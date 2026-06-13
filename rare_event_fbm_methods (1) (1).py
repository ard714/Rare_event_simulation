"""
Rare-Event Simulation for Fractional Geometric Brownian Motion (fGBM)
=====================================================================
Implements FIVE discretisation methods and compares them via
neural-network-based Importance Sampling (IS):

  0. Euler–Maruyama  (baseline)
  1. Milstein Scheme  (higher-order correction, best for H > 0.5)
  2. Exact Simulation (direct Gaussian sampling of terminal log X_T)
  3. Wong–Zakai Correction (Stratonovich-to-Itô drift correction)
  4. Rough Path / Log-ODE (exponential integrator with area term)
"""

import os
import matplotlib
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.stats import norm as scipy_norm

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset

# ── Output ──────────────────────────────────────────────────────────
OUTPUT_DIR = os.getcwd()
os.makedirs(OUTPUT_DIR, exist_ok=True)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", DEVICE)

# ── Parameters ──────────────────────────────────────────────────────
mu      = 0.05
sigma   = 0.2
X0      = 1.0
dim     = 1

T       = 1.0
dt      = 0.01
steps   = int(T / dt)

batch_size = 100
n_paths    = 500
n_eval     = 2000          # paths used in IS evaluation
epochs     = 40
lr_adam    = 0.02

threshold  = 1.5

K1, K2, K3 = 0.35, 0.45, 0.20

n_levels        = 10
beta            = np.linspace(0.1, 1, n_levels)
epsilon_schedule = 1.0 / beta

H_values = [0.25, 0.75]

METHOD_NAMES = [
    "Euler-Maruyama",
    "Milstein",
    "Exact Simulation",
    "Wong-Zakai",
    "Log-ODE (Rough Path)",
]
N_PATH_METHODS = 4   # indices 0-3 are path-based; index 2 is exact

print(f"Goal : P(X_T > {threshold}) for fGBM  mu={mu}, sigma={sigma}, X0={X0}")
print(f"Epsilon schedule: {np.round(epsilon_schedule, 4)}")
print("=" * 60)

# ── fBM covariance utilities ───────────────────────────────────────
def get_fbm_matrices(H, n_steps, dt_val, device):
    """Return Cholesky factor L and increment-transform L_diff."""
    t = np.linspace(dt_val, n_steps * dt_val, n_steps)
    T_mat, S_mat = np.meshgrid(t, t)
    C = 0.5 * (T_mat ** (2*H) + S_mat ** (2*H) - np.abs(T_mat - S_mat) ** (2*H))
    C += 1e-6 * np.eye(n_steps)
    L = np.linalg.cholesky(C)

    L_diff = np.zeros_like(L)
    L_diff[0, :] = L[0, :]
    for i in range(1, n_steps):
        L_diff[i, :] = L[i, :] - L[i - 1, :]

    return (
        torch.tensor(L,      dtype=torch.float32, device=device),
        torch.tensor(L_diff, dtype=torch.float32, device=device),
    )

# ── Neural network & control ──────────────────────────────────────
def create_nn():
    return nn.Sequential(
        nn.Linear(1 + dim, 16), nn.Tanh(),
        nn.Linear(16, 16),      nn.Tanh(),
        nn.Linear(16, 1),
    ).to(DEVICE)


def terminal_cost(x):
    return torch.clamp(threshold - x, min=0.0) ** 2


def get_optimal_control(model, t, x, epsilon):
    if model is None:
        return torch.zeros_like(x)
    with torch.enable_grad():
        x_req = x.detach().clone().requires_grad_(True)
        inp   = torch.cat([t.detach(), x_req], dim=1)
        w     = model(inp)
        dw_dx = torch.autograd.grad(w, x_req, torch.ones_like(w),
                                     create_graph=False)[0]
        u = 0.5 * sigma * x_req * dw_dx * np.sqrt(epsilon)
    return u.detach()

# ── Stepping functions ─────────────────────────────────────────────
# All receive:  x (N,1), u (N,1), eps (scalar), dw (N,1), H (scalar)
# and return updated x (N,1).

def step_em(x, u, eps, dw, H):
    """Euler–Maruyama (baseline)."""
    b = mu * x
    s = sigma * x
    return (x + (b + s * u * np.sqrt(eps)) * dt
              + s * np.sqrt(eps) * dw).clamp(min=1e-6)


def step_milstein(x, u, eps, dw, H):
    """Milstein scheme – adds O(dt^{2H}) correction."""
    b = mu * x
    s = sigma * x
    # f(x)=sigma*x  =>  f'(x)=sigma
    milstein_corr = 0.5 * sigma * s * eps * (dw**2 - dt**(2*H))
    return (x + (b + s * u * np.sqrt(eps)) * dt
              + s * np.sqrt(eps) * dw
              + milstein_corr).clamp(min=1e-6)


def step_wong_zakai(x, u, eps, dw, H):
    """Wong–Zakai corrected Euler – Stratonovich drift correction."""
    b = mu * x
    s = sigma * x
    # Correction = 0.5 * sigma^2 * eps * x * dt^{2H} per step
    wz_corr = 0.5 * sigma**2 * eps * x * dt**(2*H)
    return (x + (b + s * u * np.sqrt(eps)) * dt
              + wz_corr
              + s * np.sqrt(eps) * dw).clamp(min=1e-6)


def step_log_ode(x, u, eps, dw, H):
    """Log-ODE / rough-path exponential integrator."""
    # Step in log-space:
    #   log(X_{k+1}/X_k) = (mu + sigma*sqrt(eps)*u)*dt
    #                     + sigma*sqrt(eps)*dB^H
    #                     + 0.5*sigma^2*eps*((dB^H)^2 - dt^{2H})
    drift_part = (mu + sigma * np.sqrt(eps) * u) * dt
    diff_part  = sigma * np.sqrt(eps) * dw
    area_part  = 0.5 * sigma**2 * eps * (dw**2 - dt**(2*H))
    return (x * torch.exp(drift_part + diff_part + area_part)).clamp(min=1e-6)


STEP_FNS = [step_em, step_milstein, None, step_wong_zakai, step_log_ode]
# index 2 is Exact Simulation (handled separately)

# ── Path generation (for NN training – uses EM) ───────────────────
def generate_training_paths(eps, control_model, L_diff):
    x = torch.full((n_paths, dim), X0, device=DEVICE)
    Z = torch.randn(n_paths, steps, device=DEVICE)
    dB_H = Z @ L_diff.T

    hist_t, hist_x = [], []
    for step_i in range(steps):
        t = torch.full((n_paths, 1), step_i * dt, device=DEVICE)
        hist_t.append(t.detach())
        hist_x.append(x.detach())
        u  = get_optimal_control(control_model, t, x, eps)
        dw = dB_H[:, step_i].unsqueeze(1)
        x  = step_em(x, u, eps, dw, 0.5)  # EM for training

    hist_t.append(torch.full((n_paths, 1), steps * dt, device=DEVICE))
    hist_x.append(x.detach())
    return torch.stack(hist_t), torch.stack(hist_x)

# ── HJB loss ──────────────────────────────────────────────────────
def calculate_loss(model, t, x, epsilon):
    t = t.clone().detach().requires_grad_(True)
    x = x.clone().detach().requires_grad_(True)
    inp = torch.cat([t, x], dim=1)
    w   = model(inp)

    grads  = torch.autograd.grad(w, [t, x], torch.ones_like(w), create_graph=True)
    dw_dt  = grads[0]
    dw_dx  = grads[1]
    dw_dxx = torch.autograd.grad(dw_dx, x, torch.ones_like(dw_dx), create_graph=True)[0]

    b = mu * x
    s = sigma * x

    loss1 = -torch.mean(w)
    hjb   = (-dw_dt - b * dw_dx
             + 0.5 * (s * dw_dx)**2
             - 0.5 * epsilon * s**2 * dw_dxx)
    loss2 = torch.mean(hjb**2)

    t_mask = (t.detach() >= T - dt / 2).float()
    f      = terminal_cost(x)
    loss3  = torch.mean(t_mask * (w - 2 * f)**2)

    return K1 * loss1 + K2 * loss2 + K3 * loss3

# ── IS estimation: path-based methods ─────────────────────────────
def run_is_pathbased(model, eps, step_fn, H, L, L_diff):
    """IS estimate using a path-based stepping function."""
    model.eval()
    x = torch.full((n_eval, dim), X0, device=DEVICE)
    Z = torch.randn(n_eval, steps, device=DEVICE)
    dB_H = Z @ L_diff.T

    u_history = []
    for step_i in range(steps):
        t  = torch.full((n_eval, 1), step_i * dt, device=DEVICE)
        u  = get_optimal_control(model, t, x, eps)
        u_history.append(u)
        dw = dB_H[:, step_i].unsqueeze(1)
        x  = step_fn(x, u, eps, dw, H)

    # IS weight in innovation space
    u_tensor = torch.cat(u_history, dim=1)        # (n_eval, steps)
    m_incr   = u_tensor * dt
    m_val    = torch.cumsum(m_incr, dim=1)
    phi      = torch.linalg.solve_triangular(L, m_val.T, upper=False).T

    log_lr = -torch.sum(Z * phi, dim=1) - 0.5 * torch.sum(phi**2, dim=1)
    lr_val = torch.exp(log_lr.clamp(-50, 50)).unsqueeze(1)

    hit = (x > threshold).float()
    return torch.mean(hit * lr_val).item()

# ── IS estimation: exact simulation ──────────────────────────────
def run_is_exact(eps, H):
    """
    Exact simulation: sample log X_T ~ N(mean, var) directly.
    Uses optimal exponential-tilting IS (shift mean to threshold).
    """
    mean_orig = mu * T
    var       = sigma**2 * eps * T**(2*H)
    std       = np.sqrt(var)
    log_thr   = np.log(threshold / X0)

    # Tilt the mean to the rare-event boundary for efficient IS
    mean_tilt = log_thr

    Z      = torch.randn(n_eval, device=DEVICE)
    log_xt = mean_tilt + std * Z

    # log IS weight = log p_orig(log_xt) - log p_tilt(log_xt)
    log_w  = (-0.5 * ((log_xt - mean_orig)**2 - (log_xt - mean_tilt)**2) / var)
    w      = torch.exp(log_w.clamp(-50, 50))

    hit = (log_xt > log_thr).float()
    return torch.mean(hit * w).item()

# ── Analytical benchmark ─────────────────────────────────────────
def analytical_fgbm(eps, H):
    """P(X_T > threshold) under Stratonovich fGBM."""
    log_thr  = np.log(threshold / X0)
    mean_log = mu * T
    std_log  = sigma * np.sqrt(eps) * T**H
    return 1.0 - scipy_norm.cdf(log_thr, loc=mean_log, scale=std_log)

# ==================================================================
#  MAIN LOOP
# ==================================================================
all_results = {}

for H in H_values:
    print(f"\n{'='*25}  H = {H}  {'='*25}")
    L, L_diff = get_fbm_matrices(H, steps, dt, DEVICE)

    # Storage per method
    method_estimates = {m: [] for m in range(len(METHOD_NAMES))}
    analytical_vals  = []

    current_model  = create_nn()
    previous_model = None

    for level, eps in enumerate(epsilon_schedule):
        print(f"  Level {level+1}/{n_levels}  eps={eps:.4f}")

        # ── Train NN (using EM paths) ──
        current_model.train()
        optimizer = optim.Adam(current_model.parameters(), lr=lr_adam)
        ts, xs   = generate_training_paths(eps, previous_model, L_diff)
        flat_t   = ts.reshape(-1, 1)
        flat_x   = xs.reshape(-1, dim)
        loader   = DataLoader(TensorDataset(flat_t, flat_x),
                              batch_size=batch_size, shuffle=True)

        for _ in range(epochs):
            for bt, bx in loader:
                optimizer.zero_grad()
                loss = calculate_loss(current_model, bt, bx, eps)
                loss.backward()
                optimizer.step()

        current_model.eval()

        # ── Evaluate each method ──
        for mi, name in enumerate(METHOD_NAMES):
            if mi == 2:
                est = run_is_exact(eps, H)
            else:
                est = run_is_pathbased(current_model, eps, STEP_FNS[mi], H, L, L_diff)
            method_estimates[mi].append(est)
            print(f"    {name:25s} IS={est:.6f}")

        ana = analytical_fgbm(eps, H)
        analytical_vals.append(ana)
        print(f"    {'Analytical':25s}    {ana:.6f}")

        # ── Carry model forward ──
        previous_model = create_nn()
        previous_model.load_state_dict(current_model.state_dict())
        previous_model.eval()

    all_results[H] = {
        "methods":    method_estimates,
        "analytical": analytical_vals,
    }

# ==================================================================
#  PLOTTING
# ==================================================================
levels = range(1, n_levels + 1)
colors = ["steelblue", "darkorange", "mediumseagreen", "crimson", "mediumpurple"]
markers = ["o", "s", "D", "^", "v"]

# ── Plot 1: Convergence per H ─────────────────────────────────────
fig, axes = plt.subplots(1, len(H_values), figsize=(8 * len(H_values), 6))
if len(H_values) == 1:
    axes = [axes]

for idx, H in enumerate(H_values):
    ax  = axes[idx]
    res = all_results[H]

    for mi, name in enumerate(METHOD_NAMES):
        ax.semilogy(levels, res["methods"][mi],
                    marker=markers[mi], color=colors[mi],
                    linewidth=1.8, markersize=5, label=name)

    ax.semilogy(levels, res["analytical"],
                "k--", linewidth=2, label="Analytical (Stratonovich)")

    ax.set_xlabel("Level", fontsize=11)
    ax.set_ylabel("P(X_T > a)  estimate", fontsize=11)
    ax.set_title(f"H = {H}", fontsize=13)
    ax.legend(fontsize=8, loc="best")
    ax.grid(True, which="both", ls="--", alpha=0.4)

fig.suptitle(
    f"fGBM Rare-Event IS Convergence — All Discretisation Methods\n"
    f"(threshold a={threshold}, μ={mu}, σ={sigma})",
    fontsize=14, y=1.02,
)
fig.tight_layout()
path1 = os.path.join(OUTPUT_DIR, "fbm_methods_convergence.png")
fig.savefig(path1, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"\nSaved {path1}")

# ── Plot 2: Relative error vs level per H ────────────────────────
fig2, axes2 = plt.subplots(1, len(H_values), figsize=(8 * len(H_values), 6))
if len(H_values) == 1:
    axes2 = [axes2]

for idx, H in enumerate(H_values):
    ax  = axes2[idx]
    res = all_results[H]
    ana = np.array(res["analytical"])

    for mi, name in enumerate(METHOD_NAMES):
        ests    = np.array(res["methods"][mi])
        rel_err = np.abs(ests - ana) / (ana + 1e-15) * 100
        ax.plot(list(levels), rel_err,
                marker=markers[mi], color=colors[mi],
                linewidth=1.8, markersize=5, label=name)

    ax.set_xlabel("Level", fontsize=11)
    ax.set_ylabel("Relative Error  (%)", fontsize=11)
    ax.set_title(f"H = {H}", fontsize=13)
    ax.legend(fontsize=8, loc="best")
    ax.grid(True, which="both", ls="--", alpha=0.4)

fig2.suptitle(
    "Relative Error vs Analytical — All Discretisation Methods",
    fontsize=14, y=1.02,
)
fig2.tight_layout()
path2 = os.path.join(OUTPUT_DIR, "fbm_methods_rel_error.png")
fig2.savefig(path2, dpi=150, bbox_inches="tight")
plt.close(fig2)
print(f"Saved {path2}")

# ── Plot 3: Final-level bar comparison ────────────────────────────
fig3, axes3 = plt.subplots(1, len(H_values), figsize=(7 * len(H_values), 5))
if len(H_values) == 1:
    axes3 = [axes3]

for idx, H in enumerate(H_values):
    ax     = axes3[idx]
    res    = all_results[H]
    ana    = res["analytical"][-1]
    finals = [res["methods"][mi][-1] for mi in range(len(METHOD_NAMES))]

    x_pos = np.arange(len(METHOD_NAMES))
    bars  = ax.bar(x_pos, finals, color=colors, alpha=0.85, edgecolor="black")
    ax.axhline(ana, color="black", ls="--", lw=2, label=f"Analytical: {ana:.6f}")

    ax.set_xticks(x_pos)
    ax.set_xticklabels([n.replace(" ", "\n") for n in METHOD_NAMES], fontsize=8)
    ax.set_ylabel("P(X_T > a)", fontsize=10)
    ax.set_title(f"Final-Level Estimates  |  H = {H}", fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)

fig3.suptitle("Final-Level IS Estimates vs Analytical", fontsize=13, y=1.01)
fig3.tight_layout()
path3 = os.path.join(OUTPUT_DIR, "fbm_methods_final_bar.png")
fig3.savefig(path3, dpi=150, bbox_inches="tight")
plt.close(fig3)
print(f"Saved {path3}")

# ── Summary table ─────────────────────────────────────────────────
print("\n" + "=" * 80)
print("SUMMARY  (final level, eps=1.0)")
print("=" * 80)
header = f"{'H':>5}  {'Method':25s}  {'IS Estimate':>12}  {'Analytical':>12}  {'RelErr%':>8}"
print(header)
print("-" * 80)
for H in H_values:
    res = all_results[H]
    ana = res["analytical"][-1]
    for mi, name in enumerate(METHOD_NAMES):
        est = res["methods"][mi][-1]
        rel = abs(est - ana) / (ana + 1e-15) * 100
        print(f"{H:5.2f}  {name:25s}  {est:12.6f}  {ana:12.6f}  {rel:7.2f}%")
    print()

print(f"All plots saved to {OUTPUT_DIR}")
