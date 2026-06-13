import logging
import os
import time

import matplotlib
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.stats import norm as scipy_norm

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset

OUTPUT_DIR = os.path.join(os.getcwd(), "diagnostics_output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Logging Configuration ─────────────────────────────────────────────────────
log_filepath = os.path.join(OUTPUT_DIR, "execution.log")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.FileHandler(log_filepath), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)
# ──────────────────────────────────────────────────────────────────────────────

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {DEVICE}")

r = 0.05
sigma = 0.2
S0 = 1.0
dim = 1
T = 1.0
dt = 0.01
steps = int(T / dt)

batch_size = 500
n_paths = 1000
epochs = 200
lr_nn = 0.02
K = 1.5  # CHANGE: strike >> S0  →  deep OTM call

K1, K2, K3 = 0.35, 0.45, 0.20

n_levels = 10
beta = np.linspace(0.1, 1, n_levels)
epsilon_schedule = 1.0 / beta

N_REPLICATIONS = 10
MC_PATHS_LIST = [1000, 5_000, 50_000, 200_000]


# ── CHANGE 5: Black-Scholes call price replaces analytical probability ────────
def black_scholes_call(S, strike, rate, vol, maturity):
    d1 = (np.log(S / strike) + (rate + 0.5 * vol**2) * maturity) / (
        vol * np.sqrt(maturity)
    )
    d2 = d1 - vol * np.sqrt(maturity)
    return S * scipy_norm.cdf(d1) - strike * np.exp(-rate * maturity) * scipy_norm.cdf(
        d2
    )


C_BS = black_scholes_call(S0, K, r, sigma, T)
logger.info(
    f"Black-Scholes Call Price  C(S0={S0}, K={K}, r={r}, σ={sigma}, T={T}) = {C_BS:.10f}"
)
logger.info("=" * 70)


def create_nn():
    model = nn.Sequential(
        nn.Linear(1 + dim, 5),
        nn.Tanh(),
        nn.Linear(5, 5),
        nn.Tanh(),
        nn.Linear(5, 1),
    )
    return model.to(DEVICE)


def gbm_drift(x):
    return r * x


def gbm_diffusion(x):
    return sigma * x


def terminal_cost(x):
    return torch.clamp(x - K, min=0.0) ** 2


def get_optimal_control(model, t, x, epsilon):
    if model is None:
        return torch.zeros_like(x)
    with torch.enable_grad():
        x_req = x.detach().clone().requires_grad_(True)
        t_det = t.detach()
        inputs = torch.cat([t_det, x_req], dim=1)
        w = model(inputs)
        dw_dx = torch.autograd.grad(
            outputs=w, inputs=x_req, grad_outputs=torch.ones_like(w), create_graph=False
        )[0]
        u = 0.5 * gbm_diffusion(x_req) * dw_dx * np.sqrt(epsilon)
    return u.detach()


def generate_paths(epsilon, control_model):
    x = torch.full((n_paths, dim), S0, device=DEVICE)
    history_t, history_x = [], []
    for step in range(steps + 1):
        t = torch.full((n_paths, 1), step * dt, device=DEVICE)
        history_t.append(t.detach())
        history_x.append(x.detach())
        if step < steps:
            u = get_optimal_control(control_model, t, x, epsilon)
            dw = torch.randn_like(x) * np.sqrt(dt)
            sig_x = gbm_diffusion(x)
            x = (
                x
                + (gbm_drift(x) + sig_x * u * np.sqrt(epsilon)) * dt
                + sig_x * np.sqrt(epsilon) * dw
            )
            x = x.clamp(min=1e-6)
    return torch.stack(history_t), torch.stack(history_x)


def calculate_loss(model, t, x, epsilon):
    t = t.clone().detach().requires_grad_(True)
    x = x.clone().detach().requires_grad_(True)
    inputs = torch.cat([t, x], dim=1)
    w = model(inputs)
    grads = torch.autograd.grad(
        outputs=w, inputs=[t, x], grad_outputs=torch.ones_like(w), create_graph=True
    )
    dw_dt, dw_dx = grads[0], grads[1]
    dw_dxx = torch.autograd.grad(
        outputs=dw_dx, inputs=x, grad_outputs=torch.ones_like(dw_dx), create_graph=True
    )[0]
    sig_x = gbm_diffusion(x)
    b_x = gbm_drift(x)
    loss1 = -torch.mean(w)
    hjb = (
        -dw_dt
        - b_x * dw_dx
        + 0.5 * (sig_x * dw_dx) ** 2
        - 0.5 * epsilon * (sig_x**2) * dw_dxx
    )
    loss2 = torch.mean(hjb**2)
    terminal_mask = (t.detach() >= T - dt / 2).float()
    f = terminal_cost(x)
    loss3 = torch.mean(terminal_mask * (w - 2 * f) ** 2)
    return K1 * loss1 + K2 * loss2 + K3 * loss3


logger.info("=" * 70)
logger.info("  PHASE 1: Training the NN-IS model")
logger.info("=" * 70)

train_start = time.perf_counter()

current_model = create_nn()
previous_model = None
level_estimates_diag = []
analytical_vals_diag = []

for level, eps in enumerate(epsilon_schedule):
    logger.info(f"  Level {level+1}/{n_levels}  |  epsilon = {eps:.4f}")
    current_model.train()
    optimizer = optim.Adam(current_model.parameters(), lr=lr_nn)

    ts, xs = generate_paths(eps, previous_model)
    flat_t = ts.reshape(-1, 1)
    flat_x = xs.reshape(-1, dim)

    dataset = TensorDataset(flat_t, flat_x)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch_t, batch_x in loader:
            optimizer.zero_grad()
            loss = calculate_loss(current_model, batch_t, batch_x, eps)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.detach().item()
        if epoch % 10 == 0:
            logger.info(f"    Epoch {epoch:3d}/{epochs}  loss = {epoch_loss:.4f}")

    current_model.eval()

    x_lev = torch.full((n_paths, dim), S0, device=DEVICE)
    log_lr_lev = torch.zeros(n_paths, 1, device=DEVICE)
    for step in range(steps):
        t_lev = torch.full((n_paths, 1), step * dt, device=DEVICE)
        u_lev = get_optimal_control(current_model, t_lev, x_lev, eps)
        dw_lev = torch.randn_like(x_lev) * np.sqrt(dt)
        sx_lev = gbm_diffusion(x_lev)
        x_lev = (
            x_lev
            + (gbm_drift(x_lev) + sx_lev * u_lev * np.sqrt(eps)) * dt
            + sx_lev * np.sqrt(eps) * dw_lev
        )
        x_lev = x_lev.clamp(min=1e-6)
        log_lr_lev -= u_lev * dw_lev + 0.5 * u_lev**2 * dt

    lr_lev = torch.exp(log_lr_lev.clamp(-50, 50))
    payoff_lev = torch.clamp(x_lev - K, min=0.0)
    est_lev = np.exp(-r * T) * torch.mean(payoff_lev * lr_lev).item()
    level_estimates_diag.append(est_lev)

    sigma_eps = sigma * np.sqrt(eps)
    analytical_vals_diag.append(black_scholes_call(S0, K, r, sigma_eps, T))

    logger.info(
        f"  IS call est = {est_lev:.8f}  |  BS(σ·√ε) = {analytical_vals_diag[-1]:.8f}"
    )

    previous_model = create_nn()
    previous_model.load_state_dict(current_model.state_dict())
    previous_model.eval()

train_end = time.perf_counter()
training_time = train_end - train_start
logger.info(
    f"  >>> Total training time: {training_time:.2f} s  (EXCLUDED from comparison)"
)

trained_model = create_nn()
trained_model.load_state_dict(current_model.state_dict())
trained_model.eval()


logger.info("=" * 70)
logger.info("  PHASE 2: NN-IS Inference  (N_rep = {})".format(N_REPLICATIONS))
logger.info("=" * 70)

is_estimates = []
is_inf_times = []

for rep in range(N_REPLICATIONS):
    t0 = time.perf_counter()

    eps = 1.0
    x = torch.full((n_paths, dim), S0, device=DEVICE)
    log_lr = torch.zeros(n_paths, 1, device=DEVICE)

    for step in range(steps):
        t_step = torch.full((n_paths, 1), step * dt, device=DEVICE)
        u = get_optimal_control(trained_model, t_step, x, eps)
        dw = torch.randn_like(x) * np.sqrt(dt)
        sig_x = gbm_diffusion(x)
        x = (
            x
            + (gbm_drift(x) + sig_x * u * np.sqrt(eps)) * dt
            + sig_x * np.sqrt(eps) * dw
        )
        x = x.clamp(min=1e-6)
        log_lr -= u * dw + 0.5 * u**2 * dt

    lr_val = torch.exp(log_lr.clamp(-50, 50))
    payoff = torch.clamp(x - K, min=0.0)
    est = np.exp(-r * T) * torch.mean(payoff * lr_val).item()

    t1 = time.perf_counter()
    is_estimates.append(est)
    is_inf_times.append(t1 - t0)

is_estimates = np.array(is_estimates)
is_inf_times = np.array(is_inf_times)


logger.info("=" * 70)
logger.info("  PHASE 3: Simple Monte Carlo")
logger.info("=" * 70)

mc_results = {}

for n_mc in MC_PATHS_LIST:
    mc_ests = []
    mc_times = []

    for rep in range(N_REPLICATIONS):
        t0 = time.perf_counter()

        Z = np.random.randn(n_mc)
        S_T = S0 * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z)
        payoff_mc = np.maximum(S_T - K, 0.0)
        est = np.exp(-r * T) * np.mean(payoff_mc)

        t1 = time.perf_counter()
        mc_ests.append(est)
        mc_times.append(t1 - t0)

    mc_results[n_mc] = {
        "estimates": np.array(mc_ests),
        "times": np.array(mc_times),
    }
    logger.info(
        f"  MC  N={n_mc:>8,d}  |  mean={np.mean(mc_ests):.10f}  "
        f"std={np.std(mc_ests):.2e}  time={np.mean(mc_times):.4f}s"
    )


logger.info("=" * 70)
logger.info("  PHASE 4: Diagnostics")
logger.info("=" * 70)


def diagnostics(estimates, times, label):
    est = np.array(estimates)
    mean_est = est.mean()
    std = est.std(ddof=1)
    bias = mean_est - C_BS
    mse = np.mean((est - C_BS) ** 2)
    rmse = np.sqrt(mse)
    rel_err = abs(bias) / C_BS * 100
    ci_half = 1.96 * std / np.sqrt(len(est))
    ci_lo = mean_est - ci_half
    ci_hi = mean_est + ci_half
    covers = ci_lo <= C_BS <= ci_hi
    cv = std / abs(mean_est) if abs(mean_est) > 0 else float("inf")
    mean_time = np.mean(times)
    eff = 1.0 / (std**2 * mean_time) if std > 0 and mean_time > 0 else float("inf")

    return {
        "label": label,
        "mean": mean_est,
        "std": std,
        "bias": bias,
        "mse": mse,
        "rmse": rmse,
        "rel_err": rel_err,
        "ci_lo": ci_lo,
        "ci_hi": ci_hi,
        "covers": covers,
        "cv": cv,
        "mean_time": mean_time,
        "efficiency": eff,
    }


all_diag = []

d_is = diagnostics(is_estimates, is_inf_times, f"NN-IS  (N={n_paths})")
all_diag.append(d_is)

for n_mc in MC_PATHS_LIST:
    mc_res = mc_results[n_mc]
    d = diagnostics(mc_res["estimates"], mc_res["times"], f"MC  (N={n_mc:,d})")
    all_diag.append(d)


header = (
    f"{'Method':<22s} {'Mean':>12s} {'Bias':>12s} {'Std':>12s} "
    f"{'RMSE':>12s} {'Rel Err%':>10s} {'CV':>8s} "
    f"{'95% CI':>28s} {'Covers?':>8s} "
    f"{'Time (s)':>10s} {'Efficiency':>12s}"
)
sep = "-" * len(header)

logger.info(f"  Black-Scholes Call Price (ground truth):  C_BS = {C_BS:.10f}")
logger.info(header)
logger.info(sep)
for d in all_diag:
    ci_str = f"[{d['ci_lo']:.6e}, {d['ci_hi']:.6e}]"
    logger.info(
        f"{d['label']:<22s} {d['mean']:>12.6e} {d['bias']:>12.4e} {d['std']:>12.4e} "
        f"{d['rmse']:>12.4e} {d['rel_err']:>10.4f} {d['cv']:>8.4f} "
        f"{ci_str:>28s} {'YES' if d['covers'] else 'NO':>8s} "
        f"{d['mean_time']:>10.4f} {d['efficiency']:>12.2e}"
    )
logger.info(sep)


logger.info("=" * 70)
logger.info("  Variance Reduction Analysis")
logger.info("=" * 70)

mc_same = mc_results.get(n_paths) or mc_results[MC_PATHS_LIST[0]]
var_is = d_is["std"] ** 2
var_mc = diagnostics(mc_same["estimates"], mc_same["times"], "tmp")["std"] ** 2

if var_is > 0:
    vr_ratio = var_mc / var_is
    logger.info(f"  Var(MC, N={n_paths})  = {var_mc:.4e}")
    logger.info(f"  Var(IS, N={n_paths})  = {var_is:.4e}")
    logger.info(f"  Variance reduction ratio  = {vr_ratio:.2f}x")
else:
    logger.warning("  IS variance is zero (degenerate); cannot compute ratio.")


logger.info("=" * 70)
logger.info("  Timing Breakdown")
logger.info("=" * 70)
logger.info(
    f"  NN-IS training time (one-off, EXCLUDED from comparison): {training_time:.2f} s"
)
logger.info(
    f"  NN-IS inference time per run:  {is_inf_times.mean():.4f} s  "
    f"(std {is_inf_times.std():.4f} s)"
)
for n_mc in MC_PATHS_LIST:
    t_mc = mc_results[n_mc]["times"]
    logger.info(
        f"  MC (N={n_mc:>8,d}) time per run:  {t_mc.mean():.4f} s  (std {t_mc.std():.4f} s)"
    )


logger.info("=" * 70)
logger.info("  Generating Plots")
logger.info("=" * 70)

fig, axes = plt.subplots(2, 3, figsize=(20, 11))
fig.suptitle(
    f"Diagnostics:  NN-IS  vs  Simple Monte Carlo\n"
    f"Deep OTM European Call  –  C_BS = {C_BS:.6e}   |   "
    f"GBM(r={r}, σ={sigma}, S₀={S0}, K={K}, T={T})",
    fontsize=14,
    fontweight="bold",
    y=0.98,
)

colors_mc = plt.cm.Blues(np.linspace(0.4, 0.9, len(MC_PATHS_LIST)))
color_is = "#e74c3c"

ax = axes[0, 0]
data_for_box = [is_estimates] + [mc_results[n]["estimates"] for n in MC_PATHS_LIST]
labels_box = [f"NN-IS\nN={n_paths}"] + [f"MC\nN={n:,d}" for n in MC_PATHS_LIST]
bp = ax.boxplot(data_for_box, labels=labels_box, patch_artist=True, widths=0.5)
bp["boxes"][0].set_facecolor(color_is)
for i, box in enumerate(bp["boxes"][1:]):
    box.set_facecolor(colors_mc[i])
ax.axhline(C_BS, color="green", ls="--", lw=2, label=f"C_BS = {C_BS:.4e}")
ax.set_ylabel("Call price estimate")
ax.set_title("Distribution of Estimates")
ax.legend(fontsize=9)
ax.tick_params(axis="x", labelsize=8)

ax = axes[0, 1]
rmse_vals = [d["rmse"] for d in all_diag]
bar_labels = [d["label"] for d in all_diag]
bar_colors = [color_is] + list(colors_mc)
bars = ax.bar(range(len(all_diag)), rmse_vals, color=bar_colors, edgecolor="black")
ax.set_xticks(range(len(all_diag)))
ax.set_xticklabels(bar_labels, fontsize=7, rotation=20, ha="right")
ax.set_ylabel("RMSE")
ax.set_title("Root Mean Squared Error vs C_BS")
ax.set_yscale("log")
for b, v in zip(bars, rmse_vals):
    ax.text(
        b.get_x() + b.get_width() / 2,
        v * 1.3,
        f"{v:.2e}",
        ha="center",
        va="bottom",
        fontsize=7,
    )

ax = axes[0, 2]
bias_vals = [d["bias"] for d in all_diag]
bars = ax.bar(range(len(all_diag)), bias_vals, color=bar_colors, edgecolor="black")
ax.set_xticks(range(len(all_diag)))
ax.set_xticklabels(bar_labels, fontsize=7, rotation=20, ha="right")
ax.set_ylabel("Bias  (estimate − C_BS)")
ax.set_title("Bias vs Black-Scholes Price")
ax.axhline(0, color="black", lw=0.8)

ax = axes[1, 0]
time_vals = [d["mean_time"] for d in all_diag]
bars = ax.bar(range(len(all_diag)), time_vals, color=bar_colors, edgecolor="black")
ax.set_xticks(range(len(all_diag)))
ax.set_xticklabels(bar_labels, fontsize=7, rotation=20, ha="right")
ax.set_ylabel("Time per run (s)")
ax.set_title("Inference Time  (training excluded)")
for b, v in zip(bars, time_vals):
    ax.text(
        b.get_x() + b.get_width() / 2,
        v * 1.05,
        f"{v:.4f}s",
        ha="center",
        va="bottom",
        fontsize=7,
    )

ax = axes[1, 1]
eff_vals = [d["efficiency"] for d in all_diag]
bars = ax.bar(range(len(all_diag)), eff_vals, color=bar_colors, edgecolor="black")
ax.set_xticks(range(len(all_diag)))
ax.set_xticklabels(bar_labels, fontsize=7, rotation=20, ha="right")
ax.set_ylabel("Efficiency  1/(Var × Time)")
ax.set_title("Statistical Efficiency  (higher = better)")
ax.set_yscale("log")
for b, v in zip(bars, eff_vals):
    ax.text(
        b.get_x() + b.get_width() / 2,
        v * 1.3,
        f"{v:.1e}",
        ha="center",
        va="bottom",
        fontsize=7,
    )

ax = axes[1, 2]
cv_vals = [d["cv"] for d in all_diag]
bars = ax.bar(range(len(all_diag)), cv_vals, color=bar_colors, edgecolor="black")
ax.set_xticks(range(len(all_diag)))
ax.set_xticklabels(bar_labels, fontsize=7, rotation=20, ha="right")
ax.set_ylabel("CV  (std / mean)")
ax.set_title("Coefficient of Variation  (lower = more precise)")
for b, v in zip(bars, cv_vals):
    ax.text(
        b.get_x() + b.get_width() / 2,
        v * 1.05,
        f"{v:.3f}",
        ha="center",
        va="bottom",
        fontsize=7,
    )

plt.tight_layout(rect=[0, 0, 1, 0.94])
fig_path = os.path.join(OUTPUT_DIR, "diagnostics_comparison.png")
plt.savefig(fig_path, dpi=150, bbox_inches="tight")
plt.close()
logger.info(f"  Saved → {fig_path}")

fig2, ax2 = plt.subplots(figsize=(9, 5))

lvls = range(1, n_levels + 1)
plot_lvl_est = [max(v, 1e-15) for v in level_estimates_diag]
plot_anal = [max(v, 1e-15) for v in analytical_vals_diag]

ax2.semilogy(
    lvls,
    plot_lvl_est,
    "o-",
    lw=2,
    ms=6,
    color=color_is,
    label="IS call price estimate (per level)",
)
ax2.semilogy(
    lvls,
    plot_anal,
    "s--",
    lw=2,
    ms=6,
    color="green",
    label="BS call price at σ·√ε (per level)",
)
ax2.axhline(C_BS, color="blue", ls=":", lw=1.5, label=f"True C_BS (ε=1) = {C_BS:.6e}")
ax2.set_xlabel("Annealing Level")
ax2.set_ylabel("Call Price")
ax2.set_title(
    f"NN-IS Convergence Across Annealing Levels\n"
    f"Deep OTM Call  (K={K}, S₀={S0}, r={r}, σ={sigma})"
)
ax2.legend()
ax2.grid(True, which="both", ls=":", alpha=0.5)
plt.tight_layout()
fig2_path = os.path.join(OUTPUT_DIR, "convergence_across_levels.png")
plt.savefig(fig2_path, dpi=150, bbox_inches="tight")
plt.close()
logger.info(f"  Saved → {fig2_path}")

logger.info("=" * 70)
logger.info("  SUMMARY VERDICT")
logger.info("=" * 70)

is_rmse = d_is["rmse"]
for n_mc in MC_PATHS_LIST:
    mc_rmse_d = diagnostics(
        mc_results[n_mc]["estimates"], mc_results[n_mc]["times"], ""
    )["rmse"]
    if mc_rmse_d <= is_rmse:
        logger.info(
            f"  MC needs N={n_mc:,d} paths to match/beat IS RMSE ({is_rmse:.2e})"
        )
        logger.info(f"  MC time at that N: {mc_results[n_mc]['times'].mean():.4f} s")
        break
else:
    logger.info(
        f"  MC at N={MC_PATHS_LIST[-1]:,d} still has higher RMSE "
        f"({mc_rmse_d:.2e}) vs IS ({is_rmse:.2e})"
    )

logger.info(f"  Black-Scholes Call Price : C_BS = {C_BS:.8e}")
logger.info(
    f"  NN-IS inference time     : {d_is['mean_time']:.4f} s  "
    f"(+{training_time:.1f}s one-off training)"
)
logger.info(f"  NN-IS Call Price Est     : {d_is['mean']:.8e}")
logger.info(f"  NN-IS RMSE               : {d_is['rmse']:.4e}")
logger.info(f"  NN-IS Bias               : {d_is['bias']:.4e}")
logger.info(f"  NN-IS Efficiency         : {d_is['efficiency']:.2e}")

best_mc = min(all_diag[1:], key=lambda d: d["rmse"])
logger.info(f"  Best MC  ({best_mc['label']})")
logger.info(f"  MC  Call Price Est       : {best_mc['mean']:.8e}")
logger.info(f"  MC  RMSE                 : {best_mc['rmse']:.4e}")
logger.info(f"  MC  Bias                 : {best_mc['bias']:.4e}")
logger.info(f"  MC  Time                 : {best_mc['mean_time']:.4f} s")
logger.info(f"  MC  Efficiency           : {best_mc['efficiency']:.2e}")

logger.info("=" * 70)
logger.info(f"  All outputs saved to: {OUTPUT_DIR}")
logger.info("=" * 70)
