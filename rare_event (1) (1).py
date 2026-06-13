import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from torch.utils.data import DataLoader, TensorDataset
from scipy.stats import norm as scipy_norm
import itertools

OUTPUT_DIR = os.environ.get("PLOT_DIR", "/tmp/rare_event_plots")
os.makedirs(OUTPUT_DIR, exist_ok=True)

if torch.cuda.is_available():
    n_gpus = torch.cuda.device_count()
    DEVICE = torch.device("cuda")
    print(f"CUDA available: {n_gpus} GPU(s)")
    for i in range(n_gpus):
        props = torch.cuda.get_device_properties(i)
        print(f"  GPU {i}: {props.name}  |  {props.total_memory // 1024**2} MB  |  {props.multi_processor_count} SMs")
else:
    n_gpus = 0
    DEVICE = torch.device("cpu")
    print("CUDA not available, using CPU")
    print(f"  CPU threads available: {torch.get_num_threads()}")
    torch.set_num_threads(os.cpu_count() or 4)

T = 1.0
dt = 0.01
steps = int(T / dt)
batch_size = 64
n_paths = 2000
N_final = 10000
epochs = 50
lr_adam = 0.01
K1, K2, K3 = 0.35, 0.45, 0.20
n_levels = 10
beta_sched = np.linspace(0.1, 1.0, n_levels)
epsilon_schedule = 1.0 / beta_sched
thresholds = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5]

SDE_CONFIGS = {
    "BM": {
        "name": "Brownian Motion",
        "dim": 1,
        "X0": 0.0,
        "params": {},
    },
    "OU": {
        "name": "Ornstein-Uhlenbeck",
        "dim": 1,
        "X0": 0.0,
        "params": {"gamma": 0.01, "sigma_ou": 0.1},
    },
    "CIR": {
        "name": "Cox-Ingersoll-Ross",
        "dim": 1,
        "X0": 0.1,
        "params": {"alpha": 0.1, "beta_cir": 0.1, "sigma_cir": 0.15},
    },
    "DW": {
        "name": "Double Well",
        "dim": 1,
        "X0": -1.0,
        "params": {"kappa": 1.0},
    },
}

def bm_drift(x, params):
    return torch.zeros_like(x)

def bm_diffusion(x, params):
    return torch.ones_like(x)

def ou_drift(x, params):
    return -params["gamma"] * x

def ou_diffusion(x, params):
    return params["sigma_ou"] * torch.ones_like(x)

def cir_drift(x, params):
    return params["alpha"] * (params["beta_cir"] - x)

def cir_diffusion(x, params):
    return params["sigma_cir"] * torch.sqrt(torch.clamp(x, min=1e-8))

def dw_drift(x, params):
    return -params["kappa"] * 4.0 * x * (x**2 - 1.0)

def dw_diffusion(x, params):
    return np.sqrt(2.0) * torch.ones_like(x)

SDE_FUNCS = {
    "BM":  (bm_drift,  bm_diffusion),
    "OU":  (ou_drift,  ou_diffusion),
    "CIR": (cir_drift, cir_diffusion),
    "DW":  (dw_drift,  dw_diffusion),
}

def terminal_cost(x, threshold):
    return torch.clamp(threshold - x, min=0.0) ** 2

def create_nn(dim):
    model = nn.Sequential(
        nn.Linear(1 + dim, 32), nn.Tanh(),
        nn.Linear(32, 32), nn.Tanh(),
        nn.Linear(32, 1),
    )
    return model.to(DEVICE)

def get_optimal_control(model, t, x, epsilon, drift_fn, diff_fn, params):
    if model is None:
        return torch.zeros_like(x)
    with torch.enable_grad():
        x_req = x.detach().clone().requires_grad_(True)
        t_det = t.detach()
        inputs = torch.cat([t_det, x_req], dim=1)
        w = model(inputs)
        dw_dx = torch.autograd.grad(
            outputs=w, inputs=x_req,
            grad_outputs=torch.ones_like(w),
            create_graph=False
        )[0]
        sig_x = diff_fn(x_req, params)
        u = 0.5 * sig_x * dw_dx / np.sqrt(epsilon)
    return u.detach()

def euler_step(x, u, epsilon, drift_fn, diff_fn, params, sde_key):
    dw = torch.randn_like(x) * np.sqrt(dt)
    b_x = drift_fn(x, params)
    sig_x = diff_fn(x, params)
    x_new = x + (b_x + sig_x * u * np.sqrt(epsilon)) * dt + sig_x * np.sqrt(epsilon) * dw
    if sde_key == "CIR":
        x_new = x_new.clamp(min=1e-6)
    return x_new, dw, sig_x

def generate_paths(epsilon, control_model, sde_key, cfg):
    drift_fn, diff_fn = SDE_FUNCS[sde_key]
    params = cfg["params"]
    dim = cfg["dim"]
    X0 = cfg["X0"]
    x = torch.full((n_paths, dim), X0, device=DEVICE)
    history_t, history_x = [], []
    for step in range(steps + 1):
        t = torch.full((n_paths, 1), step * dt, device=DEVICE)
        history_t.append(t.detach())
        history_x.append(x.detach())
        if step < steps:
            u = get_optimal_control(control_model, t, x, epsilon, drift_fn, diff_fn, params)
            x, _, _ = euler_step(x, u, epsilon, drift_fn, diff_fn, params, sde_key)
    return torch.stack(history_t), torch.stack(history_x)

def calculate_loss(model, t, x, epsilon, drift_fn, diff_fn, params, threshold):
    t = t.clone().detach().requires_grad_(True)
    x = x.clone().detach().requires_grad_(True)
    inputs = torch.cat([t, x], dim=1)
    w = model(inputs)
    grads = torch.autograd.grad(
        outputs=w, inputs=[t, x],
        grad_outputs=torch.ones_like(w),
        create_graph=True
    )
    dw_dt = grads[0]
    dw_dx = grads[1]
    dw_dxx = torch.autograd.grad(
        outputs=dw_dx, inputs=x,
        grad_outputs=torch.ones_like(dw_dx),
        create_graph=True
    )[0]
    sig_x = diff_fn(x, params)
    b_x = drift_fn(x, params)
    loss1 = -torch.mean(w)
    hjb = (
        dw_dt
        - b_x * dw_dx
        + 0.5 * (sig_x * dw_dx) ** 2
        - 0.5 * epsilon * (sig_x ** 2) * dw_dxx
    )
    loss2 = torch.mean(hjb ** 2)
    terminal_mask = (t.detach() >= T - dt / 2).float()
    f = terminal_cost(x, threshold)
    loss3 = torch.mean(terminal_mask * (w - 2 * f) ** 2)
    return K1 * loss1 + K2 * loss2 + K3 * loss3

def run_is_estimate(model, epsilon, n_sample, sde_key, cfg, threshold):
    drift_fn, diff_fn = SDE_FUNCS[sde_key]
    params = cfg["params"]
    dim = cfg["dim"]
    X0 = cfg["X0"]
    model.eval()
    x = torch.full((n_sample, dim), X0, device=DEVICE)
    log_lr = torch.zeros(n_sample, 1, device=DEVICE)
    for step in range(steps):
        t = torch.full((n_sample, 1), step * dt, device=DEVICE)
        u = get_optimal_control(model, t, x, epsilon, drift_fn, diff_fn, params)
        dw = torch.randn_like(x) * np.sqrt(dt)
        b_x = drift_fn(x, params)
        sig_x = diff_fn(x, params)
        x = x + (b_x + sig_x * u * np.sqrt(epsilon)) * dt + sig_x * np.sqrt(epsilon) * dw
        if sde_key == "CIR":
            x = x.clamp(min=1e-6)
        log_lr -= (u * (dw / np.sqrt(epsilon))) + 0.5 * u ** 2 * dt
    lr_val = torch.exp(log_lr.clamp(-50, 50))
    hit = (x > threshold).float()
    return torch.mean(hit * lr_val).item()

def train_one_config(sde_key, threshold):
    cfg = SDE_CONFIGS[sde_key]
    drift_fn, diff_fn = SDE_FUNCS[sde_key]
    params = cfg["params"]
    dim = cfg["dim"]
    print(f"\n{'='*60}")
    print(f"SDE: {cfg['name']}  |  threshold a={threshold}")
    print(f"{'='*60}")
    current_model = create_nn(dim)
    if n_gpus > 1:
        current_model = nn.DataParallel(current_model)
    previous_model = None
    level_estimates = []
    for level, eps in enumerate(epsilon_schedule):
        print(f"  Level {level+1}/{n_levels}  eps={eps:.4f}", end="  ")
        current_model.train()
        optimizer = optim.Adam(current_model.parameters(), lr=lr_adam)
        ts, xs = generate_paths(eps, previous_model, sde_key, cfg)
        flat_t = ts.reshape(-1, 1)
        flat_x = xs.reshape(-1, dim)
        dataset = TensorDataset(flat_t, flat_x)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                            num_workers=0, pin_memory=False)
        for epoch in range(epochs):
            for batch_t, batch_x in loader:
                batch_t, batch_x = batch_t.to(DEVICE), batch_x.to(DEVICE)
                optimizer.zero_grad()
                inner = current_model.module if isinstance(current_model, nn.DataParallel) else current_model
                loss = calculate_loss(inner, batch_t, batch_x, eps, drift_fn, diff_fn, params, threshold)
                loss.backward()
                optimizer.step()
        inner = current_model.module if isinstance(current_model, nn.DataParallel) else current_model
        est = run_is_estimate(inner, eps, n_paths, sde_key, cfg, threshold)
        print(f"IS_est={est:.6f}")
        level_estimates.append(est)
        previous_model = create_nn(dim)
        previous_model.load_state_dict(inner.state_dict())
        previous_model.eval()
    inner = current_model.module if isinstance(current_model, nn.DataParallel) else current_model
    final_est = run_is_estimate(inner, 1.0, N_final, sde_key, cfg, threshold)
    print(f"  FINAL (N={N_final}, eps=1.0): {final_est:.6f}")
    return level_estimates, final_est

def analytical_bm(threshold, T):
    return 1.0 - scipy_norm.cdf(threshold / np.sqrt(T))

def analytical_ou(threshold, gamma, sigma_ou, T):
    var = (sigma_ou**2 / (2 * gamma)) * (1 - np.exp(-2 * gamma * T))
    mean = 0.0 * np.exp(-gamma * T)
    return 1.0 - scipy_norm.cdf(threshold, loc=mean, scale=np.sqrt(var))

def get_analytical(sde_key, cfg, threshold):
    if sde_key == "BM":
        return analytical_bm(threshold, T)
    elif sde_key == "OU":
        p = cfg["params"]
        return analytical_ou(threshold, p["gamma"], p["sigma_ou"], T)
    else:
        return None

all_results = {}
for sde_key in SDE_CONFIGS:
    all_results[sde_key] = {}
    for thr in thresholds:
        level_ests, final_est = train_one_config(sde_key, thr)
        analytical = get_analytical(sde_key, SDE_CONFIGS[sde_key], thr)
        all_results[sde_key][thr] = {
            "level_estimates": level_ests,
            "final_estimate": final_est,
            "analytical": analytical,
        }
        if analytical is not None:
            rel_err = abs(final_est - analytical) / (analytical + 1e-15) * 100
            print(f"  Analytical={analytical:.6f}  RelErr={rel_err:.2f}%")

n_sdes = len(SDE_CONFIGS)
n_thr = len(thresholds)

fig1, axes1 = plt.subplots(n_sdes, n_thr, figsize=(4 * n_thr, 4 * n_sdes))
fig1.suptitle("Per-Level IS Estimate Convergence (all SDEs × all thresholds)", fontsize=14, y=1.01)
for si, sde_key in enumerate(SDE_CONFIGS):
    sde_name = SDE_CONFIGS[sde_key]["name"]
    for ti, thr in enumerate(thresholds):
        ax = axes1[si, ti]
        res = all_results[sde_key][thr]
        levels = range(1, n_levels + 1)
        ax.semilogy(levels, res["level_estimates"], "o-", color="steelblue",
                    linewidth=1.5, markersize=4, label="Per-level IS")
        ax.axhline(res["final_estimate"], color="red", linestyle="--", linewidth=1.2,
                   label=f"Final IS: {res['final_estimate']:.4f}")
        if res["analytical"] is not None:
            ax.axhline(res["analytical"], color="green", linestyle=":", linewidth=1.2,
                       label=f"Analytical: {res['analytical']:.4f}")
        ax.set_title(f"{sde_name}\na={thr}", fontsize=9)
        ax.set_xlabel("Level", fontsize=8)
        ax.set_ylabel("P estimate", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.legend(fontsize=6)
fig1.tight_layout()
fig1.savefig(os.path.join(OUTPUT_DIR, "convergence_all.png"), dpi=150, bbox_inches="tight")
plt.close(fig1)
print("Saved convergence_all.png")

fig2, axes2 = plt.subplots(1, n_sdes, figsize=(5 * n_sdes, 5))
fig2.suptitle("Final IS Estimate vs Threshold  (all SDEs)", fontsize=13)
colors = plt.cm.plasma(np.linspace(0.1, 0.9, n_thr))
for si, sde_key in enumerate(SDE_CONFIGS):
    ax = axes2[si]
    sde_name = SDE_CONFIGS[sde_key]["name"]
    final_ests = [all_results[sde_key][thr]["final_estimate"] for thr in thresholds]
    analyticals = [all_results[sde_key][thr]["analytical"] for thr in thresholds]
    ax.semilogy(thresholds, final_ests, "o-", color="steelblue", linewidth=2,
                markersize=7, label="Final IS estimate")
    if any(a is not None for a in analyticals):
        valid = [(t, a) for t, a in zip(thresholds, analyticals) if a is not None]
        ax.semilogy([v[0] for v in valid], [v[1] for v in valid], "s--", color="green",
                    linewidth=2, markersize=7, label="Analytical")
    ax.set_title(sde_name, fontsize=10)
    ax.set_xlabel("Threshold a", fontsize=9)
    ax.set_ylabel("P(X_T > a)", fontsize=9)
    ax.legend(fontsize=8)
    ax.grid(True, which="both", alpha=0.3)
fig2.tight_layout()
fig2.savefig(os.path.join(OUTPUT_DIR, "final_vs_threshold.png"), dpi=150, bbox_inches="tight")
plt.close(fig2)
print("Saved final_vs_threshold.png")

fig3, axes3 = plt.subplots(n_thr, 1, figsize=(10, 4 * n_thr))
fig3.suptitle("Final IS Estimates: All SDEs Overlaid per Threshold", fontsize=13)
sde_colors = {"BM": "steelblue", "OU": "darkorange", "CIR": "green", "DW": "purple"}
sde_markers = {"BM": "o", "OU": "s", "CIR": "^", "DW": "D"}
for ti, thr in enumerate(thresholds):
    ax = axes3[ti]
    for sde_key in SDE_CONFIGS:
        res = all_results[sde_key][thr]
        ax.semilogy(range(1, n_levels + 1), res["level_estimates"],
                    marker=sde_markers[sde_key], color=sde_colors[sde_key],
                    linewidth=1.5, markersize=5, label=SDE_CONFIGS[sde_key]["name"])
    ax.set_title(f"Threshold a = {thr}", fontsize=10)
    ax.set_xlabel("Level", fontsize=9)
    ax.set_ylabel("P estimate", fontsize=9)
    ax.legend(fontsize=8)
    ax.grid(True, which="both", alpha=0.3)
fig3.tight_layout()
fig3.savefig(os.path.join(OUTPUT_DIR, "all_sdes_per_threshold.png"), dpi=150, bbox_inches="tight")
plt.close(fig3)
print("Saved all_sdes_per_threshold.png")

fig4, ax4 = plt.subplots(figsize=(10, 6))
for sde_key in SDE_CONFIGS:
    final_ests = [all_results[sde_key][thr]["final_estimate"] for thr in thresholds]
    ax4.semilogy(thresholds, final_ests, marker=sde_markers[sde_key],
                 color=sde_colors[sde_key], linewidth=2, markersize=8,
                 label=SDE_CONFIGS[sde_key]["name"])
ax4.set_xlabel("Threshold a", fontsize=11)
ax4.set_ylabel("P(X_T > a)", fontsize=11)
ax4.set_title("Rare Event Probability vs Threshold — All SDEs", fontsize=12)
ax4.legend(fontsize=10)
ax4.grid(True, which="both", alpha=0.3)
fig4.tight_layout()
fig4.savefig(os.path.join(OUTPUT_DIR, "summary_all_sdes.png"), dpi=150, bbox_inches="tight")
plt.close(fig4)
print("Saved summary_all_sdes.png")

print("\n" + "="*60)
print("SUMMARY TABLE")
print(f"{'SDE':<10} {'a':<6} {'Final IS':<14} {'Analytical':<14} {'RelErr%':<10}")
print("-"*60)
for sde_key in SDE_CONFIGS:
    for thr in thresholds:
        res = all_results[sde_key][thr]
        analytical_str = f"{res['analytical']:.6f}" if res["analytical"] is not None else "N/A"
        if res["analytical"] is not None:
            rel_err = abs(res["final_estimate"] - res["analytical"]) / (res["analytical"] + 1e-15) * 100
            rel_err_str = f"{rel_err:.2f}%"
        else:
            rel_err_str = "N/A"
        print(f"{sde_key:<10} {thr:<6} {res['final_estimate']:<14.6f} {analytical_str:<14} {rel_err_str:<10}")

print(f"\nAll plots saved to: {OUTPUT_DIR}")
