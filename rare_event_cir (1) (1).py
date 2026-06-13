import os

import matplotlib
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.stats import ncx2

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset

OUTPUT_DIR = os.environ.get("PLOT_DIR", "/tmp/rare_event_plots")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def save_plot(fig, filename):
    tmp_path = os.path.join(OUTPUT_DIR, filename)
    local_path = os.path.join(os.getcwd(), filename)

    fig.savefig(tmp_path, dpi=150, bbox_inches="tight")
    fig.savefig(local_path, dpi=150, bbox_inches="tight")

    print(f"Saved → {tmp_path} AND {local_path}")


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", DEVICE)

# CIR parameters
theta = 2.0
mu = 1.0
sigma = 0.5
X0 = 1.0

T = 1.0
dt = 0.01
steps = int(T / dt)

batch_size = 100
n_paths = 500
epochs = 40
lr = 0.02

threshold = 1.5

K1, K2, K3 = 0.35, 0.45, 0.20

n_levels = 10
beta = np.linspace(0.1, 1, n_levels)
epsilon_schedule = 1 / beta

print("Epsilon schedule:", np.round(epsilon_schedule, 4))


def create_nn():
    return nn.Sequential(
        nn.Linear(2, 5), nn.Tanh(), nn.Linear(5, 5), nn.Tanh(), nn.Linear(5, 1)
    ).to(DEVICE)


def cir_drift(x):
    return theta * (mu - x)


def cir_diffusion(x):
    return sigma * torch.sqrt(x.clamp(min=1e-6))


def terminal_cost(x):
    return torch.clamp(threshold - x, min=0.0) ** 2


def get_optimal_control(model, t, x, epsilon):
    if model is None:
        return torch.zeros_like(x)
    with torch.enable_grad():
        x_req = x.detach().clone().requires_grad_(True)
        inputs = torch.cat([t.detach(), x_req], dim=1)
        w = model(inputs)
        dw_dx = torch.autograd.grad(w, x_req, torch.ones_like(w), create_graph=False)[0]
        u = 0.5 * cir_diffusion(x_req) * dw_dx * np.sqrt(epsilon)
    return u.detach()


def generate_paths(epsilon, control_model):
    x = torch.full((n_paths, 1), X0, device=DEVICE)
    ts, xs = [], []

    for step in range(steps + 1):
        t = torch.full((n_paths, 1), step * dt, device=DEVICE)
        ts.append(t)
        xs.append(x)

        if step < steps:
            u = get_optimal_control(control_model, t, x, epsilon)
            dw = torch.randn_like(x) * np.sqrt(dt)
            sig = cir_diffusion(x)

            x = (
                x
                + (cir_drift(x) + sig * u * np.sqrt(epsilon)) * dt
                + sig * np.sqrt(epsilon) * dw
            )
            x = x.clamp(min=1e-6)

    return torch.stack(ts), torch.stack(xs)


def calculate_loss(model, t, x, epsilon):
    t.requires_grad_(True)
    x.requires_grad_(True)

    w = model(torch.cat([t, x], dim=1))

    dw_dt, dw_dx = torch.autograd.grad(w, [t, x], torch.ones_like(w), create_graph=True)

    dw_dxx = torch.autograd.grad(dw_dx, x, torch.ones_like(dw_dx), create_graph=True)[0]

    sig = cir_diffusion(x)
    b = cir_drift(x)

    loss1 = -torch.mean(w)

    hjb = (
        -dw_dt
        - b * dw_dx
        + 0.5 * (sig * dw_dx) ** 2
        - 0.5 * epsilon * (sig**2) * dw_dxx
    )
    loss2 = torch.mean(hjb**2)

    terminal_mask = (t >= T - dt / 2).float()
    loss3 = torch.mean(terminal_mask * (w - 2 * terminal_cost(x)) ** 2)

    return K1 * loss1 + K2 * loss2 + K3 * loss3


def cir_analytical_prob(eps):
    c = (sigma**2 * eps * (1 - np.exp(-theta * T))) / (4 * theta)
    df = 4 * theta * mu / (sigma**2 * eps)
    lam = (4 * theta * np.exp(-theta * T) * X0) / (
        sigma**2 * eps * (1 - np.exp(-theta * T))
    )
    return 1.0 - ncx2.cdf(threshold / c, df, lam)


current_model = create_nn()
previous_model = None
level_estimates = []

for level, eps in enumerate(epsilon_schedule):
    print(f"\nLevel {level+1} | epsilon={eps:.4f}")

    optimizer = optim.Adam(current_model.parameters(), lr=lr)

    ts, xs = generate_paths(eps, previous_model)
    dataset = TensorDataset(ts.reshape(-1, 1), xs.reshape(-1, 1))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(epochs):
        for t_batch, x_batch in loader:
            optimizer.zero_grad()
            loss = calculate_loss(current_model, t_batch, x_batch, eps)
            loss.backward()
            optimizer.step()

    # Importance sampling estimate
    x = torch.full((n_paths, 1), X0, device=DEVICE)
    log_lr = torch.zeros(n_paths, 1, device=DEVICE)

    for step in range(steps):
        t = torch.full((n_paths, 1), step * dt, device=DEVICE)
        u = get_optimal_control(current_model, t, x, eps)

        dw = torch.randn_like(x) * np.sqrt(dt)
        sig = cir_diffusion(x)

        x = x + (cir_drift(x) + sig * u * np.sqrt(eps)) * dt + sig * np.sqrt(eps) * dw
        x = x.clamp(min=1e-6)

        log_lr -= u * dw + 0.5 * u**2 * dt

    est = torch.mean((x > threshold).float() * torch.exp(log_lr)).item()
    print("Estimate:", est)

    level_estimates.append(est)

    previous_model = create_nn()
    previous_model.load_state_dict(current_model.state_dict())
    previous_model.eval()

# ─────────────────────────────────────────────
# Analytical for all eps
analytical_all = [cir_analytical_prob(eps) for eps in epsilon_schedule]

# ─────────────────────────────────────────────
# PLOT 1: Convergence
fig = plt.figure(figsize=(9, 5))
levels = range(1, n_levels + 1)

plt.semilogy(levels, level_estimates, "o-", label="IS Estimate")
plt.semilogy(levels, analytical_all, "s--", label="Analytical")

plt.xlabel("Level")
plt.ylabel("Probability")
plt.title("CIR Rare Event Convergence (All ε)")
plt.legend()
plt.grid(True, which="both", linestyle="--")

save_plot(fig, "convergence_plot_cir.png")
plt.close()

# ─────────────────────────────────────────────
# PLOT 2: Convergence speed
fig2 = plt.figure(figsize=(8, 4))

diff = np.diff(level_estimates, prepend=level_estimates[0])
plt.plot(levels, diff, "o-", color="purple")

plt.title("Convergence Speed (Δ estimate)")
plt.xlabel("Level")
plt.ylabel("Δ Probability")
plt.grid(True)

save_plot(fig2, "convergence_speed_cir.png")
plt.close()
