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

OUTPUT_DIR = os.getcwd()
os.makedirs(OUTPUT_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", DEVICE)

mu = 0.05
sigma = 0.2
X0 = 1.0

dim = 1
T = 1.0
dt = 0.01
steps = int(T / dt)

batch_size = 100
n_paths = 500
epochs = 40
lr = 0.02

threshold = 1.5

K1 = 0.35
K2 = 0.45
K3 = 0.20

n_levels = 10
beta = np.linspace(0.1, 1, n_levels)
epsilon_schedule = 1 / beta

print(
    "Goal: estimate P(X_T > {})  |  GBM: mu={}, sigma={}, X0={}".format(
        threshold, mu, sigma, X0
    )
)
print("Epsilon schedule:", np.round(epsilon_schedule, 4))
print("=" * 55)


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
    return mu * x


def gbm_diffusion(x):
    return sigma * x


def terminal_cost(x):
    return torch.clamp(threshold - x, min=0.0) ** 2


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
    x = torch.full((n_paths, dim), X0, device=DEVICE)
    history_t = []
    history_x = []
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
    dw_dt = grads[0]
    dw_dx = grads[1]

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


current_model = create_nn()
previous_model = None
level_estimates = []
analytical_values = []

for level, eps in enumerate(epsilon_schedule):
    print("\nLevel {}/{}  |  epsilon = {:.4f}".format(level + 1, n_levels, eps))

    current_model.train()
    optimizer = optim.Adam(current_model.parameters(), lr=lr)

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
            print("  Epoch {:3d}/{:d}  loss = {:.4f}".format(epoch, epochs, epoch_loss))

    current_model.eval()

    x = torch.full((n_paths, dim), X0, device=DEVICE)
    log_lr = torch.zeros(n_paths, 1, device=DEVICE)

    for step in range(steps):
        t = torch.full((n_paths, 1), step * dt, device=DEVICE)
        u = get_optimal_control(current_model, t, x, eps)
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
    hit = (x > threshold).float()
    est = torch.mean(hit * lr_val).item()
    level_estimates.append(est)

    log_threshold = np.log(threshold / X0)
    mean_log = (mu - 0.5 * sigma**2 * eps) * T
    std_log = sigma * np.sqrt(eps * T)

    analytical_eps = 1.0 - scipy_norm.cdf(log_threshold, loc=mean_log, scale=std_log)
    analytical_values.append(analytical_eps)

    print("  IS estimate: {:.6f}".format(est))
    print("  Analytical : {:.6f}".format(analytical_eps))

    previous_model = create_nn()
    previous_model.load_state_dict(current_model.state_dict())
    previous_model.eval()

plt.figure(figsize=(8, 4))

levels = range(1, n_levels + 1)

plt.semilogy(
    levels, level_estimates, "o-", linewidth=2, markersize=6, label="IS estimate"
)
plt.semilogy(
    levels, analytical_values, "s--", linewidth=2, markersize=6, label="Analytical"
)

plt.xlabel("Level")
plt.ylabel("Probability estimate")
plt.title("Rare Event Probability vs Levels\nP(X_T > {})".format(threshold))
plt.legend()
plt.tight_layout()

out_path = os.path.join(OUTPUT_DIR, "convergence_plot.png")
plt.savefig(out_path, dpi=150, bbox_inches="tight")
plt.close()

print(f"Plot saved → {out_path}")
