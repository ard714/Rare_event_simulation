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

S0 = 100.0
K = 150.0
T = 30 / 365
r = 0.05
sigma = 0.50

THRESHOLD = K / S0
DIM = 1

N_STEPS = 30
DT = T / N_STEPS

N_PATHS = 150
BATCH_SIZE = 75
EPOCHS = 200

LR_RATE = 1e-3
LR_GAMMA = 0.98

K1, K2, K3 = 0.35, 0.45, 0.20

N_LEVELS = 20
_beta = np.logspace(np.log10(0.1), np.log10(1.0), N_LEVELS)
EPS_SCHEDULE = (1.0 / _beta)[::-1].copy()

MC_COUNTS = [100, 500, 1_000, 5_000, 10_000, 50_000, 100_000]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUTPUT_DIR = os.getcwd()

np.random.seed(42)
torch.manual_seed(42)

print("=" * 65)
print("  Deep OTM Option Pricing  |  IS + DNN  vs  Standard MC")
print("=" * 65)
print(f"  S0={S0},  K={K},  T={T*365:.0f} days,  sigma={sigma},  r={r}")
print(f"  Normalised threshold : {THRESHOLD}")
print(f"  Device               : {DEVICE}")
print(f"  Eps schedule (first→last): {EPS_SCHEDULE[0]:.2f} → {EPS_SCHEDULE[-1]:.2f}")
print("=" * 65)


def bs_call(S0, K, T, r, sigma):
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    price = S0 * scipy_norm.cdf(d1) - K * np.exp(-r * T) * scipy_norm.cdf(d2)
    prob = scipy_norm.cdf(d2)
    return price, d1, d2, prob


BS_PRICE, d1_bs, d2_bs, BS_PROB = bs_call(S0, K, T, r, sigma)
print(f"\n  Black-Scholes price   : ${BS_PRICE:.6f}")
print(f"  Risk-neutral P(ITM)   : {BS_PROB:.6f}  ({BS_PROB*100:.4f}%)")
print(f"  d1={d1_bs:.4f},  d2={d2_bs:.4f}")


def run_standard_mc(n_paths, S0, K, T, r, sigma, n_steps):
    dt_mc = T / n_steps
    S = np.full(n_paths, S0, dtype=np.float64)
    for _ in range(n_steps):
        Z = np.random.randn(n_paths)
        S *= np.exp((r - 0.5 * sigma**2) * dt_mc + sigma * np.sqrt(dt_mc) * Z)
    payoff = np.maximum(S - K, 0.0)
    disc = np.exp(-r * T)
    price = disc * np.mean(payoff)
    se = disc * np.std(payoff, ddof=1) / np.sqrt(n_paths)
    hits = int(np.sum(payoff > 0))
    return price, se, hits


print("\n  Standard MC results:")
print(
    f"  {'Paths':>10}  {'Price ($)':>12}  {'+-2sigma':>12}  {'Hits':>6}  {'Time(s)':>9}"
)
print(f"  {'-'*57}")

mc_prices, mc_ses, mc_times, mc_cum_times = [], [], [], []
_mc_cumtime = 0.0
for n in MC_COUNTS:
    _t0 = time.perf_counter()
    p, se, hits = run_standard_mc(n, S0, K, T, r, sigma, N_STEPS)
    _elapsed = time.perf_counter() - _t0
    mc_prices.append(p)
    mc_ses.append(se)
    mc_times.append(_elapsed)
    _mc_cumtime += _elapsed
    mc_cum_times.append(_mc_cumtime)
    tag = "  <- ZERO (misprice!)" if p < 1e-9 else ""
    print(f"  {n:>10,d}  ${p:>10.6f}  +-{2*se:.6f}  {hits:>6d}  {_elapsed:>8.4f}s{tag}")


def _drift(x):
    return r * x


def _diffusion(x):
    return sigma * x


def _terminal_cost(x):
    return torch.clamp(THRESHOLD - x, min=0.0) ** 2


def make_nn():
    net = nn.Sequential(
        nn.Linear(1 + DIM, 32),
        nn.Tanh(),
        nn.Linear(32, 128),
        nn.Tanh(),
        nn.Linear(128, 64),
        nn.Tanh(),
        nn.Linear(64, 32),
        nn.Tanh(),
        nn.Linear(32, 1),
    ).to(DEVICE)
    for layer in net:
        if isinstance(layer, nn.Linear):
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)
    return net


def get_control(model, t, x, epsilon):
    if model is None:
        return torch.zeros_like(x)
    with torch.enable_grad():
        xr = x.detach().clone().requires_grad_(True)
        w = model(torch.cat([t.detach(), xr], dim=1))
        dw_dx = torch.autograd.grad(w, xr, torch.ones_like(w), create_graph=False)[0]
        u = 0.5 * _diffusion(xr) * dw_dx * np.sqrt(epsilon)
    return u.detach()


def generate_paths(ctrl_model):
    x = torch.ones(N_PATHS, DIM, device=DEVICE)
    ht, hx = [], []
    for step in range(N_STEPS + 1):
        t = torch.full((N_PATHS, 1), step * DT, device=DEVICE)
        ht.append(t.detach())
        hx.append(x.detach())
        if step < N_STEPS:
            u = get_control(ctrl_model, t, x, 1.0)
            dW = torch.randn_like(x) * np.sqrt(DT)
            sx = _diffusion(x)
            x = (x + (_drift(x) + sx * u) * DT + sx * dW).clamp(min=1e-6)
    return torch.stack(ht), torch.stack(hx)


def hjb_loss(model, t_raw, x_raw, epsilon):
    t = t_raw.clone().detach().requires_grad_(True)
    x = x_raw.clone().detach().requires_grad_(True)

    w = model(torch.cat([t, x], dim=1))

    grads = torch.autograd.grad(
        w, [t, x], torch.ones_like(w), create_graph=True, retain_graph=True
    )
    dw_dt = grads[0]
    dw_dx = grads[1]

    dw_dxx = torch.autograd.grad(
        dw_dx, x, torch.ones_like(dw_dx), create_graph=True, retain_graph=True
    )[0]

    sx = _diffusion(x)
    bx = _drift(x)

    mask_t0 = (t.detach() <= DT * 0.5).float()
    n_t0 = mask_t0.sum().clamp(min=1.0)
    L1 = -(mask_t0 * w).sum() / n_t0

    mask_interior = (t.detach() < T - DT * 0.5).float()
    hjb_res = (
        -dw_dt - bx * dw_dx + 0.5 * (sx * dw_dx) ** 2 - 0.5 * epsilon * sx**2 * dw_dxx
    )
    n_int = mask_interior.sum().clamp(min=1.0)
    L2 = (mask_interior * hjb_res**2).sum() / n_int

    mask_terminal = (t.detach() >= T - DT * 0.5).float()
    n_term = mask_terminal.sum().clamp(min=1.0)
    L3 = (mask_terminal * (w - 2.0 * _terminal_cost(x)) ** 2).sum() / n_term

    return K1 * L1 + K2 * L2 + K3 * L3


def evaluate_is(model, n_paths=N_PATHS):
    model.eval()
    x = torch.ones(n_paths, DIM, device=DEVICE)
    log_lr = torch.zeros(n_paths, 1, device=DEVICE)

    for step in range(N_STEPS):
        t = torch.full((n_paths, 1), step * DT, device=DEVICE)
        u = get_control(model, t, x, 1.0)
        dW = torch.randn_like(x) * np.sqrt(DT)
        sx = _diffusion(x)
        x = (x + (_drift(x) + sx * u) * DT + sx * dW).clamp(min=1e-6)
        log_lr -= u * dW + 0.5 * u**2 * DT

    lr_val = torch.exp(log_lr.clamp(-50, 50))

    p_est = torch.mean((x >= THRESHOLD).float() * lr_val).item()
    payoff_usd = torch.clamp(x - THRESHOLD, min=0.0) * S0
    price_est = np.exp(-r * T) * torch.mean(payoff_usd * lr_val).item()

    return p_est, price_est


print("\n" + "=" * 65)
print(f"  IS + DNN Training  (eps: {EPS_SCHEDULE[0]:.2f} → {EPS_SCHEDULE[-1]:.2f})")
print("=" * 65)

current_model = make_nn()

optimizer = optim.Adam(current_model.parameters(), lr=LR_RATE, weight_decay=1e-5)

prev_model = None
lvl_probs, lvl_prices = [], []
level_losses: list[list[float]] = []
dnn_level_times: list[float] = []
dnn_cum_times: list[float] = []
_dnn_t0 = time.perf_counter()


def get_weight_norm(model):
    return sum(p.data.norm().item() for p in model.parameters())


for lvl, eps in enumerate(EPS_SCHEDULE):
    _lv_t0 = time.perf_counter()
    w_norm_before = get_weight_norm(current_model)
    print(
        f"\n  Level {lvl+1:2d}/{N_LEVELS}  |  eps={eps:.4f}  "
        f"|  weight_norm_before={w_norm_before:.4f}"
    )

    current_model.train()

    for pg in optimizer.param_groups:
        pg["lr"] = LR_RATE * (0.90**lvl)

    ts, xs = generate_paths(prev_model)

    loader = DataLoader(
        TensorDataset(ts.reshape(-1, 1), xs.reshape(-1, DIM)),
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=True,
    )

    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=LR_GAMMA)

    epoch_losses: list[float] = []
    for ep in range(EPOCHS):
        ep_loss = 0.0
        n_batches = 0
        for bt, bx in loader:
            optimizer.zero_grad()
            loss = hjb_loss(current_model, bt, bx, eps)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(current_model.parameters(), 1.0)
            optimizer.step()
            ep_loss += loss.detach().item()
            n_batches += 1

        avg_loss = ep_loss / max(1, n_batches)
        epoch_losses.append(avg_loss)
        scheduler.step()

        if ep % 25 == 0 or ep == EPOCHS - 1:
            print(
                f"    epoch {ep:3d}/{EPOCHS}  avg_loss={avg_loss:.6f}  "
                f"lr={scheduler.get_last_lr()[0]:.2e}"
            )

    level_losses.append(epoch_losses)

    w_norm_after = get_weight_norm(current_model)
    print(
        f"    weight_norm: {w_norm_before:.4f} → {w_norm_after:.4f}  "
        f"(Δ={w_norm_after - w_norm_before:+.4f})"
    )

    p_est, price_est = evaluate_is(current_model)
    lvl_probs.append(p_est)
    lvl_prices.append(price_est)

    _lv_time = time.perf_counter() - _lv_t0
    dnn_level_times.append(_lv_time)
    dnn_cum_times.append(time.perf_counter() - _dnn_t0)

    print(f"    P(S_T>K) | IS: {p_est:.6f}   BS: {BS_PROB:.6f}")
    print(f"    Price    | IS: ${price_est:.6f}   BS: ${BS_PRICE:.6f}")
    print(f"    Level time: {_lv_time:.1f}s  |  Cumul DNN: {dnn_cum_times[-1]:.1f}s")

    prev_model = make_nn()
    prev_model.load_state_dict(current_model.state_dict())
    prev_model.eval()

IS_PRICE = lvl_prices[-1]
IS_PROB = lvl_probs[-1]
mc_best = mc_prices[-1]

print("\n" + "=" * 65)
print("  COMPARISON SUMMARY")
print("=" * 65)
print(f"  Option : S0=${S0},  K=${K},  T={T*365:.0f}d,  sigma={sigma},  r={r}")
print()
print(f"  {'Method':<32s}  {'Price':>10s}   {'P(ITM)':>10s}   {'Time(s)':>9s}")
print(f"  {'-'*70}")
print(
    f"  {'Black-Scholes (reference)':<32s}  ${BS_PRICE:>9.6f}   {BS_PROB:.6f}   {'---':>9}"
)
print(
    f"  {'Std MC  (100k paths)':<32s}  ${mc_best:>9.6f}   {'---':>10}   {mc_cum_times[-1]:>8.3f}s"
)
print(
    f"  {'IS + DNN  (eps-annealing)':<32s}  ${IS_PRICE:>9.6f}   {IS_PROB:.6f}   {dnn_cum_times[-1]:>8.1f}s"
)
print()
print(f"  MC  misprice (vs BS) : ${BS_PRICE - mc_best:+.6f}")
print(f"  IS  error    (vs BS) : ${IS_PRICE - BS_PRICE:+.6f}")
print("=" * 65)

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle(
    f"Deep OTM Call  |  S0=${S0}  K=${K}  T={T*365:.0f}d  sigma={sigma}  r={r}\n"
    f"BS=${BS_PRICE:.5f}    IS=${IS_PRICE:.5f}    MC(100k)=${mc_best:.5f}",
    fontsize=11,
    fontweight="bold",
)

levels = np.arange(1, N_LEVELS + 1)
epslbls = [f"L{lv}\ne={e:.1f}" for lv, e in zip(levels, EPS_SCHEDULE)]

ax = axes[0, 0]
ax.semilogx(
    MC_COUNTS, mc_prices, "o-", color="steelblue", lw=2, ms=6, label="Standard MC price"
)
lo = [max(0.0, p - 2 * se) for p, se in zip(mc_prices, mc_ses)]
hi = [p + 2 * se for p, se in zip(mc_prices, mc_ses)]
ax.fill_between(
    MC_COUNTS, lo, hi, alpha=0.20, color="steelblue", label="+/-2sigma band"
)
ax.axhline(
    BS_PRICE, color="crimson", ls="--", lw=2, label=f"Black-Scholes  ${BS_PRICE:.5f}"
)
ax.axhline(
    IS_PRICE,
    color="forestgreen",
    ls="-.",
    lw=2,
    label=f"IS estimate    ${IS_PRICE:.5f}",
)
ax.set_xlabel("Number of MC paths  (log scale)", fontsize=10)
ax.set_ylabel("Option price ($)", fontsize=10)
ax.set_title(
    "Standard Monte Carlo\nReturns $0 until path count is enormous", fontsize=9
)
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

ax = axes[0, 1]
ax.plot(
    levels,
    lvl_prices,
    "o-",
    color="forestgreen",
    lw=2,
    ms=6,
    label="IS price (eps=1 eval)",
)
ax.axhline(
    BS_PRICE, color="crimson", ls="--", lw=2, label=f"Black-Scholes  ${BS_PRICE:.5f}"
)
ax.set_xticks(levels)
ax.set_xticklabels(epslbls, fontsize=7)
ax.set_xlabel("Annealing level  (eps: large → 1)", fontsize=10)
ax.set_ylabel("Option price ($)", fontsize=10)
ax.set_title(
    "IS Price Convergence Across Levels\nDNN warm-started; optimizer state persists",
    fontsize=9,
)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

ax = axes[0, 2]
ax.semilogy(
    levels, lvl_probs, "s-", color="darkorange", lw=2, ms=6, label="IS  P(S_T ≥ K)"
)
ax.axhline(BS_PROB, color="crimson", ls="--", lw=2, label=f"BS prob  {BS_PROB:.5f}")
ax.set_xticks(levels)
ax.set_xticklabels(epslbls, fontsize=7)
ax.set_xlabel("Annealing level", fontsize=10)
ax.set_ylabel("P(S_T ≥ K)  [log scale]", fontsize=10)
ax.set_title(
    "Rare Event Probability Estimate\nIS finds it; standard MC would return ~0",
    fontsize=9,
)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

ax = axes[1, 0]
_cmap = plt.cm.plasma
_n = len(level_losses)
for _li, _losses in enumerate(level_losses):
    _col = _cmap(_li / max(1, _n - 1))
    _lbl = (
        f"L{_li+1} eps={EPS_SCHEDULE[_li]:.1f}"
        if _li % 4 == 0 or _li == _n - 1
        else None
    )
    ax.semilogy(
        range(len(_losses)), np.abs(_losses), color=_col, alpha=0.85, lw=1.3, label=_lbl
    )
ax.set_xlabel("Epoch", fontsize=10)
ax.set_ylabel("|Avg Batch Loss|  (log scale)", fontsize=10)
ax.set_title(
    "Loss per Epoch — all annealing levels\n"
    "Loss should DECREASE within each level (bug A fixed)",
    fontsize=9,
)
ax.legend(fontsize=7, loc="upper right")
ax.grid(True, alpha=0.3)

ax = axes[1, 1]
ax.semilogx(
    mc_cum_times, mc_prices, "o-", color="steelblue", lw=2, ms=7, label="Standard MC"
)
lo2 = [max(0.0, p - 2 * se) for p, se in zip(mc_prices, mc_ses)]
hi2 = [p + 2 * se for p, se in zip(mc_prices, mc_ses)]
ax.fill_between(mc_cum_times, lo2, hi2, alpha=0.15, color="steelblue")
ax.semilogx(
    dnn_cum_times, lvl_prices, "s-", color="forestgreen", lw=2, ms=7, label="IS + DNN"
)
ax.axhline(BS_PRICE, color="crimson", ls="--", lw=2, label=f"BS  ${BS_PRICE:.5f}")
ax.set_xlabel("Cumulative wall-clock time (s)  [log scale]", fontsize=10)
ax.set_ylabel("Option price ($)", fontsize=10)
ax.set_title("Convergence vs Wall-clock Time\nIS+DNN vs Standard MC", fontsize=9)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

ax = axes[1, 2]
ax.bar(
    levels, dnn_level_times, color="darkorange", alpha=0.85, edgecolor="white", lw=0.5
)
ax.set_xlabel("Annealing level", fontsize=10)
ax.set_ylabel("Wall-clock time (s)", fontsize=10)
ax.set_title(
    f"DNN Wall-clock Time per Level\n"
    f"Total: {dnn_cum_times[-1]:.1f}s  |  MC(100k): {mc_cum_times[-1]:.3f}s",
    fontsize=9,
)
ax.grid(True, alpha=0.3, axis="y")

plt.tight_layout()
out_path = os.path.join(OUTPUT_DIR, "pricing_deep_otm.png")
plt.savefig(out_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"\n  Plot saved → {out_path}")
