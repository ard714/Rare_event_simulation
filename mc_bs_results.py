"""
Monte Carlo European Call Option Pricing with Black-Scholes Comparison
=======================================================================
This script:
  1. Computes the Black-Scholes analytical price C_BS for a European call.
  2. Runs a Monte Carlo simulation (GBM paths) to estimate the call price C_bar.
  3. Prints a comprehensive results table including:
       - Black-Scholes price          C_BS
       - MC mean price                C_bar
       - MC standard deviation        s_M  (of discounted payoffs)
       - 95 % confidence interval     [lower, upper]
       - Absolute error               |C_bar - C_BS|
       - Relative error (%)
       - Theoretical SE               sample_std_payoffs / sqrt(N)
"""

import numpy as np
from scipy.stats import norm

# ──────────────────────────────────────────────
# 1.  Parameters
# ──────────────────────────────────────────────
np.random.seed(42)

S0      = 1000              # initial spot price
K       = 1020              # strike price
T       = 1.0 / 36          # time to maturity (in years)
r       = 0.05              # risk-free rate
sigma   = 0.25              # volatility
dt      = 1.0 / (252*24*60) # time step (minute-level within trading days)
n_steps = int(T / dt)       # number of time steps
n_paths = 1000              # number of Monte Carlo paths

# ──────────────────────────────────────────────
# 2.  Black-Scholes analytical price
# ──────────────────────────────────────────────
def black_scholes_call(S0, K, T, r, sigma):
    """
    Compute the Black-Scholes price for a European call option.

    Parameters
    ----------
    S0    : float – initial spot price
    K     : float – strike price
    T     : float – time to maturity (years)
    r     : float – risk-free rate
    sigma : float – volatility

    Returns
    -------
    C_BS  : float – Black-Scholes call price
    """
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    C_BS = S0 * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return C_BS

C_BS = black_scholes_call(S0, K, T, r, sigma)

# ──────────────────────────────────────────────
# 3.  Monte Carlo simulation  (GBM paths)
# ──────────────────────────────────────────────
paths = np.zeros((n_steps + 1, n_paths))
paths[0] = S0

for t in range(1, n_steps + 1):
    z           = np.random.standard_normal(n_paths)
    drift_part  = (r - 0.5 * sigma**2) * dt
    random_part = sigma * np.sqrt(dt) * z
    exponent    = drift_part + random_part
    paths[t]    = paths[t - 1] * np.exp(exponent)

# Terminal payoffs (un-discounted)
payoffs_undiscounted = np.maximum(paths[-1] - K, 0.0)

# Discounted payoffs
discount_factor      = np.exp(-r * T)
payoffs_discounted   = discount_factor * payoffs_undiscounted

# ──────────────────────────────────────────────
# 4.  Compute all requested statistics
# ──────────────────────────────────────────────

# MC mean price  (discounted expected payoff)
C_bar = np.mean(payoffs_discounted)

# Sample standard deviation of the *discounted* payoffs
s_M   = np.std(payoffs_discounted, ddof=1)   # ddof=1 for sample std

# Theoretical Standard Error  =  sample_std / sqrt(N)
SE    = s_M / np.sqrt(n_paths)

# 95 % confidence interval  (using normal approximation)
z_alpha_half = norm.ppf(0.975)          # ≈ 1.96
CI_lower     = C_bar - z_alpha_half * SE
CI_upper     = C_bar + z_alpha_half * SE

# Absolute error
abs_error = np.abs(C_bar - C_BS)

# Relative error (%)
rel_error_pct = (abs_error / C_BS) * 100.0 if C_BS != 0 else float('inf')

# ──────────────────────────────────────────────
# 5.  Print results
# ──────────────────────────────────────────────
print("=" * 60)
print("  Monte Carlo vs Black-Scholes  –  European Call Option")
print("=" * 60)
print()
print("  Parameters")
print("  " + "-" * 56)
print(f"  S0           = {S0}")
print(f"  K            = {K}")
print(f"  T            = {T:.6f}  ({T*365:.2f} days)")
print(f"  r            = {r}")
print(f"  sigma        = {sigma}")
print(f"  dt           = {dt:.10e}")
print(f"  n_steps      = {n_steps}")
print(f"  n_paths      = {n_paths}")
print()
print("  Results")
print("  " + "-" * 56)
print(f"  Black-Scholes price   C_BS       = {C_BS:.6f}")
print(f"  MC mean price         C_bar      = {C_bar:.6f}")
print(f"  MC std deviation      s_M        = {s_M:.6f}")
print(f"  Theoretical SE        SE         = {SE:.6f}")
print(f"  95% CI lower                     = {CI_lower:.6f}")
print(f"  95% CI upper                     = {CI_upper:.6f}")
print(f"  Absolute error  |C_bar - C_BS|   = {abs_error:.6f}")
print(f"  Relative error  (%)              = {rel_error_pct:.4f} %")
print()
print("=" * 60)
print()

# ──────────────────────────────────────────────
# 6.  Additional diagnostics (spot-price stats)
# ──────────────────────────────────────────────
print("  Additional Diagnostics")
print("  " + "-" * 56)
print(f"  Mean final spot price            = {np.mean(paths[-1]):.6f}")
print(f"  Std  final spot price            = {np.std(paths[-1]):.6f}")
print(f"  Mean payoff (undiscounted)       = {np.mean(payoffs_undiscounted):.6f}")
print(f"  Std  payoff (undiscounted)       = {np.std(payoffs_undiscounted):.6f}")
print(f"  Discount factor  exp(-rT)       = {discount_factor:.10f}")
print(f"  Fraction of paths ITM           = {np.mean(payoffs_undiscounted > 0)*100:.2f} %")
print()
print("=" * 60)

# ──────────────────────────────────────────────
# 7.  Figure 1 — GBM diagnostics (3-panel)
# ──────────────────────────────────────────────
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# ── time grid for x-axis of path plot ──
time_grid = np.linspace(0, T, n_steps + 1)

# ─────────────────────────────────────────
# Left panel: 50 simulated GBM paths
# ─────────────────────────────────────────
ax0 = axes[0]
n_show = min(50, n_paths)
for i in range(n_show):
    ax0.plot(time_grid, paths[:, i], linewidth=0.5, alpha=0.6)
ax0.axhline(y=K, color='red', linestyle='--', linewidth=1.5, label=f'K = {K}')
ax0.set_xlabel('Time (years)')
ax0.set_ylabel('Stock Price')
ax0.set_title('GBM Simulated Paths')
ax0.legend(loc='upper left', fontsize=9)

# ─────────────────────────────────────────
# Centre panel: Histogram of MC price estimates
# ─────────────────────────────────────────
ax1 = axes[1]
# Each discounted payoff is one MC "price estimate" for a single path
ax1.hist(payoffs_discounted, bins=50, color='steelblue', edgecolor='white',
         alpha=0.75, density=True, label='MC estimates')
ax1.axvline(x=C_BS, color='red', linestyle='--', linewidth=1.8,
            label=f'$C_{{BS}}$ = {C_BS:.4f}')
ax1.axvline(x=C_bar, color='blue', linestyle='-', linewidth=1.8,
            label=f'$\\bar{{C}}$ = {C_bar:.4f}')
ax1.set_xlabel('Discounted Payoff')
ax1.set_ylabel('Density')
ax1.set_title('Distribution of MC Price Estimates')
ax1.legend(loc='upper right', fontsize=9)

# ─────────────────────────────────────────
# Right panel: MC Convergence
# ─────────────────────────────────────────
ax2 = axes[2]

# Cumulative mean and cumulative SE as paths accumulate
n_runs       = np.arange(1, n_paths + 1)
cum_mean     = np.cumsum(payoffs_discounted) / n_runs
cum_var      = np.zeros(n_paths)
# Running sample variance (Welford-style, but simple here)
for j in range(n_paths):
    if j == 0:
        cum_var[j] = 0.0
    else:
        cum_var[j] = np.var(payoffs_discounted[:j+1], ddof=1)
cum_se = np.sqrt(cum_var / n_runs)

ax2.plot(n_runs, cum_mean, color='blue', linewidth=1.2, label='Cumulative Mean')
ax2.fill_between(n_runs,
                 cum_mean - 1.96 * cum_se,
                 cum_mean + 1.96 * cum_se,
                 color='blue', alpha=0.15, label='±1.96 SE band')
ax2.axhline(y=C_BS, color='red', linestyle='--', linewidth=1.5,
            label=f'$C_{{BS}}$ = {C_BS:.4f}')
ax2.set_xlabel('Number of Runs')
ax2.set_ylabel('Cumulative Mean Price')
ax2.set_title('MC Convergence')
ax2.legend(loc='upper right', fontsize=9)

# ── Overall title & save ──
fig.suptitle('GBM Monte Carlo Diagnostics', fontsize=15, fontweight='bold', y=1.02)
fig.tight_layout()

save_path = 'fig_gbm_diagnostics.png'
fig.savefig(save_path, dpi=150, bbox_inches='tight')
print(f"\n  Figure saved -> {save_path}")
plt.show()
