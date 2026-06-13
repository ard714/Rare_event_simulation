"""
Advanced Monte Carlo Models – European Call Option Pricing
GBM, Heston, CIR (stochastic rate), CGMY (Levy)
Produces Table 7 results + Figures 2-5.
"""
import numpy as np
from scipy.stats import norm, gaussian_kde
from scipy.special import gamma as gamma_fn
from scipy.integrate import quad
import matplotlib.pyplot as plt
import warnings, time
warnings.filterwarnings("ignore")

# ============================================================
# Parameters
# ============================================================
np.random.seed(42)
S0       = 1000
K        = 1020
T        = 1.0 / 36
r_const  = 0.05
n_paths  = 1000
n_steps  = 360
dt       = T / n_steps

# GBM
sigma_gbm = 0.25

# Heston
v0_h = 0.09;  kappa_h = 2.0;  theta_h = 0.09;  xi_h = 0.05;  rho_h = -0.5

# CIR
r0_c = 0.05;  kappa_c = 2.0;  theta_c = 0.05;  sig_c = 0.10;  sig_eq = 0.25

# CGMY
C_g = 1.0;  G_g = 5.0;  M_g = 5.0;  Y_g = 0.5;  eps_g = 1e-4

# ============================================================
# Helper: Pareto jump sampler for CGMY
# ============================================================
def _sample_jumps(n, lam, Y, eps):
    if n == 0:
        return np.array([])
    out = []
    rem = n
    while rem > 0:
        batch = max(rem * 3, 256)
        U = np.random.uniform(1e-15, 1.0, batch)
        u = U ** (-1.0 / Y)          # Pareto(Y) >= 1
        V = np.random.uniform(0, 1, batch)
        acc = V < np.exp(-lam * eps * (u - 1.0))
        out.append(eps * u[acc])
        rem -= int(acc.sum())
    return np.concatenate(out)[:n]

# ============================================================
# 1. GBM
# ============================================================
print("Simulating GBM ..."); t0 = time.time()
paths_gbm = np.zeros((n_steps+1, n_paths)); paths_gbm[0] = S0
for t in range(1, n_steps+1):
    z = np.random.standard_normal(n_paths)
    paths_gbm[t] = paths_gbm[t-1]*np.exp((r_const-0.5*sigma_gbm**2)*dt
                                          + sigma_gbm*np.sqrt(dt)*z)
disc_gbm = np.exp(-r_const*T)
print(f"  done  ({time.time()-t0:.1f}s)")

# ============================================================
# 2. Heston
# ============================================================
print("Simulating Heston ..."); t0 = time.time()
paths_hes = np.zeros((n_steps+1, n_paths)); paths_hes[0] = S0
var_hes   = np.zeros((n_steps+1, n_paths)); var_hes[0]   = v0_h
for t in range(1, n_steps+1):
    z1 = np.random.standard_normal(n_paths)
    z2 = np.random.standard_normal(n_paths)
    ws = z1;  wv = rho_h*z1 + np.sqrt(1-rho_h**2)*z2
    vp = np.maximum(var_hes[t-1], 0.0)
    sv = np.sqrt(vp)
    var_hes[t] = vp + kappa_h*(theta_h - vp)*dt + xi_h*sv*np.sqrt(dt)*wv
    var_hes[t] = np.maximum(var_hes[t], 0.0)
    paths_hes[t] = paths_hes[t-1]*np.exp((r_const-0.5*vp)*dt + sv*np.sqrt(dt)*ws)
disc_hes = np.exp(-r_const*T)
print(f"  done  ({time.time()-t0:.1f}s)")

# ============================================================
# 3. CIR short-rate
# ============================================================
print("Simulating CIR ..."); t0 = time.time()
paths_cir = np.zeros((n_steps+1, n_paths)); paths_cir[0] = S0
rate_cir  = np.zeros((n_steps+1, n_paths)); rate_cir[0]  = r0_c
for t in range(1, n_steps+1):
    zr = np.random.standard_normal(n_paths)
    zs = np.random.standard_normal(n_paths)
    rp = np.maximum(rate_cir[t-1], 0.0)
    rate_cir[t] = rp + kappa_c*(theta_c - rp)*dt + sig_c*np.sqrt(rp*dt)*zr
    rate_cir[t] = np.maximum(rate_cir[t], 0.0)
    paths_cir[t] = paths_cir[t-1]*np.exp((rp - 0.5*sig_eq**2)*dt + sig_eq*np.sqrt(dt)*zs)
disc_cir = np.exp(-np.trapz(rate_cir, dx=dt, axis=0))
print(f"  done  ({time.time()-t0:.1f}s)")

# ============================================================
# 4. CGMY
# ============================================================
print("Simulating CGMY ..."); t0 = time.time()
gny = gamma_fn(-Y_g)
omega_cg = -np.real(C_g*gny*((M_g+1)**Y_g - M_g**Y_g + (G_g-1)**Y_g - G_g**Y_g))

fp = lambda x: C_g*np.exp(-M_g*x)*x**(-(1+Y_g))
fn = lambda x: C_g*np.exp(-G_g*x)*x**(-(1+Y_g))
lam_p,_ = quad(fp, eps_g, np.inf)
lam_n,_ = quad(fn, eps_g, np.inf)

fvp = lambda x: x**2*C_g*np.exp(-M_g*x)*x**(-(1+Y_g))
fvn = lambda x: x**2*C_g*np.exp(-G_g*x)*x**(-(1+Y_g))
vs_p,_ = quad(fvp, 0, eps_g);  vs_n,_ = quad(fvn, 0, eps_g)
var_small = vs_p + vs_n

fmp = lambda x: x*C_g*np.exp(-M_g*x)*x**(-(1+Y_g))
fmn = lambda x: x*C_g*np.exp(-G_g*x)*x**(-(1+Y_g))
ms_p,_ = quad(fmp, 0, eps_g);  ms_n,_ = quad(fmn, 0, eps_g)
mean_small = ms_p - ms_n

paths_cgmy = np.zeros((n_steps+1, n_paths)); paths_cgmy[0] = S0
logS = np.full(n_paths, np.log(float(S0)))
for t in range(1, n_steps+1):
    # positive jumps
    np_j = np.random.poisson(lam_p*dt, n_paths)
    tp = np_j.sum()
    ps = np.zeros(n_paths)
    if tp > 0:
        aj = _sample_jumps(tp, M_g, Y_g, eps_g)
        idx=0
        for i in range(n_paths):
            if np_j[i]>0: ps[i]=aj[idx:idx+np_j[i]].sum(); idx+=np_j[i]
    # negative jumps
    nn_j = np.random.poisson(lam_n*dt, n_paths)
    tn = nn_j.sum()
    ns = np.zeros(n_paths)
    if tn > 0:
        aj = _sample_jumps(tn, G_g, Y_g, eps_g)
        idx=0
        for i in range(n_paths):
            if nn_j[i]>0: ns[i]=aj[idx:idx+nn_j[i]].sum(); idx+=nn_j[i]
    z = np.random.standard_normal(n_paths)
    small_inc = mean_small*dt + np.sqrt(var_small*dt)*z
    logS += (r_const + omega_cg)*dt + ps - ns + small_inc
    paths_cgmy[t] = np.exp(logS)
disc_cgmy = np.exp(-r_const*T)
print(f"  done  ({time.time()-t0:.1f}s)")

# ============================================================
# Compute results for each model
# ============================================================
from scipy.stats import skew, kurtosis

def results(name, ST, disc):
    payoff = disc * np.maximum(ST - K, 0.0)
    price  = np.mean(payoff)
    payoff_std = np.std(payoff, ddof=1)
    se     = payoff_std / np.sqrt(len(payoff))
    ci_lo  = price - 1.96*se
    ci_hi  = price + 1.96*se
    itm    = np.mean(ST > K)*100
    return dict(name=name, price=price, payoff_std=payoff_std, se=se,
                ci_lo=ci_lo, ci_hi=ci_hi,
                mean=np.mean(ST), std=np.std(ST), skew=skew(ST),
                kurt=kurtosis(ST), itm=itm, ST=ST, payoff=payoff)

R = [results("GBM",    paths_gbm[-1],  disc_gbm),
     results("Heston", paths_hes[-1],  disc_hes),
     results("CIR",    paths_cir[-1],  disc_cir),
     results("CGMY",   paths_cgmy[-1], disc_cgmy)]

# ============================================================
# Table 7
# ============================================================
hdr = f"{'Metric':<28}"
for d in R: hdr += f"  {d['name']:>12}"
sep = "-"*len(hdr)
print("\n" + sep)
print("  Table 7 — Monte Carlo Results (European Call, S0=1000, K=1020)")
print(sep)
print(hdr)
print(sep)
for lab,key,fmt in [("Option price",        "price",      ".4f"),
                     ("Std Dev (payoffs)",   "payoff_std", ".4f"),
                     ("Standard error",      "se",         ".4f"),
                     ("95% CI lower",        "ci_lo",      ".4f"),
                     ("95% CI upper",        "ci_hi",      ".4f"),
                     ("Mean S_T",            "mean",       ".2f"),
                     ("Std Dev S_T",         "std",        ".2f"),
                     ("Skewness S_T",        "skew",       ".4f"),
                     ("Excess kurtosis S_T", "kurt",       ".4f"),
                     ("% ITM (S_T > K)",     "itm",        ".2f")]:
    row = f"  {lab:<26}"
    for d in R: row += f"  {d[key]:>12{fmt}}"
    print(row)
print(sep + "\n")

# ============================================================
# Figure 2 — Simulated stock price paths (advanced models)
# ============================================================
time_grid = np.linspace(0, T, n_steps+1)
n_show = 50

fig2, axes2 = plt.subplots(1, 3, figsize=(17, 4.5))
for ax, pths, ttl in zip(axes2,
                         [paths_hes, paths_cir, paths_cgmy],
                         ["Heston", "CIR", "CGMY"]):
    for i in range(n_show):
        ax.plot(time_grid, pths[:, i], lw=0.5, alpha=0.55)
    ax.axhline(K, color='red', ls='--', lw=1.4, label=f'K={K}')
    ax.set_xlabel("Time");  ax.set_ylabel("Stock Price")
    ax.set_title(ttl);      ax.legend(fontsize=8, loc='upper left')
fig2.suptitle("Simulated Stock Price Paths", fontweight='bold', fontsize=14, y=1.01)
fig2.tight_layout()
fig2.savefig("fig_advanced_paths.png", dpi=150, bbox_inches='tight')
print("  Saved fig_advanced_paths.png")

# ============================================================
# Figure 3 — Auxiliary process paths
# ============================================================
fig3, (ax3a, ax3b) = plt.subplots(1, 2, figsize=(12, 4.5))
for i in range(n_show):
    ax3a.plot(time_grid, var_hes[:, i], lw=0.5, alpha=0.55)
ax3a.axhline(theta_h, color='red', ls='--', lw=1.4, label=f'theta={theta_h}')
ax3a.set_xlabel("Time"); ax3a.set_ylabel("Variance")
ax3a.set_title("Heston Variance Process $v_t$"); ax3a.legend(fontsize=9)

for i in range(n_show):
    ax3b.plot(time_grid, rate_cir[:, i], lw=0.5, alpha=0.55)
ax3b.axhline(theta_c, color='red', ls='--', lw=1.4, label=f'theta={theta_c}')
ax3b.set_xlabel("Time"); ax3b.set_ylabel("Short Rate")
ax3b.set_title("CIR Short Rate $r_t$"); ax3b.legend(fontsize=9)

fig3.tight_layout()
fig3.savefig("fig_auxiliary_paths.png", dpi=150, bbox_inches='tight')
print("  Saved fig_auxiliary_paths.png")

# ============================================================
# Figure 4 — Terminal price distributions (KDE)
# ============================================================
fig4, ax4 = plt.subplots(figsize=(8, 5))
colors = ['steelblue','darkorange','green','purple']
for d, c in zip(R, colors):
    kde = gaussian_kde(d['ST'])
    xg  = np.linspace(d['ST'].min(), d['ST'].max(), 500)
    ax4.plot(xg, kde(xg), color=c, lw=1.6, label=d['name'])
ax4.axvline(K, color='red', ls='--', lw=1.4, label=f'K={K}')
ax4.set_xlabel("$S_T$"); ax4.set_ylabel("Density")
ax4.set_title("Distribution of Terminal Stock Prices $S_T$")
ax4.legend(); fig4.tight_layout()
fig4.savefig("fig_terminal_distributions.png", dpi=150, bbox_inches='tight')
print("  Saved fig_terminal_distributions.png")

# ============================================================
# Figure 5 — Payoff distributions (2x2)
# ============================================================
fig5, axes5 = plt.subplots(2, 2, figsize=(11, 8))
for ax, d, c in zip(axes5.flat, R, colors):
    p = d['payoff']
    clip = np.percentile(p[p>0], 99) if (p>0).any() else 1.0
    ax.hist(p[p <= clip], bins=50, color=c, edgecolor='white', alpha=0.75, density=True)
    ax.set_title(d['name']); ax.set_xlabel("Discounted Payoff"); ax.set_ylabel("Density")
fig5.suptitle("Distribution of Option Payoffs", fontweight='bold', fontsize=14, y=1.01)
fig5.tight_layout()
fig5.savefig("fig_payoff_distributions.png", dpi=150, bbox_inches='tight')
print("  Saved fig_payoff_distributions.png")

plt.show()
