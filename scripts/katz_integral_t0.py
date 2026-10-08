# Reads the bolometric light curve written by bolometric_lightcurve.py (<repo>/outputs/bolometric).
# TEXP env var overrides the explosion MJD (default 60565.58).
import os as _os
import numpy as _np0; _np0.random.seed(int(_os.environ.get('SEED', 20240913)))   # seed used for the published errors
_TEXP=float(_os.environ.get('TEXP', 60565.58))
_BOL=_os.path.join(_os.path.dirname(_os.path.abspath(__file__)),'..','outputs','bolometric',f'bolometric_lightcurve_faraway_{_TEXP:.2f}.txt')
import numpy as np
from scipy import optimize
from astropy.io import ascii
from matplotlib import pyplot as plt
import matplotlib.ticker as ticker

# --- Definitions ---

def Q(t, t0):
    """Energy deposition function."""
    trap = 1.0 - np.exp(-(t0 / np.maximum(t, 1e-6))**2)
    Q_gamma = (6.54 * np.exp(-t / 8.76) + 1.38 * np.exp(-t / 111.4)) * trap
    Q_pos = 4.64*(np.exp(-t / 111.4)-np.exp(-t / 8.76))
    return Q_gamma*1e43 + Q_pos*1e41

def get_L_int_term(L_vals, t_all, idx):
    """Calculates the Observational LHS term (L(t) / Integral L t dt)."""
    # This term depends only on the data, not t0
    denom = np.trapezoid(L_vals[:idx] * t_all[:idx], x=t_all[:idx])
    return L_vals[idx] / denom

def mismatch(t0, L_vals, L_int_term=None):
    """Absolute mismatch (for minimization)."""
    if L_int_term is None:
        L_int_term = get_L_int_term(L_vals, t_all, idx)
    
    # Calculate Theoretical side (RHS)
    denom = np.trapezoid(Q(t_all[:idx], t0) * t_all[:idx], x=t_all[:idx])
    Q_int = Q(t_fit, t0) / denom
    
    return np.abs(L_int_term - Q_int)

def mismatch_percentage(t0, L_vals):
    """Percentage mismatch (for plotting)."""
    L_int = get_L_int_term(L_vals, t_all, idx)
    # We use the absolute mismatch function to ensure consistency
    abs_mis = mismatch(t0, L_vals, L_int_term=L_int)
    return 100.0 * abs_mis / L_int

def fit_t0(L_vals, bounds=(30,150), tol=1e-6):
    """Fit t0."""
    L_int = get_L_int_term(L_vals, t_all, idx)
    
    res = optimize.minimize_scalar(
        lambda t0: mismatch(t0, L_vals, L_int_term=L_int),
        bounds=bounds, method='bounded'
    )
    if res.success and res.fun < tol and bounds[0] < res.x < bounds[1]:
        return res.x
    else:
        return np.nan

# --- Main Execution ---

# 1. Load Data

L_data = ascii.read(_BOL)
L_all = np.array(L_data['Lbol(erg/s)'])
Le_all = np.array(L_data['Lbol_err(erg/s)'])
t_all = np.array(L_data['MJD']) - _TEXP
print("Data loaded successfully.")

# Find the t closest to 57 days
idx = np.argmin(np.abs(t_all - 57))
t_fit = t_all[idx]

# 2. Best fit
best_t0 = fit_t0(L_all)
print(f"Best-fit t0 = {best_t0:.2f} days")

# 3. Calculate Noise Floor (1 Sigma Mismatch)
# We need to know how much the mismatch naturally 'jitters' due to error bars
# to define what "1 sigma" means on the y-axis.
print("Calculating mismatch noise floor (MC)...")
mismatch_noise_values = []
for _ in range(1000):
    L_rand = L_all + np.random.normal(0, Le_all)
    # Measure mismatch at the best t0 using perturbed data
    mismatch_noise_values.append(mismatch(best_t0, L_rand))

sigma_abs_mismatch = np.std(mismatch_noise_values)
print(f"1-Sigma Mismatch Noise: {sigma_abs_mismatch:.2e}")

# Calculate conversion factor: Percent = (Abs / L_int) * 100
# So: Sigmas = (Percent / 100 * L_int) / sigma_abs
L_int_val = get_L_int_term(L_all, t_all, idx)
pct_to_sigma_factor = L_int_val / (100.0 * sigma_abs_mismatch)

# 4. Monte Carlo for Histogram (from your original code)
Nmc = int(_os.environ.get('NMC', 30000))   # 30000 for the published errors
t0_samples = []
for _ in range(Nmc):
    L_rand = L_all + np.random.normal(0, Le_all)
    t0 = fit_t0(L_rand)
    if not np.isnan(t0):
        t0_samples.append(t0)
t0_samples = np.array(t0_samples)

# Robust stats
t0_med = np.median(t0_samples)
t0_low_3, t0_high_3 = np.percentile(t0_samples, [0.135, 99.865])
t0_low_2, t0_high_2 = np.percentile(t0_samples, [2.275, 97.725])
t0_low_1, t0_high_1 = np.percentile(t0_samples, [15.865, 84.135])

print(f"1-Sigma range: {t0_low_1:.2f} -- {t0_high_1:.2f} days")
print(f"t0 = {best_t0:.2f} + {t0_high_1 - best_t0:.2f} / - {best_t0 - t0_low_1:.2f} days")
print(f"[rev3] 3-sigma lower limit (0.135 pct): t0 > {t0_low_3:.1f} d ; median {t0_med:.2f}")


# --- Plotting ---

# Plot 1: Histogram
plt.figure(figsize=(7,5)) 
plt.hist(t0_samples, bins=30, alpha=0.7, color="steelblue", edgecolor="k")
plt.axvline(best_t0, color="red", linestyle="--", label=f"Best-fit: {best_t0:.2f} days")
plt.axvspan(t0_low_2, t0_high_2, color="orange", alpha=0.15, label="95% CI")
plt.axvspan(t0_low_3, t0_high_3, color="red", alpha=0.1, label="99.7% CI")
plt.xlabel("t0 [days]")
plt.ylabel("Count")
plt.legend()
plt.tight_layout()
plt.show()

# Plot 2: Absolute Mismatch (skipped for brevity, focusing on the requested one)

# Plot 3: Percentage Mismatch with Sigma Axis (The Requested Plot)
t0_grid = np.linspace(30, 150, 300)
mismatches_pct = [mismatch_percentage(t0, L_all) for t0 in t0_grid]

fig, ax1 = plt.subplots(figsize=(8, 6))

# 1. Primary Axis (Percentage)
ax1.plot(t0_grid, mismatches_pct, color="darkgreen", lw=2, label="Mismatch")
ax1.set_xlabel(r"$t_0$ [days]", fontsize=12)
ax1.set_ylabel("Mismatch [%]", color="darkgreen", fontsize=12)
ax1.set_yscale('log')
ax1.tick_params(axis='y', labelcolor="darkgreen")
ax1.axvline(best_t0, color="red", linestyle="--", label=f"Best-fit: {best_t0:.1f} d")

# 2. Secondary Axis (Sigmas)
ax2 = ax1.twinx()
ax2.set_yscale('log')
ax2.set_ylabel(r"Significance [$\sigma$]", color="navy", fontsize=12)
ax2.tick_params(axis='y', labelcolor="navy")

# Sync the view limits so the scales match physically
y1_min, y1_max = ax1.get_ylim()
ax2.set_ylim(y1_min, y1_max)

# Calculate where the Sigma ticks should fall on the Percentage scale
# Relation: Percent = Sigma / pct_to_sigma_factor
sigma_levels = [1, 2, 3, 4, 5]
sigma_tick_locs = [s / pct_to_sigma_factor for s in sigma_levels]

# Force the Right Axis to put ticks ONLY at these calculated positions
# But label them as integers "1", "2", "3"...
ax2.set_yticks(sigma_tick_locs)
ax2.set_yticklabels([str(s) for s in sigma_levels])

# Add visual grid lines for these sigma levels
for pct, sig in zip(sigma_tick_locs, sigma_levels):
    ax1.axhline(pct, color="navy", alpha=0.3, linestyle=":", lw=1)
    if sig == 5:
        ax1.text(t0_grid[0], pct*1.1, r" $5\sigma$ Rejection", color="navy", fontsize=9, va="bottom")

# Final Polish
plt.title(r"Energy Conservation Mismatch vs. $t_0$", fontsize=14)
ax1.legend(loc='upper center')
plt.tight_layout()
plt.show()



# --- Minimal Katz plot for Nature ---

# sort by time
order = np.argsort(t_all)
t = t_all[order]
L = L_all[order]

# OPTIONAL: truncate to observed range you want to show (e.g. <= 57 d)
tmax = 57
m = t <= tmax
t = t[m]
L = L[m]

# cumulative Katz integrals: ∫ L(t) t dt and ∫ Q(t,t0) t dt
I_rad = np.array([np.trapezoid(L[:i+1] * t[:i+1], x=t[:i+1]) for i in range(len(t))])
I_dep = np.array([np.trapezoid(Q(t[:i+1], best_t0) * t[:i+1], x=t[:i+1]) for i in range(len(t))])

# normalize
I_rad /= I_rad[-1]
I_dep /= I_dep[-1]

# residual (%)
resid = 100 * (I_rad - I_dep) / np.maximum(I_rad, 1e-12)

fig, ax = plt.subplots(figsize=(7.0, 4.2))
ax.plot(t, I_rad, lw=2.5, label="Radiated",color='blue')
ax.plot(t, I_dep, lw=2.5, label=rf"Deposited ($t_0={best_t0:.1f}$ d)",color='red',ls='--')
ax.set_xlabel("Time since explosion [days]")
ax.set_ylabel("Normalized cumulative integral")
ax.set_ylim(-0.02, 1.05)
ax.legend(frameon=False, loc="lower right")
ax.set_title("Katz integral test")

# small residual inset
# axins = ax.inset_axes([0.15, 0.55, 0.35, 0.35])  # [x0,y0,w,h] in axes fraction
# axins.plot(t, resid, lw=2)
# axins.axhline(0, lw=1)
# axins.set_ylim(-5, 5)  # tighten for Nature
# axins.set_xticks([20, 40, 60])
# axins.set_yticks([-5, 0, 5])
# axins.set_title("Residual [%]", fontsize=9)

plt.tight_layout()
plt.show()