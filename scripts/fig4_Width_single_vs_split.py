"""
Single-population vs. split (bright/faint) linear trend for the SNe Iax
peak-magnitude / decline-rate relation.

Companion to Width.py: Width.py fits a linear trend to the FAINT Iax only.
Here we (a) refit the same relation using the WHOLE Iax sample as one
population, and (b) ask which description the data actually prefer:

    Model S (single) : one line + one intrinsic scatter for all Iax   (3 par)
    Model T (two)    : independent line + scatter for bright & faint  (6 par)

Fits are maximum-likelihood with errors on both axes and an intrinsic
scatter term:  s_i^2 = sigma_y,i^2 + (slope * sigma_x,i)^2 + sigma_int^2

Model comparison: AICc, BIC, a likelihood-ratio statistic, and -- because the
bright/faint split is made on the y-variable itself, which biases a naive
LRT -- a parametric bootstrap under the single-trend null that repeats the
whole split-and-fit procedure on simulated data.

Nothing in Width.py is modified; this script only reads the same input files.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy import stats

rng = np.random.default_rng(20240913)

plt.rcParams.update({
    'font.size': 11,
    'font.family': 'DejaVu Sans',
    'axes.linewidth': 1.2,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
})

import paths
HOME = str(paths.COMPARISON)
# diagnostics are written to outputs/figures/_work
import nature_style as ns
FIGDIR = str(ns.OUT_DIR / '_work')
os.makedirs(FIGDIR, exist_ok=True)
N_BOOT = 2000 if os.environ.get('WIDTH_FAST', '0') != '1' else 40   # WIDTH_FAST=1: quick figure iteration

# --------------------------------------------------------------------------
# 1. Likelihood machinery
# --------------------------------------------------------------------------

def _nll(theta, x, y, sx, sy):
    """Negative log-likelihood of a straight line with intrinsic scatter."""
    slope, intercept, log_sig = theta
    s2 = sy**2 + (slope * sx)**2 + np.exp(2.0 * log_sig)
    resid = y - (slope * x + intercept)
    return 0.5 * np.sum(resid**2 / s2 + np.log(2.0 * np.pi * s2))


def _num_hess(f, theta, eps=1e-4):
    """Central-difference Hessian, for MLE parameter errors."""
    theta = np.asarray(theta, float)
    n = len(theta)
    H = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            ei, ej = np.zeros(n), np.zeros(n)
            ei[i], ej[j] = eps, eps
            H[i, j] = (f(theta + ei + ej) - f(theta + ei - ej)
                       - f(theta - ei + ej) + f(theta - ei - ej)) / (4 * eps**2)
    return 0.5 * (H + H.T)


def fit_line(x, y, sx, sy, label='', fast=False):
    """MLE straight-line fit with intrinsic scatter. Returns a dict.

    fast=True uses a single starting point and looser tolerances; it is only
    used inside the bootstrap, where thousands of fits are needed and only the
    likelihood value matters.
    """
    x, y, sx, sy = map(lambda a: np.asarray(a, float), (x, y, sx, sy))
    n = len(x)
    if n < 3:
        return None

    slope0, icpt0 = np.polyfit(x, y, 1)
    sig0 = max(np.std(y - (slope0 * x + icpt0)), 0.05)
    theta0 = [slope0, icpt0, np.log(sig0)]

    if fast:
        starts = (theta0,)
        opts = dict(maxiter=2000, maxfev=2000, xatol=1e-6, fatol=1e-6)
    else:
        starts = (theta0, [slope0, icpt0, np.log(0.02)], [0.0, np.mean(y), np.log(sig0)])
        opts = dict(maxiter=40000, maxfev=40000, xatol=1e-9, fatol=1e-9)

    best = None
    for start in starts:
        try:
            res = minimize(_nll, start, args=(x, y, sx, sy), method='Nelder-Mead',
                           options=opts)
        except Exception:
            continue
        if res.success or np.isfinite(res.fun):
            if best is None or res.fun < best.fun:
                best = res
    if best is None:
        return None

    slope, icpt, log_sig = best.x
    logL = -best.fun

    if fast:
        try:
            c = np.linalg.inv(_num_hess(lambda t: _nll(t, x, y, sx, sy), best.x))
            se = float(np.sqrt(abs(c[0, 0])))
        except np.linalg.LinAlgError:
            se = np.nan
        return dict(label=label, n=n, slope=slope, intercept=icpt,
                    sig_int=np.exp(log_sig), logL=logL, k=3, slope_err=se)

    # parameter uncertainties from the curvature of the likelihood
    try:
        cov = np.linalg.inv(_num_hess(lambda t: _nll(t, x, y, sx, sy), best.x))
        err = np.sqrt(np.abs(np.diag(cov)))
    except np.linalg.LinAlgError:
        err = np.full(3, np.nan)

    # OLS, for direct comparison with the numbers printed by Width.py
    ols_slope, ols_icpt = np.polyfit(x, y, 1)
    ols_sig = np.std(y - (ols_slope * x + ols_icpt), ddof=2 if n > 2 else 1)

    return dict(label=label, n=n, x=x, y=y, sx=sx, sy=sy,
                slope=slope, intercept=icpt, sig_int=np.exp(log_sig),
                slope_err=err[0], intercept_err=err[1],
                sig_int_err=np.exp(log_sig) * err[2],
                logL=logL, k=3,
                ols_slope=ols_slope, ols_intercept=ols_icpt, ols_sigma=ols_sig)


def pred_sigma(fit, sx, sy):
    """Total predictive sigma of a fit at points with the given errors."""
    return np.sqrt(sy**2 + (fit['slope'] * sx)**2 + fit['sig_int']**2)


def info_criteria(logL, k, n):
    aic = -2 * logL + 2 * k
    aicc = aic + (2 * k * (k + 1) / (n - k - 1)) if n - k - 1 > 0 else np.inf
    bic = -2 * logL + k * np.log(n)
    return aic, aicc, bic


def place_labels(ax, xs, ys, names, fontsize=7.5, color='0.35'):
    """Annotate points, nudging each label to the first non-colliding slot."""
    fig = ax.figure
    fig.canvas.draw()
    candidates = [(6, 4), (6, -11), (-6, 4), (-6, -11), (6, 14), (6, -21),
                  (-6, 14), (-6, -21), (0, 18), (0, -25)]
    placed = []
    for x, y, name in zip(xs, ys, names):
        px, py = ax.transData.transform((x, y))
        w, h = 5.6 * len(str(name)), 9.0
        for dx, dy in candidates:
            x0 = px + dx if dx >= 0 else px + dx - w
            box = (x0, py + dy - h / 2, x0 + w, py + dy + h / 2)
            if all(box[2] < b[0] or box[0] > b[2] or box[3] < b[1] or box[1] > b[3]
                   for b in placed):
                break
        placed.append(box)
        ha = 'left' if dx >= 0 else 'right'
        ax.annotate(name, (x, y), textcoords='offset points', xytext=(dx, dy),
                    fontsize=fontsize, color=color, ha=ha, va='center')


# --------------------------------------------------------------------------
# 2. Data (identical inputs to Width.py, including SN 2024vjm itself)
# --------------------------------------------------------------------------

M_r_24vjm, M_r_err_24vjm, dm15_r_24vjm, dm15r_err_24vjm = -13.53, 0.10, 0.58, 0.13
MB_24vjm, MB_err_24vjm, dm15B_24vjm, dm15B_err_24vjm = -12.68, 0.10, 1.85, 0.05

r_band = pd.read_csv(os.path.join(HOME, 'Iax_r_band.csv'))
b_band = pd.read_csv(os.path.join(HOME, 'Iax_B_band.csv'))


def build(df, cols, vjm, faint_cut, band):
    """Assemble one band's sample (comparison SNe + 2024vjm) into a DataFrame."""
    name_c, m_c, merr_c, dm_c, dmerr_c = cols
    d = pd.DataFrame({
        'Name': list(df[name_c].astype(str).str.strip()) + ['2024vjm'],
        'M': list(df[m_c]) + [vjm[0]],
        'M_err': list(df[merr_c]) + [vjm[1]],
        'dm15': list(df[dm_c]) + [vjm[2]],
        'dm15_err': list(df[dmerr_c]) + [vjm[3]],
    })
    d['faint'] = d['M'] > faint_cut
    d['band'] = band
    # a zero reported error would give an infinitely informative point
    d['M_err'] = d['M_err'].clip(lower=0.01)
    d['dm15_err'] = d['dm15_err'].clip(lower=0.005)
    return d


BANDS = {
    'r': dict(
        data=build(r_band, ('Name', 'M_r', 'M_r_err', 'dm15_r', 'dm15_r_err'),
                   (M_r_24vjm, M_r_err_24vjm, dm15_r_24vjm, dm15r_err_24vjm),
                   faint_cut=-15.0, band='r'),
        cut=-15.0, xlim=(0.35, 1.15), color='crimson',
        xlabel=r'$\Delta m_{15}(r)$ (mag)', ylabel=r'Peak $M_r$ (mag)'),
    'B': dict(
        data=build(b_band, ('Name', 'M_B', 'M_B_err', 'dm15_B', 'dm15_B_err'),
                   (MB_24vjm, MB_err_24vjm, dm15B_24vjm, dm15B_err_24vjm),
                   faint_cut=-15.5, band='B'),
        cut=-15.5, xlim=(1.3, 2.6), color='royalblue',
        xlabel=r'$\Delta m_{15}(B)$ (mag)', ylabel=r'Peak $M_B$ (mag)'),
}


# --------------------------------------------------------------------------
# 3. Fit both models in each band
# --------------------------------------------------------------------------

def fit_models(d, cut):
    """Fit the single-population model and the bright/faint split model."""
    out = {}
    out['single'] = fit_line(d['dm15'], d['M'], d['dm15_err'], d['M_err'], 'All Iax')
    f = d[d['M'] > cut]
    b = d[d['M'] <= cut]
    out['faint'] = fit_line(f['dm15'], f['M'], f['dm15_err'], f['M_err'], 'Faint Iax')
    out['bright'] = fit_line(b['dm15'], b['M'], b['dm15_err'], b['M_err'], 'Bright Iax')
    out['n'] = len(d)
    if out['faint'] and out['bright']:
        out['logL_split'] = out['faint']['logL'] + out['bright']['logL']
        out['k_split'] = 6
    else:
        out['logL_split'] = np.nan
        out['k_split'] = np.nan
    return out


def boot_null(d, cut, fit_single, n_boot=N_BOOT):
    """Parametric bootstrap under the single-trend null.

    Simulate y from the single-population fit at the observed x and errors,
    re-apply the same luminosity cut, refit both models, and record the
    likelihood-ratio statistic.  This calibrates the LRT for the fact that the
    split is made on the simulated y-values themselves.
    """
    x = d['dm15'].to_numpy(float)
    sx = d['dm15_err'].to_numpy(float)
    sy = d['M_err'].to_numpy(float)
    mu = fit_single['slope'] * x + fit_single['intercept']
    s = np.sqrt(sy**2 + (fit_single['slope'] * sx)**2 + fit_single['sig_int']**2)

    stats_null, z_null = [], []
    for _ in range(n_boot):
        y = mu + rng.normal(0.0, s)
        sim = pd.DataFrame({'dm15': x, 'dm15_err': sx, 'M': y, 'M_err': sy})
        ff = sim[sim['M'] > cut]
        fb = sim[sim['M'] <= cut]
        if len(ff) < 3 or len(fb) < 3:
            continue
        fs = fit_line(x, y, sx, sy, fast=True)
        if fs is None:
            continue
        a = fit_line(ff['dm15'], ff['M'], ff['dm15_err'], ff['M_err'], fast=True)
        b = fit_line(fb['dm15'], fb['M'], fb['dm15_err'], fb['M_err'], fast=True)
        if a is None or b is None:
            continue
        stats_null.append(2.0 * (a['logL'] + b['logL'] - fs['logL']))
        zz = np.hypot(a['slope_err'], b['slope_err'])
        if np.isfinite(zz) and zz > 0:
            z_null.append(abs(a['slope'] - b['slope']) / zz)
    return np.array(stats_null), np.array(z_null)


def threshold_scan(d, fit_single, cuts):
    """delta-BIC (split minus single) as a function of where the cut is placed."""
    out = []
    for c in cuts:
        f = d[d['M'] > c]
        b = d[d['M'] <= c]
        if len(f) < 3 or len(b) < 3:
            out.append(np.nan)
            continue
        a = fit_line(f['dm15'], f['M'], f['dm15_err'], f['M_err'])
        bb = fit_line(b['dm15'], b['M'], b['dm15_err'], b['M_err'])
        if a is None or bb is None:
            out.append(np.nan)
            continue
        _, _, bic_split = info_criteria(a['logL'] + bb['logL'], 6, len(d))
        _, _, bic_single = info_criteria(fit_single['logL'], 3, len(d))
        out.append(bic_split - bic_single)
    return np.array(out)


results = {}
print('=' * 78)
for key, cfg in BANDS.items():
    d = cfg['data']
    m = fit_models(d, cfg['cut'])
    n = m['n']

    aic_s, aicc_s, bic_s = info_criteria(m['single']['logL'], 3, n)
    aic_t, aicc_t, bic_t = info_criteria(m['logL_split'], 6, n)
    lrt = 2.0 * (m['logL_split'] - m['single']['logL'])

    null, znull = boot_null(d, cfg['cut'], m['single'])
    p_boot = (np.sum(null >= lrt) + 1) / (len(null) + 1) if len(null) else np.nan
    p_chi2 = stats.chi2.sf(lrt, 3) if lrt > 0 else 1.0

    db = m['faint']['slope'] - m['bright']['slope']
    db_err = np.hypot(m['faint']['slope_err'], m['bright']['slope_err'])
    z_obs = abs(db) / db_err
    # The naive "N sigma" reading of z assumes large samples AND groups defined
    # in advance; neither holds here, so calibrate z on the same null.
    p_z = (np.sum(znull >= z_obs) + 1) / (len(znull) + 1) if len(znull) else np.nan

    cuts = np.arange(-17.5, -13.4, 0.1)
    scan = threshold_scan(d, m['single'], cuts)

    results[key] = dict(cfg=cfg, d=d, m=m, n=n,
                        aicc_s=aicc_s, aicc_t=aicc_t, bic_s=bic_s, bic_t=bic_t,
                        lrt=lrt, null=null, p_boot=p_boot, p_chi2=p_chi2,
                        dslope=db, dslope_err=db_err, z_obs=z_obs,
                        znull=znull, p_z=p_z, cuts=cuts, scan=scan)

    print(f'{key}-band   (N = {n};  faint cut M > {cfg["cut"]:+.1f})')
    print('-' * 78)
    for tag in ('single', 'faint', 'bright'):
        f = m[tag]
        print(f'  {f["label"]:<12s} N={f["n"]:2d}  slope = {f["slope"]:+7.3f} +/- {f["slope_err"]:.3f}'
              f'   icpt = {f["intercept"]:+8.3f}   sig_int = {f["sig_int"]:.3f}'
              f'   [OLS slope {f["ols_slope"]:+7.3f}, sigma {f["ols_sigma"]:.3f}]')
    print(f'  slope difference (faint - bright) = {db:+.3f} +/- {db_err:.3f}   z = {z_obs:.2f}')
    print(f'    -> the naive "{z_obs:.1f} sigma" is NOT {z_obs:.1f} sigma: under the single-trend')
    print(f'       null the same procedure gives z with median {np.median(znull):.2f}, '
          f'95th pct {np.percentile(znull, 95):.2f}')
    print(f'       calibrated p for the slope difference alone = {p_z:.3f}')
    print(f'  logL : single = {m["single"]["logL"]:+.3f} (k=3)   split = {m["logL_split"]:+.3f} (k=6)')
    print(f'  AICc : single = {aicc_s:+.2f}   split = {aicc_t:+.2f}   dAICc = {aicc_t - aicc_s:+.2f}'
          f'   -> {"SPLIT" if aicc_t < aicc_s else "SINGLE"} preferred')
    print(f'  BIC  : single = {bic_s:+.2f}   split = {bic_t:+.2f}   dBIC  = {bic_t - bic_s:+.2f}'
          f'   -> {"SPLIT" if bic_t < bic_s else "SINGLE"} preferred')
    print(f'  LRT  : 2*dlogL = {lrt:.2f}   p(chi2,3dof) = {p_chi2:.2e}'
          f'   p(bootstrap, {len(null)} sims) = {p_boot:.4f}')
    print('=' * 78)


# --------------------------------------------------------------------------
# 4. Figure 1 -- the two competing descriptions, side by side
# --------------------------------------------------------------------------

fig, axes = plt.subplots(1, 2, figsize=(15, 7))

for ax, key in zip(axes, BANDS):
    R = results[key]
    d, m, cfg = R['d'], R['m'], R['cfg']
    xlim = cfg['xlim']
    xg = np.linspace(*xlim, 200)

    faint = d['faint'].to_numpy()
    ax.errorbar(d['dm15'][~faint], d['M'][~faint], xerr=d['dm15_err'][~faint],
                yerr=d['M_err'][~faint], fmt='o', ms=6, color='0.25',
                elinewidth=0.8, capsize=0, label='Bright Iax', zorder=3)
    ax.errorbar(d['dm15'][faint], d['M'][faint], xerr=d['dm15_err'][faint],
                yerr=d['M_err'][faint], fmt='s', ms=6, color=cfg['color'],
                elinewidth=0.8, capsize=0, label='Faint Iax', zorder=3)
    vjm = d[d['Name'] == '2024vjm'].iloc[0]
    ax.plot(vjm['dm15'], vjm['M'], '*', ms=20, color='gold', mec='k', mew=0.8,
            zorder=6, label='SN 2024vjm')

    # Model S: one line for everything
    ys = m['single']['slope'] * xg + m['single']['intercept']
    ax.plot(xg, ys, '-', lw=2.2, color='k', zorder=4,
            label=f"Model S: one trend (slope {m['single']['slope']:+.2f})")
    ax.fill_between(xg, ys - m['single']['sig_int'], ys + m['single']['sig_int'],
                    color='k', alpha=0.10, lw=0, zorder=1)

    # Model T: separate lines
    for tag, ls, col in (('faint', '--', cfg['color']), ('bright', '--', '0.25')):
        f = m[tag]
        xs = np.linspace(f['x'].min() - 0.05, f['x'].max() + 0.05, 100)
        yy = f['slope'] * xs + f['intercept']
        ax.plot(xs, yy, ls, lw=2.0, color=col, zorder=5,
                label=f"Model T: {tag} (slope {f['slope']:+.2f})")
        ax.fill_between(xs, yy - f['sig_int'], yy + f['sig_int'],
                        color=col, alpha=0.12, lw=0, zorder=1)

    ax.axhline(cfg['cut'], color='0.6', ls=':', lw=1.0)
    ax.text(xlim[0] + 0.02, cfg['cut'], f" faint/bright cut  M = {cfg['cut']:+.1f}",
            fontsize=8, color='0.45', va='bottom')

    verdict_bic = 'two trends' if R['bic_t'] < R['bic_s'] else 'one trend'
    verdict_aic = 'two trends' if R['aicc_t'] < R['aicc_s'] else 'one trend'
    box = (f"N = {R['n']}\n"
           f"$\\Delta$BIC (T$-$S) = {R['bic_t'] - R['bic_s']:+.1f}  $\\to$ {verdict_bic}\n"
           f"$\\Delta$AICc (T$-$S) = {R['aicc_t'] - R['aicc_s']:+.1f}  $\\to$ {verdict_aic}\n"
           f"slope$_{{faint}}-$slope$_{{bright}}$ = {R['dslope']:+.2f} $\\pm$ {R['dslope_err']:.2f}"
           f"  ($p$ = {R['p_z']:.3f})\n"
           f"full-model bootstrap $p$ = {R['p_boot']:.3f}")
    ax.text(0.98, 0.04, box, transform=ax.transAxes, fontsize=9, ha='right', va='bottom',
            bbox=dict(boxstyle='round,pad=0.5', fc='white', ec='0.7', alpha=0.9))

    ax.set_xlim(*xlim)
    ax.invert_yaxis()
    ax.set_xlabel(cfg['xlabel'], fontsize=13)
    ax.set_ylabel(cfg['ylabel'], fontsize=13)
    ax.set_title(f'{key}-band', fontsize=14, weight='bold')
    ax.legend(fontsize=8, loc='upper left', framealpha=0.9)

fig.suptitle('One trend for all SNe Iax (Model S) vs. separate bright/faint trends (Model T)',
             fontsize=14, weight='bold')
fig.tight_layout(rect=(0, 0, 1, 0.96))

# SN names last, once the axes are final, so the declutter uses true pixel positions
for ax, key in zip(axes, BANDS):
    d = results[key]['d']
    place_labels(ax, d['dm15'], d['M'], d['Name'])

p1 = os.path.join(FIGDIR, 'Peak_M15_single_vs_split_fits.pdf')
fig.savefig(p1, bbox_inches='tight', dpi=200)
fig.savefig(p1.replace('.pdf', '.png'), bbox_inches='tight', dpi=150)


# --------------------------------------------------------------------------
# 5. Figure 2 -- diagnostics: residuals, bootstrap null, threshold scan
# --------------------------------------------------------------------------

fig2, axs = plt.subplots(2, 3, figsize=(16, 9))

for irow, key in enumerate(BANDS):
    R = results[key]
    d, m, cfg = R['d'], R['m'], R['cfg']
    faint = d['faint'].to_numpy()

    # (a) residuals in magnitudes, with each model's intrinsic-scatter band.
    # Raw magnitudes rather than sigma units: Model S can only accommodate the
    # data by inflating sigma_int, and that is exactly what should be visible.
    ax = axs[irow, 0]
    x = d['dm15'].to_numpy(float)
    res_s = d['M'].to_numpy(float) - (m['single']['slope'] * x + m['single']['intercept'])
    res_t = np.empty(len(d))
    for i, (isf, xi, yi) in enumerate(zip(faint, x, d['M'])):
        f = m['faint'] if isf else m['bright']
        res_t[i] = yi - (f['slope'] * xi + f['intercept'])
    ax.axhline(0, color='0.7', lw=1)
    ss = m['single']['sig_int']
    ax.axhspan(-ss, ss, color='0.85', alpha=0.6, lw=0,
               label=rf"Model S $\sigma_{{\rm int}}$ = {ss:.2f} mag")
    for tag in ('faint', 'bright'):
        sg = m[tag]['sig_int']
        ax.axhspan(-sg, sg, color=cfg['color'], alpha=0.10, lw=0)
    sf, sb = m['faint']['sig_int'], m['bright']['sig_int']
    ax.axhspan(-max(sf, sb), max(sf, sb), fc='none', ec=cfg['color'], lw=1.0, ls='--',
               label=rf"Model T $\sigma_{{\rm int}}$ = {sf:.2f} / {sb:.2f} mag")
    for xi, a_, b_ in zip(x, res_s, res_t):
        ax.plot([xi, xi], [a_, b_], '-', color='0.75', lw=0.8, zorder=0)
    ax.plot(x, res_s, 'o', ms=7, color='k', label='Model S (one trend)')
    ax.plot(x, res_t, 's', ms=7, mfc='none', mew=1.5, color=cfg['color'],
            label='Model T (two trends)')
    ax.set_xlabel(cfg['xlabel'])
    ax.set_ylabel('residual (mag)')
    ax.set_title(f'{key}-band: residuals about each model', fontsize=12, weight='bold')
    ax.legend(fontsize=7.5, ncol=2, loc='upper center')
    ax.margins(y=0.25)

    # (b) bootstrap null distribution of the LRT statistic
    ax = axs[irow, 1]
    if len(R['null']):
        ax.hist(R['null'], bins=40, color='0.75', edgecolor='0.5',
                label=f'null (single trend), {len(R["null"])} sims')
    ax.axvline(R['lrt'], color=cfg['color'], lw=2.5,
               label=f'observed = {R["lrt"]:.1f}')
    if len(R['null']):
        ax.axvline(np.percentile(R['null'], 95), color='k', ls='--', lw=1.2,
                   label='95th pct of null')
    ax.set_xlabel(r'$2\,\Delta \log \mathcal{L}$  (split $-$ single)')
    ax.set_ylabel('simulations')
    ax.set_title(f'{key}-band: is the split real?  p = {R["p_boot"]:.3f}',
                 fontsize=12, weight='bold')
    ax.legend(fontsize=8)

    # Inset: why the bare "z sigma" on the slope difference must not be read as a
    # Gaussian significance.  Splitting on luminosity inflates z even under the null.
    axi = ax.inset_axes([0.58, 0.42, 0.40, 0.34])
    if len(R['znull']):
        axi.hist(R['znull'], bins=30, color='0.8', edgecolor='0.55', lw=0.4)
        axi.axvline(np.percentile(R['znull'], 95), color='k', ls='--', lw=1.0)
    axi.axvline(R['z_obs'], color=cfg['color'], lw=2.0)
    axi.set_xlabel(r'$z$ of $\Delta$slope under null', fontsize=7)
    axi.set_yticks([])
    axi.tick_params(labelsize=6)
    axi.set_title(f"z = {R['z_obs']:.2f}  ->  p = {R['p_z']:.3f}", fontsize=7.5)

    # (c) does the answer depend on where we cut?
    ax = axs[irow, 2]
    ax.axhline(0, color='k', lw=1)
    ax.axhspan(-100, -10, color='seagreen', alpha=0.12, lw=0)
    ax.axhspan(10, 100, color='indianred', alpha=0.12, lw=0)
    # Splitting on the y-variable buys some dBIC even under the null; show how
    # much, so the observed curve is read against the right baseline.
    if len(R['null']):
        null_dbic = -R['null'] + 3.0 * np.log(R['n'])
        lo, hi = np.percentile(null_dbic, [5, 95])
        ax.axhspan(lo, hi, color='0.5', alpha=0.15, lw=0,
                   label='90% range expected under\none-trend null (same split)')
        ax.axhline(np.median(null_dbic), color='0.45', ls='-.', lw=1.0)
    ax.plot(R['cuts'], R['scan'], '-o', ms=3, color=cfg['color'], label='observed')
    ax.axvline(cfg['cut'], color='0.4', ls=':', lw=1.5,
               label=f"cut used in Width.py ({cfg['cut']:+.1f})")
    good = np.isfinite(R['scan'])
    if good.any():
        vals = list(np.abs(R['scan'][good]))
        if len(R['null']):
            vals += list(np.abs(np.percentile(-R['null'] + 3.0 * np.log(R['n']), [5, 95])))
        lim = max(12, np.nanmax(vals) * 1.15)
        ax.set_ylim(-lim, lim)
        ax.text(0.02, 0.96, 'two trends favoured', transform=ax.transAxes,
                fontsize=8, color='seagreen', va='top')
        ax.text(0.02, 0.04, 'one trend favoured', transform=ax.transAxes,
                fontsize=8, color='indianred', va='bottom')
    ax.invert_yaxis()
    ax.set_xlabel(r'faint/bright cut in peak $M$ (mag)')
    ax.set_ylabel(r'$\Delta$BIC  (two $-$ one)')
    ax.set_title(f'{key}-band: sensitivity to the cut', fontsize=12, weight='bold')
    ax.legend(fontsize=7.5, loc='lower right')

fig2.suptitle('Diagnostics for the one-trend vs. two-trend comparison',
              fontsize=14, weight='bold')
fig2.tight_layout(rect=(0, 0, 1, 0.96))
p2 = os.path.join(FIGDIR, 'Peak_M15_single_vs_split_diagnostics.pdf')
fig2.savefig(p2, bbox_inches='tight', dpi=200)
fig2.savefig(p2.replace('.pdf', '.png'), bbox_inches='tight', dpi=150)

print(f'\nFigures written to:\n  {p1}\n  {p2}')

# --------------------------------------------------------------------------
# 6. Figure 3 -- publication version, in the style of Width.py / Fig. 4
#    Same layout as the paper figure.  Three maximum-likelihood fits are
#    drawn, each with its +/-1 sigma_int band; the two group fits span only
#    their own data range.  SN names are placed by a collision-aware routine.
# --------------------------------------------------------------------------

from astropy.table import Table
from matplotlib.transforms import Bbox

ns.apply()   # [rev3] Nature artwork style (Arial, 5-7 pt, TrueType)

m15MB_ia = Table.read(os.path.join(HOME, 'm15_MB_snname_published.txt'),
                      format='ascii')

BRIGHT_COLOR = 'darkorange'

# Where the trend labels go (data coordinates of the text centre).  Each is
# rotated to follow its line.  Chosen by hand so they sit inside the bands.
# Hand-placed SN labels (offset in points) for spots the automatic placer
# cannot resolve; a leader line is drawn when the offset is large.
SN_LABEL_OVERRIDES = {
    'r': {'2025zi': (-2, -7), '2019muj': (3, -6.5), '2019gsc': (17, 3), '2003gq': (0, -7),
          '2005hk': (0, 6), '2005cc': (-4, -1)},   # [rev3] offsets in points at print size; author-requested placements
    'B': {'2024pxl': (5, -2), '2010ae': (5, -3)},   # [rev3] author: keep names close to their points
}

# Optional x-range overrides for the group lines/bands (data units); None keeps
# the automatic value (data range padded by 7% of the axis span).
GROUP_LINE_XRANGE = {
    'r': {'faint': (0.40, None)},
    'B': {},
}

TREND_LABEL_POS = {
    'r': {'bright': (0.88, -16.95), 'faint': (0.69, -14.12), 'single': (0.88, -15.65)},   # [rev3] bright label pushed right (was over 2003gq)
    'B': {'bright': (2.12, -17.05), 'faint': (2.15, -13.55), 'single': (2.00, -15.10)},
}

pub_cfg = {
    'r': dict(xlim=(0.3, 1.3), ylim=(-19.4, -12.1), color='crimson',
              xlabel=r'$\Delta M_{15}(r)$ (mag)',
              ylabel=r'Peak absolute magnitude, $M_r$ (mag)'),
    'B': dict(xlim=(0.75, 3.0), ylim=(-20.3, -12.0), color='royalblue',
              xlabel=r'$\Delta M_{15}(B)$ (mag)',
              ylabel=r'Peak absolute magnitude, $M_B$ (mag)'),
}

# candidate label offsets (points), nearest first
_CANDS_FULL = [(7, 5), (7, -7), (-7, 5), (-7, -7), (0, 10), (0, -12), (12, 0), (-12, 0),
          (14, 11), (14, -13), (-14, 11), (-14, -13), (0, 18), (0, -20),
          (22, 6), (-22, 6), (22, -8), (-22, -8),
          (24, 16), (24, -18), (-24, 16), (-24, -18), (0, 27), (0, -29),
          (34, 0), (-34, 0), (30, 24), (30, -26), (-30, 24), (-30, -26),
          (0, 40), (0, -42), (46, 0), (-46, 0), (40, 34), (40, -36), (-40, 34), (-40, -36),
          (0, 56), (0, -58), (60, 0), (-60, 0)]
# [rev3] figure is now drawn at print size (180 mm): halve all offsets in points
_CANDS = [(0.5 * dx, 0.5 * dy) for dx, dy in _CANDS_FULL]


def _overlaps(b, others, pad=1.5):
    bb = Bbox.from_extents(b.x0 - pad, b.y0 - pad, b.x1 + pad, b.y1 + pad)
    return any(bb.overlaps(o) for o in others)


def place_sn_labels(ax, df, hard, soft=(), fontsize=11, color='black', weight='bold',
                    overrides=None):
    """Annotate each row of df at the first offset that hits nothing.

    hard: display-space boxes a label must never overlap (points, other
          labels); extended in place with the boxes of the labels placed here.
    soft: boxes to avoid if possible (the fitted lines); ignored on a second
          pass when nothing else fits.
    """
    fig = ax.figure
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    axbb = ax.get_window_extent(rend)
    pts = ax.transData.transform(np.c_[df['dm15'], df['M']])
    order = np.argsort([-np.sum(np.hypot(*(pts - q).T) < 60) for q in pts])
    overrides = overrides or {}
    # hand-placed labels go first so the automatic ones work around them
    order = sorted(order, key=lambda i: df.iloc[i]['Name'] not in overrides)
    for i in order:
        row = df.iloc[i]
        col = row.get('label_color', color)
        placed = None
        if row['Name'] in overrides:
            dx, dy = overrides[row['Name']]
            ha = 'left' if dx > 0 else ('right' if dx < 0 else 'center')
            arrow = (dict(arrowstyle='-', color='0.55', lw=0.4, shrinkA=0.5, shrinkB=1.5)
                     if np.hypot(dx, dy) > 10 else None)
            t = ax.annotate(row['Name'], (row['dm15'], row['M']), textcoords='offset points',
                            xytext=(dx, dy), fontsize=fontsize, weight=weight, color=col,
                            ha=ha, va='center', zorder=6, arrowprops=arrow)
            bb = t.get_window_extent(rend)
            if _overlaps(bb, hard):
                print(f'  [warn] hand-placed label {row["Name"]} overlaps something')
            hard.append(bb)
            continue
        for obstacles in (list(hard) + list(soft), hard):
            for dx, dy in _CANDS:
                ha = 'left' if dx > 0 else ('right' if dx < 0 else 'center')
                arrow = (dict(arrowstyle='-', color='0.55', lw=0.4, shrinkA=0.5, shrinkB=1.5)
                         if np.hypot(dx, dy) > 10 else None)
                t = ax.annotate(row['Name'], (row['dm15'], row['M']), textcoords='offset points',
                                xytext=(dx, dy), fontsize=fontsize, weight=weight, color=col,
                                ha=ha, va='center', zorder=6, arrowprops=arrow)
                bb = t.get_window_extent(rend)
                inside = (bb.x0 >= axbb.x0 + 2 and bb.x1 <= axbb.x1 - 2
                          and bb.y0 >= axbb.y0 + 2 and bb.y1 <= axbb.y1 - 2)
                if inside and not _overlaps(bb, obstacles):
                    placed = bb
                    break
                t.remove()
            if placed is not None:
                break
        if placed is None:
            print(f'  [warn] no clear position for label {row["Name"]}; using default offset')
            t = ax.annotate(row['Name'], (row['dm15'], row['M']), textcoords='offset points',
                            xytext=_CANDS[0], fontsize=fontsize, weight=weight, color=col,
                            ha='left', va='center', zorder=6)
            placed = t.get_window_extent(rend)
        hard.append(placed)

def place_trend_label(ax, text, slope, intercept, xs, hard, soft=(), color='k', fontsize=14,
                      fracs=(0.5, 0.35, 0.65, 0.2, 0.8, 0.1, 0.9),
                      offsets=(-8, 8, -15, 15, -22, 22), weight='normal', alpha=0.85):
    """Write `text` along the line y = slope*x + intercept, rotated to match it.

    Tries positions at fractions `fracs` of the x-range `xs`, displaced
    perpendicular to the line by `offsets` points (nearest first), and keeps
    the first whose box lies inside the axes and clears `hard` (and `soft`
    when possible).  The accepted box is appended to `hard`.
    """
    fig = ax.figure
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    axbb = ax.get_window_extent(rend)
    p0 = ax.transData.transform((0, 0))
    p1 = ax.transData.transform((1, slope))
    ang = np.arctan2(*(p1 - p0)[::-1])
    nx, ny = -np.sin(ang), np.cos(ang)            # unit normal in display space
    for obstacles in (list(hard) + list(soft), hard):
        for off in offsets:
            for fr in fracs:
                x = xs[0] + fr * (xs[1] - xs[0])
                t = ax.annotate(text, (x, slope * x + intercept), textcoords='offset points',
                                xytext=(off * nx, off * ny), rotation=np.degrees(ang),
                                rotation_mode='anchor', ha='center', va='center',
                                fontsize=fontsize, color=color, weight=weight, alpha=alpha,
                                zorder=6)
                bb = t.get_window_extent(rend)
                inside = (bb.x0 >= axbb.x0 + 2 and bb.x1 <= axbb.x1 - 2
                          and bb.y0 >= axbb.y0 + 2 and bb.y1 <= axbb.y1 - 2)
                if inside and not _overlaps(bb, obstacles, pad=3):
                    hard.append(bb)
                    return t
                t.remove()
    return None


def _rotated_text_boxes(t, rend, n=12):
    """Approximate a rotated text by n small boxes along its baseline direction.

    The axis-aligned bounding box of a rotated label is far larger than the
    glyphs; using it as an obstacle blocks space the text does not occupy.
    """
    ang = np.radians(t.get_rotation())
    rot = t.get_rotation()
    t.set_rotation(0)
    bb0 = t.get_window_extent(rend)
    t.set_rotation(rot)
    w, h = bb0.width, bb0.height
    bb = t.get_window_extent(rend)
    cx, cy = 0.5 * (bb.x0 + bb.x1), 0.5 * (bb.y0 + bb.y1)
    ux, uy = np.cos(ang), np.sin(ang)
    boxes = []
    for f in np.linspace(-0.5, 0.5, n):
        x, y = cx + f * w * ux, cy + f * w * uy
        r = 0.5 * h
        boxes.append(Bbox.from_extents(x - r, y - r, x + r, y + r))
    return boxes


def fixed_trend_label(ax, text, slope, intercept, xy, hard, color, fontsize=14,
                      weight='normal', alpha=0.85, gaps=(8, 10, 12, 14, 17, 20),
                      shifts=(0.0, 0.03, -0.03, 0.06, -0.06, 0.1, -0.1), half_width=None,
                      pad=2):
    """Rotated text following the line y = slope*x + intercept.

    xy (data coords) picks the location along the line and which side of it
    the label sits on.  The label is then moved perpendicular to the line by
    the first gap (points) -- and, if needed, shifted along the line by one of
    `shifts` (fraction of the x-axis span) -- such that its glyphs clear every
    box in `hard` by `pad` pixels and, when `half_width` (data units, e.g.
    3*sigma_int) is given, lie entirely inside the shaded band.
    """
    fig = ax.figure
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    px_per_pt = fig.dpi / 72.0
    axbb = ax.get_window_extent(rend)
    xspan = np.diff(ax.get_xlim())[0]
    A = ax.transData.transform(xy)
    P0 = ax.transData.transform((xy[0], slope * xy[0] + intercept))
    P1 = ax.transData.transform((xy[0] + 1, slope * (xy[0] + 1) + intercept))
    u = (P1 - P0) / np.hypot(*(P1 - P0))
    n = np.array([-u[1], u[0]])
    Pfoot = P0 + u * np.dot(A - P0, u)
    side = np.sign(np.dot(A - Pfoot, n)) or 1.0
    ang = np.degrees(np.arctan2(u[1], u[0]))
    inv = ax.transData.inverted()

    best = None
    for gap in gaps:
        for sh in shifts:
            # move the foot along the line by sh * xspan, then off the line by gap
            xf = xy[0] + sh * xspan
            Pf = ax.transData.transform((xf, slope * xf + intercept))
            Q = Pf + side * gap * px_per_pt * n
            qx, qy = inv.transform(Q)
            t = ax.text(qx, qy, text, rotation=ang, rotation_mode='anchor', ha='center',
                        va='center', fontsize=fontsize, color=color, weight=weight,
                        alpha=alpha, zorder=6)
            boxes = _rotated_text_boxes(t, rend, n=16)
            ok = all(b.x0 >= axbb.x0 and b.x1 <= axbb.x1 and b.y0 >= axbb.y0 and b.y1 <= axbb.y1
                     for b in boxes)
            ok = ok and not any(_overlaps(b, hard, pad=pad) for b in boxes)
            if ok and half_width is not None:
                for b in boxes:
                    for cx, cy in ((b.x0, b.y0), (b.x1, b.y0), (b.x0, b.y1), (b.x1, b.y1)):
                        dx_, dy_ = inv.transform((cx, cy))
                        if abs(dy_ - (slope * dx_ + intercept)) > half_width:
                            ok = False
            if ok:
                hard.extend(boxes)
                return t
            if best is None:
                best = (gap, sh)
            t.remove()

    # nothing satisfied every constraint: fall back to the first candidate and say so
    gap, sh = best
    xf = xy[0] + sh * xspan
    Pf = ax.transData.transform((xf, slope * xf + intercept))
    qx, qy = inv.transform(Pf + side * gap * px_per_pt * n)
    t = ax.text(qx, qy, text, rotation=ang, rotation_mode='anchor', ha='center', va='center',
                fontsize=fontsize, color=color, weight=weight, alpha=alpha, zorder=6)
    print(f'  [warn] label "{text.split(chr(10))[0]}" could not satisfy all constraints')
    hard.extend(_rotated_text_boxes(t, rend, n=16))
    return t

def make_two_model_figure(n_sigma, fname):
    """Publication figure; group fits carry a shaded +/- n_sigma * sigma_int band,
    the single trend shows the same range as dotted lines (its band would fill the panel).

    [rev3] Drawn at final print size (180 mm wide) in the Nature style, with an
    in-figure key for every symbol, line and shading (editor request)."""
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    from matplotlib.legend_handler import HandlerTuple

    fig3, (axr, axb) = plt.subplots(1, 2, figsize=(ns.DOUBLE_COL, 0.47 * ns.DOUBLE_COL))

    for ax, key, panel in ((axr, 'r', 'a'), (axb, 'B', 'b')):
        R, cfg, m = results[key], pub_cfg[key], results[key]['m']
        d = R['d']
        xlim, ylim = cfg['xlim'], cfg['ylim']
        span = xlim[1] - xlim[0]
        xg = np.linspace(*xlim, 200)

        if key == 'B':
            ax.scatter(m15MB_ia['m15'], m15MB_ia['MB'], s=3, color='gray', alpha=0.5,
                       linewidths=0, zorder=1)

        comp = d[d['Name'] != '2024vjm']
        vjm = d[d['Name'] == '2024vjm'].iloc[0]
        ax.errorbar(comp['dm15'], comp['M'], xerr=comp['dm15_err'], yerr=comp['M_err'],
                    fmt='o', color='black', ms=2, elinewidth=0.4, alpha=0.8, capsize=0, zorder=4)
        ax.errorbar(vjm['dm15'], vjm['M'], xerr=vjm['dm15_err'], yerr=vjm['M_err'],
                    color=cfg['color'], fmt='*', ms=6, elinewidth=0.6, zorder=5)

        # Model S: single trend through all SNe Iax, full axis span
        fs = m['single']
        ys = fs['slope'] * xg + fs['intercept']
        ax.plot(xg, ys, '-', lw=0.7, color='0.45', alpha=0.6, zorder=2)
        for sgn in (-1, 1):
            ax.plot(xg, ys + sgn * n_sigma * fs['sig_int'], ':', lw=0.6, color='0.45',
                    alpha=0.7, zorder=2)

        # Model T: bright and faint trends, each over its own data range
        group_lines = {}
        for tag, ls, col in (('bright', '-.', BRIGHT_COLOR), ('faint', '--', cfg['color'])):
            f = m[tag]
            sub = d[(d['M'] <= R['cfg']['cut']) if tag == 'bright' else (d['M'] > R['cfg']['cut'])]
            x_lo, x_hi = sub['dm15'].min() - 0.07 * span, sub['dm15'].max() + 0.07 * span
            ov_lo, ov_hi = GROUP_LINE_XRANGE[key].get(tag, (None, None))
            x_lo = ov_lo if ov_lo is not None else x_lo
            x_hi = ov_hi if ov_hi is not None else x_hi
            xx = np.linspace(x_lo, x_hi, 80)
            yy = f['slope'] * xx + f['intercept']
            sig = f['sig_int']
            ax.fill_between(xx, yy - n_sigma * sig, yy + n_sigma * sig, color=col, alpha=0.15,
                            edgecolor='none', zorder=1)
            ax.plot(xx, yy, ls, lw=0.9, color=col, alpha=0.95, zorder=3)
            group_lines[tag] = (f['slope'], f['intercept'], (xx[0], xx[-1]), col)

        ax.set_xlabel(cfg['xlabel'])
        ax.set_ylabel(cfg['ylabel'])
        ax.set_xlim(*xlim)
        ax.set_ylim(ylim[1], ylim[0])            # inverted magnitude axis

        ax.text(0.015, 0.99, panel, transform=ax.transAxes, fontsize=ns.FS_PANEL,
                fontweight='bold', va='top', ha='left', zorder=10,
                bbox=dict(boxstyle='square,pad=0.1', fc='white', ec='none'))

        # --- obstacles: data points (and the CSP cloud in B) ---
        fig3.canvas.draw()
        rend = fig3.canvas.get_renderer()
        obstacles = []
        for x, y in ax.transData.transform(np.c_[d['dm15'], d['M']]):
            obstacles.append(Bbox.from_extents(x - 3, y - 3, x + 3, y + 3))
        if key == 'B':
            for x, y in ax.transData.transform(np.c_[m15MB_ia['m15'], m15MB_ia['MB']]):
                obstacles.append(Bbox.from_extents(x - 2, y - 2, x + 2, y + 2))
            t = ax.text(0.85, -18.45, 'Normal SNe Ia', fontsize=ns.FS, color='gray',
                        weight='bold', rotation=-10, zorder=6)
            obstacles.append(t.get_window_extent(rend))

        # the fitted lines are soft obstacles: avoided when possible
        soft = []
        for sl, ic, xr, _ in list(group_lines.values()) + [(fs['slope'], fs['intercept'], xlim, None)]:
            xs_ = np.linspace(xr[0], xr[1], 80)
            for x, y in ax.transData.transform(np.c_[xs_, sl * xs_ + ic]):
                soft.append(Bbox.from_extents(x - 1, y - 1, x + 1, y + 1))

        # --- trend labels at their fixed anchors, then SN names around them ---
        pos = TREND_LABEL_POS[key]
        for tag, text in (('faint', 'Faint SNe Iax'), ('bright', 'Bright SNe Iax')):
            sl, ic, xr, col = group_lines[tag]
            if key == 'r' and tag == 'bright':
                # [rev3] author-specified position: centred exactly at the anchor, rotated with the line
                _p0 = ax.transData.transform((0, ic)); _p1 = ax.transData.transform((1, sl + ic))
                ang = np.degrees(np.arctan2(_p1[1] - _p0[1], _p1[0] - _p0[0]))
                t = ax.text(*pos[tag], text, rotation=ang, rotation_mode='anchor', ha='center',
                            va='center', fontsize=ns.FS, color=col, weight='bold', alpha=0.85, zorder=6)
                obstacles.extend(_rotated_text_boxes(t, fig3.canvas.get_renderer(), n=16))
                continue
            fixed_trend_label(ax, text, sl, ic, pos[tag], obstacles, color=col, fontsize=ns.FS,
                              weight='bold', half_width=n_sigma * m[tag]['sig_int'])
        fixed_trend_label(ax, 'Single linear fit', fs['slope'], fs['intercept'],
                          pos['single'], obstacles, color='0.4', fontsize=ns.FS_SMALL,
                          gaps=(6, 8, 10, 12, 14, 17))

        lab = d.copy()
        lab['label_color'] = np.where(lab['Name'] == '2024vjm', cfg['color'], 'black')
        place_sn_labels(ax, lab, obstacles, soft, fontsize=ns.FS_TINY,
                        overrides=SN_LABEL_OVERRIDES[key])

    # --- [rev3] shared key for every symbol, line style and shading ---
    rc, bc = pub_cfg['r']['color'], pub_cfg['B']['color']
    star = lambda c: Line2D([], [], ls='none', marker='*', ms=6, color=c)
    handles = [
        (star(rc), star(bc)),
        Line2D([], [], ls='none', marker='o', ms=2, color='black'),
        Line2D([], [], ls='none', marker='o', ms=2, color='gray', alpha=0.6),
        (Patch(color=rc, alpha=0.15, lw=0), Line2D([], [], ls='--', lw=0.9, color=rc),
         Patch(color=bc, alpha=0.15, lw=0), Line2D([], [], ls='--', lw=0.9, color=bc)),
        (Patch(color=BRIGHT_COLOR, alpha=0.15, lw=0),
         Line2D([], [], ls='-.', lw=0.9, color=BRIGHT_COLOR)),
        Line2D([], [], ls='-', lw=0.7, color='0.45', alpha=0.6),
        Line2D([], [], ls=':', lw=0.6, color='0.45', alpha=0.7),
    ]
    labels = [
        'SN 2024vjm',
        'SNe Iax',
        'Normal SNe Ia',
        rf'Faint SNe Iax fit $\pm{n_sigma}\sigma$',
        rf'Bright SNe Iax fit $\pm{n_sigma}\sigma$',
        'Single fit',
        rf'Single fit $\pm{n_sigma}\sigma$',
    ]
    fig3.tight_layout(rect=(0, 0.05, 1, 1))
    fig3.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, 0.0), ncol=7,
                fontsize=ns.FS_SMALL, columnspacing=1.2, handletextpad=0.5,
                handler_map={tuple: HandlerTuple(ndivide=None, pad=0.2)})
    return ns.save(fig3, fname.replace('.pdf', ''))


fig3 = make_two_model_figure(3, 'Fig4_Peak_M15.pdf')

if os.environ.get('WIDTH_NOSHOW', '0') != '1':
    plt.show()
