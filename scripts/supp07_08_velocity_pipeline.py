# =============================================================================
# Supplementary Figs 7 and 8: line-velocity fits of the optical and NIR spectra.
# Notes:
#   * all outputs (CSVs, per-line mosaics, raw pipeline plots) -> outputs/figures/supplementary/_work/vel_pipeline
#   * final figures saved with nature_style (ns.save) as SuppFig7_velocities / SuppFig8_velocity_fits
#   * phases re-referenced to B max MJD 60572.29 AFTER the selection matrix is applied
#     (velocity_selection.csv keys were built with 60572.18; it is only read)
#   * SOAR/TripleSpec MJDs set to the observation times in Supplementary Table 1
#   * Nature styling / labels / in-figure key; 2-decimal phase labels in the mosaic
# =============================================================================
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.io import ascii
from astropy.time import Time
from scipy.optimize import curve_fit

c_kms = 299792.458

import nature_style as ns
ns.apply()
B_PEAK_NEW = 60572.29   # B-band maximum used in the paper (Supplementary Table 1)

# =============================================================================
#                                  HELPERS
# =============================================================================

def read_spec_as_restframe(path, instrument, fname, z):
    """Read 2-col ascii spectrum and return (lambda_rest_A, flux)."""
    try:
        spec = ascii.read(path)
        # Handle different column names or lack thereof by using indices
        if len(spec.colnames) >= 2:
            col0, col1 = spec.colnames[0], spec.colnames[1]
        else:
            # Fallback if astropy read doesn't give clear columns (unlikely with ascii.read)
            return np.array([]), np.array([])
            
        x = pd.to_numeric(np.array(spec[col0], dtype=str), errors='coerce').astype(float)
        y = pd.to_numeric(np.array(spec[col1], dtype=str), errors='coerce').astype(float)
        m = np.isfinite(x) & np.isfinite(y)
        x, y = x[m], y[m]
        
        if instrument == 'VLT/ X-Shooter':
            if 'UVB' in fname:
                x = x * 10.0; m = x > 3100; x, y = x[m], y[m]
            elif 'VIS' in fname:
                x = x * 10.0; m = x > 5578; x, y = x[m], y[m]
            elif 'NIR' in fname:
                m = x > 10000; x, y = x[m], y[m]
                
        return x/(1.0+z), y
    except Exception as e:
        print(f"[WARN] Failed to read {path}: {e}")
        return np.array([]), np.array([])

def apply_extinction_correction(wave_rest, flux, ebv, r_v=3.1):
    """
    Apply MW extinction correction using O'Donnell (1994) or similar via a simple 
    conversion if extinction package not available.
    """
    try:
        import extinction
        # wave_rest is in rest frame, but extinction is usually applied in OBS frame?
        # Standard: correct flux in rest frame by using de-reddened values.
        # Wave must be in Angstroms for extinction.ccm89
        # Assuming wave_rest is Angstroms.
        # We need to apply it to OBSERVED wavelengths, but pipeline works in rest.
        # wave_obs = wave_rest * (1+z)
        # However, many people just apply it to rest frame if z is small.
        # Let's do it properly if we have z. But within this function we just have wave.
        # We'll assume wave_rest is provided and we treat it as the wavelength for extinction.
        
        # extinction.ccm89(wave, a_v, r_v) expects wave in float64
        # a_v = ebv * r_v
        a_v = ebv * r_v
        # Note: extinction.ccm89 returns extinction in magnitudes.
        # A_lambda = extinction.ccm89(wave, a_v, r_v)
        # corrected_flux = flux * 10**(0.4 * A_lambda)
        
        # Check if wave is sorted (extinction might require it)
        ext = extinction.ccm89(wave_rest.astype(np.float64), a_v, r_v)
        return flux * 10**(0.4 * ext)
    except ImportError:
        # Fallback approximation if extinction package not installed
        # A_v ~ 3.1 * E(B-V). 
        # Very rough A(lambda)/A(V) for optical
        # This is just a placeholder; user likely has extinction installed in common envs.
        print("[WARN] 'extinction' package not found. Extinction correction not applied.")
        return flux


def find_file_in_tree(root_dir, fname):
    """
    Searches for fname in root_dir and its subdirectories.
    Returns full path if found, else None.
    Case-insensitive search.
    """
    fname_lower = fname.lower()
    
    # Check directly first (case-sensitive try, then insensitive)
    p = os.path.join(root_dir, fname)
    if os.path.exists(p):
        return p
        
    # Walk
    for root, dirs, files in os.walk(root_dir):
        # Case insensitive match
        for f in files:
            if f.lower() == fname_lower:
                return os.path.join(root, f)
    return None

def has_coverage(x_rest, L0, window_A=200.0, margin_A=20.0):
    if len(x_rest) == 0: return False
    lam_min = np.nanmin(x_rest); lam_max = np.nanmax(x_rest)
    return (lam_min <= (L0 - window_A + margin_A)) and (lam_max >= (L0 + window_A - margin_A))

def _robust_linear_continuum(x, y, mask_core, n_iter=4, clip_sigma=2.5):
    m = (~mask_core) & np.isfinite(x) & np.isfinite(y)
    xx, yy = x[m], y[m]
    if len(xx) < 10:
        return np.poly1d([0.0, np.nanmedian(y) if len(y)>0 else 1.0])
    w = np.ones_like(yy, float)
    p = [0.0, 1.0]
    for _ in range(n_iter):
        try:
            p = np.polyfit(xx, yy, 1, w=w)
            resid = yy - np.polyval(p, xx)
            s = np.nanstd(resid)
            if not np.isfinite(s) or s <= 0: break
            w = (np.abs(resid) < clip_sigma*s).astype(float)
            if w.sum() < 10: break
        except:
            break
    return np.poly1d(p)

def _robust_std(a):
    a = np.asarray(a); a = a[np.isfinite(a)]
    if a.size < 5: return np.nan
    med = np.nanmedian(a)
    mad = np.nanmedian(np.abs(a - med))
    return 1.4826 * mad

def gauss_absorption_4p(x, A, mu, sigma, C):
    # C is local baseline
    return C - A * np.exp(-0.5 * ((x - mu) / sigma)**2)

# =============================================================================
#                                  CA II HELPERS
# =============================================================================

def _gauss(x, A, mu, sig):
    return A * np.exp(-0.5*((x - mu)/np.maximum(sig,1e-6))**2)

def _twoP_triplet_model_shared_blue(x, theta, names, L0s, L_ref):
    """
    theta layout:
      [ vexp_b, vexp_r, vfwhm_b, vfwhm_r,   C0, C1,
        Aem_b, dmu_em_b, sigem_b, Aabs_b,
        Aem_r, dmu_em_r, sigem_r, Aabs_r ]
    """
    # configuration
    BLUE_LINES = ["Ca II 8498", "Ca II 8542"]
    RED_LINES  = ["Ca II 8662"]
    c_kms = 299792.458
    _K = np.sqrt(2.0*np.log(2.0))
    
    vexp_b, vexp_r, vfwhm_b, vfwhm_r, C0, C1 = theta[:6]
    Aem_b, dmu_em_b, sigem_b, Aabs_b = theta[6:10]
    Aem_r, dmu_em_r, sigem_r, Aabs_r = theta[10:14]

    y = C0 + C1*(x - L_ref)

    # BLUE system contributions (applied to both blue lines)
    for i, nm in enumerate(names):
        L0 = L0s[i]
        if nm in BLUE_LINES:
            mu_abs  = L0 * (1.0 - vexp_b/c_kms)
            sig_abs = (vfwhm_b / c_kms) * L0 / (2.0*_K)
            mu_em   = L0 + dmu_em_b
            y = y + _gauss(x, Aem_b, mu_em,   sigem_b) - _gauss(x, Aabs_b, mu_abs, sig_abs)

    # RED system contribution (only 8662)
    for i, nm in enumerate(names):
        if nm in RED_LINES:
            L0 = L0s[i]
            mu_abs  = L0 * (1.0 - vexp_r/c_kms)
            sig_abs = (vfwhm_r / c_kms) * L0 / (2.0*_K)
            mu_em   = L0 + dmu_em_r
            y = y + _gauss(x, Aem_r, mu_em,   sigem_r) - _gauss(x, Aabs_r, mu_abs, sig_abs)

    return y

def fit_caii_triplet_true_joint_rest(x_rest, flux,
                                     window_A=300.0):
    """
    One wide window that spans the entire triplet. Fit a single model with:
      - one shared BLUE P-Cygni system (emission & absorption shared) for 8498+8542
      - one RED P-Cygni system for 8662
      
    Returns dict with model, params, and velocities for 8662.
    """
    names = ["Ca II 8498","Ca II 8542","Ca II 8662"]
    CAII_TRIPLET = {
        "Ca II 8498": 8498.02,
        "Ca II 8542": 8542.09,
        "Ca II 8662": 8662.14,
    }
    L0s = np.array([CAII_TRIPLET[nm] for nm in names], float)
    L_ref = float(np.mean(L0s))

    # --- global slice (ONE window) ---
    lam_lo = np.min(L0s) - window_A
    lam_hi = np.max(L0s) + window_A
    m = (x_rest >= lam_lo) & (x_rest <= lam_hi)
    x = x_rest[m]; y = flux[m]
    if x.size < 120:
        return {"ok": False, "reason": "too_few_points_in_global_window"}

    # robust normalization
    q75 = np.nanquantile(y, 0.75)
    norm = q75 if np.isfinite(q75) and q75 != 0 else np.nanmedian(y)
    if not np.isfinite(norm) or norm == 0: norm = 1.0
    yn = y / norm

    # --- initial guesses & bounds ---
    vexp_b0, vexp_r0   = 1500.0, 1500.0          # km/s
    vfwhm_b0, vfwhm_r0 = 1200.0, 1200.0          # km/s
    C0_0, C1_0 = float(np.nanmedian(yn)), 0.0

    MIN_VFWHM_KMS = 700.0
    MAX_VFWHM_KMS = 5000.0
    MIN_SIGEM_A   = 0.0
    MAX_SIGEM_A   = 120.0

    dmu_em_b0 = 0.0
    dmu_em_r0 = 0.0
    DEMU_MAX  = window_A
    # Stricter bounds for blue emission to prevent overlap with 8662
    DEMU_BLUE_MAX = 20.0

    Aem_b0, Aabs_b0 = 0.5, 0.6
    Aem_r0, Aabs_r0 = 0.25, 0.6
    AEM_MAX  = 2.0
    AABS_MAX = 2.0

    sigem_b0 = 10.0
    sigem_r0 = 20.0
    MAX_SIGEM_BLUE = 20.0
    p0 = [
        vexp_b0, vexp_r0, vfwhm_b0, vfwhm_r0, C0_0, C1_0,
        Aem_b0, dmu_em_b0, sigem_b0, Aabs_b0,
        Aem_r0, dmu_em_r0, sigem_r0, Aabs_r0
    ]
    low = [
        0.0,              0.0,              MIN_VFWHM_KMS, MIN_VFWHM_KMS, 0.0,   -1.0/window_A,
        0.0,             -DEMU_MAX,         MIN_SIGEM_A,   0.0,
        0.0,             -DEMU_MAX,         MIN_SIGEM_A,   0.0
    ]
    high = [
        12000.0,          12000.0,          MAX_VFWHM_KMS, MAX_VFWHM_KMS, 2.0,    1.0/window_A,
        AEM_MAX,          DEMU_MAX,         MAX_SIGEM_A,   AABS_MAX,
        AEM_MAX,          DEMU_MAX,         MAX_SIGEM_A,   AABS_MAX
    ]

    # weights by proximity to any line center
    d = np.min([np.abs(x - L0) for L0 in L0s], axis=0)
    w = 1.0 / np.sqrt(1.0 + (d/(0.7*window_A))**2)

    def _model_for_cf(xx, *theta):
        return _twoP_triplet_model_shared_blue(xx, theta, names, L0s, L_ref)

    try:
        popt, pcov = curve_fit(
            _model_for_cf, x, yn,
            p0=p0, bounds=(low, high),
            sigma=1.0/np.clip(w, 1e-3, None),
            absolute_sigma=False, maxfev=250000
        )
    except Exception as e:
        return {"ok": False, "reason": f"curve_fit_failed: {e}"}

    # --- unpack + simple errors ---
    vexp_b, vexp_r, vfwhm_b, vfwhm_r, C0, C1 = popt[:6]
    
    def _safe_sqrt(v): 
        return float(np.sqrt(v)) if (np.isfinite(v) and v >= 0) else np.nan
    vr_err = _safe_sqrt(pcov[1,1])  if pcov is not None and pcov.shape[0] > 1 else np.nan

    # --- build full model on the λ grid ---
    ymodel_norm = _twoP_triplet_model_shared_blue(x, popt, names, L0s, L_ref)
    ymodel = ymodel_norm * norm

    # We want to return the parameters for the RED system (8662) specifically
    # Because that is what we plot in the velocity plot usually.
    # The notebook saved: v_center_kms (from model trough), v_err, etc.
    
    # Calculate trough explicitly for 8662
    # Search around predicted absorption center
    L0_red = CAII_TRIPLET["Ca II 8662"]
    mu_abs_pred = L0_red * (1.0 - vexp_r/c_kms)
    
    # Fine grid search for minimum
    xg = np.linspace(mu_abs_pred - 40, mu_abs_pred + 40, 500)
    yg = _twoP_triplet_model_shared_blue(xg, popt, names, L0s, L_ref)
    idx_min = np.argmin(yg)
    mu_min = xg[idx_min]
    
    v_abs_trough = c_kms * (L0_red - mu_min) / L0_red
    
    # Use curve_fit parameter error for formal error, plus system error ~ 0.5A
    v_err_sys = (c_kms / L0_red) * 0.5
    v_err_fit = vr_err # This is error on parameter vexp_r, not trough.
                       # But vexp_r is the parameter describing absorption center.
                       # So it is a good proxy.
    v_err = np.sqrt(v_err_fit**2 + v_err_sys**2)

    return {
        "ok": True,
        "x_obs": x,       # Fit window
        "y_obs": y,       # Obs flux
        "y_model": ymodel, 
        "fit_params": popt,
        "norm": norm,
        "red_results": {
             "line": "Ca II 8662",
             "v_exp_kms": vexp_r, # Parameter velocity
             "v_abs_kms": v_abs_trough, # Trough velocity
             "v_err_kms": v_err,
             "chi2_red": 1.0 # Placeholder
        }
    }

def fit_absorption_only(x_rest, y, L0,
                        window_A=300.0,
                        vmin_kms=0, vmax_kms=5000,
                        smooth_pts=21, smooth_poly=2,
                        core_exclude_blue_A=160.0, core_exclude_red_A=40.0,
                        fit_halfwidth_A=60.0,
                        lam_sys_A=0.5,
                        use_bootstrap=0,
                        fwhm_min_kms=0.0):
    """
    Returns dict with params, velocities (+uncertainties), chi2_red, status.
    """
    # slice
    mwin = (x_rest >= L0-window_A) & (x_rest <= L0+window_A)
    xs, ys = x_rest[mwin], y[mwin]
    if len(xs) < 40:
        return {"ok": False, "reason": "too_few_points"}

    # continuum from sidebands
    core_low, core_high = L0-core_exclude_blue_A, L0+core_exclude_red_A
    mask_core = (xs >= core_low) & (xs <= core_high)
    cont_poly = _robust_linear_continuum(xs, ys, mask_core)
    cont = cont_poly(xs)
    cont[~np.isfinite(cont)] = np.nanmedian(cont)
    cont = np.where(cont <= 0, np.nanmedian(cont[cont > 0]), cont)
    ycn = ys/cont

    # allowed region by velocity
    lam_min_allowed = L0 * (1.0 - vmax_kms/c_kms)
    lam_max_allowed = L0 * (1.0 - vmin_kms/c_kms)
    mallow = (xs >= lam_min_allowed) & (xs <= lam_max_allowed)
    if mallow.sum() < 8:
        return {"ok": False, "reason": "no_points_in_allowed_range"}

    # --- min FWHM velocity constraint translated to sigma (Å) ---
    if fwhm_min_kms and fwhm_min_kms > 0:
        sigma_min_A = (L0 * (fwhm_min_kms / c_kms)) / 2.3548
    else:
        sigma_min_A = 1.0  # original floor

    # smoothing for trough pick
    n = mallow.sum()
    win = min(smooth_pts if smooth_pts % 2 == 1 else smooth_pts + 1,
              n if n % 2 == 1 else n - 1)
    win = max(7, win)
    ysm = ycn.copy()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        try:
            ysm[mallow] = savgol_filter(ycn[mallow], win, polyorder=min(smooth_poly, win-1))
        except:
            pass

    # initial min
    try:
        idx_local = np.argmin(ysm[mallow])
        idx = np.where(mallow)[0][idx_local]
        mu0 = xs[idx]
    except:
        return {"ok": False, "reason": "min_finding_failed"}

    # fit window
    mfit = (xs >= mu0 - fit_halfwidth_A) & (xs <= mu0 + fit_halfwidth_A)
    if mfit.sum() < 15:
        return {"ok": False, "reason": "too_few_points_in_fit_window"}
    xfit, yfit = xs[mfit], ycn[mfit]

    # noise from sidebands of fit window (exclude ±20 Å)
    side_mask = (xfit < mu0 - 20.0) | (xfit > mu0 + 20.0)
    sig_y = _robust_std(yfit[side_mask])
    if not np.isfinite(sig_y) or sig_y <= 0:
        aa = (xs > mu0 - fit_halfwidth_A) & (xs < mu0 + fit_halfwidth_A) \
             & ((xs < mu0 - 20.0) | (xs > mu0 + 20.0))
        sig_y = _robust_std(ycn[aa])
    if not np.isfinite(sig_y) or sig_y <= 0:
        sig_y = 0.02
    sigma_vec = np.full_like(yfit, sig_y, dtype=float)

    # weighted fit
    A0 = np.clip(1.0 - ysm[idx], 0.05, 1.5)
    # estimate a local baseline from the sidebands of the fit window
    C0 = float(np.median(yfit[side_mask])) if np.any(side_mask) else 1.0

    p0 = [A0, mu0, 15.0, C0]
    bounds = (
        [0.0,              lam_min_allowed,  sigma_min_A,  0.5],   # A, mu, sigma, C(min)
        [2.5,              lam_max_allowed, 120.0, 3.0],   # A, mu, sigma, C(max)
    )

    try:
        pop, pcov = curve_fit(
            gauss_absorption_4p, xfit, yfit, p0=p0, bounds=bounds,
            sigma=sigma_vec, absolute_sigma=True, maxfev=20000
        )
    except Exception as e:
        return {"ok": False, "reason": str(e)}

    A, mu, sigma, C = pop
    model = gauss_absorption_4p(xfit, *pop)
    # uncertainties & quality
    mu_err = float(np.sqrt(pcov[1,1])) if (pcov is not None and np.isfinite(pcov[1,1])) else np.nan
    v_abs = c_kms * (mu - L0) / L0
    v_exp = -v_abs
    v_err_fit = (c_kms / L0) * mu_err
    v_err_sys = (c_kms / L0) * float(lam_sys_A)
    v_err = np.sqrt(v_err_fit**2 + v_err_sys**2)

    dof = max(len(yfit) - 3, 1)
    chi2 = np.sum(((yfit - model)/sig_y)**2)
    chi2_red = chi2 / dof

    return {
        "ok": True,
        "fit_params": [A, mu, sigma, C],
        "params": dict(A=A, mu=mu, sigma=sigma, mu_err=mu_err,C=C),
        "velocities": dict(v_abs=v_abs, v_exp=v_exp, v_err=v_err,
                           v_err_fit=v_err_fit, v_err_sys=v_err_sys,
                           chi2_red=chi2_red, sigma_flux=sig_y),
        "L0": L0,
        "x_fit": xfit,
        "y_fit": yfit,
        "y_obs": ycn[mfit]
    }

# =============================================================================
#                                  IR HELPERS
# =============================================================================

_K = np.sqrt(2.0*np.log(2.0))

def _line_model(lam, L0, a0, a1, A_em, mu_em, sig_em, A_abs, mu_abs, sig_abs):
    cont = a0 + a1*(lam - L0)
    em   = A_em  * np.exp(-0.5*((lam - mu_em)/np.maximum(sig_em,1e-6))**2)
    absb = A_abs * np.exp(-0.5*((lam - mu_abs)/np.maximum(sig_abs,1e-6))**2)
    return cont + em - absb

def fit_three_coii_lines_joint(wl_rest, flux, line_dict, window_A=180.0):
    # ---- prep per-line windows (rest frame) ----
    # wl_rest is already de-redshifted passed in
    arrays = []          # (name, L0, xx, yy_norm)
    for name, L0 in line_dict.items():
        m = (wl_rest >= L0 - window_A) & (wl_rest <= L0 + window_A)
        xx = wl_rest[m]
        yy = flux[m]
        if xx.size < 40:
            continue
        q75 = np.nanquantile(yy, 0.75)
        norm = q75 if np.isfinite(q75) and q75 != 0 else np.nanmedian(yy)
        if not np.isfinite(norm) or norm == 0:
            norm = 1.0
        
        arrays.append((name, L0, xx, yy / norm))

    if not arrays:
        return {"ok": False, "reason": "no_lines_with_coverage"}

    # concatenate data for curve_fit
    cat_x = np.concatenate([xx for (_, _, xx, _) in arrays])
    cat_y = np.concatenate([yy for (_, _, _, yy) in arrays])
    which_line = np.concatenate([np.full(xx.size, i, dtype=int)
                                 for i, (_, _, xx, _) in enumerate(arrays)])
    L0_per_line = np.array([L0 for (_, L0, _, _) in arrays], float)
    nL = len(arrays)

    # ---------- packing helpers ----------
    def unpack(theta):
        v_exp = theta[0]     # km/s (shared)
        v_fwhm = theta[1]    # km/s (shared)
        per_line = []
        base = 2
        for _ in range(nL):
            # (a0, a1, Aem, muem, sigem, Aabs)  -- NO per-line sigma_abs
            a0, a1, Aem, muem, sigem, Aabs = theta[base:base+6]
            per_line.append((a0, a1, Aem, muem, sigem, Aabs))
            base += 6
        return v_exp, v_fwhm, per_line

    def model_concat(lam_cat, *theta):
        v_exp, v_fwhm, per_line = unpack(theta)
        out = np.empty_like(lam_cat)
        for i in range(nL):
            idx = (which_line == i)
            if not np.any(idx):
                continue
            L0 = L0_per_line[i]
            a0, a1, Aem, muem, sigem, Aabs = per_line[i]
            mu_abs_i = L0 * (1.0 - v_exp/c_kms)
            # shared FWHM (km/s) -> sigma_abs in Å for this L0
            sigma_abs_i = (v_fwhm / c_kms) * L0 / (2.0*_K)
            out[idx] = _line_model(lam_cat[idx], L0, a0, a1, Aem, muem, sigem,
                                   Aabs, mu_abs_i, sigma_abs_i)
        return out

    # ---------- initial guesses & bounds ----------
    p0 = [1500.0, 1200.0]          # v_exp_kms, v_fwhm_kms (both shared)
    low = [500.0,  400.0]          # velocity >= 500 (constraint), modest min FWHM
    high = [10000.0, 4000.0]       # caps

    for _, L0, xx, yy in arrays:
        a0_0, a1_0 = float(np.nanmedian(yy)), 0.0
        Aem_0, muem_0, sigem_0 = 0.2, L0 + 0.15*0.5*window_A, 20.0
        Aabs_0 = 0.5
        p0 += [a0_0, a1_0, Aem_0, muem_0, sigem_0, Aabs_0]

        # Per-line bounds (emission & continuum & absorption depth only).
        low  += [0.0, -1.0/window_A, 0.0, L0 - window_A,   3.0,  0.0]
        high += [2.0,  1.0/window_A, 2.0, L0 + window_A, 120.0,  2.0]

    # mild central weighting per-point
    w = 1.0 / np.sqrt(1.0 + ((cat_x - L0_per_line[which_line])/(0.6*window_A))**2)

    # ---------- fit ----------
    try:
        popt, pcov = curve_fit(
            model_concat, cat_x, cat_y,
            p0=p0, bounds=(low, high),
            sigma=1.0/np.clip(w, 1e-3, None),
            absolute_sigma=False, maxfev=80000
        )
    except Exception as e:
        return {"ok": False, "reason": f"curve_fit_failed: {e}"}

    # unpack fitted parameters
    v_exp_kms, v_fwhm_kms, per_line = unpack(popt)
    v_exp_err_kms  = float(np.sqrt(pcov[0,0])) if (pcov is not None and np.isfinite(pcov[0,0])) else np.nan
    v_fwhm_err_kms = float(np.sqrt(pcov[1,1])) if (pcov is not None and np.isfinite(pcov[1,1])) else np.nan

    # assemble per-line rows
    rows = []
    for i, (name, L0, xx, yy) in enumerate(arrays):
        a0, a1, Aem, muem, sigem, Aabs = per_line[i]
        mu_abs = L0 * (1.0 - v_exp_kms/c_kms)
        Sigma_abs_A = (v_fwhm_kms / c_kms) * L0 / (2.0*_K)
        FWHM_A = 2.0*_K*Sigma_abs_A

        # User Request: Redefine velocity as the minimum of the full shape (profile) 
        # instead of the Gaussian component center (v_exp_kms).
        # We need to find the lambda where the model flux is minimized for this specific line.
        
        # Create a fine grid around the expected absorption
        # v_exp is approx center. Search +/- 2 FWHM
        lam_search_min = L0 * (1.0 - (v_exp_kms + 3*v_fwhm_kms)/c_kms) 
        lam_search_max = L0 * (1.0 - (v_exp_kms - 3*v_fwhm_kms)/c_kms)
        lam_fine = np.linspace(lam_search_min, lam_search_max, 500)
        
        # Evaluate FULL model for this line (emission + absorption)
        # Note: _line_model takes single L0 and its params.
        y_fine = _line_model(lam_fine, L0, a0, a1, Aem, muem, sigem, Aabs, mu_abs, Sigma_abs_A)
        
        idx_min = np.argmin(y_fine)
        lam_at_min = lam_fine[idx_min]
        
        # Convert lambda_min to velocity (Doppler shift)
        # v_at_min = c * (lam - L0) / L0. This will be negative (blueshift).
        v_at_min = c_kms * (lam_at_min - L0) / L0
        
        # Our convention: v_center_kms is a POSITIVE expansion velocity.
        # So v_center_kms = -v_at_min
        v_profile_kms = -v_at_min

        # Use this new velocity as the primary measurement
        rows.append({
            "line": name, "L0_A": L0,
            "a0": a0, "a1": a1,
            "A_em": Aem, "mu_em_A": muem, "sig_em_A": sigem,
            "A_abs": Aabs, "mu_abs_A": mu_abs, "sig_abs_A": Sigma_abs_A,
            "FWHM_A": FWHM_A,
            "v_center_kms": v_profile_kms,      "v_center_err_kms": v_exp_err_kms,   # Error is approx same
            "v_gaussian_kms": v_exp_kms,        # Keep original for reference
            "v_fwhm_kms":   v_fwhm_kms, "v_fwhm_err_kms":   v_fwhm_err_kms,  # shared
            "ok": True,
            "x_fit": xx,       # Fit window X (rest)
            "y_norm": yy, # Normalized flux for plotting
        })

    # SORT arrays to prevent zigzag in plots if windows overlap
    y_mod_eval = model_concat(cat_x, *popt)
    sort_idx = np.argsort(cat_x)
    cat_x_sorted = cat_x[sort_idx]
    y_mod_sorted = y_mod_eval[sort_idx]
    y_obs_sorted = cat_y[sort_idx]

    return {
        "ok": True, 
        "rows": rows,
        "x_obs": cat_x_sorted, 
        "y_obs": y_obs_sorted,
        "y_model": y_mod_sorted, 
        "fit_params": popt 
    }

# =============================================================================
#                                  PLOTTING
# =============================================================================

def plot_velocities_combined(
    df_abs,
    out_dir=None,
    filename="velocities_vs_phase_combined.pdf",
    max_rel_err=0.5,          # drop if σv > 50% of |v|
    max_abs_err=3000,         # or σv > 3000 km/s
    phase_limits=None,        # dict like {"Si II 6355": (-5, 30), ...}
    add_df_co=None,           # PASS THE DATAFRAME DIRECTLY instead of csv path
    # Ca-triplet display
    plot_ca="single",         # "single" | "two_systems" | "all"
    ca_rep="Ca II 8662",      # which Ca to keep when plot_ca="single" (or to represent blue pair)
):
    """
    Plot expansion velocities vs phase; optionally overlay per-line fits.
    """
    
    ca_names = {"Ca II 8498", "Ca II 8542", "Ca II 8662"}
    ca_blue = {"Ca II 8498", "Ca II 8542"}
    ca_red  = {"Ca II 8662"}

    if df_abs is None or len(df_abs) == 0:
        print("df_abs is empty.")
        return

    q = df_abs.copy()
    # Normalize column names if needed or ensure they exist
    if "phase_days" not in q.columns or "line" not in q.columns:
        print("df_abs must contain 'phase_days' and 'line' columns.")
        return

    if "ok" in q.columns:
        q = q[q["ok"] == True]
    q = q[np.isfinite(q["phase_days"])]

    # build v_plot_kms and v_err_plot_kms
    q["v_plot_kms"] = np.nan
    q["v_err_plot_kms"] = np.nan

    if {"mu_abs_A", "L0_A"}.issubset(q.columns):
        m_ca_abs = q["line"].isin(ca_names) & q["mu_abs_A"].notna() & q["L0_A"].notna()
        # For Ca II, if mu_abs_A exists, calculate velocity from it
        q.loc[m_ca_abs, "v_plot_kms"] = c_kms * (q.loc[m_ca_abs, "L0_A"] - q.loc[m_ca_abs, "mu_abs_A"]) / q.loc[m_ca_abs, "L0_A"]
        
        # Try to find error
        if "v_center_err_kms" in q.columns:
             q.loc[m_ca_abs, "v_err_plot_kms"] = q.loc[m_ca_abs, "v_center_err_kms"]
        elif "v_err_kms" in q.columns:
             q.loc[m_ca_abs, "v_err_plot_kms"] = q.loc[m_ca_abs, "v_err_kms"]

    # Fill others
    m_need = q["v_plot_kms"].isna()
    if "v_exp_kms" in q.columns:
        has_vexp = m_need & q["v_exp_kms"].notna()
        q.loc[has_vexp, "v_plot_kms"] = q.loc[has_vexp, "v_exp_kms"]
        if "v_err_kms" in q.columns:
            q.loc[has_vexp, "v_err_plot_kms"] = q.loc[has_vexp, "v_err_kms"]

    m_need = q["v_plot_kms"].isna()
    if "v_center_kms" in q.columns:
        has_vc = m_need & q["v_center_kms"].notna()
        q.loc[has_vc, "v_plot_kms"] = q.loc[has_vc, "v_center_kms"]
        if "v_center_err_kms" in q.columns:
            q.loc[has_vc, "v_err_plot_kms"] = q.loc[has_vc, "v_center_err_kms"]

    # Fill errors if missing
    m_has_v = q["v_plot_kms"].notna() & q["v_err_plot_kms"].isna()
    if "v_err_kms" in q.columns:
        q.loc[m_has_v & q["v_err_kms"].notna(), "v_err_plot_kms"] = q.loc[m_has_v & q["v_err_kms"].notna(), "v_err_kms"]
    if "v_center_err_kms" in q.columns:
        m_has_v = q["v_plot_kms"].notna() & q["v_err_plot_kms"].isna()
        q.loc[m_has_v & q["v_center_err_kms"].notna(), "v_err_plot_kms"] = q.loc[m_has_v & q["v_center_err_kms"].notna(), "v_center_err_kms"]

    q = q[np.isfinite(q["v_plot_kms"]) & np.isfinite(q["v_err_plot_kms"])]
    if q.empty:
        print("No valid rows after assembling velocities.")
        return

    # Collapse Ca if requested
    mode = str(plot_ca).lower()
    if mode == "single":
        q = q[~q["line"].isin(ca_names) | (q["line"] == ca_rep)].copy()
    elif mode == "two_systems":
        blue_rep = ca_rep if ca_rep in ca_blue else "Ca II 8542"
        keep_mask = (~q["line"].isin(ca_names)) | (q["line"] == blue_rep) | (q["line"].isin(ca_red))
        q = q[keep_mask].copy()
        q.loc[q["line"] == blue_rep, "line"] = "Ca II (blue pair)"

    # Quality cuts
    q = q[(q["v_err_plot_kms"] < max_abs_err) &
          (q["v_err_plot_kms"] < max_rel_err * q["v_plot_kms"].abs())]
          
    if q.empty:
        print("All points were purged by error cuts.")
        return

    line_styles = {
        "Si II 6355": {"color": "#44AA99", "marker": "o"},
        "Fe II 5169": {"color": "#FF2F14", "marker": "s"},
        "C II 6578":  {"color": "#332288", "marker": "D"},
        "C II 7231":  {"color": "#ee79f2", "marker": "o"},
        "Na I 5893":  {"color": "#ff7f0e", "marker": "^"},
        "Fe II 6149": {"color": "#9467bd", "marker": "v"},
        "Fe II 6247": {"color": "crimson", "marker": "P"},
        "Ca II 8498": {"color": "#e41a1c", "marker": "X"},
        "Ca II 8542": {"color": "#377eb8", "marker": "X"},
        "Ca II 8662": {"color": "#0772a3", "marker": "X"},
        "Ca II (blue pair)": {"color": "#377eb8", "marker": "X"},
    }

    # If add_df_co is provided, merge it into q for plotting
    if add_df_co is not None and not add_df_co.empty:
        # print("DEBUG: merging Co II data:", add_df_co["line"].unique())
        # Normalize columns if needed
        co = add_df_co.copy()
        
        # Calculate v_min_kms if missing (standardize column names)
        if "v_min_kms" not in co.columns:
             # Prioritize explicitly calculated v_center_kms (profile minimum)
             if "v_center_kms" in co.columns:
                 co["v_min_kms"] = co["v_center_kms"]
             elif "v_exp_kms" in co.columns:
                 co["v_min_kms"] = co["v_exp_kms"]
             elif {"L0_A", "mu_abs_A"}.issubset(co.columns):
                 # Fallback to Gaussian center if v_center_kms not present
                 co["v_min_kms"] = c_kms * (co["L0_A"] - co["mu_abs_A"]) / co["L0_A"]
                 
        if "v_min_err_kms" not in co.columns:
            if "v_err_kms" in co.columns:
                co["v_min_err_kms"] = co["v_err_kms"]
            else:
                co["v_min_err_kms"] = np.nan

        # Map to plot columns
        co["v_plot_kms"] = -co["v_min_kms"] # Plot negative velocity
        co["v_err_plot_kms"] = co["v_min_err_kms"]
        
        # Concatenate
        q = pd.concat([q, co], ignore_index=True)
        # print("DEBUG: q lines after merge:", q["line"].unique())

    # ... Now standard loop plots everyone ...

    # ---- Nature-style drawing (revision 3); data selection above is unchanged ----
    def _lam_label(name):
        parts = name.split()
        if len(parts) > 1 and parts[-1].isdigit():
            return " ".join(parts[:-1]) + r" $\lambda$" + parts[-1]
        return name

    fig, ax = plt.subplots(figsize=(0.66 * ns.DOUBLE_COL, 0.56 * ns.DOUBLE_COL))
    handles = {}
    ekw = dict(ms=3.0, lw=0.6, elinewidth=0.6, capsize=1.0, capthick=0.5,
               markeredgewidth=0.6, markerfacecolor='none', ls='none')

    for line_name in sorted(q["line"].astype(str).unique()):
        df_line = q[q["line"].astype(str) == line_name].sort_values("phase_days")
        if phase_limits and line_name in phase_limits:
            tmin, tmax = phase_limits[line_name]
            df_line = df_line[(df_line["phase_days"] >= tmin) & (df_line["phase_days"] <= tmax)]
        if df_line.empty:
            continue
        if line_name[:2] != 'Co':
            st = line_styles.get(line_name, {"color": "k", "marker": "o"})
            h = ax.errorbar(
                df_line["phase_days"].values,
                df_line["v_plot_kms"].values,
                yerr=df_line["v_err_plot_kms"].values,
                fmt=st["marker"], color=st["color"], ecolor=st["color"],
                label=_lam_label(line_name), **ekw)
            handles[line_name] = h

    line_styles = {
        "Co II 15759": {"color": "#ffae00", "marker": "o"},
        "Co II 16064": {"color": "#bcbd22", "marker": "o"},
        "Co II 16361": {"color": "#8ec103", "marker": "o"},
    }

    if add_df_co is not None and not add_df_co.empty:
        try:
            co = add_df_co.copy()
            if "v_min_kms" not in co.columns:
                 if "v_center_kms" in co.columns:
                     co["v_min_kms"] = co["v_center_kms"]
                 elif {"L0_A", "mu_abs_A"}.issubset(co.columns):
                     co["v_min_kms"] = c_kms * (co["L0_A"] - co["mu_abs_A"]) / co["L0_A"]
            if "v_min_err_kms" not in co.columns:
                if "v_center_err_kms" in co.columns:
                    co["v_min_err_kms"] = co["v_center_err_kms"]
                else:
                    co["v_min_err_kms"] = np.nan

            co_lines_to_overlay = ["Co II 16064","Co II 16361","Co II 15759"]
            for ln in co_lines_to_overlay:
                dfm = co[(co["line"] == ln) & np.isfinite(co["phase_days"])].sort_values("phase_days")
                if dfm.empty:
                    continue
                if phase_limits and ln in phase_limits:
                    tmin, tmax = phase_limits[ln]
                    dfm = dfm[(dfm["phase_days"] >= tmin) & (dfm["phase_days"] <= tmax)]
                st = line_styles.get(ln, {"color": "#555555", "marker": "*"})
                h = ax.errorbar(
                    dfm["phase_days"].values,
                    dfm["v_min_kms"].values,
                    yerr=dfm["v_min_err_kms"].values if np.any(np.isfinite(dfm["v_min_err_kms"])) else None,
                    fmt="*", color=st["color"], ecolor=st["color"],
                    label=_lam_label(ln), **{**ekw, 'ms': 5.0})
                handles[ln] = h
        except Exception as e:
            print(f"[WARN] Could not overlay Co II minima: {e}")

    ax.axhline(0, color="k", ls=":", lw=0.5)
    ax.grid(True, linestyle=':', lw=0.4, alpha=0.6)
    ax.set_xlabel("Days from $B$-band maximum")
    ax.set_ylabel(r"Velocity (km s$^{-1}$)")
    order = ["C II 6578", "C II 7231", "Si II 6355", "Fe II 5169", "Fe II 6247",
             "Ca II 8662", "Co II 15759", "Co II 16064", "Co II 16361"]
    keys = [k for k in order if k in handles] + [k for k in handles if k not in order]
    ax.legend([handles[k] for k in keys], [_lam_label(k) for k in keys],
              loc="upper right", ncol=2, handletextpad=0.3, columnspacing=1.0,
              borderaxespad=0.4, labelspacing=0.3)
    ax.set_ylim(0, 10000)
    if q["v_plot_kms"].max() < 5000:
         ax.set_ylim(0, 4000)
    ax.set_xlim(-8, 75)
    ax.minorticks_on()
    ax.tick_params(which='both', top=True, right=True)

    fig.tight_layout()
    ns.save(fig, 'supplementary/SuppFig7_velocities')
    plt.close(fig)
    # machine-readable copy of exactly what is plotted (written to outputs/velocities)
    if out_dir:
        q.to_csv(os.path.join(out_dir, "SuppFig7_plotted_points.csv"), index=False)
    



def create_master_mosaic_figure(all_results, df_allowed, out_dir):
    """
    Creates a single figure with subplots for each line, showing STACKED (waterfall) 
    plots of only the 'approved' fits (those present in df_allowed).
    """
    import math
    import matplotlib.gridspec as gridspec
    
    if not all_results:
        return

    # 1. Filter results based on df_allowed
    # Build a lookup set for fast checking: (line_str, phase_float_round_2)
    # Using rounded phase to avoid float equality issues
    allowed_set = set()
    if df_allowed is not None and not df_allowed.empty:
        for _, row in df_allowed.iterrows():
            if pd.notna(row["line"]) and pd.notna(row["phase_days"]):
                key = (str(row["line"]), round(float(row["phase_days"]), 3))
                allowed_set.add(key)
    
    kept_results = []
    for item in all_results:
        k = (str(item["line"]), round(float(item["phase"]), 3))
        if k in allowed_set:
            kept_results.append(item)
            
    if not kept_results:
        print("No results matched the allowed plot set. Skipping master mosaic.")
        return

    # Group by line
    from collections import defaultdict
    by_line = defaultdict(list)
    for item in kept_results:
        by_line[item["line"]].append(item)
    
    # Sort lines: order by line_styles keys if possible, else alphabetical
    # We really want a nice order: e.g. blue to red?
    # Simple sort is fine for now
    # Custom sort order: C -> Si -> Fe -> Ca
    custom_order = [
         "C II 6578", "C II 7231", 
         "Si II 6355", 
         "Fe II 5169", "Fe II 6247", 
         "Ca II 8662"
    ]
    # Filter out Co II lines first
    line_keys = [k for k in by_line.keys() if not k.startswith("Co II")]
    
    # Sort: items in custom_order come first in that order, then others alphabetically
    line_names = sorted(line_keys, key=lambda x: custom_order.index(x) if x in custom_order else 999)
    
    n_plots = len(line_names)
    if n_plots == 0: return
    
    # Layout: 2 columns
    n_cols = 2
    n_rows = math.ceil(n_plots / n_cols)
    
    # ---- Nature-style drawing (revision 3); selection/fits above are unchanged ----
    from matplotlib.lines import Line2D
    fig = plt.figure(figsize=(ns.DOUBLE_COL, 0.98 * ns.DOUBLE_COL))
    gs = gridspec.GridSpec(n_rows, n_cols, figure=fig, wspace=0.08, hspace=0.16,
                           left=0.03, right=0.95, bottom=0.06, top=0.9)

    c_kms = 299792.458
    lw_data, lw_fit, lw_min = 0.5, 0.7, 0.7
    label_rows = []

    for i, line_name in enumerate(line_names):
        row_idx = i // n_cols
        col_idx = i % n_cols
        ax = fig.add_subplot(gs[row_idx, col_idx])

        items = by_line[line_name]
        # Sort items by phase descending (Earliest at TOP, Latest at BOTTOM)
        items.sort(key=lambda x: x["phase"], reverse=True)

        is_ca = line_name.startswith("Ca II")
        offset_step = 1.0 if is_ca else 0.5
        xlo, xhi = (-14000, 15000) if is_ca else (-8000, 8000)

        # right-hand end of the plotted data in this panel (labels go just beyond it)
        x_right = -np.inf
        for item in items:
            xv = c_kms * (item["x_rest"] - item["L0"]) / item["L0"]
            xv = xv[(xv > xlo) & (xv < xhi)]
            if xv.size:
                x_right = max(x_right, np.nanmax(xv))
        x_lab = min(x_right, xhi) + 0.025 * (xhi - xlo)
        # one common label level per panel (regular spacing -> labels never collide)
        _lv, _ymin, _ymax = [], [], []
        for j, item in enumerate(items):
            xv = c_kms * (item["x_rest"] - item["L0"]) / item["L0"]
            yv = np.asarray(item["y_obs"], float)
            w = (xv > xlo) & (xv < xhi) & np.isfinite(yv)
            if w.sum() > 5:
                _lv.append(np.nanmedian(yv[w][xv[w] >= np.nanpercentile(xv[w], 90)]))
                _ymin.append(np.nanmin(yv[w]) + j * offset_step)
                _ymax.append(np.nanmax(yv[w]) + j * offset_step)
        lab_level = np.nanmedian(_lv) if _lv else 1.0

        for j, item in enumerate(items):
            y_shift = j * offset_step
            res = item["res"]
            L0 = item["L0"]
            x_rest = item["x_rest"]
            x_vel = c_kms * (x_rest - L0) / L0
            y_obs = item["y_obs"]

            ax.step(x_vel, y_obs + y_shift, where='mid', color='k', lw=lw_data, alpha=0.8)

            if res["ok"]:
                if "y_model" in res:
                    y_mod = res["y_model"]
                    if "x_model" in res:
                        x_mod_vel = c_kms * (res["x_model"] - L0) / L0
                        ax.plot(x_mod_vel, y_mod + y_shift, color='r', lw=lw_fit, alpha=0.9)
                elif "fit_params" in res and res["fit_params"] is not None:
                     params = res["fit_params"]
                     x_f = np.linspace(x_rest.min(), x_rest.max(), 200)
                     y_f = gauss_absorption_4p(x_f, *params)
                     x_vf = c_kms * (x_f - L0) / L0
                     ax.plot(x_vf, y_f + y_shift, color='r', lw=lw_fit, alpha=0.9)

            if "v_center_kms" in res:
                v_min = res["v_center_kms"]
                if np.isfinite(v_min):
                    ax.plot([-v_min, -v_min], [y_shift, y_shift + 1.2], color='b', ls=':',
                            lw=lw_min, alpha=0.8)

            # phase label (2 decimals: spectra taken on the same night stay distinguishable)
            ph = item['phase']
            y_lab = y_shift + lab_level
            txt = f"{ph:+.2f}".replace("-", "−")
            ax.text(x_lab, y_lab, txt, fontsize=ns.FS_TINY, color='#222', va='center', ha='left')
            label_rows.append({"line": line_name, "phase_label": txt, "phase_days": ph,
                               "file": item.get("file", ""), "v_center_kms": res.get("v_center_kms", np.nan)})

        parts = line_name.split()
        if len(parts) > 1 and parts[-1].isdigit():
            title_str = " ".join(parts[:-1]) + r" $\lambda$" + parts[-1]
        else:
            title_str = line_name
        ax.set_title(title_str, fontsize=ns.FS, pad=2)
        ax.set_xlim(xlo, xhi)
        if _ymin:
            y0, y1 = min(_ymin), max(max(_ymax), (len(items) - 1) * offset_step + 1.2)
            pad = 0.04 * (y1 - y0)
            ax.set_ylim(y0 - pad, y1 + pad)
        ax.set_yticks([])
        ax.tick_params(axis='x', which='both', top=True)
        ax.minorticks_on()
        ax.tick_params(axis='y', which='both', left=False, right=False)

        if row_idx == n_rows - 1:
            ax.set_xlabel(r"Velocity (km s$^{-1}$)")
        else:
            ax.tick_params(labelbottom=False)
        if col_idx == 0 and row_idx == 1:
            ax.set_ylabel("Normalized flux + offset")

        ax.grid(True, axis='x', linestyle=':', lw=0.4, alpha=0.4)

    key = [Line2D([], [], color='k', lw=lw_data, drawstyle='steps-mid', label='Data'),
           Line2D([], [], color='r', lw=lw_fit, label='Fit'),
           Line2D([], [], color='b', ls=':', lw=lw_min, label='Absorption minimum')]
    fig.legend(handles=key, loc='lower center', bbox_to_anchor=(0.5, 0.925), ncol=3,
               handlelength=2.2, columnspacing=1.5, fontsize=ns.FS_SMALL)

    ns.save(fig, 'supplementary/SuppFig8_velocity_fits')
    plt.close(fig)
    if out_dir:
        pd.DataFrame(label_rows).to_csv(os.path.join(out_dir, "SuppFig8_panel_labels.csv"), index=False)


def save_fit_mosaics(all_results, out_dir):
    """
    Saves mosaic plots for each line, showing all epochs stacked.
    all_results: list of dicts with keys: line, phase, res, x_rest, y_obs, file
    """
    import math
    os.makedirs(out_dir, exist_ok=True)
    
    # Group by line
    from collections import defaultdict
    by_line = defaultdict(list)
    for item in all_results:
        by_line[item["line"]].append(item)
        
    for line_name, items in by_line.items():
        # Sort by phase
        items.sort(key=lambda x: x["phase"])
        
        n_plots = len(items)
        if n_plots == 0: continue
        

        # Layout: 4 cols
        n_cols = 4
        n_rows = math.ceil(n_plots / n_cols)
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 3*n_rows), sharex=True, sharey=True)
        # axes is (n_rows, n_cols) or just (n_cols,) if n_rows=1
        if n_rows == 1: axes = np.array([axes]) # ensure 2d-like
        if n_cols == 1: axes = np.array([axes]) # ensure 2d-like
        
        # Flatten for easy iteration
        axes_flat = np.atleast_1d(axes).flatten()
        
        for i, item in enumerate(items):
            ax = axes_flat[i]
            res = item["res"]
            
            # Plot data (NORMALIZED)
            # Check if y_obs is likely raw (e.g. median << 0.1 or >> 10) compared to model ~ 1
            # But we tried to store normalized data in loops.
            # TRANSFORM TO VELOCITY SPACE
            # v = c * (rest_wl - L0) / L0
            # We use c = 299792.458 km/s (approx 3e5)
            c_kms = 299792.458
            L0 = item["L0"]
            
            x_rest = item["x_rest"]
            x_vel = c_kms * (x_rest - L0) / L0
            
            # Plot data (NORMALIZED)
            ax.step(x_vel, item["y_obs"], color='k', where='mid', lw=1)
            
            # Simple Dynamic Y-limits
            # Filter to relevant velocity range for scaling (e.g. -25k to +25k)
            # This prevents wild scaling if spectrum extends far
            mx = (x_vel > -30000) & (x_vel < 30000)
            if np.any(mx):
                y_data = item["y_obs"][mx]
                ymin, ymax = np.nanmin(y_data), np.nanmax(y_data)
                # If model exists, include it in range
                if res["ok"] and "y_model" in res:
                    x_mod_vel = c_kms * (res["x_model"] - L0) / L0
                    mx_mod = (x_mod_vel > -30000) & (x_mod_vel < 30000)
                    if np.any(mx_mod):
                        ymin = min(ymin, np.nanmin(res["y_model"][mx_mod]))
                        ymax = max(ymax, np.nanmax(res["y_model"][mx_mod]))
                
                # Add margin
                rng = ymax - ymin
                if rng == 0: rng = 1.0
                ax.set_ylim(ymin - 0.2*rng, ymax + 0.3*rng)
            else:
                ax.set_ylim(-0.5, 2.0) # fallback
            
            # Plot fit if ok
            if res["ok"]:
                if "y_model" in res and "x_model" in res:
                     # Calculate model velocity
                     x_rest_mod = res["x_model"]
                     x_vel_mod = c_kms * (x_rest_mod - L0) / L0
                     
                     ax.plot(x_vel_mod, res["y_model"], color='r', lw=1.5, alpha=0.8)
                elif "fit_params" in res and res["fit_params"] is not None:
                     # Re-evaluate
                     params = res["fit_params"]
                     x_f = np.linspace(x_rest.min(), x_rest.max(), 200)
                     y_f = gauss_absorption_4p(x_f, *params)
                     # Vel
                     x_vf = c_kms * (x_f - L0) / L0
                     ax.plot(x_vf, y_f, color='r', lw=1.5, alpha=0.8)
            
            # Mark absorption minimum
            if res.get("ok", False) and "v_center_kms" in res:
                v_min = res["v_center_kms"]
                # v_center_kms is expansion velocity (>0). In doppler plot, absorption is at -v_exp
                ax.axvline(-v_min, color='b', ls='--', lw=1.0, alpha=0.9, zorder=10)
                # print(f"DEBUG: Plotting vline for {line_name} at {-v_min:.1f} km/s")
            else:
                pass
                # if res.get("ok"):
                #    print(f"DEBUG: MISSING v_center_kms for {line_name} (Keys: {list(res.keys())})")
                     
            # Label
            # phase = item["phase"]
            # To save space, maybe just put phase in corner or title
            ax.text(0.05, 0.85, f"+{item['phase']:.1f} d", transform=ax.transAxes, fontsize=9, 
                    bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))

        # Set labels on edges
        # Axes are flattened, but we used sharex/sharey
        # It's safest to set xlabel on bottom row axes and ylabel on left col axes
        
        # Actually, with sharex=True, we interact with big subplot container?
        # Or iterating axes:
        # Standard matplotlib shared axes: only bottom plots show x-labels
        
        # Let's verify which axes are at bottom
        # Or just use fig.text for common label
        
        fig.text(0.5, 0.04, "Velocity (km/s)", ha='center', va='center', fontsize=12)
        fig.text(0.06, 0.5, "Normalized Flux", ha='center', va='center', rotation='vertical', fontsize=12)
        
        # Set X limit to a standard velocity window
        # e.g. -20000 to +10000
        # Check standard expansion velocities ~ 10000-15000 km/s?
        # Let's use -25000 to +10000 ? or symmetric?
        # Usually P-Cygni are blue-shifted.
        # Let's do -30000 to +10000
        # axes[0,0].set_xlim(-8000, 8000)
        # axes[0,0].set_ylim(0.2, 2)
        for j in range(i+1, len(axes_flat)):
            axes_flat[j].axis('off')

        fig.suptitle(f"{line_name} Evolution", y=1.0)
        fig.tight_layout()
        
        out_name = f"mosaic_{line_name.replace(' ', '_')}.png"
        out_path = os.path.join(out_dir, out_name)
        fig.savefig(out_path)
        plt.close(fig)
        print(f"Saved mosaic: {out_path}")

def save_individual_fit_plots(x_rest, y_obs, res, L0, line_name, out_dir, f_name):
    """
    Saves a plot of the individual line fit.
    """
    import matplotlib.pyplot as plt
    os.makedirs(out_dir, exist_ok=True)
    
    fig, ax = plt.subplots()
    ax.step(x_rest, y_obs, color='k', where='mid', lw=1, label='Data')
    
    save_plot = True
    clean_fname = f_name.replace("/", "_").replace(".txt", "").replace(".ascii", "").replace(".dat", "")
    
    if not res["ok"]:
        # Plot data only, title with reason
        reason = res.get("reason", "unknown failure")
        ax.set_title(f"{line_name} - {f_name}\n(Fit Failed: {reason})", color='red', fontsize=10)
    else:
        params = res["fit_params"] # A, mu, sigma, C
        x_fit = res.get("x_fit")
        if x_fit is not None:
             x_f = np.linspace(x_fit.min(), x_fit.max(), 200)
             y_f = gauss_absorption_4p(x_f, *params)
             ax.plot(x_f, y_f, color='r', lw=1.5, label='Fit')
             ax.axvline(params[1], color='r', ls='--', alpha=0.5, label='Center')
        ax.set_title(f"{line_name} - {f_name}")

    out_path = os.path.join(out_dir, f"{clean_fname}_{line_name.replace(' ', '_')}.png")
    fig.savefig(out_path)
    plt.close(fig)

# =============================================================================
#                               SELECTION CSV MANAGER
# =============================================================================

def manage_velocity_selection(df_curr, csv_path):
    """
    Manages a persistent CSV file to Select/Deselect measurements using a MATRIX format.
    
    Rows: Spectral Lines
    Cols: Phases (rounded to 2 decimals)
    Values: 1 (Include) or 0 (Exclude)
    
    1. Loads existing Matrix CSV if present.
    2. Syncs with `df_curr`:
       - If (line, phase) is new, add col/row with default 1.
       - If (line, phase) exists, respect user's 0/1.
    3. Saves updated Matrix CSV.
    4. Returns `df_curr` filtered to rows where value is 1.
    """
    if df_curr is None or df_curr.empty:
        return df_curr
        
    if not {"line", "phase_days"}.issubset(df_curr.columns):
        print("[WARN] manage_velocity_selection: df missing 'line' or 'phase_days'.")
        return df_curr

    # 1. Create a stable phase string key for columns
    # We use 2 decimals to distinguish epochs but be readable.
    # Note: If multiple spectra fall into same bin (e.g. same night), they share the column.
    # This is acceptable for a selection UI.
    df_curr = df_curr.copy()
    df_curr["phase_str"] = df_curr["phase_days"].apply(lambda x: f"{float(x):.2f}")
    
    # 2. Load Existing Matrix
    existing_matrix = None
    if os.path.exists(csv_path):
        try:
            # Index is 'line'
            existing_matrix = pd.read_csv(csv_path, index_col=0)
            existing_matrix.index.name = "line"
            
            # Robust Column Parsing: Convert to floats rounded to 2 decimals
            # This handles "13.9" vs "13.90" string mismatch
            new_cols = []
            for c in existing_matrix.columns:
                try:
                    new_cols.append(round(float(c), 2))
                except ValueError:
                    new_cols.append(c) # Keep as is if not a number
            existing_matrix.columns = new_cols
            
        except Exception as e:
            print(f"[WARN] Could not read valid matrix from {csv_path}: {e}")
            existing_matrix = None

    if existing_matrix is None:
        existing_matrix = pd.DataFrame()

    # 3. Update Matrix with Current Data
    # (Section commented out/disabled for Read-Only, but logic conceptually relies on matching)
    
    updated_matrix = existing_matrix.copy()
    
    # 5. Filter df_curr
    # We use updated_matrix to decide.
    # Iterate rows of df_curr
    keep_mask = []
    
    filtered_count = 0
    total_count = len(df_curr)
    
    for idx, row in df_curr.iterrows():
        l_ = row["line"]
        
        # Get numerical phase key
        try:
            p_val = round(float(row["phase_days"]), 2)
        except:
            p_val = row["phase_str"] # Fallback
        
        # Check matrix
        if l_ in updated_matrix.index and p_val in updated_matrix.columns:
            val = updated_matrix.loc[l_, p_val]
            # Treat NaN as 0 (if somehow valid data maps to empty cell? shouldn't happen logic above fills it)
            # Treat 1 as True, 0 as False
            keep_mask.append(val == 1)
            if val != 1: filtered_count += 1
        else:
            # Should not happen if matrix was updated, but here we read-only.
            # If not in matrix, DEFAULT INCLUDE (True)
            keep_mask.append(True)
            
    df_curr["include"] = keep_mask
    
    # Return filtered
    df_out = df_curr[df_curr["include"] == True].copy()
    
    print(f"Selection Matrix applied. Kept {len(df_out)}/{total_count} points. (Excluded {filtered_count})")
    
    # Clean up
    del df_out["phase_str"]
    del df_out["include"]
    
    return df_out

# =============================================================================
#                                  MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    # ----------------- CONFIGURATION -----------------
    B_PEAK = 60572.18
    REDSHIFT = 0.0028
    EBV = 0.205
    
    import paths
    parent_fold_opt = str(paths.SPECTRA)
    parent_fold_ir = str(paths.SPECTRA)
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "outputs",
                           "velocities")
    out_dir = os.path.abspath(out_dir)
    SELECTION_CSV_READONLY = str(paths.SPECTRA / "velocity_selection.csv")   # hand-vetted line selection
    
    # Define Optical Lines
    OPTICAL_LINES = {
        "Si II 6355": 6355.0,
        "Fe II 5169": 5169.0,
        "Fe II 6247": 6247.0,
        "C II 6578":  6578.0,
        "C II 7231":  7231.0,
        "Ca II 8498": 8498.0,
        "Ca II 8542": 8542.0,
        "Ca II 8662": 8662.0,
    }
    
    # Define IR Lines
    COII_LINES = {
        "Co II 15759": 15759.0,
        "Co II 16064": 16064.0,
        "Co II 16361": 16361.0,
    }

    # Reconstruct META from explicit file list (based on spectra_plot.ipynb content)
    # The user didn't provide meta.csv, so we will hardcode the known good spectra paths found in the notebook log output or context.
    # From context: 
    # NTT/EFOSC2: AT2024vjm_2024-09-15_00-40-56_ESO-NTT_EFOSC2-NTT_TNS.dat
    # VLT/X-Shooter: TOO_StellarMerger_SCI_SLIT_FLUX_IDP_VIS.ascii
    # VLT/X-Shooter: TOO_StellarMerger_SCI_SLIT_FLUX_IDP_UVB.ascii
    # MUSE: SN2024vjm_MUSE_spectrum_2024-10-13.txt
    # and several others from ePESSTO.
    # To be safe and robust without full list, we will try to find *supported* files in the directory? 
    # Or better, we define the ones we saw in the logs.
    
    # Let's try to list files in Spectra/, parse properties, and build a meta dataframe dynamically
    # This is safer than hardcoding if files change.
    
    print("--- Searching for optical spectra ---")
    opt_files = []
    # Simple heuristic to find spectra files
    for root, dirs, files in os.walk(parent_fold_opt):
        for f in files:
            if f.endswith(".dat") or f.endswith(".ascii") or f.endswith(".txt"):
                if "IR" in root: continue # Skip IR folder for optical search
                if "vel_plots" in root: continue
                full_path = os.path.join(root, f)
                
                # Guess instrument and phase
                inst = "Unknown"
                if "EFOSC2" in f: inst = "NTT/ EFOSC2" 
                elif "X-Shooter" in f: inst = "VLT/ X-Shooter"
                elif "MUSE" in f: inst = "VLT/ MUSE"
                elif "FLOYDS" in f: inst = "FTS/ FLOYDS"
                
                # Try to extract date/MJD to calc phase. 
                # This is hard without the `meta.csv`.
                # HOWEVER, the user has `meta.csv` in `Spectra/`. 
                # Let's try to read it.
    
    meta_path = os.path.join(parent_fold_opt, "meta.csv")
    if os.path.exists(meta_path):
        try:
            # Explicitly set comma separator and quote char
            meta = pd.read_csv(meta_path, sep=',', quotechar='"', skipinitialspace=True, on_bad_lines='skip')
            print(f"Loaded meta.csv with {len(meta)} rows.")
            print("Meta head:")
            print(meta.head(2))
            print("Meta columns:", meta.columns)
        except Exception as e:
            print(f"[WARN] meta.csv read failed: {e}. Trying alternate load...")
            meta = pd.DataFrame()
    else:
        print("[WARN] meta.csv not found in Spectra/. Velocity plot will be empty.")
        meta = pd.DataFrame()
        


    # Manually add Gemini files to meta if not already present
    gemini_entries = [
        {"file_name": "SN2024vjm_20240916_gemini.txt", "instrument": "Gemini-S/GMOS", "utc_obs_date": "2024-09-17T00:00:00"},
        {"file_name": "SN2024vjm_20240921_gemini.txt", "instrument": "Gemini-S/GMOS", "utc_obs_date": "2024-09-22T00:00:00"},
    ]
    for entry in gemini_entries:
        if not meta.empty and "file_name" in meta.columns and entry["file_name"] in meta["file_name"].values:
            continue
        meta = pd.concat([meta, pd.DataFrame([entry])], ignore_index=True)

    # ----------------- OPTICAL FITTING -----------------
    print("\n--- Starting Optical Fitting ---")
    df_abs_rows = []
    all_fit_results = []
    
    if not meta.empty:
        # Sort by MJD
        if "mjd" in meta.columns:
            meta = meta.sort_values("mjd")
        
        for idx, row in meta.iterrows():
            f_name = row["file_name"]
            # skip IR files here just in case they are in meta
            if "NIR" in f_name or "IR" in f_name: 
                # Special case: X-Shooter NIR is often in meta, but we handle it in ir_spectra_def
                continue 
            
            inst = row["instrument"]
            
            # Helper to get MJD
            mjd = np.nan
            if "mjd" in row and pd.notna(row["mjd"]):
                mjd = float(row["mjd"])
            elif "utc_obs_date" in row and pd.notna(row["utc_obs_date"]):
                try:
                    # Parse ISO string
                    t = Time(row["utc_obs_date"], format='isot', scale='utc')
                    mjd = t.mjd
                except:
                    pass
            
            if np.isnan(mjd):
                 # Try filename if date is stuck in it? 
                 # e.g. AT2024vjm_2024-09-15...
                 pass

            phase = mjd - B_PEAK
            
            # Construct path (assuming flat or known struct, usually flat in Spectra/ or subfolder?)
            # The notebook used: path = f"{parent_fold.rstrip('/')}/{f}"
            spec_path = find_file_in_tree(parent_fold_opt, f_name)
            if not spec_path:
                # Try assuming f_name might have a path prefix that logic didn't account for, 
                # or just fail.
                # One last try: if f_name has folder parts, search for basename
                spec_path = find_file_in_tree(parent_fold_opt, os.path.basename(f_name))
                
            if not spec_path:
                print(f"[WARN] Could not find {f_name} in {parent_fold_opt}")
                continue
            
            # print(f"Found {f_name} at {spec_path}")
                
            x_rest, y = read_spec_as_restframe(spec_path, inst, f_name, REDSHIFT)
            if len(x_rest) == 0: 
                print(f"[WARN] Read 0 points from {spec_path}")
                continue
            
            # Apply Extinction Correction
            y = apply_extinction_correction(x_rest, y, EBV)
            
            # Identify Ca II Triplet coverage
            # We want to run the joint fit ONCE if coverage exists.
            has_ca_triplet = has_coverage(x_rest, 8580.0, window_A=300.0) # Approx center
            
            ca_triplet_done = False
            
            for line_name, L0 in OPTICAL_LINES.items():
                is_ca = "Ca II" in line_name
                
                if is_ca and has_ca_triplet:
                    if not ca_triplet_done:
                        # Run Joint Fit
                        print(f"  Fitting Ca II Triplet Jointly for {os.path.basename(f_name)}...")
                        res_ca = fit_caii_triplet_true_joint_rest(x_rest, y)
                        
                        if res_ca["ok"]:
                            # Save Red component to CSV
                            r = res_ca["red_results"]
                            df_abs_rows.append({
                                "line": "Ca II 8662",
                                "phase_days": phase,
                                "v_exp_kms": r["v_abs_kms"],    # Use Profile Min (Trough) as main velocity
                                "v_gaussian_kms": r["v_exp_kms"], # Store Gaussian center for reference
                                "v_err_kms": r["v_err_kms"],
                                "v_abs_kms": r["v_abs_kms"],
                                "chi2_red": r["chi2_red"],
                                "ok": True,
                                "instrument": inst,
                                "file": f_name
                            })
                            
                            # Add to mosaics - we add one entry per line so they appear in each panel?
                            # OR we add one entry for the whole triplet and the mosaic plotter handles it?
                            # The mosaic plotter 'save_fit_mosaics' iterates by line.
                            # So we should append 3 results, one for each line, but sharing the same model arrays.
                            
                            for c_line, c_L0 in [("Ca II 8498", 8498.02), ("Ca II 8542", 8542.09), ("Ca II 8662", 8662.14)]:
                                # Slice model/data around this specific line for the mosaic view
                                # Model from fit_caii...["y_model"] is originally scaled to match y_obs.
                                # User wants NORMALIZED plots. 
                                # So we must divide both y_obs and y_model by res_ca["norm"].
                                
                                norm_val = res_ca.get("norm", 1.0)
                                x_mod = res_ca["x_obs"]
                                y_mod_norm = res_ca["y_model"] / norm_val
                                y_obs_norm = res_ca["y_obs"] / norm_val
                                
                                mask_line = (x_mod >= c_L0 - 300) & (x_mod <= c_L0 + 300)
                                
                                if mask_line.sum() > 10:
                                    # Use trough velocity for 8662, keeping Gaussian/fit param for others if needed
                                    # (Default was using r['v_exp_kms'] for all before)
                                    v_cen = r["v_abs_kms"] if "8662" in c_line else r["v_exp_kms"]
                                    
                                    all_fit_results.append({
                                        "line": c_line,
                                        "phase": phase,
                                        "res": {
                                            "ok": True,
                                            "x_model": x_mod[mask_line],
                                            "y_model": y_mod_norm[mask_line],
                                            "fit_params": None,
                                            "v_center_kms": v_cen
                                        },
                                        "x_rest": x_mod[mask_line], 
                                        "y_obs": y_obs_norm[mask_line],
                                        "file": f_name,
                                        "L0": c_L0
                                    })
                        else:
                             print(f"  Ca II Joint Fit terminated: {res_ca.get('reason')}")
                        
                        ca_triplet_done = True
                    continue # Skip individual fit for Ca lines
                
                # Normal fit for non-Ca or if coverage fails (fallback?)
                # If coverage fails for full triplet, maybe we should try individual?
                # For now, let's stick to the plan: if no full coverage, skip or do individual?
                # The prompt implies we want the "TRUE joint fit".
                
                if has_coverage(x_rest, L0):
                    res = fit_absorption_only(x_rest, y, L0, window_A=300.0)
                    
                    if res["ok"]:
                        v = res["velocities"]
                        df_abs_rows.append({
                            "line": line_name,
                            "phase_days": phase,
                            "v_exp_kms": v["v_exp"],
                            "v_err_kms": v["v_err"],
                            "v_abs_kms": v["v_abs"],
                            "chi2_red": v["chi2_red"],
                            "ok": True,
                            "instrument": inst,
                            "file": f_name
                        })
                        
                        # Add top-level key for mosaic plotter
                        res["v_center_kms"] = v["v_exp"]
                        
                    # Collect for mosaic
                    all_fit_results.append({
                        "line": line_name,
                        "phase": phase,
                        "res": res,
                        "x_rest": res.get("x_fit", x_rest), # Use fit window x if available, else full
                        "y_obs": res.get("y_obs", y) if res["ok"] else y, # Slice if fit ok, else full
                        "file": f_name,
                        "L0": L0
                    })
                    
                    if not res["ok"]:
                        # If failed, we want to plot the region around L0
                        mask = (x_rest > L0 - 100) & (x_rest < L0 + 100)
                        if np.sum(mask) > 10:
                             all_fit_results[-1]["x_rest"] = x_rest[mask]
                             all_fit_results[-1]["y_obs"] = y[mask]
                        else:
                             all_fit_results.pop() # remove if no data in window

    
    df_abs = pd.DataFrame(df_abs_rows)
    # Save optical results
    if not df_abs.empty:
        out_csv_opt = os.path.join(out_dir, "optical_velocity_results.csv")
        os.makedirs(out_dir, exist_ok=True)
        df_abs.to_csv(out_csv_opt, index=False)
        print(f"Saved optical results to {out_csv_opt} ({len(df_abs)} rows)")
    else:
        print("No optical velocities derived.")

    # ----------------- IR FITTING -----------------
    print("\n--- Starting IR Fitting ---")
    # IR files are hardcoded in IR_spectra.ipynb, not in meta.csv likely.
    # IR files:
    # 'SN2024vjm_0928_merged_tellcor.txt', 'SN2024vjm_1010_merged_tellcor.txt', 'SN2024vjm_1021_merged_tellcor.txt'
    # 'TOO_StellarMerger_SCI_SLIT_FLUX_IDP_NIR.ascii'
    # 'SN2024vjm_JWST_NIRSpec_60585.083.txt'
    # Define IR Spectra to load
    # Update to clipped and fixed versions as requested
    ir_spectra_def = [
        # ORIGINAL mjd: 60581.0 / 60593.0 / 60604.0 (the last is ~1 d off the observation);
        # revision 3 uses the UTC start times of Supplementary Table 1.
        {"path": "SN2024vjm_0928_clipped_spectrum.txt", "mjd": Time("2024-09-28T00:06:35.427").mjd, "label": "SOAR 09/28"},
        {"path": "SN2024vjm_1010_clipped_spectrum.txt", "mjd": Time("2024-10-10T00:25:31.077").mjd, "label": "SOAR 10/10"},
        {"path": "SN2024vjm_1021_clipped_spectrum.txt", "mjd": Time("2024-10-22T01:01:26.277").mjd, "label": "SOAR 10/21"},
        {"path": "SN2024vjm_jwst_fixed_spectrum.txt", "mjd": 60585.08, "label": "JWST"},
    ]
    
    # Need to be robust about MJD guess if not precise, but phases in plot depend on it.
    # Notebook variable names: time_a, time_b, time_e.
    # Let's search IR folder for these text files.
    
    df_co_rows = []
    
    for item in ir_spectra_def:
        fname = item["path"]
        # Resolve path relative to parent_fold_ir
        # Some had '..' which means parent_fold_opt
        if fname.startswith("../"):
            full_path = os.path.join(parent_fold_opt, fname[3:])
        else:
            full_path = os.path.join(parent_fold_ir, fname)
            
        if not os.path.exists(full_path):
            print(f"[WARN] IR file not found: {full_path}")
            continue
            
        # Read
        try:
            # Most are ascii
            spec = ascii.read(full_path)
            if len(spec.colnames) < 2: continue
            
            # X-Shooter logic inside read_spec? No, these are simple txts usually
            # But read_spec_as_restframe can handle generic 2-col
            # Note: The IR notebook manually read them as:
            # a = ascii.read(...) then x = a['col1'], y=a['col2']...
            
    # JWST is likely microns?
            # SOAR txts: wavelength usually Angstroms.
            # Use our helper for consistency
            x_rest, y = read_spec_as_restframe(full_path, "IR_GENERIC", fname, REDSHIFT)
            if len(x_rest) == 0: continue
            
            # JWST/NIRSpec data might be in different units (microns)
            # ir_spectra.ipynb: JWST was multiplied by 10000.
            if "JWST" in fname:
                if np.nanmax(x_rest) < 100.0: # likely microns
                     x_rest *= 10000.0
            
            # Apply Extinction Correction
            y = apply_extinction_correction(x_rest, y, EBV)

            # IR FIT SWITCH - Defined generally, or here?
            # It should be defined once, but here is fine if indented.
            USE_IR_JOINT_FIT = True 
            
            # ... in loop ...
            # Check coverage
            if len(x_rest) > 0:
                # Decide fitting mode
                if USE_IR_JOINT_FIT:
                     # Joint fit
                     res = fit_three_coii_lines_joint(x_rest, y, COII_LINES)
                     results_list = [res]
                else:
                     # Individual fits
                     results_list = []
                     for co_name, co_L0 in COII_LINES.items():
                         # Create single-item dict for the fitting function
                         single_line_dict = {co_name: co_L0}
                         res_single = fit_three_coii_lines_joint(x_rest, y, single_line_dict)
                         results_list.append(res_single)
                
                for res in results_list:
                    if res["ok"]:
                        for row in res["rows"]:
                            # Append metadata
                            row["phase_days"] = item["mjd"] - B_PEAK
                            row["spectrum"] = item["label"]
                            df_co_rows.append(row)
                            
                            # Add to mosaic list
                            # The Co II fit results (res) already contain NORMALIZED y_obs and y_model
                            # because fit_three_coii_lines_joint performs fitting on normalized segments.
                            # res["rows"] contains per-line normalized segments ("xx", "yy_norm") in "x_fit" and "y_norm".
                            # res["y_model"] is the model evaluated on the concatenated normalized x-axis.
                            
                            x_line = row["x_fit"]
                            y_line_norm = row["y_norm"]
                            line_L0 = row["L0_A"]

                            # To get the model corresponding to this line, we can just filter the full model 
                            # by the X-range of this line.
                            
                            x_all_sorted = res["x_obs"]
                            y_model_all = res["y_model"] # This is normalized
                            
                            min_x, max_x = x_line.min(), x_line.max()
                            mask_model = (x_all_sorted >= min_x) & (x_all_sorted <= max_x)
                            
                            all_fit_results.append({
                                "line": row["line"],
                                "phase": row["phase_days"],
                                "res": {
                                    "ok": True,
                                    "x_model": x_all_sorted[mask_model],
                                    "y_model": y_model_all[mask_model],
                                    "fit_params": None,
                                    "v_center_kms": row["v_center_kms"] # Inject for marker
                                },
                                "x_rest": x_line,
                                "y_obs": row["y_norm"], # Normalized
                                "file": fname,
                                "L0": line_L0
                            })

        except Exception as e:
            print(f"[WARN] Failed IR processing {fname}: {e}")

    df_co = pd.DataFrame(df_co_rows)
    if not df_co.empty:
        out_csv_ir = os.path.join(out_dir, "ir_velocity_results.csv")
        df_co.to_csv(out_csv_ir, index=False)
        print(f"Saved IR results to {out_csv_ir} ({len(df_co)} rows)")
    else:
        print("No IR velocities derived.")

    # Save Mosaics (Optical + IR)
    mosaic_dir = os.path.join(out_dir, "mosaics")
    save_fit_mosaics(all_fit_results, mosaic_dir)

    # ----------------- PLOTTING -----------------
    print("\n--- Generating Combined Plot ---")
    
    # --- Velocity Selection Manager (ALL LINES) ---
    # Combined Plot
    # Velocity Selection Manager (OPTICAL ONLY)
    # User requested: Co II lines should not care about velocity_selection.csv
    
    # 1. Vet Optical Lines
    if not df_abs.empty:
        selection_csv = SELECTION_CSV_READONLY   # read only (manage_velocity_selection never writes)
        df_abs_filtered = manage_velocity_selection(df_abs, selection_csv)
        df_abs_plot = df_abs_filtered.copy()
    else:
        df_abs_plot = pd.DataFrame()

    # 2. IR Lines (Always Keep)
    df_co_plot = df_co.copy()

    # ---- re-reference phases to B max = 60572.29 (after the selection, whose keys use 60572.18) ----
    dphase = B_PEAK - B_PEAK_NEW
    for _d in (df_abs_plot, df_co_plot):
        if not _d.empty:
            _d["phase_days"] = _d["phase_days"] + dphase
    for _it in all_fit_results:
        _it["phase"] = _it["phase"] + dphase
    print(f"Phases shifted by {dphase:+.3f} d (B max {B_PEAK} -> {B_PEAK_NEW})")

    # Plot
    plot_velocities_combined(
        df_abs_plot,
        out_dir=out_dir,
        add_df_co=df_co_plot,
        plot_ca="single",
        ca_rep="Ca II 8662"
    )

    # 3. Create Master Mosaic of Approved Fits
    print("--- Generating Master Fit Mosaic ---")
    if not df_abs_plot.empty or not df_co_plot.empty:
        # Combine the approved dataframes
        df_allowed = pd.concat([df_abs_plot, df_co_plot], ignore_index=True)
        # Create figure
        create_master_mosaic_figure(all_fit_results, df_allowed, out_dir=out_dir)
    else:
        print("No approved points to plot in mosaic.")

