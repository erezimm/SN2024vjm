import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
import paths
from astropy.io import ascii

# ==========================================
# 1. GLOBAL CONFIGURATION
# ==========================================
# Change this to 'near' or 'far' to toggle distance scales
SELECTED_SCALE = 'far' 

# Bolometric Correction for the i-band (M_bol - M_i)
# Standard for Iax is ~0.3-0.5 depending on temperature
BC_I = 0.

T0_LEAK = 83.15  # [rev3] Katz-integral t0 re-derived with explosion MJD 60565.58 (was 83.05)

SCALES = {
    'near': {
        'dist_pc': 7.01e6,
        'dm': 29.23,
        'mn56': 2.031e-3,
        'file': 'bolometric_lightcurve_nearby.txt',
        'label': '7.01 Mpc (Near)',
        'color': 'royalblue'
    },
    # 'far': {
    #     'dist_pc': 8.83e6,
    #     'dm': 29.73,
    #     'mn56': 3.223e-3,
    #     'file': 'bolometric_lightcurve_faraway.txt',
    #     'label': '8.83 Mpc (Far)',
    #     'color': 'darkorange'
    # }
        'far': {
        'dist_pc': 9.4e6,
        'dm': 29.86,
        'mn56': 3.705e-3,   # [rev3] tail fit with explosion MJD 60565.58 (was 3.671e-3)
        'file': str(paths.OUTPUTS / 'bolometric' / 'bolometric_lightcurve_faraway_60565.58.txt'),   # [rev3] bolometric LC rebuilt with explosion MJD 60565.58
        'label': '8.83 Mpc (Far)',
        'color': 'darkorange'
    }
}

# Physical/Temporal Constants
Z = 0.0028
T0_BMAX = 60572.29   # [rev3] GP B-band peak from lc_clean.py (= Table 1); was 60572.18
T0_BMAX_CSV = 60572.18   # reference used when coii_jointfit_results.csv phases were written
EXPLODE_T = 60565.58   # [rev3] power-law fit (explosion_time.ipynb); was 60565.89
PATH_SPECTRA = paths.SPECTRA

# Using the clipped files and JWST path from your latest prompt
SPECTRA_FILES = {
    'SOAR (Early)':  PATH_SPECTRA / "SN2024vjm_0928_clipped_spectrum.txt",
    'JWST':          PATH_SPECTRA / "SN2024vjm_jwst_fixed_spectrum.txt", 
    'SOAR (Mid)':    PATH_SPECTRA / "SN2024vjm_1010_clipped_spectrum.txt",
    'SOAR (Late)':   PATH_SPECTRA / "SN2024vjm_1021_clipped_spectrum.txt"
}

# Rest wavelengths for Co II
COII_LINES = {r"Co II 1.57 $\mu$m": 15759.0, r"Co II 1.61 $\mu$m": 16064.0, r"Co II 1.64 $\mu$m": 16361.0}

# Styling
import nature_style as ns   # [rev3] Nature artwork style (Arial 5-7 pt, TrueType, 180 mm)
ns.apply()

# Filter Paths and Config
PATH_FILTERS = paths.FILTERS
FILTER_MAP = {
    'i': 'SDSS_i.txt',
    'bg_i': 'i-bg.txt'
}

# ==========================================
# 2. DATA PROCESSING MODULES
# ==========================================

def get_ni_decay_model(t_range, m_ni, t0_leak=69.5):   # [rev3] tail-fit t0 (was 69.7)
    """Radioactive deposition model (erg/s)."""
    trap = 1.0 - np.exp(-(t0_leak/t_range)**2)
    q_gamma = m_ni * (6.45*np.exp(-t_range/8.76) + 1.38*np.exp(-t_range/111.4)) * trap
    q_pos = m_ni * 4.64*(np.exp(-t_range/111.4) - np.exp(-t_range/8.76))
    return q_gamma*1e43 + q_pos*1e41

def Q_instantaneous(t, t0):
    """Energy deposition rate Q(t, t0) without Ni-mass normalization."""
    # This matches the 'Q(t, t0)' function in your original code
    trap = 1.0 - np.exp(-(t0 /t)**2)
    Q_gamma = (6.54 * np.exp(-t / 8.76) + 1.38 * np.exp(-t / 111.4)) * trap
    Q_pos = 4.64 * (np.exp(-t / 111.4) - np.exp(-t / 8.76))
    return Q_gamma * 1e43 + Q_pos * 1e41

def load_filter_response(filter_name):
    """Loads filter response (Wavelength, Transmission)."""
    fname = FILTER_MAP.get(filter_name)
    if not fname:
        return None, None
    fpath = PATH_FILTERS / fname
    try:
        # Try default (whitespace)
        data = np.loadtxt(fpath)
    except ValueError:
        try:
            # Try comma
            data = np.loadtxt(fpath, delimiter=',')
        except Exception as e:
            print(f"Error loading filter {filter_name} from {fpath}: {e}")
            return None, None
    except Exception as e:
        print(f"Error loading filter {filter_name} from {fpath}: {e}")
        return None, None
        
    return data[:, 0], data[:, 1] # Wave, Trans

def calculate_band_flux(mag, mag_err, filter_name):
    """
    Calculates integrated flux (erg/s/cm^2) from AB magnitude and filter response.
    Assumes AB source (f_nu = 3631 Jy) for zero-point calibration.
    F_band = f_nu_source * c * Integral(T(lam)/lam^2 dlam)
    """
    lam, trans = load_filter_response(filter_name)
    if lam is None:
        return np.nan, np.nan
    
    # Constants
    c_ang = 2.99792458e18  # c in Angstrom/s
    f_nu_0 = 3631e-23      # 3631 Jy in erg/s/cm^2/Hz
    
    # Flux Density of the source at the filter
    # f_nu_source = f_nu_0 * 10**(-0.4 * mag)
    f_nu_source = f_nu_0 * 10**(-0.4 * mag)
    
    # Integral part: Integral(T(lam)/lam^2 dlam)
    # This represents the effective bandwidth in frequency space / c roughly
    integral = np.trapz(trans / lam**2, x=lam)
    
    # Total integrated flux
    flux = f_nu_source * c_ang * integral
    
    # Error prop: dF/F = 0.4 * ln(10) * dm
    flux_err = flux * (0.4 * np.log(10)) * mag_err
    
    return flux, flux_err

def i_band_to_flux_points(i_df, cfg):
    """Converts i-band magnitudes to Luminosity (erg/s) using filter integration."""
    d = i_df.copy()
    d["phase"] = d["time"] - T0_BMAX
    # Filter for nebular phase (>60d? User said 'tail' but kept >100 in old code. Keeping >60 to match previous task logic or keep 100?)
    # The previous task modified fit_bc_ni_tail.py to >60. Thermo_figure.py uses >100.
    # User didn't specify phase range change, so I'll stick to >100 or relax it if needed.
    # Let's keep >80 to be safe and inclusive? Original was >100.
    d = d[np.isfinite(d["phase"]) & (d["phase"] >= 80.0)].copy()
    
    if d.empty: return pd.DataFrame()

    # Use mag_extcorr if available, else magnitude
    mag_col = "mag_extcorr" if "mag_extcorr" in d.columns else "magnitude"
    
    # Calculate Flux per point
    lums = []
    lum_errs = []
    
    dist_cm = cfg['dist_pc'] * 3.086e18
    
    for idx, row in d.iterrows():
        filt = row['filter']
        mag = row[mag_col]
        # For error, assume the mag error is the dominant uncertainty? 
        # Using row['error'] directly.
        err = row['error']
        
        flux, f_err = calculate_band_flux(mag, err, filt)
        
        # Luminosity = 4 * pi * D^2 * Flux
        L = 4 * np.pi * dist_cm**2 * flux
        L_err = 4 * np.pi * dist_cm**2 * f_err
        
        lums.append(L)
        lum_errs.append(L_err)
        
    d["L_i"] = lums
    d["L_i_err"] = lum_errs
    
    # Remove failed calculations
    d = d.dropna(subset=["L_i"])
    
    return d[["phase", "L_i", "L_i_err"]]

def load_processed_spectrum(label, path):
    """Loads spectra, explicitly converting Path objects to strings for astropy."""
    # Convert PosixPath to string to fix the 'find' AttributeError
    path_str = str(path)
    dat = ascii.read(path_str)
    return dat['WAVE'] / (1 + Z), dat['FLUX']

def pcygni_model(x, lam0, a0, a1, A_em, mu_em, sig_em, A_abs, mu_abs, sig_abs):
    """
    Standard P-Cygni profile: Linear continuum + Gaussian emission + Gaussian absorption.
    All parameters (A, mu, sig) are derived from your joint-fit CSV.
    """
    continuum = a0 + a1 * (x - lam0)
    emission = A_em * np.exp(-(x - mu_em)**2 / (2 * sig_em**2))
    absorption = A_abs * np.exp(-(x - mu_abs)**2 / (2 * sig_abs**2))
    
    total = continuum + emission - absorption
    return total, continuum, emission, absorption
# ==========================================
# 3. PLOTTING MODULES
# ==========================================

def plot_bolometric_panel(ax, bol_data, i_pts, cfg, bc_val):
    """Populates Panel A with Bolometric data + anchors."""
    if bol_data is None: return
    
    phases = bol_data[:, 0] - T0_BMAX
    L_bol, L_err = bol_data[:, 1], bol_data[:, 2]
    
    # 1. Main GP Curve
    l_bol, = ax.plot(phases, L_bol, lw=1.1, color='black')
    f_bol = ax.fill_between(phases, L_bol - L_err, L_bol + L_err, color='black', alpha=0.15, lw=0)
    
    # 2. Ni-56 Decay Model
    t_model = np.linspace(40, 350, 500)
    L_ni = get_ni_decay_model(t_model, cfg['mn56'])
    l_ni, = ax.plot(t_model + EXPLODE_T - T0_BMAX, L_ni, lw=0.8, ls='--', color='firebrick')
    L_ni_up = get_ni_decay_model(t_model, cfg['mn56'], 69.5 + 12.5)
    L_ni_down = get_ni_decay_model(t_model, cfg['mn56'], 69.5 - 12.5)
    f_ni = ax.fill_between(t_model + EXPLODE_T - T0_BMAX, L_ni_up, L_ni_down, color='firebrick',
                           alpha=0.2, lw=0)
    
    # 3. i-band Nebular Anchors
    # [rev3] re-enabled: these points are in the published Fig. 3 and described in its caption
    h_i = None
    if not i_pts.empty:
        h_i = ax.errorbar(i_pts["phase"], i_pts["L_i"]/0.06, yerr=i_pts["L_i_err"]/0.06,
                          fmt="o", ms=3.2, mfc='crimson', mec='black', mew=0.4,
                          ecolor='crimson', elinewidth=0.6, zorder=10)
        print(i_pts.assign(L=i_pts["L_i"]/0.06, Lerr=i_pts["L_i_err"]/0.06).to_string())

    h_jwst = ax.errorbar([60766.3396-T0_BMAX],[1.45e39],yerr=[0],color='orange',fmt="*", ms=6.5,
                         mfc='orange', mec='black', mew=0.5, ecolor='orange', zorder=11)
    print(60766.3396-T0_BMAX)

    
    # Cosmetics
    ax.set_yscale('log')
    ax.set_ylim(1e38, 8e40)
    ax.set_xlim(-15, 350)
    ax.set_ylabel(r'Luminosity (erg s$^{-1}$)')
    ax.set_xlabel('Days from $B$-band maximum')
    # [rev3] key for every line, symbol and shaded band
    handles = [(f_bol, l_bol), (f_ni, l_ni)]
    labels = ['Bolometric', r'$^{56}$Ni deposition']
    if h_i is not None:
        handles.append(h_i); labels.append(r'Corrected $i$-band')
    handles.append(h_jwst); labels.append(r'$JWST$')
    ax.legend(handles, labels, loc='upper right', fontsize=ns.FS_SMALL)
    ax.text(-0.02, 1.01, 'a', transform=ax.transAxes,
                fontsize=ns.FS_PANEL, fontweight='bold', va='bottom', ha='right')
    ax.grid(True, which='both', axis='y', alpha=0.05)

def plot_katz_panel(ax, bol_data, cfg):
    """
    Panel C: Katz Integral Convergence using the 'Minimized' Ratio Method.
    This plot is independent of Ni mass and Distance.
    """
    t_exp = bol_data[:, 0] - EXPLODE_T
    L_vals = bol_data[:, 1]
    
    # Sort for consistent integration
    order = np.argsort(t_exp)
    t, L = t_exp[order], L_vals[order]
    
    # Use reliable range up to ~57 days (as in your snippet)
    mask = (t > 0) & (t <= 57)
    t_fit, L_fit = t[mask], L[mask]
    
    # 1. Compute LHS (Data): L(t) / Integral(L * t dt)
    # We compute the denominator as a running integral
    denom_L = np.array([np.trapz(L_fit[:i+1] * t_fit[:i+1], x=t_fit[:i+1]) 
                        for i in range(len(t_fit))])
    lhs_data = L_fit / np.maximum(denom_L, 1e-10)
    
    # 2. Compute RHS (Model): Q(t, t0) / Integral(Q * t dt)
    Q_vals = Q_instantaneous(t_fit, T0_LEAK)
    denom_Q = np.array([np.trapz(Q_vals[:i+1] * t_fit[:i+1], x=t_fit[:i+1]) 
                        for i in range(len(t_fit))])
    rhs_model = Q_vals / np.maximum(denom_Q, 1e-10)
    
    # 3. Plotting the convergence of the ratios
    # Solid blue line is your measured data ratio
    ax.plot(t_fit[3:], lhs_data[3:], lw=2.5, color='royalblue', label='Normalized Luminosity', zorder=2)
    # Red dashed line is the model ratio (independent of M_ni)
    ax.plot(t_fit[3:], rhs_model[3:], lw=2.5, color='firebrick', ls='--', 
            label=r'Normalized $^{56}$Ni Deposition', zorder=3)
    
    # Add the formal equation from your image to the plot
    eq = r'$\frac{L(t)}{\int_0^t L(t^{\prime})t^{\prime}dt^{\prime}} = \frac{Q(t)}{\int_0^t Q(t^{\prime}, t_0)t^{\prime}dt^{\prime}}$'
    ax.text(0.8, 0.9, eq, transform=ax.transAxes, fontsize=14, 
            ha='center', va='center', bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))

    # Aesthetics
    ax.set_xlabel('Days since explosion', fontsize=11, fontweight='bold')
    ax.set_ylabel('Ratio [day$^{-2}$]', fontsize=11)
    ax.set_yscale('log') # Ratios vary significantly; log scale shows convergence better
    ax.set_ylim(1e-4, 1e-1) # Physical range for typical Katz ratios
    ax.legend(frameon=False, loc='lower left', fontsize='small')
    ax.text(0.05, 0.92, 'c', transform=ax.transAxes, fontsize=20, fontweight='bold')
    ax.grid(True, which='both', alpha=0.1)

def plot_katz_panel(ax, bol_data, cfg):
    """
    Panel C: Normalized Energy Deposition Ratios.
    Normalized by the derived M_ni to show convergence to 1.0.
    """
    t_exp = bol_data[:, 0] - EXPLODE_T
    L_vals = bol_data[:, 1]
    
    order = np.argsort(t_exp)
    t, L = t_exp[order], L_vals[order]
    
    mask = (t > 0) & (t <= 57)
    t_fit, L_fit = t[mask], L[mask]
    
    # 1. Calculate Integrals
    int_L = np.array([np.trapz(L_fit[:i+1] * t_fit[:i+1], x=t_fit[:i+1]) 
                      for i in range(len(t_fit))])
    
    Q_vals = Q_instantaneous(t_fit, T0_LEAK)
    int_Q = np.array([np.trapz(Q_vals[:i+1] * t_fit[:i+1], x=t_fit[:i+1]) 
                      for i in range(len(t_fit))])
    
    # 2. Derive Nickel Mass from the last reliable point (Katz Method)
    # M_ni is the ratio of the integrals at the end of the diffusion phase
    derived_m_ni = int_L[-1] / int_Q[-1]

    # Print the simple (unweighted) integral mismatch up to 57 days
    # Compare \int L dt to \int (M_Ni * Q) dt over the same window.
    print(len(t_fit))
    dE_0_57 = np.trapz(derived_m_ni * Q_vals-L_fit, x=t_fit)*86400
    print(f"\u222b_0^{t_fit[-1]:.0f} L dt - \u222b_0^{t_fit[-1]:.0f} (M_Ni Q) dt = {dE_0_57:.6e} erg")
    print(f"{np.sqrt(dE_0_57/0.2e33)/1e8} ")
    
    # plt.figure()
    # plt.plot(t_fit, L_fit, label='L(t)')
    # plt.plot(t_fit, derived_m_ni * Q_vals, label=r'$M_{\rm Ni} Q(t)$')
    # plt.legend()
    # plt.show()

    # 3. Normalized Ratios
    # Now we compare L to (M_ni * Q)
    instant_ratio = L_fit / (derived_m_ni * np.maximum(Q_vals, 1e-10))
    integrated_ratio = int_L / (derived_m_ni * np.maximum(int_Q, 1e-10))

    # --- Plotting ---
    ax.plot(t_fit[3:], instant_ratio[3:], lw=1.1, color='royalblue',
            label=r'$L(t)/Q(t, t_0)$', zorder=3)
    
    katz_eq_label = 'Katz Integral'#r'$\int_0^t L t^{\prime} dt^{\prime} / \int_0^t M_{\rm Ni} Q t^{\prime} dt^{\prime}$'
    ax.plot(t_fit[3:], integrated_ratio[3:], lw=1.1, color='firebrick',
            label='Katz integral', zorder=2)
    
    # Unity reference
    ax.axhline(1.0, color='black', lw=0.6, ls='-', alpha=0.3, zorder=1)

    # Aesthetics
    ax.set_xlabel('Days since explosion')
    ax.set_ylabel(r'Luminosity / $^{56}$Ni deposition')
    ax.set_ylim(0.5, 1.3)
    ax.legend(loc='lower right', fontsize=ns.FS_SMALL)
    ax.text(-0.02, 1.01, 'c', transform=ax.transAxes, fontsize=ns.FS_PANEL, fontweight='bold',
            va='bottom', ha='right')

def plot_nir_fit_panel(gs_parent, fig):
    """
    Panel B: Multi-epoch Co II triplet mosaic in Velocity Space.
    - Added Phase (days from B-max) to y-axis labels.
    - Shared velocity label per row (no decimals, no color).
    - Fixed y-axis scale (0.4 to 2.0) and crimson velocity markers.
    - Legend now includes the blue photospheric velocity center marker.
    """
    LABEL_MAP = {
        'SOAR (Early)': 'SOAR a',
        'JWST':         'JWST/NIRSpec',
        'SOAR (Mid)':   'SOAR b',
        'SOAR (Late)':  'SOAR e'
    }

    results_path = paths.SPECTRA / "IR" / "coii_pcygni_jointfit" / "coii_jointfit_results.csv"   # from notebooks/NIR_spectra_CoII_fits.ipynb
    df_fit = pd.read_csv(results_path)
    df_fit['spectrum'] = df_fit['spectrum'].str.strip()
    df_fit['line'] = df_fit['line'].str.strip()
    
    gs_sub = gs_parent.subgridspec(4, 3, hspace=0.62, wspace=0.12)
    C_KMS = 299792.458

    # List to store handles for the legend
    legend_handles = []

    for row, label in enumerate(SPECTRA_FILES.keys()):
        wl_rest, flux = load_processed_spectrum(label, SPECTRA_FILES[label])
        
        row_velocity = None
        row_phase = None
        
        for col, (line_name, L0) in enumerate(COII_LINES.items()):
            ax = fig.add_subplot(gs_sub[row, col])
            
            csv_spec = LABEL_MAP[label]
            csv_line = f"Co II {int(L0)}"
            
            match = df_fit[(df_fit["spectrum"].str.contains(csv_spec, regex=False)) & 
                           (df_fit["line"] == csv_line)]
            
            if match.empty: continue
            r = match.iloc[0]
            
            if row_velocity is None:
                row_velocity = int(r['v_center_kms'])
                row_phase = r['phase_days'] - (T0_BMAX - T0_BMAX_CSV)   # [rev3] re-reference to 60572.29

            v_kms = C_KMS * (wl_rest - L0) / L0
            mask = (v_kms > -4500) & (v_kms < 4500)
            if not np.any(mask): continue
            
            model_norm = r['a0']
            data_norm = np.median(flux[mask])

            total, _, _, _ = pcygni_model(
                wl_rest[mask], L0, r['a0'], r['a1'], 
                r['A_em'], r['mu_em_A'], r['sig_em_A'], 
                r['A_abs'], r['mu_abs_A'], r['sig_abs_A']
            )

            # 3. Plotting Data and Fit
            l1 = ax.step(v_kms[mask], flux[mask]/data_norm, color='gray', lw=0.5, where='mid', alpha=0.6)
            l2 = ax.plot(v_kms[mask], total/model_norm, color='firebrick', ls='--', lw=0.8, zorder=10)
            
            # 4. Velocity Marker
            l3 = ax.axvline(-row_velocity, color='royalblue', ls='--', lw=0.6, alpha=0.9, zorder=5)
            
            # Capture all handles once for the legend
            l0 = ax.axvline(0, color='gray', lw=0.4, ls=':', zorder=2)
            if not legend_handles:
                legend_handles = [l1[0], l2[0], l3, l0]

            # 5. Labels and Annotation
            if col == 0:
                ax.text(0.0, 1.07, rf"$v$ = {row_velocity:,} km s$^{{-1}}$", transform=ax.transAxes,
                        fontsize=ns.FS_TINY)
                inst_name = label.split()[0]
                ax.set_ylabel(f"{inst_name}\n+{row_phase:.1f} d", fontsize=ns.FS_TINY)

            # 6. Formatting
            ax.set_xlim(-3000, 3000)
            ax.set_ylim(0.4, 2.0)
            ax.set_yticks([]); ax.set_xticks([])
            
            if row == 0: 
                ax.set_title(line_name, fontsize=ns.FS_SMALL, pad=11)
            if row == 3:
                ax.set_xticks([-2000, 0, 2000])
                ax.set_xticklabels(['$-$2', '0', '2'], fontsize=ns.FS_TINY)
                if col == 1:
                    ax.set_xlabel(r'Velocity (10$^3$ km s$^{-1}$)', fontsize=ns.FS)

            if row == 0 and col == 0:
                ax.text(-0.3, 1.2, 'b', transform=ax.transAxes,
                        fontsize=ns.FS_PANEL, fontweight='bold', va='bottom', ha='right')

    # --- UPDATED LEGEND WITH BLUE VELOCITY MARKER ---
    fig.legend(legend_handles, ['Data', 'Fit', r'$v_{\rm phot}$', r'$v = 0$'],
               loc='upper center',
               bbox_to_anchor=(0.745, 0.487),
               ncol=4, fontsize=ns.FS_SMALL, frameon=False,
               handlelength=1.5, columnspacing=1.0)
# ==========================================
# 4. MAIN ASSEMBLY
# ==========================================

def create_figure():
    """
    Assembles the multi-panel Nature figure:
    A: Bolometric energetics with late-time radioactive anchors.
    B: NIR Co II P-Cygni fits (4 epochs) showing expansion kinematics.
    C: Katz energy conservation ratio proving first-principles consistency.
    """
    cfg = SCALES[SELECTED_SCALE]
    
    # 1. Load Data Streams
    try:
        bol_data = np.loadtxt(cfg['file'])
    except Exception as e:
        print(f"Error: {cfg['file']} not found ({e}).")
        bol_data = None

    try:
        lc_df = pd.read_csv(paths.PHOTOMETRY / 'lightcurve.csv')
        late_i = lc_df[lc_df["filter"].isin(["i", "bg_i"])].copy()
        i_pts = i_band_to_flux_points(late_i, cfg)
    except Exception as e:
        print(f"Photometry anchor processing failed: {e}")
        i_pts = pd.DataFrame()

    # 2. Layout Setup (Publication Dimensions)
    fig = plt.figure(figsize=(ns.DOUBLE_COL, 0.69 * ns.DOUBLE_COL))   # [rev3] 180 mm wide
    # Left column is wider for the main lightcurve; right column contains spectral/integral panels
    gs = gridspec.GridSpec(2, 2, width_ratios=[1.1, 0.9], height_ratios=[1.0, 0.85], hspace=0.42, wspace=0.2)
    
    ax_bol = fig.add_subplot(gs[:, 0])   # Panel A (Full height)
    gs_nir = gs[0, 1]                    # Panel B (Top right grid)
    ax_katz = fig.add_subplot(gs[1, 1])  # Panel C (Bottom right)
    
    # 3. Populate Panel A: Bolometric Evolution
    if bol_data is not None:
        plot_bolometric_panel(ax_bol, bol_data, i_pts, cfg, bc_val=BC_I)
    
    # 4. Populate Panel B: NIR Co II Mosaic
    # This function uses the subgridspec (4 rows x 3 lines) internally
    plot_nir_fit_panel(gs_nir, fig)

    # 5. Populate Panel C: Katz Integral Convergence
    if bol_data is not None:
        plot_katz_panel(ax_katz, bol_data, cfg)
    else:
        ax_katz.text(-0.15, 1.05, 'c', transform=ax_katz.transAxes, 
                fontsize=24, fontweight='bold', va='bottom', ha='right')
        ax_katz.text(0.5, 0.5, 'Missing Bolometric Data', 
                     ha='center', va='center', transform=ax_katz.transAxes)

    # 6. Final Save & Export
    ns.save(fig, 'Fig3_thermonuclear')   # -> outputs/figures

if __name__ == "__main__":
    create_figure()