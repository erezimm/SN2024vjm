import numpy as np
import matplotlib.pyplot as plt
import glob
import pandas as pd
from astropy import units as u
from astropy.constants import L_sun, sigma_sb, c, h, k_B
import os
from mpl_toolkits.axes_grid1 import make_axes_locatable
import pickle
import nature_style as ns    # [rev3] Nature artwork style (Arial 5-7 pt, TrueType, 180 mm)
ns.apply()

# [rev3] input data moved since this script last ran; they are only READ here
import paths
SM_DIR = str(paths.PROGENITORS)   # WR_pop.csv, hot-subdwarf+WD binaries
CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "outputs", "cache", "bpass")
os.makedirs(CACHE_DIR, exist_ok=True)

# ==========================================
# 0. CONFIGURATION
# ==========================================
from extinction import ccm89

H_RICH_MASSES = [2, 3, 4, 6, 8]
HE_RICH_MASSES = [2, 3, 4, 6, 8]

# Color Normalization Limits (vmin, vmax)
H_RICH_NORM_LIMITS = (2, 9)
HE_RICH_NORM_LIMITS = (2, 8)

# CO Star Filtering Constants
MAX_H_SURF = 0.001  # Maximum Hydrogen surface abundance (CO Star)
MAX_HE_SURF = 1 # Maximum Helium surface abundance (CO Star)
MAX_HE_CORE_MASS = 1 # Maximum Helium Core Mass (CO Star)
MIN_MASS_LIMIT = 8.0 # Minimum initial mass for endpoints

MAX_MASS_LIMIT = 35.0 # Maximum initial mass for endpoints

# Extinction Constants (from lc_clean.py)
EBV_MW = 0.043
RV_MW = 3.1
EBV_HOST = 0.205
RV_HOST = 3.1

VIS_FILTER_PATH = str(paths.FILTERS / "Euclid_VIS.vis.dat")
V_FILTER_PATH = str(paths.FILTERS / "johnson_v.txt")

S1_MAG = 25.83
S1_MAG_ERR = 0.13
DM = 29.86
DM_ERR = 0.1

FAINT_LIM = 26.58#26.7413#27.21
FAINT_LIM_ERR = 0.16#0.183#0.21

# ==========================================

# ==========================================
# 1. CORE PHYSICS FUNCTIONS
# ==========================================

def calculate_radius(L, T):
    return np.sqrt((L * L_sun.cgs.value) / (4 * np.pi * sigma_sb.cgs.value * T**4))

def planck_lambda(wavelength_cm, T):
    T = np.atleast_1d(T)[:, None]
    wl = wavelength_cm[None, :]
    exponent = (h.cgs.value * c.cgs.value) / (wl * k_B.cgs.value * T)
    exponent = np.clip(exponent, 1e-10, 700)
    B_lambda = (2 * h.cgs.value * c.cgs.value**2) / (wl**5) / (np.exp(exponent) - 1)
    return B_lambda

def get_ab_magnitude_vectorized(L_array, T_array, filter_data):
    distance_cm = (10 * u.pc).to(u.cm).value
    wavelength_A = filter_data[:, 0]
    throughput = filter_data[:, 1]
    wavelength_cm = wavelength_A * 1e-8
    frequency_Hz = c.cgs.value / wavelength_cm
    
    sorted_idx = np.argsort(frequency_Hz)
    frequency_Hz = frequency_Hz[sorted_idx]
    throughput = throughput[sorted_idx]
    wavelength_cm = wavelength_cm[sorted_idx]

    L_array = np.atleast_1d(L_array)
    T_array = np.atleast_1d(T_array)
    
    radius = calculate_radius(L_array, T_array)
    B_lambda = planck_lambda(wavelength_cm, T_array)
    
    dilution = (radius / distance_cm)**2
    F_lambda = B_lambda * np.pi * dilution[:, None]
    F_nu = F_lambda * (wavelength_cm[None, :]**2) / c.cgs.value

    if hasattr(np, 'trapezoid'):
        num = np.trapezoid(F_nu * throughput[None, :], frequency_Hz, axis=1)
        denom = np.trapezoid(throughput, frequency_Hz)
    else:
        num = np.trapz(F_nu * throughput[None, :], frequency_Hz, axis=1)
        denom = np.trapz(throughput, frequency_Hz)
    f_nu = num / denom

    return -2.5 * np.log10(f_nu) - 48.6

# ==========================================
# 2. DATA READING FUNCTIONS
# ==========================================

# ==========================================
# 2. DATA READING & SED RECONSTRUCTION
# ==========================================

# Pivot Wavelengths (Angstroms) from calc_pivot_wl.py
PIVOT_WL = {
    'U': 3673.02,
    'B': 4458.46,
    'V': 5495.37,
    'R': 6532.82,
    'I': 7897.25,
    'J': 12386.51
}

# AB - Vega Offsets (approximate for Johnson-Cousins & 2MASS)
# m_AB = m_Vega + Offset
AB_OFFSETS = {
    'U': 0.79,
    'B': -0.09,
    'V': 0.03, # User specified 0.03 for V
    'R': 0.21,
    'I': 0.45,
    'J': 0.91
}

# BPASS Column Indices (0-based) for U, B, V, R, I, J (Cols 53-58 in 1-based documentation)
# 1-based: 53(U), 54(B), 55(V), 56(R), 57(I), 58(J) -> 0-based: 52, 53, 54, 55, 56, 57
# Wait, user said confirmed indices: U(53), B(54), V(55), R(56), I(57), J(58) in 0-based indexing?
# Let's check the previous turn's BPASS view.
# Turn 1 view showed col 53 as first mag column. 
# Let's assume the user's "U(53)" meant the index in the array (0-based).
# If the file has columns 1..N, and we read with pandas/loadtxt, index 0 is Col 1.
# BPASS manual says cols 53-57 are broad band magnitudes. 
# Let's verify with the "confirmed indices" message: "U: 53, B: 54, V: 55, R: 56, I: 57, J: 58". 
# I will use these indices.

BPASS_COLS = {
    'U': 53,
    'B': 54,
    'V': 55,
    'R': 56,
    'I': 57,
    'J': 58
}


def load_vis_filter():
    path = VIS_FILTER_PATH
    if not os.path.exists(path):
        # Local fallback or error
        print(f"Error: Filter {path} not found.")
        return np.zeros((2,2)) # returns dummy
    data = np.loadtxt(path)
    idx = np.argsort(data[:, 0])
    return data[idx]

VIS_FILTER_DATA = load_vis_filter()

# Keep V_FILTER_DATA for WR stars (observational data often single star approx)
v_filter_path = str(paths.FILTERS / 'v_band_filter.dat')
if os.path.exists(v_filter_path):
    V_FILTER_DATA = np.loadtxt(v_filter_path, delimiter=',')
else:
    V_FILTER_DATA = None

def calculate_v_to_vis_correction(T_array, vis_filter_data, v_filter_data):
    """
    Calculates the magnitude difference (M_VIS - M_V) for a blackbody of temperature T.
    Returns: Correction array where M_VIS_est = M_V_obs + Correction
    """
    # Calculate M_V for BB(T) (Arbitrary L, cancels out in difference)
    L_dummy = 1.0 # Solar luminosities
    m_v_bb = get_ab_magnitude_vectorized(L_dummy, T_array, v_filter_data)
    m_vis_bb = get_ab_magnitude_vectorized(L_dummy, T_array, vis_filter_data)
    return m_vis_bb - m_v_bb

def interpolate_sed_and_integrate(mags_ab, vis_filter_data):
    """
    mags_ab: Dictionary of AB magnitudes {'U': val, 'B': val, ...}
    Returns: M_VIS_AB
    """
    pivot_wls = []
    log_f_lambdas = []
    
    bands = ['U', 'B', 'V', 'R', 'I', 'J']
    valid_bands = []
    
    for b in bands:
        if b in mags_ab and not np.isnan(mags_ab[b]):
            wl = PIVOT_WL[b]
            mag = mags_ab[b]
            f_nu = 10**(-0.4 * (mag + 48.6))
            c_ang = 2.99792458e18
            f_lam = f_nu * c_ang / (wl**2)
            
            pivot_wls.append(wl)
            log_f_lambdas.append(np.log10(f_lam))
            valid_bands.append(b)
            
    if len(valid_bands) < 2:
        return np.nan 
        
    vis_wl = vis_filter_data[:, 0]
    vis_tr = vis_filter_data[:, 1]
    
    # Interpolate SED
    interp_log_f = np.interp(np.log10(vis_wl), np.log10(pivot_wls), log_f_lambdas)
    f_lam_interp = 10**interp_log_f
    
    c_ang = 2.99792458e18
    f_nu_interp = f_lam_interp * (vis_wl**2) / c_ang
    
    if hasattr(np, 'trapezoid'):
        trapz = np.trapezoid
    else:
        trapz = np.trapz

    num = trapz(f_nu_interp * vis_tr * (c_ang / vis_wl**2), vis_wl)
    den = trapz(vis_tr * (c_ang / vis_wl**2), vis_wl)
    
    if den == 0 or num <= 0: return np.nan
    
    f_nu_avg = num / den
    m_vis = -2.5 * np.log10(f_nu_avg) - 48.6
    return m_vis

def read_bpass_track(filename, vis_filter_data=None, v_filter_data= None):
    """ Reads full track history. """
    try:
        df = pd.read_csv(filename, sep=r'\s+', header=None, engine='python')
    except:
        return None
        
    if df.empty: return None
    
    log_Teff = df[3].values
    log_L = df[4].values
    mass = df[5].values
    He_core = df[6].values # Column 7 (He core mass)
    H_surf = df[10].values # Column 11 (X)
    He_surf = df[11].values # Column 12 (Y)
    He_mass = df[17].values # Column 18 (Mass of He)
    
    M_VIS_LIST = []
    
    # BPASS v2.2 typically has 53 columns?? 
    # User confirmed indices up to 58.
    has_phot = (df.shape[1] > 58)
    
    for i in range(len(df)):
        m_vis = np.nan
        if has_phot:
            mags = {}
            for band, col_idx in BPASS_COLS.items():
                m_vega = df.iloc[i, col_idx]
                mags[band] = m_vega + AB_OFFSETS[band]
            
            m_vis = interpolate_sed_and_integrate(mags, VIS_FILTER_DATA)
        
        # NO FALLBACK based on Blackbody, as per user request (and physics)
        # If no photometry, we have no valid VIS magnitude for this composite system.
        M_VIS_LIST.append(m_vis)
        
    M_VIS = np.array(M_VIS_LIST)
    log_g = 4.44 + np.log10(mass) - log_L + 4 * (log_Teff - 3.7617)
    
    return {'log_Teff': log_Teff, 'M_V': M_VIS, 'log_L': log_L, 
            'mass': mass, 'He_core': He_core, 'H_surf': H_surf, 'He_surf': He_surf, 'He_mass': He_mass, 'log_g': log_g, 
            'filename': os.path.basename(filename)}

def get_final_state(filename, vis_filter_data=None, v_filter_data=None):
    """ Reads last line of a track. """
    try:
        df = pd.read_csv(filename, sep=r'\s+', header=None, engine='python')
    except:
        return None
    if df.empty: return None
    
    last = df.iloc[-1]
    m_vis = np.nan
    
    if len(last) > 58:
        mags = {}
        for band, col_idx in BPASS_COLS.items():
            m_vega = last[col_idx]
            mags[band] = m_vega + AB_OFFSETS[band]
        m_vis = interpolate_sed_and_integrate(mags, VIS_FILTER_DATA)
    
    # NO FALLBACK.
    
    # Parse q/logP
    parts = os.path.basename(filename).split('-')
    try:
        q = float(parts[3])
        logP = float(parts[4])
    except:
        q = 1.0
        logP = 0.0
    
    return {'log_Teff': last[3], 'M_V': m_vis, 'mass': last[5], 
            'He_core': last[6], # Col 7
            'H_surf': last[10], 'He_surf': last[11], # Cols 11 & 12
            'He_mass': last[17], # Col 18
            'q': q, 'logP': logP,
            'age': last[1]} # Col 2: Age (years)


def _cached(fn):
    store = os.path.join(CACHE_DIR, fn.__name__ + '.pkl')
    memo = pickle.load(open(store, 'rb')) if os.path.exists(store) else {}
    def wrapper(filename, *a, **k):
        key = os.path.basename(filename)
        if key not in memo:
            memo[key] = fn(filename, *a, **k)
        return memo[key]
    wrapper.flush = lambda: pickle.dump(memo, open(store, 'wb'))
    return wrapper

read_bpass_track = _cached(read_bpass_track)   # [rev3] speeds up re-runs; results unchanged
get_final_state = _cached(get_final_state)

# ==========================================
# 3. MAIN PLOTTING
# ==========================================

if __name__ == "__main__":
    # Paths
    base_dir = os.path.join(str(paths.BPASS), "bpass-v2.2-newmodels/")
    bin_dir = os.path.join(base_dir, "NEWBINMODS/NEWBINMODS/z014/")
    sin_dir = os.path.join(base_dir, "NEWSINMODS/z014/")
    sec_dir = os.path.join(base_dir, "NEWBINMODS/NEWSECMODS/z014_2/") # Secondary models path
    filter_path = VIS_FILTER_PATH
    
    print("Loading filter data...")
    filter_data = np.loadtxt(filter_path)
    
    # V_FILTER_DATA is loaded globally at top level
    
    fig, ax = plt.subplots(figsize=(ns.SINGLE_COL * 1.45, ns.SINGLE_COL * 1.2))   # [rev3] ~128 mm wide
    
    # --- Setup Colormaps ---
    # 1. H-rich Single (Blue/Green-ish or autumn)
    track_cmap_h = plt.get_cmap('autumn')
    track_norm_h = plt.Normalize(vmin=H_RICH_NORM_LIMITS[0], vmax=H_RICH_NORM_LIMITS[1])
    
    # 2. He-rich Binary (Red - "winter")
    track_cmap_he = plt.get_cmap('winter')
    track_norm_he = plt.Normalize(vmin=HE_RICH_NORM_LIMITS[0], vmax=HE_RICH_NORM_LIMITS[1])

    # --- 1. H-rich Single Tracks (M <= 8) ---
    print("Plotting Single H-rich Tracks (Color by Mass - autumn)...")
    track_masses = H_RICH_MASSES
    H_LABELS = []
    for m in track_masses:
        fname_int = os.path.join(sin_dir, f"sneplot-z014-{int(m)}")
        fname_float = os.path.join(sin_dir, f"sneplot-z014-{float(m):g}")
        candidates = [fname_int, fname_float]
        
        found = False
        for fname in candidates:
            if os.path.exists(fname):
                track = read_bpass_track(fname, filter_data, V_FILTER_DATA)
                if track:
                    teff = track['log_Teff']
                    mv = track['M_V']
                    
                    # Apply Temperature Spike Cut (Vectorized) - Matching user's He-track logic
                    if len(teff) > 1:
                         # Calculate jumps: T[i+1] - T[i] -> positive if heating up
                         t_jumps = teff[1:] - teff[:-1]
                         
                         cut_candidates = np.where(t_jumps > 0.2)[0]
                         
                         if cut_candidates.size > 0:
                             cutoff_idx = cut_candidates[0]
                             teff = teff[:cutoff_idx+1]
                             mv = mv[:cutoff_idx+1]

                    color = track_cmap_h(track_norm_h(m))
                    ax.plot(10**teff, mv, color=color, alpha=0.7, lw=0.8, zorder=2)
                    
                    # Label end of track
                    H_LABELS.append((10**teff[-1], mv[-1], m, color))   # [rev3] placed after the loop to avoid overlaps
                    found = True
                    break

    # --- 2. BPASS He-rich Secondary Tracks (M ~ 2-5) ---
    print("Plotting BPASS He-rich Secondary Tracks (Color by Mass - Winter)...")
    
    # Gather relevant secondary files
    sec_files_all = glob.glob(os.path.join(sec_dir, 'sneplot_2-*'))
    candidates_by_mass = {m: [] for m in HE_RICH_MASSES}
    
    for f in sec_files_all:
        try:
            parts = os.path.basename(f).split('-')
            # sneplot_2-z014-M2-Mr-logP
            if len(parts) < 5: continue
            m2 = float(parts[2])
            m_round = int(round(m2))
            
            if m_round in candidates_by_mass:
                logp = float(parts[4])
                candidates_by_mass[m_round].append((logp, f))
        except:
            continue

    for m in HE_RICH_MASSES:
        tracks_list = candidates_by_mass[m]
        if not tracks_list: 
            continue
            
        # Sort by Period to pick representative selection
        tracks_list.sort(key=lambda x: x[0])
        
        # Select Low, Mid, High Period
        n = len(tracks_list)
        if n > 3:
            selected_indices = [0]
        else:
            selected_indices = range(n)
            
        unique_indices = sorted(list(set(selected_indices)))
        
        for idx in unique_indices:
            logp, pattern = tracks_list[idx]

            if os.path.exists(pattern):
                track = read_bpass_track(pattern, filter_data, V_FILTER_DATA)
                if track and 'H_surf' in track:
                    # FILTER: Keep only He-rich (H < 0.4) AND HOT (T > 10000K, logT > 4.0)
                    mask = (track['H_surf'] < 0.5) & (track['log_Teff'] > 3.5)
                    
                    if np.any(mask):
                        color = track_cmap_he(track_norm_he(m))

                        teff_masked = track['log_Teff'][mask]
                        mv_masked = track['M_V'][mask]
                        
                        # Apply Temperature Spike Cut
                        if len(teff_masked) > 1:
                            # Calculate jumps: T[i+1] - T[i] -> positive if heating up
                            t_jumps = teff_masked[1:] - teff_masked[:-1]
    
                            # Condition: Hot AND Sudden Heating > 0.5 dex
                            cut_candidates = np.where((t_jumps > 0.5))[0]
                            
                            if cut_candidates.size > 0:
                                cutoff_idx = cut_candidates[0]
                                teff_plot = teff_masked[:cutoff_idx+1]
                                mv_plot = mv_masked[:cutoff_idx+1]
                            else:
                                teff_plot = teff_masked
                                mv_plot = mv_masked
                        else:
                            teff_plot = teff_masked
                            mv_plot = mv_masked

                        # Only plot if we have points left
                        if len(teff_plot) > 0:
                            ax.plot(10**teff_plot, mv_plot, color=color, linestyle='--', lw=0.8, zorder=3)
                        
                            # Label at approx 10500 K (or closest point on track)
                            target_logt = np.log10(19000)
                            mask_eff = mv_plot < 5
                            lbl_idx = np.argmin(np.abs(teff_plot[mask_eff] - target_logt))
                            ax.text(10**teff_plot[lbl_idx], mv_plot[lbl_idx]+0.1, r"$%d\,M_{\odot}$" % int(m), rotation=45, color=color, fontsize=ns.FS_SMALL, fontweight='bold', ha='center', va='bottom', zorder=10)

    # --- 3. Secondary Stars (sneplot_2) ---
    print("Processing Secondary Stars Endpoints (sneplot_2)...")
    sec_files = glob.glob(os.path.join(sec_dir, 'sneplot_2-*'))
    endpoint_data_sec = []
    
    for f in sec_files:
        try:
            # Filename: sneplot_2-z014-M2-Mr-logP
            parts = os.path.basename(f).split('-')
            # Example: sneplot_2-z014-100-3.98...-0.4...
            # indices: 0: sneplot_2, 1: z014, 2: M2, 3: Mr, 4: logP
            
            if len(parts) < 5: continue
            
            m_init = float(parts[2]) # M2 (Secondary initial mass)
            mr = float(parts[3])     # Remnant mass
            logp_final = float(parts[4]) # log P after remnant formation

        except: continue

        
        # User Filter: Only end states of mass ranges >8 and <35 for secondary
        if not (m_init > MIN_MASS_LIMIT and m_init < MAX_MASS_LIMIT):
            continue

        # Use get_final_state for efficiency and consistency
        ep = get_final_state(f, filter_data, V_FILTER_DATA)
        if ep and (10**ep['log_Teff'] >= 10000):
             # Apply Abundance Filters (CO Stars)
             if ep['H_surf'] <= MAX_H_SURF and ep['He_core'] <= MAX_HE_CORE_MASS and ep['He_surf'] <= MAX_HE_SURF:
                 ep['m_init'] = m_init
                 ep['mr'] = mr
                 endpoint_data_sec.append(ep)

    if endpoint_data_sec:
        # Bunch everything together - all secondaries in orange
        xs = [10**e['log_Teff'] for e in endpoint_data_sec]
        ys = [e['M_V'] for e in endpoint_data_sec]
        ax.scatter(xs, ys, color='orange', s=9, marker='D', edgecolors='black', linewidth=0.25, alpha=0.7, zorder=4, label='Stripped secondaries')
        
        # Label models fainter than S1 (-3.09) per user request
        for e in endpoint_data_sec:
            if e['M_V'] > 8:#-3.09:
                # "Final mass and the mass of the binary at that stage"
                # Binary Mass = Mass(Secondary, Final) + Mass(Remnant)
                m_final = e['mass']
                label = f"$M_{{ini}}={e['m_init']:.1f}, M_{{rem}}={e['mr']:.1f}$"
                m_final = e['mass']
                # label = f"$M_{{ini}}={e['m_init']:.1f}, M_{{rem}}={e['mr']:.1f}$"
                label = f"$M_{{ini}}={e['m_init']:.1f}, M_{{fin}}={m_final:.2f}$"
                ax.text(10**e['log_Teff'], e['M_V'], label, fontsize=12, alpha=0.8, ha='left', va='bottom', zorder=10, rotation=45)

    # --- 4. High-Mass Endpoints (M > 8, Sparse) ---
    print("Plotting High-Mass Endpoints (Sparse: q=0.1,0.5,0.9; P=1,4)...")
    endpoint_data = []
    all_files = glob.glob(os.path.join(bin_dir, "sneplot-z014-*"))
    
    for f in all_files:
        parts = os.path.basename(f).split('-')
        
        m_ini = float(parts[2])
        q_val = float(parts[3])
        lp_val = float(parts[4])
        
        if np.logical_and(m_ini > MIN_MASS_LIMIT,m_ini<MAX_MASS_LIMIT) and q_val in [0.1, 0.5, 0.9] and lp_val in [1.0,2.0,3.0,4.0]:
            ep = get_final_state(f, filter_data, V_FILTER_DATA)
            if ep and (10**ep['log_Teff'] >= 10000):
                # Apply Abundance Filters (CO Stars)
                if ep['H_surf'] <= MAX_H_SURF and ep['He_core'] <= MAX_HE_CORE_MASS and ep['He_surf'] <= MAX_HE_SURF:
                    ep['m_ini'] = m_ini
                    endpoint_data.append(ep)
        
    if endpoint_data:
        # Bunch everything together - all primaries in blue
        xs = [10**e['log_Teff'] for e in endpoint_data]
        ys = [e['M_V'] for e in endpoint_data]
        ax.scatter(xs, ys, color='blue', s=18, marker='*', edgecolors='black', linewidth=0.25, label='Stripped primaries', zorder=4, alpha=0.6)
        
    # --- 5. WR Population ---
    print("Plotting WR Population...")

    wr_csv = os.path.join(SM_DIR, "WR_pop.csv")
    if os.path.exists(wr_csv):
        df_wr = pd.read_csv(wr_csv)
        T_wr = df_wr['T_kK'].values * 1000.0
        M_v_table = df_wr['M_v_table'].values
        
        # Tabulated WR magnitudes are in Vega system? 
        # If WR_pop.csv has Vega magnitudes, we should convert to AB first.
        # Standard WR catalogs (e.g. Hamann) usually give V_Vega.
        # Conversion: V_AB = V_Vega + 0.03
        M_v_ab_wr = M_v_table + 0.03
        
        correction = calculate_v_to_vis_correction(T_wr, filter_data, V_FILTER_DATA)
        M_vis_wr = M_v_ab_wr + correction
        
        yerr = 2.5 * np.array([df_wr['log_L_err_up'], df_wr['log_L_err_down']])
        ax.errorbar(T_wr, M_vis_wr, yerr=yerr, fmt='none', ecolor='magenta', elinewidth=0.5, alpha=0.3, zorder=5)
        ax.scatter(T_wr, M_vis_wr, color='magenta', alpha=0.5, s=7, marker='s', edgecolors='black', linewidth=0.25, label='WR stars', zorder=6)
        
        # Rotated label for WR stars
        # ax.text(1e5, -5, "WR stars", rotation=45, fontsize=14, color='magenta', ha='center', va='center', fontweight='bold', zorder=10)
        
        # Label WR stars fainter than S1 (-3.09)
        # for i in range(len(df_wr)):
        #     if M_vis_wr[i] > -3.09:
        #         label = f"WR {df_wr['WR'].iloc[i]}"
        #         ax.text(T_wr[i], M_vis_wr[i], label, fontsize=8, color='black', alpha=0.8, ha='left', va='center', zorder=10)

    # --- 6. Overlays & Text Labels ---
    sn_color, ha_color, pup_color = "#045D7A", "#AE001D", "#318a29"
    
    ax.errorbar([8500], [-4.790], xerr=[1003], yerr=[[0.095], [0.105]], fmt='o', ms=3, elinewidth=0.7, color=sn_color, zorder=100)
    ax.text(8500, -4.1, 'SN 2012Z', fontsize=ns.FS_SMALL, ha='center', color=sn_color, weight='bold', zorder=100)
    
    ax.errorbar([2091], [-5.450], xerr=[509], yerr=[[5.631-5.450], [-5.312+5.450]], fmt='o', ms=3, elinewidth=0.7, color=ha_color, zorder=100)
    ax.text(2091, -4.95, 'SN 2008ha', fontsize=ns.FS_SMALL, ha='center', color=ha_color, weight='bold', zorder=100)
    
    ax.errorbar([35000], [-3.795], xerr=[0], yerr=[[0.296], [0.3]], fmt='o', ms=3, elinewidth=0.7, color=pup_color, zorder=100)
    ax.text(35000, -4.2, 'V445\nPup', fontsize=ns.FS_SMALL, ha='center', color=pup_color, weight='bold', zorder=100)
    
    # Calculate Extinction for VIS
    # Calculate pivot wavelength for VIS
    if VIS_FILTER_DATA is not None:
        vis_wl = VIS_FILTER_DATA[:, 0]
        vis_tr = VIS_FILTER_DATA[:, 1]

        if hasattr(np, 'trapezoid'):
            trapz = np.trapezoid
        else:
            trapz = np.trapz

        # Simple Pivot Wavelength: sqrt( int(T l dl) / int(T/l dl) )
        num = trapz(vis_tr * vis_wl, vis_wl)
        den = trapz(vis_tr / vis_wl, vis_wl)
        vis_pivot_wl = np.sqrt(num / den)
        
        # Calculate Extinction
        a_mw = ccm89(np.array([vis_pivot_wl]), EBV_MW * RV_MW, RV_MW)[0]
        a_host = ccm89(np.array([vis_pivot_wl]), EBV_HOST * RV_HOST, RV_HOST)[0]
        A_VIS = a_mw + a_host
        
        print(f"Calculated Extinction for VIS (Pivot WL={vis_pivot_wl:.1f} A): A_MW={a_mw:.3f}, A_host={a_host:.3f}, Total={A_VIS:.3f}")
    else:
        print("Warning: VIS Filter data not found, assuming A_VIS = 0")
        A_VIS = 0.0
    
    s1_faint = S1_MAG-DM+np.sqrt(S1_MAG_ERR**2+DM_ERR**2)-A_VIS#-3.09 - A_VIS
    s1_bright = S1_MAG-DM-np.sqrt(S1_MAG_ERR**2+DM_ERR**2)-A_VIS#-4.21 - A_VIS
    print(S1_MAG-DM-A_VIS,np.sqrt(S1_MAG_ERR**2+DM_ERR**2))

    ax.axhspan(s1_bright, s1_faint, color="#FF8F8F", alpha=0.3, zorder=0)
    ax.axhspan(s1_bright, -50, color='red', alpha=0.3, zorder=0)

    ax.text(1.5e3, s1_faint, 'S1 brightness', fontsize=ns.FS, ha='center', color='black', zorder=100)
    ax.text(1.65e3, -6.5, 'Excluded region', fontsize=ns.FS, ha='center', color='black', zorder=100)
    
    # Grey Area to 27 mag (Faintest Source)
    # Distance Modulus
    #d_mpc = 7.92
    # DM = 29.86#5 * np.log10(d_mpc * 1e6) - 5
    m_faint_limit = FAINT_LIM
    m_faint_err = FAINT_LIM_ERR
    M_faint_limit = m_faint_limit- DM - A_VIS + np.sqrt(DM_ERR**2+m_faint_err**2)
    print(m_faint_limit- DM - A_VIS)
    print(m_faint_limit- DM-A_VIS,np.sqrt(DM_ERR**2+m_faint_err**2))
    
    ax.axhspan(s1_faint, M_faint_limit, color='grey', alpha=0.3, zorder=0)
    ax.text(1.5e3, (s1_faint + M_faint_limit)/2.0, r"SN location", 
            fontsize=ns.FS, ha='center', va='center', color='black', zorder=100)
    
    # WD systems
    wd_path = os.path.join(SM_DIR, "Ultracompact Hot Subdwarf-WD Binaries_with_MVIS.csv")
    if os.path.exists(wd_path):
        df_wd = pd.read_csv(wd_path)
        ax.errorbar(df_wd["T_eff"], df_wd["M_VIS_AB"], 
                    yerr=[df_wd["M_VIS_AB_err_minus"], df_wd["M_VIS_AB_err_plus"]],
                    fmt='o', color='black', markersize=2.5, capsize=1.2, elinewidth=0.6, label='sdB$-$WD binaries', zorder=3)

    # Axis Formatting
    ax.set_xscale('log')
    ax.set_xlim(2.2e5, 1000) 
    ax.set_ylim(8, -7) 
    ax.set_xlabel(r'$T_{\mathrm{eff}}$ (K)')
    ax.set_ylabel(r'$M_{\mathrm{VIS}}$ (mag)')
    ax.grid(True, linestyle='--', lw=0.3, alpha=0.4, which='both')
    
    # Clean Legend
    handles, labels = ax.get_legend_handles_labels()
    by_label = {}
    for h, l in zip(handles, labels):
        if l and l not in by_label: by_label[l] = h
    # [rev3] key for the two BPASS track types (previously only in the caption)
    from matplotlib.lines import Line2D
    from matplotlib.legend_handler import HandlerTuple
    h_cols = [track_cmap_h(track_norm_h(m)) for m in (2, 5, 8)]
    he_cols = [track_cmap_he(track_norm_he(m)) for m in (2, 5, 8)]
    by_label['H-rich tracks'] = tuple(Line2D([], [], color=c, lw=0.9) for c in h_cols)
    by_label['He-star tracks'] = tuple(Line2D([], [], color=c, lw=0.9, ls='--') for c in he_cols)
    ax.legend(list(by_label.values()), list(by_label.keys()), loc='lower left', fontsize=ns.FS_SMALL,
              frameon=False, handler_map={tuple: HandlerTuple(ndivide=None, pad=0)}, handlelength=2.4)

    # [rev3] H-rich mass labels at the track ends, nudged apart so they do not overlap
    # each other, the shaded bands' text or the SN 2008ha label
    placed_y = []
    for T_end, mv_end, m, color in sorted(H_LABELS, key=lambda r: r[1]):
        y = mv_end
        while any(abs(y - py) < 0.42 for py in placed_y):
            y += 0.42
        placed_y.append(y)
        ax.text(T_end * 0.93, y, r"$%d\,M_{\odot}$" % int(m), color=color, fontsize=ns.FS_SMALL,
                fontweight='bold', ha='left', va='center', zorder=10)
    
    # --- Statistics (Fainter than S1) ---
    print("\n--- Statistics (Values below S1 Faint Limit) ---")
    limit_mag = s1_faint 
    print(f"Threshold (S1 Faint Limit): {limit_mag:.3f} mag (brighter < threshold < fainter)")
    print(f"Counting objects fainter than this limit (M_V > {limit_mag:.3f})")

    def print_stat(name, mags, ages=None):
        total = len(mags)
        if total == 0:
            print(f"{name}: 0 total")
            return
        mags_arr = np.array(mags)
        # Fainter means M_V > limit_mag
        below_idx = np.where(mags_arr > limit_mag)[0]
        below = len(below_idx)
        pct = (below / total) * 100
        print(f"{name}: {below}/{total} ({pct:.1f}%) are below (fainter than) S1 limit")
        
        # Near limit (within 0.5 mag fainter)
        near_limit = np.sum((mags_arr > limit_mag) & (mags_arr < limit_mag + 0.5))
        pct_near = (near_limit / total) * 100
        print(f"  -> Near limit (within 0.5 mag fainter): {near_limit}/{total} ({pct_near:.1f}%)")
        
        # Age Statistics for Faint endpoints
        if ages is not None and below > 0:
            ages_arr = np.array(ages)
            faint_ages = ages_arr[below_idx] / 1e6 # Convert to Myr
            med_age = np.median(faint_ages)
            min_age = np.min(faint_ages)
            max_age = np.max(faint_ages)
            print(f"  -> Age of Faint Systems: Median = {med_age:.1f} Myr (Range: {min_age:.1f}-{max_age:.1f} Myr)")

    # Collect magnitudes and ages - bunch everything together
    primary_mags = []
    primary_ages = []
    secondary_mags = []
    secondary_ages = []
    
    # From Secondaries
    if endpoint_data_sec:
        for e in endpoint_data_sec:
            age = e.get('age', np.nan)
            secondary_mags.append(e['M_V'])
            secondary_ages.append(age)

    # From Primaries (High Mass)
    if endpoint_data:
        for e in endpoint_data:
            age = e.get('age', np.nan)
            primary_mags.append(e['M_V'])
            primary_ages.append(age)
                
    print_stat("Stripped Stars (Primaries)", primary_mags, primary_ages)
    print_stat("Stripped Secondaries", secondary_mags, secondary_ages)
    
    print_stat("WR Stars", M_vis_wr)

    # --- Statistics (Faintest Source Limit) ---
    print("\n--- Statistics (Faintest Source Limit) ---")
    # Recalculate M_faint_limit locally if needed or variable availability
    # It was calculated above: M_faint_limit = m_faint_limit - DM - A_VIS + correction
    # We'll recompute to be safe/clear
    # d_mpc = 7.92
    # DM = 29.86#5 * np.log10(d_mpc * 1e6) - 5
    m_faint_val = FAINT_LIM
    m_faint_err = FAINT_LIM_ERR
    M_faint_abs = m_faint_val - DM - A_VIS + np.sqrt(DM_ERR**2+m_faint_err**2)
    
    print(f"Threshold (Faintest Source): {M_faint_abs:.3f} mag")
    print(f"1. Detectable Gap (S1 Faint < M < Faintest Source): Candidates that should be visible")
    print(f"2. Undetectable (M > Faintest Source): Too faint to see")
    
    def print_stat_gap(name, mags):
        mags_arr = np.array(mags)
        total = len(mags)
        if total == 0: return

        # Between S1 and Faintest
        in_gap = np.sum((mags_arr > s1_faint) & (mags_arr < M_faint_abs))
        pct_gap = (in_gap / total) * 100
        
        # Fainter than Faintest
        below_faint = np.sum(mags_arr > M_faint_abs)
        pct_below = (below_faint / total) * 100
        
        print(f"{name}:")
        print(f"  -> In Detectable Gap ({s1_faint:.2f} < M < {M_faint_abs:.2f}): {in_gap}/{total} ({pct_gap:.1f}%)")
        print(f"  -> Undetectable (M > {M_faint_abs:.2f}): {below_faint}/{total} ({pct_below:.1f}%)")

    print_stat_gap("Stripped Stars (Primaries)", primary_mags)
    print_stat_gap("Stripped Secondaries", secondary_mags)
    if 'M_vis_wr' in locals():
        print_stat_gap("WR Stars", M_vis_wr)
    
    read_bpass_track.flush(); get_final_state.flush()
    plt.tight_layout()
    ns.save(fig, 'Fig2_Progenitor_limits')   # -> outputs/figures
