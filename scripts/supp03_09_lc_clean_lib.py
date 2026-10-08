# Light-curve library (lc_clean.py): photometry loading, extinction, GP interpolation.
# Data/GP functions only; the figures are made by the supp03_/supp09_ drivers.
# ===========================
# SN2024vjm — Clean Pipeline
# ===========================
import os
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib
matplotlib.rcParams['font.family'] = 'DejaVu Sans'
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from astropy.time import Time
from astropy import constants as const
from astropy import units as u
from astropy.io import ascii

from scipy.interpolate import interp1d
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (
    RBF, ConstantKernel as C, Matern, DotProduct, WhiteKernel
)
from SNeSCOPE.PhotoUtils import *   # zp_band_ab, magerr2fluxerr, maggie2cgs, f_nu2f_lam
from extinction import ccm89, remove

# ---------- GLOBALS ----------
T0_BMAX   = 60572.29
DM        = 29.86#29.48
DIST_PC   = 9.4e6#7.92e6
EXPLODE_T = 60565.58   # [rev3] power-law fit (explosion_time.ipynb); was 60565.89

# Colors (single source of truth)
COLORS = {
    'u': '#7B68EE', 'B': '#4169E1', 'g': '#00CED1', 'V': '#ADFF2F', 'q': '#FFD700',
    'r': '#FF6347', 'i': '#DC143C', 'z': '#B22222', 'J': '#D5722C', 'H': '#673C22', 'K': '#000000','Open':"#B5A16A",
    'L': '#8c8942'
}

# Magnitude-plot offsets (mag units; top legend uses these)
FILTER_OFFSETS = {
    'U': 3.5, 'u': 3.5, 'bg_u': 3.5,
    'B': 3.0, 'bg_B': 3.0, 'g': 2.5, 'bg_g': 2.5, 'V': 2.0, 'bg_V': 2.0, 'q': 1.5, 'bg_q': 1.5,
    'r': 0.5, 'bg_r': 0.5, 'i': 0.0, 'bg_i': 0.0, 'z': -0.5, 'bg_z': -0.5,
    'J': 0.0, 'H': 0.0, 'K': 0.0,'Open':1.0, 'L': 1.0
}

# Flux-plot offsets (for GP flux panel; units of 1e-16 erg/s/cm^2/Å)
GP_FLUX_OFFSETS = {
    'U': 15.0, 'u': 15.0, 'bg_u': 18.0, 'B': 12.0, 'bg_B': 12.0, 'g': 9.0, 'bg_g': 9.0,
    'V': 6.0, 'bg_V': 6.0, 'q': 3.0, 'bg_q': 3.0, 'r': 0.0, 'bg_r': 0.0,
    'i': -3.0, 'bg_i': -3.0, 'z': -6.0, 'bg_z': -6.0, 'J': -9.0, 'H': -12.0, 'K': -15.0
}

# AB - Vega zero-point offsets (AB = Vega + offset)
AB_MINUS_VEGA = {'U':0.79,'B':-0.09,'V':0.02,'R':0.21,'I':0.45,'J':0.91,'H':1.39,'K':1.85}

# Paths (set with --data / SN2024VJM_DATA, see paths.py)
import paths
PATH_ROOT   = paths.PHOTOMETRY
PATH_TELDIR = PATH_ROOT / "telescope_data"
PATH_LCO    = PATH_ROOT / "SN2024vjm_LCOGT_LCs_all.csv"
PATH_ML     = PATH_ROOT / "ML.csv"
# PATH_BG     = PATH_ROOT / "new_phot_bg.csv"
PATH_BG     = PATH_ROOT / "BG_lc_sep2025.csv"
PATH_DLT40  = PATH_ROOT / "SN2024vjm_dlt40_Open_binned_1_day.csv"
PATH_GOTO   = PATH_ROOT / "SN2024vjm_GOTO_forcedphotometry.json"
PATH_MKD    = PATH_ROOT / "mkd_bfrcat_forced_lightcurve_merged.csv"

# Filter transmission paths
path_filters = paths.FILTERS
FILTER_FILES = {
    'bg_u':'u-bg.txt','bg_g':'g-bg.txt','bg_r':'r-bg.txt','bg_i':'i-bg.txt','bg_z':'z-bg.txt','bg_q':'q-bg.txt',
    'B':'LasCumbres_LasCumbres.Bessel_B.dat','V':'johnson_v.txt','g':'SDSS_g.txt','r':'SDSS_r.txt','i':'SDSS_i.txt',
    'z':'SDSS_z.txt','J':'REM_J.txt','H':'REM_H.txt','K':'REM_K.txt',
    'L': 'q-bg.txt'
}
FILTER_FILES = {k: path_filters / v for k, v in FILTER_FILES.items()}

# --- telescope markers for plots
TELESCOPE_MARKERS = {
    'MeerLICHT':'o', 'BlackGEM':'o', 'BlackGEM_fp':'o', 'REM,REMIR':'h', 'REM,REM-ROS2-DIF':'h',
    'Prompt6,FLI':'*','Prompt6,FLI-DIF':'*','ESO-NTT,EFOSC':'D','LCO':'s', 'GOTO': '^', 'IMACS': 'X',
    'Mookodi': 'p'
}

# ============== UTILITIES ==============
def vega_to_ab(mag_vega, filt):
    """Return AB magnitude given Vega magnitude and filter."""
    return mag_vega + AB_MINUS_VEGA.get(filt, 0.0)

def flux_to_mag(flux, piv_wl):
    """F_lambda → AB mag (safe)."""
    c = 2.99792458e18  # Ang/s
    fnu0 = 3.631e-20
    flux = np.asarray(flux)
    fnu = np.maximum(flux, 1e-40) * (piv_wl**2) / c
    return -2.5 * np.log10(fnu / fnu0)

def fluxerr_to_magerr(flux, fluxerr):
    flux = np.maximum(np.asarray(flux), 1e-40)
    return (2.5/np.log(10)) * (np.asarray(fluxerr)/flux)

def bin_photometry(df, time_col='mjd', mag_col='mag', magerr_col='magerr', filter_col='filter', bin_size=0.3):
    """Weighted binning by time per filter."""
    out = []
    for filt, sub in df.groupby(filter_col):
        s = sub.sort_values(time_col).reset_index(drop=True)
        used = np.zeros(len(s), dtype=bool)
        for i in range(len(s)):
            if used[i]: continue
            dt = np.abs(s[time_col] - s.loc[i, time_col]) <= bin_size
            idx = np.where(dt & ~used)[0]
            if len(idx) == 1:
                out.append(s.loc[i])
            else:
                w = 1.0/np.clip(s.loc[idx, magerr_col].values, 1e-6, np.inf)**2
                m  = np.average(s.loc[idx, mag_col].values, weights=w)
                me = 1.0/np.sqrt(np.sum(w))
                t  = np.average(s.loc[idx, time_col].values, weights=w)
                row = s.loc[i].copy()
                row[mag_col], row[magerr_col], row[time_col] = m, me, t
                out.append(row)
            used[idx] = True
    return pd.DataFrame(out)


# ======== FILTER THROUGHPUT / PIVOT WAVELENGTHS ========
# Requires SNeSCOPE.PhotoUtils
from SNeSCOPE.PhotoUtils import zp_band_ab, magerr2fluxerr, maggie2cgs, f_nu2f_lam

def add_pivot_wavelengths(lc):
    """Add piv_wl per filter using throughput files."""
    trans = {}
    for f, path in FILTER_FILES.items():
        try:
            trans[f] = ascii.read(path)
        except Exception:
            trans[f] = ascii.read(path, delimiter=",")
    piv = []
    for _, r in lc.iterrows():
        T = trans[r['filter']]
        _, pivot_wl = zp_band_ab(T)
        piv.append(pivot_wl)
    lc['piv_wl'] = np.array(piv, dtype=float)
    return lc

def load_filter_transmissions(path_filters):
    """
    Load all filter transmission curves into a dict of astropy Tables.

    Parameters
    ----------
    path_filters : str
        Base directory containing filter transmission files.

    Returns
    -------
    dict[str, astropy.table.Table]
        Keys are filter names, values are transmission curves.
    """

    filter_files = {
        'ZTF_g': 'P48_g.txt',
        'ZTF_r': 'P48_r.txt',
        'ZTF_i': 'ZTF_transmission_i_from_twiki.ascii',
        'u_P60': 'SDSS_u.txt',
        'g_P60': 'SDSS_g.txt',
        'r_P60': 'SDSS_r.txt',
        'i_P60': 'SDSS_i.txt',
        'sdss_u': 'SDSS_u.txt',
        'g': 'SDSS_g.txt',
        'r': 'SDSS_r.txt',
        'i': 'SDSS_i.txt',
        'z': 'SDSS_z.txt',
        'LT_u': 'LT_u.txt',
        'LT_g': 'LT_g.txt',
        'LT_r': 'LT_r.txt',
        'LT_i': 'LT_i.txt',
        'LT_z': 'LT_z.txt',
        'LCO_u': 'LCO_AA_SDSS.up.txt',
        'LCO_g': 'LCO_AA_SDSS.gp.txt',
        'LCO_r': 'LCO_AA_SDSS.rp.txt',
        'LCO_i': 'LCO_AA_SDSS.ip.txt',
        'LCO_V': 'LasCumbres_LasCumbres.Bessel_V.dat',
        'LCO_B': 'LasCumbres_LasCumbres.Bessel_B.dat',
        'UVM2': 'Swift_UVM2.rtf',
        'UVW1': 'Swift_UVW1.rtf',
        'UVW2': 'Swift_UVW2.rtf',
        'u_swift': 'Swift_u.rtf',
        'v_swift': 'Swift_V.rtf',
        'b_swift': 'Swift_B.rtf',
        'NOT_u': 'NOT_u.txt',
        'NOT_g': 'NOT_g.txt',
        'NOT_r': 'NOT_r.txt',
        'NOT_i': 'NOT_i.txt',
        'NOT_z': 'NOT_z.txt',
        'KAIT_B': 'B_kait4_shifted.txt',
        'KAIT_V': 'V_kait4_shifted.txt',
        'KAIT_R': 'R_kait4_shifted.txt',
        'KAIT_I': 'I_kait4_shifted.txt',
        'KAIT_CLEAR': 'e2v_QE_midband.csv',
        'Ni_B': 'B_Nickel2.txt',
        'Ni_V': 'V_Nickel2.txt',
        'Ni_R': 'R_Nickel2.txt',
        'Ni_I': 'I_Nickel2.txt',
        'Ni_CLEAR': 'e2v_QE_midband.csv',
        'MMIRS_J': 'MMT_MMIRS.J.dat',
        'MMIRS_H': 'MMT_MMIRS.H.dat',
        'MMIRS_Ks': 'MMT_MMIRS.Ks.dat',
        '2MASS_J': '2MASS_J.txt',
        '2MASS_H': '2MASS_H.txt',
        'U': 'johnson_u.txt',
        'B': 'LasCumbres_LasCumbres.Bessel_B.dat',
        'V': 'johnson_v.txt',
        'R': 'LasCumbres_LasCumbres.Bessel_V.dat',
        'I': 'LasCumbres_LasCumbres.Bessel_B.dat',
        'R-PTF': 'P48_R_T.rtf',
        'bg_u': 'u-bg.txt',
        'bg_g': 'g-bg.txt',
        'bg_r': 'r-bg.txt',
        'bg_i': 'i-bg.txt',
        'bg_z': 'z-bg.txt',
        'bg_q': 'q-bg.txt',
        'K': 'REM_K.txt',
        'H': 'REM_H.txt',
        'J': 'REM_J.txt',
    }

    transmissions = {}
    for name, fname in filter_files.items():
        fpath = os.path.join(path_filters, fname)
        if not os.path.exists(fpath):   # only the filters used for SN 2024vjm are shipped
            continue
        try:
            transmissions[name] = ascii.read(fpath)
        except Exception:
            # fallback: try with numpy if ascii.read fails
            try:
                arr = np.loadtxt(fpath)
                transmissions[name] = arr
            except Exception as e:
                print(f"⚠️ Could not load filter {name} from {fpath}: {e}")

    return transmissions

# ============== DATA LOADERS (standardize schema) ==============
def load_meerlicht(path_csv):
    ml = pd.read_csv(path_csv)
    det = ml[(ml['Mag_zogy'] < 99) & (ml['Flux_zogy'] > 3*ml['Fluxerr_zogy'])]
    ml_bin = bin_photometry(det.rename(columns={'MJD':'mjd','Mag_zogy':'mag','Magerr_zogy':'magerr','Filter':'filter'}))
    ml_bin['source'] = 'MeerLICHT'
    ml_bin = ml_bin.rename(columns={'mjd':'time','mag':'magnitude','magerr':'error'})
    ml_bin['filter'] = 'bg_' + ml_bin['filter'].astype(str)
    return ml_bin[['time','filter','magnitude','error','source']]

def load_blackgem(path_csv):
    bg = pd.read_csv(path_csv, sep=' ')
    det = bg[(bg['MAG_ZOGY'] < 99) & (bg['FNU_ZOGY'] > 3*bg['FNUERR_ZOGY'])]
    bg_bin = bin_photometry(det.rename(columns={'MJD-OBS':'mjd','MAG_ZOGY':'mag','MAGERR_ZOGY':'magerr','FILTER':'filter'}))
    bg_bin['source'] = 'BlackGEM_fp'
    bg_bin = bg_bin.rename(columns={'mjd':'time','mag':'magnitude','magerr':'error'})
    bg_bin['filter'] = 'bg_' + bg_bin['filter'].astype(str)
    return bg_bin[['time','filter','magnitude','error','source']]

def load_andrea_txts(folder):
    # files list
    files = [
        folder/'2024vjm_REM_REMIR_photometry.txt',
        folder/'2024vjm_Prompt6_FLI-DIF_photometry.txt',
        folder/'2024vjm_REM_REM-ROS2-DIF_photometry.txt',
        folder/'2024vjm_ESO-NTT_EFOSC_photometry.txt',
        folder/'2024vjm_Prompt6_FLI_photometry.txt',
    ]
    rows = []
    for path in files:
        df = pd.read_csv(path, comment='#', sep=r'\s+')
        tel = df['Telescope'][0]
        for _, r in df.iterrows():
            if r['Flag'] != 'P': continue
            m = r['Magnitude']
            f = r['Filter']
            if r['System'] == 'Vega':
                m = vega_to_ab(m, f)
            rows.append(dict(time=r['MJD'], filter=f, magnitude=m, error=r['Error'], source=tel))
    return pd.DataFrame(rows)

def load_REM_epoch(epoch=60568.007, convert_to_ab=True):
    mags = {
        'r': [17.9693],
        'g': [18.6401],
        'i': [17.6785],
        'z': [18.0972],
        'J': [16.9493]
    }
    magerrs = {
        'r': [0.0972],
        'g': [0.1621],
        'i': [0.1159],
        'z': [0.1337],
        'J': [0.1365]
    }

    rows = []
    for f in mags:
        mag = float(mags[f][0])
        err = float(magerrs.get(f, [None])[0])
        source = 'REM,REMIR' if f == 'J' else 'REM,REM-ROS2-DIF'

        if convert_to_ab and f == 'J':
            mag += 0.91  # Vega → AB for J band

        rows.append({
            'time': epoch,
            'filter': f,
            'magnitude': mag,
            'error': err,
            'limit': '',
            'source': source
        })

    return pd.DataFrame(rows)

def load_efosc2_manual():
    """Manual entry for the late-time EFOSC2 points."""
    rows = [
        {'time': 60781.3426, 'filter': 'i', 'magnitude': 21.2057, 'error': 0.19254, 'source': 'ESO-NTT,EFOSC'},
        {'time': 60781.3426, 'filter': 'z', 'magnitude': 22.0089, 'error': 0.35782, 'source': 'ESO-NTT,EFOSC'}
    ]
    return pd.DataFrame(rows)


def load_lco(path_lco):
    """
    Load LCO photometry data from a text file.
    """
    df = pd.read_csv(path_lco)
    df['time'] = Time(df['jd'], format='jd').mjd
    df['filter'] = df['flt']
    df['magnitude'] = df['mag']
    df['error'] = df['magerr']
    df['source'] = 'LCO'
    df = df[df['error'] < 99]
    return df[['time', 'filter', 'magnitude', 'error', 'source']]



def load_bg_raw(path_bg):
    """
    Return two DataFrames from the *unbinned* BlackGEM file:
      - bg_detections: rows with significant detections (same columns as file)
      - bg_non_detections: rows with non-detections (expects LIMMAG_ZOGY in file)
    These frames are passed directly to plotting (no column renaming).
    """
    bg = pd.read_csv(path_bg, sep=r"\s+")
    # Detection threshold as in your script
    det_mask = (bg['MAG_ZOGY'] < 99) & (bg['FNU_ZOGY'] > 3 * bg['FNUERR_ZOGY'])
    bg_detections = bg[det_mask].copy()

    # Non-detections: keep original column names (FILTER, MJD-OBS, LIMMAG_ZOGY)
    nd_mask = ~det_mask
    bg_non_detections = bg[nd_mask].copy()
    # if a few rows miss LIMMAG_ZOGY, drop them rather than crash the plot
    if 'LIMMAG_ZOGY' in bg_non_detections.columns:
        bg_non_detections = bg_non_detections.dropna(subset=['LIMMAG_ZOGY', 'MJD-OBS', 'FILTER'])

    return bg_detections, bg_non_detections


def load_ml_nondet(path_ml):
    """
    Return MeerLICHT non-detections with original columns used by the plotter:
      Filter, MJD, Mag_zogy
    """
    ml = pd.read_csv(path_ml)
    ml_non = ml[ml['Mag_zogy'] >= 99].copy()
    # Keep only the columns the plotter consumes
    keep = ['Filter', 'MJD', 'Mag_zogy']
    missing = [c for c in keep if c not in ml_non.columns]
    if missing:
        raise ValueError(f"MeerLICHT file missing columns: {missing}")
    return ml_non[keep]

def load_REM_epoch_limits(epoch=60568.007):
    """
    Return a tiny DataFrame with REM H/K non-detections (limits only).
    Columns chosen to be easy to plot alongside detections.
    """
    # magnitudes are limits you provided
    rows = [
        {'time': epoch, 'filter': 'H', 'limit_mag': 17.4, 'source': 'REM,REMIR'},
        {'time': epoch, 'filter': 'K', 'limit_mag': 16.7, 'source': 'REM,REMIR'},
    ]
    return pd.DataFrame(rows)

def load_dlt40(path_dlt40):
    """
    Load DLT40 light curve data from a text file.
    """
    df = pd.read_csv(path_dlt40)
    df['mag'] = df['mag_binned'].astype(float)
    df['magerr'] = df['mag_err_binned'].astype(float)
    df['time'] = Time(df['jd_bin'], format='jd').mjd
    df['filter'] = 'Open'
    df['source'] = 'DLT40'
    df['source'] = 'DLT40'
    return df[['time', 'filter', 'mag', 'magerr', 'source']]

def load_goto(path_json):
    """
    Load GOTO forced photometry from JSON.
    Treating as 'L' band.
    """
    import json
    with open(path_json, 'r') as f:
        data = json.load(f)
    
    rows = []
    for entry in data:
        if entry.get('forced_mag') is not None:
            rows.append({
                'time': entry['mjd'],
                'filter': 'L',
                'magnitude': entry['forced_mag'],
                'error': entry['forced_mag_uncert'],
                'source': 'GOTO'
            })
    return pd.DataFrame(rows)

def load_imacs(path_txt):
    """
    Load IMACS photometry from text file.
    Format: JD mag magerr flt
    """
    df = pd.read_csv(path_txt, sep=r'\s+')
    # Filter valid errors (e.g. < 90)
    df = df[df['magerr'] < 90].copy()
    
    # JD -> MJD
    df['time'] = Time(df['JD'], format='jd').mjd
    df['magnitude'] = df['mag']
    df['error'] = df['magerr']
    df['filter'] = df['flt']
    df['source'] = 'IMACS'
    
    return df[['time', 'filter', 'magnitude', 'error', 'source']]

def load_mookodi(path_csv):
    """
    Load Mookodi forced-photometry lightcurve.
    Filters are SDSS g/r/i/z; uses mag_tot_err (photon + calibration noise).
    Multiple exposures per night per filter are binned to 0.3-day windows.
    """
    df = pd.read_csv(path_csv)
    # Keep only detections (mag < 99 and positive flux implied by finite mag)
    df = df[df['mag'] < 99].copy()
    # Use mag_tot_err (photon + calibration); drop original magerr to avoid duplicate column
    df = df[['mjd', 'filter', 'mag', 'mag_tot_err']].rename(columns={'mag_tot_err': 'magerr'})
    binned = bin_photometry(df, time_col='mjd', mag_col='mag', magerr_col='magerr', filter_col='filter')
    binned['source'] = 'Mookodi'
    binned = binned.rename(columns={'mjd': 'time', 'mag': 'magnitude', 'magerr': 'error'})
    return binned[['time', 'filter', 'magnitude', 'error', 'source']]

# ============== EXTINCTION (MW + host) ==============
def remove_extinction(lam, flam, EBV=0, R_v=3.1, z=0.0, LAW='MW'):
    """
    Return de-reddened flux and the monochromatic extinction A_lambda (mag).
    For MW/LMC/SMC, uses SNeSCOPE curves; for Calzetti and MW (ccm89) path, uses extinction.remove.
    """
    if EBV <= 0:
        return flam, np.zeros_like(np.atleast_1d(lam))

    lam = np.array(lam, dtype=float)
    if LAW == 'SMC':
        ex = SMC_Gordon03(np.unique(lam)*(1+z))
        ex.set_EBmV(EBV)
        A_v = ex.Av
        Alamm = A_v * ex.AlamAv
        Alam = np.zeros_like(lam)
        for i, l in enumerate(lam):
            Alam[i] = Alamm[np.unique(lam) == l]
        flux = flam * 10**(0.4 * Alam)

    elif LAW == 'LMC':
        ex = LMC_Gordon03(lam*(1+z))
        ex.set_EBmV(EBV)
        A_v = ex.Av
        Alam = A_v * ex.AlamAv
        flux = flam * 10**(0.4 * Alam)

    elif LAW == 'Cal00':
        A_v = R_v * EBV
        Alam_curve = calzetti00(lam, A_v, R_v, unit='aa')
        # extinction.remove expects A_lambda (mag) array
        flux = remove(Alam_curve, flam)
        Alam = Alam_curve

    else:
        # Default: MW ccm89
        A_v = R_v * EBV
        Alam_curve = ccm89(lam, A_v, R_v, unit='aa')
        flux = remove(Alam_curve, flam)
        Alam = Alam_curve

    return flux, Alam

from extinction import ccm89, remove
from astropy.io import ascii

def apply_extinction_corrections(
    lc,
    filter_transmission,      # dict: {filter_name: astropy Table} (use your loader)
    EBV_MW=0.043, Rv_MW=3.1,
    EBV_host=0.205, Rv_host=3.1,
):
    """
    Adds extinction-corrected columns to lc (mags and, if present, fluxes).

    New columns:
      - A_MW, A_host, A_total      [mag]
      - mag_extcorr                [mag]  = magnitude - A_total
      - flux_corr, fluxerr_corr    [same units as lc['flux']] (if flux columns exist)

    Also returns:
      - A_lambda_by_filter: {filter: median A_total for that filter} to correct
        raw non-detection tables in plotting.

    Notes:
      • Uses CCM89; host and MW are applied additively in magnitudes.
      • If lc['piv_wl'] is missing, we compute it from filter_transmission via zp_band_ab().
      • Does NOT change the original 'magnitude'/'flux' columns.
    """
    import numpy as np
    import pandas as pd
    from SNeSCOPE.PhotoUtils import zp_band_ab

    lc = lc.copy()

    # Ensure pivot wavelengths
    if 'piv_wl' not in lc.columns or lc['piv_wl'].isna().any():
        piv = []
        for f in lc['filter']:
            T = filter_transmission.get(f)
            if T is None:
                raise KeyError(f"No transmission curve for filter '{f}' to compute pivot wavelength.")
            _, pw = zp_band_ab(T)
            piv.append(pw)
        lc['piv_wl'] = np.array(piv, dtype=float)

    lam = lc['piv_wl'].to_numpy()  # Å

    # Compute A_lambda arrays (mag)
    if EBV_MW > 0:
        A_MW = ccm89(lam, Rv_MW * EBV_MW, Rv_MW, unit='aa')
    else:
        A_MW = np.zeros_like(lam)

    if EBV_host > 0:
        A_host = ccm89(lam, Rv_host * EBV_host, Rv_host, unit='aa')
    else:
        A_host = np.zeros_like(lam)

    A_tot = A_MW + A_host

    lc['A_MW']    = A_MW
    lc['A_host']  = A_host
    lc['A_total'] = A_tot

    # Magnitude correction
    if 'magnitude' in lc.columns:
        lc['mag_extcorr'] = lc['magnitude'] - lc['A_total']
        # Keep the same uncertainty; extinction uncertainty not propagated here
        if 'error' in lc.columns:
            lc['magerr_extcorr'] = lc['error']

    # Flux correction (only if present)
    if 'flux' in lc.columns:
        # remove() expects A_lambda in mag; apply MW then host
        flux_corr = remove(A_MW, lc['flux'].to_numpy())
        flux_corr = remove(A_host, flux_corr)
        lc['flux_corr'] = flux_corr

        if 'fluxerr' in lc.columns:
            # Scale the error by the same factor as the flux (dereddening is multiplicative)
            # factor = flux_corr / flux
            with np.errstate(divide='ignore', invalid='ignore'):
                scale = np.where(lc['flux'].to_numpy() != 0, flux_corr / lc['flux'].to_numpy(), 1.0)
            lc['fluxerr_corr'] = lc['fluxerr'].to_numpy() * scale

    # Per-filter median A_total for use on raw tables in plotting
    A_lambda_by_filter = (
        lc.groupby('filter')['A_total']
          .median()
          .to_dict()
    )

    return lc, A_lambda_by_filter

# ============== BUILD MASTER LC ==============
def build_master_lc():
    parts = [
        load_meerlicht(PATH_ML),
        load_blackgem(PATH_BG),
        load_andrea_txts(PATH_TELDIR),
        load_lco(PATH_LCO),
        load_REM_epoch(),
        load_efosc2_manual(),
        load_goto(PATH_GOTO),
        load_imacs(PATH_TELDIR / "2024vjm_IMACS-gri.txt"),
        load_mookodi(PATH_MKD),
    ]

    #Add first epoch REM
    lc = pd.concat(parts, ignore_index=True)
    lc = lc.sort_values('time').reset_index(drop=True)

    # pivot λ
    lc = add_pivot_wavelengths(lc)

    # flux columns from AB mags
    lc['f_nu']     = 10**(-0.4 * lc['magnitude'])
    lc['f_nu_err'] = magerr2fluxerr(lc['error'], lc['f_nu'])
    lc['flux']     = f_nu2f_lam(maggie2cgs(lc['f_nu']), lc['piv_wl'])
    lc['fluxerr']  = f_nu2f_lam(maggie2cgs(lc['f_nu_err']), lc['piv_wl'])
    lc['limit']    = ''  # keep schema
    return lc[['time','filter','magnitude','error','source','piv_wl','flux','fluxerr','limit']]

# ============== PLOT ================
def plot_lightcurve_mag(
    lc,
    t0,
    dm,
    colors,
    filter_offsets,
    outpath,
    bg_detections=None,
    bg_non_detections=None,
    ml_non_detections=None,
    rem_limits=None,
    DLT_40=None,
    A_map=None,                       # per-filter A_lambda (mag) lookup
    use_extinction_corrected=True,    # use lc['mag_extcorr'] if present
):
    """
    Plot measured photometry (binned + external) in magnitude space with offsets.

    - Main detections (from `lc`) are expected to already be extinction-corrected
      if `use_extinction_corrected` is True and those columns exist.
    - Unbinned BG detections and all non-detections are corrected *on the fly*
      using A_map (per-filter A_lambda, in magnitudes).
    - Non-detections (ML, BG, REM H/K limits) are plotted only for times <= t0.
    - For plotting, B V J H K are converted to Vega magnitudes; all others remain AB.
    """
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    raise RuntimeError("[rev3] original plotter disabled (this repository writes figures via nature_style.save)")  # os.makedirs(os.path.dirname(outpath), exist_ok=True)

    # Broken axis configuration
    # Left: -10 to 80 (90 days). Right: 150 to 220 (70 days).
    # Width ratio roughly 9:7 -> start with [3, 2] for aesthetics
    fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(12, 6),
                                   gridspec_kw={'width_ratios': [3, 2], 'wspace': 0.05})
    plt.rcParams.update({'font.size': 14, 'text.usetex': False})

    telescope_markers = {
        'MeerLICHT': 'o',
        'BlackGEM_fp': 'o',
        'BlackGEM': 'o',
        'REM,REMIR': 'h',
        'REM,REM-ROS2-DIF': 'h',
        'Prompt6,FLI': '*',
        'Prompt6,FLI-DIF': '*',
        'ESO-NTT,EFOSC': 'D',
        'LCO': 's',
        'DLT-40': 'P',
        'GOTO': '^',
        'IMACS': 'X',
        'Mookodi': 'p'
    }

    # Which columns to use from lc
    mag_col  = 'mag_extcorr'    if (use_extinction_corrected and 'mag_extcorr' in lc.columns) else 'magnitude'
    merr_col = 'magerr_extcorr' if (use_extinction_corrected and 'magerr_extcorr' in lc.columns) else 'error'

    # --- helpers ---
    # AB -> Vega for B,V,J,H,K when plotting
    VEGA_PLOT = {'B', 'V', 'J', 'H', 'K'}
    def to_plot_system(raw_filt, mags):
        """
        Convert AB -> Vega for B,V,J,H,K; otherwise return mags unchanged.
        `raw_filt` may be 'bg_B' etc.; we use base name for lookup.
        """
        base = raw_filt[3:] if isinstance(raw_filt, str) and raw_filt.startswith('bg_') else raw_filt
        if base in VEGA_PLOT:
            off = AB_MINUS_VEGA.get(base, 0.0)  # AB = Vega + off  => Vega = AB - off
            return np.asarray(mags, dtype=float) - off
        return np.asarray(mags, dtype=float)

    # extinction (mag) to subtract for raw filter name
    def A_of(raw_filt: str) -> float:
        if not A_map or not isinstance(raw_filt, str):
            return 0.0
        base = raw_filt[3:] if raw_filt.startswith('bg_') else raw_filt
        candidates = [raw_filt, raw_filt.strip(), base, f'bg_{base}',
                      base.upper(), base.lower(), f'bg_{base.upper()}', f'bg_{base.lower()}']
        for k in candidates:
            if k in A_map and np.isfinite(A_map[k]):
                return float(A_map[k])
        # fallback: median among same-base keys
        def collapse(k): return k[3:] if isinstance(k, str) and k.startswith('bg_') else k
        same = [A_map[k] for k in A_map.keys() if collapse(k).lower() == base.lower()]
        return float(np.median(same)) if same else 0.0

    # PLOTTING LOOP over the two axes
    for ax in [ax1, ax2]:
        # 1) Main detections (binned, already corrected if columns exist)
        if len(lc) > 0 and mag_col in lc.columns and merr_col in lc.columns:
            for filt in sorted(lc['filter'].unique()):
                fdata = lc[lc['filter'] == filt].copy()
                if 'source' in fdata.columns:
                    fdata = fdata[fdata['source'] != 'ghost_explosion']
                if len(fdata) == 0:
                    continue

                base = filt[3:] if str(filt).startswith('bg_') else filt
                if base not in colors:
                    continue
                color  = colors[base]
                offset = filter_offsets.get(filt, filter_offsets.get(base, 0.0))

                y_mag_plot = to_plot_system(filt, fdata[mag_col].to_numpy()) + offset

                for src in fdata['source'].unique():
                    sdata = fdata[fdata['source'] == src]
                    if len(sdata) == 0:
                        continue
                    # positions of sdata rows within fdata
                    pos = fdata.index.get_indexer(sdata.index)

                    if 'MeerLICHT' in str(src):
                        marker, fillstyle = telescope_markers['MeerLICHT'], 'none'
                    elif 'BlackGEM' in str(src):
                        marker, fillstyle = telescope_markers.get('BlackGEM_fp', 'o'), 'full'
                    elif 'LCO' in str(src):
                        marker, fillstyle = telescope_markers['LCO'], 'full'
                    elif 'REM,REMIR' in str(src):
                        marker, fillstyle = telescope_markers['REM,REMIR'], 'full'
                    elif 'REM,REM-ROS2-DIF' in str(src):
                        marker, fillstyle = telescope_markers['REM,REM-ROS2-DIF'], 'full'
                    elif 'Prompt6,FLI-DIF' in str(src):
                        marker, fillstyle = telescope_markers['Prompt6,FLI-DIF'], 'full'
                    elif 'Prompt6,FLI' in str(src):
                        marker, fillstyle = telescope_markers['Prompt6,FLI'], 'full'
                    elif 'ESO-NTT,EFOSC' in str(src):
                        marker, fillstyle = telescope_markers['ESO-NTT,EFOSC'], 'full'
                    elif 'GOTO' in str(src):
                        marker, fillstyle = telescope_markers['GOTO'], 'full'
                    elif 'IMACS' in str(src):
                        marker, fillstyle = telescope_markers['IMACS'], 'full'
                    elif 'Mookodi' in str(src):
                        marker, fillstyle = telescope_markers['Mookodi'], 'full'
                    else:
                        marker, fillstyle = 'o', 'full'

                    # compute y for this subset (keep the already converted array aligned)
                    ax.errorbar(
                        sdata['time'] - t0,
                        y_mag_plot[pos],
                        sdata[merr_col],
                        color=color,
                        marker=marker,
                        fillstyle=fillstyle,
                        ls='none',
                    )

        # 2) Unbinned BlackGEM detections (transparent); apply A_map then AB->Vega if needed
        if bg_detections is not None and len(bg_detections) > 0:
            need = {'FILTER', 'MJD-OBS', 'MAG_ZOGY'}
            if need.issubset(set(bg_detections.columns)):
                for filt in bg_detections['FILTER'].unique():
                    base = filt[3:] if str(filt).startswith('bg_') else filt
                    if base not in colors:
                        continue
                    color  = colors[base]
                    offset = filter_offsets.get(f'bg_{filt}', filter_offsets.get(base, 0.0))
                    sel    = (bg_detections['FILTER'] == filt)
                    tt     = bg_detections.loc[sel, 'MJD-OBS'].to_numpy()
                    mm     = bg_detections.loc[sel, 'MAG_ZOGY'].to_numpy().astype(float)
                    mm    -= A_of(filt)                      # extinction
                    mm      = to_plot_system(filt, mm)       # AB->Vega for BVJHK
                    ax.plot(tt - t0, mm + offset, color=color, alpha=0.10, marker='o', ls='none')

        # 3) MeerLICHT non-detections (ONLY epochs <= t0); apply A_map then AB->Vega if needed
        if ml_non_detections is not None and len(ml_non_detections) > 0:
            need = {'Filter', 'MJD', 'Mag_zogy'}
            if need.issubset(set(ml_non_detections.columns)):
                ml_nd = ml_non_detections[ml_non_detections['MJD'] <= t0]
                for filt in ml_nd['Filter'].unique():
                    base = filt[3:] if str(filt).startswith('bg_') else filt
                    if base not in colors:
                        continue
                    color  = colors[base]
                    offset = filter_offsets.get(f'bg_{filt}', filter_offsets.get(base, 0.0))
                    sel    = (ml_nd['Filter'] == filt)
                    tt     = ml_nd.loc[sel, 'MJD'].to_numpy()
                    mm     = ml_nd.loc[sel, 'Mag_zogy'].to_numpy().astype(float)
                    mm    -= A_of(filt)
                    mm      = to_plot_system(filt, mm)
                    ax.plot(tt - t0, mm + offset, color=color, alpha=0.50, marker='v', ls='none')

        # 4) BlackGEM non-detections (ONLY epochs <= t0); apply A_map then AB->Vega if needed
        if bg_non_detections is not None and len(bg_non_detections) > 0:
            need = {'FILTER', 'MJD-OBS', 'LIMMAG_ZOGY'}
            if need.issubset(set(bg_non_detections.columns)):
                bg_nd = bg_non_detections[bg_non_detections['MJD-OBS'] <= t0]
                for filt in bg_nd['FILTER'].unique():
                    base = filt[3:] if str(filt).startswith('bg_') else filt
                    if base not in colors:
                        continue
                    color  = colors[base]
                    offset = filter_offsets.get(f'bg_{filt}', filter_offsets.get(base, 0.0))
                    sel    = (bg_nd['FILTER'] == filt)
                    tt     = bg_nd.loc[sel, 'MJD-OBS'].to_numpy()
                    mm     = bg_nd.loc[sel, 'LIMMAG_ZOGY'].to_numpy().astype(float)
                    mm    -= A_of(filt)
                    mm      = to_plot_system(filt, mm)
                    ax.plot(tt - t0, mm + offset, color=color, alpha=0.50, marker='v', ls='none')

        # 5) REM H/K limits (ONLY epochs <= t0); apply A_map then AB->Vega
        if rem_limits is not None and len(rem_limits) > 0:
            need = {'time', 'filter', 'limit_mag'}
            if need.issubset(set(rem_limits.columns)):
                rem_nd = rem_limits[rem_limits['time'] <= t0]
                for f in rem_nd['filter'].unique():
                    base = f
                    if base not in colors:
                        continue
                    color  = colors[base]
                    offset = filter_offsets.get(base, 0.0)
                    sel    = (rem_nd['filter'] == f)
                    tt     = rem_nd.loc[sel, 'time'].to_numpy()
                    mm     = rem_nd.loc[sel, 'limit_mag'].to_numpy().astype(float)
                    mm    -= A_of(f)
                    mm      = to_plot_system(f, mm)
                    ax.plot(tt - t0, mm + offset, color=color, alpha=0.50, marker='v', ls='none')
        # 6) DLT-40
        if DLT_40 is not None and len(DLT_40) > 0:
            need = {'time', 'filter', 'mag', 'magerr', 'source'}
            if need.issubset(set(DLT_40.columns)):
                base = DLT_40['filter'].iloc[0]
                color  = colors[base]
                offset = filter_offsets.get(base, 0.0)
                sel    = (DLT_40['filter'] == base)
                tt     = DLT_40.loc[sel, 'time'].to_numpy()
                mm     = DLT_40.loc[sel, 'mag'].to_numpy().astype(float)
                mm    -= A_of('r')
                ax.plot(tt - t0, mm + offset, color=color, marker=telescope_markers['DLT-40'], ls='none')
        
        # Grid settings for both axes
        ax.grid(True, alpha=0.2)
    
    # Invert Y axis (do this once, as sharey=True means flipping one flips both. 
    # Doing it in the loop would flip it twice -> back to normal)
    ax1.invert_yaxis()

    # --- AXIS LIMITS & BROKEN EFFECT ---
    ax1.set_xlim(-10, 75)
    ax2.set_xlim(155, 330)

    # hide the spines between ax and ax2
    ax1.spines['right'].set_visible(False)
    ax2.spines['left'].set_visible(False)
    
    # tick settings
    ax1.yaxis.tick_left()
    ax1.tick_params(labelright=False)
    ax2.yaxis.tick_right()
    ax2.tick_params(labelright=False)  # we share y, so no labels needed on inner side? Actually sharey hides them on ax2 left. 
                                     # but we might want the right axis to show Dual-Y for Absolute Mag?
    
    # Start diagonal cut lines
    d = .015  # magnitude of diagonal lines
    kwargs = dict(transform=ax1.transAxes, color='k', clip_on=False)
    ax1.plot((1-d, 1+d), (-d, +d), **kwargs)        # top-left diagonal
    ax1.plot((1-d, 1+d), (1-d, 1+d), **kwargs)      # bottom-left diagonal

    kwargs.update(transform=ax2.transAxes)
    ax2.plot((-d, +d), (1-d, 1+d), **kwargs)        # top-right diagonal
    ax2.plot((-d, +d), (-d, +d), **kwargs)          # bottom-right diagonal

    # Labels
    # Shared X label?
    fig.text(0.5, 0.04, 'Days since B-band maximum', ha='center', fontsize=14)
    ax1.set_ylabel('Apparent magnitude + offset', fontsize=14)
    
    # Secondary Y axis for Absolute Magnitude (on ax2 right?)
    # Since they share Y, we can put the secondary axis on ax2 result
    secax_y2 = ax2.secondary_yaxis('right',
                                  functions=(lambda x: x - dm, lambda x: x + dm))
    secax_y2.set_ylabel('Absolute Magnitude', fontsize=14)

    filt_handles = []
    for base in ['u','B','g','V','q','r','i','z','J','H','K','Open','L']:
        if base in colors:
            off = filter_offsets.get(base, 0.0)
            filt_handles.append(Line2D([0],[0], color=colors[base], lw=2, label=f'{base} + {off:.1f}'))
    inst_handles = [
        Line2D([0],[0], marker='o', color='black', lw=0, markersize=8, label='BlackGEM'),
        Line2D([0],[0], marker='o', color='black', lw=0, fillstyle='none', markersize=8, label='MeerLICHT'),
        Line2D([0],[0], marker='h', color='black', lw=0, markersize=8, label='REM'),
        Line2D([0],[0], marker='^', color='black', lw=0, markersize=8, label='GOTO'),
        Line2D([0],[0], marker='s', color='black', lw=0, markersize=6, label='LCO'),
        Line2D([0],[0], marker='*', color='black', lw=0, markersize=6, label='Prompt6'),
        Line2D([0],[0], marker='D', color='black', lw=0, markersize=6, label='EFOSC'),
        Line2D([0],[0], marker='X', color='black', lw=0, markersize=6, label='Magellan/IMACS'),
        Line2D([0],[0], marker='P', color='black', lw=0, markersize=6, label='DLT-40'),
        Line2D([0],[0], marker='p', color='black', lw=0, markersize=6, label='Mookodi'),
        Line2D([0],[0], marker='v', color='black', alpha=0.5, lw=0, markersize=6, label='Non detection'),
    ]
    
    # Legend - putting it on the figure or one of the axes
    # We can anchor it to the top of the figure / ax1 like before
    # Note: bbox_to_anchor on subplots works slightly differently relative to figure coords?
    # Let's try placing it relative to the whole figure using fig.legend or just ax1.legend with correct coords.
    # Ax1 spans roughly 60% of width. 
    # Safest is ax1.legend with a high y anchor.
    
    ax1.legend(handles=(filt_handles + inst_handles),
              loc='upper center', bbox_to_anchor=(0.85, 1.28), # Shifted right to span gap?
              ncol=6, fontsize='small', frameon=False)

    plt.subplots_adjust(top=0.78, bottom=0.12, right=0.90, left=0.10)
    pass  # [rev3] disabled: fig.savefig(outpath, format='pdf', bbox_inches='tight', dpi=300)
    plt.show()
    plt.close(fig)

# ============== GHOST POINTS (rise anchor) ==============
def add_ghost_points(lc, t_exp=EXPLODE_T):
    ghosts=[]
    for f, sub in lc.groupby('filter'):
        piv = sub['piv_wl'].iloc[0]
        ghosts.append(dict(time=t_exp, filter=f, magnitude=30.0, error=5.0,
                           source='ghost_explosion', piv_wl=piv, flux=0.0, fluxerr=1e-18, limit='ghost'))
    return pd.concat([lc, pd.DataFrame(ghosts)], ignore_index=True).sort_values('time').reset_index(drop=True)

# ============== SAMPLING GRID ==============
def make_dates(lc, t_start=EXPLODE_T, dense_days=100, dense_n=100, regular_n=100):
    t_end = lc['time'].max()
    dense = np.linspace(t_start, t_start + dense_days, dense_n)
    reg   = np.linspace(t_start + dense_days, t_end, regular_n)
    return np.concatenate([dense, reg])

# ============== GP INTERPOLATION (FLUX-ONLY) ==============
def interpolate_flux_gp(
    lc, dates, kernel_length=2.0, min_alpha=1e-6, scale_factor=1e16,
    t0=T0_BMAX, explosion_time=EXPLODE_T,
    late_phase_start=10.0, sparse_tail_days_ref=3.0, sparse_tail_days_span=7.0,
    late_fixed_scale_min=10.0, late_scale_mult=7.5
):
    """Two-segment (early/late) GP interpolation in flux space with smooth blend.
       If extinction-corrected columns exist (flux_corr/fluxerr_corr), use them."""
    lc = lc.copy()
    if 'source' in lc.columns:
        # Exclude GOTO/DLT40 from interpolation as requested
        lc = lc[~lc['source'].isin(['GOTO', 'DLT40', 'DLT-40'])]
        lc = lc[~lc['filter'].isin(['bg_i'])]


    # --- choose columns (prefer extinction-corrected)
    flux_col    = 'flux_corr'    if 'flux_corr'    in lc.columns else 'flux'
    fluxerr_col = 'fluxerr_corr' if 'fluxerr_corr' in lc.columns else 'fluxerr'

    lc['phase'] = lc['time'] - t0
    dates = np.asarray(dates)
    phases_pred = dates - t0
    out = []

    def k_early(scale):
        return C(1.0, (1e-3, 10.0)) * Matern(length_scale=scale, length_scale_bounds=(2.0, 12.0), nu=2.5) \
               + WhiteKernel(noise_level=1e-3, noise_level_bounds=(1e-8, 1e-1))

    def k_late_fixed(scale):
        return C(1.0, (1e-3, 10.0)) * (RBF(length_scale=scale, length_scale_bounds="fixed") +
                                       DotProduct(sigma_0=0.3, sigma_0_bounds=(1e-3, 2.0))) \
               + WhiteKernel(noise_level=1e-4, noise_level_bounds="fixed")

    for filt, sub in lc.groupby('filter'):
        if sub[flux_col].isna().all():
            continue
        first_obs, last_obs = sub['time'].min(), sub['time'].max()
        use_mask = (dates >= explosion_time) & (dates <= last_obs)
        if not np.any(use_mask):
            continue

        X = (sub['time'].values - t0).reshape(-1,1)
        y = (sub[flux_col].values * scale_factor).astype(float)
        yerr = (sub[fluxerr_col].values * scale_factor).astype(float)
        is_ghost = (sub['source'] == 'ghost_explosion').values if 'source' in sub else np.zeros_like(y, bool)

        # jitter duplicate phases
        if np.unique(X).size < X.size:
            X = X + np.random.normal(0, 1e-6, size=X.shape)

        yerr = np.maximum(yerr, max(min_alpha, 0.05*np.nanmedian(yerr)))
        alpha = yerr**2
        alpha[is_ghost] *= 10.0

        early = (X.flatten() <= late_phase_start)
        late  = ~early
        if early.sum() < 3: early[:] = True
        if late.sum()  < 3: late[:]  = True

        X_late, y_late, a_late = X[late].copy(), y[late].copy(), alpha[late].copy()
        # cadence-aware inflation
        tlate = sub['time'].values[late]
        if tlate.size >= 3:
            ts = np.unique(np.sort(tlate))
            if ts.size >= 2:
                gaps = np.diff(ts); mids = (ts[:-1]+ts[1:])/2.0
                local_gap = np.interp(tlate, mids, gaps, left=gaps[0], right=gaps[-1])
                scale = 1.0 + (local_gap - sparse_tail_days_ref)/max(sparse_tail_days_span,1e-6)
                a_late *= np.clip(scale, 1.0, 1.8)

        kE = k_early(max(2.0, kernel_length*2.0))
        kL = k_late_fixed(max(late_fixed_scale_min, kernel_length*late_scale_mult))

        gpE = GaussianProcessRegressor(kernel=kE, alpha=alpha[early], n_restarts_optimizer=3,
                                       normalize_y=True, optimizer='fmin_l_bfgs_b', random_state=42)
        gpE.fit(X[early], y[early])

        gpL = GaussianProcessRegressor(kernel=kL, alpha=a_late, n_restarts_optimizer=0,
                                       normalize_y=False, optimizer='fmin_l_bfgs_b', random_state=43)
        gpL.fit(X_late, y_late)

        ph = phases_pred[use_mask].reshape(-1,1)
        muE, sE = gpE.predict(ph, return_std=True)
        muL, sL = gpL.predict(ph, return_std=True)

        w = 1.0/(1.0 + np.exp(-(ph.flatten()-late_phase_start)/1.0))
        mu  = (1-w)*muE + (w)*muL
        var = (1-w)**2*sE**2 + (w**2)*sL**2
        std = np.sqrt(var)

        out.append(pd.DataFrame({
            'time':dates[use_mask],
            'filter':filt,
            'piv_wl':sub['piv_wl'].iloc[0],
            'flux':mu/scale_factor,
            'fluxerr':std/scale_factor
        }))

    return pd.concat(out, ignore_index=True) if out else pd.DataFrame(columns=['time','filter','piv_wl','flux','fluxerr'])

# ============== GP PLOT ==============

def plot_gp_flux_with_offsets(interp_lc, lc, colors, gp_flux_offsets, outpath):
    """
    Plot GP-interpolated F_lambda with per-band offsets and shaded 1σ.
    The time axis is relative to the GP-inferred B-band peak.
    Uses extinction-corrected measured flux (flux_corr/fluxerr_corr) if present.
    """
    raise RuntimeError("[rev3] original plotter disabled (this repository writes figures via nature_style.save)")  # os.makedirs(os.path.dirname(outpath), exist_ok=True)

    # choose measured columns
    flux_col    = 'flux_corr'    if 'flux_corr'    in lc.columns else 'flux'
    fluxerr_col = 'fluxerr_corr' if 'fluxerr_corr' in lc.columns else 'fluxerr'

    def flux_to_mag(flux, piv_wl):
        c = 2.99792458e18
        fnu0 = 3.631e-20
        fnu = np.maximum(flux, 1e-99) * (piv_wl**2) / c
        return -2.5 * np.log10(fnu / fnu0)

    # get B-band peak time from GP result
    b = interp_lc[interp_lc['filter'] == 'B'].copy()
    peak_time = (interp_lc['time'].min() if b.empty
                 else b.sort_values('time').assign(mag=lambda x: flux_to_mag(x['flux'], x['piv_wl'].iloc[0]))
                         .loc[lambda x: x['mag'].idxmin(), 'time'])

    fig, ax = plt.subplots(figsize=(7.5, 7.5))
    plt.rcParams.update({'font.size': 12, 'text.usetex': False})

    telescope_markers = {
        'MeerLICHT': 'o', 'BlackGEM_fp': 'o', 'BlackGEM': 'o',
        'REM,REMIR': 'h', 'REM,REM-ROS2-DIF': 'h',
        'Prompt6,FLI': '*', 'Prompt6,FLI-DIF': '*',
        'ESO-NTT,EFOSC': 'D', 'LCO': 's', 'Mookodi': 'p'
    }

    for filt in np.unique(interp_lc['filter']):
        cond_interp = (interp_lc['filter'] == filt)
        cond_meas   = (lc['filter'] == filt)

        base = filt.replace('bg_', '') if filt.startswith('bg_') else filt
        if base not in colors:
            continue
        color = colors[base]
        gp_off = gp_flux_offsets.get(filt, gp_flux_offsets.get(base, 0.0)) * 1e-16

        # measured (extinction-corrected if available)
        if np.any(cond_meas):
            mdata = lc[cond_meas].copy()
            if 'source' in mdata.columns:
                mdata = mdata[mdata['source'] != 'ghost_explosion']
            if len(mdata) > 0:
                for src in mdata['source'].unique():
                    sdata = mdata[mdata['source'] == src]
                    if len(sdata) == 0:
                        continue
                    if 'MeerLICHT' in str(src):
                        marker, fillstyle = telescope_markers['MeerLICHT'], 'none'
                    elif 'BlackGEM' in str(src):
                        marker, fillstyle = telescope_markers.get('BlackGEM_fp', 'o'), 'full'
                    elif 'LCO' in str(src):
                        marker, fillstyle = telescope_markers['LCO'], 'full'
                    elif 'REM,REMIR' in str(src):
                        marker, fillstyle = telescope_markers['REM,REMIR'], 'full'
                    elif 'REM,REM-ROS2-DIF' in str(src):
                        marker, fillstyle = telescope_markers['REM,REM-ROS2-DIF'], 'full'
                    elif 'Prompt6,FLI-DIF' in str(src):
                        marker, fillstyle = telescope_markers['Prompt6,FLI-DIF'], 'full'
                    elif 'Prompt6,FLI' in str(src):
                        marker, fillstyle = telescope_markers['Prompt6,FLI'], 'full'
                    elif 'ESO-NTT,EFOSC' in str(src):
                        marker, fillstyle = telescope_markers['ESO-NTT,EFOSC'], 'full'
                    elif 'Mookodi' in str(src):
                        marker, fillstyle = telescope_markers['Mookodi'], 'full'
                    else:
                        marker, fillstyle = 'o', 'full'

                    ax.errorbar(
                        sdata['time'] - peak_time,
                        sdata[flux_col] + gp_off,
                        sdata[fluxerr_col],
                        color=color,
                        marker=marker,
                        fillstyle=fillstyle,
                        ls='none',
                        markersize=4,
                        alpha=0.8,
                        label=''
                    )

        # GP mean and 1σ
        if np.any(cond_interp):
            s = interp_lc[cond_interp].copy().sort_values('time')
            ax.fill_between(
                s['time'] - peak_time,
                s['flux'] + gp_off - s['fluxerr'],
                s['flux'] + gp_off + s['fluxerr'],
                alpha=0.2, color=color
            )
            ax.plot(s['time'] - peak_time, s['flux'] + gp_off,
                    color=color, alpha=0.6, linewidth=1)

    filt_handles = []
    for base in ['u','B','g','V','q','r','i','z','J','H','K']:
        if base in colors:
            off = gp_flux_offsets.get(base, 0.0)
            if off == 0:
                lab = f'{base}'
            elif off < 0:
                lab = f'{base} {off:.0f}'   # minus sign only
            else:
                lab = f'{base} +{off:.0f}'  # explicit plus for positives
            filt_handles.append(Line2D([0],[0], color=colors[base], lw=1.2, label=lab))

    inst_handles = [
        Line2D([0],[0], marker='o', color='black', lw=0, markersize=4, label='BlackGEM'),
        Line2D([0],[0], marker='o', color='black', lw=0, fillstyle='none', markersize=4, label='MeerLICHT'),
        Line2D([0],[0], marker='h', color='black', lw=0, markersize=4, label='REM'),
        Line2D([0],[0], marker='s', color='black', lw=0, markersize=3.5, label='LCO'),
        Line2D([0],[0], marker='*', color='black', lw=0, markersize=3.5, label='Prompt6'),
        Line2D([0],[0], marker='D', color='black', lw=0, markersize=3.5, label='EFOSC'),
        Line2D([0],[0], marker='p', color='black', lw=0, markersize=3.5, label='Mookodi'),
        Line2D([0],[0], color='none', lw=0, label='×10⁻¹⁶'),
    ]
    ax.legend(handles=(filt_handles + inst_handles),
            loc='upper center', bbox_to_anchor=(0.56, 1.15),
            ncol=6, fontsize='small', columnspacing=0.7,
            handlelength=0.84, handletextpad=0.3)

    ax.set_xlabel('Days since B-band GP peak')
    ax.set_ylabel(r'Flux ($\mathrm{erg\,s^{-1}\,cm^{-2}\,\AA^{-1}}$) + offset')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-10, 70)

    plt.subplots_adjust(top=0.88)
    pass  # [rev3] disabled: fig.savefig(outpath, format='pdf', bbox_inches='tight', dpi=300)
    plt.show()
    plt.close(fig)



# ============== STATS  ==============
def band_stats_from_gp(interp_lc, lc, dm=DM, dm_err=0.1, order=None):
    """
    Build per-band stats table from GP light curves.

    - Peak-time error = distance to nearest actual data point in that band.
    - Δm15 uncertainty includes GP mag uncertainties at t_peak and t_peak+15,
      plus a timing term using local slopes and σ_t.
    - Internally AB; prints B,V,J,H,K in Vega.

    Prints LaTeX lines like:
    band & t_peak±σ_t & m_peak±σ_m & M±σ_M & Δm15±σ_Δm15 \\
    """
    VEGA_OUT = {'B', 'V', 'J', 'H', 'K'}  # bands to print in Vega

    def pretty_name(f):
        return f"{f.split('_',1)[1]} (BG)" if isinstance(f, str) and f.startswith('bg_') else f

    # AB conversions
    def _flux_to_mag(flux, piv):
        c = 2.99792458e18
        fnu0 = 3.631e-20
        fnu = np.maximum(np.asarray(flux), 1e-99) * (piv**2) / c
        return -2.5 * np.log10(fnu / fnu0)

    def _fluxerr_to_magerr(flux, ferr):
        flux = np.maximum(np.asarray(flux), 1e-99)
        return (2.5/np.log(10)) * (np.asarray(ferr)/flux)

    # simple finite-difference slope helper on the GP grid
    def _local_slope(t_grid, m_grid, i):
        # central if possible, otherwise one-sided
        if i > 0 and i < len(t_grid)-1:
            dt = t_grid[i+1] - t_grid[i-1]
            if dt != 0:
                return (m_grid[i+1] - m_grid[i-1]) / dt
        if i == 0 and len(t_grid) > 1:
            dt = t_grid[1] - t_grid[0]
            if dt != 0:
                return (m_grid[1] - m_grid[0]) / dt
        if i == len(t_grid)-1 and len(t_grid) > 1:
            dt = t_grid[-1] - t_grid[-2]
            if dt != 0:
                return (m_grid[-1] - m_grid[-2]) / dt
        return 0.0

    results = []
    bands = sorted(interp_lc['filter'].unique()) if order is None else [b for b in order if b in set(interp_lc['filter'])]

    for f in bands:
        s = interp_lc[interp_lc['filter'] == f].copy()
        if s.empty:
            continue
        s = s.sort_values('time')
        tgp = s['time'].values
        piv = s['piv_wl'].iloc[0]

        # GP mags + errors (AB)
        m = _flux_to_mag(s['flux'].values, piv)
        merr = _fluxerr_to_magerr(s['flux'].values, s['fluxerr'].values)

        # peak (min mag)
        i_peak = int(np.argmin(m))
        t_peak = float(tgp[i_peak])
        m_peak = float(m[i_peak])
        m_peak_err = float(merr[i_peak])

        # Δm15 pieces
        t15 = t_peak + 15.0
        m15 = float(np.interp(t15, tgp, m))
        m15_err = float(np.interp(t15, tgp, merr))
        dm15 = m15 - m_peak


        # GP-based peak time uncertainty: find the first time after the peak where m(t)+merr(t) < m_peak-m_peak_err
        threshold = m_peak - m_peak_err
        t_err_gp = None
        for j in range(i_peak+1, len(m)):
            if (m[j] + merr[j]) < threshold:
                t_err_gp = tgp[j] - t_peak
                break
        # If not found, set to half the local grid spacing as a minimum
        if t_err_gp is None or t_err_gp <= 0:
            if i_peak < len(tgp)-1:
                t_err_gp = 0.5 * (tgp[i_peak+1] - tgp[i_peak])
            elif i_peak > 0:
                t_err_gp = 0.5 * (tgp[i_peak] - tgp[i_peak-1])
            else:
                t_err_gp = 0.05

        # Also compute the data-based uncertainty as before
        obs = lc[lc['filter'] == f].copy()
        if 'source' in obs.columns:
            obs = obs[obs['source'] != 'ghost_explosion']
        if len(obs):
            t_err_data = float(np.min(np.abs(obs['time'].values - t_peak)))
        else:
            # fallback: half local spacing around peak on GP grid
            if len(tgp) >= 2:
                if i_peak == 0:
                    dt_local = tgp[1] - tgp[0]
                elif i_peak == len(tgp)-1:
                    dt_local = tgp[-1] - tgp[-2]
                else:
                    dt_local = 0.5*((tgp[i_peak]-tgp[i_peak-1]) + (tgp[i_peak+1]-tgp[i_peak]))
                t_err_data = 0.5 * max(dt_local, 1e-6)
            else:
                t_err_data = 0.05

        # Combine the two uncertainties in quadrature
        t_err = float(np.sqrt(t_err_gp**2 + t_err_data**2))


        # slope term for timing error: [m'(t_peak+15)-m'(t_peak)] * σ_t
        # compute slope at t_peak index directly; at t_peak+15 use nearest grid index
        slope_peak = _local_slope(tgp, m, i_peak)
        i15 = int(np.clip(np.searchsorted(tgp, t15), 1, len(tgp)-1))
        if abs(tgp[i15] - t15) > abs(tgp[i15-1] - t15):
            i15 = i15 - 1
        slope_15 = _local_slope(tgp, m, i15)

        # GP-based dm15 time uncertainty: find the first time after t15 where m(t)+merr(t) < m15-m15_err
        threshold15 = m15 - m15_err
        t15_err_gp = None
        for j in range(i15+1, len(m)):
            if (m[j] + merr[j]) < threshold15:
                t15_err_gp = tgp[j] - t15
                break
        if t15_err_gp is None or t15_err_gp <= 0:
            if i15 < len(tgp)-1:
                t15_err_gp = 0.5 * (tgp[i15+1] - tgp[i15])
            elif i15 > 0:
                t15_err_gp = 0.5 * (tgp[i15] - tgp[i15-1])
            else:
                t15_err_gp = 0.05

        # Data-based uncertainty for t15: nearest observed point to t15
        if len(obs):
            t15_err_data = float(np.min(np.abs(obs['time'].values - t15)))
        else:
            if len(tgp) >= 2:
                if i15 == 0:
                    dt_local = tgp[1] - tgp[0]
                elif i15 == len(tgp)-1:
                    dt_local = tgp[-1] - tgp[-2]
                else:
                    dt_local = 0.5*((tgp[i15]-tgp[i15-1]) + (tgp[i15+1]-tgp[i15]))
                t15_err_data = 0.5 * max(dt_local, 1e-6)
            else:
                t15_err_data = 0.05

        # Combine in quadrature
        t15_err = float(np.sqrt(t15_err_gp**2 + t15_err_data**2))

        dm15_time_term = (slope_15 - slope_peak) * t15_err

        dm15_err = float(np.sqrt(m_peak_err**2 + m15_err**2 + dm15_time_term**2))

        # Absolute magnitude and uncertainty (AB)
        M_ab = m_peak - dm
        M_ab_err = float(np.hypot(m_peak_err, dm_err))

        # Convert to Vega for selected bands (printing only)
        m_out = m_peak
        M_out = M_ab
        if f in VEGA_OUT:
            off = AB_MINUS_VEGA.get(f, 0.0)  # AB = Vega + off  -> Vega = AB - off
            m_out = m_peak - off
            M_out = M_ab   - off

        results.append(dict(
            filt=f,
            pretty=pretty_name(f),
            t_peak=t_peak, t_err=t_err,
            m_peak_AB=m_peak, m_err=m_peak_err,
            M_AB=M_ab, M_err=M_ab_err,
            m_peak_print=m_out, M_print=M_out,
            dm15=dm15, dm15_err=dm15_err
        ))

    # Print LaTeX lines with Δm15 uncertainties included
    for r in results:
        print(f"{r['pretty']} & {r['t_peak']:.2f}$\\pm${r['t_err']:.2f} & "
              f"${r['m_peak_print']:.2f}\\pm{r['m_err']:.2f}$ & "
              f"${r['M_print']:.2f} \\pm {r['M_err']:.2f}$ & "
              f"${r['dm15']:.2f}\\pm{r['dm15_err']:.2f}$\\\\")
    return results
# ============== COLORS (common support; no extrapolation) ==============
def color_curve(interp_lc, lc_orig, f1, f2, t0=T0_BMAX):
    """Return dict with time (days since t0), color, color_err using only overlapping support."""
    d1 = interp_lc[interp_lc['filter']==f1].copy()
    d2 = interp_lc[interp_lc['filter']==f2].copy()
    if d1.empty or d2.empty:
        return None

    # start at later of first real detections (exclude ghost)
    real1 = lc_orig[(lc_orig['filter']==f1) & (lc_orig['source']!='ghost_explosion')]
    real2 = lc_orig[(lc_orig['filter']==f2) & (lc_orig['source']!='ghost_explosion')]
    if real1.empty or real2.empty: return None
    tstart = max(real1['time'].min(), real2['time'].min())
    d1, d2 = d1[d1['time']>=tstart], d2[d2['time']>=tstart]
    if d1.empty or d2.empty: return None

    # magnitudes + errors (AB)
    piv1, piv2 = d1['piv_wl'].iloc[0], d2['piv_wl'].iloc[0]
    m1 = flux_to_mag(d1['flux'].values, piv1); s1 = fluxerr_to_magerr(d1['flux'].values, d1['fluxerr'].values)
    m2 = flux_to_mag(d2['flux'].values, piv2); s2 = fluxerr_to_magerr(d2['flux'].values, d2['fluxerr'].values)

    # common timeline (no extrapolation)
    t1, t2 = d1['time'].values, d2['time'].values
    # Strict common range
    tmin = max(np.min(t1), np.min(t2))
    tmax = min(np.max(t1), np.max(t2))
    if not np.isfinite(tmin) or not np.isfinite(tmax) or tmax <= tmin:
        return None

    # Build times in the *closed* interval and clamp to [tmin, tmax]
    if len(np.intersect1d(np.round(t1, 8), np.round(t2, 8))) >= 3:
        times = np.intersect1d(np.round(t1, 8), np.round(t2, 8)).astype(float)
    else:
        n = max(min(len(t1), len(t2)), 8)
        times = np.linspace(tmin, tmax, num=n)

    # Tiny numerical safety: ensure no value falls outside due to fp rounding
    times = np.clip(times, tmin, tmax)

    # Interpolants that DO NOT error at boundaries (they clamp to edge values)
    f1m = interp1d(t1, m1, kind='linear', bounds_error=False,
                fill_value=(m1[0], m1[-1]), assume_sorted=False)
    f2m = interp1d(t2, m2, kind='linear', bounds_error=False,
                fill_value=(m2[0], m2[-1]), assume_sorted=False)
    f1s = interp1d(t1, s1, kind='linear', bounds_error=False,
                fill_value=(s1[0], s1[-1]), assume_sorted=False)
    f2s = interp1d(t2, s2, kind='linear', bounds_error=False,
                fill_value=(s2[0], s2[-1]), assume_sorted=False)

    mag1, mag2 = f1m(times), f2m(times)
    err1, err2 = f1s(times), f2s(times)

    color = mag1 - mag2
    color_err = np.sqrt(err1**2 + err2**2)

    # optional quality mask (drop very uncertain epochs)
    rel1 = err1/np.maximum(np.abs(mag1),1e-6)
    rel2 = err2/np.maximum(np.abs(mag2),1e-6)
    keep = (rel1<0.5) & (rel2<0.5)

    return {
        'time': times[keep] - t0,
        'color': color[keep],
        'color_err': color_err[keep],
        'filter1': f1, 'filter2': f2
    }

# ============== PEAK + Δm15 (B band) ==============
def bband_peak_and_dm15(interp_lc):
    """
    Compute B-band peak mag and Δm15 from the GP interpolation.
    Assumes interp_lc is already from extinction-corrected flux (because the GP
    was fit to flux_corr if available).
    """
    b = interp_lc[interp_lc['filter']=='B'].copy()
    if b.empty:
        return None
    b = b.sort_values('time')
    piv = b['piv_wl'].iloc[0]

    # flux -> AB mag
    m  = flux_to_mag(b['flux'].values, piv)
    me = fluxerr_to_magerr(b['flux'].values, b['fluxerr'].values)

    peak_idx = np.argmin(m)
    t_peak   = float(b['time'].iloc[peak_idx])
    m_peak   = float(m[peak_idx])
    m_peak_e = float(me[peak_idx])

    # Δm15
    t15      = t_peak + 15.0
    m15      = np.interp(t15, b['time'].values, m)
    dm15     = float(m15 - m_peak)

    return dict(peak_time=t_peak, peak_mag=m_peak, peak_mag_err=m_peak_e, dm15=dm15)

# ============== BOLOMETRIC (band-summed F_lambda) ==============
def bolometric_from_interp(interp_lc, dates, dist_pc=DIST_PC):
    """
    Integrate band fluxes (F_lambda) over wavelength at each date and propagate
    *statistical* errors (photometry/GP/band integration). Distance uncertainty
    is intentionally NOT included here—treat it later as a global systematic.

    Parameters
    ----------
    interp_lc : DataFrame with columns ['time','piv_wl','flux','fluxerr','filter']
    dates     : array of times to evaluate (must match interp_lc 'time' entries)
    dist_pc   : distance in parsec

    Returns
    -------
    L         : bolometric luminosity (erg s^-1)
    L_staterr : 1σ statistical uncertainty per point (same units as L)
    """
    import numpy as np
    from astropy import units as u

    L         = np.full_like(dates, np.nan, dtype=float)
    L_staterr = np.full_like(dates, np.nan, dtype=float)

    four_pi_d2 = 4*np.pi*(dist_pc * u.pc.to("cm"))**2

    for i, t in enumerate(dates):
        band = (interp_lc[interp_lc['time'] == t]
                .sort_values('piv_wl'))
        if band.empty:
            continue
        band = band[band['piv_wl']>4000]  # ignore UV bands below 4000 Å
        band = band[band['time'] < EXPLODE_T+56.5]  # ignore non-detections / negative fluxes

        wl = band['piv_wl'].to_numpy(float)   # Angstrom
        f  = band['flux'].to_numpy(float)     # erg s^-1 cm^-2 Å^-1
        s  = band['fluxerr'].to_numpy(float)

        N = len(wl)
        if N < 2:
            # need at least two bands to integrate over λ
            continue

        # Trapezoid rule *point* weights for ∫ f(λ) dλ
        w = np.zeros(N, dtype=float)
        w[0]     = 0.5*(wl[1]     - wl[0])
        w[1:-1]  = 0.5*(wl[2:]    - wl[:-2])
        w[-1]    = 0.5*(wl[-1]    - wl[-2])

        # Flux and variance (assuming band errors are independent)
        F       = np.sum(w * f)
        var_F   = np.sum((w**2) * (s**2))
        F_err   = np.sqrt(max(var_F, 0.0))

        # Scale to luminosity
        L[i]         = F     * four_pi_d2
        L_staterr[i] = F_err * four_pi_d2

    return L, L_staterr

def i_band_to_Lbol_points(i_df, dm=DM, BC_i=0.30, t0_bmax=T0_BMAX,
                          Mbol_sun=4.74, L_sun=3.828e33,
                          time_col="time", mag_col="mag_extcorr", magerr_col="error",
                          tmin_days_since_bmax=45.0):
    """
    Convert i-band magnitudes to pseudo-bolometric luminosity points using a BC.

    Returns a DataFrame with:
      phase (days since Bmax), Lbol, Lbol_err
    """
    import numpy as np
    import pandas as pd

    d = i_df.copy()

    # choose magnitude column (fallback if extcorr not present)
    if mag_col not in d.columns:
        mag_col = "magnitude"
    if magerr_col not in d.columns:
        magerr_col = "error"

    d["phase"] = d[time_col] - t0_bmax
    d = d[np.isfinite(d["phase"]) & (d["phase"] >= tmin_days_since_bmax)].copy()
    if d.empty:
        return pd.DataFrame(columns=["phase", "Lbol", "Lbol_err"])

    m_i = d[mag_col].astype(float).to_numpy()
    s_m = d[magerr_col].astype(float).to_numpy()

    # M_i, M_bol
    M_i   = m_i - float(dm)
    M_bol = M_i + float(BC_i)

    # Lbol from Mbol
    Lbol = L_sun * 10.0**(-0.4 * (M_bol - Mbol_sun))

    # propagate mag error: σ_L/L = (ln10/2.5) σ_M  (BC treated fixed)
    frac = (np.log(10.0)/2.5) * s_m
    Lerr = np.abs(Lbol) * frac

    out = d.copy()
    out["Lbol"] = Lbol
    out["Lbol_err"] = Lerr
    return out[["phase", "Lbol", "Lbol_err"]]


def dm_frac_sigma(dm_err_mag):
    """
    Convert distance-modulus uncertainty (mag) into a fractional luminosity
    (or Ni mass) systematic:  σ_L / L  = (2 ln 10 / 5) * σ_μ
    """
    import numpy as np
    return (2.0*np.log(10.0)/5.0) * float(dm_err_mag)

# ============== PLOTTING HELPERS (minimal) ==============
def plot_colors(curve, label, color, outpath):
    if curve is None: return
    t = curve['time']; c = curve['color']; e = curve['color_err']
    plt.figure(figsize=(7.5,5.5))
    plt.plot(t, c, color=color, lw=2, label=label)
    plt.fill_between(t, c-e, c+e, color=color, alpha=0.3)
    plt.xlabel('Days since B-band peak'); plt.ylabel(f'{label} (mag)')
    plt.grid(True, alpha=0.3); plt.xlim(-10, 60); plt.legend()
    pass  # [rev3] disabled: plt.savefig(outpath, bbox_inches='tight', dpi=300)

# ===================== MAIN FLOW =====================
