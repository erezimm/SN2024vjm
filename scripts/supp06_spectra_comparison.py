"""Supplementary Fig. 6 -- SN 2024vjm vs SN 2008ha (optical) and SN 2010ae (NIR).

Plain-python extraction of "spectra comparison.ipynb"
(cells 0-4, 7, 11, 12, 14; cell 14 produced
figures/SN2024vjm_vs_SN2010ae_SN2008ha_compare.pdf).  Exploratory cells 5, 6,
8, 9, 10, 13 (quick-look plots, SN 2019gsc) are not needed and omitted.
Edited for revision 3:
  * output written to outputs/figures.
  * lines broken (NaN) at wavelength gaps (telluric cut-outs 13,450-14,200 and
    18,100-19,356 A in the SN 2024vjm NIR spectra) instead of straight lines.
  * "SN 2024vjm", "SN 2008ha", "SN 2010ae" with a space.
  * SN 2024vjm phases from Supp. Table 1 (B max MJD 60572.29) for the files
    actually plotted:  EFOSC2 2024-09-15 -4.26 d; EFOSC2 Gr11+Gr16 2024-10-03
    +13.79 d; FLOYDS 2024-10-29 +40.13 d; EFOSC2 2024-11-29 +70.76 d;
    X-shooter NIR -2.25 d; SOAR 2024-09-28 +8.71 d; SOAR 2024-10-10 +20.73 d.
    Comparison-SN phases kept as in the notebook.
  * Nature style (Arial, 5.5-6.5 pt, 0.5 pt lines), panel letters a, b.
Data processing (binning, dereddening, normalization, offsets) unchanged.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.io import ascii
from astropy.time import Time
from extinction import ccm89, remove

import nature_style as ns
ns.apply()

import paths
SPEC = str(paths.SPECTRA)
COMP = str(paths.COMPARISON)

meta = pd.read_csv(f'{SPEC}/meta.csv', index_col=False)
meta['mjd'] = meta['utc_obs_date'].apply(lambda x: float(Time(x).mjd))

EBV_MW, RV_MW, LAW_MW = 0.043, 3.1, 'MW'
EBV_HOST, RV_HOST, LAW_HOST = 0.205, 3.1, 'MW'
Z_HOST = 0.0028


def bin_spectrum(wavelength, flux, bin_factor):
    n_bins = len(flux) // bin_factor
    bw = np.mean(np.asarray(wavelength)[:n_bins * bin_factor].reshape(-1, bin_factor), axis=1)
    bf = np.mean(np.asarray(flux)[:n_bins * bin_factor].reshape(-1, bin_factor), axis=1)
    return bw, bf


def remove_extinction(lam, flam, EBV=0, R_v=3.1, z=0.0, LAW='MW'):
    if EBV <= 0:
        return flam, np.zeros_like(np.atleast_1d(lam))
    lam = np.array(lam, dtype=float)
    A_v = R_v * EBV
    Alam = ccm89(lam, A_v, R_v, unit='aa')          # LAW='MW' path of original
    return remove(Alam, np.asarray(flam, float)), Alam


def deredden(wl, fl):
    fl, _ = remove_extinction(wl, fl, EBV=EBV_MW, R_v=RV_MW, z=0.0, LAW=LAW_MW)
    fl, _ = remove_extinction(wl, fl, EBV=EBV_HOST, R_v=RV_HOST, z=Z_HOST, LAW=LAW_HOST)
    return fl


def to_float(col):
    return np.array([np.nan if f == '--' else f for f in col], dtype=float)


# ---------------- cell 3: SN 2010ae NIR ----------------
def read_2010ae(fn, nbin):
    t = ascii.read(f'{COMP}/SN2010ae/{fn}')
    w, f = bin_spectrum(t['col1'], t['col2'], nbin)
    m = w > 10000
    return w[m], f[m]


SN2010ae_wl_a, SN2010_fl_a = read_2010ae('2010ae_2010-02-24_00-00-00_VLT-UT2_X-Shooter_None.txt', 30)
SN2010ae_wl_b, SN2010_fl_b = read_2010ae('2010ae_2010-03-06_00-00-00_VLT-UT2_X-Shooter_None.txt', 20)
SN2010ae_wl_c, SN2010_fl_c = read_2010ae('2010ae_2010-03-13_00-00-00_VLT-UT2_X-Shooter_None.txt', 20)

# ---------------- cell 4: SN 2024vjm NIR ----------------
t = ascii.read(f'{SPEC}/SN2024vjm_clipped_NIR_x-shooter_spectrum.txt')   # X-shooter, -2.25 d
SN2024vjm_wl_a, SN2024vjm_fl_a = bin_spectrum(t['WAVE'], t['FLUX'], 5)
m = (SN2024vjm_wl_a > 10000) & (SN2024vjm_wl_a < 22000)
SN2024vjm_wl_a, SN2024vjm_fl_a = SN2024vjm_wl_a[m], SN2024vjm_fl_a[m]

t = ascii.read(f'{SPEC}/SN2024vjm_0928_clipped_spectrum.txt')            # SOAR, +8.71 d
SN2024vjm_wl_b, SN2024vjm_fl_b = bin_spectrum(t['WAVE'], t['FLUX'], 3)
m = SN2024vjm_wl_b > 10000
SN2024vjm_wl_b, SN2024vjm_fl_b = SN2024vjm_wl_b[m], SN2024vjm_fl_b[m]

t = ascii.read(f'{SPEC}/SN2024vjm_1010_clipped_spectrum.txt')            # [rev3] SOAR 2024-10-10 UT, +20.73 d (author: intended epoch)
SN2024vjm_wl_c, SN2024vjm_fl_c = bin_spectrum(t['WAVE'], t['FLUX'], 3)
m = SN2024vjm_wl_c > 10000
SN2024vjm_wl_c, SN2024vjm_fl_c = SN2024vjm_wl_c[m], SN2024vjm_fl_c[m]

# ---------------- cell 7 + 11: SN 2024vjm optical ----------------
spec = ascii.read(f'{SPEC}/AT2024vjm_2024-09-15_00-40-56_ESO-NTT_EFOSC2-NTT_TNS.dat')   # -4.26 d
wl_2024vjm_a = np.array(spec['col1'], float)
fl_2024vjm_a = deredden(wl_2024vjm_a, spec['col2'])

s1 = ascii.read(f'{SPEC}/epessto/SN2024vjm_2024-10-03_01-58-23.182_ESO-NTT_EFOSC2-NTT_ePESSTOp.asci', comment='#')  # Gr16 +13.79
s2 = ascii.read(f'{SPEC}/epessto/SN2024vjm_2024-10-03_01-05-07.824_ESO-NTT_EFOSC2-NTT_ePESSTOp.asci', comment='#')  # Gr11 +13.76
wl_2024vjm_b = np.concatenate([s1['col1'], s2['col1']])
fl_2024vjm_b = np.concatenate([s1['col2'], s2['col2']])
o = np.argsort(wl_2024vjm_b)
wl_2024vjm_b = np.asarray(wl_2024vjm_b[o], float)
fl_2024vjm_b = deredden(wl_2024vjm_b, to_float(fl_2024vjm_b[o]))

spec = ascii.read(f'{SPEC}/SN2024vjm_20241029_redblu_100931.254.ascii.txt', comment='#')   # FLOYDS +40.13
wl_2024vjm_c = np.array(spec['col1'], float)
fl_2024vjm_c = deredden(wl_2024vjm_c, to_float(spec['col2']))

spec = ascii.read(f'{SPEC}/epessto/SN2024vjm_2024-11-29_01-13-41.624_ESO-NTT_EFOSC2-NTT_ePESSTOp.asci', comment='#')  # +70.76
wl_2024vjm_d = np.array(spec['col1'], float)
fl_2024vjm_d = deredden(wl_2024vjm_d, to_float(spec['col2']))

# ---------------- cell 12: SN 2008ha ----------------
def read_2008ha(fn, cut=True):
    t = ascii.read(f'{COMP}/SN2008ha/{fn}')
    w, f = np.array(t['col1'], float), np.array(t['col2'], float)
    if cut:
        m = w < 9000
        w, f = w[m], f[m]
    return w, f


SN2008ha_wl_a, SN2008ha_fl_a = read_2008ha('SN2008ha_2008-11-11_00-00-00_Other_Other_None.ascii', cut=False)
SN2008ha_wl_b, SN2008ha_fl_b = read_2008ha('SN2008ha_2008-11-27_00-00-00_Ekar_AFOSC_None.ascii')
SN2008ha_wl_c, SN2008ha_fl_c = read_2008ha('SN2008ha_2008-12-21_00-00-00_TNG_DOLORES_None.ascii')
k = np.argmax(SN2008ha_fl_c)                     # remove the max-value spike (as original)
SN2008ha_wl_c, SN2008ha_fl_c = np.delete(SN2008ha_wl_c, k), np.delete(SN2008ha_fl_c, k)
SN2008ha_wl_d, SN2008ha_fl_d = read_2008ha('SN2008ha_2009-01-19_00-00-00_TNG_DOLORES_None.ascii')


# ---------------- cell 14: figure ----------------
def gapped(x, y, factor=10):
    """Insert NaN where the wavelength step exceeds factor x the median step."""
    x = np.asarray(x, float); y = np.asarray(y, float)
    gi = np.where(np.diff(x) > factor * np.nanmedian(np.diff(x)))[0]
    return np.insert(x, gi + 1, np.nan), np.insert(y, gi + 1, np.nan)


def P(ax, x, y, **kw):
    ax.plot(*gapped(x, y), **kw)


B_peak = 60572.29
LW = 0.5
BLK = dict(color='black', lw=LW)
BLU = dict(color='#1f3fff', lw=LW, alpha=0.7)
RED = dict(color='#e8000b', lw=LW, alpha=0.7)
FS = ns.FS_SMALL
V24, V08, V10 = 'SN 2024vjm', 'SN 2008ha', 'SN 2010ae'
cB, cR = '#1f3fff', '#e8000b'

fig, axes = plt.subplots(2, 1, figsize=(ns.DOUBLE_COL, 175 * ns.MM))

# ---- a: optical, SN 2008ha ----
ax = axes[0]
offset = 3
za, zh = 1 + 0.0028, 1 + 0.004647
P(ax, wl_2024vjm_a / za, fl_2024vjm_a / np.nanmax(fl_2024vjm_a) + offset - 0.25, **BLK)
P(ax, SN2008ha_wl_a / zh, SN2008ha_fl_a / np.nanmax(SN2008ha_fl_a) + offset, **BLU)
ax.text(5500, 0.8 + offset, f'{V24} −4.26 d', fontsize=FS, color='black')
ax.text(5500, 0.55 + offset, f'{V08} −1.23 d', fontsize=FS, color=cB)

P(ax, wl_2024vjm_b / za, fl_2024vjm_b / np.nanmax(fl_2024vjm_b) + offset / 2 - 0.1, **BLK)
P(ax, SN2008ha_wl_b / zh, SN2008ha_fl_b / np.nanmax(SN2008ha_fl_b) + offset / 2, **BLU)
ax.text(5500, 1.05 + offset / 2, f'{V24} +13.79 d', fontsize=FS, color='black')
ax.text(5500, 0.8 + offset / 2, f'{V08} +15.2 d', fontsize=FS, color=cB)

P(ax, wl_2024vjm_c / za, fl_2024vjm_c / np.nanmax(fl_2024vjm_c), **BLK)
P(ax, SN2008ha_wl_c / zh, SN2008ha_fl_c / np.nanmax(SN2008ha_fl_c), **BLU)
ax.text(5500, 0.8, f'{V24} +40.13 d', fontsize=FS, color='black')
ax.text(5500, 0.55, f'{V08} +39 d', fontsize=FS, color=cB)

P(ax, wl_2024vjm_d / za, fl_2024vjm_d / np.nanmax(fl_2024vjm_d) - offset / 2, **BLK)
P(ax, SN2008ha_wl_d / zh, SN2008ha_fl_d / np.nanmax(SN2008ha_fl_d) - offset / 2, **BLU)
ax.text(5500, -0.9, f'{V24} +70.76 d', fontsize=FS, color='black')
ax.text(5500, -1.15, f'{V08} +69.2 d', fontsize=FS, color=cB)

ax.set_xlim(3000, 10000)
ax.set_ylim(-2, 4.3)

# ---- b: NIR, SN 2010ae ----
ax = axes[1]
offset = -1
za, ze = 1 + 0.0028, 1 + 0.0037
P(ax, SN2024vjm_wl_a / za, SN2024vjm_fl_a / np.nanmax(SN2024vjm_fl_a), **BLK)
P(ax, SN2010ae_wl_a / ze, SN2010_fl_a / np.nanmax(SN2010_fl_a), **RED)
ax.text(16000, 0.6, f'{V24} −2.25 d', fontsize=FS, color='black')
ax.text(16000, 0.45, f'{V10} −1 d', fontsize=FS, color=cR)

P(ax, SN2024vjm_wl_b / za, SN2024vjm_fl_b / np.nanmax(SN2024vjm_fl_b) + offset, **BLK)
P(ax, SN2010ae_wl_b / ze, SN2010_fl_b / np.nanmax(SN2010_fl_b) + offset - 0.1, **RED)
ax.text(16000, 0.6 + offset, f'{V24} +8.71 d', fontsize=FS, color='black')
ax.text(16000, 0.45 + offset, f'{V10} +9 d', fontsize=FS, color=cR)

P(ax, SN2024vjm_wl_c / za, SN2024vjm_fl_c / np.nanmax(SN2024vjm_fl_c) + 2 * offset, **BLK)
P(ax, SN2010ae_wl_c / ze, SN2010_fl_c / np.nanmax(SN2010_fl_c) + 2 * offset, **RED)
ax.text(16000, 0.6 + 2 * offset, f'{V24} +20.73 d', fontsize=FS, color='black')
ax.text(16000, 0.45 + 2 * offset, f'{V10} +18 d', fontsize=FS, color=cR)

ax.set_xlim(9900, 25000)
ax.set_xlabel('Rest wavelength (Å)')

for ax, letter in zip(axes, 'ab'):
    ax.set_yticks([])
    ax.tick_params(axis='y', which='both', left=False)
    ax.minorticks_on()
    ax.set_ylabel('Normalized flux + offset')
    ax.text(-0.045, 1.0, letter, transform=ax.transAxes, fontsize=ns.FS_PANEL,
            fontweight='bold', va='top', ha='right')
axes[0].set_xlabel('Rest wavelength (Å)')

fig.tight_layout(h_pad=1.0)
ns.save(fig, 'supplementary/SuppFig6_spectra_comparison')
