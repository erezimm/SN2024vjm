"""Supplementary Fig. 4 -- optical spectral sequence of SN 2024vjm.

Copy of scripts/spectra_plot.py (Sep 8 2026), which
produced figures/optical_spectra_2024vjm.pdf (identical to Spectra/spectra_plot.py
apart from path handling).  Edited for revision 3:
  * B-band maximum MJD 60572.29 (was 60572.18); phases printed with 2 decimals
    and a true minus sign, matching Supplementary Table 1.
  * x axis trimmed to the data (3000-10,300 A, was 3000-12,000 A);
    phase/instrument labels moved to a column right of the axes.
  * Nature style at final print size (180 mm wide), Arial 5-6.5 pt,
    0.4 pt spectra lines; line IDs above the frame.
  * Lines broken at wavelength gaps (e.g. MUSE Na laser notch) instead of
    drawing a straight segment across the gap.
  * Output -> outputs/figures/supplementary only (original savefig removed).
Data are only read.  Spectral processing is unchanged (no dereddening is
applied in the original either -- deredden_mw_plus_host() is defined but never
called; kept as in the original).
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
from astropy.io import ascii
from astropy.time import Time

import nature_style as ns
ns.apply()
from extinction import ccm89, remove

# [rev3] dereddening (author: the caption says "corrected for extinction", so apply it).
# Identical to the original's deredden_mw_plus_host() (LAW='MW' path) and to Supp. Fig. 6.
EBV_MW, RV_MW = 0.043, 3.1
EBV_HOST, RV_HOST = 0.205, 3.1


def deredden(lam, flam):
    for ebv, rv in ((EBV_MW, RV_MW), (EBV_HOST, RV_HOST)):
        flam = remove(ccm89(np.asarray(lam, float), rv * ebv, rv, unit='aa'), np.asarray(flam, float))
    return flam

import paths
PATH_SPECTRA = paths.SPECTRA

z = 0.0028
c = 299792.458
B_peak = 60572.29          # B-band maximum (MJD), revision 3

meta = pd.read_csv(PATH_SPECTRA / 'meta.csv', index_col=False)
meta['mjd'] = meta['utc_obs_date'].apply(lambda x: float(Time(x).mjd))


def fmt_phase(p):
    return f"{p:+.2f}".replace('-', '−') + ' d'


def fmt_inst(instrument, grating):
    s = instrument.replace('/ ', '/').replace('X-Shooter', 'X-shooter')
    if 'EFOSC2' in s and pd.notnull(grating):
        s += ' ' + str(grating).capitalize()
    return s


def break_gaps(x, y, factor=20):
    """Insert NaN where the wavelength step is > factor x the median step."""
    dx = np.diff(x)
    gaps = np.where(dx > factor * np.nanmedian(dx))[0]
    if len(gaps) == 0:
        return x, y
    return (np.insert(x.astype(float), gaps + 1, np.nan),
            np.insert(y.astype(float), gaps + 1, np.nan))


meta_sorted = meta.sort_values('mjd', ascending=False).reset_index(drop=True)


def _xsh_max():
    m = 0.0
    for _, r in meta[meta['instrument'] == 'VLT/ X-Shooter'].iterrows():
        f = r['file_name']
        if 'NIR' in f:
            continue
        sp = ascii.read(str(PATH_SPECTRA / f))
        x = pd.to_numeric(np.array(sp[sp.colnames[0]], dtype=str), errors='coerce')
        y = pd.to_numeric(np.array(sp[sp.colnames[1]], dtype=str), errors='coerce')
        ok = ~np.isnan(x) & ~np.isnan(y)
        x, y = x[ok] / (1 + z) * 10, y[ok]
        cut = x > (3100 if 'UVB' in f else 5578)
        m = max(m, np.nanmax(deredden(x[cut], y[cut])))
    return m


XSH_NORM = _xsh_max()
offset = -0.1

FIG_H = 250 * ns.MM
fig, ax = plt.subplots(figsize=(193 * ns.MM, FIG_H))
LW = 0.4
labels = []          # (y, text)

for i, row in meta_sorted.iterrows():
    f = row['file_name']
    instrument = row['instrument']
    grating = row['Grating']
    mjd = row['mjd']

    if instrument == 'NTT/ EFOSC2' and pd.notnull(grating):
        instrument_id = f"{instrument} {grating}"
    else:
        instrument_id = instrument

    if 'NIR' not in f:
        spec = ascii.read(str(PATH_SPECTRA / f))
        x = pd.to_numeric(np.array(spec[spec.colnames[0]], dtype=str), errors='coerce')
        y = pd.to_numeric(np.array(spec[spec.colnames[1]], dtype=str), errors='coerce')
        valid = ~np.isnan(x) & ~np.isnan(y)
        x = x[valid] / (1 + z)
        y = y[valid]

        if instrument == 'VLT/ X-Shooter':
            if 'UVB' in f:
                x *= 10
                m = x > 3100
                x, y = x[m], y[m]
            elif 'VIS' in f:
                x *= 10
                m = x > 5578
                x, y = x[m], y[m]

        y = deredden(x, y)                 # [rev3] MW + host extinction removed
        if instrument == 'VLT/ X-Shooter':
            # [rev3] one normalisation for both X-shooter arms, so dereddening
            # does not open a step at the UVB/VIS join (was per-arm max)
            yn = y / XSH_NORM + offset
        else:
            yn = y / np.nanmax(y) + offset
        xp, yp = break_gaps(np.asarray(x), np.asarray(yn))
        ax.plot(xp, yp, color='black', lw=LW)
        if 'UVB' not in f:
            labels.append((offset, f"{fmt_phase(mjd - B_peak)}  {fmt_inst(instrument, grating)}"))

    # offset step, exactly as in the original
    if i < len(meta_sorted) - 1:
        nrow = meta_sorted.iloc[i + 1]
        if nrow['instrument'] == 'NTT/ EFOSC2' and pd.notnull(nrow['Grating']):
            next_id = f"{nrow['instrument']} {nrow['Grating']}"
        else:
            next_id = nrow['instrument']
        if instrument_id != next_id or abs(mjd - nrow['mjd']) >= 0.1:
            offset += 1

YTOP = offset + 1.7
ax.set_ylim(-1, YTOP)
XMIN, XMAX = 3000, 10300
ax.set_xlim(XMIN, XMAX)

# phase / instrument labels in a column right of the frame
for yl, txt in labels:
    ax.text(1.008, yl + 0.35, txt, transform=ax.get_yaxis_transform(),
            fontsize=ns.FS_TINY, va='center', ha='left', clip_on=False)

# ---------------- line-identification bands (as in original) ----------------
COLORS = {
    'SiII': '#aec7e8', 'FeII': '#ff9896', 'CII': '#98df8a', 'SII': '#c5b0d5',
    'OI': '#ffbb78', 'CaII': '#9edae5', 'MgII': '#C3A384', 'NaI': "#f7c6c7",
}
BAND_ALPHA = 0.25
trans = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)
YLAB = 1.004


def band(x0, x1, color):
    ax.axvspan(x0, x1, color=color, alpha=BAND_ALPHA, lw=0, zorder=0)


def lab(x, text, rot=90, ha='center'):
    ax.text(x, YLAB, text, transform=trans, fontsize=ns.FS_TINY, rotation=rot,
            ha=ha, va='bottom', clip_on=False)


v = 3000
for line in [4130, 6355]:                       # Si II
    lc = line * (1 + z) * (1 - v / c)
    band(lc - 100, lc + 100, COLORS['SiII']); lab(lc, 'Si II')
line = 5958                                     # Si II 5958/5979
lc = line * (1 + z) * (1 - v / c)
band(lc - 10, lc + 100, COLORS['SiII']); lab(lc + 70, 'Si II')
band(4300, 5200, COLORS['FeII']); lab(4750, 'Fe II + IGE', rot=0)   # Fe II + IGE
for line in [4267, 6578, 7231]:                 # C II
    lc = line * (1 + z) * (1 - 2800 / c)
    band(lc - 50, lc + 50, COLORS['CII']); lab(lc, 'C II')
for line in [5500]:                             # S II
    lc = line * (1 + z) * (1 - v / c)
    band(lc - 200, lc + 200, COLORS['SII']); lab(lc, 'S II', rot=0)
for line in [6158, 7774]:                       # O I
    lc = line * (1 + z) * (1 - v / c)
    band(lc - 100, lc + 100, COLORS['OI']); lab(lc, 'O I')
lc = 5893 * (1 + z)                             # Na I
band(lc - 100, lc + 100, COLORS['NaI']); lab(lc - 65, 'Na I')
lc = 7291 * (1 + z)                             # [Ca II]
band(lc - 60, lc + 100, COLORS['CaII']); lab(lc + 20, '[Ca II]')
lc = 8542 * (1 + z)                             # Ca II NIR triplet
band(lc - 200, lc + 200, COLORS['CaII']); lab(lc, 'Ca II', rot=0)
lc = np.mean([3934, 3969]) * (1 + z) * (1 - 3500 / c)   # Ca II H&K
band(lc - 80, lc + 80, COLORS['CaII']); lab(lc, 'Ca II')
lc = 9230 * (1 + z) * (1 - v / c)               # O I / Mg II
band(lc - 100, lc + 100, '#E1AF7E'); lab(lc, 'O I/Mg II')
lc = 7890 * (1 + z) * (1 - v / c)               # Mg II
band(lc, lc + 100, COLORS['MgII']); lab(lc + 50, 'Mg II')

ax.set_xlabel('Rest wavelength (Å)')
ax.set_ylabel('Normalized flux + offset')
ax.set_yticks([])
ax.minorticks_on()
ax.tick_params(axis='y', which='both', left=False)

ns.save(fig, 'supplementary/SuppFig4_optical_spectra')
