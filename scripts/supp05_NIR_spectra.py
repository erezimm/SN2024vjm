"""Supplementary Fig. 5 -- NIR spectral sequence of SN 2024vjm.

Plain-python extraction of IR_spectra.ipynb (notebooks/NIR_spectra_CoII_fits.ipynb)
(cells 0, 1, 3, 4, 5; identical to Spectra/IR/IR_spectra.ipynb apart from
path handling).  Cell 5 produced NIR_spectra_2024vjm.pdf.
Edited for revision 3:
  * all file writes of the notebook removed (ascii.write of the clipped
    spectra into Spectra/, savefig into the notebook folder, diagnostic
    plots); the figure goes only to outputs/figures/supplementary.
  * B-band maximum MJD 60572.29 (was 60572.18); 2-decimal phases incl. JWST
    (MJD 60585.083 from the file name) -- consistent with Supp. Table 1.
  * X-shooter: plotted range restricted to 10,000 < lambda_obs < 22,620 A for
    BOTH the grey unbinned and the black binned curve (the notebook already cut
    the binned curve at 22,620 A but not the grey one).  This removes the
    blue-edge spikes (9,940-9,975 A, up to 80x the median) and the spike +
    flux step at 22,718-22,724 A.  Normalization factor unchanged (see report).
  * gaps (telluric cut-outs) broken with NaN rather than separate segments.
  * grey telluric bands defined by an in-figure key; Nature style at 180 mm.
Data files are only read.
"""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from astropy.io import ascii
from astropy.time import Time
from scipy.ndimage import median_filter

import nature_style as ns
ns.apply()

import paths
PATH_SPECTRA = paths.SPECTRA

# ---------------- cell 1: read ----------------
a = ascii.read(PATH_SPECTRA / 'IR/SN2024vjm_0928_merged_tellcor.txt')
b = ascii.read(PATH_SPECTRA / 'IR/SN2024vjm_1010_merged_tellcor.txt')
e = ascii.read(PATH_SPECTRA / 'IR/SN2024vjm_1021_merged_tellcor.txt')
d = ascii.read(PATH_SPECTRA / 'TOO_StellarMerger_SCI_SLIT_FLUX_IDP_NIR.ascii')
jwst = ascii.read(PATH_SPECTRA / 'SN2024vjm_JWST_NIRSpec_60585.083.txt')

z = 0.0028


# ---------------- cell 3: X-shooter clipping + binning (no plots / writes) ----
def sigma_clip_flux(flux, window=11, sigma=3):
    smoothed = median_filter(flux, size=window)
    residual = flux - smoothed
    mad = np.median(np.abs(residual - np.median(residual)))
    std_est = 1.4826 * mad
    clipped = np.copy(flux)
    mask = np.abs(residual) > sigma * std_est
    clipped[mask] = smoothed[mask]
    return clipped, mask


def bin_spectrum(wavelength, flux, bin_factor):
    n_bins = len(flux) // bin_factor
    bw = np.mean(wavelength[:n_bins * bin_factor].reshape(-1, bin_factor), axis=1)
    bf = np.mean(flux[:n_bins * bin_factor].reshape(-1, bin_factor), axis=1)
    return bw, bf


wavelength = np.array(d['WAVE']) * 10
flux = np.array(d['FLUX']) / np.nanmedian(d['FLUX'])
mask = np.ones_like(flux, dtype=bool)
mask[(wavelength > 13450) & (wavelength < 14200)] = False
mask[(wavelength > 18100) & (wavelength < 19356)] = False
wavelength, flux = wavelength[mask], flux[mask]

clipped_flux = np.copy(flux)
for _ in range(10):
    new, m = sigma_clip_flux(clipped_flux, window=21, sigma=5)
    if m.sum() == 0:
        break
    clipped_flux = new
binned_wavelength, binned_flux = bin_spectrum(wavelength, clipped_flux, 5)

# ---------------- cell 4: SOAR + JWST (no writes) ----------------
time_d = Time("2024-09-17T01:03:21.1465").mjd
time_a = Time("2024-09-28T00:06:35.427000").mjd
time_b = Time("2024-10-10T00:25:31.077000").mjd
time_e = Time("2024-10-22T01:01:26.277000").mjd
time_jwst = 60585.083                      # from the JWST file name


def clean_telluric_areas(wl, flux, error):
    m = np.ones_like(flux, dtype=bool)
    m[(wl > 1.345) & (wl < 1.420)] = False
    m[(wl > 1.810) & (wl < 1.936)] = False
    m[(wl < 0.98)] = False
    return np.array(wl[m]) * 10000, np.array(flux[m]), np.array(error[m])


def iter_clip(fl):
    out = np.copy(fl)
    for _ in range(10):
        new, m = sigma_clip_flux(out, window=21, sigma=5)
        if m.sum() == 0:
            break
        out = new
    return out


awl, aflux, _ = clean_telluric_areas(a['col1'], a['col2'], a['col3'])
bwl, bflux, _ = clean_telluric_areas(b['col1'], b['col2'], b['col3'])
ewl, eflux, _ = clean_telluric_areas(e['col1'], e['col2'], e['col3'])

a_clipped_flux = aflux / np.nanmedian(aflux)
awl = awl[~np.isnan(a_clipped_flux)]
a_clipped_flux = iter_clip(a_clipped_flux[~np.isnan(a_clipped_flux)])
b_clipped_flux = iter_clip(bflux / np.nanmedian(bflux))
e_clipped_flux = iter_clip(eflux / np.nanmedian(eflux))

jwst_flux = np.array(jwst['col2'][jwst['col1'] < 2.4])
jwst_wl = np.array(jwst['col1'][jwst['col1'] < 2.4]) * 10000
jwst_wl = jwst_wl[~np.isnan(jwst_flux)]
jwst_flux = jwst_flux[~np.isnan(jwst_flux)]

binned_flux = binned_flux[binned_wavelength < 22620]
binned_wavelength = binned_wavelength[binned_wavelength < 22620]

# normalization exactly as in the notebook (computed before the plot mask)
xsh_norm = np.nanmax(binned_flux)

# plot mask for the X-shooter edges (display only; data untouched)
XSH_LO, XSH_HI = 10000, 22620              # observed-frame A
mg = (wavelength > XSH_LO) & (wavelength < XSH_HI)
mb = (binned_wavelength > XSH_LO) & (binned_wavelength < XSH_HI)

# ---------------- cell 5: figure ----------------
B_peak = 60572.29


def fmt_phase(p):
    return f"{p:+.2f}".replace('-', '−') + ' d'


def plot_gaps(ax, x, y, gap_threshold=100, **kw):
    x = np.asarray(x, float); y = np.asarray(y, float)
    gi = np.where(np.diff(x) > gap_threshold)[0]
    ax.plot(np.insert(x, gi + 1, np.nan), np.insert(y, gi + 1, np.nan), **kw)


fig, ax = plt.subplots(figsize=(221 * ns.MM, 105 * ns.MM))
LW = 0.45
kw = dict(color='black', lw=LW)

plot_gaps(ax, wavelength[mg] / (1 + z), clipped_flux[mg] / xsh_norm + 0.3,
          color='0.55', lw=0.35)
plot_gaps(ax, binned_wavelength[mb] / (1 + z), binned_flux[mb] / xsh_norm + 0.3, **kw)
plot_gaps(ax, awl / (1 + z), a_clipped_flux / np.nanmax(a_clipped_flux) - 0.34, **kw)
plot_gaps(ax, jwst_wl / (1 + z), jwst_flux / np.nanmax(jwst_flux) - 1, **kw)
plot_gaps(ax, bwl / (1 + z), b_clipped_flux / np.nanmax(b_clipped_flux) - 1.5, **kw)
plot_gaps(ax, ewl / (1 + z), e_clipped_flux / np.nanmax(e_clipped_flux) - 2, **kw)

TX = 20300
for y, t, inst in [(0.40, time_d, 'VLT/X-shooter'),
                   (-0.15, time_a, 'SOAR/TripleSpec'),
                   (-0.45, time_jwst, 'JWST/NIRSpec'),
                   (-1.30, time_b, 'SOAR/TripleSpec'),
                   (-1.80, time_e, 'SOAR/TripleSpec')]:
    ax.text(TX, y, f"{fmt_phase(t - B_peak)}  {inst}", fontsize=ns.FS_SMALL,
            va='bottom', ha='left')

YLO, YHI = -2.1, 0.9
trans = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)
TELL = '#bdbdbd'
for x0, x1 in [(13400, 14200), (18000, 19356), (19900, 20200)]:
    ax.axvspan(x0, x1, color=TELL, alpha=0.45, lw=0, zorder=0)
    ax.text(0.5 * (x0 + x1), 0.985, r'$\oplus$', transform=trans,
            fontsize=ns.FS_PANEL + 1, ha='center', va='top')

# line-ID bands (as in notebook)
line = 10927
ax.axvspan(line * (1 + z) - 220, line * (1 + z) + 200, color='#C3A384', alpha=0.25, lw=0, zorder=0)
ax.text(line * (1 + z), 0.985, 'Mg II', transform=trans, fontsize=ns.FS_TINY,
        rotation=90, ha='center', va='top')
line = 16900 * (1 + z)
ax.axvspan(line - 100, line + 100, color='red', alpha=0.12, lw=0, zorder=0)
ax.text(line, 0.985, 'Fe II', transform=trans, fontsize=ns.FS_TINY,
        rotation=90, ha='center', va='top')
for x0, x1 in [(15572, line - 100), (line + 100, 18000), (20700, 23880)]:
    ax.axvspan(x0, x1, color='#4169E1', alpha=0.10, lw=0, zorder=0)
    ax.text(0.5 * (x0 + x1), 0.985, 'Co II', transform=trans, fontsize=ns.FS_TINY,
            ha='center', va='top')

ax.legend(handles=[Patch(fc=TELL, alpha=0.45, ec='none',
                         label=r'$\oplus$ telluric absorption'),
                   Line2D([], [], color='0.55', lw=0.6,
                          label='X-shooter, unbinned')],
          ncol=2, columnspacing=2.0,
          loc='lower left', bbox_to_anchor=(0.0, 1.0), fontsize=ns.FS_SMALL,
          borderaxespad=0.2, handlelength=1.4, handleheight=0.9)

ax.set_ylim(YLO, YHI)
ax.set_xlim(9700, 24000)
ax.set_xlabel('Rest wavelength (Å)')
ax.set_ylabel('Normalized flux + offset')
ax.set_yticks([])
ax.minorticks_on()
ax.tick_params(axis='y', which='both', left=False)

ns.save(fig, 'supplementary/SuppFig5_NIR_spectra')
