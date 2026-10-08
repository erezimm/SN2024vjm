"""[rev3] Supplementary Fig. 9 -- GP interpolation of the light curve in flux space.

Adapted from plot_gp_flux_with_offsets() in lc_clean.py (worktree version incl.
Mookodi, which produced the submitted gp_interpolated_lightcurve_with_offsets.pdf).
GP inputs/settings unchanged. Plotted in units of 1e-15 erg/s/cm^2/A with a
single explicit scaling on the axis label; the key's offsets use the same units.
Output: outputs/figures/supplementary/SuppFig9_gp_flux_offsets.{pdf,png}
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

import supp03_09_lc_clean_lib as L
import supp03_09_prep as P
import nature_style as ns

ns.apply()

UNIT = 1e-15      # plotted flux unit (erg s^-1 cm^-2 A^-1)
MS = 2.0
TEL = {
    'BlackGEM': ('o', 'full', 'BlackGEM'), 'MeerLICHT': ('o', 'none', 'MeerLICHT'),
    'REM': ('h', 'full', 'REM'), 'LCO': ('s', 'full', 'LCO'), 'Prompt6': ('*', 'full', 'Prompt6'),
    'EFOSC': ('D', 'full', 'NTT/EFOSC2'), 'Mookodi': ('p', 'full', 'Mookodi'),
    'IMACS': ('X', 'full', 'Magellan/IMACS'), 'GOTO': ('^', 'full', 'GOTO'),
}
MS_SCALE = {'*': 1.35, 'D': 0.85, 's': 0.9, 'h': 1.05, 'p': 1.05, 'X': 1.1}


def tel_of(src):
    s = str(src)
    for key in ['MeerLICHT', 'BlackGEM', 'LCO', 'REM', 'Prompt6', 'EFOSC', 'Mookodi', 'IMACS', 'GOTO']:
        if key in s:
            return key
    print('WARNING: unknown source', s)
    return 'BlackGEM'


def flux_to_mag(flux, piv_wl):
    fnu = np.maximum(flux, 1e-99) * (piv_wl ** 2) / 2.99792458e18
    return -2.5 * np.log10(fnu / 3.631e-20)


def plot_gp(interp_lc, lc, colors, gp_flux_offsets, t0):
    flux_col = 'flux_corr' if 'flux_corr' in lc.columns else 'flux'
    fluxerr_col = 'fluxerr_corr' if 'fluxerr_corr' in lc.columns else 'fluxerr'

    # GP B-band peak (the original used this as the time zero); report vs t0
    b = interp_lc[interp_lc['filter'] == 'B'].sort_values('time')
    gp_peak = b['time'].to_numpy()[np.argmin(flux_to_mag(b['flux'].to_numpy(), b['piv_wl'].iloc[0]))]
    print(f'GP B-band peak (argmin mag) = MJD {gp_peak:.3f}; axis zero T0_BMAX = {t0:.2f}; '
          f'diff = {gp_peak - t0:+.3f} d')

    fig, ax = plt.subplots(figsize=(110 * ns.MM, 118 * ns.MM))
    used_tel, band_off = set(), {}
    for filt in np.unique(interp_lc['filter']):
        base = filt.replace('bg_', '') if filt.startswith('bg_') else filt
        if base not in colors:
            continue
        color = colors[base]
        off16 = gp_flux_offsets.get(filt, gp_flux_offsets.get(base, 0.0))   # original units 1e-16
        off = off16 * 1e-16 / UNIT                                           # -> units of 1e-15
        band_off.setdefault(base, set()).add(round(off, 3))

        mdata = lc[(lc['filter'] == filt) & (lc['source'] != 'ghost_explosion')]
        for src in mdata['source'].unique():
            sdata = mdata[mdata['source'] == src]
            tk = tel_of(src)
            ph = (sdata['time'] - t0).to_numpy()
            if np.any((ph >= -10) & (ph <= 70)):   # key only lists symbols visible in the x range
                used_tel.add(tk)
            mk, fs, _ = TEL[tk]
            ax.errorbar(sdata['time'] - t0, sdata[flux_col] / UNIT + off, sdata[fluxerr_col] / UNIT,
                        color=color, marker=mk, fillstyle=fs, ls='none',
                        ms=MS * MS_SCALE.get(mk, 1.0), mew=0.4, elinewidth=0.4, alpha=0.8, zorder=3)

        s = interp_lc[interp_lc['filter'] == filt].sort_values('time')
        if filt == 'L':
            # [rev3] GOTO L band is excluded from the GP (interpolate_flux_gp drops source GOTO);
            # its 'GP' is fitted to the explosion anchor point only -> not drawn.
            continue
        x = s['time'] - t0
        y = s['flux'] / UNIT + off
        e = s['fluxerr'] / UNIT
        ax.fill_between(x, y - e, y + e, alpha=0.2, color=color, lw=0, zorder=1)
        ax.plot(x, y, color=color, alpha=0.7, lw=0.7, zorder=2)

    for base, offs in band_off.items():
        if len(offs) > 1:
            print(f'WARNING: band {base} plotted with several offsets {offs}')

    def lab(base):
        offs = sorted(band_off.get(base, {0.0}))
        o = offs[0]
        name = f'${base}$'
        if base == 'L':
            name += ', no GP'
        if o == 0:
            return name
        return f'{name} {"+" if o > 0 else chr(0x2212)} {abs(o):.1f}'

    band_h = [Line2D([0], [0], color=colors[b_], lw=1.2, label=lab(b_))
              for b_ in ['u', 'B', 'g', 'V', 'q', 'r', 'i', 'z', 'J', 'H', 'K', 'L'] if b_ in band_off]
    order = ['BlackGEM', 'MeerLICHT', 'REM', 'GOTO', 'LCO', 'Prompt6', 'EFOSC', 'Mookodi', 'IMACS']
    tel_h = [Line2D([0], [0], marker=TEL[k][0], fillstyle=TEL[k][1], color='k', lw=0, mew=0.5,
                    ms=MS * 1.3 * MS_SCALE.get(TEL[k][0], 1.0), label=TEL[k][2])
             for k in order if k in used_tel]
    gp_h = [Line2D([0], [0], color='0.3', lw=0.7, label='GP mean'),
            Patch(facecolor='0.3', alpha=0.25, lw=0, label='GP 1$\\sigma$')]

    common = dict(frameon=False, handletextpad=0.35, borderaxespad=0.0, fontsize=ns.FS_SMALL,
                  columnspacing=0.9)
    fig.legend(handles=band_h, loc='lower center', bbox_to_anchor=(0.5, 1.085), ncol=6,
               handlelength=1.3, bbox_transform=ax.transAxes, **common)
    fig.legend(handles=tel_h, loc='lower center', bbox_to_anchor=(0.5, 1.01), ncol=5,
               handlelength=1.0, bbox_transform=ax.transAxes, **common)
    ax.legend(handles=gp_h, loc='upper right', handlelength=1.3, frameon=False,
              fontsize=ns.FS_SMALL, borderaxespad=0.6)

    ax.set_xlabel('Days from $B$-band maximum')
    ax.set_ylabel('Flux + offset (10$^{-15}$ erg s$^{-1}$ cm$^{-2}$ Å$^{-1}$)')
    ax.grid(True, alpha=0.2, lw=0.3)
    ax.set_xlim(-10, 70)
    ax.minorticks_on()
    ax.tick_params(which='both', top=True, right=True)
    ax.ticklabel_format(axis='y', useOffset=False, style='plain')
    return fig


if __name__ == '__main__':
    lc, A_map = P.build_lc()
    interp_lc = P.gp(lc)
    print('GP filters:', sorted(interp_lc['filter'].unique()))
    fig = plot_gp(interp_lc, lc, L.COLORS, L.GP_FLUX_OFFSETS, L.T0_BMAX)
    ns.save(fig, 'supplementary/SuppFig9_gp_flux_offsets')
