"""[rev3] Supplementary Fig. 3 -- multi-band light curve of SN 2024vjm.

Adapted from plot_lightcurve_mag() in lc_clean.py (worktree version incl.
Mookodi, which produced the submitted lightcurve.pdf). Data handling is
unchanged; only styling/keys changed for the Nature artwork guide.
Output: outputs/figures/supplementary/SuppFig3_lightcurve.{pdf,png}
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import supp03_09_lc_clean_lib as L
import supp03_09_prep as P
import nature_style as ns

ns.apply()

MS = 2.2          # marker size (pt) at final print size
ELW = 0.4         # error-bar line width
ALPHA_UNBIN = 0.2
ALPHA_ND = 0.5

TEL = {  # key -> (marker, fillstyle, legend label)
    'BlackGEM': ('o', 'full', 'BlackGEM'),
    'MeerLICHT': ('o', 'none', 'MeerLICHT'),
    'REM': ('h', 'full', 'REM'),
    'GOTO': ('^', 'full', 'GOTO'),
    'LCO': ('s', 'full', 'LCO'),
    'Prompt6': ('*', 'full', 'Prompt6'),
    'EFOSC': ('D', 'full', 'NTT/EFOSC2'),
    'IMACS': ('X', 'full', 'Magellan/IMACS'),
    'DLT40': ('P', 'full', 'DLT40'),
    'Mookodi': ('p', 'full', 'Mookodi'),
}
# marker-size tweaks so that symbols look equally heavy
MS_SCALE = {'*': 1.35, 'P': 1.1, 'X': 1.1, 'D': 0.85, 's': 0.9, 'h': 1.05, 'p': 1.05}


def tel_of(src):
    s = str(src)
    for key, pat in [('MeerLICHT', 'MeerLICHT'), ('BlackGEM', 'BlackGEM'), ('LCO', 'LCO'),
                     ('REM', 'REM'), ('Prompt6', 'Prompt6'), ('EFOSC', 'EFOSC'),
                     ('GOTO', 'GOTO'), ('IMACS', 'IMACS'), ('Mookodi', 'Mookodi'),
                     ('DLT40', 'DLT')]:
        if pat in s:
            return key
    print('WARNING: unknown source', s)
    return 'BlackGEM'


def band_label(base, off):
    name = base if base == 'Open' else f'${base}$'
    if off == 0:
        return name
    sign = '+' if off > 0 else '−'
    return f'{name} {sign} {abs(off):.1f}'


def plot_lightcurve_mag(lc, t0, dm, colors, filter_offsets, bg_detections, bg_non_detections,
                        ml_non_detections, rem_limits, DLT_40, A_map, use_extinction_corrected=True):
    AB_MINUS_VEGA = L.AB_MINUS_VEGA
    fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(ns.DOUBLE_COL, 100 * ns.MM),
                                   gridspec_kw={'width_ratios': [3, 2], 'wspace': 0.04})

    mag_col = 'mag_extcorr' if (use_extinction_corrected and 'mag_extcorr' in lc.columns) else 'magnitude'
    merr_col = 'magerr_extcorr' if (use_extinction_corrected and 'magerr_extcorr' in lc.columns) else 'error'

    VEGA_PLOT = {'B', 'V', 'J', 'H', 'K'}

    def to_plot_system(raw_filt, mags):
        base = raw_filt[3:] if isinstance(raw_filt, str) and raw_filt.startswith('bg_') else raw_filt
        if base in VEGA_PLOT:
            return np.asarray(mags, dtype=float) - AB_MINUS_VEGA.get(base, 0.0)
        return np.asarray(mags, dtype=float)

    def A_of(raw_filt):
        if not A_map or not isinstance(raw_filt, str):
            return 0.0
        base = raw_filt[3:] if raw_filt.startswith('bg_') else raw_filt
        for k in [raw_filt, raw_filt.strip(), base, f'bg_{base}', base.upper(), base.lower(),
                  f'bg_{base.upper()}', f'bg_{base.lower()}']:
            if k in A_map and np.isfinite(A_map[k]):
                return float(A_map[k])
        collapse = lambda k: k[3:] if isinstance(k, str) and k.startswith('bg_') else k
        same = [A_map[k] for k in A_map if collapse(k).lower() == base.lower()]
        return float(np.median(same)) if same else 0.0

    used_tel = set()
    for ax in (ax1, ax2):
        # 1) binned detections (opaque)
        for filt in sorted(lc['filter'].unique()):
            fdata = lc[(lc['filter'] == filt) & (lc['source'] != 'ghost_explosion')]
            if len(fdata) == 0:
                continue
            base = filt[3:] if str(filt).startswith('bg_') else filt
            if base not in colors:
                continue
            color = colors[base]
            offset = filter_offsets.get(filt, filter_offsets.get(base, 0.0))
            y_all = to_plot_system(filt, fdata[mag_col].to_numpy()) + offset
            for src in fdata['source'].unique():
                sdata = fdata[fdata['source'] == src]
                pos = fdata.index.get_indexer(sdata.index)
                tk = tel_of(src)
                used_tel.add(tk)
                mk, fs, _ = TEL[tk]
                ax.errorbar(sdata['time'] - t0, y_all[pos], sdata[merr_col], color=color,
                            marker=mk, fillstyle=fs, ls='none', ms=MS * MS_SCALE.get(mk, 1.0),
                            mew=0.4, elinewidth=ELW, zorder=3)

        # 2) unbinned BlackGEM detections (semi-transparent)
        for filt in bg_detections['FILTER'].unique():
            base = filt[3:] if str(filt).startswith('bg_') else filt
            if base not in colors:
                continue
            offset = filter_offsets.get(f'bg_{filt}', filter_offsets.get(base, 0.0))
            sel = bg_detections['FILTER'] == filt
            tt = bg_detections.loc[sel, 'MJD-OBS'].to_numpy()
            mm = bg_detections.loc[sel, 'MAG_ZOGY'].to_numpy().astype(float) - A_of(filt)
            mm = to_plot_system(filt, mm)
            ax.plot(tt - t0, mm + offset, color=colors[base], alpha=ALPHA_UNBIN, marker='o',
                    ms=MS, mew=0, ls='none', zorder=2)

        # 3-5) non-detections (only epochs <= t0)
        nd_sets = [
            (ml_non_detections[ml_non_detections['MJD'] <= t0], 'Filter', 'MJD', 'Mag_zogy', True),
            (bg_non_detections[bg_non_detections['MJD-OBS'] <= t0], 'FILTER', 'MJD-OBS', 'LIMMAG_ZOGY', True),
            (rem_limits[rem_limits['time'] <= t0], 'filter', 'time', 'limit_mag', False),
        ]
        for tab, fc, tc, mc, bgkey in nd_sets:
            for filt in tab[fc].unique():
                base = filt[3:] if str(filt).startswith('bg_') else filt
                if base not in colors:
                    continue
                offset = (filter_offsets.get(f'bg_{filt}', filter_offsets.get(base, 0.0)) if bgkey
                          else filter_offsets.get(base, 0.0))
                sel = tab[fc] == filt
                mm = to_plot_system(filt, tab.loc[sel, mc].to_numpy().astype(float) - A_of(filt))
                ax.plot(tab.loc[sel, tc].to_numpy() - t0, mm + offset, color=colors[base],
                        alpha=ALPHA_ND, marker='v', ms=MS * 1.1, mew=0, ls='none', zorder=2)

        # 6) DLT40 (unfiltered 'Open'; r-band extinction applied as in the original)
        base = DLT_40['filter'].iloc[0]
        sel = DLT_40['filter'] == base
        mm = DLT_40.loc[sel, 'mag'].to_numpy().astype(float) - A_of('r')
        ax.plot(DLT_40.loc[sel, 'time'].to_numpy() - t0, mm + filter_offsets.get(base, 0.0),
                color=colors[base], marker=TEL['DLT40'][0], ms=MS * MS_SCALE['P'], mew=0,
                ls='none', zorder=3)
        used_tel.add('DLT40')
        ax.grid(True, alpha=0.2, lw=0.3)

    ax1.invert_yaxis()
    ax1.set_xlim(-10, 75)
    ax2.set_xlim(155, 330)
    ax1.spines['right'].set_visible(False)
    ax2.spines['left'].set_visible(False)
    ax1.yaxis.tick_left()
    ax2.tick_params(axis='y', which='both', left=False, right=False, labelright=False)
    ax1.tick_params(axis='x', top=True)
    ax2.tick_params(axis='x', top=True)
    ax1.minorticks_on(); ax2.minorticks_on()
    ax1.tick_params(axis='y', which='both', right=False)

    # broken-axis marks (same visual size on both panels)
    d, h = 0.012, 0.018
    kw = dict(color='k', clip_on=False, lw=0.6)
    r = 3 / 2   # width ratio -> scale x on ax2 so slashes are parallel
    for y in (0, 1):
        ax1.plot((1 - d, 1 + d), (y - h, y + h), transform=ax1.transAxes, **kw)
        ax2.plot((-d * r, d * r), (y - h, y + h), transform=ax2.transAxes, **kw)

    fig.supxlabel('Days from $B$-band maximum', fontsize=ns.FS, y=0.035)
    ax1.set_ylabel('Apparent magnitude + offset')
    sec = ax2.secondary_yaxis('right', functions=(lambda x: x - dm, lambda x: x + dm))
    sec.set_ylabel('Absolute magnitude')
    sec.minorticks_on()

    # ---- keys: row 1 = band offsets; row 2 = telescopes + data type ----
    band_h = []
    for base in ['u', 'B', 'g', 'V', 'q', 'r', 'i', 'z', 'J', 'H', 'K', 'Open', 'L']:
        if base in colors:
            band_h.append(Line2D([0], [0], color=colors[base], lw=1.4,
                                 label=band_label(base, filter_offsets.get(base, 0.0))))
    order = ['BlackGEM', 'MeerLICHT', 'REM', 'GOTO', 'LCO', 'Prompt6', 'EFOSC', 'IMACS', 'DLT40', 'Mookodi']
    tel_h = []
    for k in order:
        if k in used_tel:
            mk, fs, lab = TEL[k]
            tel_h.append(Line2D([0], [0], marker=mk, color='k', fillstyle=fs, lw=0, mew=0.5,
                                ms=MS * 1.3 * MS_SCALE.get(mk, 1.0), label=lab))
    type_h = [
        Line2D([0], [0], marker='o', color='0.35', lw=0, ms=MS * 1.3, mew=0, label='Binned'),
        Line2D([0], [0], marker='o', color='0.35', lw=0, ms=MS * 1.3, mew=0, alpha=ALPHA_UNBIN + 0.1,
               label='BlackGEM unbinned'),
        Line2D([0], [0], marker='v', color='0.35', lw=0, ms=MS * 1.4, mew=0, alpha=ALPHA_ND,
               label='Non-detection'),
    ]
    common = dict(frameon=False, handletextpad=0.3, columnspacing=1.0, borderaxespad=0.0,
                  fontsize=ns.FS_SMALL)
    fig.subplots_adjust(top=0.86, bottom=0.10, left=0.065, right=0.935)
    leg1 = fig.legend(handles=band_h, loc='lower center', ncol=len(band_h), handlelength=1.3,
                      bbox_to_anchor=(0.5, 0.93), **common)
    fig.legend(handles=tel_h + type_h, loc='lower center', ncol=len(tel_h + type_h),
               handlelength=0.9, bbox_to_anchor=(0.5, 0.885), **common)
    fig.add_artist(leg1)
    return fig


if __name__ == '__main__':
    lc, A_map = P.build_lc()
    bg_dets, bg_nondets, ml_nondets, rem_limits, dlt40 = P.extra_tables()
    print('T0_BMAX =', L.T0_BMAX, ' DM =', L.DM)
    print('filters:', sorted(lc['filter'].unique()))
    print('sources:', sorted(lc['source'].astype(str).unique()))
    fig = plot_lightcurve_mag(lc, L.T0_BMAX, L.DM, L.COLORS, L.FILTER_OFFSETS, bg_dets, bg_nondets,
                              ml_nondets, rem_limits, dlt40, A_map)
    ns.save(fig, 'supplementary/SuppFig3_lightcurve')
