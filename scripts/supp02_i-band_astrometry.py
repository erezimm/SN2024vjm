"""Supplementary Fig. 2 -- BlackGEM i-band astrometric rms vs Gaia magnitude.

Reads the astrometric scatter tables Bin1/4/16.csv (1, 4 and 16 co-added images)
from <data>/astrometry and plots DEC_RMS (mas) against Gaia G.
"""
import pandas as pd
import matplotlib.pyplot as plt

import nature_style as ns
ns.apply()

import paths
D = paths.ASTROMETRY
df1 = pd.read_csv(f'{D}/Bin1.csv')
df4 = pd.read_csv(f'{D}/Bin4.csv')
df16 = pd.read_csv(f'{D}/Bin16.csv')

fig, ax = plt.subplots(figsize=(ns.SINGLE_COL, 0.66 * ns.SINGLE_COL), layout='constrained')

# shaded region: SN brightness range in the i-band images
ax.axvspan(16, 18, alpha=0.2, color='gray', lw=0, zorder=0)
ax.text(17, 1350, 'SN brightness', fontsize=ns.FS, ha='center', va='center')

# Euclid-VIS pixel reference line at 0.1 arcsec = 100 mas
ax.axhline(100, color='black', linestyle='--', linewidth=0.8, alpha=0.7, zorder=10)
ax.text(14.1, 112, r'$Euclid$-VIS pixel', fontsize=ns.FS_SMALL, ha='left', va='bottom')

ms = 1.6
for df, lab, c in [(df1, '1 image', '#1f77b4'),
                   (df4, '4 images', '#ff7f0e'),
                   (df16, '16 images', '#2ca02c')]:
    ax.plot(df['Gaia_MAG'], df['DEC_RMS'] * 1000, 'o', ms=ms, mew=0, label=lab, color=c)

ax.axis([14, 21, 1, 2000])
ax.set_yscale('log')
ax.set_xlabel(r'Gaia $G$ (mag)')
ax.set_ylabel('rms (mas)')
ax.grid(True, which='both', linestyle='--', linewidth=0.25, alpha=0.5)
ax.tick_params(which='both', top=True, right=True)
ax.legend(loc='upper left', markerscale=1.8, handletextpad=0.2, borderaxespad=0.4,
          labelspacing=0.3, frameon=True, framealpha=0.9, edgecolor='none')

ns.save(fig, 'supplementary/SuppFig2_i-band_astrometry', dpi=600)
