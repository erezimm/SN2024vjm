"""Supplementary Fig. 1 -- MUSE NFM H-alpha slice around SN 2024vjm.

Plots the H-alpha slice of the sky-subtracted, WCS-corrected MUSE datacube
produced by notebooks/MUSE_cube.ipynb (<external>/muse/updated_wcs_datacube.fits;
see README.md and paths.py).

Run with:  python supp01_MUSE_Halpha.py [--external DIR]            (requires mpdaf)
           SUPP1_CORRECTED=1 python supp01_MUSE_Halpha.py          -> scale bar at D = 9.39 Mpc
                                                                  (version used in the paper)
"""
import os
import numpy as np
import mpdaf.obj
from astropy import units as u
import matplotlib.pyplot as plt
from astropy.visualization import ZScaleInterval

import nature_style as ns
ns.apply()

redshift = 0.0028
z = redshift

import paths
CUBE = str(paths.MUSE / 'updated_wcs_datacube.fits')
cube = mpdaf.obj.Cube(CUBE)

CORRECTED = os.environ.get('SUPP1_CORRECTED', '0') == '1'

# ---------------- cell 10 (unchanged logic) ----------------
H_alpha = 6562.8
lambda_obs = H_alpha * (1 + redshift)
index = int(cube.wave.pixel(lambda_obs))
print(f'H-alpha slice: index {index}, lambda = {cube.wave.coord(index):.2f} A '
      f'(target {lambda_obs:.2f} A)')

h_alpha_image = cube[index, :, :]
data = h_alpha_image.data

edge_trim = 20  # pixels to trim from each edge
data_cropped = data[edge_trim:data.shape[0] - edge_trim, edge_trim:data.shape[1] - edge_trim]
data_masked = np.ma.masked_invalid(data_cropped)

zscale = ZScaleInterval()
vmin, vmax = zscale.get_limits(data_masked.compressed())
print(f'zscale limits: {vmin:.1f} {vmax:.1f}; display 0 .. {vmax:.1f}')

if CORRECTED:
    # [rev3-variant] pixel scale from the cube WCS and the paper's adopted distance
    pixel_scale_mas = float(np.mean(cube.wcs.get_step(unit=u.mas)))
    distance_mpc = 9.39
else:
    pixel_scale_mas = 25    # mas per pixel (as in the notebook)
    distance_mpc = 7.92     # Mpc (as in the notebook)
pc_per_mas = distance_mpc * 1e6 / 206265e3
scale_length_pc = 20
scale_length_pixels = scale_length_pc / pc_per_mas / pixel_scale_mas
print(f'pixel scale {pixel_scale_mas:.2f} mas, D = {distance_mpc} Mpc, '
      f'20 pc = {scale_length_pixels:.2f} pix')

# ---------------- figure at print size ----------------
ny, nx = data_cropped.shape
W = ns.SINGLE_COL
img_w_in = W                                 # [rev3] colour bar removed (author request)
img_h_in = img_w_in * ny / nx
H = img_h_in + 1.0 * ns.MM
fig = plt.figure(figsize=(W, H))
ax = fig.add_axes([0, 0.5 * ns.MM / H, img_w_in / W, img_h_in / H])

im = ax.imshow(data_masked, origin='lower', cmap='gray', vmin=0, vmax=vmax,
               interpolation='nearest')

LW = 0.8
col = 'white'
# scale bar (bottom left), same data-pixel anchor as the notebook
sbx, sby = 10, 20
ax.plot([sbx, sbx + scale_length_pixels], [sby, sby], color=col, lw=LW * 1.5,
        solid_capstyle='butt')
ax.text(sbx + scale_length_pixels / 2, sby + 2.5, f'{scale_length_pc} pc', color=col,
        fontsize=ns.FS, ha='center', va='bottom', weight='bold')

# compass (top left): N up, E left -- cube WCS has diagonal CD (CD1_1<0, CD2_2>0)
cx, cy, cs = 30, ny - 20, 20
ax.plot([cx, cx], [cy - cs, cy], color=col, lw=LW, solid_capstyle='butt')
ax.text(cx, cy + 2, 'N', color=col, fontsize=ns.FS, ha='center', va='bottom',
        weight='bold')
ax.plot([cx - cs, cx], [cy - cs, cy - cs], color=col, lw=LW, solid_capstyle='butt')
ax.text(cx - cs - 2, cy - cs, 'E', color=col, fontsize=ns.FS, ha='right', va='center',
        weight='bold')

# SN label (same position as the notebook: above the SN at the cube centre)
ax.text(nx // 2 - 2, ny // 2 + 25, 'SN 2024vjm', color=col, fontsize=ns.FS,
        ha='center', va='center', weight='bold')

ax.set_xticks([])
ax.set_yticks([])
for s in ax.spines.values():
    s.set_visible(False)


name = 'supplementary/SuppFig1_MUSE_Halpha'
if CORRECTED:
    (ns.OUT_DIR / 'supplementary' / 'variants').mkdir(parents=True, exist_ok=True)
    name = 'supplementary/variants/SuppFig1_MUSE_Halpha_D9.39Mpc'
ns.save(fig, name, dpi=600)
