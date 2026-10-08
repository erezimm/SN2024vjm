import astropy.io.fits as fits
import astropy.wcs as wcs
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Ellipse,Circle
from astropy.visualization import ZScaleInterval
from astropy.coordinates import SkyCoord
import numpy as np
import matplotlib as mpl
from matplotlib.patches import Rectangle, ConnectionPatch

import paths
import nature_style as ns   # [rev3] Nature artwork style (Arial, TrueType, print-size fonts)
ns.apply()
mpl.rcParams['image.interpolation'] = 'nearest'
mpl.rcParams['image.resample'] = False


# Color Configuration
PLOT_COLORS = {
    'sn': 'darkviolet',     
    's1': 'dodgerblue',     
    'catalog': 'crimson',  
    'box': 'crimson',      
    'connector': 'crimson',
    'compass': 'black',
    'scale_bar': 'black',
    'crosshair': 'black',
}

# Standard Styling Constants
STD_LW = 0.8          # [rev3] was 2.5 at 18-inch width
STD_FONTSIZE = ns.FS  # [rev3] was 18 at 18-inch width

# Zoom Configuration (Bottom Panel Height in arcsec)
# Width will be automatically set to 2x this value.
ZOOM_CUTOUT_HEIGHT = 7.5

def _print_nearest_source_row(row, *, prefix='Nearest source'):
    if row is None:
        print(f"{prefix}: <not found>")
        return
    try:
        txt = row.to_string()
    except Exception:
        txt = str(row)
    print(f"{prefix} full row:\n{txt}")


def _draw_sn_bottom_right_guides(
    ax,
    x,
    y,
    width,
    height,
    *,
    length_pix=24,
    gap_pix=8,
    color=PLOT_COLORS['crosshair'],
    lw=2.5,
):
    """Draw short bottom+right guide lines indicating a point (x, y).

    The lines intentionally leave a small gap around (x, y) so the marker does not
    cover the SN itself.

    Parameters are in image pixel coordinates (same as imshow data coordinates).
    """
    w = int(max(width, 0))
    h = int(max(height, 0))
    if w == 0 or h == 0:
        return

    x0 = float(np.clip(x, 0, w - 1))
    y0 = float(np.clip(y, 0, h - 1))
    length = max(1.0, float(length_pix))
    gap = float(np.clip(gap_pix, 0.0, max(length - 1.0, 0.0)))

    # Bottom segment: downward (toward y=0)
    y_start = float(np.clip(y0 - gap, 0.0, h - 1))
    y_end = float(np.clip(y0 - length, 0.0, h - 1))
    if y_start > y_end:
        ax.plot([x0, x0], [y_end, y_start], color=color, lw=lw)

    # Right segment: rightward (toward increasing x)
    x_start = float(np.clip(x0 + gap, 0.0, w - 1))
    x_end = float(np.clip(x0 + length, 0.0, w - 1))
    if x_end > x_start:
        ax.plot([x_start, x_end], [y0, y0], color=color, lw=lw)

def extract_flux_magnitude_catalog(catalog_file):
    """
    Extract RA, DEC, FLUX_ISO, FLUXERR_ISO, MAG_ISO, and MAGERR_ISO from a SExtractor catalog.
    
    Parameters:
    - catalog_file (str): Path to the SExtractor catalog file.
    
    Returns:
    - pd.DataFrame: DataFrame containing the extracted columns.
    """
    # Define the column names based on the catalog file structure
    
    # Load the catalog file, skip comment lines
    catalog = pd.read_csv(catalog_file, delim_whitespace=True, comment='#', header=None)
    # print(catalog.head())
    
    # Extract the relevant columns (RA, DEC, flux/magnitude data, and ellipse parameters)
    selected_columns = [1,2,76,77,80,81,138,62,144,145,146,147,161,162,7,8,55,56,53,54,138,191,33,43,13,23,119,120,121]
    # selected_columns = [76,77,5,6,7,8]
    # selected_columns = [76,77,5,6,55,56]
    catalog_filtered = catalog[selected_columns]

    catalog_filtered.columns = ['X','Y','RA','Dec','X_err','Y_err','STAR_CLASS','PETRO_RADIUS','FLUX_PSF','FLUXERR_PSF','MAG_PSF','MAGERR_PSF','MAG_SPHEROID','MAGERR_SPHEROID','MAG_ISO','MAGERR_ISO','MAG_AUTO','MAGERR_AUTO','FLUX_AUTO','FLUXERR_AUTO','CLASS_STAR','SPREAD_MODEL','MAG_APER','MAGERR_APER','FLUX_APER','FLUXERR_APER','A_IMAGE','B_IMAGE','THETA_IMAGE']
    
    return catalog_filtered

def side_by_side_with_big_overlays(
    fits_file,
    catalog,
    ra_center,
    dec_center,
    cutout_size_arcsec=10.0,
    circle_radius_arcsec=0.1,
    large_cutout_deg=1.0,
    figsize=(18, 9),
    savepath=None
):
    with fits.open(fits_file) as hdul:
        w = wcs.WCS(hdul[0].header)
        data = np.squeeze(hdul[0].data)
        pixel_scale = np.abs(hdul[0].header['CD2_2']) * 3600.0  # arcsec/pixel

    cutout_size_pix   = cutout_size_arcsec / pixel_scale
    circle_radius_pix = circle_radius_arcsec / pixel_scale
    circle_radius_sn_pix = (50e-3) / pixel_scale   # 50 mas

    x_center, y_center = w.world_to_pixel(SkyCoord(ra_center, dec_center, unit='deg'))
    x_center, y_center = int(x_center), int(y_center)

    large_cutout_pix = (large_cutout_deg * 3600.0) / pixel_scale
    half_large = int(large_cutout_pix / 2)
    x0 = max(0, x_center - half_large); x1 = min(data.shape[1], x_center + half_large)
    y0 = max(0, y_center - half_large); y1 = min(data.shape[0], y_center + half_large)
    big = data[y0:y1, x0:x1]

    xc, yc = x_center - x0, y_center - y0

    half_size = int(cutout_size_pix / 2)
    cut = data[y_center - half_size:y_center + half_size,
               x_center - half_size:x_center + half_size]

    # Brightness scaling
    vmax_big = ZScaleInterval(contrast=0.15).get_limits(big)[1]
    vmin_big = -2.0
    vmax_big = 15
    vmin_cut, vmax_cut = ZScaleInterval().get_limits(cut)

    # 10″ box in left panel
    x_min, x_max = xc - half_size, xc + half_size
    y_min, y_max = yc - half_size, yc + half_size

    Hb, Wb = big.shape
    Hc, Wc = cut.shape

    fig = plt.figure(figsize=figsize)
    gs = plt.GridSpec(1, 2, width_ratios=[Wb / Hb, Wc / Hc])
    axL = fig.add_subplot(gs[0, 0])
    axR = fig.add_subplot(gs[0, 1])
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0, hspace=0)

    # LEFT panel (large image)
    axL.imshow(big, origin='lower', cmap='gray_r', vmin=vmin_big, vmax=vmax_big,
               interpolation='nearest', resample=False)
    axL.set_xlim(-0.5, Wb - 0.5); axL.set_ylim(-0.5, Hb - 0.5)
    axL.set_aspect('equal', adjustable='box'); axL.set_box_aspect(Hb / Wb)
    axL.set_xticks([]); axL.set_yticks([])
    axL.add_patch(Rectangle((x_min, y_min), (x_max - x_min), (y_max - y_min),
                            linewidth=2.5, edgecolor=PLOT_COLORS['box'], facecolor='none'))

    # ---------- LARGE COMPASS (top-left, scaled with image size) ----------
    m = 0.06                                 # margin as fraction of image size
    size_frac = 0.14                         # compass size as fraction of min(Hb, Wb)
    comp_size = int(size_frac * min(Hb, Wb))
    comp_size = max(comp_size, 40)           # ensure it's never tiny

    cx = 1000                         # top-left X (from left)
    cy_top = int(Hb * (1.0 - m))             # top margin Y (from bottom)
    cy = cy_top - comp_size                  # anchor so arrow fits inside

    axL.plot([cx, cx], [cy, cy + comp_size], color='black', lw=3)          # North
    axL.text(cx, cy + comp_size + max(14, comp_size*0.12), 'N',
             color='black', fontsize=16,
             ha='center', va='bottom')
    axL.plot([cx, cx - comp_size], [cy, cy], color='black', lw=3)          # East
    axL.text(cx - comp_size - max(10, comp_size*0.12), cy, 'E',
             color='black', fontsize=16,
             ha='center', va='center')

    # ---------- 5 kpc DISTANCE BAR (slightly toward center) ----------
    d_pc = 9.4e6           # distance to galaxy (pc)
    bar_pc = 5000        # 5 kpc
    bar_arcsec = bar_pc / d_pc * 206265.0
    bar_pix = bar_arcsec / pixel_scale

    y_bar = 100                                # a bit below top
    x_bar_center = int(Wb * 0.85)                           # shifted toward center
    x_bar0 = x_bar_center - bar_pix / 2.0
    x_bar1 = x_bar_center + bar_pix / 2.0

    axL.plot([x_bar0, x_bar1], [y_bar, y_bar], color='black', lw=3)
    axL.text((x_bar0 + x_bar1) / 2.0, y_bar + 10, '5 kpc',
             color='black', fontsize=16,
             ha='center', va='bottom')

    # RIGHT panel (10″ cutout)
    axR.imshow(cut, origin='lower', cmap='gray_r', vmin=vmin_cut, vmax=vmax_cut,
               interpolation='nearest', resample=False)
    axR.set_xlim(-0.5, Wc - 0.5); axR.set_ylim(-0.5, Hc - 0.5)
    axR.set_aspect('equal', adjustable='box'); axR.set_box_aspect(Hc / Wc)
    axR.set_xticks([]); axR.set_yticks([])

    # SN marker + label
    axR.add_patch(plt.Circle((Wc / 2, Hc / 2),
                             circle_radius_sn_pix, color=PLOT_COLORS['sn'], fill=False, lw=2, zorder=10))
    axR.text(Wc / 2 + 6.5, Hc / 2 + 1.5, 'SN 2024vjm',
             color=PLOT_COLORS['sn'], fontsize=20, ha='center', va='center', weight='bold')

    # Catalog filtering (as before)
    cos_dec = np.cos(np.deg2rad(dec_center))
    ra_min = ra_center - (cutout_size_arcsec / 3600.0) / cos_dec
    ra_max = ra_center + (cutout_size_arcsec / 3600.0) / cos_dec
    dec_min = dec_center - cutout_size_arcsec / 3600.0
    dec_max = dec_center + (cutout_size_arcsec / 2.0) / 3600.0
    filtered_catalog = catalog[(catalog['RA'] >= ra_min) & (catalog['RA'] <= ra_max) &
                               (catalog['Dec'] >= dec_min) & (catalog['Dec'] <= dec_max)]
    filtered_catalog = filtered_catalog[filtered_catalog['MAG_AUTO'] < 99]

    # Nearest source to center
    center_x, center_y = cutout_size_pix / 2, cutout_size_pix / 2
    nearest_index, nearest_distance = None, float('inf')
    nearest_row = None
    for idx, row in filtered_catalog.iterrows():
        xs, ys = w.world_to_pixel(SkyCoord(row['RA'], row['Dec'], unit='deg'))
        x_cut = xs - (x_center - half_size); y_cut = ys - (y_center - half_size)
        if 0 <= x_cut <= cutout_size_pix and 0 <= y_cut <= cutout_size_pix:
            d = np.hypot(x_cut - center_x, y_cut - center_y)
            if d < nearest_distance:
                nearest_distance, nearest_index = d, idx
                nearest_row = row

    _print_nearest_source_row(nearest_row, prefix='Nearest source (Euclid-only zoom)')

    # Draw catalog circles + S1
    for idx, row in filtered_catalog.iterrows():
        xs, ys = w.world_to_pixel(SkyCoord(row['RA'], row['Dec'], unit='deg'))
        x_cut = xs - (x_center - half_size); y_cut = ys - (y_center - half_size)
        if 0 <= x_cut <= cutout_size_pix and 0 <= y_cut <= cutout_size_pix:
            color = PLOT_COLORS['s1'] if idx == nearest_index else PLOT_COLORS['catalog']
            lw = 2 if idx == nearest_index else 1
            
            # For S1, draw an ellipse based on its extended size
            if idx == nearest_index:
                # Get ellipse parameters from catalog
                a_img = row['A_IMAGE']  # Semi-major axis in pixels
                b_img = row['B_IMAGE']  # Semi-minor axis in pixels
                theta = row['THETA_IMAGE']  # Position angle in degrees
                
                # Scale factor to show the full extent (using 3 sigma like inject_and_measure.py)
                scale_factor = 3.0
                
                # Draw ellipse (width and height are full diameters)
                ellipse = Ellipse((x_cut, y_cut), 
                                width=2*a_img*scale_factor, 
                                height=2*b_img*scale_factor,
                                angle=theta, 
                                edgecolor=color, 
                                facecolor='none', 
                                lw=lw)
                axR.add_patch(ellipse)
                axR.text(x_cut - a_img*scale_factor, y_cut, 'S1',
                         color=PLOT_COLORS['s1'], fontsize=16, ha='right', va='center', weight='bold')
            else:
                # For other catalog sources, use circle as before
                axR.add_patch(plt.Circle((x_cut, y_cut), circle_radius_pix,
                                         color=color, fill=False, lw=lw))

    # Right-panel distance bar (50 pc) and compass (small)
    bar_pc_r = 50
    bar_arcsec_r = bar_pc_r / d_pc * 206265.0
    bar_pix_r = bar_arcsec_r / pixel_scale
    axR.plot([Wc - 2 - bar_pix_r, Wc - 2], [2, 2], color='black', lw=2)
    axR.text(Wc - 2 - bar_pix_r / 2.0, 2, f'{bar_pc_r} pc',
             color='black', fontsize=16, ha='center', va='bottom')
    axR.plot([13, 13], [2, 2 + 8], color='black', lw=2)
    axR.text(13, 2 + 10, 'N', color='black', fontsize=16, ha='center', va='bottom')
    axR.plot([13, 13 - 10], [2, 2], color='black', lw=2)
    axR.text(13 - 12, 2, 'E', color='black', fontsize=16, ha='center', va='center')

    # Two outer dashed red connectors
    dash = (0, (8, 6))
    for (cx0, cy0), (zx, zy) in [((x_max, y_max), (0, Hc)), ((x_max, y_min), (0, 0))]:
        fig.add_artist(ConnectionPatch(
            xyA=(cx0, cy0), coordsA=axL.transData,
            xyB=(zx, zy), coordsB=axR.transData,
            color=PLOT_COLORS['connector'], lw=2.2, linestyle=dash, alpha=0.95
        ))

    if savepath:
        plt.savefig(savepath, dpi=300, bbox_inches='tight', pad_inches=0)
    plt.show()

def plot_blackgem_only(
    fits_file,
    ra_center,
    dec_center,
    fov_arcmin=10.0, # Field of view in arcminutes
    sn_circle_radius_arcsec=1.0,
    savepath=None
):
    # --- 1. Load Data ---
    with fits.open(fits_file) as hdul:
        header = hdul[0].header
        w = wcs.WCS(header)
        data = np.squeeze(hdul[0].data)
        
        # Robust Pixel Scale Check
        if 'A-PSCALE' in header:
            pixel_scale = header['A-PSCALE']
        else:
            pixel_scale = np.sqrt(np.abs(np.linalg.det(w.pixel_scale_matrix))) * 3600.0
            
    print(f"Pixel Scale: {pixel_scale:.3f} arcsec/pix")

    # --- 2. Create Cutout ---
    target_coord = SkyCoord(ra_center, dec_center, unit='deg')
    
    # Calculate cutout size in pixels
    fov_pix = int((fov_arcmin * 60.0) / pixel_scale)
    if fov_pix % 2 != 0: fov_pix += 1 # Ensure even size
    half_size = fov_pix // 2
    
    # Get pixel coordinates of center
    x_c, y_c = w.world_to_pixel(target_coord)
    x_c, y_c = int(x_c), int(y_c)
    
    # Handle boundaries (prevent crash if near edge)
    y_min = max(0, y_c - half_size); y_max = min(data.shape[0], y_c + half_size)
    x_min = max(0, x_c - half_size); x_max = min(data.shape[1], x_c + half_size)
    
    raw_cutout = data[y_min:y_max, x_min:x_max]
    
    # --- 3. Rotate to North Up ---
    # Assuming the camera is rotated ~90deg (East is Up in raw), 
    # we rotate -90 (clockwise) to make North Up.
    img_cut = np.rot90(raw_cutout, -1)

    # --- 4. Plotting ---
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)
    
    # Grayscale scaling
    vmin, vmax = ZScaleInterval().get_limits(img_cut)
    ax.imshow(img_cut, origin='lower', cmap='gray_r', vmin=vmin, vmax=vmax)
    
    # Center marker (Target): bottom + right guide lines
    center_x, center_y = img_cut.shape[1] / 2, img_cut.shape[0] / 2
    _draw_sn_bottom_right_guides(ax, center_x, center_y, img_cut.shape[1], img_cut.shape[0], lw=2.5)
    ax.text(center_x + 20, center_y + 20, 'SN 2024vjm', color='black', 
            weight='bold', fontsize=18)
    
    # Compass (top Left)
    c_len = img_cut.shape[0] * 0.15
    cx, cy = img_cut.shape[1] * 0.2, img_cut.shape[0] * 0.75
    
    ax.plot([cx, cx], [cy, cy + c_len], color='black', lw=4) # North
    ax.text(cx, cy + c_len*1.2, 'N', color='black', fontsize=20, ha='center', weight='bold')
    
    ax.plot([cx, cx - c_len], [cy, cy], color='black', lw=4) # East (Left because North is Up)
    ax.text(cx - c_len*1.2, cy, 'E', color='black', fontsize=20, va='center', weight='bold')

    # Scale Bar (5 kpc)
    d_pc = 9.4e6
    bar_pc = 5000 
    bar_len_pix = (bar_pc / d_pc * 206265.0) / pixel_scale
    
    bx = img_cut.shape[1] * 0.85
    by = img_cut.shape[0] * 0.02
    ax.plot([bx - bar_len_pix/2, bx + bar_len_pix/2], [by, by], color='black', lw=4)
    ax.text(bx, by + 15, '5 kpc', color='black', fontsize=18, ha='center', va='bottom', weight='bold')

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("Blackgem Discovery Image (Rotated North-Up)", fontsize=20, pad=15)
    
    if savepath:
        plt.savefig(savepath, dpi=300, bbox_inches='tight')
        print(f"Saved to {savepath}")
    plt.show()


def bg_left_of_euclid_two_images(
    bg_fits_file,
    euclid_fits_file,
    euclid_catalog,
    ra_center,
    dec_center,
    bg_fov_arcmin=10.0,
    euclid_cutout_size_arcsec=10.0,
    euclid_circle_radius_arcsec=0.1,
    euclid_large_cutout_deg=1.0,
    figsize=(27, 9),
    savepath=None,
):
    """Create a 3-panel figure: BlackGEM (left) + Euclid (big + zoom) (right).

    Panel order:
      [BG discovery image] [Euclid large cutout w/ overlays] [Euclid 10\" zoom]
    """

    # -------------------- BlackGEM panel (left) --------------------
    with fits.open(bg_fits_file) as hdul:
        header_bg = hdul[0].header
        w_bg = wcs.WCS(header_bg)
        data_bg = np.squeeze(hdul[0].data)

        if 'A-PSCALE' in header_bg:
            pixel_scale_bg = header_bg['A-PSCALE']
        else:
            pixel_scale_bg = np.sqrt(np.abs(np.linalg.det(w_bg.pixel_scale_matrix))) * 3600.0

    target_coord = SkyCoord(ra_center, dec_center, unit='deg')
    fov_pix = int((bg_fov_arcmin * 60.0) / pixel_scale_bg)
    if fov_pix % 2 != 0:
        fov_pix += 1
    half_size_bg = fov_pix // 2

    x_bg, y_bg = w_bg.world_to_pixel(target_coord)
    x_bg, y_bg = int(x_bg), int(y_bg)

    y_min = max(0, y_bg - half_size_bg)
    y_max = min(data_bg.shape[0], y_bg + half_size_bg)
    x_min = max(0, x_bg - half_size_bg)
    x_max = min(data_bg.shape[1], x_bg + half_size_bg)

    raw_cutout_bg = data_bg[y_min:y_max, x_min:x_max]
    bg_img = np.rot90(raw_cutout_bg, -1)
    vmin_bg, vmax_bg = ZScaleInterval().get_limits(bg_img)

    # -------------------- Euclid panels (middle + right) --------------------
    with fits.open(euclid_fits_file) as hdul:
        w_e = wcs.WCS(hdul[0].header)
        data_e = np.squeeze(hdul[0].data)
        pixel_scale_e = np.abs(hdul[0].header['CD2_2']) * 3600.0  # arcsec/pixel

    cutout_size_pix = euclid_cutout_size_arcsec / pixel_scale_e
    circle_radius_pix = euclid_circle_radius_arcsec / pixel_scale_e
    circle_radius_sn_pix = (50e-3) / pixel_scale_e  # 50 mas

    x_center, y_center = w_e.world_to_pixel(SkyCoord(ra_center, dec_center, unit='deg'))
    x_center, y_center = int(x_center), int(y_center)

    large_cutout_pix = (euclid_large_cutout_deg * 3600.0) / pixel_scale_e
    half_large = int(large_cutout_pix / 2)
    x0 = max(0, x_center - half_large)
    x1 = min(data_e.shape[1], x_center + half_large)
    y0 = max(0, y_center - half_large)
    y1 = min(data_e.shape[0], y_center + half_large)
    big = data_e[y0:y1, x0:x1]

    xc, yc = x_center - x0, y_center - y0

    # For the hybrid layout we want the bottom zoom panel to be ~2× wider than tall
    # without distorting pixels. Do that by using a rectangular cutout (2× width).
    zoom_height_pix = int(round(cutout_size_pix))
    if zoom_height_pix % 2 != 0:
        zoom_height_pix += 1
    zoom_width_pix = 2 * zoom_height_pix
    half_size_y = zoom_height_pix // 2
    half_size_x = zoom_width_pix // 2

    y0_zoom = max(0, y_center - half_size_y)
    y1_zoom = min(data_e.shape[0], y_center + half_size_y)
    x0_zoom = max(0, x_center - half_size_x)
    x1_zoom = min(data_e.shape[1], x_center + half_size_x)
    cut = data_e[y0_zoom:y1_zoom, x0_zoom:x1_zoom]

    vmax_big = ZScaleInterval(contrast=0.15).get_limits(big)[1]
    vmin_big = -2.0
    vmax_big = 15
    vmin_cut, vmax_cut = ZScaleInterval().get_limits(cut)

    # Red box on the big Euclid panel should match the zoom panel region
    x_min_e, x_max_e = xc - half_size_x, xc + half_size_x
    y_min_e, y_max_e = yc - half_size_y, yc + half_size_y

    Hb, Wb = big.shape
    Hc, Wc = cut.shape
    Hbg, Wbg = bg_img.shape

    # -------------------- Figure layout --------------------
    fig = plt.figure(figsize=figsize)
    gs = plt.GridSpec(
        1,
        3,
        width_ratios=[Wbg / Hbg, Wb / Hb, Wc / Hc],
    )
    axBG = fig.add_subplot(gs[0, 0])
    axL = fig.add_subplot(gs[0, 1])
    axR = fig.add_subplot(gs[0, 2])
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0, hspace=0)

    # Panel labels
    label_kwargs = dict(
        transform=None,
        ha='left',
        va='top',
        fontsize=22,
        color='black',
        weight='bold',
    )
    axBG.text(0.03, 0.97, 'a', transform=axBG.transAxes, **{k: v for k, v in label_kwargs.items() if k != 'transform'})
    axL.text(0.03, 0.97, 'b', transform=axL.transAxes, **{k: v for k, v in label_kwargs.items() if k != 'transform'})
    axR.text(0.03, 0.97, 'c', transform=axR.transAxes, **{k: v for k, v in label_kwargs.items() if k != 'transform'})

    # -------------------- BG panel draw --------------------
    axBG.imshow(bg_img, origin='lower', cmap='gray_r', vmin=vmin_bg, vmax=vmax_bg,
                interpolation='nearest', resample=False)
    axBG.set_xlim(-0.5, Wbg - 0.5)
    axBG.set_ylim(-0.5, Hbg - 0.5)
    axBG.set_aspect('equal', adjustable='box')
    axBG.set_box_aspect(Hbg / Wbg)
    axBG.set_anchor('S')
    axBG.set_xticks([])
    axBG.set_yticks([])

    # Center marker + label: bottom + right guide lines
    center_x_bg, center_y_bg = Wbg / 2.0, Hbg / 2.0
    _draw_sn_bottom_right_guides(axBG, center_x_bg, center_y_bg, Wbg, Hbg, lw=2.5)
    axBG.text(center_x_bg + 20, center_y_bg + 20, 'SN 2024vjm', color='black',
              weight='bold', fontsize=18)

    # Compass (top-left)
    c_len = Hbg * 0.15
    cx, cy = Wbg * 0.2, Hbg * 0.75
    axBG.plot([cx, cx], [cy, cy + c_len], color='black', lw=4)
    axBG.text(cx, cy + c_len * 1.2, 'N', color='black', fontsize=20,
              ha='center', weight='bold')
    axBG.plot([cx, cx - c_len], [cy, cy], color='black', lw=4)
    axBG.text(cx - c_len * 1.2, cy, 'E', color='black', fontsize=20,
              va='center', weight='bold')

    # Scale bar (5 kpc)
    d_pc = 9.4e6
    bar_pc = 5000
    bar_len_pix_bg = (bar_pc / d_pc * 206265.0) / pixel_scale_bg
    bx = Wbg * 0.85
    by = Hbg * 0.02
    axBG.plot([bx - bar_len_pix_bg / 2, bx + bar_len_pix_bg / 2], [by, by],
              color='black', lw=4)
    axBG.text(bx, by + 15, '5 kpc', color='black', fontsize=18,
              ha='center', va='bottom', weight='bold')

    # -------------------- Euclid large panel draw --------------------
    axL.imshow(big, origin='lower', cmap='gray_r', vmin=vmin_big, vmax=vmax_big,
               interpolation='nearest', resample=False)
    axL.set_xlim(-0.5, Wb - 0.5)
    axL.set_ylim(-0.5, Hb - 0.5)
    axL.set_aspect('equal', adjustable='box')
    axL.set_box_aspect(Hb / Wb)
    axL.set_anchor('S')
    axL.set_xticks([])
    axL.set_yticks([])
    axL.add_patch(Rectangle((x_min_e, y_min_e), (x_max_e - x_min_e), (y_max_e - y_min_e),
                            linewidth=2.5, edgecolor=PLOT_COLORS['box'], facecolor='none'))

    # Large compass (top-left)
    m = 0.06
    size_frac = 0.14
    comp_size = int(size_frac * min(Hb, Wb))
    comp_size = max(comp_size, 40)
    cx = 1000
    cy_top = int(Hb * (1.0 - m))
    cy = cy_top - comp_size
    axL.plot([cx, cx], [cy, cy + comp_size], color='black', lw=3)
    axL.text(cx, cy + comp_size + max(14, comp_size * 0.12), 'N',
             color='black', fontsize=16, ha='center', va='bottom')
    axL.plot([cx, cx - comp_size], [cy, cy], color='black', lw=3)
    axL.text(cx - comp_size - max(10, comp_size * 0.12), cy, 'E',
             color='black', fontsize=16, ha='center', va='center')

    # Distance bar (5 kpc)
    bar_arcsec = bar_pc / d_pc * 206265.0
    bar_pix = bar_arcsec / pixel_scale_e
    y_bar = 100
    x_bar_center = int(Wb * 0.85)
    x_bar0 = x_bar_center - bar_pix / 2.0
    x_bar1 = x_bar_center + bar_pix / 2.0
    axL.plot([x_bar0, x_bar1], [y_bar, y_bar], color='black', lw=3)
    axL.text((x_bar0 + x_bar1) / 2.0, y_bar + 10, '5 kpc',
             color='black', fontsize=16, ha='center', va='bottom')

    # -------------------- Euclid zoom panel draw --------------------
    axR.imshow(cut, origin='lower', cmap='gray_r', vmin=vmin_cut, vmax=vmax_cut,
               interpolation='nearest', resample=False)
    axR.set_xlim(-0.5, Wc - 0.5)
    axR.set_ylim(-0.5, Hc - 0.5)
    axR.set_aspect('equal', adjustable='box')
    axR.set_box_aspect(Hc / Wc)
    axR.set_anchor('N')
    axR.set_xticks([])
    axR.set_yticks([])

    # SN marker + label
    center_x = x_center - x0_zoom
    center_y = y_center - y0_zoom
    axR.add_patch(plt.Circle((center_x, center_y),
                             circle_radius_sn_pix, color=PLOT_COLORS['sn'], fill=False, lw=2, zorder=10))
    axR.text(center_x + 5.5, center_y + 1.5, 'SN 2024vjm',
             color=PLOT_COLORS['sn'], fontsize=20, ha='center', va='center', weight='bold')

    # Catalog filtering
    half_width_arcsec = half_size_x * pixel_scale_e
    half_height_arcsec = half_size_y * pixel_scale_e
    cos_dec = np.cos(np.deg2rad(dec_center))
    ra_min = ra_center - (half_width_arcsec / 3600.0) / cos_dec
    ra_max = ra_center + (half_width_arcsec / 3600.0) / cos_dec
    dec_min = dec_center - half_height_arcsec / 3600.0
    dec_max = dec_center + half_height_arcsec / 3600.0
    filtered_catalog = euclid_catalog[
        (euclid_catalog['RA'] >= ra_min) & (euclid_catalog['RA'] <= ra_max) &
        (euclid_catalog['Dec'] >= dec_min) & (euclid_catalog['Dec'] <= dec_max)
    ]
    filtered_catalog = filtered_catalog[filtered_catalog['MAG_AUTO'] < 99]

    # Nearest source to center
    # Nearest source to the SN position inside the zoom cutout
    nearest_index, nearest_distance = None, float('inf')
    nearest_row = None
    for idx, row in filtered_catalog.iterrows():
        xs, ys = w_e.world_to_pixel(SkyCoord(row['RA'], row['Dec'], unit='deg'))
        x_cut = xs - x0_zoom
        y_cut = ys - y0_zoom
        if 0 <= x_cut <= Wc and 0 <= y_cut <= Hc:
            d = np.hypot(x_cut - center_x, y_cut - center_y)
            if d < nearest_distance:
                nearest_distance, nearest_index = d, idx
                nearest_row = row

    _print_nearest_source_row(nearest_row, prefix='Nearest source (BG|Euclid horizontal zoom)')

    # Draw catalog circles + S1
    for idx, row in filtered_catalog.iterrows():
        xs, ys = w_e.world_to_pixel(SkyCoord(row['RA'], row['Dec'], unit='deg'))
        x_cut = xs - x0_zoom
        y_cut = ys - y0_zoom
        if 0 <= x_cut <= Wc and 0 <= y_cut <= Hc:
            color = PLOT_COLORS['s1'] if idx == nearest_index else PLOT_COLORS['catalog']
            lw = 2 if idx == nearest_index else 1
            
            # For S1, draw an ellipse based on its extended size
            if idx == nearest_index:
                # Get ellipse parameters from catalog
                a_img = row['A_IMAGE']  # Semi-major axis in pixels
                b_img = row['B_IMAGE']  # Semi-minor axis in pixels
                theta = row['THETA_IMAGE']  # Position angle in degrees
                
                # Scale factor to show the full extent (using 3 sigma like inject_and_measure.py)
                scale_factor = 3.0
                
                # Draw ellipse (width and height are full diameters)
                ellipse = Ellipse((x_cut, y_cut), 
                                width=2*a_img*scale_factor, 
                                height=2*b_img*scale_factor,
                                angle=theta, 
                                edgecolor=color, 
                                facecolor='none', 
                                lw=lw)
                axR.add_patch(ellipse)
                axR.text(x_cut - a_img*scale_factor, y_cut, 'S1',
                         color=PLOT_COLORS['s1'], fontsize=16, ha='right', va='center', weight='bold')
            else:
                # For other catalog sources, use circle as before
                axR.add_patch(plt.Circle((x_cut, y_cut), circle_radius_pix,
                                         color=color, fill=False, lw=lw))

    # Right-panel distance bar (50 pc) and compass (small)
    bar_pc_r = 50
    bar_arcsec_r = bar_pc_r / d_pc * 206265.0
    bar_pix_r = bar_arcsec_r / pixel_scale_e
    axR.plot([Wc - 2 - bar_pix_r, Wc - 2], [2, 2], color='black', lw=2)
    axR.text(Wc - 2 - bar_pix_r / 2.0, 2, f'{bar_pc_r} pc',
             color='black', fontsize=16, ha='center', va='bottom')
    axR.plot([13, 13], [2, 2 + 8], color='black', lw=2)
    axR.text(13, 2 + 10, 'N', color='black', fontsize=16, ha='center', va='bottom')
    axR.plot([13, 13 - 10], [2, 2], color='black', lw=2)
    axR.text(13 - 12, 2, 'E', color='black', fontsize=16, ha='center', va='center')

    # Euclid dashed red connectors (between middle and right panels)
    dash = (0, (8, 6))
    for (cx0, cy0), (zx, zy) in [((x_max_e, y_max_e), (0, Hc)), ((x_max_e, y_min_e), (0, 0))]:
        fig.add_artist(ConnectionPatch(
            xyA=(cx0, cy0), coordsA=axL.transData,
            xyB=(zx, zy), coordsB=axR.transData,
            color=PLOT_COLORS['connector'], lw=2.2, linestyle=dash, alpha=0.95
        ))

    if savepath:
        plt.savefig(savepath, dpi=300, bbox_inches='tight', pad_inches=0)
        print(f"Saved to {savepath}")
    plt.show()


def bg_above_euclid_two_images(
    bg_fits_file,
    euclid_fits_file,
    euclid_catalog,
    ra_center,
    dec_center,
    bg_fov_arcmin=10.0,
    euclid_cutout_size_arcsec=10.0,
    euclid_circle_radius_arcsec=0.1,
    euclid_large_cutout_deg=1.0,
    figsize=(9, 27),
    savepath=None,
):
    """Create a 3-panel figure stacked vertically.

    Panel order (top -> bottom):
      a) BlackGEM discovery image
      b) Euclid large cutout w/ overlays
      c) Euclid 10\" zoom
    """

    # -------------------- BlackGEM panel (top) --------------------
    with fits.open(bg_fits_file) as hdul:
        header_bg = hdul[0].header
        w_bg = wcs.WCS(header_bg)
        data_bg = np.squeeze(hdul[0].data)

        if 'A-PSCALE' in header_bg:
            pixel_scale_bg = header_bg['A-PSCALE']
        else:
            pixel_scale_bg = np.sqrt(np.abs(np.linalg.det(w_bg.pixel_scale_matrix))) * 3600.0

    target_coord = SkyCoord(ra_center, dec_center, unit='deg')
    fov_pix = int((bg_fov_arcmin * 60.0) / pixel_scale_bg)
    if fov_pix % 2 != 0:
        fov_pix += 1
    half_size_bg = fov_pix // 2

    x_bg, y_bg = w_bg.world_to_pixel(target_coord)
    x_bg, y_bg = int(x_bg), int(y_bg)

    y_min = max(0, y_bg - half_size_bg)
    y_max = min(data_bg.shape[0], y_bg + half_size_bg)
    x_min = max(0, x_bg - half_size_bg)
    x_max = min(data_bg.shape[1], x_bg + half_size_bg)

    raw_cutout_bg = data_bg[y_min:y_max, x_min:x_max]
    bg_img = np.rot90(raw_cutout_bg, -1)
    vmin_bg, vmax_bg = ZScaleInterval().get_limits(bg_img)

    # -------------------- Euclid panels (middle + bottom) --------------------
    with fits.open(euclid_fits_file) as hdul:
        w_e = wcs.WCS(hdul[0].header)
        data_e = np.squeeze(hdul[0].data)
        pixel_scale_e = np.abs(hdul[0].header['CD2_2']) * 3600.0  # arcsec/pixel

    cutout_size_pix = euclid_cutout_size_arcsec / pixel_scale_e
    circle_radius_pix = euclid_circle_radius_arcsec / pixel_scale_e
    circle_radius_sn_pix = (50e-3) / pixel_scale_e  # 50 mas

    x_center, y_center = w_e.world_to_pixel(SkyCoord(ra_center, dec_center, unit='deg'))
    x_center, y_center = int(x_center), int(y_center)

    large_cutout_pix = (euclid_large_cutout_deg * 3600.0) / pixel_scale_e
    half_large = int(large_cutout_pix / 2)
    x0 = max(0, x_center - half_large)
    x1 = min(data_e.shape[1], x_center + half_large)
    y0 = max(0, y_center - half_large)
    y1 = min(data_e.shape[0], y_center + half_large)
    big = data_e[y0:y1, x0:x1]

    xc, yc = x_center - x0, y_center - y0

    # For the hybrid layout we want the bottom zoom panel to be ~2× wider than tall
    # without distorting pixels. Do that by using a rectangular cutout (2× width).
    zoom_height_pix = int(round(cutout_size_pix))
    if zoom_height_pix % 2 != 0:
        zoom_height_pix += 1
    zoom_width_pix = 2 * zoom_height_pix
    half_size_y = zoom_height_pix // 2
    half_size_x = zoom_width_pix // 2

    y0_zoom = max(0, y_center - half_size_y)
    y1_zoom = min(data_e.shape[0], y_center + half_size_y)
    x0_zoom = max(0, x_center - half_size_x)
    x1_zoom = min(data_e.shape[1], x_center + half_size_x)
    cut = data_e[y0_zoom:y1_zoom, x0_zoom:x1_zoom]

    vmax_big = ZScaleInterval(contrast=0.15).get_limits(big)[1]
    vmin_big = -2.0
    vmax_big = 15
    vmin_cut, vmax_cut = ZScaleInterval().get_limits(cut)

    # Red box on the big Euclid panel should match the zoom panel region
    x_min_e, x_max_e = xc - half_size_x, xc + half_size_x
    y_min_e, y_max_e = yc - half_size_y, yc + half_size_y

    Hb, Wb = big.shape
    Hc, Wc = cut.shape
    Hbg, Wbg = bg_img.shape

    # -------------------- Figure layout --------------------
    fig = plt.figure(figsize=figsize)
    gs = plt.GridSpec(
        3,
        1,
        height_ratios=[Hbg / Wbg, Hb / Wb, Hc / Wc],
    )
    axBG = fig.add_subplot(gs[0, 0])
    axL = fig.add_subplot(gs[1, 0])
    axR = fig.add_subplot(gs[2, 0])
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0, hspace=0)

    # Panel labels
    axBG.text(0.03, 0.97, 'a', transform=axBG.transAxes,
              ha='left', va='top', fontsize=22, color='black', weight='bold')
    axL.text(0.03, 0.97, 'b', transform=axL.transAxes,
             ha='left', va='top', fontsize=22, color='black', weight='bold')
    axR.text(0.03, 0.97, 'c', transform=axR.transAxes,
             ha='left', va='top', fontsize=22, color='black', weight='bold')

    # -------------------- BG panel draw --------------------
    axBG.imshow(bg_img, origin='lower', cmap='gray_r', vmin=vmin_bg, vmax=vmax_bg,
                interpolation='nearest', resample=False)
    axBG.set_xlim(-0.5, Wbg - 0.5)
    axBG.set_ylim(-0.5, Hbg - 0.5)
    axBG.set_aspect('equal', adjustable='box')
    axBG.set_box_aspect(Hbg / Wbg)
    axBG.set_xticks([])
    axBG.set_yticks([])

    center_x_bg, center_y_bg = Wbg / 2.0, Hbg / 2.0
    _draw_sn_bottom_right_guides(axBG, center_x_bg, center_y_bg, Wbg, Hbg, lw=2.5)
    axBG.text(center_x_bg + 20, center_y_bg + 20, 'SN 2024vjm', color='black',
              weight='bold', fontsize=18)

    c_len = Hbg * 0.15
    cx, cy = Wbg * 0.2, Hbg * 0.75
    axBG.plot([cx, cx], [cy, cy + c_len], color='black', lw=4)
    axBG.text(cx, cy + c_len * 1.2, 'N', color='black', fontsize=20,
              ha='center', weight='bold')
    axBG.plot([cx, cx - c_len], [cy, cy], color='black', lw=4)
    axBG.text(cx - c_len * 1.2, cy, 'E', color='black', fontsize=20,
              va='center', weight='bold')

    d_pc = 9.4e6
    bar_pc = 5000
    bar_len_pix_bg = (bar_pc / d_pc * 206265.0) / pixel_scale_bg
    bx = Wbg * 0.85
    by = Hbg * 0.02
    axBG.plot([bx - bar_len_pix_bg / 2, bx + bar_len_pix_bg / 2], [by, by],
              color='black', lw=4)
    axBG.text(bx, by + 15, '5 kpc', color='black', fontsize=18,
              ha='center', va='bottom', weight='bold')

    # -------------------- Euclid large panel draw --------------------
    axL.imshow(big, origin='lower', cmap='gray_r', vmin=vmin_big, vmax=vmax_big,
               interpolation='nearest', resample=False)
    axL.set_xlim(-0.5, Wb - 0.5)
    axL.set_ylim(-0.5, Hb - 0.5)
    axL.set_aspect('equal', adjustable='box')
    axL.set_box_aspect(Hb / Wb)
    axL.set_xticks([])
    axL.set_yticks([])
    axL.add_patch(Rectangle((x_min_e, y_min_e), (x_max_e - x_min_e), (y_max_e - y_min_e),
                            linewidth=2.5, edgecolor=PLOT_COLORS['box'], facecolor='none'))

    m = 0.06
    size_frac = 0.14
    comp_size = int(size_frac * min(Hb, Wb))
    comp_size = max(comp_size, 40)
    cx = 1000
    cy_top = int(Hb * (1.0 - m))
    cy = cy_top - comp_size
    axL.plot([cx, cx], [cy, cy + comp_size], color='black', lw=3)
    axL.text(cx, cy + comp_size + max(14, comp_size * 0.12), 'N',
             color='black', fontsize=16, ha='center', va='bottom')
    axL.plot([cx, cx - comp_size], [cy, cy], color='black', lw=3)
    axL.text(cx - comp_size - max(10, comp_size * 0.12), cy, 'E',
             color='black', fontsize=16, ha='center', va='center')

    bar_arcsec = bar_pc / d_pc * 206265.0
    bar_pix = bar_arcsec / pixel_scale_e
    y_bar = 100
    x_bar_center = int(Wb * 0.85)
    x_bar0 = x_bar_center - bar_pix / 2.0
    x_bar1 = x_bar_center + bar_pix / 2.0
    axL.plot([x_bar0, x_bar1], [y_bar, y_bar], color='black', lw=3)
    axL.text((x_bar0 + x_bar1) / 2.0, y_bar + 10, '5 kpc',
             color='black', fontsize=16, ha='center', va='bottom')

    # -------------------- Euclid zoom panel draw --------------------
    axR.imshow(cut, origin='lower', cmap='gray_r', vmin=vmin_cut, vmax=vmax_cut,
               interpolation='nearest', resample=False)
    axR.set_xlim(-0.5, Wc - 0.5)
    axR.set_ylim(-0.5, Hc - 0.5)
    axR.set_aspect('equal', adjustable='box')
    axR.set_box_aspect(Hc / Wc)
    axR.set_xticks([])
    axR.set_yticks([])

    axR.add_patch(plt.Circle((cutout_size_pix / 2, cutout_size_pix / 2),
                             circle_radius_sn_pix, color=PLOT_COLORS['sn'], fill=False, lw=2, zorder=10))
    axR.text(cutout_size_pix / 2 + 5.5, cutout_size_pix / 2 + 1.5, 'SN 2024vjm',
             color=PLOT_COLORS['sn'], fontsize=20, ha='center', va='center', weight='bold')

    cos_dec = np.cos(np.deg2rad(dec_center))
    ra_min = ra_center - (euclid_cutout_size_arcsec / 3600.0) / cos_dec
    ra_max = ra_center + (euclid_cutout_size_arcsec / 3600.0) / cos_dec
    dec_min = dec_center - euclid_cutout_size_arcsec / 3600.0
    dec_max = dec_center + (euclid_cutout_size_arcsec / 2.0) / 3600.0
    filtered_catalog = euclid_catalog[
        (euclid_catalog['RA'] >= ra_min) & (euclid_catalog['RA'] <= ra_max) &
        (euclid_catalog['Dec'] >= dec_min) & (euclid_catalog['Dec'] <= dec_max)
    ]
    filtered_catalog = filtered_catalog[filtered_catalog['MAG_AUTO'] < 99]

    center_x, center_y = Wc / 2, Hc / 2
    nearest_index, nearest_distance = None, float('inf')
    nearest_row = None
    for idx, row in filtered_catalog.iterrows():
        xs, ys = w_e.world_to_pixel(SkyCoord(row['RA'], row['Dec'], unit='deg'))
        x_cut = xs - x0_zoom
        y_cut = ys - y0_zoom
        if 0 <= x_cut <= Wc and 0 <= y_cut <= Hc:
            d = np.hypot(x_cut - center_x, y_cut - center_y)
            if d < nearest_distance:
                nearest_distance, nearest_index = d, idx
                nearest_row = row

    _print_nearest_source_row(nearest_row, prefix='Nearest source (BG above Euclid zoom)')

    for idx, row in filtered_catalog.iterrows():
        xs, ys = w_e.world_to_pixel(SkyCoord(row['RA'], row['Dec'], unit='deg'))
        x_cut = xs - x0_zoom
        y_cut = ys - y0_zoom
        if 0 <= x_cut <= Wc and 0 <= y_cut <= Hc:
            color = PLOT_COLORS['s1'] if idx == nearest_index else PLOT_COLORS['catalog']
            lw = 2 if idx == nearest_index else 1
            
            # For S1, draw an ellipse based on its extended size
            if idx == nearest_index:
                # Get ellipse parameters from catalog
                a_img = row['A_IMAGE']  # Semi-major axis in pixels
                b_img = row['B_IMAGE']  # Semi-minor axis in pixels
                theta = row['THETA_IMAGE']  # Position angle in degrees
                
                # Scale factor to show the full extent (using 3 sigma like inject_and_measure.py)
                scale_factor = 3.0
                
                # Draw ellipse (width and height are full diameters)
                ellipse = Ellipse((x_cut, y_cut), 
                                width=2*a_img*scale_factor, 
                                height=2*b_img*scale_factor,
                                angle=theta, 
                                edgecolor=color, 
                                facecolor='none', 
                                lw=lw)
                axR.add_patch(ellipse)
                axR.text(x_cut - a_img*scale_factor, y_cut, 'S1',
                         color=PLOT_COLORS['s1'], fontsize=16, ha='right', va='center', weight='bold')
            else:
                # For other catalog sources, use circle as before
                axR.add_patch(plt.Circle((x_cut, y_cut), circle_radius_pix,
                                         color=color, fill=False, lw=lw))


    bar_pc_r = 50
    bar_arcsec_r = bar_pc_r / d_pc * 206265.0
    bar_pix_r = bar_arcsec_r / pixel_scale_e
    axR.plot([Wc - 2 - bar_pix_r, Wc - 2], [2, 2], color='black', lw=2)
    axR.text(Wc - 2 - bar_pix_r / 2.0, 2, f'{bar_pc_r} pc',
             color='black', fontsize=16, ha='center', va='bottom')
    axR.plot([13, 13], [2, 2 + 8], color='black', lw=2)
    axR.text(13, 2 + 10, 'N', color='black', fontsize=16, ha='center', va='bottom')
    axR.plot([13, 13 - 10], [2, 2], color='black', lw=2)
    axR.text(13 - 12, 2, 'E', color='black', fontsize=16, ha='center', va='center')

    # Dashed red connectors between Euclid middle and bottom panels
    # Connect the *bottom* corners of the red box to the *top* corners of the zoom panel.
    dash = (0, (8, 6))
    for (cx0, cy0), (zx, zy) in [
        ((x_min_e, y_min_e), (0, Hc)),
        ((x_max_e, y_min_e), (Wc, Hc)),
    ]:
        fig.add_artist(ConnectionPatch(
            xyA=(cx0, cy0), coordsA=axL.transData,
            xyB=(zx, zy), coordsB=axR.transData,
            color=PLOT_COLORS['connector'], lw=2.2, linestyle=dash, alpha=0.95
        ))

    if savepath:
        plt.savefig(savepath, dpi=300, bbox_inches='tight', pad_inches=0)
        print(f"Saved to {savepath}")
    plt.show()


def bg_and_big_euclid_on_top_zoom_bottom(
    bg_fits_file,
    euclid_fits_file,
    euclid_catalog,
    ra_center,
    dec_center,
    bg_fov_arcmin=10.0,
    euclid_cutout_size_arcsec=10.0,
    euclid_circle_radius_arcsec=0.1,
    euclid_large_cutout_deg=1.0,
    figsize=(18, 18),
    savepath=None,
):
    """Create a 3-panel figure with 2 panels on top and a wide zoom on bottom.

    Layout:
      a) BlackGEM discovery image (top-left)
      b) Euclid large cutout w/ overlays (top-right)
      c) Euclid 10\" zoom spanning both columns (bottom)
    """

    # -------------------- BlackGEM image prep --------------------
    with fits.open(bg_fits_file) as hdul:
        header_bg = hdul[0].header
        w_bg = wcs.WCS(header_bg)
        data_bg = np.squeeze(hdul[0].data)

        if 'A-PSCALE' in header_bg:
            pixel_scale_bg = header_bg['A-PSCALE']
        else:
            pixel_scale_bg = np.sqrt(np.abs(np.linalg.det(w_bg.pixel_scale_matrix))) * 3600.0

    target_coord = SkyCoord(ra_center, dec_center, unit='deg')
    fov_pix = int((bg_fov_arcmin * 60.0) / pixel_scale_bg)
    if fov_pix % 2 != 0:
        fov_pix += 1
    half_size_bg = fov_pix // 2

    x_bg, y_bg = w_bg.world_to_pixel(target_coord)
    x_bg, y_bg = int(x_bg), int(y_bg)

    y_min = max(0, y_bg - half_size_bg)
    y_max = min(data_bg.shape[0], y_bg + half_size_bg)
    x_min = max(0, x_bg - half_size_bg)
    x_max = min(data_bg.shape[1], x_bg + half_size_bg)

    raw_cutout_bg = data_bg[y_min:y_max, x_min:x_max]
    bg_img = np.rot90(raw_cutout_bg, -1)
    vmin_bg, vmax_bg = ZScaleInterval().get_limits(bg_img)
    Hbg, Wbg = bg_img.shape

    # -------------------- Euclid image prep --------------------
    with fits.open(euclid_fits_file) as hdul:
        w_e = wcs.WCS(hdul[0].header)
        data_e = np.squeeze(hdul[0].data)
        pixel_scale_e = np.abs(hdul[0].header['CD2_2']) * 3600.0  # arcsec/pixel

    cutout_size_pix = euclid_cutout_size_arcsec / pixel_scale_e
    circle_radius_pix = euclid_circle_radius_arcsec / pixel_scale_e
    circle_radius_sn_pix = (50e-3) / pixel_scale_e  # 50 mas

    x_center, y_center = w_e.world_to_pixel(SkyCoord(ra_center, dec_center, unit='deg'))
    x_center, y_center = int(x_center), int(y_center)

    large_cutout_pix = (euclid_large_cutout_deg * 3600.0) / pixel_scale_e
    half_large = int(large_cutout_pix / 2)
    x0 = max(0, x_center - half_large)
    x1 = min(data_e.shape[1], x_center + half_large)
    y0 = max(0, y_center - half_large)
    y1 = min(data_e.shape[0], y_center + half_large)
    big = data_e[y0:y1, x0:x1]

    xc, yc = x_center - x0, y_center - y0

    # For the hybrid layout we want the bottom zoom panel to be ~2× wider than tall
    # without distorting pixels. Do that by using a rectangular cutout (2× width).
    zoom_height_pix = int(round(cutout_size_pix))
    if zoom_height_pix % 2 != 0:
        zoom_height_pix += 1
    zoom_width_pix = 2 * zoom_height_pix
    half_size_y = zoom_height_pix // 2
    half_size_x = zoom_width_pix // 2

    y0_zoom = max(0, y_center - half_size_y)
    y1_zoom = min(data_e.shape[0], y_center + half_size_y)
    x0_zoom = max(0, x_center - half_size_x)
    x1_zoom = min(data_e.shape[1], x_center + half_size_x)
    cut = data_e[y0_zoom:y1_zoom, x0_zoom:x1_zoom]

    vmax_big = ZScaleInterval(contrast=0.15).get_limits(big)[1]
    vmin_big = -2.0
    vmax_big = 15
    vmin_cut, vmax_cut = ZScaleInterval().get_limits(cut)

    # Red box on the big Euclid panel should match the zoom panel region
    x_min_e, x_max_e = xc - half_size_x, xc + half_size_x
    y_min_e, y_max_e = yc - half_size_y, yc + half_size_y

    Hb, Wb = big.shape
    Hc, Wc = cut.shape

    # -------------------- Figure layout --------------------
    # Use explicit axes positions to avoid GridSpec + equal-aspect leaving internal gaps.
    col0 = Wbg / Hbg
    col1 = Wb / Hb
    colsum = col0 + col1
    top_height_w = 1.0 / colsum
    bottom_height_w = (Hc / Wc)
    total_height_w = top_height_w + bottom_height_w

    fig_width = float(figsize[0])
    fig_height = fig_width * total_height_w
    fig = plt.figure(figsize=(fig_width, fig_height))

    top_frac = top_height_w / total_height_w
    bottom_frac = bottom_height_w / total_height_w
    w0 = col0 / colsum
    w1 = col1 / colsum

    axBG = fig.add_axes([0.0, bottom_frac, w0, top_frac])
    axL = fig.add_axes([w0, bottom_frac, w1, top_frac])
    axR = fig.add_axes([0.0, 0.0, 1.0, bottom_frac])

    # Panel labels
    axBG.text(0.03, 0.97, 'a', transform=axBG.transAxes,
              ha='left', va='top', fontsize=ns.FS_PANEL, color='black', weight='bold')
    axL.text(0.03, 0.97, 'b', transform=axL.transAxes,
             ha='left', va='top', fontsize=ns.FS_PANEL, color='black', weight='bold')
    axR.text(0.03, 0.97, 'c', transform=axR.transAxes,
             ha='left', va='top', fontsize=ns.FS_PANEL, color='black', weight='bold')

    # -------------------- BG panel draw --------------------
    axBG.imshow(bg_img, origin='lower', cmap='gray_r', vmin=vmin_bg, vmax=vmax_bg,
                interpolation='nearest', resample=False)
    axBG.set_xlim(-0.5, Wbg - 0.5)
    axBG.set_ylim(-0.5, Hbg - 0.5)
    axBG.set_aspect('equal', adjustable='box')
    axBG.set_xticks([])
    axBG.set_yticks([])

    center_x_bg, center_y_bg = Wbg / 2.0, Hbg / 2.0
    _draw_sn_bottom_right_guides(axBG, center_x_bg, center_y_bg, Wbg, Hbg, lw=STD_LW)
    axBG.text(center_x_bg + 20, center_y_bg + 10, 'SN 2024vjm', color='black',
              weight='bold', fontsize=STD_FONTSIZE)

    c_len = Hbg * 0.15
    cx, cy = Wbg * 0.2, Hbg * 0.75
    axBG.plot([cx, cx], [cy, cy + c_len], color='black', lw=STD_LW)
    axBG.text(cx, cy + c_len * 1.2, 'N', color='black', fontsize=STD_FONTSIZE,
              ha='center', weight='bold')
    axBG.plot([cx, cx - c_len], [cy, cy], color='black', lw=STD_LW)
    axBG.text(cx - c_len * 1.2, cy, 'E', color='black', fontsize=STD_FONTSIZE,
              va='center', weight='bold')

    d_pc = 9.4e6
    bar_pc = 5000
    bar_len_pix_bg = (bar_pc / d_pc * 206265.0) / pixel_scale_bg
    bx = Wbg * 0.85
    by = Hbg * 0.02
    axBG.plot([bx - bar_len_pix_bg / 2, bx + bar_len_pix_bg / 2], [by, by],
              color='black', lw=STD_LW)
    axBG.text(bx, by + 15, '5 kpc', color='black', fontsize=STD_FONTSIZE,
              ha='center', va='bottom', weight='bold')

    # -------------------- Euclid large panel draw --------------------
    axL.imshow(big, origin='lower', cmap='gray_r', vmin=vmin_big, vmax=vmax_big,
               interpolation='nearest', resample=False)
    axL.set_xlim(-0.5, Wb - 0.5)
    axL.set_ylim(-0.5, Hb - 0.5)
    axL.set_aspect('equal', adjustable='box')
    axL.set_xticks([])
    axL.set_yticks([])
    axL.add_patch(Rectangle((x_min_e, y_min_e), (x_max_e - x_min_e), (y_max_e - y_min_e),
                            linewidth=STD_LW, edgecolor=PLOT_COLORS['box'], facecolor='none'))

    m = 0.06
    size_frac = 0.14
    comp_size = int(size_frac * min(Hb, Wb))
    comp_size = max(comp_size, 40)
    cx = 1000
    cy_top = int(Hb * (1.0 - m))
    cy = cy_top - comp_size
    axL.plot([cx, cx], [cy, cy + comp_size], color='black', lw=STD_LW)
    axL.text(cx, cy + comp_size + max(14, comp_size * 0.12), 'N',
             color='black', fontsize=STD_FONTSIZE, ha='center', va='bottom', weight='bold')
    axL.plot([cx, cx - comp_size], [cy, cy], color='black', lw=STD_LW)
    axL.text(cx - comp_size - max(10, comp_size * 0.12), cy, 'E',
             color='black', fontsize=STD_FONTSIZE, ha='center', va='center', weight='bold')

    bar_arcsec = bar_pc / d_pc * 206265.0
    bar_pix = bar_arcsec / pixel_scale_e
    y_bar = 100
    x_bar_center = int(Wb * 0.85)
    x_bar0 = x_bar_center - bar_pix / 2.0
    x_bar1 = x_bar_center + bar_pix / 2.0
    axL.plot([x_bar0, x_bar1], [y_bar, y_bar], color='black', lw=STD_LW)
    axL.text((x_bar0 + x_bar1) / 2.0, y_bar + 10, '5 kpc',
             color='black', fontsize=STD_FONTSIZE, ha='center', va='bottom', weight='bold')

    # -------------------- Euclid zoom panel draw (wide) --------------------
    axR.imshow(cut, origin='lower', cmap='gray_r', vmin=vmin_cut, vmax=vmax_cut,
               interpolation='nearest', resample=False)
    axR.set_xlim(-0.5, Wc - 0.5)
    axR.set_ylim(-0.5, Hc - 0.5)
    axR.set_aspect('equal', adjustable='box')
    axR.set_xticks([])
    axR.set_yticks([])

    center_x = x_center - x0_zoom
    center_y = y_center - y0_zoom
    axR.add_patch(plt.Circle((center_x, center_y),
                             circle_radius_sn_pix, color=PLOT_COLORS['sn'], fill=False, lw=STD_LW, zorder=10))
    axR.text(center_x + 5.5, center_y + 3, 'SN 2024vjm',
             color=PLOT_COLORS['sn'], fontsize=STD_FONTSIZE, ha='center', va='center', weight='bold')

    half_width_arcsec = half_size_x * pixel_scale_e
    half_height_arcsec = half_size_y * pixel_scale_e
    cos_dec = np.cos(np.deg2rad(dec_center))
    ra_min = ra_center - (half_width_arcsec / 3600.0) / cos_dec
    ra_max = ra_center + (half_width_arcsec / 3600.0) / cos_dec
    dec_min = dec_center - half_height_arcsec / 3600.0
    dec_max = dec_center + half_height_arcsec / 3600.0
    filtered_catalog = euclid_catalog[
        (euclid_catalog['RA'] >= ra_min) & (euclid_catalog['RA'] <= ra_max) &
        (euclid_catalog['Dec'] >= dec_min) & (euclid_catalog['Dec'] <= dec_max)
    ]
    filtered_catalog = filtered_catalog[filtered_catalog['MAG_AUTO'] < 99]

    if not filtered_catalog.empty:
        faintest_row = filtered_catalog.loc[filtered_catalog['MAG_AUTO'].idxmax()]
        print("\n--- Faintest Source in Frame ---")
        print(faintest_row[['RA', 'Dec', 'MAG_AUTO', 'CLASS_STAR']])
        print("--------------------------------\n")

    nearest_index, nearest_distance = None, float('inf')
    nearest_row = None
    for idx, row in filtered_catalog.iterrows():
        xs, ys = w_e.world_to_pixel(SkyCoord(row['RA'], row['Dec'], unit='deg'))
        x_cut = xs - x0_zoom
        y_cut = ys - y0_zoom
        if 0 <= x_cut <= Wc and 0 <= y_cut <= Hc:
            d = np.hypot(x_cut - center_x, y_cut - center_y)
            if d < nearest_distance:
                nearest_distance, nearest_index = d, idx
                nearest_row = row

    _print_nearest_source_row(nearest_row, prefix='Nearest source (hybrid wide zoom)')

    for idx, row in filtered_catalog.iterrows():
        xs, ys = w_e.world_to_pixel(SkyCoord(row['RA'], row['Dec'], unit='deg'))
        x_cut = xs - x0_zoom
        y_cut = ys - y0_zoom
        if 0 <= x_cut <= Wc and 0 <= y_cut <= Hc:
            # Color Logic
            mag = row['MAG_AUTO']
            s1_mag = 25.8342
            
            if idx == nearest_index:
                color = PLOT_COLORS['s1'] # Keep S1 special
            elif mag > 26.9:
                color = PLOT_COLORS['catalog'] # Fainter than 26.9
            elif mag > s1_mag:
                color =  PLOT_COLORS['catalog'] # Fainter than S1
            else:
                color =  PLOT_COLORS['catalog'] # Brighter (Different color for standard sources)

            lw = STD_LW if idx == nearest_index else 0.5   # [rev3] was 1
            
            # For S1, draw an ellipse based on its extended size
            if idx == nearest_index:
                # Get ellipse parameters from catalog
                a_img = row['A_IMAGE']  # Semi-major axis in pixels
                b_img = row['B_IMAGE']  # Semi-minor axis in pixels
                theta = row['THETA_IMAGE']  # Position angle in degrees
                
                # Scale factor to show the full extent (using 3 sigma like inject_and_measure.py)
                scale_factor = 3.0
                
                # Draw ellipse (width and height are full diameters)
                ellipse = Ellipse((x_cut, y_cut), 
                                width=2*a_img*scale_factor, 
                                height=2*b_img*scale_factor,
                                angle=theta, 
                                edgecolor=color, 
                                facecolor='none', 
                                lw=lw)
                axR.add_patch(ellipse)
                axR.text(x_cut - a_img*scale_factor, y_cut, 'S1',
                         color=PLOT_COLORS['s1'], fontsize=STD_FONTSIZE, ha='right', va='center', weight='bold')
            else:
                # For other catalog sources, use circle as before
                axR.add_patch(plt.Circle((x_cut, y_cut), circle_radius_pix,
                                         color=color, fill=False, lw=lw))


    bar_pc_r = 50
    bar_arcsec_r = bar_pc_r / d_pc * 206265.0
    bar_pix_r = bar_arcsec_r / pixel_scale_e
    axR.plot([Wc - 2 - bar_pix_r, Wc - 2], [2, 2], color='black', lw=STD_LW)
    axR.text(Wc - 2 - bar_pix_r / 2.0, 2 + 0.5, f'{bar_pc_r} pc',
             color='black', fontsize=STD_FONTSIZE, ha='center', va='bottom', weight='bold')
    
    # Move compass to top-left (matching relative position of other panels: ~20% W, ~75% H)
    c_x_anchor = int(Wc * 0.09)
    c_y_anchor = int(Hc * 0.8)
    c_height = 8
    c_width = 10
    
    axR.plot([c_x_anchor, c_x_anchor], [c_y_anchor, c_y_anchor + c_height], color='black', lw=STD_LW)
    axR.text(c_x_anchor, c_y_anchor + c_height + 2, 'N', color='black', fontsize=STD_FONTSIZE, ha='center', va='bottom', weight='bold')
    axR.plot([c_x_anchor, c_x_anchor - c_width], [c_y_anchor, c_y_anchor], color='black', lw=STD_LW)
    axR.text(c_x_anchor - c_width - 2, c_y_anchor, 'E', color='black', fontsize=STD_FONTSIZE, ha='center', va='center', weight='bold')

    # Legend for Source Colors
    # from matplotlib.lines import Line2D
    # legend_elements = [
    #     Line2D([0], [0], marker='o', color='w', label='Bright (Mag < S1)',
    #            markerfacecolor='none', markeredgecolor='lime', markersize=10, markeredgewidth=1.5),
    #     Line2D([0], [0], marker='o', color='w', label='S1',
    #            markerfacecolor='none', markeredgecolor=PLOT_COLORS['s1'], markersize=10, markeredgewidth=STD_LW),
    #     Line2D([0], [0], marker='o', color='w', label='Faint (S1 < Mag < 26.9)',
    #            markerfacecolor='none', markeredgecolor='orange', markersize=10, markeredgewidth=1.5),
    #     Line2D([0], [0], marker='o', color='w', label='Very Faint (Mag > 26.9)',
    #            markerfacecolor='none', markeredgecolor='magenta', markersize=10, markeredgewidth=1.5)
    # ]
    # axR.legend(handles=legend_elements, loc='upper right', fontsize=12, framealpha=0.9)

    # [rev3] in-figure key for the panel-c markers (editor: define symbols in the image)
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    key = [
        Line2D([], [], ls='none', marker='o', ms=5, mfc='none', mec=PLOT_COLORS['sn'], mew=STD_LW),
        Line2D([], [], ls='none', marker='o', ms=5, mfc='none', mec=PLOT_COLORS['catalog'], mew=0.5),
        Patch(facecolor='none', edgecolor=PLOT_COLORS['s1'], lw=STD_LW),
    ]
    key_labels = [
        'SN 2024vjm',
        r'$Euclid$ sources',
        'S1',
    ]
    axR.legend(key, key_labels, loc='upper right', fontsize=ns.FS_SMALL, frameon=True,
               framealpha=0.85, facecolor='white', edgecolor='none', borderpad=0.4,
               handletextpad=0.4, handlelength=1.2)

    # Dashed connectors from the red box (big Euclid) down to the wide zoom panel
    dash = (0, (4, 3))
    for (cx0, cy0), (zx, zy) in [
        ((x_min_e, y_min_e), (0, Hc)),
        ((x_max_e, y_min_e), (Wc, Hc)),
    ]:
        fig.add_artist(ConnectionPatch(
            xyA=(cx0, cy0), coordsA=axL.transData,
            xyB=(zx, zy), coordsB=axR.transData,
            color=PLOT_COLORS['connector'], lw=STD_LW, linestyle=dash, alpha=0.95
        ))

    if savepath:
        # [rev3] 600 dpi keeps the Euclid detail in the embedded images; overlays stay vector
        ns.save(fig, savepath, dpi=600, pad_inches=0)

if __name__ == "__main__":
    CREATE_CATALOG = False
    RUN_BG_LEFT_EUCLID = True
    BG_EUCLID_LAYOUT = 'top2_zoom_bottom'  # 'horizontal' or 'vertical'
    catalog_file = paths.EUCLID_CAT

    if CREATE_CATALOG or not paths.EUCLID_CSV.exists():
        filtered_catalog = extract_flux_magnitude_catalog(catalog_file)
        paths.EUCLID_CSV.parent.mkdir(parents=True, exist_ok=True)
        filtered_catalog.to_csv(paths.EUCLID_CSV, index=False)
    else:
        filtered_catalog = pd.read_csv(paths.EUCLID_CSV)

    if RUN_BG_LEFT_EUCLID:
        common_kwargs = dict(
            bg_fits_file=paths.BLACKGEM_IMAGE,
            euclid_fits_file=paths.EUCLID_IMAGE,
            euclid_catalog=filtered_catalog,
            ra_center=287.357448360426,
            dec_center=-63.8338255006658,
            bg_fov_arcmin=10.0,
            euclid_cutout_size_arcsec=ZOOM_CUTOUT_HEIGHT,
            euclid_circle_radius_arcsec=0.1,
            euclid_large_cutout_deg=10/60,
        )

        layout = BG_EUCLID_LAYOUT.lower().strip()

        if layout in ('vertical', 'v'):
            bg_above_euclid_two_images(
                **common_kwargs,
                figsize=(9, 27),
                savepath='Fig1_alt_layout',
            )
        elif layout in ('top2', 'hybrid', 'two_top_one_bottom', 'top2_zoom_bottom'):
            bg_and_big_euclid_on_top_zoom_bottom(
                **common_kwargs,
                figsize=(ns.DOUBLE_COL, ns.DOUBLE_COL),   # [rev3] 180 mm wide
                savepath='Fig1_BG_plus_Euclid',            # -> outputs/figures/
            )
        else:
            bg_left_of_euclid_two_images(
                **common_kwargs,
                figsize=(27, 9),
                savepath='Fig1_alt_layout',
            )
    else:
        plot_blackgem_only(
            fits_file=paths.BLACKGEM_IMAGE,
            ra_center=287.357447,
            dec_center=-63.833826,
            fov_arcmin=10.0,
            savepath=None
        )