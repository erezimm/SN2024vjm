import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
from astropy.nddata import Cutout2D
from photutils.psf import PSFPhotometry, FittableImageModel
from photutils.background import LocalBackground, MMMBackground, MeanBackground
from scipy.ndimage import zoom
from astropy.table import Table
import copy
import paths

def create_gaussian_psf(size=31, pixel_scale=0.1, fwhm_arcsec=0.16):
    """
    Create a Gaussian PSF with specified FWHM.
    
    Parameters:
    - size: PSF stamp size in pixels
    - pixel_scale: arcsec/pixel
    - fwhm_arcsec: desired FWHM in arcsec
    
    Returns:
    - FittableImageModel of normalized Gaussian PSF
    """
    # Convert FWHM to sigma: FWHM = 2.355 * sigma
    fwhm_pix = fwhm_arcsec / pixel_scale
    sigma_pix = fwhm_pix / 2.355
    
    # Create coordinate grid
    center = size // 2
    y, x = np.mgrid[0:size, 0:size]
    
    # Distance from center in pixels
    r_squared = (x - center)**2 + (y - center)**2
    
    # Gaussian: I(r) = exp(-r^2 / (2*sigma^2))
    psf = np.exp(-r_squared / (2 * sigma_pix**2))
    
    # Normalize to unit sum
    psf /= np.sum(psf)
    
def load_psf(psf_path, zoom_factor=None, oversampling=1):
    """Load PSF from file with optional resampling for integer oversampling"""
    print(f"Loading PSF: {psf_path}")
    with fits.open(psf_path) as psf_hdul:
        psf_raw = psf_hdul[0].data
        n_prototypes = 6
        psf_size = 31
        psfs = []
        if psf_raw.shape == (186, 31):
             for i in range(n_prototypes):
                y_start = i * psf_size
                y_end = (i + 1) * psf_size
                psf_cut = psf_raw[y_start:y_end, :]
                psfs.append(psf_cut)
        elif psf_raw.shape == (31, 186):
             for i in range(n_prototypes):
                x_start = i * psf_size
                x_end = (i + 1) * psf_size
                psf_cut = psf_raw[:, x_start:x_end]
                psfs.append(psf_cut)
        elif len(psf_raw.shape) == 3 and psf_raw.shape[0] == 6:
             psfs = [psf_raw[i] for i in range(6)]
        else:
             print(f"  Error: Unrecognized PSF shape {psf_raw.shape}. using raw as is if square")
             if psf_raw.shape[0] == psf_raw.shape[1]:
                 psf_2d = psf_raw
                 if zoom_factor is not None:
                     psf_2d = zoom(psf_2d, zoom_factor, order=3)
                 psf_2d /= np.sum(psf_2d)
                 psf_2d *= (oversampling**2)
                 return FittableImageModel(psf_2d, oversampling=oversampling)
             return None

        mean_psf = np.mean(psfs, axis=0)
        
        if zoom_factor is not None:
            mean_psf = zoom(mean_psf, zoom_factor, order=3)
            
        mean_psf /= np.sum(mean_psf)
        mean_psf *= (oversampling**2)
        return FittableImageModel(mean_psf, oversampling=oversampling)

def inject_and_measure():
    image_path = paths.EUCLID_IMAGE
    psf_path = paths.EUCLID_PSF
    
    # SN Coordinates
    sn_ra = 287.357447567
    sn_dec = -63.83382612
    sn_coord = SkyCoord(sn_ra, sn_dec, unit='deg')

    print(f"Loading image: {image_path}")
    with fits.open(image_path) as hdul:
        full_data = hdul[0].data
        # Use same limits as euclid_image.py for consistency
        vmin, vmax = -2.0, 8.0
        header = hdul[0].header
        wcs = WCS(header)

    # Load Weight Map
    weight_path = paths.EUCLID_WEIGHT
    print(f"Loading weight map: {weight_path}")
    with fits.open(weight_path) as w_hdul:
        full_weight = w_hdul[0].data
    
    psf_samp = 0.45
    # Target oversampling: 2 (Integer required by FittableImageModel)
    target_oversampling = 2
    # Current effective oversampling: 1 / 0.45 = 2.222...
    current_oversampling = 1.0 / psf_samp
    
    # Resample factor
    zoom_factor = target_oversampling / current_oversampling
    
    print(f"Loading PSF: {psf_path}")
    print(f"  Native PSF_SAMP = {psf_samp} (Oversampling ~{current_oversampling:.2f})")
    print(f"  Resampling to Integer Oversampling = {target_oversampling} (Zoom factor {zoom_factor:.4f})")
    
    psf_model = load_psf(psf_path, zoom_factor=zoom_factor, oversampling=target_oversampling)
    if psf_model is None:
        print("Failed to load PSF model.")
        return
    
    print(f"  PSF loaded and resampled: {psf_model._data.shape[0]}x{psf_model._data.shape[1]} pixels")
    
    # Zeropoint
    zp = 30.1320
    
    # Create Cutout for Analysis
    cutout_size_analysis = 100
    position = sn_coord
    position = sn_coord
    cutout_analysis = Cutout2D(full_data, position, (cutout_size_analysis, cutout_size_analysis), wcs=wcs)
    data = cutout_analysis.data
    
    # Weight Cutout
    cutout_weight = Cutout2D(full_weight, position, (cutout_size_analysis, cutout_size_analysis), wcs=wcs)
    weight_data = cutout_weight.data
    # wcs is now cutout wcs
    wcs = cutout_analysis.wcs
    
    # Update SN pixel coords to cutout coords
    sn_x, sn_y = wcs.world_to_pixel(sn_coord)
    
    # Load Host Properties from JSON (to avoid loading massive catalog)
    import json
    with open(paths.HOST_PROPS, 'r') as f:
        host_props = json.load(f)
        
    host_ra = host_props['ra']
    host_dec = host_props['dec']
    host_coord = SkyCoord(host_ra, host_dec, unit='deg')
    host_x, host_y = wcs.world_to_pixel(host_coord)
    
    # Symmetric Point: P_sym = Host + (Host - SN) = 2*Host - SN
    sym_x = 2 * host_x - sn_x+1
    sym_y = 2 * host_y - sn_y+1
    
    print(f"SN Location:   ({sn_x:.2f}, {sn_y:.2f})")
    print(f"Host Location: ({host_x:.2f}, {host_y:.2f})")
    print(f"Symmetric Loc: ({sym_x:.2f}, {sym_y:.2f})")
    
    # Nearest source parameters (for ellipse)
    nearest_a = host_props['a_image']
    nearest_b = host_props['b_image']
    nearest_theta = host_props['theta_image']
    nearest_x, nearest_y = host_x, host_y
    
    print(f"Nearest source: A={nearest_a:.2f}, B={nearest_b:.2f}, θ={nearest_theta:.1f}°")
    
    # MAG_AUTO Flux
    mag_auto = 25.8342 # From catalog ID 524331
    mag_auto_flux = 10**((zp - mag_auto) / 2.5)
    print(f"Catalog MAG_AUTO: {mag_auto} -> Flux: {mag_auto_flux:.2f}")
    
    # Mags to inject
    # Mags to inject
    # Mags to inject
    mags = [None, 25.0, 26.0, 26.5, 27.0]
    titles_template = ["Original", "Inj Mag 25.0", "Inj Mag 26.0", "Inj Mag 26.5", "Inj Mag 27.0"]
    
    # 2 Rows: Top = SN, Bottom = Sym
    fig, axes = plt.subplots(2, 5, figsize=(25, 10))
    
    scenarios = [
        {"name": "SN Loc", "x": sn_x, "y": sn_y, "row": 0},
        {"name": "Sym Loc", "x": sym_x, "y": sym_y, "row": 1}
    ]

    for scen in scenarios:
        target_x = scen['x']
        target_y = scen['y']
        row_idx = scen['row']
        scen_name = scen['name']
        
        # Single pixel comparison (Target vs SN) - Raw Data
        ty_i, tx_i = int(np.round(target_y)), int(np.round(target_x))
        sy_i, sx_i = int(np.round(sn_y)), int(np.round(sn_x))
        t_pix_val = data[ty_i, tx_i]
        s_pix_val = data[sy_i, sx_i]
        pix_diff = t_pix_val - s_pix_val
        print(f"[{scen_name}] Raw Pixel: {t_pix_val:.2f}, SN Pixel: {s_pix_val:.2f}, Diff: {pix_diff:.2f}")

        # Pre-calculate Background-Only 3-sigma Limit for this location
        # This represents the image depth, independent of the source brightness
        with np.errstate(divide='ignore', invalid='ignore'):
            base_bkg_error = 1.0 / np.sqrt(weight_data)
        base_bkg_error[weight_data <= 0] = 1e10
        
        # Setup dummy fit for error calculation
        limit_fit_model = psf_model.copy()
        limit_fit_model.x_0.fixed = True
        limit_fit_model.y_0.fixed = True
        limit_psfphot = PSFPhotometry(limit_fit_model, fit_shape=(21, 21), localbkg_estimator=None)
        
        init_limit = Table()
        init_limit['x_0'] = [target_x]
        init_limit['y_0'] = [target_y]
        init_limit['flux_0'] = [1.0] # Dummy flux
        
        try:
            # We use the raw data (or zeros) - it doesn't matter for the error calc, 
            # only the error_arr matters for flux_err
            res_limit = limit_psfphot(data, error=base_bkg_error, init_params=init_limit)
            bkg_flux_err = res_limit['flux_err'][0]
            limit_3sigma_mag = zp - 2.5 * np.log10(3 * bkg_flux_err)
        except:
             limit_3sigma_mag = 99.9

        for i, mag in enumerate(mags):
            ax = axes[row_idx, i]
            current_data = data.copy()
            
            res_str = ""
            if mag is not None:
                flux = 10**((zp - mag) / 2.5)
                # Inject a box around target
                box = 20
                y_i, x_i = int(np.round(target_y)), int(np.round(target_x))
                yy, xx = np.mgrid[y_i-box:y_i+box, x_i-box:x_i+box]
                
                if (np.min(yy) >= 0 and np.max(yy) < data.shape[0] and 
                    np.min(xx) >= 0 and np.max(xx) < data.shape[1]):
                    # Evaluate PSF at target location
                    psf_stamp = psf_model.evaluate(xx, yy, flux, target_x, target_y)
                    current_data[yy, xx] += psf_stamp
            
            # Measure
            # Calculate Error Map
            # Total Error = sqrt( Background_Variance + Source_Poisson_Variance )
            # Background_Var = 1 / Weight (assuming weight is inverse variance)
            # Source_Poisson_Var = |Data| / Gain
            gain = 76.137 # From user info
            
            with np.errstate(divide='ignore', invalid='ignore'):
                bkg_var = 1.0 / weight_data
                # Handle pixels with 0 or negative weight
                bkg_var[weight_data <= 0] = 1e10
                
                # Add Source Poisson Noise
                # Use abs(current_data) to handle negative noise fluctuations safely
                source_var = current_data / gain
                
                error_arr = np.sqrt(bkg_var + source_var)
            
            # Mask bad pixels in error map
            error_arr[weight_data <= 0] = 1e10 
            
            fit_model = psf_model.copy()
            fit_model.x_0.fixed = True
            fit_model.y_0.fixed = True
            
            # Background: Image is flattened (Mean=0), so we disable local subtraction
            bkg_estimator = None 
            psfphot = PSFPhotometry(fit_model, fit_shape=(21, 21), localbkg_estimator=None)
            
            # Measure at Target
            init_target = Table()
            init_target['x_0'] = [target_x]
            init_target['y_0'] = [target_y]
            init_target['flux_0'] = [100.0]
            
            # Measure at SN (for difference)
            init_sn = Table()
            init_sn['x_0'] = [sn_x]
            init_sn['y_0'] = [sn_y]
            init_sn['flux_0'] = [100.0]
            
            try:
                # 1. Measurement at target
                results = psfphot(current_data, error=error_arr, init_params=init_target)
                m_flux = results['flux_fit'][0]
                m_err = results['flux_err'][0]
                m_snr = m_flux / m_err
                m_mag = zp - 2.5 * np.log10(m_flux) if m_flux > 0 else 99.9
                m_mag_err = 1.0857 * (m_err / m_flux) if m_flux > 0 else 99.9
                 # Note: limit_3sigma_mag is now the Background-Only limit
                
                # 2. Measurement at SN (for delta) - Optional but requested
                results_sn = psfphot(current_data, error=error_arr, init_params=init_sn)
                sn_measured_flux = results_sn['flux_fit'][0]
                flux_diff = m_flux - sn_measured_flux
                
                res_str = f"Meas: {m_mag:.2f}±{m_mag_err:.2f} (SNR {m_snr:.1f})\nLim(3$\sigma$): {limit_3sigma_mag:.2f}\n$\Delta$Flux(Tgt-SN): {flux_diff:.2f}"
                print(f"  {scen_name} {titles_template[i]}: Recov Flux {m_flux:.2f}, Diff(Tgt-SN) {flux_diff:.2f}, Excess vs Auto: {m_flux - mag_auto_flux:.2f}")
            except:
                res_str = "Fit Failed"

            # Plot - Zoom on Target
            y_int, x_int = int(np.round(target_y)), int(np.round(target_x))
            box = 10
            y_min = max(0, y_int-box)
            y_max = min(current_data.shape[0], y_int+box)
            x_min = max(0, x_int-box)
            x_max = min(current_data.shape[1], x_int+box)

            zoom_data = current_data[y_min:y_max, x_min:x_max]
            ax.imshow(zoom_data, origin='lower', cmap='gray_r', vmin=vmin, vmax=vmax, 
                      extent=[x_min, x_max, y_min, y_max])
            
            # Markers
            ax.plot(sn_x, sn_y, 'rx', ms=8, mew=2, label='SN Loc',alpha=0.25)
            ax.plot(sym_x, sym_y, 'cx', ms=8, mew=2, label='Sym Loc',alpha=0.25)
            ax.plot(host_x, host_y, 'm.', ms=5, label='Host Center',alpha=0.25)
            
            # Ellipse
            from matplotlib.patches import Ellipse
            e = Ellipse((nearest_x, nearest_y), width=2*nearest_a*3, height=2*nearest_b*3, 
                        angle=nearest_theta, edgecolor='magenta', facecolor='none', lw=1.5, alpha=0.7)
            ax.add_patch(e)
                    
            if i == 0 and row_idx == 0:
                from matplotlib.lines import Line2D
                legend_elements = [
                    Line2D([0], [0], color='r', marker='x', lw=0, label='SN Loc',alpha=0.1),
                    Line2D([0], [0], color='c', marker='x', lw=0, label='Sym Loc',alpha=0.1),
                    Line2D([0], [0], color='magenta', marker='.', lw=0, label='Host Ctr',alpha=0.1),
                ]
                ax.legend(handles=legend_elements, loc='upper right', fontsize=6, framealpha=0.9)
                
            ax.set_title(f"{scen_name}: {titles_template[i]}\n{res_str}", fontsize=10)
            ax.axis('off')

    plt.tight_layout()

    plt.show()
    print("Saved plot to injection_test_plot.pdf")

if __name__ == "__main__":
    inject_and_measure()
