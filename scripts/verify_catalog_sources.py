"""Check of the forced-PSF photometry against the Euclid VIS catalogue.

Fits every point source (CLASS_STAR >= 0.8) within 20 arcsec of SN 2024vjm with the
same PSF model used for the progenitor limit and compares the result with the
catalogue MAG_PSF (Methods: mean offset 0.06 mag). Needs outputs/euclid_vis.csv,
which fig1_euclid_image.py writes from the Euclid catalogue.
Writes outputs/verification_pointsources.csv and outputs/verification_pointsources_plot.pdf.
"""
import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
import astropy.units as u
from astropy.nddata import Cutout2D
from photutils.psf import PSFPhotometry, FittableImageModel
from photutils.background import LocalBackground, MeanBackground, MMMBackground
from scipy.ndimage import zoom
from astropy.table import Table
import paths

def load_psf(psf_path, zoom_factor=None, oversampling=1):
    """Load PSF with corrected normalization"""
    print(f"Loading PSF: {psf_path}")
    with fits.open(psf_path) as psf_hdul:
        psf_raw = psf_hdul[0].data
        if len(psf_raw.shape) == 3 and psf_raw.shape[0] == 6:
             psfs = [psf_raw[i] for i in range(6)]
             mean_psf = np.mean(psfs, axis=0)
        else:
             psf_size = 31
             n_prototypes = 6
             psfs = []
             if psf_raw.shape == (186, 31):
                 for i in range(n_prototypes):
                    psf_cut = psf_raw[i*psf_size:(i+1)*psf_size, :]
                    psfs.append(psf_cut)
                 mean_psf = np.mean(psfs, axis=0)
             elif psf_raw.shape == (31, 186):
                 # Horizontal concatenation
                 for i in range(n_prototypes):
                    psf_cut = psf_raw[:, i*psf_size:(i+1)*psf_size]
                    psfs.append(psf_cut)
                 mean_psf = np.mean(psfs, axis=0)
             else:
                 mean_psf = psf_raw 

        if zoom_factor is not None:
            mean_psf = zoom(mean_psf, zoom_factor, order=3)
            
        mean_psf /= np.sum(mean_psf)
        mean_psf *= (oversampling**2) # FIX APPLIED
        return FittableImageModel(mean_psf, oversampling=oversampling)

def verify_catalog():
    # Paths
    image_path = paths.EUCLID_IMAGE
    weight_path = paths.EUCLID_WEIGHT
    psf_path = paths.EUCLID_PSF
    csv_path = paths.EUCLID_CSV   # written by fig1_euclid_image.py
    
    # Standard ZP
    zp = 30.1320
    
    # Load Image Data
    with fits.open(image_path) as hdul:
        full_data = hdul[0].data
        header = hdul[0].header
        wcs = WCS(header)
    
    with fits.open(weight_path) as hdul:
        full_weight = hdul[0].data
        
    # Load PSF (Corrected Oversampling)
    psf_samp = 0.45
    target_oversampling = 2
    current_oversampling = 1.0 / psf_samp
    zoom_factor = target_oversampling / current_oversampling
    psf_model = load_psf(psf_path, zoom_factor=zoom_factor, oversampling=target_oversampling)
    
    # DEBUG MODEL
    print(f"PSF Model Sum: {np.sum(psf_model.data)}")
    print(f"PSF Model Max: {np.max(psf_model.data)}")
    print(f"PSF Model Shape: {psf_model.data.shape}")
    
    # Load Catalog
    print(f"Loading Catalog: {csv_path}")
    df = pd.read_csv(csv_path)
    
    # SN Location
    sn_ra = 287.357447567
    sn_dec = -63.83382612
    sn_coord = SkyCoord(sn_ra, sn_dec, unit='deg')
    
    # Calculate Separation
    cat_coords = SkyCoord(df['RA'].values, df['Dec'].values, unit='deg')
    seps = sn_coord.separation(cat_coords)
    df['sep_arcsec'] = seps.arcsec
    
    # Filter
    # 1. Within 20 arcsec
    # 2. Point Sources (CLASS_STAR > 0.8)
    # 3. Valid Magnitude
    subset = df[(df['sep_arcsec'] <= 20.0) & (df['MAG_PSF'] < 90.0) & (df['CLASS_STAR'] >= 0.8)].copy()
    
    print(f"Found {len(subset)} point sources (CLASS_STAR >= 0.8) within 20''.")
    
    if len(subset) == 0:
        print("No sources found matching criteria.")
        return

    # Use ALL filtered sources
    sample = subset
    
    print(f"\nVerifying {len(sample)} sources:")
    print(f"{'ID':<6} {'Sep':<6} {'Cat Mag':<8} {'Meas Mag':<8} {'Diff':<6} {'MagErr':<6} {'Cat Flux':<10} {'Meas Flux':<10}")
    
    cutout_size = 100
    
    results_list = []
    
    # Filename for results
    csv_file = paths.OUTPUTS / 'verification_pointsources.csv'
    
    for idx, row in sample.iterrows():
        ra, dec = row['RA'], row['Dec']
        cat_mag = row['MAG_PSF']
        cat_flux = row['FLUX_PSF'] 
        
        # Pixel Coords
        target_x, target_y = wcs.world_to_pixel_values(ra, dec)
        
        # Cutout
        try:
            cutout_img = Cutout2D(full_data, (target_x, target_y), (cutout_size, cutout_size), wcs=wcs)
            cutout_wht = Cutout2D(full_weight, (target_x, target_y), (cutout_size, cutout_size), wcs=wcs)
        except Exception as e:
            continue
            
        curr_data = cutout_img.data
        curr_weight = cutout_wht.data
        
        # Error Map
        with np.errstate(divide='ignore', invalid='ignore'):
            error_arr = 1.0 / np.sqrt(curr_weight)
        error_arr[curr_weight <= 0] = 1e10
        
        # Center in cutout
        cx, cy = cutout_size/2, cutout_size/2
        
        # Refine Position: Find peak within 5 pixels radius
        # This handles small astrometric offsets/PM to avoid fitting local noise
        search_rad = 5
        y_slice, x_slice = np.indices(curr_data.shape)
        mask = (x_slice - cx)**2 + (y_slice - cy)**2 <= search_rad**2
        
        # Mask out data outside search radius to find peak ONLY near target
        # (Avoid jumping to a brighter neighbor far away)
        masked_data = curr_data.copy()
        masked_data[~mask] = -np.inf
        
        py, px = np.unravel_index(np.argmax(masked_data), masked_data.shape)
        
        # Use Peak as Initial Guess
        fit_model = psf_model.copy()
        fit_model.x_0.fixed = False 
        fit_model.y_0.fixed = False
        
        init_params = Table()
        init_params['x_0'] = [px]
        init_params['y_0'] = [py]
        init_params['flux_0'] = [cat_flux]
        
        # Background: Flattened image logic
        bkg_estimator = None 
        psfphot = PSFPhotometry(fit_model, fit_shape=(21, 21), localbkg_estimator=None)
        
        try:
            results = psfphot(curr_data, error=error_arr, init_params=init_params)
            meas_flux = results['flux_fit'][0]
            meas_flux_err = results['flux_err'][0] # Get Flux Error
            
            if meas_flux > 0:
                meas_mag = zp - 2.5 * np.log10(meas_flux)
                # Propagate error: dM = 1.0857 * (dF / F)
                meas_mag_err = 1.0857 * (meas_flux_err / meas_flux)
            else:
                meas_mag = 99.9
                meas_mag_err = 0.0
            
            if meas_mag < 90:
                diff_mag = meas_mag - cat_mag
                print(f"{idx:<6} {row['sep_arcsec']:<6.2f} {cat_mag:<8.3f} {meas_mag:<8.3f} {diff_mag:<6.3f} {meas_mag_err:<6.3f} {cat_flux:<10.1f} {meas_flux:<10.1f}")
                results_list.append({
                    'id': idx,
                    'sep': row['sep_arcsec'],
                    'cat': cat_mag, 
                    'meas': meas_mag, 
                    'diff': diff_mag, 
                    'err': meas_mag_err
                })
            else:
                print(f"{idx:<6} {row['sep_arcsec']:<6.2f} {cat_mag:<8.3f} {'Failed':<8} {'NaN':<6} {'0.0':<6} {cat_flux:<10.1f} {meas_flux:<10.1f}")
            
        except Exception as e:
            pass

    # Save Results
    if len(results_list) > 0:
        res_df = pd.DataFrame(results_list)
        res_df.to_csv(csv_file, index=False)
        print(f"\nSaved results to {csv_file}")

    # Load Results (if list is empty, try loading)
    if len(results_list) == 0:
        try:
             res_df = pd.read_csv(csv_file)
             results_list = res_df.to_dict('records')
             print(f"Loaded previous results from {csv_file}")
        except FileNotFoundError:
             print("No results to analyze.")
             return

    if len(results_list) > 0:
        diffs = [r['diff'] for r in results_list]
        cats = [r['cat'] for r in results_list]
        meass = [r['meas'] for r in results_list]
        errs = [r['err'] for r in results_list]
        
        mean_diff = np.mean(diffs)
        std_diff = np.std(diffs)
        print(f"\nStatistics ({len(results_list)} sources):")
        print(f"Mean Offset (Meas - Cat): {mean_diff:.4f} mag")
        print(f"Std Dev: {std_diff:.4f} mag")
        print(f"Median Offset: {np.median(diffs):.4f} mag")
        
        # Plotting
        try:
            import matplotlib.pyplot as plt
            from scipy.stats import linregress
            
            plt.figure(figsize=(8, 6))
            # Errorbar Plot
            plt.errorbar(cats, meass, yerr=errs, fmt='o', alpha=0.6, label='Data', capsize=2)
            
            # 1:1 Line
            min_mag = min(min(cats), min(meass)) - 0.5
            max_mag = max(max(cats), max(meass)) + 0.5
            plt.plot([min_mag, max_mag], [min_mag, max_mag], 'k--', label='1:1 Ideal')
            
            # Fit
            slope, intercept, r_value, p_value, std_err = linregress(cats, meass)
            x_vals = np.array([min_mag, max_mag])
            y_vals = slope * x_vals + intercept
            plt.plot(x_vals, y_vals, 'r-', label=f'Fit: y={slope:.3f}x + {intercept:.3f}\n$R^2$={r_value**2:.3f}')
            
            plt.xlabel('Catalog Magnitude (MAG_PSF)')
            plt.ylabel('Measured Magnitude (PSFPhotometry)')
            plt.title(f'Point Sources < 20\'\' (N={len(results_list)})\nMean Offset: {mean_diff:.3f} mag')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.gca().invert_yaxis()
            plt.gca().invert_xaxis()
            
            out_file = paths.OUTPUTS / "verification_pointsources_plot.pdf"
            plt.savefig(out_file)
            plt.show()
            print(f"Saved plot to {out_file}")
            
        except ImportError:
            print("Matplotlib not installed, skipping plot.")

if __name__ == "__main__":
    verify_catalog()
