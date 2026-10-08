
import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
from astropy.nddata import Cutout2D
from photutils.psf import PSFPhotometry, FittableImageModel
from photutils.background import LocalBackground, MMMBackground
from scipy.ndimage import zoom
from astropy.table import Table
import paths

def load_psf(psf_path, zoom_factor=None, oversampling=1):
    with fits.open(psf_path) as psf_hdul:
        psf_raw = psf_hdul[0].data
        if len(psf_raw.shape) == 3 and psf_raw.shape[0] == 6:
             psfs = [psf_raw[i] for i in range(6)]
             mean_psf = np.mean(psfs, axis=0)
        else:
             psfs = []
             if psf_raw.shape == (186, 31):
                 for i in range(6):
                    psf_cut = psf_raw[i*31:(i+1)*31, :]
                    psfs.append(psf_cut)
                 mean_psf = np.mean(psfs, axis=0)
             else:
                 mean_psf = psf_raw 

        if zoom_factor is not None:
            mean_psf = zoom(mean_psf, zoom_factor, order=3)
            
        mean_psf /= np.sum(mean_psf)
        mean_psf *= (oversampling**2)
        return FittableImageModel(mean_psf, oversampling=oversampling)

def calibrate_zp():
    image_path = paths.EUCLID_IMAGE
    weight_path = paths.EUCLID_WEIGHT
    psf_path = paths.EUCLID_PSF
    csv_path = paths.EUCLID_CSV   # written by fig1_euclid_image.py
    
    # Original ZP from header/script
    zp_header = 30.1320
    
    with fits.open(image_path) as hdul:
        full_data = hdul[0].data
        header = hdul[0].header
        wcs = WCS(header)
    with fits.open(weight_path) as hdul:
        full_weight = hdul[0].data
        
    psf_samp = 0.45
    target_oversample = 2
    zoom_f = target_oversample / (1.0/psf_samp)
    psf_model = load_psf(psf_path, zoom_factor=zoom_f, oversampling=target_oversample)
    
    df = pd.read_csv(csv_path)
    sn_ra, sn_dec = 287.357447567, -63.83382612
    sn_coord = SkyCoord(sn_ra, sn_dec, unit='deg')
    seps = sn_coord.separation(SkyCoord(df['RA'], df['Dec'], unit='deg')).arcsec
    df['sep'] = seps
    
    # Select stars
    # Within 30 arcsec (to get more stars), Mag < 25 (Reliable)
    subset = df[(df['sep'] <= 30.0) & (df['MAG_PSF'] < 25.0) & (df['MAG_PSF'] > 15.0)].copy()
    
    print(f"Calibrating with {len(subset)} stars (Mag 15-25, <30'').")
    
    offsets = []
    
    cutout_size = 100
    bkg_est = LocalBackground(inner_radius=6, outer_radius=10, bkg_estimator=MMMBackground())
    psfphot = PSFPhotometry(psf_model, fit_shape=(21, 21), localbkg_estimator=bkg_est)
    
    print(f"{'Mag_Cat':<8} {'Mag_Meas':<8} {'Offset':<8}")
    
    for idx, row in subset.iterrows():
        tx, ty = wcs.world_to_pixel_values(row['RA'], row['Dec'])
        if not (50 < tx < 36000-50 and 50 < ty < 36000-50): continue
        
        try:
            cut = Cutout2D(full_data, (tx, ty), (cutout_size, cutout_size), wcs=wcs)
            wht = Cutout2D(full_weight, (tx, ty), (cutout_size, cutout_size), wcs=wcs)
            
            with np.errstate(divide='ignore'): err = 1.0/np.sqrt(wht.data)
            err[wht.data<=0] = 1e10
            
            init = Table()
            init['x_0'] = [cutout_size/2]
            init['y_0'] = [cutout_size/2]
            init['flux_0'] = [row['FLUX_PSF']]
            
            # Allow position float
            model = psf_model.copy()
            model.x_0.fixed = False
            model.y_0.fixed = False
            
            res = psfphot(cut.data, error=err, init_params=init)
            flux = res['flux_fit'][0]
            
            if flux > 0:
                mag_meas = zp_header - 2.5*np.log10(flux)
                offset = row['MAG_PSF'] - mag_meas # Positive if Cat is fainter (implying ZP needs increase?)
                # Offset = Cat - (ZP - 2.5logF)
                # Cat = (ZP + Delta) - 2.5logF
                # Delta = Cat - Meas
                
                print(f"{row['MAG_PSF']:<8.3f} {mag_meas:<8.3f} {offset:<8.3f}")
                offsets.append(offset)
        except:
            pass
            
    if offsets:
        med_offset = np.median(offsets)
        std_offset = np.std(offsets)
        print(f"\nMedian Offset: {med_offset:.3f} +/- {std_offset:.3f}")
        print(f"New ZP should be: {zp_header + med_offset:.3f}")
        
if __name__ == "__main__":
    calibrate_zp()
