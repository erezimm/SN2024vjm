"""Input/output locations for all scripts and notebooks in this repository.

Two input roots can be set, in order of precedence, on the command line, through
environment variables, or left at their defaults:

    --data DIR       / SN2024VJM_DATA      (default: <repo>/data)
        small inputs shipped with the repository (spectra, photometry,
        filter curves, comparison samples, astrometry tables)

    --external DIR   / SN2024VJM_EXTERNAL  (default: <data>/external)
        large public data sets that are NOT shipped (Euclid ERO image and
        catalogue, MUSE cube, BPASS models, BlackGEM image); see README.md
        for where to download them and how to lay them out

Outputs are always written below <repo>/outputs.

Example:
    python scripts/fig2_plot_composite_tracks.py --external /path/to/external
    SN2024VJM_EXTERNAL=/path/to/external jupyter lab
"""
import argparse
import os
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

_p = argparse.ArgumentParser(add_help=False)
_p.add_argument('--data')
_p.add_argument('--external')
_args, _ = _p.parse_known_args()

DATA = Path(_args.data or os.environ.get('SN2024VJM_DATA') or REPO / 'data').expanduser()
EXTERNAL = Path(_args.external or os.environ.get('SN2024VJM_EXTERNAL') or DATA / 'external').expanduser()

# ---- shipped inputs (data/)
SPECTRA = DATA / 'spectra'              # SN 2024vjm spectra + meta.csv (IR/, epessto/ subfolders)
PHOTOMETRY = DATA / 'photometry'        # SN 2024vjm photometry (telescope_data/ subfolder)
FILTERS = DATA / 'filters'              # filter transmission curves
COMPARISON = DATA / 'comparison'        # literature comparison SNe (SN2008ha/, SN2010ae/, Iax light-curve tables)
ASTROMETRY = DATA / 'astrometry'        # i-band astrometric scatter tables (Bin1/4/16.csv)
PROGENITORS = DATA / 'progenitor_models'  # WR population, hot-subdwarf+WD binaries
HOST_PROPS = DATA / 'host_props.json'

# ---- large external inputs (not shipped)
EUCLID = EXTERNAL / 'euclid'            # ERO NGC 6744 VIS image, weight map, PSF, source catalogue
EUCLID_IMAGE = EUCLID / 'Euclid-VIS-ERO-NGC6744-Flattened.DR3.fits'
EUCLID_WEIGHT = EUCLID / 'Euclid-VIS-ERO-NGC6744-Flattened.DR3.weight.fits'
EUCLID_PSF = EUCLID / 'prototypes_Euclid-VIS-ERO-NGC6744-Flattened.DR3.psfex.fits'
EUCLID_CAT = EUCLID / 'Euclid-VIS-ERO-NGC6744-Flattened.DR3.cat'
BLACKGEM_IMAGE = EXTERNAL / 'blackgem' / 'BG4_20240923_020840_red.fits'
MUSE = EXTERNAL / 'muse'                # MUSE datacube (ESO programme 114.28GZ) and products
BPASS = EXTERNAL / 'bpass'              # BPASS v2.2.1 detailed models (bpass-v2.2-newmodels)

# ---- outputs
OUTPUTS = REPO / 'outputs'
EUCLID_CSV = OUTPUTS / 'euclid_vis.csv'   # column cut of EUCLID_CAT, written by fig1_euclid_image.py
OUTPUTS.mkdir(parents=True, exist_ok=True)
