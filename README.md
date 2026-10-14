# SN 2024vjm — analysis code

Code accompanying **Zimmerman et al. 2026, "A Faint Progenitor System for the Faint Iax Supernova 2024vjm"** (*Nature Astronomy*).

## Layout
```
scripts/     Python scripts (run from this folder; they import each other)
notebooks/   Jupyter notebooks (outputs stripped)
data/        input data shipped with the code (spectra, photometry, filters, comparison samples)
data/external/  large public data sets, NOT included (see below)
outputs/     created on first run: figures, bolometric light curve, velocity tables
```

## What produces what

| Result in the paper | Code |
|---|---|
| Explosion epoch (MJD 60565.58 ± 0.31, power-law index 0.91) | `notebooks/explosion_time.ipynb` |
| Host extinction from Na I D, E(B−V)_host | `notebooks/extinction_NaID.ipynb` |
| GP light-curve interpolation, peak times and magnitudes, ΔM₁₅ (Table 1), bolometric light curve | `scripts/bolometric_lightcurve.py` (uses `supp03_09_lc_clean_lib.py`) |
| ⁵⁶Ni mass and γ-ray escape time from the radioactive tail | `scripts/ni56_tail_fit.py` |
| γ-ray escape time from the Katz integral (t₀ = 83.15 +14.83/−15.47 d, 3σ lower limit) | `scripts/katz_integral_t0.py` |
| Forced PSF photometry at the SN position in the *Euclid* VIS image; injection tests | `scripts/inject_and_measure.py` |
| Check of the PSF photometry against the *Euclid* catalogue (point sources within 20″, mean offset 0.06 mag) | `scripts/verify_catalog_sources.py` |
| Precision astrometry of the BlackGEM images: SN position and astrometric scatter (original analysis code, MATLAB) | `scripts/astrometry_analyze.m` |
| MUSE Hα slice, SN spectrum extraction | `notebooks/MUSE_cube.ipynb` |
| MUSE emission-line fluxes and O3N2 metallicity | `notebooks/MUSE_emission_lines_metallicity.ipynb` |
| NIR Co II P Cygni joint fits (velocities in Fig. 3b) | `notebooks/NIR_spectra_CoII_fits.ipynb` |
| Absorption-line velocities | `scripts/supp07_08_velocity_pipeline.py` |
| Fig. 1 (BlackGEM and *Euclid* images) | `scripts/fig1_euclid_image.py` |
| Fig. 2 (H–R diagram with BPASS tracks, progenitor limits) | `scripts/fig2_plot_composite_tracks.py` |
| Fig. 3 (bolometric light curve, Co II fits, Katz integral) | `scripts/fig3_Thermo_figure.py` |
| Fig. 4 (peak magnitude vs ΔM₁₅; single vs split linear fits, model comparison) | `scripts/fig4_Width_single_vs_split.py` |
| Supplementary Figs 1–9 | `scripts/supp01_…` to `scripts/supp09_…` |

`scripts/nature_style.py` holds the shared figure style. All figures are written to `outputs/figures/`.

## Order of running
1. `notebooks/explosion_time.ipynb` gives the explosion epoch, which is used as the default (`TEXP=60565.58`).
2. `python scripts/bolometric_lightcurve.py` writes `outputs/bolometric/`.
3. `python scripts/ni56_tail_fit.py` and `python scripts/katz_integral_t0.py`. The published errors use `SEED=20240913` and `NMC=30000`, which are the defaults.
4. The figure scripts.

## Requirements
Python 3 with `numpy`, `scipy`, `astropy`, `pandas`, `matplotlib`, `scikit-learn`, `photutils`, `extinction` and `mpdaf` (MUSE only).
`scripts/astrometry_analyze.m` is the original astrometry code. It runs in MATLAB with [AstroPack](https://github.com/EranOfek/AstroPack) (MAATv2), from the folder containing the BlackGEM images (`BG*.fits`).
The light-curve library also imports `SNeSCOPE.PhotoUtils` for filter zero points; install it from GitHub with `pip install git+https://github.com/idoirani/shock_cooling.git`. NumPy ≥ 2.0 is needed (`numpy.trapezoid`). Everything can be installed with `pip install -r requirements.txt`.

## Data
All input locations are defined in `scripts/paths.py`. Two roots can be set on the command line, through environment variables, or left at their defaults:

| Root | Option | Environment variable | Default |
|---|---|---|---|
| Shipped inputs | `--data DIR` | `SN2024VJM_DATA` | `data/` |
| Large external data | `--external DIR` | `SN2024VJM_EXTERNAL` | `data/external/` |

For example, `python scripts/fig1_euclid_image.py --external /path/to/external`. For the notebooks, set the environment variables before starting Jupyter.

### Included in `data/`
```
spectra/            SN 2024vjm optical and NIR spectra, meta.csv (observation log),
                    IR/ (TripleSpec), epessto/ (EFOSC2), SN2024vjm_SALT/ (RSS),
                    velocity_selection.csv (vetted line selection for the velocity fits),
                    IR/coii_pcygni_jointfit/coii_jointfit_results.csv (output of the NIR notebook)
photometry/         SN 2024vjm photometry: MeerLICHT (ML.csv), BlackGEM (BG_lc_sep2025.csv,
                    new_phot_bg.csv), LCO, DLT40, GOTO, Mookodi, telescope_data/ (REM, PROMPT6,
                    EFOSC2, IMACS); lightcurve.csv is the merged light curve
filters/            filter transmission curves
comparison/         literature comparison data: SN 2008ha and SN 2010ae spectra (from WISeREP),
                    SN Iax peak magnitudes and decline rates
astrometry/         BlackGEM i-band astrometric scatter (Supplementary Fig. 2)
progenitor_models/  Galactic WR population and hot-subdwarf+WD binaries (Fig. 2)
host_props.json     host position used by the Euclid injection tests
```
The SN 2024vjm photometry and spectra are also available on WISeREP (https://www.wiserep.org/object/26253).

### Not included (download into `data/external/`)
| Data | Source | Place as |
|---|---|---|
| *Euclid* ERO NGC 6744 VIS image, weight map, PSF and source catalogue | https://doi.org/10.57780/esa-qmocze3 | `euclid/Euclid-VIS-ERO-NGC6744-Flattened.DR3.fits`, `…DR3.weight.fits`, `prototypes_…DR3.psfex.fits`, `…DR3.cat` |
| BlackGEM image (Fig. 1) | available from the authors on request | `blackgem/BG4_20240923_020840_red.fits` |
| MUSE NFM datacube (ESO programme 114.28GZ) | ESO Science Archive, https://archive.eso.org | `muse/Object.fits`; `notebooks/MUSE_cube.ipynb` writes the corrected cubes and `spectrum.txt` there |
| BPASS v2.2.1 detailed stellar models (z = 0.014) | https://bpass.auckland.ac.nz | extract so that `bpass/bpass-v2.2-newmodels/NEWBINMODS/…` exists |
| *JWST* NIRSpec spectrum (reduced version already in `data/spectra/`) | https://doi.org/10.17909/m3aj-1c81 | — |

`fig1_euclid_image.py` writes the column-selected *Euclid* catalogue to `outputs/euclid_vis.csv` on its first run; `verify_catalog_sources.py` reads it from there.
`notebooks/MUSE_emission_lines_metallicity.ipynb` also needs the `eml_tools` module (Balmer decrement and extinction helpers) on the `PYTHONPATH`.
