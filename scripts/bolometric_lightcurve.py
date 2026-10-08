"""[rev3] Explosion-time chain, step 1: GP light curve -> B peak, band stats, bolometric LC.
Same settings as lc_clean.py __main__ (Mookodi/worktree version); only EXPLODE_T varies.
Usage: python bolometric_lightcurve.py   (TEXP env var overrides the explosion MJD; default 60565.58)
Writes band statistics (Table 1) and the bolometric light curve to <repo>/outputs/bolometric.
"""
import os, numpy as np, pandas as pd
import supp03_09_lc_clean_lib as L
TEXP = float(os.environ.get('TEXP', 60565.58))   # explosion epoch from explosion_time.ipynb
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'outputs', 'bolometric')
os.makedirs(OUT, exist_ok=True)
L.EXPLODE_T = TEXP                                   # used at runtime inside bolometric_from_interp
lc = L.build_master_lc()
lc = L.add_ghost_points(lc, t_exp=TEXP)
ft = L.load_filter_transmissions(L.path_filters)
lc, A_map = L.apply_extinction_corrections(lc, ft, EBV_MW=0.043, Rv_MW=3.1, EBV_host=0.205, Rv_host=3.1)
dates = np.linspace(TEXP, TEXP + 56, 10000)
interp = L.interpolate_flux_gp(lc, dates, kernel_length=2.0, min_alpha=1e-4, scale_factor=1e16,
                               late_phase_start=10, explosion_time=TEXP)
b = L.bband_peak_and_dm15(interp)
print(f"RESULT TEXP={TEXP}  B_peak_MJD={b['peak_time']:.3f}  mB={b['peak_mag']:.3f}  dm15B={b['dm15']:.3f}")
res = L.band_stats_from_gp(interp, lc)
pd.DataFrame(res).to_csv(os.path.join(OUT, f'band_stats_{TEXP:.2f}.csv'), index=False)
Lf, Lfe = L.bolometric_from_interp(interp.copy(), dates, dist_pc=L.DIST_PC)
np.savetxt(os.path.join(OUT, f'bolometric_lightcurve_faraway_{TEXP:.2f}.txt'),
           np.column_stack([dates, Lf, Lfe]), header='MJD  Lbol(erg/s)  Lbol_err(erg/s)')
ipk = np.nanargmax(Lf)
print(f"RESULT bol_peak_MJD={dates[ipk]:.3f}  rise_from_explosion={dates[ipk]-TEXP:.2f} d  Lpeak={Lf[ipk]:.3e}")
