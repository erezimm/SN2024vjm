"""[rev3] Shared data preparation for Supplementary Figs 3 and 9.

Reproduces the __main__ block of lc_clean.py (worktree version incl. Mookodi)
up to the GP interpolation. Reads data only; writes nothing.
"""
import numpy as np
import supp03_09_lc_clean_lib as L


def build_lc():
    lc = L.build_master_lc()
    lc = L.add_ghost_points(lc)
    ft = L.load_filter_transmissions(L.path_filters)
    lc, A_map = L.apply_extinction_corrections(
        lc, ft, EBV_MW=0.043, Rv_MW=3.1, EBV_host=0.205, Rv_host=3.1)
    return lc, A_map


def extra_tables():
    bg_dets, bg_nondets = L.load_bg_raw(L.PATH_BG)
    ml_nondets = L.load_ml_nondet(L.PATH_ML)
    rem_limits = L.load_REM_epoch_limits(epoch=60568.007)
    dlt40 = L.load_dlt40(L.PATH_DLT40)
    return bg_dets, bg_nondets, ml_nondets, rem_limits, dlt40


def gp(lc):
    dates = np.linspace(L.EXPLODE_T, L.EXPLODE_T + 56, 10000)
    return L.interpolate_flux_gp(lc, dates, kernel_length=2.0, min_alpha=1e-4,
                                 scale_factor=1e16, late_phase_start=10)
