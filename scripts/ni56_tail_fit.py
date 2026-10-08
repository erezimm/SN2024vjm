# Reads the bolometric light curve written by bolometric_lightcurve.py (<repo>/outputs/bolometric).
# TEXP env var overrides the explosion MJD (default 60565.58).
import os as _os
_TEXP=float(_os.environ.get('TEXP', 60565.58))
_BOL=_os.path.join(_os.path.dirname(_os.path.abspath(__file__)),'..','outputs','bolometric',f'bolometric_lightcurve_faraway_{_TEXP:.2f}.txt')

import numpy as np
from dataclasses import dataclass
from astropy.io import ascii
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

def Ni56_depo(t, MN56):
    # t in days, MN56 in Msun, returns L/(1e43 erg/s) when you fit y = L/1e43
    return MN56 * (6.45*np.exp(-t/8.76) + 1.38*np.exp(-t/111.4))

def Ni56_depo_leaky(t, MN56, t0):
    # t in days, t0 is leakage timescale (days)
    trap = 1.0 - np.exp(-(t0/np.maximum(t,1e-6))**2)
    Q_gamma = MN56 * (6.45*np.exp(-t/8.76) + 1.38*np.exp(-t/111.4)) * trap
    Q_pos = MN56 * 4.64*(np.exp(-t/111.4)-np.exp(-t/8.76))*1e-2
    return Q_gamma + Q_pos

# Load bolometric LC
L_data = ascii.read(_BOL)
t_all = np.array(L_data['MJD'])
L_all = np.array(L_data['Lbol(erg/s)'])
Le_all = np.array(L_data['Lbol_err(erg/s)'])

t0 = _TEXP  # explosion MJD

# keep 20–57 d post-explosion
m = (t_all >= t0 + 45) & (t_all <= t0 + 57)
t = t_all[m]
L = L_all[m]
Le = Le_all[m]

# Fit y = L/1e43 with correctly scaled uncertainties
y  = L  / 1e43
ye = Le / 1e43

popt, pcov = curve_fit(Ni56_depo_leaky, t-_TEXP, y,
                       p0=[1e-4, 50.0], bounds=(0, np.inf),
                       sigma=ye, absolute_sigma=True, maxfev=50000)

MN, t0 = popt
MN_err, t0_err = np.sqrt(np.diag(pcov))
print(f"^56Ni = {MN:.3e} ± {MN_err:.3e} Msun,  t0 = {t0:.1f} ± {t0_err:.1f} d")

yfit = Ni56_depo_leaky(t-_TEXP, *popt)
chi2 = np.sum(((y - yfit)/ye)**2)
dof  = (len(y) - len(popt))
print(f"chi^2/dof = {chi2} = {chi2/dof}")

x   = t-_TEXP
plt.figure()
plt.loglog(x, L, 'o', label='Data')
plt.fill_between(x, y*1e43-ye*1e43, y*1e43+ye*1e43, alpha=0.2)
plt.loglog(x, yfit*1e43, '-', label='Leaky fit')
plt.xlabel('Time since explosion (days)')
plt.ylabel('L (erg/s)')
plt.legend()
plt.tight_layout()
plt.show()