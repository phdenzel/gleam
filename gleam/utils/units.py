#!/usr/bin/env python
"""
@author: phdenzel

Framework to work with Planckian units
"""
import numpy as np

# Planckian units
marb = 2.177e-8   # kg
lap = 1.615e-35   # m
tick = 5.383e-44  # s
therm = 1.419e32  # K

# Quantum-mechanical fundamentals
m_e = 4.184e-23  # marb
m_b = 7.688e-20  # marb
alpha = 1./137.036
H_0 = 70 * 1000 / 3.08567758149137e6 * tick  # tick-1

# Derived quantities
sigma_thompson = 8*3.14159265359/3.*(alpha*alpha/m_e/m_e) * lap * lap
M_landau = 1./(m_b*m_b) * marb
r_bohr = 1./(alpha*m_e) * lap

# Non-planckian units
c = 2.99792458e8          # m s-1
G = 6.6743e-11            # m3 kg-1 s-2
hbar2pi = 6.62607004e-34  # J s
k_B = 1.38064852e-23      # J K-1

# Astronomical units
AU = 499.0                     # sec
parsec = 1.0292712503794876e8  # sec
M_sol = 0.54*M_landau          # kg
M_earth = 1.5e-6*M_landau      # kg

# other constants
pi = 3.1415926535897932384626433832795028841971693993751058209749445923078164062862089986280348253421170679821480865132823066470938446095505822317253594081284811174502841027019385211055596446229489549303819644288109756659334461284756482337867831652712019091456485669234603486104543266482133936072602491412737245870066063155881748815209209628292540917153643678925903600113305305488204665213841469519415116094330572703657595919530921861173819326117931051185480744623799627495673518857527248912279381830119491298336733624406566430860213949463952247371907021798609437027705392171762931767523846748184676694051320005681271452635608277857713427577896091736371787214684409012249534301465495853710507922796892589235420199561121290219608640344181598136297747713099605187072113499999983729780499510597317328160963185950244594553469083026425223082533446850352619311881710100031378387528865875332083814206171776691473035982534904287554687311595628638823537875937519577818577805321712268066130019278766111959092164201989


# conversion 
def H02aHz(H0arr):
    H0arr = np.asarray(H0arr)  # assume in units of km/s/Mpc
    H0 = H0arr.copy()
    H0 /= 3.08567758149137e1   # aHz=1e-18 sec
    return H0

def aHz2H0(vH0):
    vH0 *= 3.08567758149137e1
    return vH0

def H02Gyrs(H0arr):
    H0arr = np.asarray(H0arr)    # assume in units of km/s/Mpc
    invH0 = 1./H0arr.copy()      # s Mpc/km
    invH0 *= 3.08567758149137e3/np.pi  # Gyrs (Mpc=3.08567758e22 m; Gyr=pi*1e16 sec; km=1e3 m)
    return invH0

def H02critdens(H0arr):
    c = 299792458.0            # m s^-1
    G = 6.67430e-11            # kg^-1 m^3 s^-2
    GeV = 1.602176634e-10      # J/GeV
    H0arr = np.asarray(H0arr)  # assume in units of km/s/Mpc
    H02 = H0arr.copy()**2
    H02 /= 3.08567758149137e19*3.08567758149137e19  # s^-2
    c2_8piG = 3*c**2/(8*np.pi*G) # kg m^2 m^-3
    rhocrit = H02 * c2_8piG    # J / m^3
    rhocrit /= 1.602176634e-10 # GeV / m^3
    return rhocrit


class units(object):
    pass


localvars = vars()
allkeys = list(localvars.keys())
for name in allkeys:
    if not name == 'units' and not name.startswith('__'):
        setattr(units, name, localvars[name])
