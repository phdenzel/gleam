#!/usr/bin/env python
"""
@author: phdenzel

Useful transformations for data from MegaPrime/MegaCam from CFHT


Note: Some of the functions are only effective, if all the bands are provided
"""
###############################################################################
# Imports
###############################################################################
import numpy as np
from scipy.integrate import trapz
from astropy import cosmology, units
import sys
import warnings  # remove RuntimeWarning caused by log10
warnings.filterwarnings("ignore", module="magnitudes")


###############################################################################
def ADU2MegaCam(ADUdata, photzp=30., nan_filter=True):
    """
    Converts input to AB magnitude

    Args:
       ADUdata: np.array or double; flux measured from observations

    Kwargs:
       photzp:     int or double; photometric zero-point
       nan_filter: filter out NaNs, occurring in applying log
    """
    ADUdata = np.array(ADUdata)
    try:
        np.array(photzp).__len__
        photzp = np.array([p*np.ones(ADUdata.shape[1:]) for p in photzp])
        # transform
        transf = -2.5*np.log10(ADUdata)[:]+photzp
    except:
        # transform
        transf = -2.5*np.log10(ADUdata)+photzp
    if nan_filter:
        # filter NaNs and replace by maximum mag (i.e. minimum brightness)
        mag_max = np.max(np.nan_to_num(transf))
        loc_nan = np.isnan(transf)
        transf[loc_nan] = mag_max
    return transf


def MegaCam2SSDS(ABdata):
    """
    Converts MegaCam's AB magnitudes to SDSS magnitudes
    Args:
       ABdata: list(np.array) or list(double); AB magnitude data of all bands
               ordered as (U,G,R,I,I2,Z or U,G,R,I,Z)
    """
    SDSS_transformed = []
    try:
        N = len(ABdata)
    except:
        N = ABdata.size
    if (N == 6):
        # U band
        SDSS_transformed.append(ABdata[0] + 0.181 * (ABdata[0] - ABdata[1]))
        # G band
        SDSS_transformed.append(ABdata[1] + 0.195 * (ABdata[1] - ABdata[2]))
        # R band
        SDSS_transformed.append(ABdata[2] + 0.011 * (ABdata[1] - ABdata[2]))
        # I band (old)
        SDSS_transformed.append(ABdata[3] + 0.079 * (ABdata[2] - ABdata[3]))
        # I2 band (new)
        SDSS_transformed.append(ABdata[3] + 0.001 * (ABdata[2] - ABdata[3]))
        # Z band
        SDSS_transformed.append(ABdata[5] - 0.099 * (ABdata[3] - ABdata[5]))
    elif (N == 5):
        # U band
        SDSS_transformed.append(ABdata[0] + 0.181 * (ABdata[0] - ABdata[1]))
        # G band
        SDSS_transformed.append(ABdata[1] + 0.195 * (ABdata[1] - ABdata[2]))
        # R band
        SDSS_transformed.append(ABdata[2] + 0.011 * (ABdata[1] - ABdata[2]))
        # I band (old)
        SDSS_transformed.append(ABdata[3] + 0.079 * (ABdata[2] - ABdata[3]))
        # Z band
        SDSS_transformed.append(ABdata[4] - 0.099 * (ABdata[3] - ABdata[4]))
    else:
        print("SDSS transform only possible with all filter bands"
              + " ordered as (U,G,R,I,Z or U,G,R,I,I2,Z)...")
        sys.exit(1)
    return SDSS_transformed


def magnitude2ADU(ABdata, system=None, photzp=30.):
    """
    Inverse AB magnitude transform to ADU data

    Args:
       ABdata: list(np.array) or list(double); AB magnitude data of all bands
               ordered as (U,G,R,I,I2,Z or U,G,R,I,Z)

    Kwargs:
       system: string; magnitude system type, can be either 'MegaCam' or 'SDSS'
       photzp: double; photometric zero-point
    """
    AB_transformed = []
    try:
        N = len(ABdata)
    except:
        N = ABdata.size
    if (system == 'MegaCam'):
        return 10**(0.4*(photzp-ABdata))
    elif (system == 'SDSS'):
        AB_transformed = []
        if (N == 6):
            # U band
            AB_transformed.append(ABdata[0] - 0.241 * (ABdata[0] - ABdata[1]))
            # G band
            AB_transformed.append(ABdata[1] - 0.153 * (ABdata[1] - ABdata[2]))
            # R band
            AB_transformed.append(ABdata[2] - 0.024 * (ABdata[1] - ABdata[2]))
            # I band (old)
            AB_transformed.append(ABdata[3] - 0.085 * (ABdata[2] - ABdata[3]))
            # I2 band (new)
            AB_transformed.append(ABdata[3] - 0.003 * (ABdata[2] - ABdata[3]))
            # Z band
            AB_transformed.append(ABdata[5] + 0.074 * (ABdata[3] - ABdata[5]))
        elif (N == 5):
            # U band
            AB_transformed.append(ABdata[0] - 0.241 * (ABdata[0] - ABdata[1]))
            # G band
            AB_transformed.append(ABdata[1] - 0.153 * (ABdata[1] - ABdata[2]))
            # R band
            AB_transformed.append(ABdata[2] - 0.024 * (ABdata[1] - ABdata[2]))
            # I band (old)
            AB_transformed.append(ABdata[3] - 0.085 * (ABdata[2] - ABdata[3]))
            # Z band
            AB_transformed.append(ABdata[4] + 0.074 * (ABdata[3] - ABdata[4]))
        else:
            print("For SSDS magnitudes, all bands have to be provided...")
            return None
        return 10**(0.4*(photzp-np.array(AB_transformed)))
    else:
        print("Assuming input is in MegaCam AB magnitudes...")
        return 10**(0.4*(photzp-ABdata))


def _read_filters(bands=['u', 'g', 'r', 'i', 'z'],
                  directory="glfits/stelmass/"):
    """
    Read MegaCam filter list into a dictionary
    """
    filters = {}
    filters['bands'] = bands
    filters['indices'] = range(len(filters['bands']))
    for i, band in enumerate(filters['bands']):
        filters[band] = {}
        filters.setdefault(band, {})['index'] = i
        # read data files
        prefix = [s for s in sys.path if s.endswith('/glfits')][0]+"/"
        path = prefix+directory+"CFHT_Mega_"+band[0]+".flt"
        filters.setdefault(band, {})['filename'] = path
        lam, resp = np.loadtxt(path, unpack=True, usecols=(0, 1))
        # normalize filter responses
        norm = trapz(resp, lam)
        resp /= norm
        # fzp - 1A interpolation  - AB reference: Fnu = constant
        w = np.arange(np.amin(lam), np.amax(lam), 1.0)
        fzp = 25000000./(w*w) * np.interp(w, lam, resp)
        # write to dictionary
        filters.setdefault(band, {})['wavelength'] = lam
        filters.setdefault(band, {})['response'] = resp
        filters.setdefault(band, {})['fZP'] = trapz(fzp, w)
    return filters


def ABfromFilters(wavelengths, flux, z, filters):
    """
    Calculate magnitudes from an SED value off MegaCam filters

    Args:
       flux: np.array; sed information from base spectra
       z:    double; redshift

    Kwargs:
       filters: np.array; (output from read_filters) band filter data from
                MegaCam (u,g,r,i,z); including bands, wavelengths, responses,
                photometric flux zero-points
    """
    mag = []
    for b in filters['bands']:
        # wavelength range
        wlmin = np.amin(filters[b]['wavelength'])
        wlmax = np.amax(filters[b]['wavelength'])
        # filter wavelengths
        wl = np.arange(wlmin, wlmax, 1.0)
        # wavelengths from base models
        wl_bm = wavelengths*(1.0+z)
        # 1A interpolation
        fl = np.interp(wl, wl_bm, flux)
        res = np.interp(wl, filters[b]['wavelength'], filters[b]['response'])
        f_res = fl * res
        # magnitude
        m = -2.5*np.log10(trapz(f_res, wl)/filters[b]['fZP']) - 4.69427
        if z > 0.0:
            dm = float(cosmology.Planck15.distmod(z)/units.mag)
            m += dm + 2.5*np.log10(1.0+z)
        mag.append(m)
    return np.array(mag)


def _dust_CCM89(w):
    """
    Account for dust attenuation (CCM89) in base model spectra
    """
    nu = 10000./w
    k_wl = 0*nu
    # IR
    IR_msk = (nu > 0.3) & (nu < 1.1)
    k_wl[IR_msk] = 0.404*np.power(10.0, 1.61)
    # Optical / Near-IR
    opt_msk = (nu > 1.1) & (nu < 3.3)
    y = nu[opt_msk] - 1.82
    a0 = (1.0 + y*(0.17699 + y*(-0.50447 + y*(-0.02427 + y*(0.72085
          + y*(0.01979 + y*(-0.77530+0.32999*y)))))))
    b0 = (y*(1.41338 + y*(2.28305 + y*(1.07233 + y*(-5.38434 + y*(-0.62251
          + y*(5.30260 - 2.09002*y)))))))
    k_wl[opt_msk] = a0 + b0/3.1
    # UV
    UV1_msk = (nu > 3.3) & (nu < 5.9)
    UV2_msk = (nu > 5.9) & (nu < 8.0)
    aux = nu[UV2_msk] - 5.9
    Fa = aux*aux * (-0.04473 - 0.009779*aux)
    Fb = aux*aux * (0.2130 + 0.1207*aux)
    a1 = (1.752 - 0.316*nu[UV1_msk] - 0.104/(0.341 + (nu[UV1_msk]-4.67)
          * (nu[UV1_msk]-4.67)))
    a2 = (1.752 - 0.316*nu[UV2_msk] - 0.104/(0.341 + (nu[UV2_msk]-4.67)
          * (nu[UV2_msk]-4.67)) + Fa)
    b1 = (-3.090 + 1.825*nu[UV1_msk] + 1.206/(0.263 + (nu[UV1_msk]-4.62)
          * (nu[UV1_msk]-4.62)))
    b2 = (-3.090 + 1.825*nu[UV2_msk] + 1.206/(0.263 + (nu[UV2_msk]-4.62)
          * (nu[UV2_msk]-4.62)) + Fb)
    k_wl[UV1_msk] = a1 + b1/3.1
    k_wl[UV2_msk] = a2 + b2/3.1
    # Far-UV
    fUV_msk = (nu > 8.0) & (nu < 10.0)
    aux = nu[fUV_msk] - 8.0
    af = -1.073 + aux*(-0.628 + aux*(0.137 - 0.070*aux))
    bf = 13.670 + aux*(4.257 + aux*(-0.420 + 0.374*aux))
    k_wl[fUV_msk] = af + bf/3.1
    return 3.1*k_wl


def run_bpz(magnitudes, errors, catalogue):
    """
    TODO
    """
    # N = len(magnitudes)
    pass
