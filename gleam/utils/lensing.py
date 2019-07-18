#!/usr/bin/env python
"""
@author: phdenzel

Lensing utility functions for calculating various analytics from lensing mass maps
"""
###############################################################################
# Imports
###############################################################################
import numpy as np
import scipy
from scipy import interpolate
from scipy.integrate import odeint
from matplotlib import pyplot as plt
from gleam.utils.linalg import eigvals, eigvecs, angle


###############################################################################
def downsample_model(kappa, extent, shape, pixel_scale=1., verbose=False):
    """
    Resample (usually downsample) a model's kappa grid to match the specified scale and size

    Args:
        kappa <np.ndarray> - the model's with (data, hdr)
        extent <tuple/list> - extent of the output
        shape <tuple/list> - shape of the output

    Kwargs:
        pixel_scale <float> - the pixel scale of the input kappa grid
        verbose <bool> - verbose mode; print command line statements

    Return:
        kappa_resmap <np.ndarray> - resampled kappa grid
    """
    pixrad = tuple(r//2 for r in kappa.shape)
    maprad = pixrad[1]*pixel_scale

    if verbose:
        print("Kappa grid: {}".format(kappa.shape))
        print("Pixrad {}".format(pixrad))
        print("Maprad {}".format(maprad))

    xmdl = np.linspace(-maprad, maprad, kappa.shape[0])
    ymdl = np.linspace(-maprad, maprad, kappa.shape[1])
    newx = np.linspace(extent[0], extent[1], shape[0])
    newy = np.linspace(extent[2], extent[3], shape[1])

    rescale = interpolate.interp2d(xmdl, ymdl, kappa)
    kappa_resmap = rescale(newx, newy)
    kappa_resmap[kappa_resmap < 0] = 0

    return kappa_resmap


def upsample_model(gls_model, extent, shape, verbose=False):
    """
    Resample (usually upsample) a model's kappa grid to match the specified scales and size

    Args:
        gls_model <glass.LensModel object> - GLASS ensemble model
        extent <tuple/list> - extent of the output
        shape <tuple/list> - shape of the output

    Kwargs:
        verbose <bool> - verbose mode; print command line statements

    Return:
        kappa_resmap <np.ndarray> - resampled kappa grid
    """
    obj, data = gls_model['obj,data'][0]
    kappa_map = obj.basis._to_grid(data['kappa'], 1)
    pixrad = obj.basis.pixrad
    maprad = obj.basis.top_level_cell_size * (obj.basis.pixrad)
    mapextent = (-obj.basis.top_level_cell_size * (obj.basis.pixrad+0.5),
                 obj.basis.top_level_cell_size * (obj.basis.pixrad+0.5))
    cell_size = obj.basis.top_level_cell_size

    if verbose:
        print(obj)
        print("Kappa map: {}".format(kappa_map.shape))
        print("Pixrad {}".format(pixrad))
        print("Maprad {}".format(maprad))
        print("Mapextent {}".format(mapextent))
        print("Cellsize {}".format(cell_size))

    xmdl = np.linspace(-maprad, maprad, kappa_map.shape[0])
    ymdl = np.linspace(-maprad, maprad, kappa_map.shape[1])
    Xmdl, Ymdl = np.meshgrid(xmdl, ymdl)
    xnew = np.linspace(extent[0], extent[1], shape[0])
    ynew = np.linspace(extent[2], extent[3], shape[1])
    Xnew, Ynew = np.meshgrid(xnew, ynew)

    rescale = interpolate.Rbf(Xmdl, Ymdl, kappa_map)
    # rescale = interpolate.interp2d(xmdl, ymdl, kappa_map)
    kappa_resmap = rescale(xnew, ynew)
    kappa_resmap[kappa_resmap < 0] = 0

    return kappa_resmap


def radial_profile(data, center=None, bins=None):
    """
    Calculate radial profiles of some data maps

    Args:
        data <np.ndarray> - 2D data array

    Kwarg:
        center <tuple/list> - center indices of profile

    Return:
        radial_profile <np.ndarray> - radially-binned 1D profile
    """
    if data is None:
        return None
    N = data.shape[0]
    if bins is None:
        bins = N//2
    if center is None:
        # center = np.unravel_index(data.argmax(), data.shape)
        center = [c//2 for c in data.shape][::-1]
    dta_cent = data[center[0], center[1]]
    x, y = np.indices((data.shape))
    r = np.sqrt((x - center[0])**2 + (y - center[1])**2)
    r = r.reshape(r.size)
    data = data.reshape(data.size)
    rbins = np.linspace(0, N//2, bins)
    encavg = [np.sum(data[r < ri])/len(data[r < ri]) if len(data[r < ri]) else dta_cent for ri in rbins]
    return encavg


def complex_ellipticity(a, b, phi):
    """
    Calculate the complex ellipticity

    Args:
        a <float> - semi-major axis
        b <float> - semi-minor axis
        phi <float> - position angle

    Kwargs:
        None

    Return:
        e1, e2 <float,float> - complex ellipticty vector components
    """
    a = np.asarray(a)
    b = np.asarray(b)
    phi = np.asarray(phi)
    Q = (a-b)/(a+b)
    return np.array([Q*np.cos(2*phi), Q*np.sin(2*phi)])


def Dcomov(r, a, cosmology=None):
    """
    Calculate the comoving distance from scale factor, for odeint

    Args:
        r <float> - distance of
    """
    if cosmology is None:
        Omega_l = 0.72
        Omega_r = 4.8e-5
        Omega_k = 0
        Omega_m = 1. - Omega_r - Omega_l - Omega_k
    else:
        Omega_l = cosmology['Omega_l']
        Omega_r = cosmology['Omega_r']
        Omega_k = cosmology['Omega_k']
        Omega_m = cosmology['Omega_m']
    H = (Omega_m/a**3 + Omega_r/a**4 + Omega_l + Omega_k/a**2)**.5
    return 1./(a*a*H)


def DL(zl, zs):
    """
    Distance D_L (for scaling kpc distances)

    Args:
        zl <float> - lens redshift
        zs <float> - source redshift

    Kwargs:
        None

    Return:
        Dl <float> - distance D_L
    """
    alens = 1./(1+zl)
    asrc = 1./(1+zs)
    a = [asrc, alens, 1]
    r = odeint(Dcomov, [0], a)[:, 0]
    Dl = alens * (r[2] - r[1])
    return Dl


def DLDSDLS(zl, zs):
    """
    Distance D_L*D_S/D_LS (for scaling masses)

    Args:
        zl <float> - lens redshift
        zs <float> - source redshift

    Kwargs:
        None

    Return:
        DlDsDls <float> - distance D_L*D_S/D_LS
    """
    alens = 1./(1+zl)
    asrc = 1./(1+zs)
    a = [asrc, alens, 1]
    r = odeint(Dcomov, [0], a)[:, 0]
    Dl = alens * (r[2] - r[1])
    Ds = asrc * r[2]
    Dls = asrc * r[1]
    return Dl*Ds/Dls


def DLSDS(zl, zs):
    """
    Distance ratio D_LS/D_S (for scaling kappa maps)

    Args:
        zl <float> - lens redshift
        zs <float> - source redshift

    Kwargs:
        None

    Return:
        ratio <float> - distance ratio D_S/D_L
    """
    alens = 1./(1+zl)
    asrc = 1./(1+zs)
    a = [asrc, alens, 1]
    r = odeint(Dcomov, [0], a)[:, 0]
    Ds = asrc * r[2]
    Dls = asrc * r[1]
    return Dls/Ds


def center_of_mass(kappa, pixel_scale=1, center=True):
    """
    Calculate the 2D position of the center of mass

    Args:
        kappa <np.ndarray> - a 2D grid of kappa tiles/pixels

    Kwargs:
        pixel_scale <float> - the pixel scale
        center <bool> - return the COM relative to the map center

    Return:
        com <np.ndarray> - center of mass on the kappa coordinate grid as (x, y)
    """
    invM = 1./np.sum(kappa)
    x1 = np.linspace(-(kappa.shape[0]//2), kappa.shape[0]//2, kappa.shape[0])
    y1 = np.linspace(-(kappa.shape[1]//2), kappa.shape[1]//2, kappa.shape[1])
    x1 *= pixel_scale
    y1 *= pixel_scale
    x, y = np.meshgrid(x1, y1)
    rk = np.array([x*kappa, y*kappa])
    com = np.sum(invM*rk, axis=(2, 1))
    if not center:
        com += pixel_scale * (np.array(kappa.shape) // 2)
    return com


def inertia_tensor(kappa, pixel_scale=1, activation=None, com_correct=True):
    """
    Tensor of inertia for a kappa grid

    Args:
        kappa <np.ndarray> - a 2D grid of kappa tiles/pixels

    Kwargs:
        pixel_scale <float> - the pixel scale
        activation <float> - a threshold value below which pixel values are ignored
        com_correct <bool> - if True, the coordinates shift to the com

    Note:
        The diag. matrix will have a^2/4, b^2/4 (semi-major/semi-minor axes) as entries
    """
    if activation is not None:
        # kappa_map[kappa_map >= activation] = 1
        kappa[kappa < activation] = 0

    x1 = np.linspace(-(kappa.shape[0]//2), kappa.shape[0]//2,
                     kappa.shape[0]) * pixel_scale
    y1 = np.linspace(-(kappa.shape[1]//2), kappa.shape[1]//2,
                     kappa.shape[1]) * pixel_scale
    x, y = np.meshgrid(x1, y1)
    if com_correct:
        com = center_of_mass(kappa, pixel_scale=pixel_scale)
        x -= com[0]
        y -= com[1]
    yx = xy = y*x
    N = 1./np.sum(kappa)
    Ixx = N*np.sum(kappa*x*x)
    Ixy = N*np.sum(kappa*xy)
    Iyx = N*np.sum(kappa*yx)
    Iyy = N*np.sum(kappa*y*y)
    return np.matrix([[Ixx, Ixy], [Iyx, Iyy]])


def qpm_props(qpm, verbose=False):
    """
    Calculate properties of the quadrupole moment:
        semi-major axis, semi-minor axis, and position angle

    Args:
        qpm <np.matrix> - a 2x2 matrix of the quadrupole moment, i.e. inertia tensor

    Kwargs:
        None

    Return:
        a, b, phi <float> - semi-major, semi-minor axes, position angle

    Note:
        The diag. matrix will have a^2/4, b^2/4 (semi-major/semi-minor axes) as entries
    """
    evl = eigvals(qpm)
    evc = eigvecs(qpm)
    a, b = 2*np.sqrt(evl)
    cosphi = angle(evc[0], [1, 0])
    sinphi = angle(evc[0], [0, 1])
    if verbose:
        print("Eigen vectors: {}".format(evc))
    return a, b, np.arctan2(sinphi, cosphi)


def lnr_indef(x, y, x2=None, y2=None):
    """
    Indefinite ln(theta) integral for a lensing potential

    Args:
        x, y <float/np.ndarray> - theta coordinate components

    Kwargs:
        x2, y2 <float/np.ndarray> - optionally the squared arguments can be passed
                                    to increase efficiency
    """
    if x2 is None:
        x2 = x*x
    if y2 is None:
        y2 = y*y
    return x*y*(np.log(x2+y2) - 3) + x2*np.arctan(y/x) + y2*np.arctan(x/y)


def lnr(x, y, xn, yn, a):
    """
    Potential ln(r) contribution of the n-th pixel

    Args:
        x, y <float/np.ndarray> - theta coordinate components of the potential
        xn, yn <float/np.ndarray> - pixel coordinates of the kappa grid
        a <float> - pixel scale of the kappa grid
    """
    xm, xp = x-xn-0.5*a, x-xn+0.5*a
    ym, yp = y-yn-0.5*a, y-yn+0.5*a
    xm2, xp2 = xm*xm, xp*xp
    ym2, yp2 = ym*ym, yp*yp
    return lnr_indef(xm, ym, xm2, ym2) + lnr_indef(xp, yp, xp2, yp2) \
        - lnr_indef(xm, yp, xm2, yp2) - lnr_indef(xp, ym, xp2, ym2)


def potential_grid(kappa, N, grid_size, verbose=False):
    """
    The entire potential grid

    Args:
        kappa <np.ndarray> - the kappa grid
        N <int> - number of grid points along the axes of the potential grid
        grid_size <float> - the length of the grid along the axes of the kappa grid

    Kwargs:
        verbose <bool> - verbose mode; print command line statements

    Return:
        gx, gy, psi <np.ndarray> - the x and y grid coordinates and the potential grid
    """
    N += (N % 2 == 0) and 1
    x = np.linspace(-grid_size/2., grid_size/2., N)
    y = np.linspace(-grid_size/2., grid_size/2., N)
    xkappa = np.linspace(-grid_size/2., grid_size/2., kappa.shape[0])
    ykappa = np.linspace(-grid_size/2., grid_size/2., kappa.shape[1])
    pixel_size = xkappa[-1] - xkappa[-2]

    gx, gy = np.meshgrid(x, y)

    if verbose:
        print("kappa:     \t{}".format(kappa.shape))
        print("pixel_size:\t{}".format(pixel_size))
        print("N:         \t{}".format(N))
        print("grid_size: \t{}".format(grid_size))

    psi = np.zeros((N, N))
    for m, ym in enumerate(ykappa):
        for n, xn in enumerate(xkappa):
            psi += kappa[m, n]*lnr(gx, gy, xn, ym, pixel_size)
    psi *= -1./(2*np.pi)

    return gx, gy, psi


def degarr_grid(*args, **kwargs):
    """
    The degenerate arrival time surface without source shift: 1/2 theta^2 - psi

    Args:
        kappa <np.ndarray> - the kappa grid
        N <int> - number of grid points along the axes of the potential grid
        grid_size <float> - the length of the grid along the axes of the kappa grid

    Kwargs:
        verbose <bool> - verbose mode; print command line statements

    Return:
        gx, gy, psi <np.ndarray> - the x and y grid coordinates and the potential grid
    """
    gx, gy, psi = potential_grid(*args, **kwargs)
    return gx, gy, 0.5*(gx*gx + gy*gy) + psi


def kappa_map_plot(model, obj_index=0, subcells=1, extent=None,
                   contours=False, levels=3, delta=0.2, log=True,
                   oversample=True,
                   cmap='magma', colorbar=False):
    """
    Plot the convergence map of a GLASS model with auto-adjusted contour levels

    Args:
        model <glass.LensModel object> - GLASS ensemble model

    Kwargs:
        obj_index <int> - object index of the GLASS object within the model
        subcells <int> - number of subcells
        extent <tuple/list> - map extent
        contours <bool> - plot contours onto map
        levels <int> - half-number of contours; total number w/ log is 2*N+1
        delta <float> - contour distances
        log <bool> - plot in log scale
        oversample <bool> - oversample the map to show characteristic pixel structure
        cmap <str/mpl.cm.ColorMap object> - color map for plotting
        colorbar <bool> - plot colorbar next to convergence map
    """
    obj, dta = model['obj,data'][obj_index]
    dlsds = DLSDS(obj.z, obj.sources[obj_index].z)
    grid = dlsds * obj.basis._to_grid(dta['kappa'], subcells)

    if extent is None:
        # R = obj.basis.maprad
        R = obj.basis.mapextent
        extent = [-R, R, -R, R]
    # pixrad = obj.basis.pixrad

    # masking if necessary
    msk = grid != 0
    if not np.any(msk):
        vmin = -15
        grid += 10**vmin
    else:
        vmin = np.log10(np.amin(grid[msk]))

    # interpolate
    zoomlvl = 3
    order = 0
    if oversample:
        grid = scipy.ndimage.zoom(grid, zoomlvl, order=order)
    grid[grid <= 10**vmin] = 10**vmin

    # contour levels
    clev2 = np.arange(delta, levels*delta, delta)
    clevels = np.concatenate((-clev2[::-1], (0,), clev2))

    if log:
        kappa1 = 0
        grid = np.log10(grid)
        grid[grid < clevels[0]] = clevels[0]
    else:
        kappa1 = 1
        clevels = np.concatenate(((0,), 10**clev2))

    plt.contourf(grid, cmap=cmap, antialiased=True,
                 extent=extent, origin='lower', levels=clevels)
    if colorbar:
        cbar = plt.colorbar()
        if log:
            lvllbls = ['{:1.1f}'.format(l) if i > 0 else '0'
                       for (i, l) in enumerate(10**cbar._tick_data_values)]
            cbar.ax.set_yticklabels(lvllbls)

    if contours:
        plt.contour(grid, levels=(kappa1,), colors=['k'],
                    extent=extent, origin='lower')
