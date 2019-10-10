#!/usr/bin/env python
"""
@author: phdenzel

Lensing utility functions for calculating various analytics from lensing observables
"""
###############################################################################
# Imports
###############################################################################
import os
import numpy as np
from functools import partial
from scipy import interpolate
from scipy import optimize
from scipy.integrate import odeint
from gleam.utils.linalg import eigvals, eigvecs, angle
import gleam.utils.units as gu
# print(globals().keys())
from gleam.glass_interface import glass_basis, glass_renv
glass = glass_renv()


###############################################################################
class ModelArray(object):
    """
    Model class which contains data and some grid information
    """
    __framework__ = 'array'

    def __init__(self, data, filename=None,
                 grid_size=None, pixrad=None, maprad=None, kappa=None,
                 verbose=False):
        """
        Args:
            data <np.ndarray> - ensemble data

        Kwargs:
            filename <str> - filename containing the lens data
            pixrad <int> - pixel radius of the grid
            maprad <float> - map radius of the grid in arcsec
            kappa <float> - kappa conversion factor in Msol/arcsec^2
        """
        self.filepath = filename
        self.filename = os.path.basename(filename) if filename else filename
        self.data = np.asarray(data) if np.any(data) else None
        if grid_size is not None:
            self.grid_size = grid_size
        if pixrad is not None:
            self.pixrad = int(pixrad) if pixrad else self.pixrad
        self.maprad = maprad
        self.kappa = kappa
        if data is None:
            pass
        elif len(self.data.shape) == 3:
            self.models = [d for d in self.data]
        elif len(self.data.shape) == 2:
            self.models = [self.data]
        elif len(self.data.shape) == 1 and self.data.size == self.grid_size**2:
            self.data = self.data.reshape((self.grid_size, self.grid_size))
            self.models = [self.data]
        elif len(self.data.shape) == 1 and (self.data.size % self.grid_size**2) == 1:
            N = self.data.size // (self.grid_size**2)
            self.data = self.data.reshape((N, self.grid_size, self.grid_size))
            self.models = [d for d in self.data]

    @property
    def tests(self):
        return ['filename', 'filepath', 'N', 'pixrad', 'maprad', 'kappa']

    @property
    def __v__(self):
        return "\n".join([t.ljust(20)+"\t{}".format(self.__getattribute__(t))
                          for t in self.tests])

    def __getitem__(self, index):
        return self.models[index]

    @property
    def N(self):
        if hasattr(self, 'models'):
            return len(self.models)
        elif len(self.data.shape) == 3:
            return self.data.shape[0]
        elif len(self.data.shape) == 2:
            return 1
        else:
            return None

    @property
    def extent(self):
        return [-self.maprad, self.maprad, -self.maprad, self.maprad]

    @property
    def grid_size(self):
        return 2*self.pixrad + 1

    @grid_size.setter
    def grid_size(self, grid_size):
        self.pixrad = grid_size//2

    def kappa_grid(self, model_index=0):
        return self[model_index]

    def sigma_grid(self, model_index=0):
        return self.kappa*self[model_index]

    def potential_grid(self, model_index=0, N=None, factor=1, ext=None):
        N = self.data.shape[-1] if N is None else N
        model = self.data_hires[model_index] if hasattr(self, 'data_hires') else self[model_index]
        factor = 1./self.dlsds[model_index] if hasattr(self, 'dlsds') else 1
        ext = [self.shear_grid(model_index)]
        _, _, grid = potential_grid(model, N, 2*self.maprad, factor=factor, ext=ext)
        return grid

    def shear_grid(self, model_index=0, N=None):
        N = self.data.shape[-1] if N is None else N
        shear = self.shears[model_index] if hasattr(self, 'shears') else np.array([0, 0])
        gx, gy, grid = shear_grid(shear, N, 2*self.maprad)
        return grid

    def arrival_grid(self, model_index=0, N=None, beta=None, factor=None, ext=None):
        N = self.data.shape[-1] if N is None else N
        model = self.data[model_index]
        beta = self.betas[model_index] if hasattr(self, 'betas') else np.array([0, 0])
        factor = 1./self.dlsds[model_index] if hasattr(self, 'dlsds') else 1
        ext = [self.ext_potentials[model_index]] if hasattr(self, 'ext_potentials') else None
        _, _, grid = arrival_grid(model, N, 2*self.maprad, beta, factor=factor, ext=ext)
        return grid

    def roche_potential_grid(self, model_index=0, N=None, factor=1, ext=None):
        N = self.data.shape[-1] if N is None else N
        model = self.data[model_index]
        _, _, grid = roche_potential_grid(model, N, 2*self.maprad, factor=factor, ext=ext)
        return grid


class PixeLensModel(ModelArray):
    """
    Model class which loads PixeLens .state, .dat, or .txt files
    """
    __framework__ = 'PixeLens'

    fields = {
        'sigcrit': 'kappa',
        'x range': 'maprad',
        'y range': 'maprad',
        'grid size': 'grid_size'}

    def __init__(self, filename, **kwargs):
        with open(os.path.abspath(filename), 'rb') as f:
            self.content = f.readlines()
        hdr_info, hdr_offset = self.parse_header(self.content)
        data = self.parse_dat(self.content, offset=hdr_offset)
        super(PixeLensModel, self).__init__(data, filename=filename, **hdr_info)

    @staticmethod
    def parse_header(text):
        """
        Parse PixeLens .dat files and extract header information

        Args:
            text <list(str)> - read lines from .dat file

        Kwargs:
            None

        Return:
            hdr <dict> - header info with variables names and values
            hdr_offset <int> - index where header ends
        """
        hdr = {}
        for i, line in enumerate(text):
            hdr_offset = i+1
            if line == '\n':
                break
            items = [[ie.strip() for ie in e.split(' ') if ie]
                     if idx > 0 else e for idx, e in enumerate(line.split(':'))]
            label = items[0]
            value = [float(v) for v in items[1] if v[0].isdigit()][0]
            if label in PixeLensModel.fields:
                hdr[PixeLensModel.fields[label]] = value
        return hdr, hdr_offset

    @staticmethod
    def parse_dat(text, offset=5):
        """
        Parse PixeLens .dat files and extract data

        Args:
            text <list(str)> - read lines from .dat file

        Kwargs:
            offset <int> - offset in case there is a header

        Return:
            dta <np.ndarray> - data array
        """
        dta = []
        model = []
        for line in text[offset:]:
            if line == '\n':
                dta.append(model)
                model = []
                continue
            dta_column = [float(e) for e in line.strip().split(' ')]
            model.append(dta_column)
        dta = np.asarray(dta)
        return dta


class GLASSModel(ModelArray):
    """
    Wrapper class for glass models

    TODO:
        - <src_idx> not yet handled correctly
    """
    __framework__ = 'GLASS'

    def __init__(self, env, filename=None, obj_index=0, src_index=0, verbose=False):
        """
        Args:
            gls <glass.environment.Environment object> - all model data

        Kwargs:
            filename <str> - filename of the state file
            obj_index <int> - index of the current object index
            src_index <int> - index of the source index
        """
        self.env = env
        cosmo = cosmology(self.env.omega_matter, self.env.omega_lambda)
        if not hasattr(self, 'all_data'):
            dta = {'hires': [], 'toplevel': [], 'dlsds': [], 'cosmology': cosmo}
            for obj_idx in range(self.N_obj):
                mdta_toplvl = []
                mdta_hires = []
                mdta_dlsds = []
                for m in self.env.models:
                    obj, data = m['obj,data'][obj_idx]
                    dlsds = DLSDS(obj.z, obj.sources[src_index].z, cosmo=cosmo)
                    toplvl = dlsds * obj.basis._to_grid(data['kappa'], 1)
                    hires = dlsds * obj.basis.kappa_grid(data)
                    mdta_toplvl.append(toplvl)
                    mdta_hires.append(hires)
                    mdta_dlsds.append(dlsds)
                dta['toplevel'].append(mdta_toplvl)
                dta['hires'].append(mdta_hires)
                dta['dlsds'].append(mdta_dlsds)
            for k in dta:
                if isinstance(dta[k], list):
                    dta[k] = np.array(dta[k], dtype=np.float32)
            self.all_data = dta
        self.filepath = filename
        self.filename = os.path.basename(filename)
        self.obj_idx = obj_index  # calls super(GLASSModel, self)__init__()
        self.src_idx = src_index
        if self.N_src > 1:  # TODO: can be removed once multiple sources are handled correctly
            print("Warning! There are multiple sources!")

    @classmethod
    def from_filename(cls, filename, **kwargs):
        env = glass.glcmds.loadstate(filename)
        kwargs.update({'filename': filename})
        return cls(env, **kwargs)

    @classmethod
    def from_filenames(cls, *filenames):
        return (cls(glass.glcmds.loadstate(f), filename=f) for f in filenames)

    @property
    def tests(self):
        return super(GLASSModel, self).tests + ['N_obj', 'N_src', 'obj_idx', 'src_idx']

    @property
    def data_hires(self):
        return self.all_data['hires'][self.obj_idx]

    @property
    def data_toplevel(self):
        return self.all_data['toplevel'][self.obj_idx]

    @property
    def dlsds(self):
        return self.all_data['dlsds'][self.obj_idx]

    @property
    def gcm(self):
        """
        Get current models from GLASS environment
        """
        return [m['obj,data'][self.obj_idx] for m in self.env.models]

    @property
    def betas(self):
        """
        All modeled source positions
        """
        betas = np.array([m[1]['src'] for m in self.gcm]).view(np.float)
        return betas

    @property
    def shears(self):
        """
        All modeled shear components
        """
        shears = np.array([m[1]['shear'] for m in self.gcm])
        return shears

    @property
    def ext_potentials(self):
        """
        Get the external potential terms
        """
        ext_potentials = [self.shear_grid(i) for i in range(self.N)]
        return np.asarray(ext_potentials)

    @property
    def obj_idx(self):
        return self._obj_idx

    @obj_idx.setter
    def obj_idx(self, obj_index):
        self._obj_idx = obj_index
        dta = self.all_data['hires'][obj_index]
        obj, data = self.env.models[0]['obj,data'][obj_index]
        kappa = glass.scales.convert('kappa to Msun/arcsec^2', 1,
                                     obj.dL, self.env.nu[obj_index])
        super(GLASSModel, self).__init__(dta, filename=self.filepath,
                                         pixrad=obj.basis.pixrad,
                                         maprad=obj.basis.mapextent,
                                         kappa=kappa)

    @property
    def src_idx(self):
        return self._src_idx

    @src_idx.setter
    def src_idx(self, src_index):
        self._src_idx = src_index

    @property
    def N_obj(self):
        return len(self.env.models[0]['obj,data'])

    @property
    def N_src(self):
        return len(self.env.models[0]['obj,data'][self.obj_idx][0].sources)


class MetaModel(type):
    def __new__(mcls, name, bases, cdict):
        return super(MetaModel, mcls).__new__(mcls, name, bases, cdict)

    def __instancecheck__(cls, instance):
        stype = cls.__name__ if hasattr(cls, '__name__') else type(cls).__name__
        otype = instance.__name__ if hasattr(instance, '__name__') else type(instance).__name__
        return stype == otype


class LensModel(MetaModel):
    """
    General model class which standardizes convergence model data from various formats
    (This is the only class one needs to import from this module)

    Note:
        - inherits from metaclass, i.e. instances should be classes too,
          however the constructor was overidden and immediately returns
          an instance of said class depending on the format of the input
    """
    __metaclass__ = MetaModel

    def __new__(mcls, args, **kwargs):
        # default
        bases = (ModelArray, object)
        cdict = {}
        # GLASS model
        if isinstance(args, str) and args.endswith('.state'):
            bases = (GLASSModel, ModelArray, object)
            cls = mcls.__class__.__new__(mcls, mcls.__name__, bases, cdict)
            return cls.from_filename(args, **kwargs)
        elif isinstance(args, object) and hasattr(args, 'models') and 'obj,data' in args.models[0]:
            bases = (GLASSModel, ModelArray, object)
        elif isinstance(args, dict) and 'obj,data' in args:
            NotImplemented
        # PixeLens model
        elif isinstance(args, str) and args.endswith('.dat'):
            bases = (PixeLensModel, ModelArray, object)
        else:
            NotImplemented
        cls = super(LensModel, mcls).__new__(mcls, mcls.__name__, bases, cdict)
        instance = cls(args, **kwargs)
        return instance


###############################################################################
def cosmology(Omega_m, Omega_l, Omega_k=0, Omega_r=None):
    cosmo = {}
    cosmo['Omega_m'] = Omega_m
    cosmo['Omega_l'] = Omega_l
    cosmo['Omega_k'] = Omega_k
    if Omega_r is None:
        Omega_r = (1 + Omega_k) - Omega_m - Omega_l
    cosmo['Omega_r'] = Omega_r
    return cosmo


def Dcomov(r, a, cosmo=None):
    """
    Calculate the comoving distance from scale factor, for odeint

    Args:
        r <float> - distance
    """
    if cosmo is None:
        Omega_l = 0.72
        Omega_r = 4.8e-5
        Omega_k = 0
        Omega_m = 1. - Omega_r - Omega_l - Omega_k
    else:
        Omega_l = cosmo['Omega_l']
        Omega_r = cosmo['Omega_r']
        Omega_k = cosmo['Omega_k']
        Omega_m = cosmo['Omega_m']
    H = (Omega_m/a**3 + Omega_r/a**4 + Omega_l + Omega_k/a**2)**.5
    return 1./(a*a*H)


def DL(zl, zs, cosmo=None):
    """
    Distance D_L (for scaling kpc distances)

    Args:
        zl <float> - lens redshift
        zs <float> - source redshift

    Kwargs:
        None

    Return:
        Dl <float> - distance D_L

    Note:
        - only if wrong or no redshift has been used
        - * DL(zl_actual,zs_actual)/DL(zl_used,zs_used)
    """
    global Dcomov
    comov = partial(Dcomov, cosmo=cosmo)
    alens = 1./(1+zl)
    asrc = 1./(1+zs)
    a = [asrc, alens, 1]
    r = odeint(comov, [0], a)[:, 0]
    Dl = alens * (r[2] - r[1])
    return Dl


def DLDSDLS(zl, zs, cosmo=None):
    """
    Distance D_L*D_S/D_LS (for scaling masses)

    Args:
        zl <float> - lens redshift
        zs <float> - source redshift

    Kwargs:
        None

    Return:
        DlDsDls <float> - distance D_L*D_S/D_LS

    Note:
        - only if wrong redshift has been used
        - * DLDSDLS(zl_actual,zs_actual)/DLDSDLS(zl_used,zs_used)
    """
    global Dcomov
    comov = partial(Dcomov, cosmo=cosmo)
    alens = 1./(1+zl)
    asrc = 1./(1+zs)
    a = [asrc, alens, 1]
    r = odeint(comov, [0], a)[:, 0]
    Dl = alens * (r[2] - r[1])
    Ds = asrc * r[2]
    Dls = asrc * r[1]
    return Dl*Ds/Dls


def DLSDS(zl, zs, cosmo=None):
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
    global Dcomov
    comov = partial(Dcomov, cosmo=cosmo)
    alens = 1./(1+zl)
    asrc = 1./(1+zs)
    a = [asrc, alens, 1]
    r = odeint(comov, [0], a)[:, 0]
    Ds = asrc * r[2]
    Dls = asrc * r[1]
    return Dls/Ds


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
    encavg = [np.sum(data[r < ri])/len(data[r < ri])
              if len(data[r < ri]) else dta_cent for ri in rbins]
    return encavg


def kappa_profile(model, obj_index=0, correct_distances=True, maprad=None, pixrad=None):
    """
    Calculate radial kappa profiles for GLASS models or (for other models) by
    simply radially binning a generally kappa grid

    Args:
        model <GLASS model dict/np.ndarray> - GLASS model dictionary or some other kappa grid model

    Kwargs:
        obj_index <int> - object index of the GLASS object within the model
        correct_distances <bool> - correct with distance ratios and redshifts
        maprad <float> - map radius or physical scale of the profile

    Return:
        radii, profile <np.ndarrays> - radii and profiles ready to be plotted
    """
    if isinstance(model, dict) and 'obj,data' in model:  # a glass model
        obj, dta = model['obj,data'][obj_index]
        dlsds = DLSDS(obj.z, obj.sources[obj_index].z) if correct_distances else 1
        pixrad = obj.basis.pixrad
        maprad = obj.basis.top_level_cell_size * obj.basis.pixrad
        kappa_grid = dlsds * obj.basis._to_grid(dta['kappa'], 1)
    else:  # otherwise assume kappa grid was directly inputted
        kappa_grid = model
        maprad = 1 if maprad is None else maprad
        pixrad = kappa_grid.shape[0] if pixrad is None else pixrad
    profile = radial_profile(kappa_grid, bins=pixrad)
    radii = np.linspace(0, maprad, len(profile))
    return radii, profile


def dispersion_profile(model, obj_index=0, units=None):
    """
    Calculate velocity dispersion profiles for GLASS models or (for other models)

    Args:
        model <GLASS model dict/np.ndarray> - GLASS model dictionary or some other model data
                                              with shape(2, N) with M(<R) of length N at index 0
                                              and R data of length N at index 1

    Kwargs:
        obj_index <int> - object index of the GLASS object within the model
        units <dict> - dictionary of units for M=mass and R=length of the input data

    Return:
        radii <np.ndarray> - profile radii in light-seconds
        disp <np.ndarray> - dispersion profile in light units
    """
    units = {'mass': 'Msun', 'M': 'Msun',
             'R': 'kpc', 'length': 'kpc', 'L': 'kpc'} if units is None else units
    if isinstance(model, dict) and 'obj,data' in model:  # a glass model
        obj, dta = model['obj,data'][obj_index]
        MltR = dta['M(<R)'][units['M']]
        R = dta['R'][units['R']]
    else:
        MltR = model[0, :]
        R = model[1, :]
    if units['M'] == 'Msun' and units['R'] == 'kpc':
        MltR = MltR*gu.M_sol/gu.marb*gu.tick  # convert M_sol to seconds
        R = R*1e3*gu.parsec                   # convert to seconds
        disp = np.sqrt(2./3/gu.pi*MltR/R)     # dispersion in light units
    elif units['M'] == 'Msun' and units['R'] == 'arcsec':
        NotImplemented
    elif units['M'] == 'kg' and units['R'] == 'arcsec':
        NotImplemented
    elif units['M'] == 'kg' and units['R'] == 'kpc':
        NotImplemented
    elif units['M'] == 'kg' and units['R'] == 'km':
        NotImplemented
    elif units['M'] == 'kg' and units['R'] == 'm':
        NotImplemented
    else:
        print("Units are not yet convertable...")
        NotImplemented
    return np.asarray(R), np.asarray(disp)


def interpolate_profile(radii, profile, Nx=None):
    """
    Interpolate profile to increase resolution

    Args:
        radii <list/np.ndarray> - profile radii
        profile <list/np.ndarray> - kappa profi

    Kwargs:
        Nx <int> - new number of points along x-axis

    Return:
        r_ref, prof_ref <np.ndarrays> - refined arrays
    """
    if Nx is None:
        Nx = 5*len(radii)
    x, y = np.asarray(radii), np.asarray(profile)
    poly = interpolate.BPoly.from_derivatives(x, y[:, np.newaxis])
    newr = np.linspace(x.min(), x.max(), Nx)
    return newr, poly(newr)


def find_einstein_radius(radii, profile):
    """
    Find the Einstein radius of a kappa profile using the kappa=1 line

    Args:
        radii <list/np.ndarray> - profile radii
        profile <list/np.ndarray> - kappa profile

    Kwargs:
        None

    Return:
        einstein_radius <float> - Einstein radius of the lens profile
    """
    x, y = np.asarray(radii), np.asarray(profile)
    poly = interpolate.BPoly.from_derivatives(x, y[:, np.newaxis])
    x_min = np.min(x)
    x_max = np.max(x)
    midpoint = poly(x[len(x)//2])

    def kappaone(x): return poly(x) - 1

    rE, infodict, ierr, msg = optimize.fsolve(kappaone, midpoint, full_output=True)
    if (ierr == 1 or ierr == 5) and len(rE) == 1 and x_min < rE < x_max:
        einstein_radius = rE[0]
    elif len(rE) > 1:
        for r in rE:
            if x_min < r < x_max:
                einstein_radius = r
    else:
        einstein_radius = 0
    return einstein_radius


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

    Return:
        qpm <np.matrix object> - the quadrupole moment tensor

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
    phi = np.arctan2(sinphi, cosphi)
    if phi < 0:
        phi += np.pi
    return a, b, phi


def xy_grid(N, grid_size, a=None):
    """
    Coordinate grid of x and y given a particular pixel range and length

    Args:
        N <int> - number of grid points along the axes of the potential grid
        grid_size <float> - the length of the grid along the axes of the kappa grid

    Kwargs:
        a <float> - pixel scale of the corresponding kappa map

    Kwargs:
        None

    Return:
        xy <np.ndarray> - grid with x and y coordinates for corresponding 2D indices
    """
    N += (N % 2 == 0) and 1
    R = grid_size/2.
    pixel_size = grid_size/N if a is None else a
    x = np.linspace(-R+pixel_size/2., R-pixel_size/2., N)
    y = np.linspace(-R+pixel_size/2., R-pixel_size/2., N)
    gx, gy = np.meshgrid(x, y[::-1])
    return np.array([gx, gy])


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
    # a = x[0, -1] - x[0, -2]
    xylogx2y2 = x*y*(np.log(x2+y2) - 3)
    xarctanyx = x2*np.arctan(y/x)
    yarctanxy = y2*np.arctan(x/y)
    xylogx2y2 = np.nan_to_num(xylogx2y2)
    xarctanyx = np.nan_to_num(xarctanyx)
    yarctanxy = np.nan_to_num(yarctanxy)
    return xylogx2y2 + xarctanyx + yarctanxy


def lnr(x, y, xn, yn, a):
    """
    Potential ln(r) contribution of the n-th pixel as a sum of its potentials at the corners

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


def potential_grid(kappa, N, grid_size, factor=1, ext=None, verbose=False):
    """
    The entire potential map

    Args:
        kappa <np.ndarray> - the kappa grid
        N <int> - number of grid points along the axes of the potential grid
        grid_size <float> - the length of the grid along the axes of the kappa grid

    Kwargs:
        factor <float> - arbitrary scaling factor (e.g. redshift correction for kappa map)
        ext <list/np.ndarray> - external potential maps to be added
        verbose <bool> - verbose mode; print command line statements

    Return:
        gx, gy, psi <np.ndarray> - the x and y grid coordinates and the potential grid
    """
    gx, gy = xy_grid(N, grid_size)
    N = gx.shape[-1]
    pixel_size = (gx[0, 1] - gx[0, 0])*float(N)/kappa.shape[0]
    R = gx[0, -1] + 0.5*(gx[0, 1] - gx[0, 0])

    psi = np.zeros((N, N))
    xkappa = np.linspace(-R+pixel_size/2., R-pixel_size/2., kappa.shape[0])
    ykappa = xkappa[::-1]
    for m, ym in enumerate(ykappa):
        for n, xn in enumerate(xkappa):
            psi += kappa[m, n]*lnr(gx, gy, xn, ym, pixel_size)

    psi *= -1./(2*np.pi)
    psi *= factor
    # external terms
    if ext is not None:
        for e in ext:
            if e.shape != psi.shape:
                break
            psi -= e
    return gx, gy, psi


def shear_grid(gamma, N, grid_size, a=None):
    """
    Shear potential map

    Args:
        gamma <tuple/list/np.ndarray> - two-component shear parameter
        N <int> - number of grid points along the axes of the potential grid
        grid_size <float> - the length of the grid along the axes of the kappa grid
        a <float> - pixel scale of the corresponding kappa map

    Kwargs:
        None

    Return:
        gx, gy, ext_pot <np.ndarray> - the x and y grid coordinates and the shear potential grid
    """
    gamma = np.asarray(gamma)
    xy = xy_grid(N, grid_size)
    n0 = (xy[0]**2 - xy[1]**2)/2
    n1 = np.prod(xy, axis=0)
    N = np.asarray([n0, n1])
    p = np.sum(gamma*N.T, axis=-1).T
    return xy[0], xy[1], p


def arrival_grid(kappa, N, grid_size, beta, factor=1, ext=None, verbose=False):
    """
    The entire arrival-time surface map

    Args:
        kappa <np.ndarray> - the kappa grid
        N <int> - number of grid points along the axes of the potential grid
        grid_size <float> - the length of the grid along the axes of the kappa grid
        beta <list/np.ndarray> - source position

    Kwargs:
        factor <float> - arbitrary scaling factor (e.g. redshift correction for kappa map)
        ext <list/np.ndarray> - external potential maps to be added
        verbose <bool> - verbose mode; print command line statements

    Return:
        gx, gy, psi <np.ndarray> - the x and y grid coordinates and the arrival-time grid
    """
    gx, gy, psi = potential_grid(kappa, N, grid_size, factor=factor, ext=ext, verbose=verbose)
    xy = np.vectorize(complex)(gx, gy)
    beta = np.vectorize(complex)(beta[0], beta[1])
    return gx, gy, 0.5*abs(xy - beta)**2*factor + psi


def roche_potential_grid(*args, **kwargs):
    """
    The Roche potential map (degenerate arrival-time surface without source shift): theta^2/2 - psi
    from a kappa map

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


# def roche_potential(model, obj_index=0, N=85, src_index=0,
#                     correct_distances=True, verbose=False):
#     """
#     The Roche potential (degenerate arrival time surface without source shift): theta^2/2 - psi
#     from a GLASS model
#     Args:
#         model <GLASS model dict/np.ndarray> - GLASS model dictionary
#     Kwargs:
#         obj_index <int> - object index of the GLASS object within the model
#         N <int> - number of grid points along the axes of the potential grid
#         correct_distances <bool> - correct with distance ratios and redshifts
#         verbose <bool> - verbose mode; print command line statements
#     Return:
#         gx, gy, psi <np.ndarray> - the x and y grid coordinates and the potential grid
#     """
#     obj, dta = model['obj,data'][obj_index]
#     dlsds = DLSDS(obj.z, obj.sources[src_index].z) if correct_distances else 1
#     maprad = obj.basis.top_level_cell_size * obj.basis.pixrad
#     kappa = obj.basis._to_grid(dta['kappa'], 1)
#     kappa = dlsds * kappa
#     grid_size = 2 * maprad
#     gx, gy, degarr = roche_potential_grid(kappa, N, grid_size, verbose=verbose)
#     return gx, gy, degarr
