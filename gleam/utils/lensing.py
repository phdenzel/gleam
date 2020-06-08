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
from scipy import ndimage
from astropy.io import fits
from gleam.utils.linalg import eigvals, eigvecs, angle_between
from gleam.utils.units import units as gu
from gleam.glass_interface import glass_basis, glass_renv, _detect_cpus, _detect_omp
glass = glass_renv()


###############################################################################
class ModelArray(object):
    """
    Model class which contains data and some grid information
    """
    __framework__ = 'array'

    def __init__(self, data, filename=None,
                 grid_size=None, pixrad=None, maprad=None, kappa=None,
                 obj_index=None, src_index=None,
                 minima=[], saddle_points=[], maxima=[],
                 betas=[], shears=[],
                 zl=None, zs=None,
                 cosmo={'Omega_m': 0.3, 'Omega_l': 0.7, 'Omega_k': 0, 'Omega_r': 0.0},
                 verbose=False):
        """
        Args:
            data <np.ndarray> - ensemble data

        Kwargs:
            filename <str> - filename containing the lens data
            pixrad <int> - pixel radius of the grid
            maprad <float> - map radius of the grid in arcsec
            kappa <float> - kappa conversion factor in Msol/arcsec^2
            obj_index <int> - only for GLASSModel instances
            src_index <int> - only for GLASSModel instances
        """
        self.filepath = filename
        self.filename = os.path.basename(filename) if filename else filename
        self.data = np.asarray(data) if np.any(data) else np.ones((3, 3, 1))
        if grid_size is not None:
            self.grid_size = grid_size
        if pixrad is not None:
            self.pixrad = int(pixrad) if pixrad else self.pixrad
        self.maprad = maprad
        self.kappa = kappa
        self.minima = minima
        self.saddle_points = saddle_points
        self.maxima = maxima
        self.zl = zl
        self.zs = zs
        self.cosmo = cosmo
        # Convergence data
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
        # Shear
        if not hasattr(self, 'shears'):
            if shears:
                self.shears = shears
            else:
                self.shears = None
        # Source position
        if not hasattr(self, 'betas'):
            if betas:
                self.betas = betas
            else:
                self.betas = [np.array([0, 0])]*self.N
        if verbose:
            print(self.__v__)

    @classmethod
    def from_fitsfile(cls, filename, **kwargs):
        with open(filename) as f:
            hdu = fits.open(f, memmap=False)
            dta, hdr = hdu[0].data, hdu[0].header
        hdu.close()
        pixrad = dta.shape[-1]//2
        maprad = pixrad * hdr['CD2_2']*3600
        kwargs.update({'filename': filename, 'pixrad': pixrad, 'maprad': maprad})
        return cls(dta, **kwargs)

    @classmethod
    def from_fitsfiles(cls, filenames, **kwargs):
        data = None
        pixrad = None
        maprad = None
        index = 0
        for filename in filenames:
            with open(filename) as f:
                hdu = fits.open(f, memmap=False)
                dta, hdr = hdu[0].data, hdu[0].header
            hdu.close()
            if np.sum(dta) == 0:
                if data is not None:
                    data = data[:-1]
                continue
            if pixrad is None:
                pixrad = dta.shape[-1]//2
            if dta.shape[-1]//2 != pixrad:
                if data is not None:
                    data = data[:-1]
                continue
            if maprad is None:
                maprad = pixrad * hdr['CD2_2']*3600
            if (pixrad * hdr['CD2_2']*3600) != maprad:
                if data is not None:
                    data = data[:-1]
                continue
            if data is None:
                data = np.empty((len(filenames),) + dta.shape)
            data[index] = dta
            index += 1
        kwargs.update({'pixrad': pixrad, 'maprad': maprad})
        kwargs.setdefault('filename', 'lensmodels.fits')
        return cls(data, **kwargs)

    def subset(self, indices=None, mask=None, **kwargs):
        kw = [('filepath', 'filename'),
              ('grid_size', 'grid_size'), ('pixrad', 'pixrad'), ('maprad', 'maprad'),
              ('kappa', 'kappa'), ('_obj_idx', 'obj_index'), ('_src_idx', 'src_index'),
              ('minima', 'minima'), ('saddle_points', 'saddle_points'), ('maxima', 'maxima'),
              ('zl', 'zl'), ('zs', 'zs'), ('cosmo', 'cosmo')]
        for key in kw:
            if hasattr(self, key[0]):
                v = self.__getattribute__(key[0])
                kwargs.setdefault(key[1], v)
        if mask is not None:
            data = self.data[mask]
        elif indices is not None:
            data = np.select(self.data, indices)
        else:
            data = self.data
        return self.__class__(data, **kwargs)

    def save(self, savename=None, **kwargs):
        import cPickle as pickle
        kw = [('filepath', 'filename'),
              ('grid_size', 'grid_size'), ('pixrad', 'pixrad'), ('maprad', 'maprad'),
              ('kappa', 'kappa'), ('_obj_idx', 'obj_index'), ('_src_idx', 'src_index'),
              ('minima', 'minima'), ('saddle_points', 'saddle_points'), ('maxima', 'maxima'),
              ('_zl', 'zl'), ('_zs', 'zs')]
        for key in kw:
            if hasattr(self, key[0]):
                v = self.__getattribute__(key[0])
                kwargs.setdefault(key[1], v)
        if savename is not None:
            with open(savename, 'wb') as f:
                pickle.dump((self.data.copy(), kwargs), f)
        else:
            return self.data.copy(), kwargs        

    def __str__(self):
        return "<LensModel@{}>".format("".join(self.filename.split('.')[:-1]))

    @property
    def tests(self):
        return ['filename', 'filepath', 'N', 'pixrad', 'maprad', 'pixel_size', 'kappa',
                'minima', 'saddle_points', 'maxima', 'zl', 'zs']

    @property
    def __v__(self):
        return "\n".join([t.ljust(20)+"\t{}".format(self.__getattribute__(t))
                          for t in self.tests])

    def __getitem__(self, index):
        return self.models[index]

    @property
    def obj_name(self):
        return '.'.join(self.filename.split('.')[:-1])

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
    def ensemble_average(self):
        if len(self.data.shape) > 2:
            return np.average(self.data, axis=0)
        else:
            return self.data

    @property
    def extent(self):
        if self.maprad:
            return [-1*self.maprad, self.maprad, -1*self.maprad, self.maprad]
        else:
            return np.array([-1, 1, -1, 1])

    @property
    def grid_size(self):
        return 2*self.pixrad + 1

    @grid_size.setter
    def grid_size(self, grid_size):
        if grid_size:
            self.pixrad = int(grid_size//2)

    @property
    def pixel_size(self):
        return 2*self.maprad / self.data.shape[-1]

    @property
    def minima(self):
        if not hasattr(self, '_minima'):
            self._minima = []
        return np.asarray(self._minima)

    @minima.setter
    def minima(self, minima):
        self._minima = minima

    @property
    def saddle_points(self):
        if not hasattr(self, '_saddle_points'):
            self._saddle_points = []
        return np.asarray(self._saddle_points)

    @saddle_points.setter
    def saddle_points(self, saddle_points):
        self._saddle_points = saddle_points

    @property
    def maxima(self):
        if not hasattr(self, '_maxima'):
            self._maxima = []
        return np.asarray(self._maxima)

    @maxima.setter
    def maxima(self, maxima):
        self._maxima = maxima

    @property
    def cosmo(self):
        if not hasattr(self, '_cosmo'):
            self._cosmo = {'Omega_m': 0.3, 'Omega_l': 0.7,
                           'Omega_k': 0, 'Omega_r': 0.0}
        return self._cosmo

    @cosmo.setter
    def cosmo(self, cosmo):
        self._cosmo = cosmo
        if not (self.zl is None or self.zs is None):
            self._dlsds = DLSDS(self.zl, self.zs, cosmo=self._cosmo)

    @property
    def zl(self):
        if not hasattr(self, '_zl'):
            self._zl = None
        return self._zl

    @zl.setter
    def zl(self, zl):
        self._zl = zl
        if not (self.zs is None):
            self._dlsds = DLSDS(self._zl, self.zs, cosmo=self.cosmo)

    @property
    def zs(self):
        if not hasattr(self, '_zs'):
            self._zs = None
        return self._zs

    @zs.setter
    def zs(self, zs):
        self._zs = zs
        if not self.zl is None:
            self._dlsds = DLSDS(self.zl, self._zs, cosmo=self.cosmo)

    @property
    def dlsds(self):
        if not hasattr(self, '_dlsds'):
            if not (self.zl is None and self.zs is None):
                self._dlsds = DLSDS(self.zl, self.zs, cosmo=self.cosmo)
            else:
                self._dlsds = 1
        return self._dlsds

    @dlsds.setter
    def dlsds(self, dlsds):
        self._dslds = dlsds

    @property
    def obj_idx(self):
        if not hasattr(self, '_obj_idx'):
            self._obj_idx = 0
        return self._obj_idx

    @obj_idx.setter
    def obj_idx(self, obj_index):
        self._obj_idx = obj_index

    @property
    def src_idx(self):
        return self._src_idx

    @src_idx.setter
    def src_idx(self, src_index):
        self._src_idx = src_index

    @property
    def N_obj(self):
        return 1

    @property
    def N_src(self):
        return 1

    def rescale(self, zl_new, zs_new, zl=None, zs=None, cosmo=None):
        """
        Rescale the lens model to new lens and source redshifts
        """
        if cosmo is None:
            cosmo = cosmology()
        if zl is None:
            zl = self.zl
        if zs is None:
            zs = self.zs
        dldsdls = DLDSDLS(zl, zs)
        dldsdls_new = DLDSDLS(zl_new, zs_new)
        sigma = dldsdls / dldsdls_new
        # dl = DL(zl, zs)
        # dl_new = DL(zl_new, zs_new)
        # delta = dl_new / dl
        self.data = self.data * sigma
        self.models = [m * sigma for m in self.models]
        self.kappa = self.kappa * sigma if self.kappa is not None else None
        self.zl = zl_new
        self.zs = zs_new
        # self.maprad = self.maprad * delta if self.maprad is not None else None
        # self.minima = [[m[0]*delta, m[1]*delta] for m in self.minima]
        # self.saddle_points = [[m[0]*delta, m[1]*delta] for m in self.saddle_points]
        # self.maxima = [[m[0]*delta, m[1]*delta] for m in self.maxima]
        # self.shears  #  rescale like kappa?
        # self.betas = [[b[0]*delta, b[1]*delta] for b in self.betas]

    def rotate(self, angle, data_attr='data', create_instance=True,
               **kwargs):
        """
        Rotate map data by angle(s)
        """
        d = self.__getattribute__(data_attr).copy()
        if isinstance(d, dict):
            d = d['data'].copy()
        if isinstance(angle, (int, float)):
            angle = np.array([angle]*self.N)
        kwargs.setdefault('pixrad', self.pixrad)
        kwargs.setdefault('maprad', self.maprad)
        rotation = np.vectorize(partial(ndimage.rotate, reshape=False),
                                signature='(m,n),()->(m,n)')
        self.rotated = {}
        self.rotated['data'] = rotation(d, angle)
        self.rotated['angle'] = angle.copy()
        self.rotated.update(kwargs)
        if create_instance:
            obj = self.__class__(self.rotated['data'], **kwargs)
            self.rotated['obj'] = obj
            return obj
        

    def resample(self, pixrad, data_attr='data', create_instance=True,
                 **kwargs):
        """
        Resample map data to a lower pixrad
        """
        d = self.__getattribute__(data_attr).copy()
        if isinstance(d, dict):
            d = d['data'].copy()
        kwargs.setdefault('pixrad', pixrad)
        kwargs.setdefault('maprad', self.maprad)
        kwargs.setdefault('filename', self.filepath)
        kwargs.setdefault('zl', None)
        kwargs.setdefault('zs', None)
        zoom_factor = (2 * pixrad + 1.)/(2 * self.pixrad + 1)
        self.resampled = {}
        self.resampled['data'] = ndimage.interpolation.zoom(d, [1, zoom_factor, zoom_factor],
                                                            order=0)
        self.resampled.update(kwargs)
        if create_instance:
            obj = self.__class__(self.resampled['data'], **kwargs)
            self.resampled['obj'] = obj
            return obj

    def xy_grid(self, N=None, maprad=None, pixel_size=None, as_complex=False, refined=True):
        N = self.kappa_grid(refined=refined).shape[-1] if N is None else N
        maprad = self.maprad if maprad is None else maprad
        pixel_size = self.pixel_size if pixel_size is None else pixel_size
        xy = xy_grid(N, 2*maprad, pixel_size)
        if as_complex:
            xy_compl = np.empty((N, N), dtype=np.complex)
            xy_compl.real = xy[0, ...]
            xy_compl.imag = xy[1, ...]
            return xy_compl
        return xy

    def cellsize_grid(self, N=None, maprad=None, pixel_size=None, refined=True):
        N = self.kappa_grid(refined=refined).shape[-1] if N is None else N
        maprad = self.maprad if maprad is None else maprad
        cell_size = 2*maprad / N
        cell_sizes = np.ones((N, N)) * cell_size
        return cell_sizes

    def kappa_grid(self, model_index=-1, refined=True):
        if refined and hasattr(self, 'data_hires'):
            return self.data_hires[model_index] if model_index >= 0 \
                else self.ensemble_average
        elif refined:
            return self[model_index] if model_index >= 0 \
                else self.ensemble_average
        elif hasattr(self, 'data_toplevel'):
            return self.data_toplevel[model_index] if model_index >= 0 \
                else np.average(self.data_toplevel, axis=0)
        return self[model_index] if model_index >= 0 else self.ensemble_average

    def sigma_grid(self, model_index=-1):
        return self.kappa*self.kappa_grid(model_index=model_index)

    def potential_grid(self, model_index=-1, N=None, factor=1, ext=None):
        model = self.kappa_grid(model_index)
        N = model.shape[-1] if N is None else N
        factor = 1./self.dlsds if hasattr(self, 'dlsds') else 1
        no_ext = ext is None
        ext = [self.shear_grid(model_index, N)] if no_ext else ext
        ext += [g for g in self.extmass_grid(model_index, N)] if no_ext else []
        _, _, grid = potential_grid(model, N, 2*self.maprad, factor=factor, ext=ext)
        return grid

    def shear_grid(self, model_index=-1, N=None):
        N = self.data.shape[-1] if N is None else N
        nil = np.array([0, 0])
        if model_index >= -1:
            shear = self.shears[model_index] if hasattr(self, 'shears') and np.any(self.shears) \
                    else nil
        else:
            shear = nil
        _, _, grid = shear_grid(shear, N, 2*self.maprad)
        return grid

    def extmass_grid(self, model_index=-1, N=None):
        N = self.data.shape[-1] if N is None else N
        nil = [[np.array([1, 1]), 0]]
        if model_index >= -1:
            extm = self.ptmasses[model_index] if hasattr(self, 'ptmasses') and len(self.ptmasses[model_index]) > 0 \
                else nil
        _, _, grid = extmass_grid(extm, N, 2*self.maprad)
        return grid

    def arrival_grid(self, model_index=-1, N=None, beta=None, factor=None, ext=None):
        model = self.kappa_grid(model_index)
        N = model.shape[-1] if N is None else N
        nil = np.array([0, 0])
        if model_index >= -1:
            beta = self.betas[model_index] if hasattr(self, 'betas') else nil
        else:
            beta = nil
        factor = 1./self.dlsds if factor is None and hasattr(self, 'dlsds') else 1
        no_ext = ext is None
        ext = [self.shear_grid(model_index, N)] if no_ext else ext
        ext += [g for g in self.extmass_grid(model_index, N)] if no_ext else []
        _, _, grid = arrival_grid(model, N, 2*self.maprad, beta, factor=factor, ext=ext)
        return grid

    def roche_potential_grid(self, model_index=-1, N=None, factor=1, ext=None):
        model = self.kappa_grid(model_index)
        N = self.data.shape[-1] if N is None else N
        # factor = 1./self.dlsds if hasattr(self, 'dlsds') else 1
        no_ext = ext is None
        ext = [self.shear_grid(model_index, N)] if no_ext else ext
        ext += [g for g in self.extmass_grid(model_index, N)] if no_ext else []
        _, _, grid = roche_potential_grid(model, N, 2*self.maprad, factor=factor, ext=ext)
        return grid

    def saddle_contour_levels(self, model_index=-1, saddle_points=None, N=None, maprad=None,
                              factor=None, shear=None, extm=None):
        model = self.kappa_grid(model_index)
        N = model.shape[-1] if N is None else N
        maprad = self.maprad if maprad is None else maprad
        L = 2*maprad
        saddle_points = self.saddle_points if saddle_points is None else saddle_points
        beta = self.betas[model_index] if hasattr(self, 'betas') else np.array([0, 0])
        factor = 1./self.dlsds if factor is None and hasattr(self, 'dlsds') else 1
        shear = self.shears[model_index] \
            if shear is None and hasattr(self, 'shears') and np.any(self.shears) \
            else shear
        extm = self.ptmasses[model_index] \
            if extm is None and hasattr(self, 'ptmasses') and len(self.ptmasses[model_index]) > 0 \
            else extm
        kwargs = dict(factor=factor, shear=shear, extm=extm)
        return sorted([arrival_time(sad, beta, model, N, L, **kwargs) for sad in saddle_points])


class PixeLensStateList(type):

    __propagate__ = ['append']

    def __new__(mcls, *args, **cdict):
        name = mcls.__name__
        bases = (list, object, )
        cdict.update(list.__dict__)
        cls = super(PixeLensStateList, mcls).__new__(mcls, name, bases, cdict)
        cls.__init__ = mcls.assertion_init
        for fn in mcls.__propagate__:
            if fn in mcls.__dict__:
                func = mcls.__dict__[fn]
            else:
                continue
            setattr(cls, func.__name__, func)
        return cls

    def __init__(self, *args, **kwargs):
        list.__init__([])
        self._type = args[0]

    @classmethod
    def propagate(cls, func):
        print(cls)

    @staticmethod
    def assertion_init(self, *args, **kwargs):
        for a in args:
            print(type(a))
            if hasattr(a, '__len__'):
                element = []
                for ai in a:
                    try:
                        element.append(self._type(ai))
                    except ValueError:
                        continue
            elif isinstance(a, str): #and self._type != str:
                print("Split str")
            else:
                try:
                    element = self._type(a)
                except ValueError:
                    continue
            self.append(element)

    # @PixeLensStateList.propagate
    # def append(self, *args):
    #     print("CUSTOM APPEND")
    #     print(self)


class PixeLensEnsem(PixeLensStateList):
    pass


class PixeLensModelState(PixeLensStateList):
    pass


class PixeLensModel(ModelArray):
    """
    Model class which loads PixeLens .state, .dat, or .txt files
    """
    __framework__ = 'PixeLens'

    dat_fields = {
        'sigcrit': 'kappa',
        'x range': 'maprad',
        'y range': 'maprad',
        'grid size': 'grid_size'}

    def __init__(self, filename, **kwargs):
        with open(os.path.abspath(filename), 'rb') as f:
            self.content = f.readlines()
        if filename.endswith('state.txt'):
            state = self.parse_state(self.content)
            data, hdr_info = self.state2data(state)
        elif filename.endswith('.dat'):
            hdr_info, hdr_offset = self.parse_header(self.content)
            data = self.parse_dat(self.content, offset=hdr_offset)
        super(PixeLensModel, self).__init__(data, filename=filename, **hdr_info)

    @staticmethod
    def parse_state(text, sections={'INPUT': PixeLensStateList(str),
                                    'PMAP': PixeLensStateList(str),
                                    'MODEL': PixeLensModelState(float),
                                    'ENSEM': 'MODEL'},
                    ):
        """
        Parse through a PixeLensModel text file

        Args:
            text <list(str)> - text file content

        Kwargs:
            sections <dict> - sections which to parse by

        Return:
            TODO
        """
        fields = {}
        for line in text:
            if line.startswith('#BEGIN'):
                key = line[7:].strip()
                parse_type = sections[key]
                if isinstance(parse_type, str) and parse_type in sections:
                    parse_type = sections[parse_type]
                fields[key] = parse_type()
                continue
            if line.startswith('#END'):
                end_key = line[4:].strip()
                if end_key in sections.values():
                    finalize_section = [k for k in sections if sections[k] == end_key][0]
                    fields[finalize_section].append(fields[end_key])
                # if end_key == 'MODEL':
                #     fields[end_key].copy()
                # if end_key == 'ENSEM':
                #     pass
                continue
            fields[key].append(line)
        print(fields["PMAP"])
        pass

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
            if label in PixeLensModel.dat_fields:
                hdr[PixeLensModel.dat_fields[label]] = value
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
            dta = {'hires': [], 'toplevel': [], 'dlsds': [],
                   'minima': [], 'saddle_points': [], 'maxima': [],
                   'zl': [], 'zs': [], 'H0': [], 'cosmology': cosmo}
            for obj_idx in range(self.N_obj):
                mdta = {'toplevel': [], 'hires': [], 'dlsds': [], 'H0': []}
                for m in self.env.models:
                    obj, data = m['obj,data'][obj_idx]
                    src = obj.sources[src_index]
                    zl, zs = obj.z, src.z
                    dlsds = DLSDS(zl, zs, cosmo=cosmo)
                    mdta['toplevel'].append(dlsds * obj.basis._to_grid(data['kappa'], 1))
                    mdta['hires'].append(dlsds * obj.basis.kappa_grid(data))
                    mdta['H0'].append(data['H0'])
                imgs = src.images
                mdta['dlsds'] = dlsds
                mdta['zl'] = zl
                mdta['zs'] = zs
                mdta['minima'] = [i._pos for i in imgs if i.parity == 0]
                mdta['saddle_points'] = [i._pos for i in imgs if i.parity == 1]
                mdta['maxima'] = [i._pos for i in imgs if i.parity == 2]
                for k in mdta:
                    dta[k].append(mdta[k])
            for k in ['toplevel', 'hires']:
                dta[k] = np.array(dta[k], dtype=np.float32)
            self.all_data = dta
        self.filepath = filename if filename is not None else ""
        self.filename = os.path.basename(filename) if filename is not None else ""
        self.src_idx = src_index
        self.obj_idx = obj_index  # calls super(GLASSModel, self)__init__()
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
    def obj_name(self):
        return self.gcm[0][0].name

    @property
    def data_hires(self):
        return self.all_data['hires'][self.obj_idx]

    @property
    def data_toplevel(self):
        return self.all_data['toplevel'][self.obj_idx]

    # @property
    # def dlsds(self):
    #     return self.all_data['dlsds'][self.obj_idx]

    @property
    def H0(self):
        return self.all_data['H0'][self.obj_idx]

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
        betas = [m[1]['src'] for m in self.gcm]
        if hasattr(self, 'env'):
            self.env.make_ensemble_average()
            betas.append(self.env.ensemble_average['obj,data'][self.obj_idx][1]['src'])
        betas = np.array(betas).view(np.float)
        return betas

    @property
    def shears(self):
        """
        All modeled shear components
        """
        shears = [m[1]['shear'] for m in self.gcm if 'shear' in m[1]]
        self.env.make_ensemble_average()
        ensdta = self.env.ensemble_average['obj,data'][self.obj_idx][1]
        if 'shear' in ensdta:
            shears.append(ensdta['shear'])
        shears = np.asarray(shears)
        return shears

    @property
    def N_ptmasses(self):
        """
        Number of point masses in the model
        """
        iptm = [i for i, ep in enumerate(self.gcm[0][0].extra_potentials) if ep.name == 'ptmass']
        return len(iptm)

    @property
    def ptmasses(self):
        """
        All modeled point mass components
        """
        has_ptmasses = 'ptmass' in self.gcm[0][1]
        iptm = [i for i, ep in enumerate(self.gcm[0][0].extra_potentials) if ep.name == 'ptmass']
        ptms = [[(np.array([m[0].extra_potentials[i].r.real, m[0].extra_potentials[i].r.imag]),
                 m[1]['ptmass'][n]) for n, i in enumerate(iptm)]
                for m in self.gcm if has_ptmasses]
        self.env.make_ensemble_average()
        ens = self.env.ensemble_average['obj,data'][self.obj_idx]
        ptms.append([(np.array([ens[0].extra_potentials[i].r.real,
                                ens[0].extra_potentials[i].r.imag]),
                      ens[1]['ptmass'][n])
                     for n, i in enumerate(iptm)])
        return ptms

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
        kappa = None
        if self.env.nu is not None:
            kappa = glass.scales.convert('kappa to Msun/arcsec^2', 1,
                                         obj.dL, self.env.nu[self.src_idx])
        minima = np.asarray(self.all_data['minima'][obj_index])
        saddle_points = np.asarray(self.all_data['saddle_points'][obj_index])
        maxima = np.asarray(self.all_data['maxima'][obj_index])
        zl = self.all_data['zl'][obj_index]
        zs = self.all_data['zs'][obj_index]
        cosmo = self.all_data['cosmology']
        super(GLASSModel, self).__init__(
            dta, filename=self.filepath, pixrad=obj.basis.pixrad, maprad=obj.basis.mapextent,
            kappa=kappa, minima=minima, maxima=maxima, saddle_points=saddle_points,
            zl=zl, zs=zs, cosmo=cosmo)

    @property
    def src_idx(self):
        if not hasattr(self, '_src_idx'):
            self._src_idx = 0
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
    (This is the only "*Model" class one needs to import from this module)

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
        if isinstance(args, str) and args.endswith('.fits'):
            bases = (ModelArray, object)
            cls = mcls.__class__.__new__(mcls, mcls.__name__, bases, cdict)
            return cls.from_fitsfile(args, **kwargs)
        elif isinstance(args, list) and isinstance(args[0], str) and args[0].endswith('.fits'):
            bases = (ModelArray, object)
            cls = mcls.__class__.__new__(mcls, mcls.__name__, bases, cdict)
            return cls.from_fitsfiles(args, **kwargs)
        # GLASS model
        if isinstance(args, str) and args.endswith('.state'):
            bases = (GLASSModel, ModelArray, object)
            cls = mcls.__class__.__new__(mcls, mcls.__name__, bases, cdict)
            return cls.from_filename(args, **kwargs)
        elif isinstance(args, glass.environment.Environment):
            bases = (GLASSModel, ModelArray, object)
        # elif isinstance(args, object) and hasattr(args, 'models') and 'obj,data' in args.models[0]:
        #     bases = (GLASSModel, ModelArray, object)
        elif isinstance(args, dict) and 'obj,data' in args:
            NotImplemented
        # PixeLens model
        elif isinstance(args, str) and args.endswith('.dat'):
            bases = (PixeLensModel, ModelArray, object)
        elif isinstance(args, str) and args.endswith('state.txt'):
            bases = (PixeLensModel, ModelArray, object)
        else:
            NotImplemented
        cls = super(LensModel, mcls).__new__(mcls, mcls.__name__, bases, cdict)
        instance = cls(args, **kwargs)
        return instance


###############################################################################
def cosmology(Omega_m=0.279952, Omega_l=0.72, Omega_k=0, Omega_r=None):
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


def downsample_model(kappa, extent, shape, pixel_scale=1., sanitize=True, verbose=False):
    """
    Resample a model's kappa grid (usually downsample to a coarser grid) to match
    the specified scale and size

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
    if sanitize:
        kappa_resmap[kappa_resmap < 0] = 0

    return kappa_resmap


def upsample_model(kappa, extent, shape, pixel_scale=1., sanitize=True, verbose=False):
    """
    Resample a model's kappa grid (usually upsample to a finer grid) to match
    the specified scales and size

    Args:
        gls_model <glass.LensModel object> - GLASS ensemble model
        extent <tuple/list> - extent of the output
        shape <tuple/list> - shape of the output

    Kwargs:
        verbose <bool> - verbose mode; print command line statements

    Return:
        kappa_resmap <np.ndarray> - resampled kappa grid
    """
    # obj, data = gls_model['obj,data'][0]
    # kappa_map = obj.basis._to_grid(data['kappa'], 1)
    # pixrad = obj.basis.pixrad
    # maprad = obj.basis.top_level_cell_size * (obj.basis.pixrad)
    # mapextent = (-obj.basis.top_level_cell_size * (obj.basis.pixrad+0.5),
    #              obj.basis.top_level_cell_size * (obj.basis.pixrad+0.5))
    # cell_size = obj.basis.top_level_cell_size
    pixrad = tuple(r//2 for r in kappa.shape)
    maprad = pixrad[1]*pixel_scale

    if verbose:
        print("Kappa map: {}".format(kappa.shape))
        print("Pixrad {}".format(pixrad))
        print("Maprad {}".format(maprad))
        # print(obj)
        # print("Mapextent {}".format(mapextent))
        # print("Cellsize {}".format(cell_size))

    xmdl = np.linspace(-maprad, maprad, kappa.shape[0])
    ymdl = np.linspace(-maprad, maprad, kappa.shape[1])
    Xmdl, Ymdl = np.meshgrid(xmdl, ymdl)
    xnew = np.linspace(extent[0], extent[1], shape[0])
    ynew = np.linspace(extent[2], extent[3], shape[1])
    Xnew, Ynew = np.meshgrid(xnew, ynew)

    # rescale = interpolate.Rbf(Xmdl, Ymdl, kappa)
    rescale = interpolate.interp2d(xmdl, ymdl, kappa)
    kappa_resmap = rescale(xnew, ynew)
    if sanitize:
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


def kappa_profile(model, obj_index=0, mdl_index=-1, maprad=None, pixrad=None, refined=True, factor=1):
    """
    Calculate radial kappa profiles for GLASS models or (for other models) by
    simply radially binning a generally kappa grid

    Args:
        model <LensModel object/np.ndarray> - LensModel object or a raw kappa grid array

    Kwargs:
        obj_index <int> - object index for the LensModel object
        maprad <float> - map radius or physical scale of the profile
        pixrad <int> - pixel radius of the grid; used to estimate number of bins

    Return:
        radii, profile <np.ndarrays> - radii and profiles ready to be plotted
    """
    if isinstance(model, LensModel):
        model.obj_idx = min(obj_index, model.N_obj-1)
    elif isinstance(model, np.ndarray):  # kappa grid was directly input
        pixrad = model.shape[-1] if pixrad is None else pixrad
        maprad = 1 if maprad is None else maprad
        model = LensModel(model, maprad=maprad, pixrad=pixrad)
        kappa_grid = model.data
    else:
        model = LensModel(model)
    kappa_grid = model.kappa_grid(model_index=mdl_index, refined=refined)*factor
    maprad = model.maprad if maprad is None else maprad
    pixrad = model.pixrad if pixrad is None else pixrad
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


def qpm_abphi(qpm, verbose=False):
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
    cosphi = angle_between(evc[0], [1, 0])
    sinphi = angle_between(evc[0], [0, 1])
    if verbose:
        print("Eigen vectors: {}".format(evc))
    phi = np.arctan2(sinphi, cosphi)
    if phi < 0:
        phi += np.pi
    if phi > np.pi:
        phi -= np.pi
    return a, b, phi


def xy_grid(N, grid_size, a=None, as_complex=False):
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
    xy = np.array([gx, gy])
    if as_complex:
        xy_compl = np.empty((N, N), dtype=np.complex)
        xy_compl.real = xy[0, ...]
        xy_compl.imag = xy[1, ...]
        return xy_compl
    return xy


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


def grad(W, r0, r, a, threads=1, use_omp=False):
    """
    Args:
        W <np.ndarray> - kappa grid
        r0 <complex> - theta point of reference
        r <np.ndarray(complex)> - array of all pixel coordinates
        a <np.ndarray> - pixel size
    """
    import weave
    from numpy import pi
    code = """
    int i;
    std::complex<double> v(0,0);
    //Py_BEGIN_ALLOW_THREADS
    double xx=0,yy=0;
    #ifdef WITH_OMP
    omp_set_num_threads(threads);
    #endif
    #pragma omp parallel for reduction(+:xx) reduction(+:yy)
    for (i=0; i < l; i++)
    {
        double vx,vy;
        std::complex<double> dr = r0-r[i];
        const double xi = std::real(dr);
        const double yi = std::imag(dr);
        const double xm = xi - a[i]/2;
        const double xp = xi + a[i]/2;
        const double ym = yi - a[i]/2;
        const double yp = yi + a[i]/2;
        const double xm2 = xm*xm;
        const double xp2 = xp*xp;
        const double ym2 = ym*ym;
        const double yp2 = yp*yp;
        const double log_xm2_ym2 = log(xm2 + ym2);
        const double log_xp2_yp2 = log(xp2 + yp2);
        const double log_xp2_ym2 = log(xp2 + ym2);
        const double log_xm2_yp2 = log(xm2 + yp2);
        vx = (xm*atan(ym/xm) + xp*atan(yp/xp)) + (ym*log_xm2_ym2 + yp*log_xp2_yp2) / 2
           - (xm*atan(yp/xm) + xp*atan(ym/xp)) - (ym*log_xp2_ym2 + yp*log_xm2_yp2) / 2;
        vx /= pi;
        vx *= W[i];
        vy = (ym*atan(xm/ym) + yp*atan(xp/yp)) + (xm*log_xm2_ym2 + xp*log_xp2_yp2) / 2
           - (ym*atan(xp/ym) + yp*atan(xm/yp)) - (xm*log_xm2_yp2 + xp*log_xp2_ym2) / 2;
        vy /= pi;
        vy *= W[i];
        //v += std::complex<double>(vx,vy);
        xx += vx;
        yy += vy;
    }
    //Py_END_ALLOW_THREADS
    return_val = std::complex<double>(xx,yy);
    """
    l = len(r)
    if use_omp:
        kw = _detect_omp(force_gcc=True)
    else:
        kw = {}
    # threads = _detect_cpus()
    v = weave.inline(code, ['l', 'W','r0','r', 'pi', 'a', 'threads'], **kw)
    return v


def external_shear_grad(shear, theta):
    """
    Returns potential gradient of external shear (as complex two-component point)
    """
    shear = np.asarray(shear)
    theta = np.asarray(theta)
    sheargrad = theta.copy()
    sheargrad.real = shear[0]*theta.real + shear[1]*theta.imag
    sheargrad.imag = shear[1]*theta.real - shear[0]*theta.imag
    return sheargrad


def external_ptmass_grad(extmass, theta):
    """
    Returns potential gradient of external mass (point mass)
    """
    # extmass = 2*extmass
    r, m = [complex(*e[0]) for e in extmass], [e[1] for e in extmass]
    r = np.asarray(r)
    m = np.asarray(m)
    theta = np.asarray(theta)
    d = theta[np.newaxis, ...] - r[..., np.newaxis]
    d2 = np.abs(d)**2
    gradptm = np.sum(m[..., np.newaxis]*d/d2, axis=0)
    return gradptm


def deflect(theta, kappa, ploc, cell_size):
    """
    Returns the deflection, i.e. potential gradient

    Args:
        theta <> - angular position
        kappa <> - kappa map
        ploc <> - pixel locations of the kappa map
        cell_size <> - array of pixel sizes
    """
    s = grad(kappa, theta, ploc, cell_size)
    return s


def arrival_time(theta, beta, kappa, N, grid_size, factor=1,
                 shear=None, extm=None, verbose=False):
    """
    Calculate the arrival time for a given kappa map and beta at position theta
    """
    geom = sum(abs(theta - beta)**2) / 2 * factor
    xn, yn = xy_grid(N, grid_size)
    pixel_size = (xn[0, 1] - xn[0, 0])*float(N)/kappa.shape[0]
    Q = lnr(theta[0], theta[1], xn, yn, pixel_size)
    pot = -np.sum(kappa*Q) / (2*np.pi) * factor
    if shear is not None:
        extshear = external_shear(shear, theta)
        pot -= extshear
    if extm is not None:
        extmass = external_ptmass(extm, theta)
        pot -= extmass
    return geom + pot


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


def external_shear(shear, theta):
    """
    Shear value at theta

    Args:
        shear <tuple/list/np.ndarray> - two-component shear parameter value(s)
        theta <tuple/list/np.ndarray> - angular position on the lens plane, i.e. theta

    Kwargs:
        None

    Return:
        shear <np.ndarray> - potential contribution of two-component shear at theta
    """
    shear = np.asarray(shear)
    x, y = theta
    n0 = (x**2 - y**2)/2
    n1 = x*y
    return np.sum(shear*np.array([n0, n1]).T)


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


def external_ptmass(extmass, theta):
    """
    External mass value at theta

    Args:
        extmass <list/tuple> - (r:np.ndarray, m:float) parameters of ext. masses
        theta <tuple/list/np.ndarray> - angular position on the lens plane, i.e. theta

    Kwargs:
        None
    """
    r, m = [e[0] for e in extmass], [e[1] for e in extmass]
    r = np.asarray(r)
    m = np.asarray(m)
    if isinstance(theta, complex):
        theta = np.array([theta.real, theta.imag])
    elif isinstance(theta, (list, tuple)):
        theta = np.asarray(theta)
    d = theta - r
    dx, dy = d[..., 0], d[..., 1]
    return m * np.log(np.sqrt(dx**2 + dy**2)) / np.pi


def extmass_grid(extmass, N, grid_size, a=None):
    """
    External mass potential map

    Args:
        extmass <list/tuple> - (r:np.ndarray, m:float) parameters of ext. mass
        N <int> - number of grid points along the axes of the potential grid
        grid_size <float> - the length of the grid along the axes of the kappa grid
    """
    r, m = [e[0] for e in extmass], [e[1] for e in extmass]
    r = np.asarray(r)
    m = np.asarray(m)
    xy = xy_grid(N, grid_size)
    d = np.stack(r.shape[0]*[xy])
    d = d - r[..., np.newaxis, np.newaxis]
    dists = np.log(np.sqrt(d[:, 0, ...]**2 + d[:, 1, ...]**2))
    extm_grids = m[..., np.newaxis, np.newaxis] * dists / np.pi
    return xy[0], xy[1], extm_grids


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


# TODO: remove when it is not needed anymore
def roche_potential(model, obj_index=0, N=85, src_index=0,
                    correct_distances=True, verbose=False):
    """
    The Roche potential (degenerate arrival time surface without source shift): theta^2/2 - psi
    from a GLASS model
    Args:
        model <GLASS model dict/np.ndarray> - GLASS model dictionary
    Kwargs:
        obj_index <int> - object index of the GLASS object within the model
        N <int> - number of grid points along the axes of the potential grid
        correct_distances <bool> - correct with distance ratios and redshifts
        verbose <bool> - verbose mode; print command line statements
    Return:
        gx, gy, psi <np.ndarray> - the x and y grid coordinates and the potential grid
    """
    obj, dta = model['obj,data'][obj_index]
    dlsds = DLSDS(obj.z, obj.sources[src_index].z) if correct_distances else 1
    maprad = obj.basis.top_level_cell_size * obj.basis.pixrad
    kappa = obj.basis._to_grid(dta['kappa'], 1)
    kappa = dlsds * kappa
    grid_size = 2 * maprad
    gx, gy, degarr = roche_potential_grid(kappa, N, grid_size, verbose=verbose)
    return gx, gy, degarr
