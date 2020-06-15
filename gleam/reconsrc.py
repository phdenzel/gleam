#!/usr/bin/env python
"""
@author: phdenzel

Climb every peak in search for lens and source candidates

TODO:
    - add a main method
    - complete tests
    - add checks for antialiasing in cache, if different recompute
    - derive more info to cache, N_AA from size of matrix, N_nil from
    - PROBLEM:
        o mask should be selected differently: try adding ROISelector.Ring class
        o mask should only be used to determine the size of the source plane
        o filter should then compare all pixels and not just masked values!!!
"""
###############################################################################
# Imports
###############################################################################
import sys
import os
import numpy as np
import time
from functools import partial
from pathos.multiprocessing import ProcessingPool as Pool
from scipy import ndimage
from scipy.sparse import lil_matrix as sparse_Lmatrix
# from scipy.sparse import csc_matrix as sparse_Cmatrix
from scipy.sparse import csr_matrix as sparse_Rmatrix
from scipy.sparse import diags as sparse_Dmatrix
from scipy.sparse.linalg import spsolve, lsqr, lsmr, cgs, lgmres, minres, qmr
from sklearn.preprocessing import normalize as row_norm
import matplotlib.pyplot as plt

from gleam.skyf import SkyF
from gleam.lensobject import LensObject
from gleam.skypatch import SkyPatch
from gleam.multilens import MultiLens
from gleam.utils.lensing import LensModel
from gleam.utils.lensing import deflect, external_shear_grad, external_ptmass_grad
from gleam.utils.lensing import downsample_model
from gleam.utils.linalg import is_symm2D
import gleam.utils.optimized as cython_optimized
import gleam.utils.rgb_map as glmrgb
from gleam.glass_interface import glass_renv, filter_env, export_state
glass = glass_renv()

__all__ = ['ReconSrc', 'synth_filter', 'synth_filter_mp', 'eval_residuals']


class KeyboardInterruptError(Exception):
    pass


###############################################################################
class ReconSrc(object):
    """
    Framework for source reconstruction
    """
    def __init__(self, gleamobject, model, M=40, M_fullres=None, mask_keys=[], verbose=False):
        """
        Initialize

        Args:
            gleamobject <GLEAM object> - a GLEAM object instance with .fits file's data
            model <str> - filename of a GLASS .state file, or a loaded GLASS state, or a LensModel object

        Kwargs:
            M <int> - source plane pixel radius; the total source plane will be (2*M+1)x(2*M+1)

        Return:
            <ReconSrc object> - standard initializer for ReconSrc
        """
        self.gleamobject = gleamobject
        # load input data
        if isinstance(self.gleamobject, (SkyPatch, MultiLens)):
            self.lens_objects = self.gleamobject.lens_objects
        elif isinstance(self.gleamobject, (SkyF, LensObject)):
            self.gleamobject.mag_formula = None  # to enable pickling
            self.lens_objects = [self.gleamobject]
            self.lensobject = self.gleamobject
        else:
            raise TypeError("ReconSrc needs a GLEAM object (LensObject/MultiLens) as input!")
        self.lensobject = self.lens_objects[0]  # grab first lensobject if there are more than one; TODO: make method with more lenses
        self.mask_keys = mask_keys  # sets mask
        self.rotation = 0
        # load model data
        if isinstance(model, str) and model.endswith('.state'):
            self.model = LensModel(glass.glcmds.loadstate(model))
        elif isinstance(model, glass.environment.Environment):
            self.model = LensModel(model)
        elif isinstance(model, LensModel) and model.__class__.__base__.__name__ == 'GLASSModel':
            self.model = model
        elif isinstance(model, LensModel):
            self.model = model
        else:
            self.model = LensModel(model)
        self.model_index = -1
        self.obj_index = self.model.obj_idx

        # source plane
        # projection happens full resolution and is antialiased afterwards
        self.M_fullres = M_fullres  # full resolution pixel radius, sets N_fullres
        self.M = M                  # resolution pixel radius after antialiasing, sets N
        self.r_fullres = None       # full resolution scale
        self.r_max = None           # antialiasing radius
        self.N_nil = 0              # skipped pixels during antialiasing
        if self.M_fullres is None:
            self.M_fullres = 2*self.lensobject.naxis1
        if self.M is None:
            self.M = self.M_fullres

        # psf
        self.psf = None

        # cache for high demand operations
        self._cache = {}
        for v in ['inv_proj', 'M_fullres', 'r_fullres', 'M', 'r_max',
                  'N_nil', 'reproj_d_ij']:
            self._cache[v] = {k: {i: None for i in range(-1, self.model.N)}
                              for k in range(self.model.N_obj)}
        self._cache['rotation'] = {k: {i: [] for i in range(-1, self.model.N)}
                                   for k in range(self.model.N_obj)}

        # some verbosity
        if verbose:
            print(self.__v__)

    def __getitem__(self, key):
        if isinstance(key, int):
            self.chmdl(key)
        else:
            NotImplemented

    def __str__(self):
        return "ReconSrc{}".format((self.N, self.N))

    def __repr__(self):
        return self.__str__()

    @property
    def tests(self):
        """
        A list of attributes being tested when calling __v__

        Args/Kwargs:
            None

        Return:
            tests <list(str)> - a list of test variable strings
        """
        return ['lensobject', 'model', 'mask_keys', 'M', 'N', 'r_max',
                'M_fullres', 'N_fullres', 'r_fullres', 'N_nil', 'rotation']

    @property
    def __v__(self):
        """
        Info string for test printing

        Args/Kwargs:
            None

        Return:
            <str> - test of ReconSrc attributes
        """
        return "\n".join([t.ljust(20)+"\t{}".format(self.__getattribute__(t)) for t in self.tests])

    @property
    def rotation(self):
        """
        TODO
        """
        if not hasattr(self, '_rotation'):
            self._rotation = 0
        return self._rotation

    @rotation.setter
    def rotation(self, angle):
        """
        TODO
        """
        if hasattr(self, '_cache'):
            self._cache['rotation'][self.obj_index][self.model_index].append(angle)
        self._rotation = angle % 360

    @property
    def N_fullres(self):
        """
        The full-resolution source plane row and column size,
        derived from the pixel radius M_fullres

        Args/Kwargs:
            None

        Return:
            N_fullres <int> - number of full-resolution source plane pixels in each row and column
        """
        return 2*self.M_fullres+1

    @N_fullres.setter
    def N_fullres(self, N_fullres):
        """
        Set the full-resolution source plane row and column size

        Args:
            N_fullres <int> - number of full-resolution source plane pixels in each row and column

        Kwargs/Return:
            None
        """
        self.M_fullres = int(N_fullres/2)

    @property
    def N(self):
        """
        The source plane row and column size, derived from the pixel radius M

        Args/Kwargs:
            None

        Return:
            N <int> - number of source plane pixels in each row and column
        """
        return 2*self.M+1

    @N.setter
    def N(self, N):
        """
        Set the source plane row and column size

        Args:
            N <int> - number of source plane pixels in each row and column

        Kwargs/Return:
            None
        """
        self.M = int(N/2)

    @property
    def mask(self):
        """
        The boolean mask selecting the data to be included in the mapping

        Args/Kwargs:
            None

        Return:
            mask <np.ndarray(bool)> - boolean array mask
        """
        mask = False*np.empty(self.lensobject.data.shape, dtype=bool)
        for k in self.mask_keys:
            mask = np.logical_or.reduce([mask]+self.lensobject.roi._masks[k])
        return mask

    def image_mask(self, f=0.8, n_sigma=3):
        """
        Estimate a mask which includes most of the images

        Args:
            None

        Kwargs:
            n_sigma <float> - sigma level for threshold estimation

        Return:
            mask <np.ndarray(bool)> - mask splitting signal from background
        """
        data = self.lens_map().copy()
        if np.any(self.mask):
            lmsk = self.mask
            data[lmsk] = 0
        threshold_map = self.lensobject.finder.threshold_estimate(data, sigma=n_sigma)
        mask = np.abs(data) >= f*threshold_map
        return mask

    def ring_mask(self, dr=10, inner=None):
        """
        Kwargs:
            dr <int> - thickness of the ring mask
            inner <int> - inner radius of the ring
        """
        if np.any(self.mask):
            rmsk = self.mask
            # get center and radius of rmsk
            # expand by dr -> Rmsk
            # return Rmsk and not rmsk

    def chbnd(self, index=None):
        """
        Change the lensobject/band; moves to the next lensobject/band in the list by default

        Args:
            None

        Kwargs:
            index <int> - the index which to swtich to next

        Return:
            None
        """
        if index is None:
            index = (self.lens_objects.index(self.lensobject)+1) % len(self.lens_objects)
        self.lensobject = self.lens_objects[index]

    def chobj(self, index=None):
        """
        Change the glass object; moves to the next glass object in the list by default

        Args:
            None

        Kwargs:
            index <int> - the index which to swtich to next

        Return:
            None
        """
        if index is None:
            index = (self.obj_index + 1) % self.model.N_obj
        self.obj_index = index
        self.model.obj_idx = self.obj_index

    def chmdl(self, index=None):
        """
        Change the glass model; moves to the next model in the list by default

        Args:
            None

        Kwargs:
            index <int> - the index which to swtich to next

        Return:
            None
        """
        if index is None:
            index = (self.model_index + 1) % self.model.N
        self.model_index = index
        # self.ploc = self.model.xy_grid(as_complex=True).flatten()
        # self.cell_sizes = self.model.cellsize_grid().flatten()

    @property
    def extent(self):
        """
        Physical extent of the lens plane map

        Args/Kwargs:
            None

        Return:
            extent <list> - physical map extent
        """
        return self.lensobject.extent

    @property
    def model_extent(self):
        """
        Physical extent of the lens model's kappa map

        Args/Kwargs:
            None

        Return:
            extent <list> - physical map extent
        """
        return self.model.extent

    @property
    def src_extent(self):
        """
        Physical extent of the source plane map

        Args/Kwargs:
            None

        Return:
            extent <list> - physical map extent
        """
        return [-1*self.r_max, self.r_max, -1*self.r_max, self.r_max]

    @property
    def src_pxscale(self):
        """
        Physical pixel size of the source plane map

        Args/Kwargs:
            None

        Return:
            pxscale <list> - physical pixel scale
        """
        pxscale = 2*self.r_max/self.N
        return pxscale

    def ensavg_mdl(self, selection=None):
        """
        Get the ensemble average model of all models in the GLASS state (or a subselection thereof)

        Args:
            None

        Kwargs:
            selection <list(int)> - a subselection of the models used for ensemble averaging

        Return:
            obj, data <glass.environment Object, dict> - GLASS environment and data dictionary
                                                         of the ensemble average
        """
        if selection is None:
            return self.model.ensemble_average()
        else:
            mdlcpy = self.model.subset(selection)
            return mdlcpy.ensemble_average()

    def update_cache(self, cache, variables=['inv_proj', 'M_fullres', 'r_fullres',
                                             'M', 'r_max', 'N_nil', 'reproj_d_ij'],
                     verbose=False):
        """
        Update cache with another cache dictionary, overwriting None values

        Args:
            cache <dict> - constructed cache or cache of another object

        Kwargs:
            variables <list(str)> - variables in cache to be updated
            verbose <bool> - verbose mode; print command line statements

        Return:
            None
        """
        for v in variables:
            for k in cache[v].keys():
                for i in cache[v][k].keys():
                    original = self._cache[v][k][i]
                    new_var = cache[v][k][i]
                    self._cache[v][k][i] = new_var if new_var is not None else original

    def flush_cache(self, variables=['inv_proj', 'M_fullres', 'r_fullres',
                                     'M', 'r_max', 'N_nil', 'reproj_d_ij'],
                    verbose=False):
        """
        Remove cached matrix and reprojection maps for current object/model index

        Args:
            None

        Kwargs:
            variables <list(str)> - variables in cache to be updated
            verbose <bool> - verbose mode; print command line statements

        Return:
            None
        """
        for v in variables:
            self._cache[v] = {k: {i: None for i in range(-1, self.model.N)}
                              for k in range(self.model.N_obj)}
        self._cache['rotation'] = {k: {i: [] for i in range(-1, self.model.N)}
                                   for k in range(self.model.N_obj)}

    @staticmethod
    def _psf_f(psf_data, dx, dy, center=None):
        """
        PSF function

        Args:
            psf_data <np.ndarray> - data of the PSF
            dx <float> - distance from the center along x
            dy <float> - distance from the center along y

        Kwargs:
            center <tuple/list> - center/symmetry point of the PSF

        Return:
            psf_val <float> - value of the PSF at the specified point
        """
        if center is None:
            yc, xc = [X//2 for X in psf_data.shape]
        else:
            yc, xc = center[::-1]
        y, x = yc+dy, xc+dx
        if (0 <= y < psf_data.shape[1]) and (0 <= x < psf_data.shape[0]):
            return psf_data[y, x]
        return 0

    def calc_psf(self, psf_file, window_size=6, cy_opt=False, normalize=True,
                 adjust_scale=False, verbose=False):
        """
        Calculate the PSF matrix attribute

        Args:
            psf_file <str> - path to the PSF .fits file, or a SkyF object

        Kwargs:
            window_size <int> - window size of the PSF which is applied
            cy_opt <bool> - use optimized cython method to construct psf matrix
            verbose <bool> - verbose mode; print command line statements

        Return:
            P_kl <scipy.sparse.csc.csc_matrix> - the PSF matrix as blurring operator
        """
        if isinstance(psf_file, str):
            psf = SkyF(psf_file, verbose=False)
        elif isinstance(psf_file, SkyF):
            psf = psf_file
        if adjust_scale and psf.px2arcsec[0] != self.lensobject.px2arcsec[0]:
            f_zoom = psf.px2arcsec[0]/self.lensobject.px2arcsec[0]
            shape = [int(f_zoom*s) for s in psf.data.shape]
            d_psf = downsample_model(psf.data, extent=psf.extent,
                                     shape=shape, pixel_scale=psf.px2arcsec[0])
        else:
            d_psf = psf.data
        if normalize:
            rmsk = glmrgb.radial_mask(d_psf, radius=window_size+1+window_size//5)
            rsum = np.sum(d_psf[rmsk])
            d_psf = d_psf / rsum
        d_psf = d_psf.astype('f8')
        Ny, Nx = self.lensobject.data.shape    # image plane dimensions
        # Ypsf, Xpsf = d_psf.shape               # PSF window dimensions
        c_psf = [d//2 for d in d_psf.shape]    # center of the PSF
        # is_symm = is_symm2D(d_psf)           # test for symmetry
        if cy_opt:
            P_kl = np.zeros((Ny*Ny, Nx*Nx), dtype=np.float64)
            P_kl = cython_optimized.calc_psf(P_kl, d_psf, c_psf[1], c_psf[0], Nx, Ny, window_size)
            P_kl = sparse_Rmatrix(P_kl)
        else:
            P_kl = sparse_Lmatrix((Ny*Ny, Nx*Nx))
            idcs = {kl: self.lensobject.idx2yx(kl, cols=Ny) for kl in range(Nx*Ny)}
            coords = {self.lensobject.idx2yx(kl, cols=Ny): kl for kl in range(Nx*Ny)}
            # # set up the PSF matrix
            for k in range(Nx*Ny):
                yk, xk = idcs[k]
                for xl in range(max(xk-window_size, 0), min(xk+window_size+1, Nx)):
                    for yl in range(max(yk-window_size, 0), min(yk+window_size+1, Ny)):
                        l = coords[(yl, xl)]
                        dx, dy = xk-xl, yk-yl
                        P_kl[k, l] = P_kl[l, k] = self._psf_f(d_psf, dx, dy, center=c_psf)
            P_kl = P_kl.tocsr()
        self.psf = P_kl
        if verbose:
            print("PSF: P_kl{}".format(P_kl.shape))
            rs = np.sum(d_psf[c_psf[0]-window_size:c_psf[0]+window_size+1,
                              c_psf[1]-window_size:c_psf[1]+window_size+1])
            print("Sum: {}".format(rs))
        return P_kl

    def d_ij(self, flat=True, include_rotation=True, mask=False, composite=False):
        """
        Lens plane data array

        Args:
            None

        Kwargs:
            flat <bool> - return the flattened array
            composite <bool> - return the composite data array [not yet tested]

        Return:
            dij <np.ndarray> - the lens plane data array
        """
        LxL = self.lensobject.naxis1*self.lensobject.naxis2
        dta = self.lensobject.data
        if mask and np.any(self.mask):
            mskij = self.mask
            dta[mskij] = 0
        if self.rotation != 0 and include_rotation:
            dta = ndimage.rotate(dta, self.rotation, reshape=False)
        if composite and isinstance(self.gleamobject, MultiLens):
            dta = self.lensobject.composite
        if flat:
            dij = np.array([dta[self.lensobject.idx2yx(i)] for i in range(LxL)])
        else:
            dij = dta
        return dij

    def lens_map(self, flat=False, include_rotation=True, mask=False, composite=False):
        """
        Lens plane data array

        Args:
            None

        Kwargs:
            flat <bool> - return the flattened array
            mask <bool> - return only the masked part of the lens plane data
            composite <bool> - return the composite data array [not yet tested]

        Return:
            dij <np.ndarray> - the lens plane data array
        """
        dij = self.d_ij(flat=flat, include_rotation=include_rotation, mask=mask,
                        composite=composite)
        return dij

    def delta_beta(self, ang_pos, beta=0j, zcap=None):
        """
        Delta beta from lens equation using the current kappa model

        Args:
            theta <np.ndarray/list/tuple/complex> - 2D angular position coordinates

        Kwargs:
            beta <complex> - source point position

        Return:
            delta_beta <np.ndarray> - beta from lens equation
        """
        data = self.model.kappa_grid(model_index=self.model_index, refined=False).flatten()
        ploc = self.model.xy_grid(as_complex=True, refined=False).flatten()
        cell_sizes = self.model.cellsize_grid(refined=False).flatten()
        if isinstance(ang_pos, np.ndarray) and ang_pos.shape[-1] == 2:
            theta = np.empty(ang_pos.shape[:-1], dtype=np.complex)
            theta.real = ang_pos[..., 1]
            theta.imag = ang_pos[..., 0]
            grad_phi = np.vectorize(deflect, signature='(),(m),(m),(m)->()')
        elif isinstance(ang_pos, (list, tuple)) and len(ang_pos) == 2:
            theta = complex(*ang_pos)
            grad_phi = deflect
        else:
            theta = ang_pos
            grad_phi = deflect
        zcap = 1.  # if the model's have been rescaled to kappa_inf use 1./dlsds
        if np.any(self.model.betas[self.model_index]):
            beta = self.model.betas[self.model_index]
            if not isinstance(beta, complex):
                beta = complex(*beta)
        if hasattr(self.model, 'shears') and np.any(self.model.shears):
            shear = self.model.shears[self.model_index]
            s = external_shear_grad(shear, theta)
        else:
            s = 0
        if hasattr(self.model, 'ptmasses') and len(self.model.ptmasses[self.model_index]) > 0:
            extm = self.model.ptmasses[self.model_index]
            p = external_ptmass_grad(extm, theta)
        else:
            p = 0
        alpha = grad_phi(theta, data, ploc, cell_sizes) + s + p
        dbeta = beta - theta + alpha * zcap
        return np.array([dbeta.real, dbeta.imag]).T

    def srcgrid_deflections(self, mask=None, limit=True):
        """
        Calculate the map radius of the source plane for a given model

        Args:
            None

        Kwargs:
            mask <np.ndarray(bool)> - boolean mask for image plane pixel selection

        Return:
            r_max <float> - maprad of the source plane
        """
        LxL = self.lensobject.naxis1*self.lensobject.naxis2
        ij = range(LxL)
        if mask is not None:
            ij = [self.lensobject.yx2idx(ix, iy) for ix, iy in zip(*np.where(mask))]
        theta = np.array([self.lensobject.theta(i) for i in ij])  # grid coords [arcsec]
        dbeta = self.delta_beta(theta)  # distance components from center on source plane [arcsec]
        # r_max = np.nanmax(np.abs(dbeta))  # maximal distance [arcsec]
        r_max = np.nanmax(np.sqrt(dbeta[:, 0]**2+dbeta[:, 1]**2))
        if limit:
            r_max = max(r_max, self.model.maprad/2.)
            if r_max > 12:
                r_max = self.model.maprad/2.
        return dbeta, r_max

    def srcgrid_mapping(self, dbeta=None, pixrad=None, maprad=None, mask=None):
        """
        Calculate the mapping from image to source plane for a given model

        Args:
            None

        Kwargs:
            dbeta <np.ndarray> - deflections on the source plane
            pixrad <int> - pixel radius of the deflection grid
            maprad <float> - map radius of the deflection grid
            mask <np.ndarray(bool)> - boolean mask for image plane pixel selection

        Return:
            xy <np.ndarray(int)> - pixel mapping coordinates from image to source plane
        """
        if dbeta is None:
            LxL = self.lensobject.naxis1*self.lensobject.naxis2
            ij = range(LxL)
            if mask is not None:
                ij = [self.lensobject.yx2idx(ix, iy) for ix, iy in zip(*np.where(mask))]
            theta = np.array([self.lensobject.theta(i) for i in ij])  # grid coords [arcsec]
            dbeta = self.delta_beta(theta)  # distances from center on source plane [arcsec]
        if pixrad is None:
            pixrad = self.M_fullres
        if maprad is None:
            maprad = np.nanmax(np.abs(dbeta))
            # maprad = np.nanmax(np.sqrt(dbeta[:, 0]**2+dbeta[:, 1]**2))  # maximal distance [arcsec]
        xy = np.int16(np.floor(pixrad*(1+dbeta/maprad))+.5)
        return xy

    def inv_proj_matrix(self, cy_opt=False, use_mask=False, r_max=None,
                        return_props=False, asarray=False):
        """
        The projection matrix to get from the image plane to the source plane

        Args:
            None

        Kwargs:
            cy_opt <bool> - use optimized cython method to construct inverse projection matrix
                            [currently broken]
            asarray <bool> - if True, return matrix as array, otherwise as scipy.sparse matrix

        Return:
            Mij_p <scipy.sparse.lil.lil_matrix> - inverse projection map from image to source plane
        """
        if cy_opt:
            return self.inv_proj_matrix_cython(use_mask=use_mask, r_max=r_max,
                                               r_fullres=r_fullres, asarray=asarray)
        else:
            # # lens plane
            LxL = self.lensobject.naxis1*self.lensobject.naxis2
            ij = range(LxL)
            # calculate optimal mapping resolution: r_max
            msk = self.image_mask()   # msk = self.mask if use_mask else None
            if r_max is None:
                _, self.r_max = self.srcgrid_deflections(mask=msk)
            else:
                self.r_max = float(r_max)
            # calculate mapping: r_fullres, dbeta
            if use_mask and np.any(self.mask):
                ij = [self.lensobject.yx2idx(ix, iy) for ix, iy in zip(*np.where(~self.mask))]
                dbeta, self.r_fullres = self.srcgrid_deflections(mask=~self.mask)
            else:
                dbeta, self.r_fullres = self.srcgrid_deflections(mask=None, limit=False)
            map2px = self.M/self.r_max
            self.r_fullres = max(self.r_fullres, self.r_max)
            self.M_fullres = int(self.r_fullres*map2px + .5)
            # get model's source grid mapping
            xy = self.srcgrid_mapping(dbeta=dbeta, pixrad=self.M_fullres, maprad=self.r_fullres)
            # # source plane dimensions
            NxN = self.N*self.N
            N_l, N_r = self.N_fullres//2-self.N//2, self.N_fullres//2+self.N//2
            # # construct projection matrix
            self.N_nil = 0
            Mij_p = sparse_Lmatrix((LxL, NxN), dtype=np.int16)
            for i, (x, y) in enumerate(xy):
                # antialiasing
                if (N_l < x < N_r) and (N_l < y < N_r):
                    x -= N_l
                    y -= N_l
                else:
                    self.N_nil += 1
                    continue
                # fill matrix entry
                p_idx = self.lensobject.yx2idx(y, x, cols=self.N)
                if p_idx >= NxN:
                    message = "Warning!"
                    message += " Projection discovered pixel out of range in matrix construction!"
                    message += " Something might be wrong!"
                    print(message)
                    continue
                ij_idx = ij[i]
                Mij_p[ij_idx, p_idx] = 1
            Mij_p = Mij_p.tocsr()
        if self._cache['inv_proj'][self.obj_index][self.model_index] is None:
            self._cache['inv_proj'][self.obj_index][self.model_index] = Mij_p.copy()
        if self._cache['N_nil'][self.obj_index][self.model_index] is None:
            self._cache['N_nil'][self.obj_index][self.model_index] = self.N_nil
        if self._cache['M_fullres'][self.obj_index][self.model_index] is None:
            self._cache['M_fullres'][self.obj_index][self.model_index] = self.M_fullres
        if self._cache['r_fullres'][self.obj_index][self.model_index] is None:
            self._cache['r_fullres'][self.obj_index][self.model_index] = self.r_fullres
        if self._cache['r_max'][self.obj_index][self.model_index] is None:
            self._cache['r_max'][self.obj_index][self.model_index] = self.r_max
        if asarray:
            return Mij_p.toarray()
        if return_props:
            return dbeta, xy, self.r_max, self.r_fullres
        return Mij_p

    def inv_proj_matrix_cython(self, use_mask=False, r_max=None, r_fullres=None, asarray=False):
        """
        Same as function <inv_proj_matrix>, but optimized using cython
        """
        # Lx, Ly = self.lensobject.naxis1, self.lensobject.naxis2
        # if hasattr(self.lensobject, 'lens'):
        #     origin = self.lensobject.lens
        # else:
        #     origin = self.lensobject.center
        # # msk = self.image_mask() if use_mask else None
        # msk = self.mask if use_mask else None
        # image_ij = np.array([self.lensobject.yx2idx(ix, iy)
        #                         for ix, iy in zip(*np.where(msk))], dtype=np.int32)
        # if np.any(self.model.betas[self.obj_index]):
        #     src = self.model.betas[self.obj_index]
        #     if not isinstance(src, complex):
        #         src = complex(*src)
        # zcap = 1./self.model.dlsds
        # kappa = self.model.kappa_grid(model_index=self.model_index, refined=False).flatten()
        # ploc = self.model.xy_grid(as_complex=True, refined=False).flatten()
        # cell_size = self.model.cellsize_grid(refined=False).flatten()
        # extra_potentials = [(data[e.name], e) for e in obj.extra_potentials]  # fix
        # Mij_p = sparse_Lmatrix((Lx*Ly, self.N*self.N), dtype=np.int32)
        # Mij_p, N_nil, M_fullres, r_fullres, r_max = cython_optimized.inv_proj_matrix(
        #     Mij_p, Lx, Ly, self.N, image_ij,
        #     origin.x, origin.y, self.lensobject.px2arcsec[0],
        #     src.real, src.imag, zcap, kappa, ploc, cell_size,
        #     [])
        # Mij_p = Mij_p.tocsr()
        # self.N_nil = N_nil
        # self.M_fullres = M_fullres
        # self.r_fullres = r_fullres
        # self.r_max = r_max
        return NotImplemented

    def proj_matrix(self, **kwargs):
        """
        The projection matrix to get from the source plane to the image plane

        Args:
            None

        Kwargs:
            cy_opt <bool> - use optimized cython method to construct inverse projection matrix
            asarray <bool> - if True, return matrix as array, otherwise as scipy.sparse matrix

        Return:
            Mp_ij <scipy.sparse.csc.csc_matrix> - projection map from source to image plane
        """
        Mij_p = self.inv_proj_matrix(**kwargs)
        Mp_ij = Mij_p.T.to_csr()
        return Mp_ij

    def d_p(self, method='minres', iter_lim=10000, cy_opt=False, use_psf=False, flat=True,
            cached=False, sigma=1, sigma2=None, sigmaM2=None, use_mask=False, use_filter=False,
            composite=False):
        """
        Recover the source plane data by minimizing
        (M^q_i.T * sigma_i^-2) M^i_p d_p = (M^q_i.T * sigma_i^-2) d_i

        Args:
            None

        Kwargs:
            method <str> - option to choose the minimizing method ['lsqr','lsmr','row_norm']
            iterations <int> - limit of iterations
            cy_opt <bool> - use optimized cython method to construct inverse projection matrix
            use_psf <bool> - use the PSF to smooth the projection matrix
            flat <bool> - return the flattened array
            cached <bool> - use a cached inverse projection matrix rather than computing it
            sigma <float/np.ndarray> - uncertainty map
            sigma2 <float/np.ndarray> - squared uncertainty map
            sigmaM2 <scipy.sparse.diags> - sparse diagonal matrix with inverse sigma2s
            use_mask <bool> - masked areas will not be projected
            composite <bool> - return the composite data array

        Return:
            dp <np.ndarray> - the source plane data
        """
        # LxL = self.lensobject.naxis1*self.lensobject.naxis2  # lens plane size
        # projection data
        dij = self.d_ij(flat=True, mask=use_mask, composite=composite)
        # projection mapping
        Mij_p = None
        if cached:
            Mij_p = self._cache['inv_proj'][self.obj_index][self.model_index]
            self.r_max = self._cache['r_max'][self.obj_index][self.model_index]
            self.M_fullres = self._cache['M_fullres'][self.obj_index][self.model_index]
            self.r_fullres = self._cache['r_fullres'][self.obj_index][self.model_index]
            self.N_nil = self._cache['N_nil'][self.obj_index][self.model_index]
        if Mij_p is None:
            Mij_p = self.inv_proj_matrix(use_mask=use_mask, cy_opt=cy_opt)
            self._cache['inv_proj'][self.obj_index][self.model_index] = Mij_p.copy()
        if use_psf:
            Mij_p = self.psf * Mij_p
        # uncertainty
        if sigma2 is None:
            sigma2 = sigma*sigma
        if sigmaM2 is None:
            sigmaM2 = 1./sigma2
            if hasattr(sigmaM2, '__len__') and len(sigmaM2.shape) > 1:
                sigmaM2 = np.array(
                    [sigmaM2[self.lensobject.idx2yx(i)] for i in range(sigmaM2.size)])
                sigmaM2 = sparse_Dmatrix(sigmaM2)
                sigmaM2 = sigmaM2.tocsr()
        if method == 'row_norm':
            Mij_p = row_norm(Mij_p, norm='l1', axis=0)
            dp = Mij_p.T * dij
        Qij_p = sigmaM2 * Mij_p
        A = Mij_p.T.tocsr() * Qij_p
        b = dij * Qij_p
        if method == 'minres':
            dp = minres(A, b)[0]
        elif method == 'lsqr':
            dp = lsqr(A, b, iter_lim=iter_lim)[0]
        elif method == 'lsmr':
            dp = lsmr(A, b)[0]
        elif method == 'spsolve':
            dp = spsolve(A, b)
        elif method == 'cgs':
            dp = cgs(A, b)[0]
        elif method == 'lgmres':
            dp = lgmres(A, b, atol=1e-05)[0]
        elif method == 'qmr':
            dp = qmr(A, b)[0]
        if use_filter:
            dp = dp.reshape((self.N, self.N))
            dp = ndimage.median_filter(dp, size=2)
            dp = dp.reshape(self.N*self.N)
        if not flat:
            dp = dp.reshape((self.N, self.N))
        return dp

    def plane_map(self, flat=False, **kwargs):
        """
        Recover the source plane data map using the inverse projection matrix: d_p = M^ij_p * d_ij

        Args:
            None

        Kwargs:
            method <str> - option to choose the minimizing method ['lsqr','lsmr','row_norm']
            cy_opt <bool> - use optimized cython method to construct inverse projection matrix
            use_psf <bool> - use the PSF to smooth the projection matrix
            flat <bool> - return the flattened array
            cached <bool> - use a cached inverse projection matrix rather than computing it
            sigma <float/np.ndarray> - uncertainty map
            sigma2 <float/np.ndarray> - squared uncertainty map
            sigmaM2 <scipy.sparse.diags> - sparse diagonal matrix with inverse sigma2s
            composite <bool> - return the composite data array

        Return:
            dp <np.ndarray> - the source plane data
        """
        return self.d_p(flat=flat, **kwargs)

    def reproj_d_ij(self, flat=True, use_mask=False, use_filter=False,
                    from_cache=False, save_to_cache=False,
                    **kwargs):
        """
        Solve the inverse projection problem to Mij_p * d_ij = d_p, where Mij_p and d_p are known

        Args:
            None

        Kwargs:
            method <str> - option to choose the minimizing method ['lsqr','lsmr','row_norm']
            iterations <int> - limit of iterations
            cy_opt <bool> - use optimized cython method to construct inverse projection matrix
            use_psf <bool> - use the PSF to smooth the projection matrix
            flat <bool> - return the flattened array
            use_mask <bool> - replace masked areas if original was masked
            from_cache <bool> - use the cached reprojected map
            cached <bool> - use a cached inverse projection matrix rather than computing it
            save_to_cache <bool> - save the reprojected map to cache
            sigma <float/np.ndarray> - uncertainty map
            sigma2 <float/np.ndarray> - squared uncertainty map
            sigmaM2 <scipy.sparse.diags> - sparse diagonal matrix with inverse sigma2s
            composite <bool> - return the composite data array

        Return:
            dij <np.ndarray> - reprojection data based on the source reconstruction
        """
        cached = kwargs.get('cached', False)
        use_psf = kwargs.get('use_psf', True)
        dij = None
        if from_cache:
            dij = self._cache['reproj_d_ij'][self.obj_index][self.model_index]
            self.r_max = self._cache['r_max'][self.obj_index][self.model_index]
            self.M_fullres = self._cache['M_fullres'][self.obj_index][self.model_index]
            self.r_fullres = self._cache['r_fullres'][self.obj_index][self.model_index]
            self.N_nil = self._cache['N_nil'][self.obj_index][self.model_index]
        if dij is None:
            dp = self.d_p(flat=True, use_mask=use_mask, **kwargs)
            if cached:
                Mij_p = self._cache['inv_proj'][self.obj_index][self.model_index]
                self.r_max = self._cache['r_max'][self.obj_index][self.model_index]
                self.M_fullres = self._cache['M_fullres'][self.obj_index][self.model_index]
                self.r_fullres = self._cache['r_fullres'][self.obj_index][self.model_index]
                self.N_nil = self._cache['N_nil'][self.obj_index][self.model_index]
            else:
                Mij_p = self.inv_proj_matrix(use_mask=use_mask, cy_opt=kwargs.get('cy_opt', False))
                self._cache['inv_proj'][self.obj_index][self.model_index] = Mij_p.copy()
            if use_psf:
                Mij_p = self.psf * Mij_p

            dij = Mij_p * dp

            if save_to_cache:
                self._cache['reproj_d_ij'][self.obj_index][self.model_index] = dij.copy()
        if use_filter:
            dij = ndimage.median_filter(dij, size=2)
            # dij = ndimage.filters.gaussian_filter(dij, sigma=0.5)
        if use_mask and np.any(self.mask):
            msk = self.mask.flatten()
            dij[msk] = self.d_ij(flat=True, composite=kwargs.get('composite', False))[msk]
            # dij[msk] = self.d_ij(flat=True, include_rotation=False, composite=kwargs.get('composite', False))[msk]
        if not flat:
            dij = dij.reshape((self.lensobject.naxis1, self.lensobject.naxis2))
        return dij

    def reproj_map(self, flat=False, **kwargs):
        """
        Solve the inverse projection problem to Mij_p * d_ij = d_p, where Mij_p and d_p are known

        Args:
            None

        Kwargs:
            method <str> - option to choose the minimizing method ['lsqr','lsmr','row_norm']
            iterations <int> - limit of iterations
            cy_opt <bool> - use optimized cython method to construct inverse projection matrix
            use_psf <bool> - use the PSF to smooth the projection matrix
            flat <bool> - return the flattened array
            from_cache <bool> - use the cached reprojected map
            cached <bool> - use a cached inverse projection matrix rather than computing it
            save_to_cache <bool> - save the reprojected map to cache
            sigma <float/np.ndarray> - uncertainty map
            sigma2 <float/np.ndarray> - squared uncertainty map
            sigmaM2 <scipy.sparse.diags> - sparse diagonal matrix with inverse sigma2s
            composite <bool> - return the composite data array

        Return:
            dij <np.ndarray> - reprojection data based on the source reconstruction
        """
        return self.reproj_d_ij(flat=flat, **kwargs)

    def residual_map(self, flat=False, nonzero_only=False, within_radius=None, **kwargs):
        """
        Evaluate the reprojection by calculating the residual map to the lens map data

        Args:
            None

        Kwargs:
            method <str> - option to choose the minimizing method ['lsqr','lsmr','row_norm']
            iterations <int> - limit of iterations
            use_psf <bool> - use the PSF to smooth the projection matrix
            flat <bool> - return the flattened array
            cached <bool> - use a cached inverse projection matrix rather than computing it
            sigma <float/np.ndarray> - uncertainty map
            sigma2 <float/np.ndarray> - squared uncertainty map
            sigmaM2 <scipy.sparse.diags> - sparse diagonal matrix with inverse sigma2s
            ignore_null <bool> - 
            composite <bool> - return the composite data array

        Return:
            residual <np.ndarray> - reprojection data based on the source reconstruction
        """
        data = self.lens_map(flat=flat, composite=kwargs.get('composite', False))
        # data = self.lens_map(flat=flat, include_rotation=False, composite=kwargs.get('composite', False))
        reproj = self.reproj_map(flat=flat, **kwargs)
        # set defaults
        sigma = kwargs.pop('sigma', 1)
        sigma2 = kwargs.pop('sigma2', None)
        if sigma2 is None:
            sigma2 = sigma*sigma
        residuals = (data-reproj)**2/sigma2
        if nonzero_only:
            residuals[reproj == 0] = 0
            residuals[data == 0] = 0
        if within_radius is not None:
            rad = int(within_radius * (residuals.shape[-1] // 2))
            rmsk = glmrgb.radial_mask(residuals, radius=rad)
            residuals[~rmsk] = 0
        return residuals

    def reproj_chi2(self, data=None, reduced=False, output_all=False, nonzero_only=False,
                    noise=0, sigma=1, sigma2=None, sigmaM2=None, within_radius=None,
                    **kwargs):
        """
        Evaluate the reprojection by calculating the absolute squared residuals to the data

        Args:
            None

        Kwargs:
            reduced <bool> - return the reduced chi^2 (i.e. chi2 divided by the number of pixels)
            nonzero_only <bool> - only sum over pixels which were actually reprojected
            method <str> - option to choose the minimizing method ['lsqr','lsmr','row_norm']
            iterations <int> - limit of iterations
            use_psf <bool> - use the PSF to smooth the projection matrix
            cy_opt <bool> - use optimized cython method to construct inverse projection matrix
            from_cache <bool> - use the cached reprojected map
            save_to_cache <bool> - save the reprojected map to cache
            cached <bool> - use a cached inverse projection matrix rather than computing it
            sigma <float/np.ndarray> - sigma for chi2 calculation
            sigma2 <float/np.ndarray> - squared sigma for the chi2 calculation
            sigmaM2 <scipy.sparse.diags> - sparse diagonal matrix with inverse sigma2s
            noise <float/np.ndarray> - noise added to the data
            use_mask <bool> - use mask in the data to avoid sending selected light to source plane
            composite <bool> - return the composite data array

        Return:
            residual <float> - the residual between data and reprojection

        Note:
            - if mask is too big, reduced will adjust the degrees-of-freedom
        """
        # set defaults
        if sigma2 is None:
            sigma2 = sigma*sigma
        if sigmaM2 is None:
            sigmaM2 = 1./sigma2
            if hasattr(sigmaM2, '__len__') and len(sigmaM2.shape) > 1:
                sigmaM2 = np.array(
                    [sigmaM2[self.lensobject.idx2yx(i)] for i in range(sigmaM2.size)])
                sigmaM2 = sparse_Dmatrix(sigmaM2)
                sigmaM2 = sigmaM2.tocsr()
        # calculate synthetic image and residual map
        reproj = self.reproj_map(sigma=sigma, sigma2=sigma2, sigmaM2=sigmaM2, **kwargs)
        if data is None:
            data = self.lens_map(flat=False, composite=kwargs.get('composite', False))
            # data = self.lens_map(flat=False, include_rotation=False, composite=kwargs.get('composite', False))
            # data = data + noise
        residuals = (data - reproj)**2/sigma2
        if nonzero_only:
            residuals[reproj == 0] = 0
            # n_i = sigma2[msk] if hasattr(sigma2, '__len__') else sigma2
            # chi2 = np.sum(residual[msk]**2/n_i)
        if within_radius is not None:
            rad = int(within_radius * (residuals.shape[-1] // 2))
            rmsk = glmrgb.radial_mask(residuals, radius=rad)
            residuals[~rmsk] = 0
        chi2 = np.sum(residuals) + noise
        if reduced:
            N = data.size - np.sum(reproj == 0)
            chi2 = chi2 / (N - self.N_nil)
        elif output_all:
            N = data.size - np.sum(reproj == 0)
            chi2red = chi2 / (N - self.N_nil)
            return chi2, chi2red, self.N_nil
        return chi2



def run_model(reconsrc, mdl_index=0, angle=0, dzsrc=0,
              reduced=False, nonzero_only=True, within_radius=None,
              method='lsqr', use_psf=False, use_mask=True, use_filter=True,
              noise=0, sigma=1, sigma2=None, sigmaM2=None,
              cached=True, from_cache=True, save_to_cache=True,
              flush_cache=True, output_maps=False):
    """
    TODO
    """
    reconsrc.chmdl(mdl_index)
    if dzsrc:
        reconsrc.model.rescale(reconsrc.model.zl, reconsrc.model.zs+dzsrc,
                               zl=reconsrc.model.zl, zs=reconsrc.model.zs,
                               cosmo=reconsrc.model.cosmo)
        reconsrc.flush_cache()
    reconsrc.rotation = angle
    s2 = None
    if sigma2 is not None:
        s2 = ndimage.rotate(sigma2, angle, reshape=False)
        s2[s2 <= 0] = np.min(sigma2[sigma2 > 0])
    kw = dict(method=method, use_psf=use_psf, use_mask=use_mask, sigma2=s2.copy())
    chi2 = reconsrc.reproj_chi2(cached=True, from_cache=False, save_to_cache=save_to_cache,
                                nonzero_only=nonzero_only, within_radius=within_radius,
                                use_filter=use_filter,
                                reduced=reduced, noise=noise, **kw)
    if output_maps:
        srcplane = reconsrc.plane_map(cached=cached, **kw)
        synth = reconsrc.reproj_map(cached=True, from_cache=True, save_to_cache=False,
                                    use_filter=use_filter, **kw)
        resids = reconsrc.residual_map(cached=True, from_cache=True, save_to_cache=False,
                                       nonzero_only=nonzero_only, within_radius=within_radius,
                                       **kw)
        return chi2, (srcplane, synth, resids, s2)
    if flush_cache:
        reconsrc.flush_cache(variables=['reproj_dij'])
    return chi2


def synth_filter(statefile=None, gleamobject=None, reconsrc=None, psf_file=None,
                 percentiles=[], cy_opt=False,
                 reduced=False, method='minres', use_psf=True,
                 from_cache=True, cached=True, save_to_cache=True,
                 noise=0, sigma=1, sigma2=None, sigmaM2=None,
                 use_mask=False, nonzero_only=True,
                 N_models=None, save=False, return_obj=False,
                 stdout_flush=True, verbose=False):
    """
    Filter a GLASS state file using GLEAM's source reconstruction feature

    Args:
        None

    Kwargs:
        statefile <str> - the GLASS statefile
        gleamobject <GLEAM object> - a GLEAM object instance with .fits file's data
        reconsrc <ReconSrc object> - use a ReconSrc object from input
        psf_file <str> - path to the PSF .fits file
        percentiles <list(float)> - percentages the filter retains
        reduced <bool> - return the reduced chi^2 (i.e. chi2 divided by the number of pixels)
        nonzero_only <bool> - only sum over pixels which were actually reprojected
        cy_opt <bool> - use optimized cython method to construct inverse projection matrix
        method <str> - option to choose the minimizing method ['lsqr','lsmr','row_norm']
        use_psf <bool> - use the PSF to smooth the projection matrix
        from_cache <bool> - use the cached reprojected map
        cached <bool> - use a cached inverse projection matrix rather than computing it
        save_to_cache <bool> - save the reprojected map to cache
        sigma <float/np.ndarray> - sigma for chi2 calculation
        sigma2 <float/np.ndarray> - squared sigma for the chi2 calculation
        sigmaM2 <scipy.sparse.diags> - sparse diagonal matrix with inverse sigma2s
        use_mask <bool> - use mask in the data to avoid sending selected light to source plane
        N_models <int> - number of models to loop through
        save <bool> - save the filtered states automatically
        return_obj <bool> - return the object with all reprojections cached instead
        stdout_flush <bool> - flush stdout and update line in verbose mode
        verbose <bool> - verbose mode; print command line statements

    Return:
      if return_obj:
        recon_src <ReconSrc object> - the object with all reprojections cached
      else:
        filtered_states <list(glass.Environment object)> - the filtered states ready for export
        selected <list(int)> - index list of selected models
        chi2s <list(float)> - list of chi2s
    """
    if verbose and statefile is not None:
        print(statefile)

    if reconsrc is None:
        if gleamobject is not None and statefile is not None:
            recon_src = ReconSrc(gleamobject, statefile, M=40, verbose=verbose)
        else:
            return None
        if psf_file is not None and os.path.exists(psf_file):
            recon_src.calc_psf(psf_file, cy_opt=cy_opt)
    else:
        recon_src = reconsrc
        if psf_file is not None and os.path.exists(psf_file):
            recon_src.calc_psf(psf_file, cy_opt=cy_opt)

    chi2s = []
    if N_models is None:
        N_models = len(recon_src.gls.models)
    if sigma2 is None:
        sigma2 = sigma*sigma
    if sigmaM2 is None:
        sigmaM2 = 1./sigma2
        if hasattr(sigmaM2, '__len__') and len(sigmaM2.shape) > 1:
            sigmaM2 = np.array(
                [sigmaM2[recon_src.lensobject.idx2yx(i)] for i in range(sigmaM2.size)])
            sigmaM2 = sparse_Dmatrix(sigmaM2)

    data = recon_src.lens_map() + noise
    for i in range(N_models):
        recon_src.chmdl(i)
        chi2 = recon_src.reproj_chi2(data=data, reduced=reduced, nonzero_only=nonzero_only,
                                     use_mask=use_mask,
                                     sigma2=sigma2, sigmaM2=sigmaM2,
                                     cy_opt=cy_opt,
                                     save_to_cache=save_to_cache, from_cache=from_cache,
                                     cached=cached, method=method, use_psf=use_psf)
        chi2s.append(chi2)
        if verbose:
            message = "{:4d} / {:4d}: {:4.8f}\r".format(i+1, N_models, chi2)
            if stdout_flush:
                sys.stdout.write(message)
                sys.stdout.flush()
            else:
                print(message)

    if verbose:
        print("Number of residual models: {}".format(len(chi2s)))

    if return_obj:
        return recon_src

    rhi = [np.percentile(chi2s, p, interpolation='higher') for p in percentiles]
    rlo = [0 for r in rhi]
    selected = [[i for i, r in enumerate(chi2s) if rh > r > rl] for rh, rl in zip(rhi, rlo)]

    filtered = [filter_env(recon_src.gls, s) for s in selected]

    if save:
        dirname = os.path.dirname(statefile)
        basename = ".".join(os.path.basename(statefile).split('.')[:-1])
        saves = [dirname+'/'+basename+'_synthf{}.state'.format(p) for p in percentiles]
        for f, s in zip(filtered, saves):
            export_state(f, name=s)

    return filtered, selected, chi2s


def eval_residuals(index, reconsrc, data=None,
                   reduced=True, nonzero_only=True,
                   sigma=1, sigma2=None, sigmaM2=None,
                   save_to_cache=True, from_cache=True,
                   cy_opt=False, use_mask=False,
                   cached=True, method='minres', use_psf=True,
                   N_total=0, stdout_flush=True, verbose=False):
    """
    Helper function to evaluate the residual of a single model within a multiprocessing loop

    Args:
        reconsrc <ReconSrc object> - ReconSrc object which handles the source reconstruction
        index <int> - index of the model to analyse

    Kwargs:
        data <np.ndarray> - data to be fitted
        reduced <bool> - return the reduced chi^2 (i.e. chi2 divided by the number of pixels)
        nonzero_only <bool> - only sum over pixels which were actually reprojected
        cy_opt <bool> - use optimized cython method to construct inverse projection matrix
        method <str> - option to choose the minimizing method ['lsqr','lsmr','row_norm']
        use_psf <bool> - use the PSF to smooth the projection matrix
        from_cache <bool> - use the cached reprojected map
        cached <bool> - use a cached inverse projection matrix rather than computing it
        save_to_cache <bool> - save the reprojected map to cache
        sigma <float/np.ndarray> - sigma for chi2 calculation
        sigma2 <float/np.ndarray> - squared sigma for the chi2 calculation
        sigmaM2 <scipy.sparse.diags> - sparse diagonal matrix with inverse sigma2s
        use_mask <bool> - use mask in the data to avoid sending selected light to source plane
        N_total <int> - the total size of loop range
        stdout_flush <bool> - flush stdout and update line in verbose mode
        verbose <bool> - verbose mode; print command line statements

    Return:
        delta <float> - the chi2 sum of a single model
    """
    reconsrc.chmdl(index)
    try:
        delta = reconsrc.reproj_chi2(data=data, reduced=reduced, nonzero_only=nonzero_only,
                                     use_mask=use_mask,
                                     from_cache=from_cache, save_to_cache=save_to_cache,
                                     sigma=sigma, sigma2=sigma2, sigmaM2=sigmaM2,
                                     cy_opt=cy_opt,
                                     cached=cached, method=method, use_psf=use_psf)
        if verbose:
            message = "{:4d} / {:4d}: {:4.4f}\r".format(index+1, N_total, delta)
            if stdout_flush:
                sys.stdout.write(message)
                sys.stdout.flush()
            else:
                print(message)
        return reconsrc._cache, delta
    except KeyboardInterrupt:
        raise KeyboardInterruptError()


def synth_filter_mp(statefile=None, gleamobject=None, reconsrc=None, psf_file=None,
                    percentiles=[],
                    nproc=2,
                    cy_opt=False, use_mask=False,
                    reduced=False, nonzero_only=True, method='minres', use_psf=True,
                    from_cache=True, cached=True, save_to_cache=True,
                    noise=0, sigma=1, sigma2=None, sigmaM2=None,
                    N_models=None, save=False, return_obj=False,
                    stdout_flush=True, verbose=False):
    """
    Filter a GLASS state file using GLEAM's source reconstruction feature

    Args:

    Kwargs:
        statefile <str> - the GLASS statefile
        gleamobject <GLEAM object> - a GLEAM object instance with .fits file's data
        reconsrc <ReconSrc object> - use a ReconSrc object from input
        psf_file <str> - path to the PSF .fits file
        percentiles <list(float)> - percentages the filter retains
        reduced <bool> - return the reduced chi^2 (i.e. chi2 divided by the number of pixels)
        nonzero_only <bool> - only sum over pixels which were actually reprojected
        cy_opt <bool> - use optimized cython method to construct inverse projection matrix
        method <str> - option to choose the minimizing method ['lsqr','lsmr','row_norm']
        use_psf <bool> - use the PSF to smooth the projection matrix
        from_cache <bool> - use the cached reprojected map
        cached <bool> - use a cached inverse projection matrix rather than computing it
        save_to_cache <bool> - save the reprojected map to cache
        noise <np.ndarray> - noise with is artificially added to data
        sigma <float/np.ndarray> - sigma for chi2 calculation
        sigma2 <float/np.ndarray> - squared sigma for the chi2 calculation
        sigmaM2 <scipy.sparse.diags> - sparse diagonal matrix with inverse sigma2s
        use_mask <bool> - use mask in the data to avoid sending selected light to source plane
        N_models <int> - number of models to loop through
        save <bool> - save the filtered states automatically
        return_obj <bool> - return the object with all reprojections cached instead
        stdout_flush <bool> - flush stdout and update line in verbose mode
        verbose <bool> - verbose mode; print command line statements

    Return:
      if return_obj:
        recon_src <ReconSrc object> - the object with all reprojections cached
      else:
        filtered_states <list(glass.Environment object)> - the filtered states ready for export
        selected <list(int)> - index list of selected models
        chi2s <list(float)> - list of chi2s
    """
    if verbose and statefile is not None:
        print(statefile)

    if reconsrc is None:
        if gleamobject is not None and statefile is not None:
            recon_src = ReconSrc(gleamobject, statefile, M=20, verbose=verbose)
        else:
            return None
        if psf_file is not None and os.path.exists(psf_file):
            recon_src.calc_psf(psf_file, cy_opt=cy_opt)
    else:
        recon_src = reconsrc
        if psf_file is not None and os.path.exists(psf_file):
            recon_src.calc_psf(psf_file, cy_opt=cy_opt)

    if N_models is None:
        N_models = len(recon_src.gls.models)
    if sigma2 is None:
        sigma2 = sigma*sigma
    if sigmaM2 is None:
        sigmaM2 = 1./sigma2
        if hasattr(sigmaM2, '__len__') and len(sigmaM2.shape) > 1:
            sigmaM2 = np.array(
                [sigmaM2[recon_src.lensobject.idx2yx(i)] for i in range(sigmaM2.size)])
            sigmaM2 = sparse_Dmatrix(sigmaM2)

    data = recon_src.lens_map() + noise

    pool = Pool(processes=nproc)
    chi2s = [None]*N_models
    try:
        f = partial(eval_residuals, reconsrc=recon_src, data=data,
                    reduced=reduced, nonzero_only=nonzero_only,
                    cy_opt=cy_opt, use_mask=use_mask,
                    method=method, use_psf=use_psf,
                    from_cache=from_cache, cached=cached, save_to_cache=save_to_cache,
                    sigma2=sigma2, sigmaM2=sigmaM2,
                    N_total=N_models, stdout_flush=stdout_flush, verbose=verbose)
        output = pool.map(f, range(N_models))
        chi2s = [o[1] for o in output]
        caches = [o[0] for o in output]
        for c in caches:
            recon_src.update_cache(c)
        pool.clear()
    except KeyboardInterrupt:
        pool.terminate()

    if verbose:
        print("Number of residual models: {}".format(len(chi2s)))

    if return_obj:
        return recon_src

    rhi = [np.percentile(chi2s, p, interpolation='higher') for p in percentiles]
    rlo = [0 for r in rhi]
    selected = [[i for i, r in enumerate(chi2s) if rh > r > rl] for rh, rl in zip(rhi, rlo)]

    filtered = [filter_env(recon_src.gls, s) for s in selected]

    if save:
        dirname = os.path.dirname(statefile)
        basename = ".".join(os.path.basename(statefile).split('.')[:-1])
        saves = [dirname+'/'+basename+'_synthf{}.state'.format(p) for p in percentiles]
        for f, s in zip(filtered, saves):
            export_state(f, name=s)

    return filtered, selected, chi2s


# MAIN FUNCTION ###############################################################
def main(*args):
    pass


def parse_arguments():
    """
    Parse command line arguments
    """
    from argparse import ArgumentParser, RawTextHelpFormatter
    parser = ArgumentParser(description=__doc__, formatter_class=RawTextHelpFormatter)
    # main args
    parser.add_argument("case", nargs='?',
                        help="Path input to .fits file for reconsrc",
                        default=os.path.join(os.path.abspath(os.path.dirname(__file__)),
                                             'test'))

    # mode args
    parser.add_argument("-v", "--verbose", dest="verbose", action="store_true",
                        help="Run program in verbose mode",
                        default=False)
    parser.add_argument("-t", "--test", "--test-mode", dest="test_mode", action="store_true",
                        help="Run program in testing mode",
                        default=False)

    args = parser.parse_args()
    case = args.case
    delattr(args, 'case')
    return parser, case, args


###############################################################################
if __name__ == '__main__':
    jsons = ["/Users/phdenzel/adler/json/H1S0A0B90G0.json",
             "/Users/phdenzel/adler/json/H1S1A0B90G0.json",
             "/Users/phdenzel/adler/json/H2S1A0B90G0.json",
             "/Users/phdenzel/adler/json/H2S2A0B90G0.json",
             "/Users/phdenzel/adler/json/H2S7A0B90G0.json",
             "/Users/phdenzel/adler/json/H3S0A0B90G0.json",
             "/Users/phdenzel/adler/json/H3S1A0B90G0.json",
             "/Users/phdenzel/adler/json/H4S3A0B0G90.json",
             "/Users/phdenzel/adler/json/H10S0A0B90G0.json",
             "/Users/phdenzel/adler/json/H13S0A0B90G0.json",
             "/Users/phdenzel/adler/json/H23S0A0B90G0.json",
             "/Users/phdenzel/adler/json/H30S0A0B90G0.json",
             "/Users/phdenzel/adler/json/H36S0A0B90G0.json",
             "/Users/phdenzel/adler/json/H160S0A90B0G0.json",
             "/Users/phdenzel/adler/json/H234S0A0B90G0.json"]
    statefiles = ["/Users/phdenzel/adler/states/v1/H1S0A0B90G0.state",
                  "/Users/phdenzel/adler/states/v1/H1S1A0B90G0.state",
                  "/Users/phdenzel/adler/states/v1/H2S1A0B90G0.state",
                  "/Users/phdenzel/adler/states/v1/H2S2A0B90G0.state",
                  "/Users/phdenzel/adler/states/v1/H2S7A0B90G0.state",
                  "/Users/phdenzel/adler/states/v1/H3S0A0B90G0.state",
                  "/Users/phdenzel/adler/states/v1/H3S1A0B90G0.state",
                  "/Users/phdenzel/adler/states/v1/H4S3A0B0G90.state",
                  "/Users/phdenzel/adler/states/v1/H10S0A0B90G0.state",
                  "/Users/phdenzel/adler/states/v1/H13S0A0B90G0.state",
                  "/Users/phdenzel/adler/states/v1/H23S0A0B90G0.state",
                  "/Users/phdenzel/adler/states/v1/H30S0A0B90G0.state",
                  "/Users/phdenzel/adler/states/v1/H36S0A0B90G0.state",
                  "/Users/phdenzel/adler/states/v1/H160S0A90B0G0.state",
                  "/Users/phdenzel/adler/states/v1/H234S0A0B90G0.state"]
    # i = 7
    for i in range(len(jsons)):
        i=2
        with open(jsons[i]) as f:
            ml = MultiLens.from_json(f)
        # recon_src = ReconSrc(ml, statefiles[i], M=80, verbose=1)
        recon_src = ReconSrc(ml, statefiles[i], M=161, mask_keys=[], verbose=1)
        recon_src.chobj()
        import time
        ti = time.time()
        #print(recon_src.plane_map())
        #plt.imshow(recon_src.reproj_map())
        #plt.show()
        chi2 = recon_src.reproj_chi2(sigma=recon_src.lensobject.sigma(factor=0.1))
        print(u'\u03C7^2 = {}'.format(chi2))
        # print(all(self.lensobject.data.reshape(LxL) == np.array([self.lensobject.data[self.lensobject.idx2yx(i)] for i in range(LxL)])))
        # nevertheless, as data vector use: np.array([self.lensobject.data[self.lensobject.idx2yx(i)] for i in range(LxL)])
        tf = time.time()
        print(tf-ti)
        break

    # recon_src.mask_plot()
    # parser, case, args = parse_arguments()
    # testdir = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'test')
    # no_input = len(sys.argv) <= 1 and testdir in case
    # if no_input:
    #     parser.print_help()
    # elif args.test_mode:
    #     sys.argv = sys.argv[:1]
    #     from gleam.test.test_reconsrc import TestReconSrc
    #     TestReconSrc.main()
    # else:
    #     main(case, args)
