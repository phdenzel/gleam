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
from functools import partial
from pathos.multiprocessing import ProcessingPool as Pool
from scipy.sparse import lil_matrix as sparse_Lmatrix
from scipy.sparse import csc_matrix as sparse_Cmatrix
from scipy.sparse import diags as sparse_Dmatrix
from scipy.sparse.linalg import lsqr, lsmr, cgs, lgmres, minres, qmr
from sklearn.preprocessing import normalize as row_norm
import matplotlib.pyplot as plt

from gleam.skyf import SkyF
from gleam.lensobject import LensObject
from gleam.skypatch import SkyPatch
from gleam.multilens import MultiLens
from gleam.utils.linalg import is_symm2D
# import gleam.utils.optimized as cython_optimized
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
    def __init__(self, gleamobject, glsfile, M=40, M_fullres=None, mask_keys=[], verbose=False):
        """
        Initialize

        Args:
            gleamobject <GLEAM object> - a GLEAM object instance with .fits file's data
            glsfile <str> - filename of a GLASS .state file

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
            self.lens_objects = [self.gleamobject]
            self.lensobject = self.gleamobject
        else:
            raise TypeError("ReconSrc needs a GLEAM object (LensObject/MultiLens) as input!")
        # grab first lensobject if there are more than one; TODO: make method with more lenses
        self.lensobject = self.lens_objects[0]
        self.mask_keys = mask_keys  # sets mask
        self.gls = glass.glcmds.loadstate(glsfile)
        self.model_index = -1
        self.obj_index = 0

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
        self._cache = {v: {k: {i: None for i in range(-1, len(self.gls.models))}
                           for k in range(len(self.gls.objects))}
                       for v in ['inv_proj', 'M_fullres', 'r_fullres',
                                 'M', 'r_max', 'N_nil', 'reproj_d_ij']}

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
        return ['lensobject', 'gls', 'mask_keys', 'M', 'N', 'r_max',
                'M_fullres', 'N_fullres', 'r_fullres', 'N_nil']

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

    def image_mask(self, f=0.5, n_sigma=5):
        """
        Estimate a mask which includes most of the images

        Args:
            None

        Kwargs:
            n_sigma <float> - sigma level for threshold estimation

        Return:
            mask <np.ndarray(bool)> - mask splitting signal from background
        """
        threshold_map = self.lensobject.finder.threshold_estimate(self.lens_map(), sigma=n_sigma)
        mask = np.abs(self.lens_map()) >= f*threshold_map
        return mask

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
            index = (self.obj_index + 1) % len(self.gls.objects)
        self.obj_index = index

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
            index = (self.model_index + 1) % len(self.gls.models)
        self.model_index = index

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
            self.gls.make_ensemble_average()
            return self.gls.ensemble_average['obj,data'][self.obj_index]
        else:
            envcpy = filter_env(self.gls, selection)
            envcpy.make_ensemble_average()
            return envcpy.ensemble_average['obj,data'][self.obj_index]

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
            self._cache[v] = {k: {i: None for i in range(-1, len(self.gls.models))}
                              for k in range(len(self.gls.objects))}

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

    def calc_psf(self, psf_file, window_size=6, cy_opt=False, verbose=False):
        """
        Calculate the PSF matrix attribute

        Args:
            psf_file <str> - path to the PSF .fits file

        Kwargs:
            window_size <int> - window size of the PSF which is applied
            cy_opt <bool> - use optimized cython method to construct psf matrix
            verbose <bool> - verbose mode; print command line statements

        Return:
            P_kl <scipy.sparse.csc.csc_matrix> - the PSF matrix as blurring operator
        """
            
        psf = SkyF(psf_file, verbose=False)
        d_psf = psf.data
        Ny, Nx = self.lensobject.data.shape    # image plane dimensions
        Ypsf, Xpsf = d_psf.shape               # PSF window dimensions
        c_psf = [d//2 for d in d_psf.shape]    # center of the PSF
        # is_symm = is_symm2D(d_psf)           # test for symmetry

        if cy_opt:
            P_kl = np.zeros((Ny*Ny, Nx*Nx))
            P_kl = cython_optimized.calc_psf(P_kl, d_psf, c_psf[1], c_psf[0], Nx, Ny, window_size)
            P_kl = sparse_Cmatrix(P_kl)
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
                        P_kl[k, l] = P_kl[l, k] = self._psf_f(psf.data, dx, dy, center=c_psf)
            P_kl = P_kl.tocsc()
        self.psf = P_kl
        if verbose:
            print("PSF: P_kl{}".format(P_kl.shape))
        return P_kl

    def d_ij(self, flat=True, composite=False):
        """
        Lens plane data array

        Args:
            None

        Kwargs:
            flat <bool> - return the flattened array
            composite <bool> - return the composite data array

        Return:
            dij <np.ndarray> - the lens plane data array
        """
        LxL = self.lensobject.naxis1*self.lensobject.naxis2
        data = self.lensobject.data
        if composite and isinstance(self.gleamobject, MultiLens):
            data = self.lensobject.composite
        if flat:
            dij = np.array([data[self.lensobject.idx2yx(i)] for i in range(LxL)])
        else:
            dij = data
        return dij

    def lens_map(self, flat=False, mask=False, composite=False):
        """
        Lens plane data array

        Args:
            None

        Kwargs:
            flat <bool> - return the flattened array
            mask <bool> - return only the masked part of the lens plane data
            composite <bool> - return the composite data array

        Return:
            dij <np.ndarray> - the lens plane data array
        """
        dij = self.d_ij(flat=flat, composite=composite)
        if mask and np.any(self.mask):
            if flat:
                msk = self.mask.flatten()
            else:
                msk = self.mask
            dij[~msk] = 0
        return dij

    def delta_beta(self, theta):
        """
        Delta beta from lens equation using the current GLASS model

        Args:
            theta <list/tuple/complex> - 2D angular position coordinates

        Kwargs:
            None

        Return:
            delta_beta <np.ndarray> - beta from lens equation
        """
        if self.model_index >= 0:
            obj, data = self.gls.models[self.model_index]['obj,data'][self.obj_index]
        else:
            obj, data = self.ensavg_mdl()
        if isinstance(theta, np.ndarray) and len(theta) > 2:  # seems to work with
            ang_pos = np.empty(theta.shape[:-1], dtype=np.complex)
            ang_pos.real = theta[..., 0]
            ang_pos.imag = theta[..., 1]
            deflect = np.vectorize(obj.basis.deflect)  # , excluded=['data'])
        elif not isinstance(theta, (list, tuple)) and len(theta) == 2:
            ang_pos = complex(*theta)
            deflect = obj.basis.deflect
        else:
            ang_pos = theta
            deflect = obj.basis.deflect
        src = data['src'][0]
        zcap = obj.sources[0].zcap
        dbeta = src - ang_pos + deflect(ang_pos, data) / zcap
        return np.array([dbeta.real, dbeta.imag]).T

    def srcgrid_deflections(self, mask=None):
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
        r_max = np.nanmax(np.abs(dbeta))  # maximal distance [arcsec]
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
            maprad = np.nanmax(np.abs(dbeta))  # maximal distance [arcsec]
        xy = np.int16(np.floor(pixrad*(1+dbeta/maprad))+.5)
        return xy

    def inv_proj_matrix(self, asarray=False):
        """
        The projection matrix to get from the image plane to the source plane

        Args:
            None

        Kwargs:
            asarray <bool> - if True, return matrix as array, otherwise as scipy.sparse matrix

        Return:
            Mij_p <scipy.sparse.lil.lil_matrix> - inverse projection map from image to source plane
        """
        # # lens plane
        LxL = self.lensobject.naxis1*self.lensobject.naxis2
        ij = range(LxL)
        # calculate optimal mapping resolution
        _, self.r_max = self.srcgrid_deflections(mask=self.image_mask())
        if np.any(self.mask):
            ij = [self.lensobject.yx2idx(ix, iy) for ix, iy in zip(*np.where(self.mask))]
            dbeta, self.r_fullres = self.srcgrid_deflections(mask=self.mask)
        else:
            dbeta, self.r_fullres = self.srcgrid_deflections(mask=None)
        map2px = self.M/self.r_max
        self.M_fullres = int(self.r_fullres*map2px+.5)
        # get model's source grid mapping
        xy = self.srcgrid_mapping(dbeta=dbeta, pixrad=self.M_fullres, maprad=self.r_fullres)
        # # source plane dimensions
        NxN = self.N*self.N
        N_l, N_r = self.N_fullres//2-self.N//2, self.N_fullres//2+self.N//2
        # # construct projection matrix
        self.N_nil = 0
        Mij_p = sparse_Lmatrix((LxL, NxN), dtype=np.int16)
        for i, (x, y) in enumerate(xy):  # TODO: vectorize this loop
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
                print("Projection discovered a pixel out of range in matrix construction!")
                continue
            ij_idx = ij[i]
            Mij_p[ij_idx, p_idx] = 1
        self._cache['N_nil'][self.obj_index][self.model_index] = self.N_nil
        self._cache['M_fullres'][self.obj_index][self.model_index] = self.M_fullres
        self._cache['r_fullres'][self.obj_index][self.model_index] = self.r_fullres
        self._cache['r_max'][self.obj_index][self.model_index] = self.r_max
        if asarray:
            return Mij_p.toarray()
        Mij_p = Mij_p.tocsr()
        return Mij_p

    def proj_matrix(self, **kwargs):
        """
        The projection matrix to get from the source plane to the image plane

        Args:
            None

        Kwargs:
            asarray <bool> - if True, return matrix as array, otherwise as scipy.sparse matrix

        Return:
            Mp_ij <scipy.sparse.csc.csc_matrix> - projection map from source to image plane
        """
        Mij_p = self.inv_proj_matrix(**kwargs)
        Mp_ij = Mij_p.T
        return Mp_ij

    def d_p(self, method='lsmr', use_psf=True, flat=True, cached=False,
            sigma=1, sigma2=None, sigmaM2=None, composite=False):
        """
        Recover the source plane data by minimizing
        (M^q_i.T * sigma_i^-2) M^i_p d_p = (M^q_i.T * sigma_i^-2) d_i

        Args:
            None

        Kwargs:
            method <str> - option to choose the minimizing method ['lsqr','lsmr','row_norm']
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
        # LxL = self.lensobject.naxis1*self.lensobject.naxis2  # lens plane size
        # projection data
        dij = self.d_ij(flat=True, composite=composite)
        # projection mapping
        Mij_p = None
        if cached:
            Mij_p = self._cache['inv_proj'][self.obj_index][self.model_index]
            self.r_max = self._cache['r_max'][self.obj_index][self.model_index]
            self.M_fullres = self._cache['M_fullres'][self.obj_index][self.model_index]
            self.r_fullres = self._cache['r_fullres'][self.obj_index][self.model_index]
            self.N_nil = self._cache['N_nil'][self.obj_index][self.model_index]
        if Mij_p is None:
            Mij_p = self.inv_proj_matrix()
            self._cache['inv_proj'][self.obj_index][self.model_index] = Mij_p.copy()
        if use_psf and self.psf is not None:
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
        if method == 'lsmr':
            Qij_p = sigmaM2 * Mij_p
            A = Mij_p.T * Qij_p
            b = dij * Qij_p
            dp = lsmr(A, b)[0]
        elif method == 'lsqr':
            Qij_p = sigmaM2 * Mij_p
            A = Mij_p.T * Qij_p
            b = dij * Qij_p
            dp = lsqr(A, b)[0]
        elif method == 'cgs':
            Qij_p = sigmaM2 * Mij_p
            A = Mij_p.T * Qij_p
            b = dij * Qij_p
            dp = cgs(A, b)[0]
        elif method == 'lgmres':
            Qij_p = sigmaM2 * Mij_p
            A = Mij_p.T * Qij_p
            b = dij * Qij_p
            dp = lgmres(A, b, atol=1e-05)[0]
        elif method == 'minres':
            Qij_p = sigmaM2 * Mij_p
            A = Mij_p.T * Qij_p
            b = dij * Qij_p
            dp = minres(A, b)[0]
        elif method == 'qmr':
            Qij_p = sigmaM2 * Mij_p
            A = Mij_p.T * Qij_p
            b = dij * Qij_p
            dp = qmr(A, b)[0]
        elif method == 'row_norm':
            Mij_p = row_norm(Mij_p, norm='l1', axis=0)
            dp = Mij_p.T * dij
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

    def reproj_d_ij(self, flat=True, from_cache=False, save_to_cache=False, **kwargs):
        """
        Solve the inverse projection problem to Mij_p * d_ij = d_p, where Mij_p and d_p are known

        Args:
            None

        Kwargs:
            method <str> - option to choose the minimizing method ['lsqr','lsmr','row_norm']
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
        cached = kwargs.get('cached', False)
        use_psf = kwargs.get('use_psf', True)
        dij = None
        if from_cache:
            dij = self._cache['reproj_d_ij'][self.obj_index][self.model_index]
        if dij is None:
            dp = self.d_p(flat=True, **kwargs)
            if cached:
                Mij_p = self._cache['inv_proj'][self.obj_index][self.model_index]
            else:
                Mij_p = self.inv_proj_matrix()
                self._cache['inv_proj'][self.obj_index][self.model_index] = Mij_p.copy()
            if use_psf and self.psf is not None:
                Mij_p = self.psf * Mij_p

            dij = Mij_p * dp

            if save_to_cache:
                self._cache['reproj_d_ij'][self.obj_index][self.model_index] = dij.copy()

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

    def residual_map(self, flat=False, **kwargs):
        """
        Evaluate the reprojection by calculating the residual map to the lens map data

        Args:
            None

        Kwargs:
            method <str> - option to choose the minimizing method ['lsqr','lsmr','row_norm']
            use_psf <bool> - use the PSF to smooth the projection matrix
            flat <bool> - return the flattened array
            cached <bool> - use a cached inverse projection matrix rather than computing it
            sigma <float/np.ndarray> - uncertainty map
            sigma2 <float/np.ndarray> - squared uncertainty map
            sigmaM2 <scipy.sparse.diags> - sparse diagonal matrix with inverse sigma2s
            composite <bool> - return the composite data array

        Return:
            residual <np.ndarray> - reprojection data based on the source reconstruction
        """
        data = self.lens_map(flat=flat, mask=True, composite=kwargs.get('composite', False))
        reproj = self.reproj_map(flat=flat, **kwargs)
        return data-reproj

    def reproj_chi2(self, data=None, reduced=False, nonzero_only=True,
                    noise=0, sigma=1, sigma2=None, sigmaM2=None,
                    **kwargs):
        """
        Evaluate the reprojection by calculating the absolute squared residuals to the data

        Args:
            None

        Kwargs:
            reduced <bool> - return the reduced chi^2 (i.e. chi2 divided by the number of pixels)
            nonzero_only <bool> - only sum over pixels which were actually reprojected
            method <str> - option to choose the minimizing method ['lsqr','lsmr','row_norm']
            use_psf <bool> - use the PSF to smooth the projection matrix
            from_cache <bool> - use the cached reprojected map
            save_to_cache <bool> - save the reprojected map to cache
            cached <bool> - use a cached inverse projection matrix rather than computing it
            sigma <float/np.ndarray> - sigma for chi2 calculation
            sigma2 <float/np.ndarray> - squared sigma for the chi2 calculation
            sigmaM2 <scipy.sparse.diags> - sparse diagonal matrix with inverse sigma2s
            noise <float/np.ndarray> - noise added to the data
            composite <bool> - return the composite data array

        Return:
            residual <float> - the residual between data and reprojection
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
        # calculate synthetic image and residual map
        reproj = self.reproj_map(sigma=sigma, sigma2=sigma2, sigmaM2=sigmaM2, **kwargs)
        if data is None:
            composite = kwargs.get('composite', False)
            data = self.lens_map(flat=False, composite=composite)
            data = data + noise
        residual = data-reproj
        if nonzero_only:
            msk = reproj != 0
            chi2 = np.sum(residual[msk]**2/sigma2[msk])
        else:
            chi2 = np.sum(residual**2/sigma2)
        if reduced:
            N = self.lens_map().size
            chi2 = chi2 / (N - self.N_nil)
        return chi2

    def mask_plot(self, data=None, bg=None):
        """
        Plot a background in grayscale with masked data in color

        Args:
            None

        Kwargs:
            data <np.ndarray> - data to be masked with shape (N, M, i={0,1,4})
            bg <np.ndarray> - background image to be put in grayscale

        Return:
            img <np.ndarray> - masked data rgba array
        """
        if data is None:
            data = self.lensobject.data
            if isinstance(self.gleamobject, (SkyPatch, MultiLens)):
                if self.gleamobject.composite:
                    data = self.gleamobject.composite
        if bg is None:
            bg = data
        img = glmrgb.grayscale(bg)
        rgba = glmrgb.rgba(data)
        img[self.mask] = rgba[self.mask]
        plt.imshow(img)
        return img


def synth_filter(statefile=None, gleamobject=None, reconsrc=None, psf_file=None,
                 percentiles=[],
                 reduced=False, nonzero_only=True, method='lsmr', use_psf=True,
                 from_cache=True, cached=True, save_to_cache=True,
                 noise=0, sigma=1, sigma2=None, sigmaM2=None,
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
        method <str> - option to choose the minimizing method ['lsqr','lsmr','row_norm']
        use_psf <bool> - use the PSF to smooth the projection matrix
        from_cache <bool> - use the cached reprojected map
        cached <bool> - use a cached inverse projection matrix rather than computing it
        save_to_cache <bool> - save the reprojected map to cache
        sigma <float/np.ndarray> - sigma for chi2 calculation
        sigma2 <float/np.ndarray> - squared sigma for the chi2 calculation
        sigmaM2 <scipy.sparse.diags> - sparse diagonal matrix with inverse sigma2s
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
            recon_src.calc_psf(psf_file)
    else:
        recon_src = reconsrc
        if psf_file is not None and os.path.exists(psf_file):
            recon_src.calc_psf(psf_file, )

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
                                     sigma2=sigma2, sigmaM2=sigmaM2,
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
                   cached=True, method='lsmr', use_psf=True,
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
        method <str> - option to choose the minimizing method ['lsqr','lsmr','row_norm']
        use_psf <bool> - use the PSF to smooth the projection matrix
        from_cache <bool> - use the cached reprojected map
        cached <bool> - use a cached inverse projection matrix rather than computing it
        save_to_cache <bool> - save the reprojected map to cache
        sigma <float/np.ndarray> - sigma for chi2 calculation
        sigma2 <float/np.ndarray> - squared sigma for the chi2 calculation
        sigmaM2 <scipy.sparse.diags> - sparse diagonal matrix with inverse sigma2s
        N_total <int> - the total size of loop range
        stdout_flush <bool> - flush stdout and update line in verbose mode
        verbose <bool> - verbose mode; print command line statements

    Return:
        delta <float> - the chi2 sum of a single model
    """
    reconsrc.chmdl(index)
    try:
        delta = reconsrc.reproj_chi2(data=data, reduced=reduced, nonzero_only=nonzero_only,
                                     from_cache=from_cache, save_to_cache=save_to_cache,
                                     sigma=sigma, sigma2=sigma2, sigmaM2=sigmaM2,
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
                    reduced=False, nonzero_only=True, method='lsmr', use_psf=True,
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
        method <str> - option to choose the minimizing method ['lsqr','lsmr','row_norm']
        use_psf <bool> - use the PSF to smooth the projection matrix
        from_cache <bool> - use the cached reprojected map
        cached <bool> - use a cached inverse projection matrix rather than computing it
        save_to_cache <bool> - save the reprojected map to cache
        noise <np.ndarray> - noise with is artificially added to data
        sigma <float/np.ndarray> - sigma for chi2 calculation
        sigma2 <float/np.ndarray> - squared sigma for the chi2 calculation
        sigmaM2 <scipy.sparse.diags> - sparse diagonal matrix with inverse sigma2s
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
            recon_src.calc_psf(psf_file)
    else:
        recon_src = reconsrc
        if psf_file is not None and os.path.exists(psf_file):
            recon_src.calc_psf(psf_file)

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
