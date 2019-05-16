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
from scipy.sparse import diags as sparse_Dmatrix
from scipy.sparse.linalg import lsqr, lsmr
from sklearn.preprocessing import normalize as row_norm
import matplotlib.pyplot as plt

from gleam.skyf import SkyF
from gleam.lensobject import LensObject
from gleam.skypatch import SkyPatch
from gleam.multilens import MultiLens
# from gleam.utils.sparse_funcs import (inplace_csr_row_normalize)
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
    def __init__(self, gleamobject, glsfile, M=20, mask_keys=[], verbose=False):
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
        self.M = M  # full resolution pixel radius, sets N
        self.N_AA = self.N
        self.r_antialias = None
        self.N_nil = 0  # skipped pixels during antialiasing
        self.plane = np.zeros((self.N, self.N), dtype=list)

        # some cache containers
        self._cache = {'inv_proj': {k: {i: None for i in range(len(self.gls.models))}
                                    for k in range(len(self.gls.objects))},
                       'N_nil': {k: {i: None for i in range(len(self.gls.models))}
                                 for k in range(len(self.gls.objects))},
                       'reproj_d_ij': {k: {i: None for i in range(len(self.gls.models))}
                                       for k in range(len(self.gls.objects))}}

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
        return ['lensobject', 'gls', 'mask', 'M', 'N']

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
    def N(self):
        """
        The source plane row and column size, derived from the pixel radius

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

        Note:
            - even integers will increase by one
        """
        self.M = int(N/2)

    @property
    def r_antialias(self):
        """
        The antialiasing maprad, i.e. r_AA

        Args/Kwargs:
            None

        Return:
            r_antialias <float> - antialiasing maprad radius
        """
        if self._r_antialias is None:
            _, r = self.srcgrid_mapping(pixrad=20, mask=self.image_mask())
            self._r_antialias = r
        return self._r_antialias

    @r_antialias.setter
    def r_antialias(self, r_AA):
        """
        Setter for the antialiasing pixrad

        Args:
            r_AA <float> - antialiasing radius in units of arcsec
        """
        self._r_antialias = r_AA

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
        # self.plane = np.zeros((self.N, self.N), dtype=list)

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
        # self.plane = np.zeros((self.N, self.N), dtype=list)

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
        # self.plane = np.zeros((self.N, self.N), dtype=list)

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

    def update_cache(self, cache, variables=['inv_proj', 'N_nil', 'reproj_d_ij'], verbose=False):
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

    def flush_cache(self, variables=['inv_proj', 'N_nil', 'reproj_d_ij'], verbose=False):
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
            self._cache[v] = {k: {i: None for i in range(len(self.gls.models))}
                              for k in range(len(self.gls.objects))}

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

    def srcgrid_mapping(self, pixrad=None, mask=None):
        """
        Calculate the mapping from image to source plane for a given model

        Args:
            None

        Kwargs:
            mask <np.ndarray(bool)> - boolean mask for image plane pixel selection

        Return:
            xy <np.ndarray(int)> - pixel mapping coordinates from image to source plane
        """
        LxL = self.lensobject.naxis1*self.lensobject.naxis2
        ij = range(LxL)
        if pixrad is None:
            pixrad = self.M
        if mask is not None:
            ij = [self.lensobject.yx2idx(ix, iy) for ix, iy in zip(*np.where(mask))]
        theta = np.array([self.lensobject.theta(i) for i in ij])  # grid coords [arcsec]
        dbeta = self.delta_beta(theta)  # distance components from center on source plane [arcsec]
        r_max = np.nanmax(np.abs(dbeta))  # maximal distance [arcsec]
        xy = np.int16(np.floor(pixrad*(1+dbeta/r_max))+.5)
        return (xy, r_max)

    def inv_proj_matrix(self, antialias=True, asarray=False):
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
        if np.any(self.mask):
            ij = [self.lensobject.yx2idx(ix, iy) for ix, iy in zip(*np.where(self.mask))]
            xy, self.r_max = self.srcgrid_mapping(mask=self.mask)
        else:
            xy, self.r_max = self.srcgrid_mapping(mask=None)
        # # src plane
        N = self.N
        NxN = N*N
        N_0 = N//2
        self.N_AA = N
        NxN_AA = NxN
        N_l, N_r = N_0-self.N_AA//2, N_0+self.N_AA//2
        self.N_nil = 0
        if antialias:
            # f_AA = int(self.r_max/self.r_antialias)
            f_AA = int(self.r_max/self.r_antialias)
            self.N_AA = int(N//f_AA)
            NxN_AA = self.N_AA*self.N_AA
            N_l, N_r = N_0-self.N_AA//2, N_0+self.N_AA//2
        # # init inverse projection matrix
        Mij_p = sparse_Lmatrix((LxL, NxN_AA), dtype=np.int16)
        for i, (x, y) in enumerate(xy):
            if (N_l < x < N_r) and (N_l < y < N_r):
                x -= N_l
                y -= N_l
            else:
                self.N_nil += 1
                continue
            p_idx = self.lensobject.yx2idx(y, x, cols=self.N_AA)
            if p_idx >= NxN_AA:  # should never be True (simply there for stability)
                print("Projection discovered a pixel out of range!")
                self.N_nil += 1
                continue
            ij_idx = ij[i]
            Mij_p[ij_idx, p_idx] = 1
        self._cache['N_nil'][self.obj_index][self.model_index] = self.N_nil
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

    def d_p(self, method='lsqr', antialias=True, flat=True, cached=False,
            sigma=1, sigma2=None, sigmaM2=None, composite=False):
        """
        Recover the source plane data by minimizing
        (M^q_i.T * sigma_i^-2) M^i_p d_p = (M^q_i.T * sigma_i^-2) d_i

        Args:
            None

        Kwargs:
            method <str> - option to choose the minimizing method ['lsqr','lsmr','row_norm']
            antialias <bool> - antialias the source plane (runs faster)
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
            self.N_nil = self._cache['N_nil'][self.obj_index][self.model_index]
        if Mij_p is None:
            Mij_p = self.inv_proj_matrix(antialias=antialias)
            self._cache['inv_proj'][self.obj_index][self.model_index] = Mij_p.copy()
        # uncertainty
        if sigma2 is None:
            sigma2 = sigma*sigma
        if sigmaM2 is None:
            sigmaM2 = 1./sigma2
            if hasattr(sigmaM2, '__len__') and len(sigmaM2.shape) > 1:
                sigmaM2 = np.array(
                    [sigmaM2[self.lensobject.idx2yx(i)] for i in range(sigmaM2.size)])
                sigmaM2 = sparse_Dmatrix(sigmaM2)
        if method == 'lsqr':
            Qij_p = sigmaM2 * Mij_p
            A = Mij_p.T * Qij_p
            b = dij * Qij_p
            dp = lsqr(A, b)[0]
        elif method == 'lsmr':
            Qij_p = sigmaM2 * Mij_p
            A = Mij_p.T * Qij_p
            b = dij * Qij_p
            dp = lsmr(A, b)[0]
        elif method == 'row_norm':
            Mij_p = row_norm(Mij_p, norm='l1', axis=0)
            dp = Mij_p.T * dij
        N = self.N
        if antialias:
            if hasattr(self, 'r_max'):
                f_AA = int(self.r_max/self.r_antialias)
                N = int(N//f_AA)
            else:
                N = int(np.sqrt(dp.size))
        if not flat:
            dp = dp.reshape((N, N))
        return dp

    def plane_map(self, flat=False, **kwargs):
        """
        Recover the source plane data map using the inverse projection matrix: d_p = M^ij_p * d_ij

        Args:
            None

        Kwargs:
            flat <bool> - return the flattened array
            composite <bool> - return the composite data array

        Return:
            dp <np.ndarray> - the source plane data
        """
        return self.d_p(flat=flat, **kwargs)

    def reproj_d_ij(self, flat=True, **kwargs):
        """
        Solve the inverse projection problem to Mij_p * d_ij = d_p, where Mij_p and d_p are known

        Args:
            None

        Kwargs:
            method <str> - option to choose the minimizing method ['lsqr','lsmr','row_norm']
            antialias <bool> - antialias the source plane (runs faster)
            flat <bool> - return the flattened array
            cached <bool> - use a cached inverse projection matrix rather than computing it
            sigma <float/np.ndarray> - uncertainty map
            sigma2 <float/np.ndarray> - squared uncertainty map
            composite <bool> - return the composite data array

        Return:
            dij <np.ndarray> - reprojection data based on the source reconstruction
        """
        cached = kwargs.get('cached', False)
        antialias = kwargs.get('antialias', True)
        dij = None
        if cached:
            dij = self._cache['reproj_d_ij'][self.obj_index][self.model_index]
        if dij is None:
            dp = self.d_p(flat=True, **kwargs)
            if cached:
                Mij_p = self._cache['inv_proj'][self.obj_index][self.model_index]
            else:
                Mij_p = self.inv_proj_matrix(antialias=antialias)
                self._cache['inv_proj'][self.obj_index][self.model_index] = Mij_p.copy()

            dij = Mij_p * dp

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
            antialias <bool> - antialias the source plane (runs faster)
            flat <bool> - return the flattened array
            cached <bool> - use a cached inverse projection matrix rather than computing it
            sigma <float/np.ndarray> - uncertainty map
            sigma2 <float/np.ndarray> - squared uncertainty map
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
            antialias <bool> - antialias the source plane (runs faster)
            flat <bool> - return the flattened array
            cached <bool> - use a cached inverse projection matrix rather than computing it
            sigma <float/np.ndarray> - uncertainty map
            sigma2 <float/np.ndarray> - squared uncertainty map
            composite <bool> - return the composite data array

        Return:
            residual <np.ndarray> - reprojection data based on the source reconstruction
        """
        reproj = self.reproj_map(flat=flat, **kwargs)
        data = self.lens_map(flat=flat, mask=True, composite=kwargs.get('composite', False))
        return data-reproj

    def reproj_chi2(self, data=None, reduced=True, mask=True,
                    noise=0, sigma=1, sigma2=None, sigmaM2=None,
                    **kwargs):
        """
        Evaluate the reprojection by calculating the absolute squared residuals to the data

        Args:
            None

        Kwargs:
            reduced <bool> - return the reduced chi^2 (i.e. chi2 divided by the number of pixels)
            method <str> - option to choose the minimizing method ['lsqr','lsmr','row_norm']
            antialias <bool> - antialias the source plane (runs faster)
            cached <bool> - use a cached inverse projection matrix rather than computing it
            sigma <float/np.ndarray> - sigma for chi2 calculation
            sigma2 <float/np.ndarray> - squared sigma for the chi2 calculation
            noise <float/np.ndarray> - noise added to the data
            composite <bool> - return the composite data array

        Return:
            residual <float> - the residual between data and reprojection
        """
        reproj = self.reproj_map(sigma=sigma, sigma2=sigma2, **kwargs)
        if data is None:
            composite = kwargs.get('composite', False)
            data = self.lens_map(flat=False, composite=composite)
            data = data + noise
        residual = data-reproj
        if sigma2 is None:
            sigma2 = sigma*sigma
        if sigmaM2 is None:
            sigmaM2 = 1./sigma2
            if hasattr(sigmaM2, '__len__') and len(sigmaM2.shape) > 1:
                sigmaM2 = np.array(
                    [sigmaM2[self.lensobject.idx2yx(i)] for i in range(sigmaM2.size)])
                sigmaM2 = sparse_Dmatrix(sigmaM2)
        if mask:
            msk = reproj != 0
            chi2 = np.sum(residual[msk]**2/sigma2[msk])
        else:
            chi2 = np.sum(residual**2/sigma2)
        N = self.lens_map().size
        N_mask = N_src = N_empty = 0
        if reduced:
            if np.any(self.mask):
                N_mask = N - np.count_nonzero(self.mask)
            else:
                N_mask = 0
            N_src = self.N_AA * self.N_AA
            N_empty = self.N_nil
            chi2 = (chi2 - N_empty) / (N-N_mask - N_empty - N_src)
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


def synth_filter(statefile=None, gleamobject=None, reconsrc=None, percentiles=[],
                 reduced=False, noise=0, sigma=1, sigma2=None, sigmaM2=None, N_models=None,
                 save=False, return_obj=False,
                 verbose=False):
    """
    Filter a GLASS state file using GLEAM's source reconstruction feature

    Args:
        statefile <str> - the GLASS statefile
        gleamobject <GLEAM object> - a GLEAM object instance with .fits file's data

    Kwargs:
        percentiles <list(float)> - percentages the filter retains
        reduced <bool> - return the reduced chi^2 (i.e. chi2 divided by the number of pixels)
        sigma <float/np.ndarray> - sigma for chi2 calculation
        sigma2 <float/np.ndarray> - squared sigma for the chi2 calculation
        save <bool> - save the filtered states automatically
        return_obj <bool> - return the object with all reprojections cached instead
        verbose <bool> - verbose mode; print command line statements

    Return:
      if return_obj:
        recon_src <ReconSrc object> - the object with all reprojections cached
      else:
        filtered_states <list(glass.Environment object)> - the filtered states ready for export
        selected <list(int)> - index list of selected models
        residuals <list(float)> - list of chi2s
    """
    if verbose and statefile is not None:
        print(statefile)

    if reconsrc is None:
        if gleamobject is not None and statefile is not None:
            recon_src = ReconSrc(gleamobject, statefile, M=20, verbose=verbose)
        else:
            return None
    else:
        recon_src = reconsrc

    residuals = []
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
        delta = recon_src.reproj_chi2(data=data, reduced=reduced,
                                      sigma2=sigma2, sigmaM2=sigmaM2,
                                      method='lsqr', antialias=True, cached=True)
        residuals.append(delta)
        if verbose:
            message = "{:4d} / {:4d}: {:4.8f}\r".format(i+1, N_models, delta)
            sys.stdout.write(message)
            sys.stdout.flush()

    if verbose:
        print("Number of residual models: {}".format(len(residuals)))

    rhi = [np.percentile(residuals, p, interpolation='higher') for p in percentiles]
    rlo = [0 for r in rhi]
    selected = [[i for i, r in enumerate(residuals) if rh > r > rl] for rh, rl in zip(rhi, rlo)]

    filtered = [filter_env(recon_src.gls, s) for s in selected]

    if save:
        dirname = os.path.dirname(statefile)
        basename = ".".join(os.path.basename(statefile).split('.')[:-1])
        saves = [dirname+'/'+basename+'_synthf{}.state'.format(p) for p in percentiles]
        for f, s in zip(filtered, saves):
            export_state(f, name=s)

    if return_obj:
        return recon_src
    return filtered, selected, residuals


def eval_residuals(index, reconsrc, data=None,
                   reduced=True, noise=0, sigma=1, sigma2=None, sigmaM2=None,
                   N_total=0, verbose=False):
    """
    Helper function to evaluate the residual of a single model within a multiprocessing loop

    Args:
        reconsrc <ReconSrc object> - pass
        index <int> - index of the model to analyse

    Kwargs:
        reduced <bool> - return the reduced chi^2 (i.e. chi2 divided by the number of pixels)
        sigma <float/np.ndarray> - sigma for chi2 calculation
        sigma2 <float/np.ndarray> - squared sigma for the chi2 calculation
        N_total <int> - the total size of loop range
        verbose <bool> - verbose mode; print command line statements

    Return:
        delta <float> - the residual
    """
    reconsrc.chmdl(index)
    try:
        delta = reconsrc.reproj_chi2(data=data, reduced=reduced,
                                     noise=0, sigma=sigma, sigma2=sigma2, sigmaM2=None,
                                     method='lsqr', antialias=True, cached=True)
        if verbose:
            message = "{:4d} / {:4d}: {:4.4f}\r".format(index+1, N_total, delta)
            sys.stdout.write(message)
            sys.stdout.flush()
        return reconsrc._cache, delta
    except KeyboardInterrupt:
        raise KeyboardInterruptError()


def synth_filter_mp(statefile=None, gleamobject=None, reconsrc=None, percentiles=[],
                    nproc=2,
                    reduced=False, noise=0, sigma=1, sigma2=None, sigmaM2=None, N_models=None,
                    save=False, return_obj=False,
                    verbose=False):
    """
    Filter a GLASS state file using GLEAM's source reconstruction feature

    Args:
        statefile <str> - the GLASS statefile
        gleamobject <GLEAM object> - a GLEAM object instance with .fits file's data

    Kwargs:
        percentiles <list(float)> - percentages the filter retains
        reduced <bool> - return the reduced chi^2 (i.e. chi2 divided by the number of pixels)
        sigma <float/np.ndarray> - sigma for chi2 calculation
        sigma2 <float/np.ndarray> - squared sigma for the chi2 calculation
        save <bool> - save the filtered states automatically
        verbose <bool> - verbose mode; print command line statements

    Return:
        filtered_states <list(glass.Environment object)> - the filtered states ready for export
        selected <list(int)> - index list of selected models
        residuals <list(float)> - list of chi2s
    """
    if verbose and statefile is not None:
        print(statefile)

    if reconsrc is None:
        if gleamobject is not None and statefile is not None:
            recon_src = ReconSrc(gleamobject, statefile, M=20, verbose=verbose)
        else:
            return None
    else:
        recon_src = reconsrc

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
    residuals = [None]*N_models
    try:
        f = partial(eval_residuals, reconsrc=recon_src, data=data,
                    reduced=reduced, sigma2=sigma2, sigmaM2=sigmaM2,
                    N_total=N_models, verbose=verbose)
        output = pool.map(f, range(N_models))
        residuals = [o[1] for o in output]
        caches = [o[0] for o in output]
        for c in caches:
            recon_src.update_cache(c)
        pool.clear()
    except KeyboardInterrupt:
        pool.terminate()

    if verbose:
        print("Number of residual models: {}".format(len(residuals)))

    rhi = [np.percentile(residuals, p, interpolation='higher') for p in percentiles]
    rlo = [0 for r in rhi]
    selected = [[i for i, r in enumerate(residuals) if rh > r > rl] for rh, rl in zip(rhi, rlo)]

    filtered = [filter_env(recon_src.gls, s) for s in selected]

    if save:
        dirname = os.path.dirname(statefile)
        basename = ".".join(os.path.basename(statefile).split('.')[:-1])
        saves = [dirname+'/'+basename+'_synthf{}.state'.format(p) for p in percentiles]
        for f, s in zip(filtered, saves):
            export_state(f, name=s)

    if return_obj:
        return recon_src
    return filtered, selected, residuals


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
