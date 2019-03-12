#!/usr/bin/env python
"""
@author: phdenzel

Climb every peak in search for lens and source candidates

TODO:
    - add a main method
    - complete tests
"""
###############################################################################
# Imports
###############################################################################
import sys
import os
import copy
import numpy as np
from scipy.sparse import lil_matrix as sparse_matrix
from scipy.sparse.linalg import lsqr, lsmr
from sklearn.preprocessing import normalize as row_norm
import matplotlib.pyplot as plt

from gleam.skyf import SkyF
from gleam.lensobject import LensObject
from gleam.skypatch import SkyPatch
from gleam.multilens import MultiLens
# from gleam.utils.sparse_funcs import (inplace_csr_row_normalize)
import gleam.utils.rgb_map as glmrgb
from gleam.glass_interface import glass_renv, filter_env
glass = glass_renv()

__all__ = ['ReconSrc']


###############################################################################
class ReconSrc(object):
    """
    Framework for source reconstruction
    """
    def __init__(self, gleamobject, glsfile, M=20, mask_keys=['polygon'], verbose=False):
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
        # grab first lensobject if there are more than one
        self.lensobject = self.lens_objects[0]
        self.gls = glass.glcmds.loadstate(glsfile)
        self.model_index = -1

        # source plane
        self.M = M
        self.mask_keys = mask_keys
        # self.plane = np.zeros((self.N, self.N), dtype=list)

        # some verbosity
        if verbose:
            print(self.__v__)

    def __str__(self):
        return "ReconSrc{}".format(self.plane.shape)

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

    def chobj(self, index=None):
        """
        Change the lensobject; moves to the next lensobject in the list  by default

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
            index = self.model_index + 1
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
            return self.gls.ensemble_average['obj,data'][0]
        else:
            envcpy = filter_env(self.gls, selection)
            envcpy.make_ensemble_average()
            return envcpy.ensemble_average['obj,data'][0]

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
            obj, data = self.gls.models[self.model_index]['obj,data'][0]
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
        if mask:
            if np.any(self.mask):
                if flat:
                    msk = self.mask.flatten()
                else:
                    msk = self.mask
                dij[~msk] = 0
        return dij

    def inv_proj_matrix(self, asarray=False):
        """
        The projection matrix to get from the image plane to the source plane

        Args:
            None

        Kwargs:
            asarray <bool> - if True, return matrix as array, otherwise as scipy.sparse matrix

        Return:
            Mij_p <scipy.sparse.csc.csc_matrix> - inverse projection map from image to source plane
        """
        # # src plane
        NxN = self.N*self.N
        # p = range(NxN)
        # # lens plane
        LxL = self.lensobject.naxis1*self.lensobject.naxis2
        ij = range(LxL)
        if np.any(self.mask):
            ij_masked = [self.lensobject.yx2idx(ix, iy) for ix, iy in zip(*np.where(self.mask))]
        else:
            ij_masked = ij  # in case self.mask is not available
        theta = np.array([self.lensobject.theta(i) for i in ij_masked])  # grid coords [arcsec]
        dbeta = self.delta_beta(theta)  # distance from source center on source plane [arcsec]
        r = np.nanmax(np.abs(dbeta))  # maximal distance [arcsec]
        xy = np.int16(np.floor(self.M*(1+dbeta/r))+.5)
        # # init inverse projection matrix
        Mij_p = sparse_matrix((LxL, NxN), dtype=np.int16)
        for i, (x, y) in enumerate(xy):
            ij_idx = ij_masked[i]
            p_idx = self.lensobject.yx2idx(y, x, cols=self.N)
            Mij_p[ij_idx, p_idx] = 1

        Mij_p = row_norm(Mij_p, norm='l1', axis=0)
        # Mij_p.data = Mij_p.data / np.repeat(
        #     np.add.reduceat(Mij_p.data, Mij_p.indptr[:-1]), np.diff(Mij_p.indptr))
        if asarray:
            return Mij_p.toarray()
        return Mij_p

    def d_p(self, flat=True, composite=False):
        """
        Recover the source plane data using the inverse projection matrix: d_p = M^ij_p * d_ij

        Args:
            None

        Kwargs:
            flat <bool> - return the flattened array
            composite <bool> - return the composite data array

        Return:
            dp <np.ndarray> - the source plane data
        """
        # LxL = self.lensobject.naxis1*self.lensobject.naxis2  # lens plane size
        # projection data
        dij = self.d_ij(flat=True, composite=composite)
        Mij_p = self.inv_proj_matrix()
        dp = dij * Mij_p
        if not flat:
            dp = dp.reshape((self.N, self.N))
        return dp

    def plane_map(self, flat=False, composite=False):
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
        return self.d_p(flat=flat, composite=composite)

    def proj_matrix(self, asarray=False):
        """
        The projection matrix to get from the source plane to the image plane

        Args:
            None

        Kwargs:
            asarray <bool> - if True, return matrix as array, otherwise as scipy.sparse matrix

        Return:
            Mp_ij <scipy.sparse.csc.csc_matrix> - projection map from source to image plane
        """
        Mij_p = self.inv_proj_matrix(asarray=asarray)
        # TODO
        # d_p = 1
        # Mp_ij = splin_pinv(Mij_p.toarray(), d_p)
        # TODO
        Mp_ij = Mij_p
        return Mp_ij

    def reproj_d_ij(self, method='lsqr', flat=True, composite=False):
        """
        Solve the inverse projection problem to Mij_p * d_ij = d_p, where Mij_p and d_p are known

        Args:
            None

        Kwargs:
            method <str> - method which solves the inverse problem [lsqr, lsmr,]
            flat <bool> - return the flattened array
            composite <bool> - return the composite data array

        Return:
            dij <np.ndarray> - reprojection data based on the source reconstruction
        """
        dp = self.d_p(flat=True, composite=composite)
        Mij_p = self.inv_proj_matrix()
        if method == 'lsqr':
            dij = lsqr(Mij_p.T, dp)[0]
        elif method == 'lsmr':
            dij = lsmr(Mij_p.T, dp)[0]
        if not flat:
            dij = dij.reshape((self.lensobject.naxis1, self.lensobject.naxis2))
        return dij

    def reproj_map(self, method='lsqr', flat=False, composite=False):
        """
        Solve the inverse projection problem to Mij_p * d_ij = d_p, where Mij_p and d_p are known

        Args:
            None

        Kwargs:
            method <str> - method which solves the inverse problem [lsqr, lsmr,]
            flat <bool> - return the flattened array
            composite <bool> - return the composite data array

        Return:
            dij <np.ndarray> - reprojection data based on the source reconstruction
        """
        return self.reproj_d_ij(method=method, flat=flat, composite=composite)

    def synth_map(self, method='lsqr', flat=True, composite=False):
        """
        TODO
        """
        pass

    def residual_map(self, method='lsqr', flat=False, composite=False):
        """
        Evaluate the reprojection by calculating the residual map to the lens map data

        Args:
            None

        Kwargs:
            method <str> - method which solves the inverse problem [lsqr, lsmr,]
            flat <bool> - return the flattened array
            composite <bool> - return the composite data array

        Return:
            residual <np.ndarray> - reprojection data based on the source reconstruction
        """
        reproj = self.reproj_map(method=method, flat=flat, composite=composite)
        data = self.lens_map(flat=flat, mask=True, composite=composite)
        return data-reproj

    def reproj_residual(self, method='lsqr', composite=False):
        """
        Evaluate the reprojection by calculating the absolute squared residuals to the data

        Args:
            None

        Kwargs:
            method <str> - method which solves the inverse problem [lsqr, lsmr,]
            composite <bool> - return the composite data array

        Return:
            residual <float> - the residual between data and reprojection
        """
        residual = self.residual_map(method=method, flat=True, composite=composite)
        return np.sum(residual**2)

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
        plt.show()
        return img


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
        with open(jsons[i]) as f:
            ml = MultiLens.from_json(f)
        # recon_src = ReconSrc(ml, statefiles[i], M=80, verbose=1)
        recon_src = ReconSrc(ml, statefiles[i], M=20, verbose=1)
        recon_src.chobj()
        import time
        ti = time.time()
        print(recon_src.plane_map())
        # print(all(self.lensobject.data.reshape(LxL) == np.array([self.lensobject.data[self.lensobject.idx2yx(i)] for i in range(LxL)])))
        # nevertheless, as data vector use: np.array([self.lensobject.data[self.lensobject.idx2yx(i)] for i in range(LxL)])
        tf = time.time()
        print(tf-ti)

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
