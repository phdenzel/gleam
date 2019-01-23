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
import numpy as np

from gleam.skyf import SkyF
from gleam.lensobject import LensObject
from gleam.skypatch import SkyPatch
from gleam.multilens import MultiLens
from gleam.glass_interface import glass_renv
glass = glass_renv()

__all__ = ['ReconSrc']


###############################################################################
class ReconSrc(object):
    """
    Framework for source reconstruction
    """
    def __init__(self, gleamobject, statefile, M=20, mask_keys=['polygon'], verbose=False):
        """
        Initialize

        Args:
            gleamobject <GLEAM object> - a GLEAM object instance with .fits file's data
            statefile <str> - filename of a GLASS .state file

        Kwargs:
            M <int> - source plane pixel radius; the total source plane will be (2*M+1) x (2*M+1)

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
        self.gls = glass.glcmds.loadstate(statefile)
        self.model_index = 0

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

    # @property
    # def plane(self):
    #     """
    #     The source plane data array

    #     Args/Kwargs:
    #         None

    #     Return:
    #         plane <np.ndarray> - the source plane data array
    #     """
    #     if not hasattr(self, '_plane'):
    #         self._plane = np.zeros((self.N, self.N), dtype=list)
    #     if self._plane.shape[0] > self.N:
    #         self._plane = self._plane[:self.N, :self.N]
    #     elif self._plane.shape[0] < self.N:
    #         self._plane = np.pad(self._plane, int((self.N-self._plane.shape)/2),
    #                              mode='constant', constant_values=0)
    #     return self._plane

    # @plane.setter
    # def plane(self, plane):
    #     """
    #     (Re)set the source plane data array

    #     Args:
    #         plane <np.ndarray> - the source plane data array

    #     Kwargs/Return:
    #         None
    #     """
    #     self._plane = plane
    #     if self._plane.shape[0] != self.N:
    #         raise ValueError("Shape mismatch: source plane needs shape {0}x{0}".format(self.N))

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

    def inv_proj_matrix(self):
        """
        The projection matrix to get from the image plane to the source plane

        Args:
        Kwargs:
        Return:

        d_p = M^ij_p * dij
        """
        # src plane
        NxN = self.N*self.N
        p = range(NxN)
        # lens plane
        LxL = self.lensobject.naxis1*self.lensobject.naxis2
        ij = range(LxL)
        # ij_masked = [self.mask]
        theta = np.array([self.lensobject.theta(i) for i in ij])
        #
        # d_source = np.array(map(self.delta_beta, theta))
        d_source = self.delta_beta(theta)
        print(d_source.shape)
        
        # inverse projection matrix
        M_ij = np.zeros((LxL, NxN), dtype=int)


        # DEBUG
        print(self.mask.shape)

        return M_ij

    def delta_beta(self, theta):
        """
        Delta beta from lens equation using the current GLASS model

        Args:
            theta <list/tuple/complex> - 2D angular position coordinates
            model_object

        Kwargs:
            None

        Return:
            delta_beta <list/tuple> - beta from lens equation
        """
        if not isinstance(theta, (list, tuple)) and len(theta) == 2:
            ang_pos = complex(*theta)
        elif isinstance(theta, np.ndarray) and len(theta) > 2:
            ang_pos = np.empty(theta.shape[:-1], dtype=np.complex)
            ang_pos.real = theta[..., 0]
            ang_pos.imag = theta[..., 1]
        obj, data = self.gls.models[self.model_index+1]['obj,data'][0]
        src = data['src'][0]
        zcap = obj.sources[0].zcap
        print(ang_pos.shape)
        print((src - ang_pos).shape)
        return src - ang_pos + obj.basis.deflect(ang_pos, data) / zcap

    # @staticmethod
    # def mapping(model, theta, srcplane):
    #     """
    #     Collect pixel indices from the lens plane mapped onto the source plane
    #     for a specific model from a GLASS ensemble and theta coordinates

    #     Args:
    #         model
    #     """
    #     obj, data = model['obj,data'][0]
    #     src = data['src'][0]
    #     zcap = obj.sources[0].zcap

    #     # lensing equation
    #     def delta_beta(theta):
    #         return src - theta + obj.basis.deflect(theta, data) / zcap

    #     d_source = np.array(map(delta_beta, theta))
    #     r = max(np.max(np.abs(np.real(d_source))),
    #             np.max(np.abs(np.imag(d_source))))
    #     X = np.int32(np.floor(M*(1+np.real(d_source)/r)+.5))
    #     Y = np.int32(np.floor(M*(1+np.imag(d_source)/r)+.5))
    # collect on source plane

    # srcplane = np.zeros((2*M+1, 2*M+1), dtype=list)
    # for (x, y), _ in np.ndenumerate(srcplane):
    #     srcplane[x, y] = []
    # for i, _ in enumerate(d_source):
    #     x, y = (X[i], Y[i])
    #     srcplane[x, y].append(i)
    # return srcplane

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
        import gleam.utils.rgb_map as glmrgb
        import matplotlib.pyplot as plt
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
    statefiles = ["/Users/phdenzel/adler/H1S0A0B90G0.state",
                  "/Users/phdenzel/adler/H1S1A0B90G0.state",
                  "/Users/phdenzel/adler/H2S1A0B90G0.state",
                  "/Users/phdenzel/adler/H2S2A0B90G0.state",
                  "/Users/phdenzel/adler/H2S7A0B90G0.state",
                  "/Users/phdenzel/adler/H3S0A0B90G0.state",
                  "/Users/phdenzel/adler/H3S1A0B90G0.state",
                  "/Users/phdenzel/adler/H4S3A0B0G90.state",
                  "/Users/phdenzel/adler/H10S0A0B90G0.state",
                  "/Users/phdenzel/adler/H13S0A0B90G0.state",
                  "/Users/phdenzel/adler/H23S0A0B90G0.state",
                  "/Users/phdenzel/adler/H30S0A0B90G0.state",
                  "/Users/phdenzel/adler/H36S0A0B90G0.state",
                  "/Users/phdenzel/adler/H160S0A90B0G0.state",
                  "/Users/phdenzel/adler/H234S0A0B90G0.state"]
    # for i in range(len(jsons)):
    i = 4
    with open(jsons[i]) as f:
        ml = MultiLens.from_json(f)
    recon_src = ReconSrc(ml, statefiles[i], verbose=1)
    recon_src.chobj()
    recon_src.inv_proj_matrix()
    # import time
    # ti = time.time()
    # for m in recon_src.gls.models[:10]:
    #     recon_src.inv_proj_matrix()
    #     recon_src.chmdl()
    # tf = time.time()
    # print(tf-ti)

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
