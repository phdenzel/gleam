#!/usr/bin/env python
"""
@author: phdenzel

Elliptical 2D Sersic model profile

Base parameters (see modelbase.py):
    - x, y: center coordinates
    - phi: position angle
    - e: ellipticity
    - I_0: intensity in the center
    - c_0: box parameter
Model-specific parameters:
    - n: Sersic index
    - r_s: half-light radius
    - b_n: internal non-free parameter describing constant exponent dependent on n
           (approximated after Ciotti & Bertin 1999 and MacArthur et al. 2003)
Image parameters (see modelbase.py):
    - Nx, Ny: image size (in pixels)
    - n_subsamples: number of sub-pixels

Notes:
    - total number of model parameters is 8
"""
from gleam.model.modelbase import _BaseModel
import numpy as np


class Sersic(_BaseModel):
    """
    Sersic intensity profile class
    """
    __modelname__ = 'sersic'
    parameter_keys = ['x', 'y', 'phi', 'e', 'I_0', 'c_0', 'n', 'r_s']

    def __init__(self, **kwargs):
        """
        Define model parameters as attribute variables and instantiate

        Args:
            None

        Kwargs:
            x <float> - first center coordinate on x-axis
            y <float> - second center coordinate on y-axis
            phi <float> - position angle (in radians [0, 2\pi])
            e <float> - ellipticity [0, 1]
            I_0 <float> - intensity at r=0
            c_0 <float> - box parameter
            n <float> - Sersic index
            r_s <float> - half-light radius
            Nx <int> - number of pixels along the x-axis
            Ny <int> - number of pixels along the y-axis
            n_subsamples: number of subsampling pixels
            verbose <bool> - verbose mode; print command line statements

        Return:
            <Sersic object> - Sersic model object instance
        """
        # Model specific parameters
        self.n = kwargs.pop('n', 4.)
        self.r_s = kwargs.pop('r_s', 1.)
        # Base parameters
        super(Sersic, self).__init__(**kwargs)

    @property
    def bn(self):
        """
        Internal non-free parameter describing constant exponent dependent on n.
        Approximated after (Ciotti & Bertin 1999) and (MacArthur et al. 2003)

        Args/Kwargs:
            None

        Return:
            bn <float> - constant exponent dependent on n
        """
        if hasattr(self, '_bn'):
            if self.n in self._bn:
                return self._bn[self.n]
        else:
            self._bn = {}
        n2 = self.n*self.n
        if self.n > 0.36:
            self._bn[self.n] = 2.*self.n-1./3.+4./(405.*self.n)+46./(25515.*n2) \
                               + 131./(1148175.*n2*self.n)-2194697./(30690717750.*n2*n2)
        else:
            self._bn[self.n] = .01945-.8902*self.n+10.95*n2-19.67*n2+13.43*n2*self.n
        return self._bn[self.n]

    @property
    def invr_s(self):
        """
        Inverse half-light radius

        Args/Kwargs:
            None

        Return:
            invr_s <float> - the inverse of r_s
        """
        if hasattr(self, '_invr_s'):
            if self.r_s in self._invr_s:
                return self._invr_s[self.r_s]
        else:
            self._invr_s = {}
        self._invr_s[self.r_s] = 1./self.r_s
        return self._invr_s[self.r_s]

    @property
    def invn(self):
        """
        Inverse Sersic index

        Args/Kwargs:
            None

        Return:
            invn <float> - inverse of n
        """
        if hasattr(self, '_invn'):
            if self.n in self._invn:
                return self._invn[self.n]
        else:
            self._invn = {}
        self._invn[self.n] = 1./self.n
        return self._invn[self.n]

    @property
    def profile_scale(self):
        """
        The specific scale variable of the model

        Args/Kwargs:
            None

        Return:
            scale <float> - intrinsic scale of the model
        """
        return self.r_s

    @profile_scale.setter
    def profile_scale(self, pscale):
        """
        Set the specific scale variable of the model

        Args:
            pscale <float> - profile scale

        Kwargs/Return:
            None
        """
        self.r_s = pscale

    @property
    def priors(self):
        """
        Parameter space limits

        Args/Kwargs:
            None

        Return:
            priors <list> - a list of min and max values the parameters should take
        """
        if not hasattr(self, '_priors'):
            self._priors = super(Sersic, self).priors + [
                [0.5, 25.],
                [2., np.sqrt(2)*.125*(self.Nx+self.Ny)]]
        return self._priors

    @priors.setter
    def priors(self, priors):
        """
        Set different parameter space limits than the default
        (simple step function as prior probability)

        Args:
            priors <list> - a list of min and max values the parameters should take

        Kwargs/Return:
            None
        """
        self._priors = priors

    def get_profile(self, a):
        """
        Intensity at radius a along the semi-major axis

        Args:
            a <float> - radial distance from the center along the semi-major axis
        Kwargs:
            None

        Return:
            I(r=a) <float> - intensity at radius a
        """
        return self.I_0 * np.exp(
            -self.bn * (np.power(a*self.invr_s, self.invn) - 1.))


def parse_arguments():
    """
    Parse command line arguments
    """
    from argparse import ArgumentParser, RawTextHelpFormatter
    parser = ArgumentParser(description=__doc__, formatter_class=RawTextHelpFormatter)
    # main args
    # TODO
    # mode args
    parser.add_argument("-v", "--verbose", dest="verbose", action="store_true",
                        help="Run program in verbose mode",
                        default=False)
    parser.add_argument("-t", "--test", "--test-mode", dest="test_mode", action="store_true",
                        help="Run program in testing mode",
                        default=False)
    args = parser.parse_args()
    return parser, args


if __name__ == "__main__":
    import sys
    parser, args = parse_arguments()
    no_input = len(sys.argv) <= 1
    if no_input:
        parser.print_help()
    elif args.test_mode:
        sys.argv = sys.argv[:1]
        from gleam.test.test_sersic import TestSersic
        TestSersic.main(verbosity=1)
