#!/usr/bin/env python
"""
@author: phdenzel

SPEMD - Singular Power-law Elliptical Mass Distribution

Base parameters (see modelbase.py):
    - x, y: center coordinates
    - phi: position angle
    - e: ellipticity
Model-specific parameters:
    - theta_E: Einstein radius (sets specific scale)
    - q: minor-to-major axis ratio
    - phi_G: position angle
    - gamma: power-law slope
Image parameters (see modelbase.py):
    - Nx, Ny: image size (in pixels)
    - n_subsamples: number of sub-pixels

Notes:
    - total number of model parameters is 8
"""
from gleam.model.modelbase import _BaseModel
import numpy as np


class SPEMD(_BaseModel):
    """
    Singular power-law elliptical mass distribution
    """
    __modelname__ = 'spemd'
    parameter_keys = ['x', 'y', 'phi_G', 'q', 'gamma', 'theta_E']

    def __init__(self, **kwargs):
        """
        Define model parameters as attribute variables and instantiate

        Args:
            None

        Kwargs:
            x <float> - first center coordinate on x-axis
            y <float> - second center coordinate on y-axis
            phi_G <float> - position angle with respect to y-axis
                            (in radians [0, 2\pi])
            q <float> - minor-to-major axis ratio [0.5, 1]
            gamma <float> - power-law slope index
            s <float> - smoothing/core radius
            phi <float> - position angle (in radians [0, 2\pi]) will be overridden by phi_G
            e <float> - ellipticity [0, 1] will be overridden by q
            Nx <int> - number of pixels along the x-axis
            Ny <int> - number of pixels along the y-axis
            n_subsamples: number of subsampling pixels
            verbose <bool> - verbose mode; print command line statements

        Return:
            <SPEMD object> - SPEMD model object instance
        """
        # Model specific parameters
        theta_E = kwargs.pop('theta_E', 1.)
        q = kwargs.pop('q', 1)
        phi_G = kwargs.pop('phi_G', 0.)
        gamma = kwargs.pop('gamma', 2.)
        s = kwargs.pop('s', 1e-8)
        self.theta_E = theta_E
        self.phi_G = phi_G
        self.gamma = gamma
        self.s2 = s*s
        # Base parameters
        kwargs['phi'] = np.pi/2-self.phi_G
        kwargs['e'] = 1 - q
        super(SPEMD, self).__init__(**kwargs)
        # finally apply parameter constraints
        self.apply_constaints()

    def apply_constaints(self):
        """
        Apply parameter constraints to avoid invalid results

        Args/Kwargs/Return:
            None
        """
        if self.gamma < 1.4:
            self.gamma = 1.4
            self.theta_E = 0
        if self.gamma > 2.9:
            self.gamma = 2.9
            self.theta_E = 0
        if self.q < 0.5:
            self.q = 0.5
            self.theta_E = 0
        if self.q > 1:
            self.q = 1.
            self.theta_E = 0

    @property
    def E(self):
        """
        Fixes overall normalization

        Args/Kwargs:
            None

        Return:
            E <float> - normalization factor
        """
        if hasattr(self, '_E'):
            if (self.gamma, self.theta_E, self.q) in self._E:
                return self._E[(self.gamma, self.theta_E, self.q)]
        else:
            self._E = {}
        # E = self.theta_E / ((3-self.gamma) / 2.)**(1./(1-self.gamma))
        E = (3.-self.gamma)/2. * (self.theta_E*self.theta_E / self.q)**((self.gamma-1.)/2.)
        self._E[(self.gamma, self.theta_E, self.q)] = E
        return self._E[(self.gamma, self.theta_E, self.q)]

    @property
    def eta(self):
        """
        Power-law index from Barkana (1998) [arXiv:astro-ph/9802002]

        Args/Kwargs:
            None

        Return:
            eta <float> - power-law index derived from gamma
        """
        if hasattr(self, '_eta'):
            if (self.gamma) in self._eta:
                return self._eta[self.gamma]
        else:
            self._eta = {}
        self._eta[self.gamma] = 3-self.gamma
        return self._eta[self.gamma]

    @property
    def profile_scale(self):
        """
        The specific scale variable of the model

        Args/Kwargs:
            None

        Return:
            scale <float> - intrinsic scale of the model
        """
        return self.theta_E

    @profile_scale.setter
    def profile_scale(self, pscale):
        """
        Set the specific scale variable of the model

        Args:
            pscale <float> - profile scale

        Kwargs/Return:
            None
        """
        self.theta_E = pscale

    def get_profile(self, a):
        """
        Surface mass density at radius a along the profile axis

        Args:
            a <float> - radial distance from the center along the profile axis
        Kwargs:
            None

        Return:
            I(r=a) <float> - intensity at radius a
        """
        return self.E * (a*a + self.s2)**(self.eta/2. - 1)


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
        from gleam.test.test_spemd import TestSPEMD
        TestSPEMD.main(verbosity=1)
