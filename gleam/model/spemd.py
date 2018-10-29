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
    - gamma: power-law slope index
Image parameters (see modelbase.py):
    - Nx, Ny: image size (in pixels)
    - n_subsamples: number of sub-pixels

Notes:
    - total number of model parameters is 8
"""
import __init__
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
        theta_E = kwargs.pop('theta_E', 1.)
        q = kwargs.pop('q', 1)
        phi_G = kwargs.pop('phi_G', 0.)
        gamma = kwargs.pop('gamma', 2.)
        # Base parameters
        kwargs['phi'] = phi_G
        kwargs['e'] = 1 - q
        super(SPEMD, self).__init__(**kwargs)
        # Model specific parameters
        self.theta_E = theta_E
        self.phi_G = self.phi
        # apply parameter constraints
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

    def get_major_profile(self, a):
        """
        Intensity at radius a along the semi-major axis

        Args:
            a <float> - radial distance from the center along the semi-major axis
        Kwargs:
            None

        Return:
            I(r=a) <float> - intensity at radius a
        """
        return 0


if __name__ == "__main__":
    from gleam.test.test_spemd import TestSPEMD
    TestSPEMD.main(verbosity=1)
    
