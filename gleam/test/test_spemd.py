#!/usr/bin/env python
"""
@author: phdenzel

Make really, really sure you have the right...
Singular Power-law Elliptical Potential
"""
###############################################################################
# Imports
###############################################################################
from gleam.model.spemd import SPEMD
import os
import numpy as np
from gleam.test.utils import UnitTestPrototype


# CLASS TESTING ###############################################################
class TestSPEMD(UnitTestPrototype):

    def setUp(self):
        # arguments and keywords
        self.model_kwargs = {
            'phi_G': 1.731,
            'q': .787,
            'e': 1.-.787,
            'theta_E': 1.161,
            'gamma': 2.044
        }
        self.v = {'verbose': 1}
        # __init__ test
        self.spemd = SPEMD(Nx=200, Ny=100, **self.model_kwargs)
        # verbosity
        print("")
        print(self.separator)
        print(self.shortDescription())

    def tearDown(self):
        print("")

    def test_SPEMD(self):
        """ # SPEMD """
        print(">>> {}".format(self.model_kwargs))
        spemd = SPEMD(Nx=100, Ny=100, **dict(self.model_kwargs, **self.v))
        self.assertIsInstance(spemd, SPEMD)

    def test_q(self):
        """ # q """
        print(">>> {}".format(()))
        q = self.spemd.q
        self.assertIsNotNone(q)
        self.assertIsInstance(q, float)
        self.assertEqual(q, 1-self.model_kwargs['e'])
        print(q)

    def test_cosPA(self):
        """ # cosPA """
        print(">>> {}".format(()))
        cosPA = self.spemd.cosPA
        self.assertIsNotNone(cosPA)
        self.assertIsInstance(cosPA, np.float64)
        self.assertEqual(cosPA, np.cos(self.model_kwargs['phi_G']))
        print(cosPA)

    def test_sinPA(self):
        """ # sinPA """
        print(">>> {}".format(()))
        sinPA = self.spemd.sinPA
        self.assertIsNotNone(sinPA)
        self.assertIsInstance(sinPA, np.float64)
        self.assertEqual(sinPA, np.sin(self.model_kwargs['phi_G']))
        print(sinPA)

    def test_model_parameters(self):
        """ # model_parameters """
        print(">>> {}".format(()))
        pars = self.spemd.model_parameters
        self.assertEqual(len(pars), self.spemd.N)

    def test_model_parameters_setter(self):
        """ # model_parameters.setter """
        kwargs = {  # double all parameters except for gamma
            'phi_G': 2*self.model_kwargs['phi_G'],
            'q': 2*self.model_kwargs['q'],
            'theta_E': 2*self.model_kwargs['theta_E'],
            'gamma': 2.044
        }
        print(">>> {}".format(kwargs))
        before = self.spemd.model_parameters[:]
        self.spemd.model_parameters = kwargs
        after = self.spemd.model_parameters[:]
        self.assertFalse(before == after)
        self.assertEqual(before[SPEMD.parameter_keys.index('gamma')],
                         after[SPEMD.parameter_keys.index('gamma')])
        print(SPEMD.parameter_keys)
        print(before)
        print(after)

    def test_set_model_parameters(self):
        """ # set_model_parameters """
        kwargs = {  # double all parameters except for gamma
            'phi_G': 2*self.model_kwargs['phi_G'],
            'q': 2*self.model_kwargs['q'],
            'theta_E': 2*self.model_kwargs['theta_E'],
            'gamma': 2.044
        }
        print(">>> {}".format(kwargs))
        before = self.spemd.model_parameters[:]
        self.spemd.set_model_parameters(kwargs)
        after = self.spemd.model_parameters[:]
        self.assertFalse(before == after)
        self.assertEqual(before[SPEMD.parameter_keys.index('gamma')],
                         after[SPEMD.parameter_keys.index('gamma')])
        print(SPEMD.parameter_keys)
        print(before)
        print(after)

    def test_map_parameters(self):
        """ # map_parameters """
        print(">>> {}".format(()))
        pars = self.spemd.map_parameters
        self.assertEqual(len(pars), len(self.spemd.map_keys))
        print(SPEMD.map_keys)
        print(pars)

    def test_calc_map(self):
        """ # calc_map """
        print(">>> {}".format(()))
        self.spemd.calc_map(smooth_center=True, **self.v)
        self.assertTrue(np.any(self.spemd.map2D))

    def test_get_map(self):
        """ # get_map """
        print(">>> {}".format(()))
        map2D = self.spemd.get_map(smooth_center=True, **self.v)
        self.assertTrue(np.any(map2D))

    def test_calc_pixel(self):
        """ # calc_pixel """
        print(">>> {}".format((self.spemd.x, self.spemd.y)))
        center = self.spemd.calc_pixel(self.spemd.x, self.spemd.y)
        self.assertIsInstance(center, np.float64)
        print(center)

    def test_plot_map(self):
        """ # plot_map """
        self.spemd.calc_map(smooth_center=True)
        fig = self.spemd.plot_map(log=True, contours=True, show=False)
        print(fig)

if __name__ == "__main__":
    TestSPEMD.main(verbosity=1)
