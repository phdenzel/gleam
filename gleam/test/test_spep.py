#!/usr/bin/env python
"""
@author: phdenzel

Make really, really sure you have the right...
Singular Power-law Elliptical Potential
"""
###############################################################################
# Imports
###############################################################################
from gleam.model.spep import SPEP
import os
import numpy as np
from gleam.test.utils import UnitTestPrototype


# CLASS TESTING ###############################################################
class TestSPEP(UnitTestPrototype):

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
        self.spep = SPEP(Nx=200, Ny=200, **self.model_kwargs)
        # verbosity
        print("")
        print(self.separator)
        print(self.shortDescription())

    def tearDown(self):
        print("")

    def test_SPEP(self):
        """ # SPEP """
        print(">>> {}".format(self.model_kwargs))
        spep = SPEP(Nx=200, Ny=200, **dict(self.model_kwargs, **self.v))
        self.assertIsInstance(spep, SPEP)

    def test_q(self):
        """ # q """
        print(">>> {}".format(()))
        q = self.spep.q
        self.assertIsNotNone(q)
        self.assertIsInstance(q, float)
        self.assertEqual(q, 1-self.model_kwargs['e'])
        print(q)

    def test_cosPA(self):
        """ # cosPA """
        print(">>> {}".format(()))
        cosPA = self.spep.cosPA
        self.assertIsNotNone(cosPA)
        self.assertIsInstance(cosPA, np.float64)
        self.assertEqual(cosPA, np.cos(np.pi/2-self.model_kwargs['phi_G']))
        print(cosPA)

    def test_sinPA(self):
        """ # sinPA """
        print(">>> {}".format(()))
        sinPA = self.spep.sinPA
        self.assertIsNotNone(sinPA)
        self.assertIsInstance(sinPA, np.float64)
        self.assertEqual(sinPA, np.sin(np.pi/2-self.model_kwargs['phi_G']))
        print(sinPA)

    def test_model_parameters(self):
        """ # model_parameters """
        print(">>> {}".format(()))
        pars = self.spep.model_parameters
        self.assertEqual(len(pars), self.spep.N)

    def test_model_parameters_setter(self):
        """ # model_parameters.setter """
        kwargs = {  # double all parameters except for gamma
            'phi_G': 2*self.model_kwargs['phi_G'],
            'q': 2*self.model_kwargs['q'],
            'theta_E': 2*self.model_kwargs['theta_E'],
            'gamma': 2.044
        }
        print(">>> {}".format(kwargs))
        before = self.spep.model_parameters[:]
        self.spep.model_parameters = kwargs
        after = self.spep.model_parameters[:]
        self.assertFalse(before == after)
        self.assertEqual(before[SPEP.parameter_keys.index('gamma')],
                         after[SPEP.parameter_keys.index('gamma')])
        print(SPEP.parameter_keys)
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
        before = self.spep.model_parameters[:]
        # self.spep.model_parameters = kwargs
        self.spep.set_model_parameters(kwargs)
        after = self.spep.model_parameters[:]
        self.assertFalse(before == after)
        self.assertEqual(before[SPEP.parameter_keys.index('gamma')],
                         after[SPEP.parameter_keys.index('gamma')])
        print(SPEP.parameter_keys)
        print(before)
        print(after)

    def test_map_parameters(self):
        """ # map_parameters """
        print(">>> {}".format(()))
        pars = self.spep.map_parameters
        self.assertEqual(len(pars), len(self.spep.map_keys))
        print(SPEP.map_keys)
        print(pars)

    def test_calc_map(self):
        """ # calc_map """
        print(">>> {}".format(()))
        self.spep.calc_map(smooth_center=True, **self.v)
        self.assertTrue(np.any(self.spep.map2D))

    def test_get_map(self):
        """ # get_map """
        print(">>> {}".format(()))
        map2D = self.spep.get_map(smooth_center=True, **self.v)
        self.assertTrue(np.any(map2D))

    def test_calc_pixel(self):
        """ # calc_pixel """
        print(">>> {}".format((self.spep.x, self.spep.y)))
        center = self.spep.calc_pixel(self.spep.x, self.spep.y)
        self.assertIsInstance(center, np.float64)
        print(center)

    def test_plot_map(self):
        """ # plot_map """
        self.spep.calc_map()
        fig = self.spep.plot_map(contours=True, show=False)
        print(fig)


if __name__ == "__main__":
    TestSPEP.main(verbosity=1)
