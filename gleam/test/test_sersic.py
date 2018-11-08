#!/usr/bin/env python
"""
@author: phdenzel

Make really, really sure you have the right...
Elliptical 2D Sersic model profile
"""
###############################################################################
# Imports
###############################################################################
from gleam.model.sersic import Sersic
import os
import numpy as np
from gleam.test.utils import UnitTestPrototype


# CLASS TESTING ###############################################################
class TestSersic(UnitTestPrototype):

    def setUp(self):
        # arguments and keywords
        self.model_kwargs = {
            'x': 100,
            'y': 100,
            'phi': np.pi,
            'e': 0.5,
            'I_0': 10.,
            'c_0': 0.5,
            'n': 2.,
            'r_s': 10.
        }
        self.v = {'verbose': 1}
        # __init__ test
        self.sersic = Sersic(Nx=200, Ny=200, **self.model_kwargs)
        # verbosity
        print("")
        print(self.separator)
        print(self.shortDescription())

    def tearDown(self):
        print("")

    def test_Sersic(self):
        """ # Sersic """
        print(">>> {}".format(self.model_kwargs))
        sersic = Sersic(Nx=200, Ny=200, **dict(self.model_kwargs, **self.v))
        self.assertIsInstance(sersic, Sersic)

    def test_q(self):
        """ # q """
        print(">>> {}".format(()))
        q = self.sersic.q
        self.assertIsNotNone(q)
        self.assertIsInstance(q, float)
        self.assertEqual(q, 1-self.model_kwargs['e'])
        print(q)

    def test_cosPA(self):
        """ # cosPA """
        print(">>> {}".format(()))
        cosPA = self.sersic.cosPA
        self.assertIsNotNone(cosPA)
        self.assertIsInstance(cosPA, np.float64)
        self.assertEqual(cosPA, np.cos(self.model_kwargs['phi']))
        print(cosPA)

    def test_sinPA(self):
        """ # sinPA """
        print(">>> {}".format(()))
        sinPA = self.sersic.sinPA
        self.assertIsNotNone(sinPA)
        self.assertIsInstance(sinPA, np.float64)
        self.assertEqual(sinPA, np.sin(self.model_kwargs['phi']))
        print(sinPA)

    def test_model_parameters(self):
        """ # model_parameters """
        print(">>> {}".format(()))
        pars = self.sersic.model_parameters
        self.assertEqual(len(pars), self.sersic.N)

    def test_model_parameters_setter(self):
        """ # model_parameters.setter """
        kwargs = {  # double all parameters except for half-light radius
            'x': 200,
            'y': 200,
            'phi': 2*np.pi,
            'e': 1.,
            'I_0': 20.,
            'c_0': 1.,
            'n': 4.,
        }
        print(">>> {}".format(kwargs))
        before = self.sersic.model_parameters[:]
        self.sersic.model_parameters = kwargs
        after = self.sersic.model_parameters[:]
        self.assertFalse(before == after)
        self.assertEqual(before[Sersic.parameter_keys.index('r_s')],
                         after[Sersic.parameter_keys.index('r_s')])
        print(Sersic.parameter_keys)
        print(before)
        print(after)

    def test_set_model_parameters(self):
        """ # set_model_parameters """
        kwargs = {  # double all parameters except for half-light radius
            'x': 200,
            'y': 200,
            'phi': 2*np.pi,
            'e': 1.,
            'I_0': 20.,
            'c_0': 1.,
            'n': 4.,
        }
        print(">>> {}".format(kwargs))
        before = self.sersic.model_parameters[:]
        # self.sersic.model_parameters = kwargs
        self.sersic.set_model_parameters(kwargs)
        after = self.sersic.model_parameters[:]
        self.assertFalse(before == after)
        self.assertEqual(before[Sersic.parameter_keys.index('r_s')],
                         after[Sersic.parameter_keys.index('r_s')])
        print(Sersic.parameter_keys)
        print(before)
        print(after)

    def test_map_parameters(self):
        """ # map_parameters """
        print(">>> {}".format(()))
        pars = self.sersic.map_parameters
        self.assertEqual(len(pars), len(self.sersic.map_keys))
        print(Sersic.map_keys)
        print(pars)
        print(list(zip(Sersic.map_keys, pars)))

    def test_get_map(self):
        """ # get_map """
        print(">>> {}".format(()))
        print("TODO")

    def test_calc_map(self):
        """ # calc_map """
        print(">>> {}".format(()))
        print("TODO")

    def test_calc_pixel(self):
        """ # calc_pixel """
        print(">>> {}".format((self.sersic.x, self.sersic.y)))
        print("TODO")


if __name__ == "__main__":
    TestSersic.main(verbosity=1)
