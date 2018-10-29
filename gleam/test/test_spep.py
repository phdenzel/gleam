#!/usr/bin/env python
"""
@author: phdenzel

Make really, really sure you have the right...
Singular Power-law Elliptical Potential
"""
###############################################################################
# Imports
###############################################################################
import __init__
from gleam.model.spep import SPEP
import os
import numpy as np
from gleam.test.utils import UnitTestPrototype


# CLASS TESTING ###############################################################
class TestSPEP(UnitTestPrototype):

    def setUp(self):
        # arguments and keywords
        self.model_kwargs = {
            'x': 100,
            'y': 100,
            'phi_G': np.pi,
            'phi': np.pi,
            'q': 1.,
            'theta_E': 1.,
            'gamma': 2.
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
        self.assertEqual(cosPA, np.cos(self.model_kwargs['phi']))
        print(cosPA)

    def test_sinPA(self):
        """ # sinPA """
        print(">>> {}".format(()))
        sinPA = self.spep.sinPA
        self.assertIsNotNone(sinPA)
        self.assertIsInstance(sinPA, np.float64)
        self.assertEqual(sinPA, np.sin(self.model_kwargs['phi']))
        print(sinPA)

    def test_model_parameters(self):
        """ # model_parameters """
        print(">>> {}".format(()))
        pars = self.spep.model_parameters
        self.assertEqual(len(pars), self.spep.N)

    def test_model_parameters_setter(self):
        """ # model_parameters.setter """
        kwargs = { # double all parameters except for gamma
            'x': 200,
            'y': 200,
            'phi_G': 2*np.pi,
            'phi': 2*np.pi,
            'q': 2.,
            'theta_E': 2.,
            'gamma': 2.
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
        kwargs = { # double all parameters except for gamma
            'x': 200,
            'y': 200,
            'phi_G': 2*np.pi,
            'q': 2.,
            'theta_E': 2.,
            'gamma': 2.
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
        print(list(zip(SPEP.map_keys, pars)))

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
        print(">>> {}".format((self.spep.x, self.spep.y)))
        print("TODO")



if __name__ == "__main__":
    TestSPEP.main(verbosity=1)
