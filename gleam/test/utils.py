#!/usr/bin/env python
"""
@author: phdenzel

Testing utility module
"""
###############################################################################
# Imports
###############################################################################
import unittest

###############################################################################
class UnitTestPrototype(unittest.TestCase):
    OKAY = u'\033[92m'+u'\u2713'+u'\x1b[0m'
    FAIL = u'\033[91m'+u'\u2717'+u'\x1b[0m'
    separator = "#"*80

    @classmethod
    def setUpClass(cls):
        classname = cls.__name__.replace("Test", "").upper()
        test_descr = "# " + classname
        test_intro = "\n".join([cls.separator, test_descr, cls.separator])
        print("")
        print(test_intro)

    @classmethod
    def tearDownClass(cls):
        classname = cls.__name__.replace("Test", "").upper()
        test_descr = "# " + "End test of " + classname
        test_intro = "\n".join([cls.separator, test_descr, cls.separator])
        print("")
        print(test_intro)

    @classmethod
    def main(cls, **kwargs):
        v = kwargs.pop('verbosity', 1)
        unittest.main(verbosity=v, **kwargs)
