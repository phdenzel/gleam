#!/usr/bin/env python
"""
@author: phdenzel

Make really, really sure you can choose amongst all shapes...
What are you interested in? Squares, circles, polygons, or an amorph collection of pixels...

Features:
  - select
    + circle
    + square
    + rect
    + polygon
    + amorph
    + by_color
  - focus
    o use a buffer to save selections and switch between them with focus
    + circle
    + square
    + rect
    + polygon
    + amorph
    + by_color
"""
###############################################################################
# Imports
###############################################################################
from gleam.skyf import SkyF
from gleam.skypatch import SkyPatch
from gleam.roiselector import ROISelector
import os
import numpy as np
from gleam.test.utils import UnitTestPrototype


# CLASS TESTING ###############################################################
class TestROISelector(UnitTestPrototype):

    def setUp(self):
        # arguments and keywords
        self.test_fits = os.path.abspath(os.path.dirname(__file__)) \
                         + '/W3+3-2.I.12907_13034_7446_7573.fits'
        self.test_ftss = os.path.dirname(self.test_fits)
        # __init__ test
        self.skyf = SkyF(self.test_fits)
        self.skypatch = SkyPatch(self.test_ftss)
        # verbosity
        self.v = {'verbose': 1}
        print("")
        print(self.separator)
        print(self.shortDescription())

    def tearDown(self):
        print("")

    def test_ROISelector(self):
        """ # ROISelector """
        print(">>> {}".format(None))
        select = ROISelector(None, **self.v)
        self.assertIsInstance(select, ROISelector)
        self.assertIsNone(select.data)
        print(">>> {}".format(self.skyf.data))
        select = ROISelector(self.skyf.data, **self.v)
        self.assertIsInstance(select, ROISelector)
        self.assertIsNotNone(select.data)

    def test_from_skyf(self):
        """ # from_skyf """
        print(">>> {}".format(self.skyf))
        select = ROISelector.from_skyf(self.skyf, **self.v)
        self.assertIsInstance(select, ROISelector)
        self.assertIsNotNone(select.data)

    def test_from_skypatch(self):
        """ # from_skypatch """
        print(">>> {}".format(self.skypatch))
        select = ROISelector.from_skypatch(self.skypatch, **self.v)
        self.assertIsInstance(select, ROISelector)
        self.assertIsNotNone(select.data)


if __name__ == "__main__":
    TestROISelector.main(verbosity=1)
