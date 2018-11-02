#!/usr/bin/env python
"""
@author: phdenzel

Make really, really sure you can...
Climb every peak in search for lens and source candidates
"""
###############################################################################
# Imports
###############################################################################
from gleam.skycoords import SkyCoords
from gleam.lensobject import LensObject
from gleam.lensfinder import LensFinder
import os
import numpy as np
from gleam.test.utils import UnitTestPrototype


# CLASS TESTING ###############################################################
class TestLensFinder(UnitTestPrototype):

    def setUp(self):
        # arguments and keywords
        self.test_fits = os.path.abspath(os.path.dirname(__file__)) \
                         + '/W3+3-2.I.12907_13034_7446_7573.fits'
        self.kwargs = {
            'n': 5,
            'min_q': 0.1,
            'sigma': (1, 3),
            'centroid': 5
        }
        self.v = {'verbose': 1}
        # __init__ test
        self.lo = LensObject(self.test_fits)
        self.finder = LensFinder(self.lo, **self.kwargs)
        # verbosity
        self.kwargs.update(self.v)
        print("")
        print(self.separator)
        print(self.shortDescription())

    def tearDown(self):
        print("")

    def test_LensFinder(self):
        """ # LensFinder """
        print(">>> {}".format(self.lo))
        finder = LensFinder(self.lo, **self.kwargs)
        self.assertIsInstance(finder, LensFinder)
        self.assertTrue(hasattr(finder, 'peaks'))
        self.assertTrue(hasattr(finder, 'lens_candidate'))
        self.assertTrue(hasattr(finder, 'source_candidates'))

    def test_threshold_estimate(self):
        """ # threshold_estimate """
        print(">>> {}".format(self.lo.data))
        threshold = LensFinder.threshold_estimate(
            self.lo.data, sigma=self.kwargs['sigma'], **self.v)
        self.assertTrue(self.finder.threshold.shape == self.lo.data.shape)

    def test_peak_candidates(self):
        """ # peak_candidates """
        print(">>> {}, {}".format(self.lo.data, self.finder.threshold))
        peaks, vals = LensFinder.peak_candidates(
            self.lo.data, self.finder.threshold, min_d=self.kwargs['min_q']*self.lo.naxis1,
            n=self.kwargs['n'], centroid=self.kwargs['centroid'], **self.v)
        p_idcs = [(int(round(p[1])), int(round(p[0]))) for p in peaks]
        self.assertTrue(len(peaks) == len(vals))
        self.assertEqual([self.lo.data[p] for p in p_idcs], vals)

    def test_detect_lens(self):
        """ # detect_lens """
        print(">>> {}, {}".format(self.finder.peaks, self.finder.peak_values))
        candidate = LensFinder.detect_lens(self.finder.peaks, self.finder.peak_values, **self.v)
        self.assertIsNotNone(candidate[0])
        self.assertIsInstance(candidate[0], SkyCoords)
        self.assertIsInstance(candidate[1], type(self.lo.data[0, 0]))
        self.assertIsInstance(candidate[2], int)

    def test_order_by_distance(self):
        """ # order_by_distance """
        print(">>> {}, {}".format(self.finder.source_candidates, self.finder.lens_candidate))
        reordered, order = LensFinder.order_by_distance(
            self.finder.source_candidates, self.finder.lens_candidate, **self.v)
        self.assertTrue(len(reordered) == len(self.finder.source_candidates))

    def test_relative_positions(self):
        """ # relative_positions """
        print(">>> {}, {}".format(self.finder.source_candidates, self.finder.lens_candidate))
        relp = LensFinder.relative_positions(
            self.finder.source_candidates, self.finder.lens_candidate, **self.v)
        self.assertTrue(len(relp) == len(self.finder.source_candidates))


if __name__ == "__main__":
    TestLensFinder.main(verbosity=1)
