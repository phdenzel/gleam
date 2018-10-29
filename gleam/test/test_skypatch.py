#!/usr/bin/env python
"""
@author: phdenzel

Make really, really sure you can...
Phase through a region in the sky with SkyPatch
"""
###############################################################################
# Imports
###############################################################################
import __init__
from gleam.skypatch import SkyPatch
import os
import matplotlib.pyplot as plt
from gleam.test.utils import UnitTestPrototype


# CLASS TESTING ###############################################################
class TestSkyPatch(UnitTestPrototype):

    def setUp(self):
        # arguments and keywords
        self.test_fits = os.path.abspath(os.path.dirname(__file__))
        # __init__ test
        self.skyp = SkyPatch(self.test_fits)
        # verbosity
        self.v = {'verbose': 1}
        print("")
        print(self.separator)
        print(self.shortDescription())

    def tearDown(self):
        print("")

    def test_SkyPatch(self):
        """ # SkyPatch """
        print(">>> {}".format(self.test_fits))
        skyp = SkyPatch(self.test_fits, **self.v)
        self.assertIsInstance(skyp, SkyPatch)
        self.assertEqual(skyp, self.skyp)

    def test_copy(self):
        """ # copy """
        print(">>> {}".format(self.skyp))
        copy = self.skyp.copy(**self.v)
        self.assertEqual(copy, self.skyp)
        self.assertFalse(copy is self.skyp)

    def test_deepcopy(self):
        """ # deepcopy """
        print(">>> {}".format(self.skyp))
        copy = self.skyp.deepcopy(**self.v)
        self.assertEqual(copy, self.skyp)
        self.assertFalse(copy is self.skyp)

    def test_find_files(self):
        """ # find_files """
        print(">>> {}".format(self.test_fits))
        fpaths = SkyPatch.find_files(self.test_fits, **self.v)
        self.assertTrue([self.test_fits in fpaths])

    def test_add_to_patch(self):
        """ # add_to_patch """
        print(">>> {}".format(self.skyp.filepaths[0]))
        before = self.skyp.filepaths[:]
        self.skyp.add_to_patch(self.skyp.filepaths[0], **self.v)
        after = self.skyp.filepaths
        self.assertEqual(len(before)+1, len(after))
        self.assertEqual(before, after[:-1])

    def test_remove_from_patch(self):
        """ # remove_from_patch """
        print(">>> {}".format(-1))
        before = self.skyp.filepaths[:]
        self.skyp.remove_from_patch(-1, **self.v)
        after = self.skyp.filepaths
        self.assertEqual(len(before)-1, len(after))
        self.assertEqual(before[:-1], after)

    def test_reorder_patch(self):
        """ # reorder_patch """
        print(">>> {}".format(list(range(self.skyp.N))[::-1]))
        before = self.skyp.filepaths[:]
        self.skyp.reorder_patch(list(range(self.skyp.N))[::-1], **self.v)
        after = self.skyp.filepaths
        self.assertEqual(len(before), len(after))
        self.assertEqual(before[::-1], after)

    def test_plot_composite(self):
        """ # plot_composite """
        print(">>> {}".format(self.skyp))
        fig, ax = self.skyp.plot_composite(plt.figure(), **self.v)
        self.assertIsNotNone(fig, ax)


if __name__ == "__main__":
    TestSkyPatch.main(verbosity=1)
