#!/usr/bin/env python
"""
@author: phdenzel

Make really, really sure that...
MultiLens sees same lens in different bands
"""
###############################################################################
# Imports
###############################################################################
from gleam.skycoords import SkyCoords
from gleam.multilens import MultiLens
import os
import matplotlib.pyplot as plt
from gleam.test.utils import UnitTestPrototype


# CLASS TESTING ###############################################################
class TestMultiLens(UnitTestPrototype):

    def setUp(self):
        # arguments and keywords
        self.test_fits = os.path.abspath(os.path.dirname(__file__))
        # __init__ test
        self.ml = MultiLens(self.test_fits)
        # verbosity
        self.v = {'verbose': 1}
        print("")
        print(self.separator)
        print(self.shortDescription())

    def tearDown(self):
        print("")

    def test_MultiLens(self):
        """ # MultiLens """
        print(">>> {}".format(self.test_fits))
        ml = MultiLens(self.test_fits, **self.v)
        self.assertIsInstance(ml, MultiLens)
        self.assertEqual(ml, self.ml)

    def test_copy(self):
        """ # copy """
        print(">>> {}".format(self.ml))
        copy = self.ml.copy(**self.v)
        self.assertEqual(copy, self.ml)
        self.assertFalse(copy is self.ml)

    def test_deepcopy(self):
        """ # deepcopy """
        print(">>> {}".format(self.ml))
        copy = self.ml.deepcopy(**self.v)
        self.assertEqual(copy, self.ml)
        self.assertFalse(copy is self.ml)

    def test_from_json(self):
        """ # from_json """
        filename = 'test.json'
        self.ml.lens_objects[5].photzp = 25.67
        filename = self.ml.jsonify(name='test.json')
        print(">>> {}".format(filename))
        with open(filename, 'r') as j:
            jcopy = MultiLens.from_json(j, **self.v)
            self.assertEqual(jcopy, self.ml)
            self.assertFalse(jcopy is self.ml)
            self.assertEqual(jcopy.lens_objects[0], self.ml.lens_objects[0])
            self.assertEqual(jcopy.lens_objects[5].photzp, self.ml.lens_objects[5].photzp)
        try:
            os.remove(filename)
        except OSError:
            pass

    def test_jsonify(self):
        """ # jsonify """
        self.ml.lens_objects[5].photzp = 25.67
        print(">>> {}".format(self.ml))
        jsnstr = self.ml.jsonify(**self.v)
        self.assertIsInstance(jsnstr, str)

    def test_find_files(self):
        """ # find_files """
        print(">>> {}".format(self.test_fits))
        fpaths = MultiLens.find_files(self.test_fits, **self.v)
        self.assertTrue([self.test_fits in fpaths])

    def test_add_srcimg(self):
        """ # add_srcimg """
        print(">>> {}".format((-1.234, 0.567)))
        before = self.ml.srcimgs[:]
        self.ml.add_srcimg((-1.234, 0.567), unit='arcsec', relative=True, **self.v)
        after = self.ml.srcimgs
        self.assertEqual([len(b)+1 for b in before], [len(a) for a in after])

    def test_add_to_patch(self):
        """ # add_to_patch"""
        print(">>> {}".format(self.ml.filepaths[0]))
        before = self.ml.copy()
        self.ml.add_to_patch(self.ml.filepaths[0])
        after = self.ml
        self.assertEqual(before.N+1, after.N)
        self.assertEqual(before.filepaths, after.filepaths[:-1])
        self.assertEqual(before.bands, after.bands[:-1])
        self.assertEqual(self.ml.filepaths[0], after.filepaths[-1])

    def test_remove_from_patch(self):
        """ # remove_from_patch """
        print(">>> {}".format(-1))
        before = self.ml.copy()
        self.ml.remove_from_patch(-1, **self.v)
        after = self.ml
        self.assertEqual(before.N-1, after.N)
        self.assertEqual(before.filepaths[:-1], after.filepaths)

    def test_reorder_patch(self):
        """ # reorder_patch """
        print(">>> {}".format(list(range(self.ml.N))[::-1]))
        before = self.ml.copy()
        self.ml.reorder_patch(list(range(self.ml.N))[::-1], **self.v)
        after = self.ml
        self.assertEqual(before.N, after.N)
        self.assertEqual(before.filepaths[::-1], after.filepaths)
        self.assertEqual(before.bands[::-1], after.bands)

    def test_plot_composite(self):
        """ # plot_composite """
        print(">>> {}".format(self.ml))
        fig, ax = self.ml.show_composite(savefig='test.pdf', **self.v)
        self.assertIsNotNone(fig, ax)
        try:
            os.remove('test.pdf')
        except OSError:
            pass

if __name__ == "__main__":
    TestMultiLens.main(verbosity=1)
