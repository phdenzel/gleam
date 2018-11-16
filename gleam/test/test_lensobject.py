#!/usr/bin/env python
"""
@author: phdenzel

Make really, really sure you look at...
Gravitational lenses and all their properties
"""
###############################################################################
# Imports
###############################################################################
from gleam.skycoords import SkyCoords
from gleam.lensobject import LensObject
import os
import numpy as np
import matplotlib.pyplot as plt
from gleam.test.utils import UnitTestPrototype


# CLASS TESTING ###############################################################
class TestLensObject(UnitTestPrototype):

    def setUp(self):
        # arguments and keywords
        self.test_fits = os.path.abspath(os.path.dirname(__file__)) \
                         + '/W3+3-2.U.12907_13034_7446_7573.fits'
        # __init__ test
        self.lobject = LensObject(self.test_fits)
        # verbosity
        self.v = {'verbose': 1}
        print("")
        print(self.separator)
        print(self.shortDescription())

    def tearDown(self):
        print("")

    def test_LensObject(self):
        """ # LensObject """
        print(">>> {}".format(self.test_fits))
        lobject = LensObject(self.test_fits, **self.v)
        self.assertIsInstance(lobject, LensObject)
        self.assertEqual(lobject, self.lobject)
        kwargs = {'auto': True,
                  'n': 5,
                  'min_q': 0.1,
                  'sigma': (1, 3),
                  'centroid': 5}
        print(">>> {}, {}".format(self.test_fits, kwargs))
        auto = kwargs.pop('auto')
        lobject = LensObject(self.test_fits, auto=auto, finder_options=kwargs, **self.v)
        self.assertIsInstance(lobject, LensObject)
        print(">>> {}".format(None))
        lobject = LensObject(None, **self.v)
        self.assertIsInstance(lobject, LensObject)

    def test_copy(self):
        """ # copy """
        print(">>> {}".format(self.lobject))
        self.lobject.zl = 0.5
        copy = self.lobject.copy(**self.v)
        self.assertEqual(copy, self.lobject)
        self.assertFalse(copy is self.lobject)
        self.assertEqual(copy.zl, self.lobject.zl)

    def test_deepcopy(self):
        """ # deepcopy """
        print(">>> {}".format(self.lobject))
        self.lobject.zl = 0.5
        copy = self.lobject.deepcopy(**self.v)
        self.assertEqual(copy, self.lobject)
        self.assertFalse(copy is self.lobject)
        self.assertEqual(copy.zl, self.lobject.zl)

    def test_from_json(self):
        """ # from_json """
        filename = 'test.json'
        self.lobject.zl = 0.5
        self.lobject.add_srcimg((24, 36), unit='pixels')
        filename = self.lobject.jsonify(savename='test.json')
        print(">>> {}".format(filename))
        with open(filename, 'r') as j:
            jcopy = LensObject.from_json(j, **self.v)
            self.assertEqual(jcopy, self.lobject)
            self.assertFalse(jcopy is self.lobject)
            self.assertEqual(jcopy.zl, self.lobject.zl)
            self.assertEqual(jcopy.srcimgs, self.lobject.srcimgs)
        try:
            os.remove(filename)
        except OSError:
            pass

    def test_jsonify(self):
        """ # jsonify """
        print(">>> {}".format(self.lobject))
        self.lobject.zl = 0.5
        self.lobject.add_srcimg((24, 36), unit='pixels')
        jsnstr = self.lobject.jsonify(**self.v)
        self.assertIsInstance(jsnstr, str)

    def test_check_path(self):
        """ # check_path """
        print(">>> {}".format(self.test_fits))
        fpath = LensObject.check_path(self.test_fits, **self.v)
        self.assertEqual(fpath, self.test_fits)

    def test_parse_fitsfile(self):
        """ # parse_fitsfile """
        print(">>> {}".format(self.test_fits))
        pdta, phdr = LensObject.parse_fitsfile(self.test_fits, header=True, **self.v)
        self.assertIsNotNone(phdr)
        self.assertEqual(pdta.shape, (128, 128))
        self.assertIsInstance(pdta, np.ndarray)
        self.assertIsInstance(phdr, dict)

    def test_mag_formula(self):
        """ # mag_formula """
        print(">>> {}".format(1))
        self.assertEqual(self.lobject.mag_formula(1, **self.v), self.lobject.photzp)
        print(">>> {}".format(0))
        self.assertEqual(self.lobject.mag_formula(0, **self.v), np.inf)
        print(">>> {}".format(1e12))
        self.assertEqual(self.lobject.mag_formula(10**(self.lobject.photzp/2.5), **self.v), 0)
        print(">>> {}".format(10))
        self.assertEqual(self.lobject.mag_formula(10, **self.v), self.lobject.photzp-2.5)

    def test_total_magnitude(self):
        """ # total_magnitude """
        print(">>> {}".format(0))
        self.assertEqual(self.lobject.total_magnitude(0, **self.v), np.inf)
        print(">>> {}".format(128))
        self.assertEqual(self.lobject.total_magnitude(128, **self.v), 22.723195552825928)
        print(">>> {}".format(10000))
        self.assertEqual(self.lobject.total_magnitude(10000, **self.v), 22.723195552825928)

    def test_cutout(self):
        """ # cutout """
        print(">>> {}".format(10))
        self.assertEqual(self.lobject.cutout(10, **self.v).shape, (10, 10))
        print(">>> {}".format(5))
        self.assertEqual(self.lobject.cutout(5, **self.v).shape, (5, 5))
        print(">>> {}".format(1))
        self.assertEqual(self.lobject.cutout(1, **self.v).shape, (1, 1))
        print(">>> {}".format(0))
        self.assertEqual(self.lobject.cutout(0, **self.v).shape, (0, 0))

    def test_gain(self):
        """ # gain """
        print(">>> {}".format(5))
        gain = self.lobject.gain(5, **self.v)
        self.assertIsInstance(gain, float)
        self.assertGreater(gain, 0)
        print(">>> {}".format(10))
        gain = self.lobject.gain(10, **self.v)
        self.assertIsInstance(gain, float)
        self.assertGreater(gain, 0)
        print(">>> {}".format(20))
        gain = self.lobject.gain(20, **self.v)
        self.assertIsInstance(gain, float)
        self.assertGreater(gain, 0)
        print(">>> {}".format(40))
        gain = self.lobject.gain(40, (20, 20), **self.v)
        self.assertIsInstance(gain, float)
        self.assertGreater(gain, 0)

    def test_pxscale_from_hdr(self):
        """ # pxscale_from_hdr """
        print(">>> {}".format(self.lobject.hdr))
        scale = LensObject.pxscale_from_hdr(self.lobject.hdr)
        self.assertListEqual(scale, [0.185733387468, 0.185733387468])

    def test_crota2_from_hdr(self):
        """ # crota2_from_hdr """
        print(">>> {}".format(self.lobject.hdr))
        crota2 = LensObject.crota2_from_hdr(self.lobject.hdr, **self.v)
        self.assertListEqual(crota2, [0, 0])

    def test_refpx_from_hdr(self):
        """ # refpx_from_hdr """
        print(">>> {}".format(self.lobject.hdr))
        refpx = LensObject.refpx_from_hdr(self.lobject.hdr, **self.v)
        self.assertListEqual(refpx, [-2710.0, 3245.0])

    def test_mag_formula_from_hdr(self):
        """ # mag_formula_from_hdr """
        print(">>> {}".format(self.lobject.hdr))
        formula = LensObject.mag_formula_from_hdr(self.lobject.hdr, **self.v)
        self.assertTrue(hasattr(formula, '__call__'))

    def test_p2skycoords(self):
        """ # p2skycoords """
        print(">>> {}".format((-1.234, 0.567)))
        skyc = self.lobject.p2skycoords((-1.234, 0.567), unit='arcsec', relative=True, **self.v)
        self.assertIsInstance(skyc, SkyCoords)

    def test_add_srcimg(self):
        """ # add_srcimg """
        print(">>> {}".format((-1.234, 0.567)))
        before = self.lobject.srcimgs[:]
        srcp = self.lobject.add_srcimg((-1.234, 0.567), unit='arcsec', relative=True, **self.v)
        after = self.lobject.srcimgs
        self.assertIsInstance(srcp, SkyCoords)
        self.assertEqual(len(before)+1, len(after))

    def test_src_shifts(self):
        """ # src_shifts """
        print(">>> {}".format(()))
        self.lobject.add_srcimg((-1.234, 0.567), unit='arcsec', relative=True)
        shifts = self.lobject.src_shifts(unit='arcsec', **self.v)
        self.assertIsInstance(shifts, list)

    def test_show_f(self):
        """ # plot_f """
        print(">>> {}".format(self.lobject))
        fig, ax = self.lobject.show_f(savefig='test.pdf', lens=True, source_images=True, **self.v)
        self.assertIsNotNone(fig)
        self.assertIsNotNone(ax)
        try:
            os.remove('test.pdf')
        except OSError:
            pass

if __name__ == "__main__":
    TestLensObject.main(verbosity=1)
