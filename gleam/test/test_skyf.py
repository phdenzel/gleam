#!/usr/bin/env python
"""
@author: phdenzel

Make really, really sure you...
Learn everything about .fits files with SkyF
"""
###############################################################################
# Imports
###############################################################################
from gleam.skyf import SkyF
import os
import numpy as np
from PIL import Image
from gleam.test.utils import UnitTestPrototype


# CLASS TESTING ###############################################################
class TestSkyF(UnitTestPrototype):

    def setUp(self):
        # arguments and keywords
        self.test_fits = os.path.abspath(os.path.dirname(__file__)) \
                         + '/W3+3-2.U.12907_13034_7446_7573.fits'
        # __init__ test
        self.skyf = SkyF(self.test_fits)
        # verbosity
        self.v = {'verbose': 1}
        print("")
        print(self.separator)
        print(self.shortDescription())

    def tearDown(self):
        print("")

    def test_SkyF(self):
        """ # SkyF """
        print(">>> {}".format(self.test_fits))
        skyf = SkyF(self.test_fits, **self.v)
        self.assertIsInstance(skyf, SkyF)
        self.assertEqual(skyf, self.skyf)

    def test_copy(self):
        """ # copy """
        print(">>> {}".format(self.skyf))
        self.skyf.photzp = 25.67
        copy = self.skyf.copy(**self.v)
        self.assertEqual(copy, self.skyf)
        self.assertFalse(copy is self.skyf)
        self.assertEqual(copy.photzp, self.skyf.photzp)

    def test_deepcopy(self):
        """ # deepcopy """
        print(">>> {}".format(self.skyf))
        self.skyf.photzp = 25.67
        copy = self.skyf.deepcopy(**self.v)
        self.assertEqual(copy, self.skyf)
        self.assertFalse(copy is self.skyf)
        self.assertEqual(copy.photzp, self.skyf.photzp)

    def test_from_json(self):
        """ # from_json """
        filename = 'test.json'
        self.skyf.photzp = 25.67
        self.skyf.roi.select['circle']((64, 64), 20)
        self.skyf.roi.select['polygon']((64, 64), (32, 32), (32, 64))
        self.skyf.roi.select['rect']((32, 32), (64, 64))
        self.skyf.light_model = {'sersic': {
            'phi': 1.8052729641945808, 'e': 0.3648069769263307,
            'r_s': 18.520443508692196, 'n': 1.12780508392599,
            'c_0': 0.0, 'y': 68, 'x': 61, 'I_0': 35.71338473276946}}
        filename = self.skyf.jsonify(name='test.json')
        print(">>> {}".format(filename))
        with open(filename, 'r') as j:
            jcopy = SkyF.from_json(j, **self.v)
            self.assertEqual(jcopy, self.skyf)
            self.assertFalse(jcopy is self.skyf)
            self.assertEqual(jcopy.photzp, self.skyf.photzp)
            self.assertEqual(jcopy.roi, self.skyf.roi)
        try:
            os.remove(filename)
        except OSError:
            pass

    def test_jsonify(self):
        """ # jsonify """
        print(">>> {}".format(self.skyf))
        self.skyf.roi.select['circle']((64, 64), 20)
        self.skyf.roi.select['polygon']((64, 64), (32, 32), (32, 64))
        self.skyf.roi.select['rect']((32, 32), (64, 64))
        self.skyf.light_model = {'sersic': {
            'phi': 1.8052729641945808, 'e': 0.3648069769263307,
            'r_s': 18.520443508692196, 'n': 1.12780508392599,
            'c_0': 0.0, 'y': 68, 'x': 61, 'I_0': 35.71338473276946}}
        jsnstr = self.skyf.jsonify(**self.v)
        self.assertIsInstance(jsnstr, str)

    def test_check_path(self):
        """ # check_path """
        print(">>> {}".format(self.test_fits))
        fpath = SkyF.check_path(self.test_fits, **self.v)
        self.assertEqual(fpath, self.test_fits)

    def test_parse_fitsfile(self):
        """ # parse_fitsfile """
        print(">>> {}".format(self.test_fits))
        pdta, phdr = SkyF.parse_fitsfile(self.test_fits, header=True, **self.v)
        self.assertIsNotNone(phdr)
        self.assertEqual(pdta.shape, (128, 128))
        self.assertIsInstance(pdta, np.ndarray)
        self.assertIsInstance(phdr, dict)

    def test_mag_formula(self):
        """ # mag_formula """
        print(">>> {}".format(1))
        self.assertEqual(self.skyf.mag_formula(1, **self.v), self.skyf.photzp)
        print(">>> {}".format(0))
        self.assertEqual(self.skyf.mag_formula(0, **self.v), np.inf)
        print(">>> {}".format(1e12))
        self.assertEqual(self.skyf.mag_formula(10**(self.skyf.photzp/2.5), **self.v), 0)
        print(">>> {}".format(10))
        self.assertEqual(self.skyf.mag_formula(10, **self.v), self.skyf.photzp-2.5)

    def test_total_magnitude(self):
        """ # total_magnitude """
        print(">>> {}".format(0))
        self.assertEqual(self.skyf.total_magnitude(0, **self.v), np.inf)
        print(">>> {}".format(128))
        self.assertEqual(self.skyf.total_magnitude(128, **self.v), 22.723195552825928)
        print(">>> {}".format(10000))
        self.assertEqual(self.skyf.total_magnitude(10000, **self.v), 22.723195552825928)

    def test_roi(self):
        """ # roi """
        print(">>> {}, {}".format('circle', [(64, 64), 20]))
        self.skyf.roi.select['circle']((64, 64), 20, **self.v)
        print(">>> {}, {}".format('polygon', [(64, 64), (32, 32), (32, 64)]))
        self.skyf.roi.select['polygon']((64, 64), (32, 32), (32, 64), **self.v)
        print(">>> {}, {}".format('rect', [(32, 32), (64, 64)]))
        self.skyf.roi.select['rect']((32, 32), (64, 64), **self.v)

    def test_cutout(self):
        """ # cutout """
        print(">>> {}".format(10))
        self.assertEqual(self.skyf.cutout(10, **self.v).shape, (10, 10))
        print(">>> {}".format(5))
        self.assertEqual(self.skyf.cutout(5, **self.v).shape, (5, 5))
        print(">>> {}".format(1))
        self.assertEqual(self.skyf.cutout(1, **self.v).shape, (1, 1))
        print(">>> {}".format(0))
        self.assertEqual(self.skyf.cutout(0, **self.v).shape, (0, 0))

    def test_gain(self):
        """ # gain """
        print(">>> {}".format(5))
        gain = self.skyf.gain(5, **self.v)
        self.assertIsInstance(gain, float)
        self.assertGreater(gain, 0)
        print(">>> {}".format(10))
        gain = self.skyf.gain(10, **self.v)
        self.assertIsInstance(gain, float)
        self.assertGreater(gain, 0)
        print(">>> {}".format(20))
        gain = self.skyf.gain(20, **self.v)
        self.assertIsInstance(gain, float)
        self.assertGreater(gain, 0)
        print(">>> {}".format(40))
        gain = self.skyf.gain(40, (20, 20), **self.v)
        self.assertIsInstance(gain, float)
        self.assertGreater(gain, 0)

    def test_yx2idx(self):
        """ # yx2idx """
        print(">>> {}".format((0, 0)))
        i = self.skyf.yx2idx(0, 0, **self.v)
        self.assertIsInstance(i, int)
        self.assertEqual(i, 0)
        # if self.skyf.naxis1 and self.skyf.naxis2 are uneven, the tests will fail!
        print(">>> {}".format((self.skyf.naxis1-1, self.skyf.naxis2-1)))
        i = self.skyf.yx2idx(self.skyf.naxis1-1, self.skyf.naxis2-1, **self.v)
        self.assertIsInstance(i, int)
        self.assertEqual(i, self.skyf.naxis1*self.skyf.naxis2-1)
        print(">>> {}".format((self.skyf.naxis1//2, self.skyf.naxis2//2)))
        i = self.skyf.yx2idx(self.skyf.naxis1//2, self.skyf.naxis2//2, **self.v)
        self.assertIsInstance(i, int)
        self.assertEqual(i, ((self.skyf.naxis1+1)*self.skyf.naxis2)//2)

    def test_idx2yx(self):
        """ # idx2yx """
        print(">>> {}".format(0))
        yx = self.skyf.idx2yx(0, **self.v)
        self.assertIsInstance(yx, tuple)
        self.assertEqual(yx, (0, 0))
        print(">>> {}".format(self.skyf.naxis1*self.skyf.naxis2-1))
        yx = self.skyf.idx2yx(self.skyf.naxis1*self.skyf.naxis2-1, **self.v)
        self.assertIsInstance(yx, tuple)
        self.assertEqual(yx, (self.skyf.naxis1-1, self.skyf.naxis2-1))
        print(">>> {}".format(((self.skyf.naxis1+1)*self.skyf.naxis2)//2))
        yx = self.skyf.idx2yx(((self.skyf.naxis1+1)*self.skyf.naxis2)//2, **self.v)
        self.assertIsInstance(yx, tuple)
        self.assertEqual(yx, (self.skyf.naxis1//2, self.skyf.naxis2//2))

    def test_theta(self):
        """ # theta """
        print(">>> {}".format(((self.skyf.naxis1+1)*self.skyf.naxis2)//2))
        t = self.skyf.theta(0, **self.v)
        self.assertTrue(len(t) == 2)
        print(">>> {}".format(((self.skyf.naxis1+1)*self.skyf.naxis2)//2))
        t = self.skyf.theta(((self.skyf.naxis1+1)*self.skyf.naxis2)//2, **self.v)
        self.assertTrue(len(t) == 2)
        print(">>> {}".format(self.skyf.center.xy))
        t = self.skyf.theta(self.skyf.center.xy, **self.v)
        self.assertTrue(len(t) == 2)
        print(">>> {}, {}".format([36, 65], [61, 68]))
        theta = self.skyf.theta([36, 65], origin=[61, 68], **self.v)
        self.assertEqual(len(theta), 2)

    def test_pxscale_from_hdr(self):
        """ # pxscale_from_hdr """
        print(">>> {}".format(self.skyf.hdr))
        scale = SkyF.pxscale_from_hdr(self.skyf.hdr)
        self.assertListEqual(scale, [0.185733387468, 0.185733387468])

    def test_crota2_from_hdr(self):
        """ # crota2_from_hdr """
        print(">>> {}".format(self.skyf.hdr))
        crota2 = SkyF.crota2_from_hdr(self.skyf.hdr, **self.v)
        self.assertListEqual(crota2, [0, 0])

    def test_refpx_from_hdr(self):
        """ # refpx_from_hdr """
        print(">>> {}".format(self.skyf.hdr))
        refpx = SkyF.refpx_from_hdr(self.skyf.hdr, **self.v)
        self.assertListEqual(refpx, [-2710.0, 3245.0])

    def test_mag_formula_from_hdr(self):
        """ # mag_formula_from_hdr """
        print(">>> {}".format(self.skyf.hdr))
        formula = SkyF.mag_formula_from_hdr(self.skyf.hdr, **self.v)
        self.assertTrue(hasattr(formula, '__call__'))

    def test_show_f(self):
        """ # plot_f """
        print(">>> {}".format(self.skyf))
        fig, ax = self.skyf.show_f(savefig='test.pdf', **self.v)
        self.assertIsNotNone(fig)
        self.assertIsNotNone(ax)
        try:
            os.remove('test.pdf')
        except OSError:
            pass

    def test_image_f(self):
        """ # image_f """
        self.skyf.roi.select['circle']((64, 64), 20, **self.v)
        self.skyf.roi.select['polygon']((64, 32), (32, 32), (32, 64), **self.v)
        self.skyf.roi.select['rect']((28, 28), (82, 82), **self.v)
        self.skyf.roi.close_all()
        img = self.skyf.image_f(draw_roi=True)
        # img.show()


if __name__ == "__main__":
    TestSkyF.main(verbosity=1)
