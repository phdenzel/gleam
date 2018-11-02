#!/usr/bin/env python
"""
@author: phdenzel

Make really, really sure you...
Find your way in the sky with SkyCoords
"""
###############################################################################
# Imports
###############################################################################
from gleam.skycoords import SkyCoords
from gleam.test.utils import UnitTestPrototype


# CLASS TESTING ###############################################################
class TestSkyCoords(UnitTestPrototype):

    def setUp(self):
        # arguments and keywords
        self.kw = {
            'px2arcsec_scale': [0.185733387468, 0.185733387468],
            'reference_value': [218.9618304, 52.64492319],
            'reference_pixel': [-2710.0, 3245.0]
        }
        self.ek = {
            'px2arcsec_scale': [None, None],
            'reference_value': [None, None],
            'reference_pixel': [None, None]
        }
        # __init__ test
        self.skyc = SkyCoords(218.72676666666666, 52.480555555555554, **self.kw)
        # verbosity
        self.v = {'verbose': 1}
        self.kw.update(self.v)
        self.ek.update(self.v)
        print("")
        print(self.separator)
        print(self.shortDescription())

    def tearDown(self):
        print("")

    def test_empty(self):
        """ # empty """
        print(">>> {}".format(None))
        skyc = SkyCoords.empty(**self.v)
        self.assertIsInstance(skyc, SkyCoords)
        self.assertEqual(skyc.dec, None)
        self.assertEqual(skyc.ra, None)

    def test_copy(self):
        """ # copy """
        print(">>> {}".format(()))
        copy = self.skyc.copy(**self.v)
        self.assertEqual(copy, self.skyc)
        self.assertFalse(copy is self.skyc)

    def test_deepcopy(self):
        """ # deepcopy """
        print(">>> {}".format(()))
        copy = self.skyc.deepcopy(**self.v)
        self.assertEqual(copy, self.skyc)
        self.assertFalse(copy is self.skyc)

    def test_from_J2000(self):
        """ # from_J2000"""
        print(">>> {}".format("J143454.4+522850"))
        skyc = SkyCoords.from_J2000("J143454.4+522850", **self.kw)
        self.assertIsInstance(skyc, SkyCoords)
        self.assertEqual(skyc, self.skyc)
        self.assertEqual(skyc.ra, 218.72676666666666)
        self.assertEqual(skyc.dec, 52.480555555555554)
        self.assertEqual(skyc.arcsecs, [787416.36, 188930.0])
        self.assertEqual(skyc.xy, [54.45757848892799, 59.12430976120095])
        self.assertEqual(skyc.J2000C.J2000, "J143454.4+522850")
        self.assertEqual(skyc.J2000C.c1_str, "14:34:54.4")
        self.assertEqual(skyc.J2000C.c2_str, "52:28:50.0")
        # using None
        print(">>> {}".format(None))
        empty = SkyCoords.from_J2000(None, **self.ek)
        self.assertIsInstance(empty, SkyCoords)
        self.assertEqual(empty.ra, None)
        self.assertEqual(empty.dec, None)
        self.assertEqual(empty.arcsecs, [None, None])
        self.assertEqual(empty.xy, [None, None])
        self.assertEqual(empty.J2000C.J2000, None)
        self.assertEqual(empty.J2000C.c1_str, None)
        self.assertEqual(empty.J2000C.c2_str, None)

    def test_from_arcsec(self):
        """ # from_arcsec """
        print(">>> {}".format((787416.36, 188930.0)))
        skyc = SkyCoords.from_arcsec(787416.36, 188930.0, **self.kw)
        self.assertIsInstance(skyc, SkyCoords)
        self.assertEqual(skyc, self.skyc)
        self.assertEqual(skyc.ra, 218.72676666666666)
        self.assertEqual(skyc.dec, 52.480555555555554)
        self.assertEqual(skyc.arcsecs, [787416.36, 188930.0])
        self.assertEqual(skyc.xy, [54.45757848892799, 59.12430976120095])
        self.assertEqual(skyc.J2000C.J2000, "J143454.4+522850")
        self.assertEqual(skyc.J2000C.c1_str, "14:34:54.4")
        self.assertEqual(skyc.J2000C.c2_str, "52:28:50.0")
        # using None
        print(">>> {}".format((None, None)))
        empty = SkyCoords.from_arcsec(None, None, **self.ek)
        self.assertIsInstance(empty, SkyCoords)
        self.assertEqual(empty.ra, None)
        self.assertEqual(empty.dec, None)
        self.assertEqual(empty.arcsecs, [None, None])
        self.assertEqual(empty.xy, [None, None])
        self.assertEqual(empty.J2000C.J2000, None)
        self.assertEqual(empty.J2000C.c1_str, None)
        self.assertEqual(empty.J2000C.c2_str, None)

    def test_from_degrees(self):
        """ # from_degrees """
        print(">>> {}".format((218.72676666666666, 52.480555555555554)))
        skyc = SkyCoords.from_degrees(218.72676666666666, 52.480555555555554, **self.kw)
        self.assertIsInstance(skyc, SkyCoords)
        self.assertEqual(skyc, self.skyc)
        self.assertEqual(skyc.ra, 218.72676666666666)
        self.assertEqual(skyc.dec, 52.480555555555554)
        self.assertEqual(skyc.arcsecs, [787416.36, 188930.0])
        self.assertEqual(skyc.xy, [54.45757848892799, 59.12430976120095])
        self.assertEqual(skyc.J2000C.J2000, "J143454.4+522850")
        self.assertEqual(skyc.J2000C.c1_str, "14:34:54.4")
        self.assertEqual(skyc.J2000C.c2_str, "52:28:50.0")
        # using None
        print(">>> {}".format((None, None)))
        empty = SkyCoords.from_degrees(None, None, **self.ek)
        self.assertIsInstance(empty, SkyCoords)
        self.assertEqual(empty.ra, None)
        self.assertEqual(empty.dec, None)
        self.assertEqual(empty.arcsecs, [None, None])
        self.assertEqual(empty.xy, [None, None])
        self.assertEqual(empty.J2000C.J2000, None)
        self.assertEqual(empty.J2000C.c1_str, None)
        self.assertEqual(empty.J2000C.c2_str, None)

    def test_from_pixels(self):
        """ # from_pixels """
        print(">>> {}".format((54.45757848892799, 59.12430976120095)))
        skyc = SkyCoords.from_pixels(54.45757848892799, 59.12430976120095, **self.kw)
        self.assertIsInstance(skyc, SkyCoords)
        self.assertEqual(skyc, self.skyc)
        self.assertEqual(skyc.ra, 218.72676666666666)
        self.assertEqual(skyc.dec, 52.480555555555554)
        self.assertEqual(skyc.arcsecs, [787416.36, 188930.0])
        self.assertEqual(skyc.xy, [54.45757848892799, 59.12430976120095])
        self.assertEqual(skyc.J2000C.J2000, "J143454.4+522850")
        self.assertEqual(skyc.J2000C.c1_str, "14:34:54.4")
        self.assertEqual(skyc.J2000C.c2_str, "52:28:50.0")
        # using None
        print(">>> {}".format((None, None)))
        empty = SkyCoords.from_degrees(None, None, **self.ek)
        self.assertIsInstance(empty, SkyCoords)
        self.assertEqual(empty.ra, None)
        self.assertEqual(empty.dec, None)
        self.assertEqual(empty.arcsecs, [None, None])
        self.assertEqual(empty.xy, [None, None])
        self.assertEqual(empty.J2000C.J2000, None)
        self.assertEqual(empty.J2000C.c1_str, None)
        self.assertEqual(empty.J2000C.c2_str, None)

    def test_J2000(self):
        """ # J2000 """
        print(">>> {}".format((218.72676666666666, 52.480555555555554)))
        j2000 = SkyCoords.J2000(218.72676666666666, 52.480555555555554, **self.v)
        self.assertIsInstance(j2000, SkyCoords.J2000)
        self.assertEqual(j2000, self.skyc.J2000C)
        self.assertEqual(j2000.c1, [14.0, 34.0, 54.4])
        self.assertEqual(j2000.c2, [52.0, 28.0, 50.0])
        self.assertEqual(j2000.c1_str, "14:34:54.4")
        self.assertEqual(j2000.c2_str, "52:28:50.0")
        # using None
        print(">>> {}".format((None, None)))
        empty = SkyCoords.J2000(None, None, **self.v)
        self.assertIsInstance(empty, SkyCoords.J2000)
        self.assertEqual(empty.c1, [None, None, None])
        self.assertEqual(empty.c2, [None, None, None])
        self.assertEqual(empty.c1_str, None)
        self.assertEqual(empty.c2_str, None)


    def test_J2000_2arcsec(self):
        """ # J2000._2arcsec """
        print(">>> {}".format("J143454.4+522850"))
        arcs = SkyCoords.J2000._2arcsec("J143454.4+522850", **self.v)
        self.assertEqual(arcs, [787416.36, 188930.0])
        # using None
        print(">>> {}".format(None))
        empty = SkyCoords.J2000._2arcsec(None, **self.v)
        self.assertEqual(empty, [None, None])
    
    def test_J2000_2deg(self):
        """ # J2000._2deg """
        print(">>> {}".format("J143454.4+522850"))
        degs = SkyCoords.J2000._2deg("J143454.4+522850", **self.v)
        self.assertEqual(degs, [218.72676666666666, 52.480555555555554])
        # using None
        print(">>> {}".format(None))
        empty = SkyCoords.J2000._2deg(None, **self.v)
        self.assertEqual(empty, [None, None])

    def test_J2000_2pixels(self):
        """ # J2000._2pixels """
        print(">>> {}".format("J143454.4+522850"))
        pxls = SkyCoords.J2000._2pixels("J143454.4+522850", **self.kw)
        self.assertEqual(pxls, [54.45757848892799, 59.12430976120095])
        # using None
        print(">>> {}".format(None))
        empty = SkyCoords.J2000._2deg(None, **self.v)
        self.assertEqual(empty, [None, None])

    def test_deg2J2000(self):
        """ # deg2J2000 """
        print(">>> {}".format((218.72676666666666, 52.480555555555554)))
        j2ks = SkyCoords.deg2J2000(218.72676666666666, 52.480555555555554, **self.v)
        self.assertEqual(j2ks, "J143454.4+522850")
        # using None
        print(">>> {}".format((None, None)))
        empty = SkyCoords.deg2J2000(None, None, **self.v)
        self.assertEqual(empty, None)

    def test_deg2arcsec(self):
        """ # deg2arcsec """
        print(">>> {}".format((218.72676666666666, 52.480555555555554)))
        arcs = SkyCoords.deg2arcsec(218.72676666666666, 52.480555555555554, **self.v)
        self.assertEqual(arcs, [787416.36, 188930.0])
        # using None
        print(">>> {}".format((None, None)))
        empty = SkyCoords.deg2arcsec(None, None, **self.v)
        self.assertEqual(empty, [None, None])

    def test_deg2pixels(self):
        """ # deg2pixels """
        print(">>> {}".format((218.72676666666666, 52.480555555555554)))
        pxls = SkyCoords.deg2pixels(218.72676666666666, 52.480555555555554, **self.kw)
        self.assertEqual(pxls, [54.45757848892799, 59.12430976120095])
        # using None
        print(">>> {}".format((None, None)))
        empty = SkyCoords.deg2pixels(None, None, **self.ek)
        self.assertEqual(empty, [None, None])

    def test_arcsecs2J2000(self):
        """ # arcsec2J2000 """
        print(">>> {}".format((787416.36, 188930.0)))
        j2ks = SkyCoords.arcsec2J2000(787416.36, 188930.0, **self.v)
        self.assertEqual(j2ks, "J143454.4+522850")
        # using None
        print(">>> {}".format((None, None)))
        empty = SkyCoords.arcsec2J2000(None, None, **self.v)
        self.assertEqual(empty, None)

    def test_arcsec2deg(self):
        """ # arcsec2deg """
        print(">>> {}".format((787416.36, 188930.0)))
        degs = SkyCoords.arcsec2deg(787416.36, 188930.0, **self.v)
        self.assertEqual(degs, [218.72676666666666, 52.480555555555554])
        # using None
        print(">>> {}".format((None, None)))
        empty = SkyCoords.arcsec2deg(None, None, **self.v)
        self.assertEqual(empty, [None, None])

    def test_arcsec2pixels(self):
        """ # arcsec2pixels """
        print(">>> {}".format((787416.36, 188930.0)))
        pxls = SkyCoords.arcsec2pixels(787416.36, 188930.0, **self.kw)
        self.assertEqual(pxls, [54.45757848892799, 59.12430976120095])
        # using None
        print(">>> {}".format((None, None)))
        empty = SkyCoords.arcsec2pixels(None, None, **self.ek)
        self.assertEqual(empty, [None, None])

    def test_pixels2J2000(self):
        """ # pixels2J2000 """
        print(">>> {}".format((54.45757848892799, 59.12430976120095)))
        j2ks = SkyCoords.pixels2J2000([54.45757848892799, 59.12430976120095], **self.kw)
        self.assertEqual(j2ks, "J143454.4+522850")
        # using None
        print(">>> {}".format((None, None)))
        empty = SkyCoords.pixels2J2000([None, None], **self.ek)
        self.assertEqual(empty, None)

    def test_pixels2arcsec(self):
        """ # pixels2arcsec """
        print(">>> {}".format((54.45757848892799, 59.12430976120095)))
        arcs = SkyCoords.pixels2arcsec([54.45757848892799, 59.12430976120095], **self.kw)
        self.assertEqual(arcs, [787416.36, 188930.0])
        # using None
        print(">>> {}".format((None, None)))
        empty = SkyCoords.pixels2arcsec([None, None], **self.ek)
        self.assertEqual(empty, [None, None])

    def test_pixels2deg(self):
        """ # pixels2deg """
        print(">>> {}".format((54.45757848892799, 59.12430976120095)))
        degs = SkyCoords.pixels2deg([54.45757848892799, 59.12430976120095], **self.kw)
        self.assertEqual(degs, [218.72676666666666, 52.480555555555554])
        # using None
        print(">>> {}".format((None, None)))
        empty = SkyCoords.pixels2deg([None, None], **self.ek)
        self.assertEqual(empty, [None, None])

    def test_deg2rad(self):
        """ # deg2rad """
        print(">>> {}".format((218.72676666666666, 52.480555555555554)))
        rads = SkyCoords.deg2rad(218.72676666666666, 52.480555555555554, **self.v)
        self.assertEqual(rads, [3.8175022405747154, 0.9159584877202462])

    def test_rad2deg(self):
        """ # rad2deg """
        print(">>> {}".format((3.8175022405747154, 0.9159584877202462)))
        degs = SkyCoords.rad2deg(3.8175022405747154, 0.9159584877202462, **self.v)
        self.assertEqual(degs, [218.72676666666666, 52.480555555555554])

    def test_add(self):
        """ # __add__ """
        print(">>> {}, {}".format(self.skyc, [0.1, 0.1]))
        shift = self.skyc.__add__([0.1, 0.1])
        print(shift)
        self.assertEqual(shift, [218.82676666666666, 52.580555555555554])

    def test_distance(self):
        """ # distance """
        print(">>> {}, {}".format(self.skyc, self.skyc))
        d = self.skyc.distance(self.skyc, **self.v)
        self.assertEqual(d, 0)

    def test_angle(self):
        """ # angle """
        print(">>> {}, {}".format(self.skyc, self.skyc))
        a = self.skyc.angle(self.skyc, **self.v)
        self.assertEqual(a, 0)

    def test_shift(self):
        """ # shift """
        print(">>> {}, {}".format(self.skyc, [0, 0]))
        shifted = self.skyc.shift([0, 0], **self.v)
        self.assertEqual(self.skyc, self.skyc)

    def test_get_shift_to(self):
        """ # get_shift_to """
        print(">>> {}, {}".format(self.skyc, self.skyc))
        shift = self.skyc.get_shift_to(self.skyc, **self.v)
        self.assertEqual(shift, [0, 0])


if __name__ == "__main__":
    TestSkyCoords.main(verbosity=1)
