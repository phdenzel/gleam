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
from matplotlib import pyplot as plt
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
        self.skyp = SkyPatch(self.test_ftss)
        self.roi = ROISelector(self.skyf.data)
        # verbosity
        self.v = {'verbose': 1}
        print("")
        print(self.separator)
        print(self.shortDescription())

    def tearDown(self):
        print("")

    def test_ROISelector(self):
        """ # ROISelector """
        # None
        print(">>> {}".format(None))
        roi = ROISelector(None, **self.v)
        self.assertIsInstance(roi, ROISelector)
        self.assertEqual(roi.data, np.zeros((1, 1)))
        # with data shape (128, 128)
        print(">>> {}".format(self.skyf.data))
        roi = ROISelector(self.skyf.data, **self.v)
        self.assertIsInstance(roi, ROISelector)
        self.assertIsNotNone(roi.data)
        # with data shape (128, 128, 2)
        print(">>> {}".format(self.skyp.data))
        roi = ROISelector(self.skyp.data, **self.v)
        self.assertIsInstance(roi, ROISelector)
        self.assertIsNotNone(roi.data)

    def test_from_gleamobj(self):
        """ # from_gleamobj """
        # from skyf
        print(">>> {}".format(self.skyf))
        roi = ROISelector.from_gleamobj(self.skyf, **self.v)
        self.assertIsInstance(roi, ROISelector)
        self.assertIsNotNone(roi.data)
        # from skypatch
        print(">>> {}".format(self.skyp))
        roi = ROISelector.from_gleamobj(self.skyp, **self.v)
        self.assertIsInstance(roi, ROISelector)
        self.assertIsNotNone(roi.data)

    def test_select_circle(self):
        """ # select_circle """
        # on shape (128, 128)
        center, radius = (0, 128), 30
        print(">>> {}, {}".format(center, radius))
        s = self.roi.select_circle(center, radius, **self.v)
        self.assertIsInstance(s, np.ndarray)
        self.assertEqual(s.dtype, bool)
        self.assertTrue(np.any(s))
        self.roi.data[self.roi()] = 100
        plt.imshow(self.roi.data, cmap='bone', origin='lower')
        if 0:
            plt.show()
        plt.close()
        # on shape (200, 300, 4)
        W, H = 300, 200
        e = np.ones((H, W, 4), dtype=float)
        center, radius = (50, 150), 40
        print(">>> {}, {}".format(center, radius))
        roi = ROISelector(e)
        s = roi.select['circle'](center, radius, **self.v)
        e[~roi()] = 0
        e[:, :, 3] = 1
        plt.imshow(e, origin='lower')
        if 0:
            plt.show()
        plt.close()

    def test_select_rect(self):
        """ # select_rect """
        # on shape (128, 128)
        anchor, dv = (0, 0), (32, 64)
        print(">>> {}, {}".format(anchor, dv))
        s = self.roi.select_rect(anchor, dv, **self.v)
        self.assertIsInstance(s, np.ndarray)
        self.assertEqual(s.dtype, bool)
        self.assertTrue(np.any(s))
        self.roi.data[self.roi()] = 100
        plt.imshow(self.roi.data, cmap='bone', origin='lower')
        if 0:
            plt.show()
        plt.close()
        # on shape (200, 300, 4)
        W, H = 300, 200
        e = np.ones((H, W, 4), dtype=float)
        anchor, dv = (0, 0), (100, 150)
        print(">>> {}, {}".format(anchor, dv))
        roi = ROISelector(e)
        s = roi.select['rect'](anchor, dv, **self.v)
        e[~roi()] = 0, 0, 1, 0
        e[:, :, 3] = 1
        plt.imshow(e, origin='lower')
        if 0:
            plt.show()
        plt.close()

    def test_select_square(self):
        """ # select_square """
        # on shape (128, 128)
        anchor, dv = (0, 79), 48
        print(">>> {}, {}".format(anchor, dv))
        s = self.roi.select_square(anchor, dv, **self.v)
        self.assertIsInstance(s, np.ndarray)
        self.assertEqual(s.dtype, bool)
        self.assertTrue(np.any(s))
        self.roi.data[self.roi()] = 100
        plt.imshow(self.roi.data, cmap='bone', origin='lower')
        if 0:
            plt.show()
        plt.close()
        # on shape (200, 300, 4)
        W, H = 300, 200
        e = np.ones((H, W, 4), dtype=float)
        anchor, dv = (0, 100), 100
        print(">>> {}, {}".format(anchor, dv))
        roi = ROISelector(e)
        s = roi.select['square'](anchor, dv, **self.v)
        e[~roi()] = 0, 0, 1, 0
        e[:, :, 3] = 1
        plt.imshow(e, origin='lower')
        if 0:
            plt.show()
        plt.close()

    def test_select_polygon(self):
        """ # test_polygon """
        # on shape (128, 128)
        polygon = [(43, 45), (42, 54), (39, 62), (38, 71), (31, 60), (36, 51)]
        print(">>> {}".format(polygon))
        s = self.roi.select_polygon(*polygon, **self.v)
        self.assertIsInstance(s, np.ndarray)
        self.assertEqual(s.dtype, bool)
        self.assertTrue(np.any(s))
        self.roi.data[self.roi()] = 100
        plt.imshow(self.roi.data, cmap='bone', origin='lower')
        if 0:
            plt.show()
        plt.close()
        # on shape (200, 300, 4)
        W, H = 300, 200
        e = np.ones((H, W, 4), dtype=float)
        polygon = [(125, 50), (175, 50), (200, 100), (175, 150), (125, 150), (100, 100)]
        print(">>> {}".format(polygon))
        roi = ROISelector(e)
        s = roi.select['polygon'](*polygon, **self.v)
        e[~roi()] = 0, 0, 1, 0
        e[:, :, 3] = 1
        plt.imshow(e, origin='lower')
        if 0:
            plt.show()
        plt.close()

    def test_select_amorph(self):
        """ # test_amorph """
        # interrupted diagonal on shape (128, 128)
        points = [(i, i) for i in range(128)]
        points = points[0:32]+points[-32:]
        print(">>> {}".format(points))
        s = self.roi.select_amorph(*points, **self.v)
        self.assertIsInstance(s, np.ndarray)
        self.assertEqual(s.dtype, bool)
        self.assertTrue(np.any(s))
        self.roi.data[self.roi()] = 100
        plt.imshow(self.roi.data, cmap='bone', origin='lower')
        if 0:
            plt.show()
        plt.close()
        # scattered points on shape (200, 300, 4)
        W, H = 300, 200
        e = np.ones((H, W, 4), dtype=float)
        points = [(2*i, i) for i in range(128)]
        points = points[0:32]+points[-32:]
        print(">>> {}".format(points))
        roi = ROISelector(e)
        s = roi.select['amorph'](*points, **self.v)
        e[~roi()] = 0, 0, 1, 0
        e[:, :, 3] = 1
        plt.imshow(e, origin='lower')
        if 0:
            plt.show()
        plt.close()

    def test_mpl_interface(self):
        """ # mpl_interface"""
        pass


class TestPolygon(UnitTestPrototype):

    def setUp(self):
        # arguments and keywords
        self.points = [(0, 0), (1, 0), (2, 1), (3, 2.5), (3, 3), (2, 3), (2, 2.5), (0.5, 1)]
        self.polygon = ROISelector.Polygon(*self.points)
        # # verbosity
        self.v = {'verbose': 1}
        print("")
        print(self.separator)
        print(self.shortDescription())

    def tearDown(self):
        print("")

    def test_Polygon(self):
        """ # ROISelector.Polygon """
        # from list
        print(">>> {}".format(self.points))
        p = ROISelector.Polygon(self.points, **self.v)
        self.assertIsInstance(p, ROISelector.Polygon)
        self.assertEqual(p, self.polygon)
        self.assertEqual(p.N, 8)
        self.assertEqual(p.area, 3.625)
        self.assertGreater(p.is_ccw, 0)
        self.assertFalse(p.is_closed)
        # from tuple
        print(">>> {}".format(tuple(self.points)))
        p = ROISelector.Polygon(*self.points, **self.v)
        self.assertIsInstance(p, ROISelector.Polygon)
        self.assertEqual(p, self.polygon)
        self.assertEqual(p.N, 8)
        self.assertEqual(p.area, 3.625)
        self.assertGreater(p.is_ccw, 0)
        self.assertFalse(p.is_closed)
        # from single point
        print(">>> {}".format((1, 1)))
        p = ROISelector.Polygon((1, 1), **self.v)
        self.assertIsInstance(p, ROISelector.Polygon)
        self.assertEqual(p.N, 1)
        self.assertEqual(p.area, 0)
        self.assertEqual(p.is_ccw, 0)
        self.assertFalse(p.is_closed)

    def test_add_point(self):
        """ # ROISelector.Polygon.add_point """
        # add point not in polygon
        print(">>> {}".format((0.25, 0.50)))
        self.polygon.add_point((0.25, 0.50), **self.v)
        self.assertEqual(self.polygon.N, 9)
        # add point closing the polygon
        print(">>> {}".format((0, 0)))
        self.polygon.add_point((0, 0), **self.v)
        self.assertEqual(self.polygon.N, 9)

    def test_close(self):
        """ # ROISelector.Polygon.close """
        print(">>> {}".format(()))
        self.assertFalse(self.polygon.is_closed)
        p = self.polygon.close(**self.v)
        self.assertTrue(p.is_closed)

    def test_contains(self):
        """ # ROISelector.Polygon.contains """
        # multi-point input
        print(">>> {} {}".format([0, 3, 2, 7], [4, 3, 5, 2]))
        self.polygon.contains([0, 3, 2, 7], [4, 3, 5, 2], **self.v)
        # single-point input
        print(">>> {}".format((0, 4)))
        self.polygon.contains(0, 4, **self.v)


class TestRectangle(UnitTestPrototype):

    def setUp(self):
        # arguments and keywords
        self.anchor = [0, 0]
        self.rectangle = ROISelector.Rectangle(self.anchor, [1, 1])
        # # verbosity
        self.v = {'verbose': 1}
        print("")
        print(self.separator)
        print(self.shortDescription())

    def tearDown(self):
        print("")

    def test_Rectangle(self):
        """ # ROISelector.Rectangle """
        # with dv
        print(">>> {}, {}".format(self.anchor, (1, 1)))
        r = ROISelector.Rectangle(self.anchor, (1, 1), **self.v)
        self.assertIsInstance(r, ROISelector.Rectangle)
        self.assertEqual(r, self.rectangle)
        self.assertEqual(r.N, 4)
        self.assertEqual(r.area, 1)
        # with corner
        print(">>> {}, {}".format(self.anchor, {'corner': (1, 1)}))
        r = ROISelector.Rectangle(self.anchor, corner=(1, 1), **self.v)
        self.assertIsInstance(r, ROISelector.Rectangle)
        self.assertEqual(r, self.rectangle)
        self.assertEqual(r.N, 4)
        self.assertEqual(r.area, 1)
        # no second input
        print(">>> {}".format(self.anchor))
        r = ROISelector.Rectangle(self.anchor, **self.v)
        self.assertIsInstance(r, ROISelector.Rectangle)
        self.assertNotEqual(r, self.rectangle)
        self.assertEqual(r.N, 4)
        self.assertEqual(r.area, 0)

    def test_anchor(self):
        """ # ROISelector.Rectangle.anchor """
        anchor = (1, 1)
        print(">>> {}".format(anchor))
        a = self.rectangle.area
        dv = self.rectangle.dv
        self.rectangle.anchor = anchor
        print(self.rectangle.__v__)
        self.assertEqual(self.rectangle.dv, dv)
        self.assertEqual(self.rectangle.anchor, tuple(c+da for c, da in zip(self.anchor, anchor)))
        self.assertEqual(self.rectangle.area, a)

    def test_dv(self):
        """ # ROISelector.Rectangle.dv """
        dv = (2, 2)
        print(">>> {}".format(dv))
        a = self.rectangle.anchor
        area = self.rectangle.area
        self.rectangle.dv = dv
        print(self.rectangle.__v__)
        self.assertEqual(self.rectangle.dv, dv)
        self.assertEqual(self.rectangle.anchor, a)
        self.assertEqual(self.rectangle.area, area*sum(dv))

    def test_add_point(self):
        """ # ROISelector.Rectangle.add_point """
        # add point not in rectangle
        print(">>> {}".format((0.25, 0.50)))
        self.rectangle.add_point((0.25, 0.50), **self.v)
        self.assertEqual(self.rectangle.N, 4)
        # add point closing the rectangle
        print(">>> {}".format((0, 0)))
        self.rectangle.add_point((0, 0), **self.v)
        self.assertEqual(self.rectangle.N, 4)
        self.assertTrue(self.rectangle.is_closed)

    def test_close(self):
        """ # ROISelector.Rectangle.close """
        print(">>> {}".format(()))
        print(self.rectangle.__v__)
        self.assertFalse(self.rectangle.is_closed)
        r = self.rectangle.close(**self.v)
        self.assertTrue(r.is_closed)

    def test_contains(self):
        """ # ROISelector.Rectangle.contains """
        # multi-point input
        print(">>> {} {}".format([0, 2, 3, 0.2], [0, 3, 5, 0.6]))
        self.rectangle.contains([0, 2, 3, 0.2], [0, 3, 5, 0.6], **self.v)
        # single-point input
        print(">>> {}".format((0.25, 0.75)))
        self.rectangle.contains(0.25, 0.75, **self.v)


class TestSquare(UnitTestPrototype):

    def setUp(self):
        # arguments and keywords
        self.anchor = (0, 0)
        self.square = ROISelector.Square(self.anchor, 1)
        # # verbosity
        self.v = {'verbose': 1}
        print("")
        print(self.separator)
        print(self.shortDescription())

    def tearDown(self):
        print("")

    def test_Square(self):
        """ # ROISelector.Square """
        # with scalar dv
        print(">>> {}, {}".format(self.anchor, 1))
        s = ROISelector.Square(self.anchor, 1, **self.v)
        self.assertIsInstance(s, ROISelector.Square)
        self.assertEqual(s, self.square)
        self.assertEqual(s.N, 4)
        self.assertEqual(s.area, 1)
        # with corner
        print(">>> {}, {}".format(self.anchor, {'corner': (1, 2)}))
        s = ROISelector.Square(self.anchor, corner=(1, 2), **self.v)
        self.assertIsInstance(s, ROISelector.Square)
        self.assertNotEqual(s, self.square)
        self.assertEqual(s.N, 4)
        self.assertEqual(s.area, 4)
        # no second input
        print(">>> {}".format(self.anchor))
        s = ROISelector.Square(self.anchor, **self.v)
        self.assertIsInstance(s, ROISelector.Square)
        self.assertNotEqual(s, self.square)
        self.assertEqual(s.N, 4)
        self.assertEqual(s.area, 0)

    def test_anchor(self):
        """ # ROISelector.Square.anchor """
        anchor = (1, 1)
        print(">>> {}".format(anchor))
        a = self.square.area
        dv = self.square.dv
        self.square.anchor = anchor
        print(self.square.__v__)
        self.assertEqual(self.square.dv, dv)
        self.assertEqual(self.square.anchor, tuple(c+da for c, da in zip(self.anchor, anchor)))
        self.assertEqual(self.square.area, a)

    def test_dv(self):
        """ # ROISelector.Square.dv """
        dv = 2
        print(">>> {}".format(dv))
        a = self.square.anchor
        area = self.square.area
        self.square.dv = dv
        print(self.square.__v__)
        self.assertEqual(self.square.dv, (dv, dv))
        self.assertEqual(self.square.anchor, a)
        self.assertEqual(self.square.area, area*sum((dv, dv)))

    def test_add_point(self):
        """ # ROISelector.Square.add_point """
        # add point not in square
        print(">>> {}".format((0.25, 0.50)))
        self.square.add_point((0.25, 0.50), **self.v)
        self.assertEqual(self.square.N, 4)
        # add point closing the square
        print(">>> {}".format((0, 0)))
        self.square.add_point((0, 0), **self.v)
        self.assertEqual(self.square.N, 4)
        self.assertTrue(self.square.is_closed)

    def test_close(self):
        """ # ROISelector.Square.close """
        print(">>> {}".format(()))
        print(self.square.__v__)
        print("")
        self.assertFalse(self.square.is_closed)
        r = self.square.close(**self.v)
        self.assertTrue(r.is_closed)

    def test_contains(self):
        """ # ROISelector.Square.contains """
        # multi-point input
        print(">>> {} {}".format([0, 2, 3, 0.2], [0, 3, 5, 0.6]))
        self.square.contains([0, 2, 3, 0.2], [0, 3, 5, 0.6], **self.v)
        # single-point input
        print(">>> {}".format((0.25, 0.75)))
        self.square.contains(0.25, 0.75, **self.v)


class TestCircle(UnitTestPrototype):

    def setUp(self):
        # arguments and keywords
        self.center = (0, 0)
        self.radius = 1
        self.circle = ROISelector.Circle(self.center, self.radius)
        # # verbosity
        self.v = {'verbose': 1}
        print("")
        print(self.separator)
        print(self.shortDescription())

    def tearDown(self):
        print("")

    def test_Circle(self):
        """ # ROISelector.Circle """
        # with no radius
        print(">>> {}".format(self.center))
        circle = ROISelector.Circle(self.center, radius=0, **self.v)
        self.assertIsInstance(circle, ROISelector.Circle)
        self.assertNotEqual(circle, self.circle)
        self.assertEqual(circle.diameter, 0)
        self.assertEqual(circle.area, 0)
        # with scalar radius
        print(">>> {}".format((self.center, self.radius)))
        circle = ROISelector.Circle(self.center, self.radius, **self.v)
        self.assertIsInstance(circle, ROISelector.Circle)
        self.assertEqual(circle, self.circle)
        self.assertEqual(circle.diameter, 2*self.radius)
        self.assertEqual(circle.area, np.pi)
        # with radial point
        print(">>> {}".format((self.center, (self.radius, 0))))
        circle = ROISelector.Circle(self.center, (self.radius, 0), **self.v)
        self.assertIsInstance(circle, ROISelector.Circle)
        self.assertEqual(circle, self.circle)
        self.assertEqual(circle.diameter, 2*self.radius)
        self.assertEqual(circle.area, np.pi)

    def test_contains(self):
        """ ROISelector.Circle.contains """
        # multi-point input
        print(">>> {} {}".format([0, 2, 3, 0.2, 1, 1], [0, 3, 5, 0.6, 1, 0]))
        self.circle.contains([0, 2, 3, 0.2, 1, 1], [0, 3, 5, 0.6, 1, 0], **self.v)
        # single-point input
        print(">>> {} {}".format(0, 1))
        self.circle.contains(0, 1, **self.v)


class TestAmorph(UnitTestPrototype):

    def setUp(self):
        # arguments and keywords
        self.points = [(0, 0), (1, 0), (0, 2)]
        self.amorph = ROISelector.Amorph(self.points)
        # # verbosity
        self.v = {'verbose': 1}
        print("")
        print(self.separator)
        print(self.shortDescription())

    def tearDown(self):
        print("")

    def test_Amorph(self):
        """ # ROISelector.Amorph """
        # with list
        print(">>> {}".format(self.points))
        amorph = ROISelector.Amorph(self.points, **self.v)
        self.assertIsInstance(amorph, ROISelector.Amorph)
        self.assertTrue(np.all(amorph == self.amorph))
        # with tuple
        print(">>> {}".format(tuple(self.points)))
        amorph = ROISelector.Amorph(*self.points, **self.v)
        self.assertIsInstance(amorph, ROISelector.Amorph)
        self.assertTrue(np.all(amorph == self.amorph))
        # with numpy array
        print(">>> {}".format(np.flip(np.indices((3, 3)).T, 2)))
        amorph = ROISelector.Amorph(np.flip(np.indices((3, 3)).T, 2), **self.v)
        self.assertIsInstance(amorph, ROISelector.Amorph)
        self.assertFalse(np.all(amorph == self.amorph))
        self.assertIsInstance(amorph[:2, :2, :], ROISelector.Amorph)
        # single-point input (separate)
        print(">>> {}".format((0, 0)))
        amorph = ROISelector.Amorph(0, 0, **self.v)
        self.assertIsInstance(amorph, ROISelector.Amorph)
        self.assertFalse(np.all(amorph == self.amorph))

    def test_add_point(self):
        """ # ROISelector.Amorph.add_point """
        # add point to grid points
        print(">>> {}".format((5, 5)))
        amorph = ROISelector.Amorph(np.flip(np.indices((3, 3)).T, 2))
        amorph = amorph.add_point((5, 5), **self.v)
        print(amorph.N)
        self.assertIsInstance(amorph, ROISelector.Amorph)
        # add point to flat points
        print(">>> {}".format((5, 5)))
        self.amorph.add_point((5, 5), **self.v)
        self.assertIsInstance(self.amorph, ROISelector.Amorph)

    def test_contains(self):
        """ # ROISelector.Amorph.contains """
        # in flat points
        points = np.array([(5, 5), (0, 0), (1, 0)]).T
        print(">>> {}".format(points))
        self.amorph = self.amorph.add_point((5, 0), **self.v)
        self.amorph.contains(*points, **self.v)
        # in grid points
        print(">>> {}".format(points))
        amorph = ROISelector.Amorph(np.flip(np.indices((4, 4)).T, -1), **self.v)
        print(np.array([(5, 5), (3, 2), (0, 0), (1, 0)]).T)
        amorph.contains(*np.array([(5, 5), (3, 2), (0, 0), (1, 0)]).T, **self.v)


if __name__ == "__main__":
    TestROISelector.main(verbosity=1)
    TestPolygon.main(verbosity=1)
    TestRectangle.main(verbosity=1)
    TestSquare.main(verbosity=1)
    TestCircle.main(verbosity=1)
    TestAmorph.main(verbosity=1)
