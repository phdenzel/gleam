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
        self.skyp = SkyPatch(self.test_ftss)
        # verbosity
        self.v = {'verbose': 1}
        print("")
        print(self.separator)
        print(self.shortDescription())

    def tearDown(self):
        print("")

    # def test_ROISelector(self):
    #     """ # ROISelector """
    #     print(">>> {}".format(None))
    #     select = ROISelector(None, **self.v)
    #     self.assertIsInstance(select, ROISelector)
    #     self.assertIsNone(select.data)
    #     print(">>> {}".format(self.skyf.data))
    #     select = ROISelector(self.skyf.data, **self.v)
    #     self.assertIsInstance(select, ROISelector)
    #     self.assertIsNotNone(select.data)
    #     print(">>> {}".format(self.skyp.data))
    #     select = ROISelector(self.skyp.data, **self.v)
    #     self.assertIsInstance(select, ROISelector)
    #     self.assertIsNotNone(select.data)

    # def test_from_gleamobj(self):
    #     """ # from_gleamobj """
    #     print(">>> {}".format(self.skyf))
    #     select = ROISelector.from_gleamobj(self.skyf, **self.v)
    #     self.assertIsInstance(select, ROISelector)
    #     self.assertIsNotNone(select.data)
    #     print(">>> {}".format(self.skyp))
    #     select = ROISelector.from_gleamobj(self.skyp, **self.v)
    #     self.assertIsInstance(select, ROISelector)
    #     self.assertIsNotNone(select.data)


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
        print(">>> {}".format(self.points))
        p = ROISelector.Polygon(*self.points, **self.v)
        self.assertIsInstance(p, ROISelector.Polygon)
        self.assertEqual(p, self.polygon)
        self.assertEqual(p.N, 8)
        self.assertEqual(p.area, 3.625)
        self.assertGreater(p.is_ccw, 0)
        self.assertFalse(p.is_closed)
        print(">>> {}".format((1, 1)))
        p = ROISelector.Polygon((1, 1), **self.v)
        self.assertIsInstance(p, ROISelector.Polygon)
        self.assertEqual(p.N, 1)
        self.assertEqual(p.area, 0)
        self.assertEqual(p.is_ccw, 0)
        self.assertFalse(p.is_closed)

    def test_add_point(self):
        """ # ROISelector.Polygon.add_point """
        print(">>> {}".format((0.25, 0.50)))
        self.polygon.add_point((0.25, 0.50), **self.v)
        self.assertEqual(self.polygon.N, 9)
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
        print(">>> {} {}".format([0, 3, 2, 7], [4, 3, 5, 2]))
        self.polygon.contains([0, 3, 2, 7], [4, 3, 5, 2], **self.v)
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
        print(">>> {}, {}".format(self.anchor, (1, 1)))
        r = ROISelector.Rectangle(self.anchor, (1, 1), **self.v)
        self.assertIsInstance(r, ROISelector.Rectangle)
        self.assertEqual(r, self.rectangle)
        self.assertEqual(r.N, 4)
        self.assertEqual(r.area, 1)
        print(">>> {}, {}".format(self.anchor, {'corner': (1, 1)}))
        r = ROISelector.Rectangle(self.anchor, corner=(1, 1), **self.v)
        self.assertIsInstance(r, ROISelector.Rectangle)
        self.assertEqual(r, self.rectangle)
        self.assertEqual(r.N, 4)
        self.assertEqual(r.area, 1)
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
        print(">>> {}".format((0.25, 0.50)))
        self.rectangle.add_point((0.25, 0.50), **self.v)
        self.assertEqual(self.rectangle.N, 4)
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
        print(">>> {} {}".format([0, 2, 3, 0.2], [0, 3, 5, 0.6]))
        self.rectangle.contains([0, 2, 3, 0.2], [0, 3, 5, 0.6], **self.v)
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
        print(">>> {}, {}".format(self.anchor, 1))
        s = ROISelector.Square(self.anchor, 1, **self.v)
        self.assertIsInstance(s, ROISelector.Square)
        self.assertEqual(s, self.square)
        self.assertEqual(s.N, 4)
        self.assertEqual(s.area, 1)
        print(">>> {}, {}".format(self.anchor, {'corner': (1, 2)}))
        s = ROISelector.Square(self.anchor, corner=(1, 2), **self.v)
        self.assertIsInstance(s, ROISelector.Square)
        self.assertNotEqual(s, self.square)
        self.assertEqual(s.N, 4)
        self.assertEqual(s.area, 4)
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
        print(">>> {}".format((0.25, 0.50)))
        self.square.add_point((0.25, 0.50), **self.v)
        self.assertEqual(self.square.N, 4)
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
        print(">>> {} {}".format([0, 2, 3, 0.2], [0, 3, 5, 0.6]))
        self.square.contains([0, 2, 3, 0.2], [0, 3, 5, 0.6], **self.v)
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
        print(">>> {}".format(self.center))
        circle = ROISelector.Circle(self.center, radius=0, **self.v)
        self.assertIsInstance(circle, ROISelector.Circle)
        self.assertNotEqual(circle, self.circle)
        self.assertEqual(circle.diameter, 0)
        self.assertEqual(circle.area, 0)
        print(">>> {}".format((self.center, self.radius)))
        circle = ROISelector.Circle(self.center, self.radius, **self.v)
        self.assertIsInstance(circle, ROISelector.Circle)
        self.assertEqual(circle, self.circle)
        self.assertEqual(circle.diameter, 2*self.radius)
        self.assertEqual(circle.area, np.pi)

    def test_contains(self):
        """ ROISelector.Circle.contains """
        print(">>> {}".format(()))


if __name__ == "__main__":
    TestROISelector.main(verbosity=1)
    TestPolygon.main(verbosity=1)
    TestRectangle.main(verbosity=1)
    TestSquare.main(verbosity=1)
    TestCircle.main(verbosity=1)
