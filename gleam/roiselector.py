#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: phdenzel

What are you interested in? Is it squares, circles, polygons, or an amorph
collection of pixels...
"""
###############################################################################
# Imports
###############################################################################
import copy
import numpy as np

__all__ = ['ROISelector']


###############################################################################
class ROISelector(object):
    """
    Selects pixels from fits files
    """

    selection_modes = ['circle', 'rect', 'polygon', 'amorph']

    def __init__(self, data, verbose=False):
        """
        Initialize from a pure data array

        Args:
            data <np.ndarray> - the data on which the selection is based

        Kwargs:
            verbose <bool> - verbose mode; print command line statements

        Return:
           <ROISelector object> - standard initializer (also see other classmethod initializers)
        """
        self.data = np.asarray(data) if data is not None else data
        self._buffer = {k: [] for k in ROISelector.selection_modes}
        if verbose:
            print(self.__v__)

    @property
    def __v__(self):
        """
        Info string for test printing

        Args/Kwargs:
            None

        Return:
            <str> - test of ROISelector attributes
        """
        return "\n".join([t.ljust(20)+"\t{}".format(self.__getattribute__(t)) for t in self.tests])

    @property
    def tests(self):
        """
        A list of attributes being tested when calling __v__

        Args/Kwargs:
            None

        Return:
            tests <list(str)> - a list of test variable strings
        """
        return ['data', 'shape', '_buffer']

    @classmethod
    def from_gleamobj(cls, gleam, **kwargs):
        """
        Initialize from a gleam instance
        (gleam.[skyf.SkyF, skypatch.SkyPatch, lensobject.LensObject, multilens.MultiLens])

        Args:
            gleam <gleam object> - contains data of a or more than one .fits files

        Kwargs:
            verbose <bool> - verbose mode; print command line statements

        Return:
           <ROISelector object> - initializer with gleam object
        """
        return cls(gleam.data, **kwargs)

    @property
    def shape(self):
        """
        The data shape of the selector

        Args/Kwargs:
            None

        Return:
            shape
        """
        if self.data is not None:
            return self.data.shape

    @property
    def select(self):
        """
        A collection of selection methods

        Args/Kwargs:
            None

        Return:
            selectables <dict(func)> - collection of functions
        """
        return {
            'circle': self.select_circle,
            'rect': self.select_rect,
            'square': self.select_square,
            'polygon': self.select_polygon,
            'amorph': self.select_amorph,
            'color': self.select_by_color
        }

    def select_circle(self, center, radius, verbose=False):
        """
        Select all pixels within a circle

        Args:
            center <float,float> - center coordinates of the selection circle
            radius <float> - radius of the selection circle

        Kwargs:
            verbose <bool> - verbose mode; print command line statements

        Return:
            selection <np.ndarray(bool)> - boolean array
        """
        pass

    def select_rect(self, anchor, dv, verbose=False):
        """
        Select all pixels within a rectangle

        Args:
            anchor <float,float> - anchor coordinates of the selection rectangle
            dv <float,float> - diagonal vector of the selection rectangle

        Kwargs:
            verbose <bool> - verbose mode; print command line statements

        Return:
            selection <np.ndarray(bool)> - boolean array
        """
        pass

    def select_square(self, anchor, dv, verbose=False):
        """
        """
        pass

    def select_polygon(self, *polygon, **kwargs):
        """
        """
        verbose = kwargs.pop('verbose', False)
        if verbose:
            pass
        pass

    def select_amorph(self, *points, **kwargs):
        """
        """
        verbose = kwargs.pop('verbose', False)
        if verbose:
            pass
        pass

    def select_by_color(self, **kwargs):
        """
        """
        verbose = kwargs.pop('verbose', False)
        if verbose:
            pass
        pass

    @property
    def focus(self):
        """
        A collection of focus methods

        Args/Kwargs:
            None

        Return:
            focusable <dict(func)> - collection of functions
        """
        return {
            'circle': self.focus_circle,
            'rect': self.focus_rect,
            'square': self.focus_rect,
            'polygon': self.focus_polygon,
            'amorph': self.focus_amorph,
            'color': self.focus_amorph
        }

    @staticmethod
    def inside_polygon(point, polygon, verbose=False):
        """
        Test whether point is inside polygon

        Args:
            point <float,float> - point to be tested
            polygon <list(float,float)> - set of points connecting to a polygon

        Kwargs:
            verbose <bool> - verbose mode; print command line statements
        """
        n_edges = len(polygon)
        inside = False
        x, y = point
        p1x, p1y = polygon[0]
        for i in range(n_edges+1):
            p2x, p2y = polygon[i % n_edges]
            if y > min(p1y, p2y) and y <= max(p1y, p2y) and x <= max(p1x, p2x):
                if p1y != p2y:
                    xinters = (y-p1y)*(p2x-p1x)/(p2y-p1y)+p1x
                if p1x == p2x or x <= xinters:
                    inside = not inside
            p1x, p1y = p2x, p2y
        return inside

###############################################################################
    class Polygon(object):
        """
        A data structure describing a polygon
        """

        def __init__(self, *points, **kwargs):
            """
            Initialize a polygon data structure

            Args:
                points <float,float> - arbitrary number of point tuples/lists with x and y coords

            Kwargs:
                verbose <bool> - verbose mode; print command line statements

            Return:
                <ROISelector.Polygon object> - standard initializer
            """
            self.points = points
            if len(self.x) != len(self.y):  # by construction this should never happen
                raise IndexError("Shape of x and y coordinates do not match!")
            self._2ccw()
            verbose = kwargs.pop('verbose', False)
            if verbose:
                print(self.__v__)

        def __eq__(self, other):
            if isinstance(other, self.__class__):
                if self.N == other.N:
                    return np.all(self.x == other.x) and np.all(self.y == other.y)
            else:
                NotImplemented

        def __copy__(self):
            args = self.points
            return self.__class__(*args)

        def __deepcopy__(self, memo):
            args = copy.deepcopy(self.points, memo)
            return self.__class__(*args)

        def copy(self, verbose=False):
            cpy = copy.copy(self)
            if verbose:
                print(cpy.__v__)
            return cpy

        def deepcopy(self, verbose=False):
            cpy = copy.deepcopy(self)
            if verbose:
                print(cpy.__v__)
            return cpy

        def encode(self):
            """
            Using md5 to encode specific information
            """
            import hashlib
            s = ', '.join([str(p) for p in self.points]).encode('utf-8')
            code = hashlib.md5(s).hexdigest()
            return code

        def __hash__(self):
            """
            Using encode to create hash
            """
            return hash(self.encode())

        def __str__(self):
            return "{}({})#{}".format(self.__class__.__name__, self.N, self.encode())

        @property
        def __v__(self):
            """
            Info string for test printing

            Args/Kwargs:
                None

            Return:
                <str> - test of ROISelector.Polygon attributes
            """
            return "\n".join([t.ljust(20)+"\t{}".format(self.__getattribute__(t))
                              for t in self.tests])

        @property
        def tests(self):
            """
            A list of attributes being tested when calling __v__

            Args/Kwargs:
                None

            Return:
                tests <list(str)> - a list of test variable strings
            """
            return ['N', 'N_unique', 'N_edges', 'points', 'x', 'y', 'area', 'is_closed', 'is_ccw']

        @property
        def points(self):
            """
            The points of the polygon as an array of shape (N, 2)

            Args/Kwargs:
                None

            Return:
                points <np.ndarray> - point array of shape (N, 2)
            """
            return np.stack((self.x, self.y), axis=-1)

        @points.setter
        def points(self, points):
            """
            Setter for points of the polygon from a list of point tuples

            Args:
                points <list(float,float)> - new set of points

            Kwargs/Return:
                None
            """
            self.x = np.array([p[0] for p in points])
            self.y = np.array([p[1] for p in points])

        @property
        def N(self):
            """
            Count the number of vertices of the polygon

            Args/Kwargs:
                None

            Return:
                N <int> - number of vertices defining the polygon
            """
            N = len(self.x)
            if self.is_closed:
                return N-1
            else:
                return N

        @property
        def N_unique(self):
            """
            Count the number of unique vertices of the polygon

            Args/Kwargs:
                None

            Return:
                N <int> - number of vertices defining the polygon
            """
            return len(np.unique(self.points, axis=0))

        @property
        def N_edges(self):
            """
            Count the number of edges of the polygon

            Args/Kwargs:
                None

            Return:
                N <int> - number of edges connecting the vertices of the polygon
            """
            if self.is_closed:
                return self.N
            else:
                return self.N-1

        @property
        def is_closed(self):
            """
            Check whether polygon is closed or not

            Args/Kwargs:
                None

            Return:
                is_closed <bool> - True if closed, False otherwise
            """
            if len(self.x) > 2 and len(np.unique(self.points, axis=0)) > 2:
                return self.x[0] == self.x[-1] and self.y[0] == self.y[-1]
            return False

        @staticmethod
        def determinant(x, y, verbose=False):
            """
            Calculate the determinant formula for a set of points (x, y)

            Args:
                x - x coordinates of the points
                y - y coordinates of the points

            Kwargs:
                verbose <bool> - verbose mode; print command line statements

            Return:
                det <float> - the value of the determinant
            """
            x = np.asfarray(x)
            y = np.asfarray(y)
            x_p = np.concatenate(([x[-1]], x[:-1]))
            y_p = np.concatenate(([y[-1]], y[:-1]))
            det = np.sum(y * x_p - x * y_p, axis=0)
            if verbose:
                print(det)
            return det

        @property
        def area(self):
            """
            Compute the area of the polygon

            Args/Kwargs:
                None

            Return:
                area <float> - the absolute area of the polygon
            """
            return 0.5*abs(self.is_ccw)

        @property
        def is_ccw(self):
            """
            Check if polygon points are ordered counter-clockwise

            Args/Kwargs:
                None

            Return:
                is_ccw <-, 0, +> - positive if polygon points are in counter-clockwise order
                                   negative if polygon points are in clockwise order
                                   zero if polygon points are co-linear or coincident

            Note:
                - it actually computes twice the area of the polygon with the determinante formula
            """
            return self.determinant(self.x, self.y)

        def _2ccw(self, verbose=False):
            """
            Reorder the polygon points from clockwise to counter-clockwise

            Args:
                None

            Kwargs:
                verbose <bool> - verbose mode; print command line statements

            Return:
                None
            """
            if self.is_ccw < 0:
                self.x = self.x[::-1]
                self.y = self.y[::-1]
            if verbose:
                print(self.__v__)

        def add_point(self, p, verbose=False):
            """
            Add a point to the polygon

            Args:
                p <float,float> - the point's x and y coordinates

            Kwargs
                verbose <bool> - verbose mode; print command line statements

            Return:
                None
            """
            if len(p) != 2:
                raise IndexError("Point has wrong dimensions!")
            if self.is_closed:
                self.x = np.concatenate((self.x[:-1], [p[0]], self.x[-1:]))
                self.y = np.concatenate((self.y[:-1], [p[1]], self.y[-1:]))
            else:
                self.x = np.concatenate((self.x, [p[0]]))
                self.y = np.concatenate((self.y, [p[1]]))
            if verbose:
                print(self.__v__)

        def close(self, verbose=False):
            """
            Close the polygon

            Args:
                None

            Kwargs:
                verbose <bool> - verbose mode; print command line statements

            Return:
                closed <ROISelector.Polygon object> - a copy of the input polygon but closed
            """
            closed = self.copy()
            if closed.is_closed:
                return closed
            closed.add_point(self.points[0])
            if verbose:
                print(closed.__v__)
            return closed

        def mindst(self, xp, yp, small_dist=1e-12, verbose=False):
            """
            Point-in-Polygon algorithm following S.W. Sloan (1985)

            Args:
                xp <np.ndarray(float)> - x coordinate of the point(s)
                yp <np.ndarray(float)> - y coordinate of the point(s)

            Kwargs:
                small_dist <float> - threshold distance to determine limit
                verbose <bool> - verbose mode; print command line statementsx

            Return:
                mindst <np.ndarray(float)> - distance from (xp, yp) to nearest side

            Note:
                if mindst < 0: point (xp, yp) is outside the polygon
                          = 0: point (xp, yp) is on a side of the polygon
                          > 0: point (xp, yp) is inside the polygon
            """
            xp = np.asfarray(xp)
            yp = np.asfarray(yp)
            # handle scalar input
            scalar = False
            if xp.shape is tuple() or yp.shape is tuple():
                xp = np.array([xp], dtype=float)
                yp = np.array([yp], dtype=float)
                scalar = True
            if xp.shape != yp.shape:
                raise IndexError("Input x and y have different shapes!")

            mindst = np.ones_like(xp, dtype=float) * np.inf
            # if snear = True:  distance to nearest side < nearest vertex
            #          = False: distance to nearest vertex < nearest side
            snear = np.ma.masked_all(xp.shape, dtype=bool)
            j = np.ma.masked_all(xp.shape, dtype=int)
            for i in range(self.N_edges):  # loop over each edge
                d = np.ones_like(xp, dtype=float) * np.inf
                x21 = self.x[i+1] - self.x[i]
                y21 = self.y[i+1] - self.y[i]
                x1p = self.x[i] - xp
                y1p = self.y[i] - yp
                # line through edge with t=[0,1]
                t = -(x1p*x21 + y1p*y21)/(x21**2 + y21**2)
                tlt0 = t < 0
                tle1 = (0 <= t) & (t <= 1)
                # normal distance for t<0 and 0<t<1
                d[tle1] = ((x1p[tle1]+t[tle1]*x21)**2 + (y1p[tle1]+t[tle1]*y21)**2)
                d[tlt0] = x1p[tlt0]**2 + y1p[tlt0]**2
                # store distances
                mask = d < mindst
                mindst[mask] = d[mask]
                j[mask] = i
                snear[mask & tlt0] = False  # point closest to x[i], y[i]
                snear[mask & tle1] = True   # point closest to edge
            if np.ma.count(snear) != snear.size:
                raise IndexError('Error computing distances')
            mindst **= 0.5
            # if snear, check if nearest vertex concave
            jo = j.copy()
            jo[j == 0] -= 1
            area = self.determinant([self.x[j+1], self.x[j], self.x[jo-1]],
                                    [self.y[j+1], self.y[j], self.y[jo-1]])
            mindst[~snear] = np.copysign(mindst, area)[~snear]
            # if not snear, check if (xp, yp) to the left (i.e. inside) or the right (i.e. outside)
            area = self.determinant([self.x[j], self.x[j+1], xp], [self.y[j], self.y[j+1], yp])
            mindst[snear] = np.copysign(mindst, area)[snear]
            mindst[np.fabs(mindst) < small_dist] = 0
            if scalar:
                mindst = float(mindst)

            if verbose:
                reslt = mindst
                if scalar:
                    reslt = np.array([mindst])
                msg = ''.join(['({}, {}) \t{}\n'.format(xp[i], yp[i], reslt[i])
                               for i in range(len(reslt))])
                print(msg)
            return mindst

        def contains(self, xp, yp, small_dist=1e-12, verbose=False):
            """
            Point-in-Polygon algorithm following S.W. Sloan (1985)

            Args:
                xp <np.ndarray(float)> - x coordinate of the point(s)
                yp <np.ndarray(float)> - y coordinate of the point(s)

            Kwargs:
                small_dist <float> - threshold distance to determine limit
                verbose <bool> - verbose mode; print command line statements

            Return:
                mindst <np.ndarray(float)> - distance from (xp, yp) to nearest side

            Note:
                if mindst < 0: point (xp, yp) is outside the polygon
                          = 0: point (xp, yp) is on a side of the polygon
                          > 0: point (xp, yp) is inside the polygon
            """
            xp = np.asfarray(xp)
            yp = np.asfarray(yp)

            contained = self.mindst(xp, yp, small_dist=small_dist) >= 0
            if xp.shape is tuple():
                xp = np.array([xp])
                yp = np.array([yp])
                contained = [contained]
            if verbose:
                is_inside = ['is inside' if c else 'is outside' for c in contained]
                msg = ''.join(['({}, {}) {}\n'.format(xp[i], yp[i], is_inside[i])
                               for i in range(len(is_inside))])
                print(msg)
            return contained

###############################################################################
    class Rectangle(Polygon):
        """
        A data structure describing a rectangle
        """

        def __init__(self, anchor, dv=None, corner=None, **kwargs):
            """
            Initialize a rectangle data structure

            Args:
                anchor <float,float> - anchor vertex of the rectangle (origin of diagonal vector)

            Kwargs:
                dv <float,float> - diagonal vector spanning the rectangle
                                   (takes priority over corner keyword)
                corner <float,float> - the corner across anchor (alternative keyword for dv)
                points <list(float,float)> - list of the vertices
                verbose <bool> - verbose mode; print command line statements

            Return:
                <ROISelector.Rectangle object> - standard initializer
            """
            if dv is None:
                if corner:
                    dv = [c-a for a, c in zip(anchor, corner)]
                else:
                    dv = (0, 0)
            points = [tuple(anchor),
                      (anchor[0]+dv[0], anchor[1]),
                      (anchor[0]+dv[0], anchor[1]+dv[1]),
                      (anchor[0], anchor[1]+dv[1])]
            super(ROISelector.Rectangle, self).__init__(*points, **kwargs)

        def __copy__(self):
            args = self.anchor, self.dv
            return self.__class__(*args)

        def __deepcopy__(self, memo):
            args = copy.deepcopy(self.anchor, memo), copy.deepcopy(self.dv, memo)
            return self.__class__(*args)

        def __str__(self):
            return "{}#{}".format(self.__class__.__name__, self.encode())

        @property
        def tests(self):
            """
            A list of attributes being tested when calling __v__

            Args/Kwargs:
                None

            Return:
                tests <list(str)> - a list of test variable strings
            """
            return super(ROISelector.Rectangle, self).tests + ['anchor', 'dv']

        @property
        def anchor(self):
            """
            Anchor of the rectangle

            Args/Kwargs:
                None

            Return:
                anchor <float,float> - anchor, i.e. corner point of rectangle
            """
            return self.x[0], self.y[0]

        @anchor.setter
        def anchor(self, anchor):
            """
            Move the anchor to a new positions while preserving dv

            Args:
                anchor <float,float> - new anchor position

            Kwargs/Return:
                None
            """
            old = self.anchor
            mv = (anchor[0]-old[0], anchor[1] - old[1])
            self.x = self.x + mv[0]
            self.y = self.y + mv[1]

        @property
        def dv(self):
            """
            Diagonal spanning vecotr of the rectangle

            Args/Kwargs:
                None

            Return:
                dv <float,float> - vector spanning the diagonal of the rectangle
            """
            A = self.anchor
            C = (self.x[2], self.y[2])
            return C[0]-A[0], C[1]-A[0]

        @dv.setter
        def dv(self, dv):
            """
            Move diagonal spanning vector while leaving the anchor at the same position

            Args:
                dv <float,float> - new spanning vector

            Kwargs/Return:
                None
            """
            if isinstance(dv, (int, float)) or len(dv) == 1:
                dv = (dv, dv)
            A = self.anchor
            B = (A[0]+dv[0], A[1])
            C = (B[0], A[1]+dv[1])
            D = (A[0], C[1])
            self.points = [A, B, C, D]
            self._2ccw()

        def add_point(self, p, verbose=False):
            """
            Only add point if rectangle doesn't have all it's vertices

            Args:
                p <float,float> - coordinates of the point to be added

            Kwargs:
                verbose <bool> - verbose mode; print command line statements

            Return:
                None
            """
            if not self.is_closed \
               and p[0] == self.x[0] and p[1] == self.y[0] \
               and self.N_unique == 4:
                super(ROISelector.Rectangle, self).add_point(p, verbose=verbose)
            elif verbose:
                print(self.__v__)

###############################################################################
    class Square(Rectangle):
        """
        A data structure describing a square
        """

        def __init__(self, anchor, dv=None, corner=None, **kwargs):
            """
            Initialize a square data structure

            Args:
                anchor <float,float> - anchor vertex of the square (origin of diagonal vector)

            Kwargs:
                dv <float> - magnitude of the diagonal vector spanning the of the square
                             (takes priority over corner keyword)
                corner <float,float> - the corner across anchor (alternative keyword for dv)
                points <list(float,float)> - list of the vertices
                verbose <bool> - verbose mode; print command line statements

            Return:
                <ROISelector.Square object> - standard initializer
            """
            if dv is None:
                if corner:
                    dv = max([c-a for a, c in zip(anchor, corner)], key=abs)
                else:
                    dv = 0
            if hasattr(dv, '__len__'):
                dv = dv[0]
            kwargs['dv'] = [dv]*2
            super(ROISelector.Square, self).__init__(anchor, **kwargs)

###############################################################################
    class Circle(object):
        """
        A data structure describing a circle
        """

        def __init__(self, center, radius=0, verbose=False):
            """
            Initialize a circle data structure

            Args:
                center <float,float> - center coordinates of the circle
                radius <float> - radius of the circle

            Kwargs:
                verbose <bool> - verbose mode; print command line statements

            Return:
                <ROISelector.Circle object> - standard initializer
            """
            self.center = np.asarray(center)
            if isinstance(radius, (int, float)):
                self.radius = abs(radius)
            elif hasattr(radius, '__len__') and len(radius) == 2:
                dx = radius[0]-center[0]
                dy = radius[1]-center[1]
                self.radius = np.sqrt(dx*dx+dy*dy)
            if verbose:
                print(self.__v__)

        def __eq__(self, other):
            if isinstance(other, self.__class__):
                return self.radius == other.radius \
                    and self.center[0] == other.center[0] \
                    and self.center[1] == other.center[1]
            else:
                NotImplemented

        def encode(self):
            """
            Using md5 to encode specific information
            """
            import hashlib
            s = ', '.join([str(self.radius), str(self.center)]).encode('utf-8')
            code = hashlib.md5(s).hexdigest()
            return code

        def __hash__(self):
            """
            Using encode to create hash
            """
            return hash(self.encode())

        def __str__(self):
            return "{}({})@{}#{}".format(
                self.__class__.__name__, self.radius, self.center, self.encode())

        @property
        def __v__(self):
            """
            Info string for test printing

            Args/Kwargs:
                None

            Return:
                <str> - test of ROISelector.Circle attributes
            """
            return "\n".join([t.ljust(20)+"\t{}".format(self.__getattribute__(t))
                              for t in self.tests])

        @property
        def tests(self):
            """
            A list of attributes being tested when calling __v__

            Args/Kwargs:
                None

            Return:
                tests <list(str)> - a list of test variable strings
            """
            return ['center', 'radius', 'diameter', 'circumference', 'area']

        @property
        def r2(self):
            """
            The squared radius (useful for a lot of mathematical operations)

            Args/Kwargs:
                None

            Return:
                r2 <float> - squared radius
            """
            return self.radius*self.radius

        @property
        def diameter(self):
            """
            Diameter of the circle

            Args/Kwargs:
                None

            Return:
                diameter <float> - the diameter of the circle is twice its radius
            """
            return 2.*self.radius

        @diameter.setter
        def diameter(self, d):
            """
            Setter of the diameter of the circle

            Args:
                d <float> - the diameter of the circle

            Kwargs/Return:
                None
            """
            self.radius = .5*d

        @property
        def circumference(self):
            """
            The circumference of the circle

            Args/Kwargs:
                None

            Return:
                circumference <float> - the circumference is 2*pi multiplied by its radius
            """
            return 2*np.pi*self.radius

        @circumference.setter
        def circumference(self, c):
            """
            Setter of the circumference of the circle

            Args:
                c <float> - the circumference of the circle

            Kwargs/Return:
                None
            """
            self.radius = .5*c/np.pi

        @property
        def area(self):
            """
            Area of the circle

            Args/Kwargs:
                None

            Return:
                area <float> - the area of the circle is pi multiplied by its radius squared
            """
            return np.pi*self.radius*self.radius

        @area.setter
        def area(self, a):
            """
            Setter of the area of the circle

            Args:
                a <float> - the area of the circle

            Kwargs/Return:
                None
            """
            self.radius = np.sqrt(a)/np.pi

        def mindst(self, xp, yp, small_dist=1e-12, verbose=False):
            """
            Minimal distances from the circle's radius to a given set of points

            Args:
                xp <np.ndarray(float)> - x coordinate of the point(s)
                yp <np.ndarray(float)> - y coordinate of the point(s)

            Kwargs:
                small_dist <float> - threshold distance to determine limit
                verbose <bool> - verbose mode; print command line statements

            Return:
                mindst <np.ndarray(float)> - distance from (xp, yp) to the center + radius
            """
            pass

        def contains(self, xp, yp, small_dist=1e-12, verbose=False):
            """
            Check whether given points are in or outside of the circle

            Args:
                xp <np.ndarray(float)> - x coordinate of the point(s)
                yp <np.ndarray(float)> - y coordinate of the point(s)

            Kwargs:
                 small_dist <float> - threshold distance to determine limit
                verbose <bool> - verbose mode; print command line statements

            Return:
                mindst <np.ndarray(float)> - distance from (xp, yp) to nearest side

            Note:
                if mindst < 0: point (xp, yp) is outside the polygon
                          = 0: point (xp, yp) is on a side of the polygon
                          > 0: point (xp, yp) is inside the polygon
            """
            xp = np.asfarray(xp)
            yp = np.asfarray(yp)
            if xp.shape is tuple():
                xp = np.array([xp])
                yp = np.array([yp])

            contained = None #TODO
            if verbose:
                is_inside = ['is inside' if c else 'is outside' for c in contained]
                msg = ''.join(['({}, {}) {}\n'.format(xp[i], yp[i], is_inside[i])
                               for i in range(len(is_inside))])
                print(msg)
            return contained


def parse_arguments():
    """
    Parse command line arguments
    """
    from argparse import ArgumentParser, RawTextHelpFormatter
    parser = ArgumentParser(description=__doc__, formatter_class=RawTextHelpFormatter)
    # main args
    # TODO
    # mode args
    parser.add_argument("-v", "--verbose", dest="verbose", action="store_true",
                        help="Run program in verbose mode",
                        default=False)
    parser.add_argument("-t", "--test", "--test-mode", dest="test_mode", action="store_true",
                        help="Run program in testing mode",
                        default=False)
    args = parser.parse_args()
    return parser, args


if __name__ == "__main__":
    import sys
    parser, args = parse_arguments()
    no_input = len(sys.argv) <= 1
    if no_input:
        parser.print_help()
    elif args.test_mode:
        sys.argv = sys.argv[:1]
        from gleam.test.test_roiselector \
            import TestROISelector, TestPolygon, TestRectangle, TestSquare, TestCircle
        TestROISelector.main(verbosity=1)
        TestPolygon.main(verbosity=1)
        TestRectangle.main(verbosity=1)
        TestSquare.main(verbosity=1)
        TestCircle.main(verbosity=1)
