#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: phdenzel

What are you interested in? Is it squares, circles, polygons, or an amorph
collection of pixels...

Usage example:
    data = np.empty((128, 128))
    s = ROISelector(data)
    s.select['circle']((64, 64), 10)
    data[s()]

"""
###############################################################################
# Imports
###############################################################################
import copy
import numpy as np
from PIL import Image, ImageDraw

from gleam.utils import colors as glmc

__all__ = ['ROISelector']


###############################################################################
class ROISelector(object):
    """
    Selects pixels from fits files
    """
    params = ['shape', '_buffer']
    selection_modes = ['circle', 'rect', 'square', 'polygon', 'amorph', 'color']

    def __init__(self, data=None, shape=None, _buffer=None, verbose=False):
        """
        Initialize from a pure data array

        Args:
            data <np.ndarray> - the data on which the selection is based

        Kwargs:
            verbose <bool> - verbose mode; print command line statements

        Return:
           <ROISelector object> - standard initializer (also see other classmethod initializers)
        """
        if data is not None:
            if len(data.shape) > 3:
                raise IndexError("ROISelector requires data shape of (N, M, D), "
                                 + "NxM grid with D dimensional points!")
            self.data = np.asarray(data)
        elif shape is not None:
            self.data = np.zeros(shape)
        else:
            self.data = np.zeros((1, 1))
        # height, widht = a.shape, i.e. matrix coordinates i, j = y, x and not x, y!
        self.yidcs, self.xidcs = np.indices(self.shape[0:2])
        self._buffer = {k: [] for k in ROISelector.selection_modes}
        self._selection = self.selection_modes[0]
        self._focus = -1
        if _buffer is not None:
            self._buffer = _buffer
        if verbose:
            print(self.__v__)

    def __call__(self, key=None, index=-1):
        if key in self._masks:
            return self._masks[key][index]
        elif hasattr(self, '_focus') and hasattr(self, '_selection'):
            return self._masks[self._selection][self._focus]
        else:
            NotImplemented

    def __getitem__(self, key):
        if key == 0:
            return self()
        else:
            raise IndexError('Not a valid key "{}"!'.format(key))

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.shape == other.shape and self._buffer == other._buffer
        else:
            NotImplemented

    def __copy__(self):
        args = (self.data,)
        kwargs = {k: self.__getattribute__(k) for k in self.__class__.params}
        return self.__class__(*args, **kwargs)

    def __deepcopy__(self, memo):
        args = (copy.deepcopy(self.data, memo),)
        kwargs = {k: copy.deepcopy(self.__getattribute__(k), memo)
                  for k in self.__class__.params}
        return self.__class__(*args, **kwargs)

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

    @property
    def __json__(self):
        """
        Select attributes for json write (link to data is not considered)
        """
        jsn_dict = {}
        for k in self.__class__.params:
            if hasattr(self, k):
                jsn_dict[k] = self.__getattribute__(k)
        jsn_dict['__type__'] = self.__class__.__name__
        return jsn_dict

    @classmethod
    def from_gleamobj(cls, gleam, **kwargs):
        """
        Initialize from a gleam instance
        gleam.[skyf.SkyF, skypatch.SkyPatch, lensobject.LensObject, multilens.MultiLens]

        Args:
            gleam <gleam object> - contains data of a or more than one .fits files

        Kwargs:
            verbose <bool> - verbose mode; print command line statements

        Return:
           <ROISelector object> - initializer with gleam object
        """
        return cls(gleam.data, **kwargs)

    def __str__(self):
        return "{}{}".format(self.__class__.__name__, self.shape)

    def __repr__(self):
        return self.__str__()

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
        return ['shape', '_buffer']

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
    def buffer(self):
        """
        The most current ROI object from the buffer

        Args/Kwargs:
            None

        Return:
            buffer <ROISelector.[] object> - current ROI object
        """
        if self._buffer[self._selection]:
            return self._buffer[self._selection][self._focus]

    @buffer.setter
    def buffer(self, roi):
        """
        Setter for the most current ROI object

        Args:
            roi <ROISelector.[] object> - current ROI object for buffer
        """
        self._buffer[self._selection].append(roi)

    @property
    def mask(self):
        """
        The most current ROI object from the buffer

        Args/Kwargs:
            None

        Return:
            buffer <ROISelector.[] object> - current ROI object
        """
        if self._masks[self._selection]:
            return self._masks[self._selection][self._focus]

    @property
    def _masks(self):
        """
        The masks for ROI selection derived from buffer

        Args/Kwargs:
            None

        Return:
            masks <dict(list(np.ndarray(bool)))> - selections derived from buffer
        """
        masks = {k: [] for k in ROISelector.selection_modes}
        for k in self._buffer:
            for s in self._buffer[k]:
                masks[k].append(s.contains(self.xidcs, self.yidcs))
        return masks

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
        Select all pixels within the circle

        Args:
            center <float,float> - center coordinates of the selection circle
            radius <float> - radius of the selection circle

        Kwargs:
            verbose <bool> - verbose mode; print command line statements

        Return:
            selection <np.ndarray(bool)> - boolean array with same shape as data
        """
        circle = ROISelector.Circle(center, radius)
        self._selection = 'circle'
        self._focus = -1
        self._buffer[self._selection].append(circle)
        selection = circle.contains(self.xidcs, self.yidcs)
        if verbose:
            print(self.__v__)
        return selection

    def select_rect(self, anchor, dv=None, corner=None, verbose=False):
        """
        Select all pixels within the rectangle

        Args:
            anchor <float,float> - anchor coordinates of the selection rectangle
            dv <float,float> - diagonal vector of the selection rectangle

        Kwargs:
            verbose <bool> - verbose mode; print command line statements

        Return:
            selection <np.ndarray(bool)> - boolean array with same shape as data
        """
        rect = ROISelector.Rectangle(anchor, dv=dv, corner=corner)
        self._selection = 'rect'
        self._focus = -1
        self._buffer[self._selection].append(rect)
        rect = rect.close()
        selection = rect.contains(self.xidcs, self.yidcs)
        if verbose:
            print(self.__v__)
        return selection

    def select_square(self, anchor, dv=None, corner=None, verbose=False):
        """
        Select all pixels within the square

        Args:
            anchor <float,float> - anchor coordinates of the selection square
            dv <float> - diagonal vector of the selection rectangle

        Kwargs:
            verbose <bool> - verbose mode; print command line statements

        Return:
            selection <np.ndarray(bool)> - boolean array with same shape as data
        """
        square = ROISelector.Square(anchor, dv=dv, corner=corner)
        self._selection = 'square'
        self._focus = -1
        self._buffer[self._selection].append(square)
        selection = square.contains(self.xidcs, self.yidcs)
        if verbose:
            print(self.__v__)
        return selection

    def select_polygon(self, *polygon, **kwargs):
        """
        Select all pixels within the polygon

        Args:
            polygon <list(float,float)> - vertices of the selection polygon

        Kwargs:
            verbose <bool> - verbose mode; print command line statements

        Return:
            selection <np.ndarray(bool)> - boolean array with same shape as data
        """
        verbose = kwargs.pop('verbose', False)
        polygon = ROISelector.Polygon(*polygon, **kwargs)
        self._selection = 'polygon'
        self._focus = -1
        self._buffer[self._selection].append(polygon)
        polygon = polygon.close()
        selection = polygon.contains(self.xidcs, self.yidcs)
        if verbose:
            print(self.__v__)
        return selection

    def select_amorph(self, *points, **kwargs):
        """
        Select all pixels belonging to a set of points

        Args:
            points <list(float,float)> - point coordinates of the selection set

        Kwargs:
            verbose <bool> - verbose mode; print command line statements

        Return:
            selection <np.ndarray(bool)> - boolean array with same shape as data
        """
        verbose = kwargs.pop('verbose', False)
        amorph = ROISelector.Amorph(*points, **kwargs)
        self._selection = 'amorph'
        self._focus = -1
        self._buffer[self._selection].append(amorph)
        selection = amorph.contains(self.xidcs, self.yidcs)
        if verbose:
            print(self.__v__)
        return selection

    def select_by_color(self, **kwargs):
        """
        TODO
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
            'square': self.focus_square,
            'polygon': self.focus_polygon,
            'amorph': self.focus_amorph,
            'color': self.focus_color
        }

    def close_all(self):
        """
        Close all ROI objects in the buffer

        Args/Kwargs/Return:
            None
        """
        for k in self._buffer:
            for i, s in enumerate(self._buffer[k]):
                if 'close' in dir(s):
                    self._buffer[k][i] = s.close()

    def clear_buffer(self):
        """
        Clear the buffer

        Args/Kwargs/Return:
            None
        """
        self._buffer = {k: [] for k in ROISelector.selection_modes}
        self._selection = None
        self._focus = 0

    def draw_rois(self, img, current_only=False, verbose=False):
        """
        Draw all selection ROI objects in the buffer on the input image

        Args:
            img <PIL.Image object> - an image object

        Kwargs:
            current_only <bool> - only plot currently focussed selection
            verbose <bool> - verbose mode; print command line statements

        Return:
            img <PIL.Image object> - the image object with selections drawn
        """
        if any([self._buffer[k] for k in self.selection_modes]):
            for k in self._buffer:
                for s in self._buffer[k]:
                    img = s.draw2img(img)
        return img

    @staticmethod
    def r_integrate(data, mask=None, center=None, R=None):
        """
        Sum up all pixels from a data map at center radially outwards

        Args:
            data <np.ndarray> - the data to be integrated

        Kwargs:
            mask <np.ndarray(bool)> - boolean mask used for integration
            center <float,float> - center from which to sum up the pixels (in pixels)
            R <float> - the radius up to which to sum up the pixels (in pixels)

        Return:
            sum <float> - the sum of all pixels in the data map
        """
        data = np.asarray(data)
        if center is None:
            center = 0.5*data.shape[0], 0.5*data.shape[1]
        if R is None:
            R = 0.5*data.shape[0]
        if mask is None:
            roi = ROISelector(data)
            mask = roi.select['circle'](center, R)
        return np.sum(data[mask])

    @staticmethod
    def cumr_profile(data, center=None, radii=None, R=None):
        """
        Sum up all pixels from a data map at center radially outwards for multiple radii

        Args:
            data <np.ndarray> - the data to be integrated

        Kwargs:
            mask <np.ndarray(bool)> - boolean mask used for integration
            center <float,float> - center from which to sum up the pixels (in pixels)
            R <float> - the radius up to which to sum up the pixels (in pixels)
        """
        data = np.asarray(data)
        if center is None:
            center = 0.5*data.shape[0], 0.5*data.shape[1]
        if R is None:
            R = 0.5*data.shape[0]
        if radii is None:
            radii = np.linspace(0., R, 0.5*data.shape[0])
        if len(radii) == 0:
            radii = np.array([0])
        roi = ROISelector(data)
        profile = radii[:]
        roi.select['circle'](center, radii[0])
        for i, r in enumerate(radii):
            roi.buffer.radius = r
            profile[i] = np.sum(data[roi.mask])
        return profile

    def mpl_interface(self):
        """
        Plot the data and provide a GUI to select ROI objects manually
        """
        import matplotlib as mpl
        mpl.pyplot.plot(self.data)

###############################################################################
    class Polygon(object):
        """
        A data structure describing a polygon
        """
        params = ['points']

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
            points = kwargs.pop('points', points)
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

        @property
        def __json__(self):
            """
            Select attributes for json write
            """
            jsn_dict = {}
            for k in self.__class__.params:
                if hasattr(self, k):
                    jsn_dict[k] = self.__getattribute__(k)
            jsn_dict['__type__'] = 'ROISelector.' + self.__class__.__name__
            return jsn_dict

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
            if len(points) == 1 and len(points[0]) > 2:
                points = points[0]
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
            if len(self.points) < 2:
                return
            if self.is_ccw < 0:  # clockwise
                self.x = np.concatenate((np.array([self.x[0]]), self.x[:0:-1]))
                self.y = np.concatenate((np.array([self.y[0]]), self.y[:0:-1]))
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

            if verbose:
                msg = ''.join(['({}, {}) \t{}\n'.format(xp[i], yp[i], mindst[i])
                               for i in range(len(mindst))])
                print(msg)

            if scalar:
                mindst = float(mindst)
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
                contained <np.ndarray(bool)> - boolean array of points which are within

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
                ctemp = [contained]
            else:
                ctemp = contained
            if verbose:
                is_inside = ['is inside' if c else 'is outside' for c in ctemp]
                msg = ''.join(['({}, {}) {}\n'.format(xp[i], yp[i], is_inside[i])
                               for i in range(len(is_inside))])
                print(msg)
            return contained

        def draw2img(self, img, point_size=1, fill=None, outline=None, verbose=False):
            """
            Draw the circle on the input image

            Args:
                img <PIL.Image object> - an image object

            Kwargs:
                point_size <int/float> - point size, i.e. radius in pixels
                fill <str> - color to use for points
                outline <str> - color to use for lines and borders
                verbose <bool> - verbose mode; print command line statements

            Return:
                img <PIL.Image object> - the image object with the shape drawn
            """
            if fill is None:
                if self.__class__.__name__ == 'Polygon':
                    fill = glmc.green
                elif self.__class__.__name__ == 'Rectangle':
                    fill = glmc.golden
                elif self.__class__.__name__ == 'Square':
                    fill = glmc.blue
            if outline is None:
                outline = glmc.black
            draw = ImageDraw.Draw(img)
            draw.line([tuple(p) for p in self.points], fill=outline, width=1)
            pts = [(p[0]-point_size, p[1]-point_size, p[0]+point_size, p[1]+point_size)
                   for p in self.points]
            for p in pts:
                draw.ellipse(p, fill=fill, outline=outline)
            del draw
            return img

###############################################################################
    class Rectangle(Polygon):
        """
        A data structure describing a rectangle
        """
        params = ['anchor', 'dv']

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
            Diagonal spanning vector of the rectangle

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

        @property
        def dcorner(self):
            """
            Diagonal corner opposite of anchor of the rectangle

            Args/Kwargs:
                None

            Return:
                corner <float,float> - corner position opposite of the anchor
            """
            return self.points[2]

        @dcorner.setter
        def dcorner(self, corner):
            """
            Move the corner opposite to the anchor to the given position

            Args:
                corner <float,float> - new corner position

            Kwargs/Return:
                None
            """
            dv = [c-a for a, c in zip(self.anchor, corner)]
            A = self.anchor
            B = (A[0]+dv[0], A[1])
            C = corner
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
        params = ['anchor', 'dv']

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
        params = ['center', 'radius']

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
                dx = radius[0]-self.center[0]
                dy = radius[1]-self.center[1]
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

        @property
        def __json__(self):
            """
            Select attributes for json write
            """
            jsn_dict = {}
            for k in self.__class__.params:
                if hasattr(self, k):
                    jsn_dict[k] = self.__getattribute__(k)
            jsn_dict['__type__'] = 'ROISelector.' + self.__class__.__name__
            return jsn_dict

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

        def mv_r(self, position):
            """
            Move the radius from the center to the input position
            """
            dx = position[0]-self.center[0]
            dy = position[1]-self.center[1]
            self.radius = np.sqrt(dx*dx+dy*dy)

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
            xp = np.asfarray(xp)
            yp = np.asfarray(yp)
            if xp.shape is tuple():
                xp = np.array([xp])
                yp = np.array([yp])
                scalar = True
            else:
                scalar = False

            dx = xp - self.center[0]
            dy = yp - self.center[1]
            mindst = self.r2 - (dx**2 + dy**2)
            mindst[np.fabs(mindst) < small_dist] = 0
            if verbose:
                msg = ''.join(['({}, {}) \t{}\n'.format(xp[i], yp[i], mindst[i])
                               for i in range(len(mindst))])
                print(msg)
            if scalar:
                mindst = float(mindst)
            return mindst

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
                contained <np.ndarray(bool)> - boolean array of points which are within

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
                ctemp = [contained]
            else:
                ctemp = contained
            if verbose:
                is_inside = ['is inside' if c else 'is outside' for c in ctemp]
                msg = ''.join(['({}, {}) {}\n'.format(xp[i], yp[i], is_inside[i])
                               for i in range(len(is_inside))])
                print(msg)
            return contained

        @staticmethod
        def draw_radius(image, bounds, width=1, outline=None, antialias=1):
            """
            Improved ellipse drawing function, based on PIL.ImageDraw

            Args:
                image <PIL.Image object> - the image to draw on
                bounds <list(tuple(float,float))> - four points to define the bounding box

            Kwargs:
                width <float> - width of the circle/ellipse outline
                outline <str> - color to use for lines and borders
                antialias <int> - degree of antialiasing (1 for no antialiasing, needs more memory)

            Return:
                image <PIL.Image objet> - the image object with the circle drawn
            """
            if outline is None:
                outline = glmc.black
            # single channel mask
            mask = Image.new(size=[int(dim * antialias) for dim in image.size],
                             mode='L', color='black')
            draw = ImageDraw.Draw(mask)

            # draw outer shape in color and inner shape transparent (black)
            for offset, fill in [(width/-2.0, 'white'), (width/2.0, 'black')]:
                x0, y0 = [(value + offset) * antialias for value in bounds[:2]]
                x1, y1 = [(value - offset) * antialias for value in bounds[2:]]
                draw.ellipse((x0, y0, x1, y1), fill=fill)
            mask = mask.resize(image.size, Image.LANCZOS)
            image.paste(outline, mask=mask)
            del draw, mask
            return image

        def draw2img(self, img, point_size=1, arc_width=1, fill=None, outline=None, verbose=False):
            """
            Draw the circle on the input image

            Args:
                img <PIL.Image object> - an image object

            Kwargs:
                point_size <float> - point size, i.e. radius in pixels
                arc_width <float> - width of the circle borders
                fill <str> - color to use for points
                outline <str> - color to use for lines and borders
                verbose <bool> - verbose mode; print command line statements

            Return:
                img <PIL.Image object> - the image object with the circle drawn
            """
            if fill is None:
                fill = glmc.red
            if outline is None:
                outline = glmc.black
            x0, y0 = self.center[0]-self.radius, self.center[1]-self.radius
            x1, y1 = self.center[0]+self.radius, self.center[1]+self.radius
            cx0, cy0 = self.center[0]-point_size, self.center[1]-point_size
            cx1, cy1 = self.center[0]+point_size, self.center[1]+point_size
            img = self.draw_radius(img, (x0, y0, x1, y1), width=arc_width, outline=outline)
            draw = ImageDraw.Draw(img)
            draw.ellipse((cx0, cy0, cx1, cy1), fill=fill, outline=outline)
            del draw
            return img

###############################################################################
    class Amorph(np.ndarray):
        """
        A data structure describing a non-shaped assembly of points
        (in contrast to the other geometrical shape classes in ROISelector)
        """

        def __new__(cls, *arrlike, **kwargs):
            if len(arrlike) == 1:
                obj = np.asarray(*arrlike).view(cls)
            else:
                obj = np.asarray(arrlike).view(cls)
            # obj.info = kwargs.pop('info', None)
            return obj

        def __init__(self, *arrlike, **kwargs):
            """
            Initialize an amorph data structure (sub-classed from nupy.ndarray)

            Args:
                arrlike <tuple/list/np.ndarray> - collection of points covertible into a np.ndarray
                                                  (last dimension needs to be 2!)

            Kwargs:
                verbose <bool> - verbose mode; print command line statements

            Return:
                <ROISelector.Amorph object> - standard initializer
            """
            shape = self.shape
            if shape[-1] != 2:
                raise IndexError("Wrong input shape! Amorph requires axis[-1] = 2!")
            if kwargs.pop('verbose', False):
                print(self.__v__)

        def __array_finalize__(self, obj):
            if obj is None:
                return
            # self.info = getattr(obj, 'info', None)

        def __array_prepare__(self, out_arr, context=None):
            return super(ROISelector.Amorph, self).__array_prepare(self, out_arr, context)

        def __array_wrap__(self, out_arr, context=None):
            return super(ROISelector.Amorph, self).__array_wrap__(self, out_arr, context)

        def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
            args = []
            in_no = []
            for i, input_ in enumerate(inputs):
                if isinstance(input_, ROISelector.Amorph):
                    in_no.append(i)
                    args.append(input_.view(np.ndarray))
                else:
                    args.append(input_)

            outputs = kwargs.pop('out', None)
            out_no = []
            if outputs:
                out_args = []
                for j, output in enumerate(outputs):
                    if isinstance(output, ROISelector.Amorph):
                        out_no.append(j)
                        out_args.append(output.view(np.ndarray))
                    else:
                        out_args.append(output)
                kwargs['out'] = tuple(out_args)
            else:
                outputs = (None,) * ufunc.nout

            results = super(ROISelector.Amorph, self).__array_ufunc__(ufunc, method,
                                                                      *args, **kwargs)
            if results is NotImplemented:
                return NotImplemented

            if method == 'at':
                return

            if ufunc.nout == 1:
                results = (results,)

            results = tuple((np.asarray(result).view(ROISelector.Amorph)
                             if output is None else output)
                            for result, output in zip(results, outputs))

            return results[0] if len(results) == 1 else results

        def encode(self):
            """
            Using md5 to encode specific information
            """
            import hashlib
            s = ', '.join([str(p) for p in self]).encode('utf-8')
            code = hashlib.md5(s).hexdigest()
            return code

        @property
        def __v__(self):
            """
            Info string for test printing

            Args/Kwargs:
                None

            Return:
                <str> - test of ROISelector attributes
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
            return ['str', 'N', 'x', 'y']

        def __str__(self):
            if self.N > 8:
                return '{}({})#{}'.format(self.__class__.__name__, self.N, self.encode())
            else:
                return '{}({})'.format(
                    self.__class__.__name__, super(ROISelector.Amorph, self).tolist().__str__())

        def __repr__(self):
            return '{}({})#{}'.format(self.__class__.__name__, self.N, self.encode())

        @property
        def str(self):
            return str(self)

        @property
        def x(self):
            """
            The x coordinates of the array of points
            """
            if len(self.shape) > 1:
                return np.array(self.reshape(np.prod(self.shape[:-1]), 2)[:, 0])
            else:
                return self[0]

        @property
        def y(self):
            """
            The y coordinatse of the array of points
            """
            if len(self.shape) > 1:
                return np.array(self.reshape(np.prod(self.shape[:-1]), 2)[:, 1])
            else:
                return self[1]

        @property
        def N(self):
            """
            The number of points in point collection

            Args/Kwargs:
                None

            Return:
                N <int> - the number of points
            """
            shape = self.shape
            if len(shape) == 1:
                return 1
            if len(shape) > 2:
                return np.prod(shape[:-1])
            return shape[0]

        def add_point(self, p, verbose=False):
            """
            Add a point to the polygon

            Args:
                p <float,float> - the point's x and y coordinates

            Kwargs
                verbose <bool> - verbose mode; print command line statements

            Return:
                added <ROISelector.Amorph object> - copy of instance with point added
            """
            if len(p) != 2:
                raise IndexError("Point has wrong dimensions!")
            if len(self.shape) > 2:
                if verbose:
                    print(self.__v__)
                return self
            self = ROISelector.Amorph(np.append(self, [p], axis=0))
            if verbose:
                print(self.__v__)
            return self

        def contains(self, xp, yp, verbose=False):
            """
            Check whether given points are in or outside of the circle

            Args:
                xp <np.ndarray(float)> - x coordinate of the point(s)
                yp <np.ndarray(float)> - y coordinate of the point(s)

            Kwargs:
                small_dist <float> - threshold distance to determine limit
                verbose <bool> - verbose mode; print command line statements

            Return:
                contained <np.ndarray(bool)> - boolean array of points which are within

            Note:
                How it works:
                - Amorph instance holds a set of unordered points of shape (N, 2) or (N, M, 2)
                - input can be of shape (K,), (K,)
                - return: boolean array of shape (N,) or (N, M) (matching Amorph instance shape)
                  where True values indicate a match
            """
            xp = np.asarray(xp)
            yp = np.asarray(yp)
            if xp.shape == tuple():
                xp = np.array([xp])
                yp = np.array([yp])
                scalar = True
            else:
                scalar = False

            if len(xp.shape) == len(self.x.shape) and len(xp.shape) < 3:
                contained = np.isin(xp, self.x) * np.isin(yp, self.y)
            else:
                contained = np.full(xp.shape[0:2], False)
                contained[(self.y, self.x)] = True
            if scalar:
                return bool(contained)
            if verbose:
                print(contained)
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
            import TestROISelector, TestPolygon, TestRectangle, TestSquare, TestCircle, TestAmorph
        TestROISelector.main(verbosity=1)
        TestPolygon.main(verbosity=1)
        TestRectangle.main(verbosity=1)
        TestSquare.main(verbosity=1)
        TestCircle.main(verbosity=1)
        TestAmorph.main(verbosity=1)
