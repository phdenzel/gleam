#!/usr/bin/env python
"""
@author: phdenzel

Find your way in the sky with SkyCoords

TODO:
   - center calculations have an issue coming from referece calculation (precision ?)
"""
###############################################################################
# Imports
###############################################################################
import sys
import re
import copy
import math


__all__ = ['SkyCoords']


###############################################################################
class SkyCoords(object):
    """
    Framework for sky coordinates, particularly from .fits files
    """
    params = ['ra', 'dec', 'px2arcsec_scale', 'reference_pixel', 'reference_value']

    def __init__(self, ra, dec, px2arcsec_scale=[None, None],
                 reference_pixel=[None, None], reference_value=[None, None], verbose=False):
        """
        Initialize instance with RA,Dec in arcsec

        Args:
            ra  <float> - Right Ascension (RA) coordinate in degrees
            dec <float> - Declination (Dec) coordinate in degrees

        Kwargs:
            px2arcsec_scale <float,float> - conversion factor from pixels to arcsecs
            reference_pixel <int,int> - reference pixel in .fits file (default: [0, 0])
            reference_value <float,float> - reference coordinates in the .fits file
                                            (in degrees; default: [0., 0.])
            verbose <bool> - verbose mode; print command line statements

        Return:
            <SkyCoords object> - standard initializer (see also classmethods)

        Note:
            - RA,Dec origin at vernal equinox
            - origin attribute in this class refers to pixel origin and depends on the
              reference pixel/value
            - RA increases on the celestial plane eastwards (to the left) and decreases westwards
        """
        self.ra = ra    # RA in [0, 360]
        self.dec = dec  # Dec in [-90, 90]
        if isinstance(reference_pixel, list) and None in reference_pixel:
            self.reference_pixel = [0, 0]
        else:
            self.reference_pixel = reference_pixel
        if isinstance(reference_value, list) and None in reference_value:
            self.reference_value = [360., 0.]
        else:
            self.reference_value = reference_value
        if isinstance(px2arcsec_scale, list) and None in px2arcsec_scale:
            self.px2arcsec_scale = [1., 1.]
        else:
            self.px2arcsec_scale = px2arcsec_scale
        if verbose:
            print(self.__v__)

    def __add__(self, other):
        # Note: adding to RA means moving to the left on the celestial plane (WCS)
        if isinstance(other, (list, tuple, SkyCoords)) and len(other) >= 2:
            # Note: for a shift in rectangular coordinates use self.shift
            return SkyCoords(self.ra+other[0], self.dec+other[1], **self.coordkw)
        else:
            NotImplemented

    def __radd__(self, other):
        # Note: adding to RA means moving to the left on the celestial plane (WCS)
        # Note: for a shift in rectangular coordinates use self.shift
        return self.__add__(other)

    def __sub__(self, other):
        # Note: subtracting from RA means moving to the right on the celestial plane (WCS)
        if isinstance(other, (list, tuple)) and len(other) >= 2:
            # Note: for a shift in rectangular coordinates use self.shift
            return SkyCoords(self.ra-other[0], self.dec-other[1], **self.coordkw)
        else:
            NotImplemented

    def __rsub__(self, other):
        # Note: subtracting from RA means moving to the right on the celestial plane (WCS)
        if isinstance(other, (list, tuple)) and len(other) >= 2:
            # Note: for a shift in rectangular coordinates use self.shift
            return SkyCoords(other[0]-self.ra, other[1]-self.dec, **self.coordkw)
        else:
            NotImplemented

    def __mul__(self, other):
        if isinstance(other, (int, float)):
            # Note: multiplying RA means moving to the left on the celestial plane (WCS)
            return SkyCoords(self.ra*other, self.dec*other, **self.coordkw)
        else:
            NotImplemented

    def __rmul__(self, other):
        # Note: multiplying RA means moving to the left on the celestial plane (WCS)
        if isinstance(other, (int, float)):
            return SkyCoords(self.ra*other, self.dec*other, **self.coordkw)
        else:
            NotImplemented

    def __lt__(self, other):
        if isinstance(other, SkyCoords):
            return self.ra < other.ra and self.dec < other.dec
        elif isinstance(other, list) and len(other) == 2:
            return self.ra < other[0] and self.dec < other[1]
        else:
            NotImplemented

    def __le__(self, other):
        if isinstance(other, SkyCoords):
            return self.ra <= other.ra and self.dec <= other.dec
        elif isinstance(other, list) and len(other) == 2:
            return self.ra <= other[0] and self.dec <= other[1]
        else:
            NotImplemented

    def __eq__(self, other):
        if isinstance(other, SkyCoords):
            return self.ra == other.ra and self.dec == other.dec
        elif isinstance(other, list) and len(other) == 2:
            return self.ra == other[0] and self.dec == other[1]
        else:
            NotImplemented

    def __ne__(self, other):
        if isinstance(other, SkyCoords):
            return self.ra != other.ra and self.dec != other.dec
        elif isinstance(other, list) and len(other) == 2:
            return self.ra != other[0] and self.dec != other[1]
        else:
            NotImplemented

    def __gt__(self, other):
        if isinstance(other, SkyCoords):
            return self.ra > other.ra and self.dec > other.dec
        elif isinstance(other, list) and len(other) == 2:
            return self.ra > other[0] and self.dec > other[1]
        else:
            NotImplemented

    def __ge__(self, other):
        if isinstance(other, SkyCoords):
            return self.ra >= other.ra and self.dec >= other.dec
        elif isinstance(other, (list, tuple)) and len(other) == 2:
            return self.ra >= other[0] and self.dec >= other[1]
        else:
            NotImplemented

    def __len__(self):
        return 2

    def __getitem__(self, i):
        if i == 0:
            return self.ra
        elif i == 1:
            return self.dec
        else:
            raise IndexError

    def __setitem__(self, i, ra_or_dec):
        if not isinstance(ra_or_dec, (int, float)):
            raise TypeError("RA,Dec coordinates need to be integer or float")
        if i == 0:
            self.ra = ra_or_dec
        elif i == 1:
            self.dec = ra_or_dec
        else:
            raise IndexError

    def __contains__(self, key):
        return key in self.radec

    def __abs__(self):
        return abs(self.ra), abs(self.dec)

    def __copy__(self):
        kwargs = {k: self.__getattribute__(k) for k in self.__class__.params}
        return self.__class__(**kwargs)

    def __deepcopy__(self, memo):
        kwargs = {k: copy.deepcopy(self.__getattribute__(k), memo)
                  for k in self.__class__.params}
        return self.__class__(**kwargs)

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

        Args/Kwargs:
            None

        Return:
            jsn_dict <dict> - a first-layer serialized json dictionary
        """
        jsn_dict = {k: self.__getattribute__(k) for k in self.__class__.params}
        jsn_dict['__type__'] = self.__class__.__name__
        return jsn_dict

    @classmethod
    def empty(cls, *args, **kwargs):
        """
        Initialize empty instance

        Args:
            None (any argument will be ignored)

        Kwargs:
            px2arcsec_scale <float,float> - conversion factor from pixels to arcsecs
            reference_pixel <int,int> - reference pixel from .fits file
            reference_value <float,float> - reference coordinates in the .fits file (in degrees)
            verbose <bool> - verbose mode; print command line statements

        Return:
            <SkyCoords object> - empty SkyCoords instance
        """
        return cls(None, None, **kwargs)
    
    def __str__(self):
        if self.ra is not None and self.dec is not None:
            return "<{:.4f}, {:.4f}>".format(self.ra, self.dec)
        else:
            return "<{}, {}>".format(self.ra, self.dec)

    def __repr__(self):
        return self.__str__()

    @property
    def tests(self):
        """
        A list of attributes being tested when calling __v__

        Args/Kwargs:
            None

        Return:
            tests <list(str)> - a list of test variable strings
        """
        return ['J2000C', 'radec', 'degrees', 'arcsecs', 'xy', 'origin',
                'reference_pixel', 'reference_value', 'px2arcsec_scale',
                'arcsec2px_scale', 'px2deg_scale', 'deg2px_scale']

    @property
    def __v__(self):
        """
        Info string for test printing

        Args/Kwargs:
            None

        Return:
            <str> - test of SkyCoords attributes
        """
        return "\n".join([t.ljust(20)+"\t{}".format(self.__getattribute__(t)) for t in self.tests])

    @classmethod
    def from_J2000(cls, J2000, dec_as_hms=False, **kwargs):
        """
        Initialize instance with J2000 coordinates

        Args:
            J2000 <str> - format: 'J*[0-9][+\-]*[0-9]'; Julian equinox

        Kwargs:
            dec_as_hms <bool> - read the Dec in <hours,minutes,seconds>
            px2arcsec_scale <float,float> - conversion factor from pixels to arcsecs
            reference_pixel <int,int> - reference pixel from .fits file
            reference_value <float,float> - reference coordinates in the .fits file (in degrees)
            verbose <bool> - verbose mode; print command line statements

        Return:
            <SkyCoords object> - initialize with J2000 coordinates
        """
        return cls(*cls.J2000._2deg(J2000, dec_as_hms=dec_as_hms), **kwargs)

    @classmethod
    def from_arcsec(cls, ra_arcs, dec_arcs, **kwargs):
        """
        Initialize instance with RA,Dec in arcsec

        Args:
            ra_arcs  <float> - Right Ascension (RA) coordinate in arcsec
            dec_arcs <float> - Declination (Dec) coordinate in arcsec

        Kwargs:
            px2arcsec_scale <float,float> - conversion factor from pixels to arcsecs
            reference_pixel <int,int> - reference pixel from .fits file
            reference_value <float,float> - reference coordinates in the .fits file (in degrees)
            verbose <bool> - verbose mode; print command line statements

        Return:
            <SkyCoords object> - initialize with coordinates in arcsecs
        """
        return cls(*cls.arcsec2deg(ra_arcs, dec_arcs), **kwargs)

    @classmethod
    def from_degrees(cls, ra_deg, dec_deg, **kwargs):
        """
        Initialize instance with RA,Dec in degrees

        Args:
            ra_deg  <float> - Right Ascension (RA) coordinate in degrees
            dec_deg <float> - Declination (Dec) coordinate in degrees

        Kwargs:
            px2arcsec_scale <float,float> - conversion factor from pixels to arcsecs
            reference_pixel <int,int> - reference pixel from .fits file
            reference_value <float,float> - reference coordinates in the .fits file (in degrees)
            verbose <bool> - verbose mode; print command line statements

        Return:
            <SkyCoords object> - initialize with coordinates in degrees
        """
        return cls(ra_deg, dec_deg, **kwargs)

    @classmethod
    def from_pixels(cls, x, y, **kwargs):
        """
        Initialize instance with x/y in pixels

        Args:
            x <int/float> - Right Ascension (RA) coordinate in pixels
            y <int/float> - Declination (Dec) coordinate in pixels

        Kwargs:
            px2arcsec_scale <float,float> - conversion factor from pixels to arcsecs
            reference_pixel <int,int> - reference pixel from .fits file
            reference_value <float,float> - reference coordinates in the .fits file (in degrees)
            verbose <bool> - verbose mode; print command line statements

        Return:
            <SkyCoords object> - initialize with coordinates in pixels
        """
        v = kwargs.pop('verbose', False)
        return cls(*cls.pixels2deg([x, y], **kwargs), verbose=v, **kwargs)

    def distance(self, other, from_pixels=False, verbose=False):
        """
        Distance between two angular positions (in degrees)

        Args:
            other <SkyCoords object/list/tuple(float,float)> - position of the other object

        Kwargs:
            from_pixels <bool> - calculate distance in pixels with normal pythagorean formula
            verbose <bool> - verbose mode; print command line statements

        Return:
            d <float> - angular separation in degrees
        """
        assert isinstance(other, (SkyCoords, list, tuple))
        dra, ddec = self.ra-other[0], self.dec-other[1]
        cosdel2 = math.cos(*SkyCoords.deg2rad(self.dec))**2
        if from_pixels:
            if isinstance(other, SkyCoords):
                dra, ddec = self.x-other.x, self.y-other.y
            else:
                dra, ddec = self.x-other[0], self.y-other[1]
            cosdel2 = 1
        # estimate separation with pseudo-Pythagorean formula
        d = math.sqrt(dra*dra*cosdel2+ddec*ddec)
        if verbose:
            print(d)
        return d

    def angle(self, other, verbose=False):
        """
        Angle bewtween two angluar positions on the skyplane (in degrees)

        Args:
            other <SkyCoords object/list/tuple(float,float)> - position of the other object

        Kwargs:
            verbose <bool> - verbose mode; print command line statements

        Return:
            a <float> - planar angle between the two objects in degrees
        """
        assert isinstance(other, (SkyCoords, list, tuple))
        a = SkyCoords.rad2deg(math.atan2(self.dec-other[1], self.ra-other[0]))[0]
        if verbose:
            print(a)
        return a

    def shift(self, shift, right_increase=True, verbose=False):
        """
        Shift the coordinates on a rectangular plane

        Args:
            shift <tuple/list> - the amount of shift in degrees

        Kwargs:
            right_increase <bool> - positive shift points to the right

        Return:
            shifted <SkyCoords object> - a new SkyCoords object shifted by other

        Note:
            - the sign of the RA in shift is flipped, because adding to RA means moving to the left
              in the celestial plane (WCS), which is counter-intuitive.
        """
        shift = list(shift)
        if right_increase:
            shift[0] = -shift[0]
        if isinstance(shift, (list, tuple)) and len(shift) >= 2:
            newra = self.ra+shift[0]/math.cos(*SkyCoords.deg2rad(self.dec))
            newdec = self.dec+shift[1]
            shifted = SkyCoords(newra, newdec, **self.coordkw)
            if verbose:
                print(shifted)
            return shifted
        else:
            NotImplemented

    def get_shift_to(self, other, unit='degree', verbose=False):
        """
        Recover rectangular shift of the coordinates to another coordinate point

        Args:
            other <SkyCoords object> - another coordinate point

        Kwargs:
            unit <str> - unit of the shift (arcsec, degree, pixel)

        Return:
            shift <list> - the rectangular shift (in degrees by default)

        Note:
            - the sign of the RA in shift is flipped, because adding to RA means moving to the left
              in the celestial plane (WCS), which is counter-intuitive.
        """
        dra = (self.ra-other.ra)*math.cos(*SkyCoords.deg2rad(other.dec))
        ddec = (self.dec-other.dec)
        if unit in ['pixel', 'pixels']:
            shift = SkyCoords.deg2pixels(dra, ddec, px2arcsec_scale=self.px2arcsec_scale,
                                         reference_pixel=[0, 0], reference_value=[0., 0.])
        elif unit in ['arcsec', 'arcsecs']:
            shift = SkyCoords.deg2arcsec(-dra, ddec)
        elif unit in ['degree', 'degrees']:
            shift = [-dra, ddec]
        else:
            shift = [-dra, ddec]
        if verbose:
            print(shift)
        return shift

    @property
    def coordkw(self):
        """
        Dictionary getter of all keyword arguments of the instance

        Args/Kwargs:
            None

        Return:
            coordkw <dict> - dictionary of all keyword arguments
        """
        return {'px2arcsec_scale': self.px2arcsec_scale,
                'reference_pixel': self.reference_pixel, 'reference_value': self.reference_value}

    @property
    def arcsec2px_scale(self):
        """
        Conversion factor from arcsec to pixels

        Args/Kwargs:
            None

        Return
            arcsec2px <float,float> - conversion factor for RA,Dec in pixels/arcsec
        """
        if None not in self.px2arcsec_scale:
            return [1./s for s in self.px2arcsec_scale]
        else:
            return [None, None]

    @arcsec2px_scale.setter
    def arcsec2px_scale(self, arcsec2px):
        """
        Setter for conversion factor from arcsec to pixels

        Args:
            arcsec2px <float,float> - conversion factor for RA,Dec in pixels/arcsecs

        Kwargs/Return:
            None
        """
        self.px2arcsec_scale = [1./s for s in arcsec2px]

    @property
    def px2deg_scale(self):
        """
        Conversion factor from pixels to degrees

        Args/Kwargs:
            None

        Return:
            px2deg <float,float> - conversion factor for RA,Dec in degrees/pixels
        """
        if None not in self.px2arcsec_scale:
            return [s/3600. for s in self.px2arcsec_scale]
        else:
            return [None, None]

    @px2deg_scale.setter
    def px2deg_scale(self, px2deg):
        """
        Conversion factor setter from pixels to degrees

        Args:
            px2deg <float> - conversion factor for RA,Dec in degrees/pixels

        Kwargs/Return:
            None
        """
        self.px2arcsec_scale = [s*3600. for s in px2deg]

    @property
    def deg2px_scale(self):
        """
        Conversion factor from degrees to pixels

        Args/Kwargs:
            None

        Return:
            deg2px <float,float> - conversion factor for RA,Dec in pixels/degrees
        """
        if None not in self.px2arcsec_scale:
            return [3600./s for s in self.px2arcsec_scale]
        else:
            return [None, None]

    @deg2px_scale.setter
    def deg2px_scale(self, deg2px):
        """
        Conversion factor setter from degrees to pixels

        Args:
            deg2px <float> - conversion factor for RA,Dec in pixels/degrees

        Kwargs/Return:
            None
        """
        self.px2arcsec_scale = [3600./s for s in deg2px]

    @property
    def J2000C(self):
        """
        J2000 coordinate getter

        Args/Kwargs:
            None

        Return:
            <J2000 object> - J2000 coordinate w/ c1(RA), c2(Dec) in hms, dms
        """
        return SkyCoords.J2000(self.ra, self.dec)

    @property
    def ra(self):
        """
        RA coordinate (in degrees)

        Args/Kwargs:
            None

        Return:
            ra <float> - Right Ascension coordinate in degrees
        """
        return self._ra

    @ra.setter
    def ra(self, ra):
        """
        RA coordinate setter (in degrees)

        Args:
            ra <float> - Right Ascension coordinate in degrees

        Kwargs/Return:
            None
        """
        if ra is None:
            self._ra = ra
        else:
            self._ra = ra % 360

    @property
    def dec(self):
        """
        Dec coordinate (in degrees)

        Args/Kwargs:
            None

        Return:
            dec <float> - Declination coordinate in degrees
        """
        return self._dec

    @dec.setter
    def dec(self, dec):
        """
        Dec coordinate setter (in degrees)

        Args:
            dec <float> - Declination coordinate in degrees

        Kwargs/Return:
            None
        """
        if dec is None:
            self._dec = dec
        else:
            self._dec = dec % 90*math.copysign(1, dec)

    @property
    def radec(self):
        """
        Coordinate point getter (in degrees)

        Args/Kwargs:
            None

        Return:
            radec <float,float> - (RA, Dec) point in degrees
        """
        return [self.ra, self.dec]

    @radec.setter
    def radec(self, radec):
        """
        Coordinate point setter (in degrees)

        Args:
            radec <float,float> - (RA, Dec) point in degrees

        Kwargs/Return:
            None
        """
        self.ra = radec[0]
        self.dec = radec[1]

    @property
    def xdeg(self):
        """
        Right Ascension coordinate getter in degrees

        Args/Kwargs:
            None

        Return:
            xdeg <float> - Right Ascension coordinate in degrees
        """
        return self.ra

    @xdeg.setter
    def xdeg(self, xdeg):
        """
        Right Ascension coordinate setter in degrees

        Args:
            xdeg <float> - Right Ascension coordinate in degrees

        Kwargs/Return:
            None
        """
        self.ra = xdeg

    @property
    def ydeg(self):
        """
        Declination coordinate getter in degrees

        Args/Kwargs:
            None

        Return:
            ydeg <float> - Declination coordinate in degrees
        """
        return self.dec

    @ydeg.setter
    def ydeg(self, ydeg):
        """
        Declination coordinate setter in degrees

        Args:
            ydeg <float> - Declination coordinate in degrees

        Kwargs/Return:
            None
        """
        self.dec = ydeg

    @property
    def degrees(self):
        """
        Coordinate point getter (in degrees)

        Args/Kwargs:
            None

        Return:
            degrees <float,float> - (RA, Dec) point in degrees
        """
        return self.radec

    @degrees.setter
    def degrees(self, degrees):
        """
        Coordinate point setter (in degrees)

        Args:
            degrees <float,float> - (RA, Dec) point in degrees

        Kwargs/Return:
            None
        """
        self.ra = degrees[0]
        self.dec = degrees[1]

    @property
    def xarcsec(self):
        """
        Right Ascension coordinate getter (in arcsec)

        Args/Kwargs:
            None

        Return:
            xarcsec <float> - Right Ascension coordinate in arcsecs
        """
        if self.ra is not None:
            return self.ra*3600

    @xarcsec.setter
    def xarcsec(self, xarcsec):
        """
        Right Ascension coordinate setter (in arcsec)

        Args:
            xarcsec <float> - Right Ascension coordinate in arcsecs

        Kwargs/Return:
            None
        """
        self.ra = xarcsec/3600.

    @property
    def yarcsec(self):
        """
        Declination coordinate getter (in arcsec)

        Args/Kwargs:
            None

        Return:
            yarcsec <float> - Declination coordinate in arcsecs
        """
        if self.dec is not None:
            return self.dec*3600

    @yarcsec.setter
    def yarcsec(self, yarcsec):
        """
        Declination coordinate setter (in arcsec)

        Args:
            yarcsec <float> - Declination coordinate in arcsecs

        Kwargs/Return:
            None
        """
        self.dec = yarcsec/3600.

    @property
    def arcsecs(self):
        """
        Coordinate point getter (in arcsec)

        Args:
            degrees <float,float> - (RA, Dec) point in arcsecs

        Kwargs/Return:
            None
        """
        if None not in self.radec:
            return [rd*3600 for rd in self.radec]
        else:
            return [None, None]

    @arcsecs.setter
    def arcsecs(self, arcsecs):
        """
        Coordinate point setter (in arcsec)

        Args:
            degrees <float,float> - (RA, Dec) point in arcsecs

        Kwargs/Return:
            None
        """
        self.radec = SkyCoords.arcsec2deg(*arcsecs)

    @property
    def x(self):
        """
        Right Ascension pixel coordinate getter

        Args/Kwargs:
            None

        Return:
            x <int> - Right Ascension coordinate in pixels
        """
        if not any([c is None for c in [self.refx, self.refra, self.ra,
                                        self.refdec, self.px2deg_scale[0]]]):
            return self.refx+(self.refra-self.ra) * math.cos(*self.deg2rad(self.refdec)) \
                / self.px2deg_scale[0]

    @x.setter
    def x(self, x):
        """
        Right Ascension pixel coordinate setter

        Args:
            x <int> - Right Ascension coordinate in pixels

        Kwargs/Return:
            None
        """
        self.ra = self.refra-(x-self.refx)*self.px2deg_scale[0] \
            / math.cos(*self.deg2rad(self.refdec))

    @property
    def y(self):
        """
        Declination pixel coordinate getter

        Args/Kwargs:
            None

        Return:
            y <int> - Declination coordinate in pixels
        """
        if not any([c is None for c in [self.refy, self.refdec, self.dec, self.px2deg_scale[1]]]):
            return self.refy-(self.refdec-self.dec)/self.px2deg_scale[1]

    @y.setter
    def y(self, y):
        """
        Declination pixel coordinate setter

        Args:
            y <int> - Declination coordinate in pixels

        Kwargs/Return:
            None
        """
        self.dec = self.refdec+(y-self.refy)*self.px2deg_scale[1]

    @property
    def xy(self):
        """
        Coordinate point getter (in pixels)

        Args/Kwargs:
            None

        Return:
            xy <int,int> - (RA, Dec) point in pixels
        """
        return self.deg2pixels(
            *self.radec, integers=False, px2arcsec_scale=self.px2arcsec_scale,
            reference_pixel=self.reference_pixel, reference_value=self.reference_value)

    @xy.setter
    def xy(self, xy):
        """
        Coordinate point setter (in pixels)

        Args:
            xy <int,int> - (RA, Dec) point in pixels

        Kwargs/Return:
            None
        """
        self.ra = self.refra-(xy[0]-self.refx) \
            * self.px2deg_scale[0]/math.cos(self.deg2rad(self.refdec))
        self.dec = self.refdec+(xy[1]-self.refy)*self.px2deg_scale[1]

    @property
    def refra(self):
        """
        Reference point's RA coordinate in arcsec

        Args/Kwargs:
            None

        Return:
            refra <float> - reference point's RA coordinate in degrees
        """
        return self.reference_value[0]

    @refra.setter
    def refra(self, refra):
        """
        Reference point's RA coordinate setter in arcsec

        Args:
            refra <float> - reference point's RA coordinate in degrees

        Kwargs/Return:
            None
        """
        self.reference_value[0] = refra

    @property
    def refdec(self):
        """
        Reference point's Dec coordinate in arcsec

        Args/Kwargs:
            None

        Return:
            refdec <float> - reference point's Dec coordinate in degrees
        """
        return self.reference_value[1]

    @refdec.setter
    def refdec(self, refdec):
        """
        Reference point's Dec coordinate setter in arcsec

        Args:
            refdec <float> - reference point's Dec coordinate in degrees

        Kwargs/Return:
            None
        """
        self.reference_pixel[1] = refdec

    @property
    def refx(self):
        """
        Reference pixel's x-coordinate

        Args/Kwargs:
            None

        Return:
            refx <int> - reference point's x-coordinate in pixels
        """
        return self.reference_pixel[0]

    @refx.setter
    def refx(self, x):
        """
        Reference pixel's x-coordinate setter

        Args:
            refx <int> - reference point's x-coordinate in pixels

        Kwargs/Return:
            None
        """
        self.reference_pixel[0] = x

    @property
    def refy(self):
        """
        Reference pixel's y-coordinate

        Args/Kwargs:
            None

        Return:
            refy <int> - reference point's y-coordinate in pixels
        """
        return self.reference_pixel[1]

    @refy.setter
    def refy(self, y):
        """
        Reference pixel's y-coordinate setter

        Args:
            refy <int> - reference point's y-coordinate in pixels

        Kwargs/Return:
            None
        """
        self.reference_pixel[1] = y

    @property
    def origin(self):
        """
        Origin of the pixel coordinates (in degrees)

        Args/Kwargs:
            None

        Return:
            origin <float,float> - RA,Dec coordinates of the origin relative to reference value
        """
        if not any([c is None for c in [self.refra, self.refdec, self.refx, self.refy,
                                        self.px2deg_scale[0], self.px2deg_scale[1]]]):
            return [self.refra+self.refx*self.px2deg_scale[0]/math.cos(*self.deg2rad(self.refdec)),
                    self.refdec-self.refy*self.px2deg_scale[1]]
        else:
            return [None, None]

    @origin.setter
    def origin(self, origin):
        """
        Setter for the origin of the map coordinates (in degrees) relative to reference value

        Args:
            origin <float,float> - RA,Dec coordinates of the origin relative to reference value

        Kwargs/Return:
            None
        """
        self.refdec = origin[1]+self.refy*self.px2deg_scale[1]
        self.refra = origin[0]-self.refx*self.px2deg_scale[0]/math.cos(*self.deg2rad(self.refdec))

    @staticmethod
    def hms2deg(h, m, s, add_roundup=True, verbose=False):
        """
        Convert hms (hours, minutes, seconds) coordinates to degrees

        Args:
            h <int> - number of hours
            m <int> - number of minutes
            s <float> - number of seconds

        Kwargs:
            add_roundup <bool> - return result with an added 0.0001
            verbose <bool> - verbose mode; print command line statements

        Return:
            degree <float> - converted degrees
        """
        degree = h*15.+m*0.25+s/240.
        if add_roundup:
            degree = h*15.+m*0.25+s/240.+0.0001
        if verbose:
            print(degree)
        return degree

    @staticmethod
    def dms2deg(d, m, s, verbose=False):
        """
        Convert dms (degrees, minutes, seconds) coordinates to degrees

        Args:
            d <int> - number of degrees
            m <int> - number of minutes
            s <float> - number of seconds

        Kwargs:
            verbose <bool> - verbose mode; print command line statements

        Return:
            degree <float> - converted degrees
        """
        degree = d+math.copysign(1, d)*(m/60.+s/3600.)
        if verbose:
            print(degree)
        return degree

    @staticmethod
    def deg2hms(degs, round_=False, verbose=False):
        """
        Convert degrees to hms (hours, minutes, seconds) coordinates

        Args:
            degs <float> - degrees

        Kwargs:
            round_ <bool> - round to integer seconds
            verbose <bool> - verbose mode; print command line statements

        Return:
            h <int>, m <int>, s <float> - number of hours, minutes, and seconds
        """
        h = int(degs/15.)
        m = int(abs(degs/15.-h)*60.)
        s = (abs(degs/15.-h)-m/60.)*3600.
        if round_:
            s = int(round(s))
        if verbose:
            print([h, m, s])
        return h, m, s

    @staticmethod
    def deg2dms(degs, round_=False, verbose=False):
        """
        Convert degrees to dms (degrees, minutes, seconds) coordinates

        Args:
            degs <float> - degrees

        Kwargs:
            round_ <bool> - round to integer seconds
            verbose <bool> - verbose mode; print command line statements

        Return:
            d <int>, m <int>, s <float> - number of degrees, minutes, and seconds
         """
        d = int(degs)
        m = int(abs((degs-d)*60.))
        s = (abs(degs-d)-m/60.)*3600.
        if round_:
            s = int(round(s))
        if verbose:
            print([d, m, s])
        return d, m, s

    @staticmethod
    def deg2J2000(ra_deg, dec_deg, verbose=False):
        """
        Convert RA,Dec in degrees to a J2000 string

        Args:
            ra <float>, dec <float> - RA,Dec coordinates in degrees

        Kwargs:
            verbose <bool> - verbose mode; print command line statements

        Return:
            J2000 <str> - 'J[02d:02d:02.1f][+\-][02d:02d:02d]'; Julian equinox
        """
        def sign_number(i):
            return ("+" if i > 0 else "-") + "{:02d}".format(abs(i))
        if ra_deg is None or dec_deg is None:
            J2000 = None
        else:
            hms = SkyCoords.deg2hms(ra_deg)
            dms = SkyCoords.deg2dms(dec_deg, round_=True)
            J2000 = "".join(("J", "{0:02d}{1:02d}{2:02.1f}".format(*hms),
                             sign_number(dms[0])+"{0:02d}{1:02d}".format(*dms[1:])))
        if verbose:
            print(J2000)
        return J2000

    @staticmethod
    def deg2arcsec(*degs, **kwargs):
        """
        Convert degrees to arcsecs

        Args:
            degs *<float> - degrees to be converted

        Kwargs:
            verbose <bool> - verbose mode; print command line statements

        Return:
            arcsecs *<float> - converted arcsecs
        """
        verbose = kwargs.pop('verbose', False)
        if None in degs:
            arcsecs = [None]*len(degs)
        else:
            arcsecs = [d*3600. for d in degs]
        if verbose:
            print(arcsecs)
        return arcsecs

    @staticmethod
    def deg2pixels(*degs, **kwargs):
        """
        Convert degrees to pixels with given pixel scale

        Args:
            degs *<float> - RA,Dec coordinates (in degrees) to be converted

        Kwargs:
            integers <bool> - return pixels as integers
            px2arcsec_scale <float,float> - pixel scale in arcsec/pixels
            reference_pixel <int,int> - reference pixel from .fits file
            reference_value <float,float> - reference coordinates in the .fits file (in degrees)
            verbose <bool> - verbose mode; print command line statements

        Return:
            pixels *<int> - converted pixels
        """
        verbose = kwargs.pop('verbose', False)
        integers = kwargs.pop('integers', False)
        px2arcsec_scale = kwargs.pop('px2arcsec_scale', [None, None])
        reference_pixel = kwargs.pop('reference_pixel', [None, None])
        reference_value = kwargs.pop('reference_value', [None, None])
        # take care of keywords
        if None in px2arcsec_scale:
            px2arcsec_scale = [1., 1.]
        else:
            px2arcsec_scale = px2arcsec_scale
        px2deg_scale = SkyCoords.arcsec2deg(*px2arcsec_scale)
        if None in reference_pixel:
            refx, refy = 0, 0
        else:
            refx, refy = reference_pixel
        if None in reference_value:
            refra, refdec = 360., 0.
        else:
            refra, refdec = reference_value
        # calculate pixel position
        if None in degs:
            pixels = [None]*len(degs)
        else:
            pixels = [refx+(refra-degs[0])*math.cos(*SkyCoords.deg2rad(refdec))/px2deg_scale[0],
                      refy-(refdec-degs[1])/px2deg_scale[1]]
        if integers:
            pixels = [int(round(p)) for p in pixels]
        if verbose:
            print(pixels)
        return pixels

    @staticmethod
    def arcsec2deg(*arcsecs, **kwargs):
        """
        Convert arcsecs to degrees

        Args:
            arcsecs *<float> - arcsecs to be converted

        Kwargs:
            verbose <bool> - verbose mode; print command line statements

        Return:
            degrees *<float> - converted degrees
        """
        verbose = kwargs.pop('verbose', False)
        if None in arcsecs:
            degrees = [None]*len(arcsecs)
        else:
            degrees = [a/3600. for a in arcsecs]
        if verbose:
            print(degrees)
        return degrees

    @staticmethod
    def arcsec2J2000(ra_arcsec, dec_arcsec, verbose=False):
        """
        Convert RA,Dec in arcsec to a J2000 string

        Args:
            ra_arcsec <float>, dec_arcsec <float> - RA,Dec coordinates in arcsec

        Kwargs:
            verbose <bool> - verbose mode; print command line statements

        Return:
            J2000 <str> - 'J[02d:02d:02.1f][+\-][02d:02d:02d]'; Julian equinox
        """
        ra_deg, dec_deg = SkyCoords.arcsec2deg(ra_arcsec, dec_arcsec)
        J2000 = SkyCoords.deg2J2000(ra_deg, dec_deg)
        if verbose:
            print(J2000)
        return J2000

    @staticmethod
    def arcsec2pixels(*arcsecs, **kwargs):
        """
        Convert arcsecs to pixels with given pixel scale

        Args:
            arcsecs *<float> - RA,Dec coordinates (in arcsecs) to be converted

        Kwargs:
            integers <bool> - return pixels as integers
            px2arcsec_scale <float,float> - pixel scale in arcsec/pixels
            reference_pixel <int,int> - reference pixel from .fits file
            reference_value <float,float> - reference coordinates in the .fits file (in degrees)
            verbose <bool> - verbose mode; print command line statements

        Return:
            pixels *<int> - converted pixels
        """
        verbose = kwargs.pop('verbose', False)
        degs = SkyCoords.arcsec2deg(*arcsecs)
        pixels = SkyCoords.deg2pixels(*degs, **kwargs)
        if verbose:
            print(pixels)
        return pixels

    @staticmethod
    def pixels2deg(pixels, px2arcsec_scale=[None, None],
                   reference_pixel=[None, None], reference_value=[None, None], verbose=False):
        """
        Convert pixel coordinates to degrees with given pixel scale

        Args:
            pixels *<int/float> - pixels to be converted

        Kwargs:
            px2arcsec_scale <float,float> - pixel scale in arcsec/pixels
            reference_pixel <int,int> - reference pixel in .fits file
            reference_value <float,float> - reference coordinates in the .fits file (in degrees)
            verbose <bool> - verbose mode; print command line statements

        Return:
            degs *<float> - converted degrees
        """
        # take care of keywords
        if reference_pixel == [None, None]:
            refx, refy = 0, 0
        else:
            refx, refy = reference_pixel
        if reference_value == [None, None]:
            refra, refdec = 360., 0.
        else:
            refra, refdec = reference_value
        if px2arcsec_scale == [None, None]:
            px2arcsec_scale = 1., 1.
        else:
            px2arcsec_scale = px2arcsec_scale
        # calculate position in degrees
        px2deg_scale = SkyCoords.arcsec2deg(*px2arcsec_scale)
        if None in pixels:
            degs = [None]*len(pixels)
        else:
            degs = [refra-(pixels[0]-refx)*px2deg_scale[0] / math.cos(*SkyCoords.deg2rad(refdec)),
                    refdec+(pixels[1]-refy)*px2deg_scale[1]]
        if verbose:
            print(degs)
        return degs

    @staticmethod
    def pixels2J2000(pixels, **kwargs):
        """
        Convert pixel coordinates to a J2000 string with given pixel scale

        Args:
            pixels *<int/float> - pixels to be converted

        Kwargs:
            px2arcsec_scale <float,float> - pixel scale in arcsec/pixels
            reference_pixel <int,int> - reference pixel in .fits file
            reference_value <float,float> - reference coordinates in the .fits file (in degrees)
            verbose <bool> - verbose mode; print command line statements

        Return:
            J2000 <str> - 'J[02d:02d:02.1f][+\-][02d:02d:02d]'; Julian equinox
        """
        verbose = kwargs.pop('verbose', False)
        ra, dec = SkyCoords.pixels2deg(pixels, **kwargs)
        J2000 = SkyCoords.deg2J2000(ra, dec)
        if verbose:
            print(J2000)
        return J2000

    @staticmethod
    def pixels2arcsec(pixels, **kwargs):
        """
        Convert pixel coordinates to arcsecs with given pixel scale

        Args:
            pixels *<int/float> - pixels to be converted

        Kwargs:
            px2arcsec_scale <float,float> - pixel scale in arcsec/pixels
            reference_pixel <int,int> - reference pixel in .fits file
            reference_value <float,float> - reference coordinates in the .fits file (in degrees)
            verbose <bool> - verbose mode; print command line statements

        Return:
            arcsecs *<float> - converted arcsecs
        """
        # if None in pixels:
        #     arcsec [None]*len(pixels)
        verbose = kwargs.pop('verbose', False)
        degs = SkyCoords.pixels2deg(pixels, **kwargs)
        arcsecs = [d*3600. if d else None for d in degs]
        if verbose:
            print(arcsecs)
        return arcsecs

    @staticmethod
    def deg2rad(*degs, **kwargs):
        """
        Convert degrees to radians

        Args:
            degs *<float> - degrees to be converted

        Kwargs:
            verbose <bool> - verbose mode; print command line statements

        Return:
            rads *<float> - converted radians
        """
        verbose = kwargs.pop('verbose', False)
        rads = [d*math.pi/180. for d in degs]
        if verbose:
            print(rads)
        return rads

    @staticmethod
    def rad2deg(*rads, **kwargs):
        """
        Convert radians to degrees

        Args:
            rads *<float> - radians to be converted

        Kwargs:
            verbose <bool> - verbose mode; print command line statements

        Return:
            degrees *<float> - converted degrees
        """
        verbose = kwargs.pop('verbose', False)
        rads = [r*180./math.pi for r in rads]
        if verbose:
            print(rads)
        return rads

    # J2000 SUBCLASS ##########################################################
    class J2000(object):
        """
        J2000 coordinate framework for easy handling in SkyCoords class
        """
        def __init__(self, ra, dec, verbose=False):
            """
            Initialize a J2000 instance

            Args:
                ra <float>, dec <float> - RA,Dec coordinates in degrees

            Kwargs:
                verbose <bool> - verbose mode; print command line statements

            Return:
                J2000 <SkyCoords.J2000 object> - an instance of SkyCoords.J2000
            """
            self.J2000 = SkyCoords.deg2J2000(ra, dec)
            if verbose:
                print(self.__v__)

        def __eq__(self, other):
            if isinstance(other, SkyCoords.J2000):
                return self.J2000 == other.J2000
            else:
                NotImplemented

        def __str__(self):
            return str(self.J2000)

        @property
        def __v__(self):
            """
            Info string for test printing

            Args/Kwargs:
                None

            Return:
                <str> - test of SkyCoords attributes
            """
            tests = ['c1', 'c2', 'c1_str', 'c2_str']
            return "\n".join([t.ljust(20)+"\t{}".format(self.__getattribute__(t)) for t in tests])

        @property
        def c1(self):
            """
            RA coordinate in hms

            Args/Kwargs:
                None

            Return:
                c1 <float,float,float> - RA in number of hours, minutes, seconds
            """
            if self.J2000:
                return SkyCoords.J2000._2hmsdms(self.J2000)[0]
            else:
                return [None, None, None]

        @property
        def c2(self):
            """
            Dec coordinate in dms

            Args/Kwargs:
                None

            Return:
                c2 <float,float,float> - Dec in number of degrees, minutes, seconds
            """
            if self.J2000:
                return SkyCoords.J2000._2hmsdms(self.J2000)[1]
            else:
                return [None, None, None]

        @property
        def c1_str(self):
            """
            RA coordinate in hms

            Args/Kwargs:
                None

            Return:
                c1 <str> - RA string in number of hours, minutes, seconds
            """
            if self.J2000:
                return "{0:02.0f}:{1:02.0f}:{2:02.1f}".format(*SkyCoords.J2000._2hmsdms(self.J2000)[0])
            else:
                return None

        @property
        def c2_str(self):
            """
            Dec coordinate in dms

            Args/Kwargs:
                None

            Return:
                c2 <str> - Dec string in number of degrees, minutes, seconds
            """
            if self.J2000:
                return "{0:02.0f}:{1:02.0f}:{2:02.1f}".format(*SkyCoords.J2000._2hmsdms(self.J2000)[1])
            else:
                return None

        @staticmethod
        def split(J2000, verbose=False):
            """
            Convert a J2000 string to RA,Dec strings

            Args:
                J2000 <str> - 'J[02d:02d:02.1f][+\-][02d:02d:02d]'; Julian equinox

            Kwargs:
                verbose <bool> - verbose mode; print command line statements

            Return:
                ra_str <str>, dec_str <str> - RA,Dec string in hms,dms (or hms) format
            """
            if J2000 is None:
                return None, None
            if not isinstance(J2000, str):
                raise TypeError("J2000 coordinates need to be a string")
            matcher = r"J\s*([0-9\.]+)\s*([\+\-\.0-9]+)"
            pattrn = re.compile(matcher)
            try:
                matched = pattrn.search(J2000.decode('utf-8'))
            except AttributeError:
                matched = pattrn.search(J2000)
            if matched is None:
                raise ValueError(
                    "J2000 string needs format 'J[02d:02d:02.1f][+\-][02d:02d:02d]'")
            ra_str, dec_str = matched.groups()
            if verbose:
                print(ra_str, dec_str)
            return ra_str, dec_str

        @staticmethod
        def _2hmsdms(J2000, round_=False, verbose=False):
            """
            Convert a J2000 string to RA,Dec in hms,dms

            Args:
                J2000 <str> - 'J[02d:02d:02.1f][+\-][02d:02d:02d]'; Julian equinox

            Kwargs:
                round_ <bool> - round to integer seconds
                verbose <bool> - verbose mode; print command line statements

            Return:
                hms, dms <float,float,float> - RA,Dec coordinates in hms,dms format
            """
            rastr, decstr = SkyCoords.J2000.split(J2000)
            matcher = r"\s*(\+?\-?[0-9]{2})\:?\s*([0-9]{2})\:?\s*([0-9\.]{2,5})"
            pattrn = re.compile(matcher)
            try:  # python 2
                ra_match = pattrn.search(rastr.decode('utf-8'))
            except AttributeError:  # python 3
                ra_match = pattrn.search(rastr)
            try:  # python 2
                dec_match = pattrn.search(decstr.decode('utf-8'))
            except AttributeError:  # python 3
                dec_match = pattrn.search(decstr)
            if ra_match is None:
                raise ValueError(
                    'Could not match Right Ascension in J2000 coordinate')
            if dec_match is None:
                raise ValueError(
                    'Could not match Declination in J2000 coordinate')
            hms = [float(ra) for ra in ra_match.groups()]
            dms = [float(dec) for dec in dec_match.groups()]
            if verbose:
                print(hms, dms)
            return hms, dms

        @staticmethod
        def _2deg(J2000, dec_as_hms=False, round_=False, verbose=False):
            """
            Convert a J2000 string to RA,Dec in degrees

            Args:
                J2000 <str> - 'J[02d:02d:02.1f][+\-][02d:02d:02d]'; Julian equinox

            Kwargs:
                dec_as_hms <bool> - read the Dec in <hours,minutes,seconds>
                round_ <bool> - round to integer seconds
                verbose <bool> - verbose mode; print command line statements

            Return:
                ra <float>, dec <float> - RA,Dec coordinates in degrees
            """
            if J2000 is None:
                degs = [None, None]
                if verbose:
                    print(degs)
                return degs
            if not isinstance(J2000, str):
                raise ValueError('J2000 coordinates need to be a string')
            hms, dms = SkyCoords.J2000._2hmsdms(J2000, round_=round_)
            ra_deg = SkyCoords.hms2deg(*hms)
            dec_deg = SkyCoords.dms2deg(*dms)
            if dec_as_hms:
                dec_deg = SkyCoords.hms2deg(*dms)
            degs = [ra_deg, dec_deg]
            if verbose:
                print(degs)
            return degs

        @staticmethod
        def _2arcsec(J2000, **kwargs):
            """
            Convert a J2000 string to RA,Dec in arcsecs

            Args:
                J2000 <str> - 'J[02d:02d:02.1f][+\-][02d:02d:02d]'; Julian equinox

            Kwargs:
                dec_as_hms <bool> - read the Dec in <hours,minutes,seconds>
                verbose <bool> - verbose mode; print command line statements

            Return:
                ra <float>, dec <float> - RA,Dec coordinates in arcsec
            """
            verbose = kwargs.pop('verbose', False)
            radec = SkyCoords.J2000._2deg(J2000, **kwargs)
            arcsecs = SkyCoords.deg2arcsec(*radec)
            if verbose:
                print(arcsecs)
            return arcsecs

        @staticmethod
        def _2pixels(J2000, **kwargs):
            """
            Convert a J2000 string to RA,Dec in pixels

            Args:
                J2000 <str> - 'J[02d:02d:02.1f][+\-][02d:02d:02d]'; Julian equinox

            Kwargs:
                integers <bool> - return pixels as integers
                px2arcsec_scale <float,float> - pixel scale in arcsec/pixels
                reference_pixel <int,int> - reference pixel in .fits file
                reference_value <float,float> - reference coordinates in the .fits file (in degs)
                dec_as_hms <bool> - read the Dec in <hours,minutes,seconds>
                verbose <bool> - verbose mode; print command line statements

            Return:
                pixels <int,int> - converted pixel coordinates
            """
            verbose = kwargs.pop('verbose', False)
            dec_as_hms = kwargs.pop('dec_as_hms', False)
            radec = SkyCoords.J2000._2deg(J2000, dec_as_hms=dec_as_hms)
            pixels = SkyCoords.deg2pixels(*radec, **kwargs)
            if verbose:
                print(pixels)
            return pixels


# MAIN FUNCTION ###############################################################
def main(case, args):
    """
    Main function to use SkyCoords from command line

    Args:
        case <str> - test case argument (J2000 coordinate string)
        args <namespace> - namespace of keyword arguments for all functions

    Kwargs:
        end_message <str> - optional message for printing at the end

    Return:
        None
    """
    refpx = args.refpx
    refval = args.refval
    pxscale = args.scale
    if args.ra is None or args.dec is None:
        J2000 = case
        SkyCoords.from_J2000(J2000, px2arcsec_scale=pxscale,
                             reference_pixel=refpx, reference_value=refval,
                             verbose=args.verbose)
    elif args.ra is not None and args.dec is not None:
        SkyCoords(args.ra, args.dec, reference_pixel=refpx, reference_value=refval,
                  px2arcsec_scale=pxscale, verbose=args.verbose)


def parse_arguments():
    """
    Parse command line arguments

    Args/Kwargs:
        None (only from command-line)

    Return:
        parser <ArguementParser object> - return the parsed argument
    """
    from argparse import ArgumentParser, RawTextHelpFormatter
    parser = ArgumentParser(description=__doc__, formatter_class=RawTextHelpFormatter)
    # main args
    parser.add_argument("case", nargs='?',
                        help="J2000 coordinate for skycoords to read",
                        default='J143454.4+522850')
    parser.add_argument("-r", "--ra", dest="ra", type=float,
                        help="The Right Ascension coordinate in degrees")
    parser.add_argument("-d", "--dec", dest="dec", type=float,
                        help="The Declination coordinate in degrees")
    parser.add_argument("--scale", metavar=("<dx", "dy>"), nargs=2, type=float,
                        help="Pixel-to-arcsec scale for x (-RA) and y (Dec) direction",
                        default=[None, None])
    parser.add_argument("--refpx", metavar=("<x", "y>"), nargs=2, type=float,
                        help="Coordinates of the reference pixel",
                        default=[None, None])
    parser.add_argument("--refval", metavar=("<ra", "dec>"), nargs=2, type=float,
                        help="Values of the reference pixel",
                        default=[None, None])
    # mode args
    parser.add_argument("-v", "--verbose", dest="verbose", action="store_true",
                        help="Run program in verbose mode",
                        default=False)
    parser.add_argument("-t", "--test", "--test-mode", dest="test_mode", action="store_true",
                        help="Run program in testing mode",
                        default=False)
    args = parser.parse_args()
    case = args.case
    delattr(args, 'case')
    return parser, case, args


###############################################################################
if __name__ == "__main__":
    parser, case, args = parse_arguments()
    no_input = len(sys.argv) <= 1 and 'J143454.4+522850' in case
    if no_input:
        parser.print_help()
    elif args.test_mode:
        sys.argv = sys.argv[:1]
        from gleam.test.test_skycoords import TestSkyCoords
        TestSkyCoords.main()
    else:
        main(case, args)
