#!/usr/bin/env python
"""
@author: phdenzel

Learn everything about .fits files with SkyF

TODO:
   - center estimate is sometimes off by ~ 1-23 pixels... what's the issue?
   - rewrite cutout to use roi
"""
###############################################################################
# Imports
###############################################################################
from gleam.skycoords import SkyCoords
from gleam.roiselector import ROISelector
from gleam.megacam import MEGACAM_FPROPS
from gleam.utils.encode import GLEAMEncoder, GLEAMDecoder

import sys
import os
import re
import copy
import math
import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt


__all__ = ['SkyF']


###############################################################################
class SkyF(object):
    """
    Framework for patches of the sky (.fits files) of a single band
    """
    params = ['data', 'hdr', 'px2arcsec', 'refval', 'refpx', 'photzp', 'roi']
    hdr_keys = ['FILTER', 'NAXIS1', 'NAXIS2', 'CRPIX1', 'CRPIX2', 'CRVAL1', 'CRVAL2',
                'CD1_1', 'CD1_2', 'CD2_1', 'CD2_2', 'OBJECT', 'PHOTZP']

    def __init__(self, filepath, data=None, hdr=None,
                 px2arcsec=None, refpx=None, refval=None,
                 photzp=None, roi=None, _light_model=None,
                 verbose=False):
        """
        Initialize parsing of a fits file with a file name

        Args:
            filepath <str> - path to .fits file (shortcuts are automatically resolved)

        Kwargs:
            data <list/np.ndarray> - add the data of the .fits file directly
            hdr <dict> - add the header of the .fits file directly
            px2arcsec <float,float> - overwrite the pixel scale in the .fits header (in arcsecs)
            refpx <int,int> - overwrite reference pixel coordinates in .fits header (in pixels)
            refval <float,float> - overwrite reference pixel values in .fits header (in degrees)
            photzp <float> - overwrite photometric zero-point information
            verbose <bool> - verbose mode; print command line statements

        Return:
            <SkyF object> - standard initializer
        """
        # .fits file data
        self.filepath = self.check_path(filepath)
        self.data, self.hdr = self.parse_fitsfile(filepath)  # data[Dec, RA] and .fits header
        if data is not None:
            self.data = np.asarray(data)
        if hdr is not None:
            self.hdr = hdr
        if px2arcsec is not None and len(px2arcsec) == 2:
            self.px2arcsec = px2arcsec
        if refpx is not None and len(refpx) == 2:
            self.refpx = refpx
        if refval is not None and len(refval) == 2:
            self.refval = refval
        if photzp is not None:
            self.photzp = photzp
        self.mag_formula = None
        if self.hdr is not None:
            self.mag_formula = self.mag_formula_from_hdr(self.hdr, photzp=self.photzp)
        self.roi = ROISelector.from_gleamobj(self)
        if roi is not None:
            self.roi = roi
        # get rid of methods when inherited
        if self.__class__.__name__ != SkyF.__name__ and hasattr(SkyF, 'show_fullsky'):
            del SkyF.show_fullsky
        # some verbosity
        if verbose:
            print(self.__v__)

    def __eq__(self, other):
        if isinstance(other, SkyF):
            return \
                np.array_equal(self.data, other.data) \
                and self.center == other.center \
                and self.photzp == other.photzp
        else:
            NotImplemented

    def encode(self):
        """
        Using md5 to encode specific information
        """
        import hashlib
        j = self.__json__
        s = ', '.join([
            str(j[k]) for k in self.__class__.params if not isinstance(j[k], dict)
        ]).encode('utf-8')
        code = hashlib.md5(s).hexdigest()
        return code

    def __hash__(self):
        """
        Using encode to create hash
        """
        return hash(self.encode())

    def __copy__(self):
        args = (self.filepath,)
        kwargs = {k: self.__getattribute__(k) for k in self.__class__.params}
        return self.__class__(*args, **kwargs)

    def __deepcopy__(self, memo):
        args = (copy.deepcopy(self.filepath, memo),)
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
        Select attributes for json write

        Args/Kwargs:
            None

        Return:
            jsn_dict <dict> - a first-layer serialized json dictionary
        """
        jsn_dict = {}
        for k in self.__class__.params:
            if hasattr(self, k):
                val = self.__getattribute__(k)
                if isinstance(val, np.ndarray):
                    val = val.tolist()
                jsn_dict[k] = val
        jsn_dict['__type__'] = self.__class__.__name__
        return jsn_dict

    @classmethod
    def from_jdict(cls, jdict, filepath=None, verbose=False):
        """
        Initialize from a json-originated dictionary

        Args:
            jdict <dict> - serialized object dictionary

        Kwargs:
            filepath <str> - separate filepath setter
            verbose <bool> - verbose mode; print command line statements

        Return:
            <cls object> - initializer from a json-originated dictionary

        Note:
            - used by GLEAMDecoder in from_json
        """
        self = cls(None, **jdict)
        self.data = np.asarray(self.data)
        self.roi.data = self.data
        if filepath:
            self.filepath = filepath
        if verbose:
            print(self.__v__)
        return self

    @classmethod
    def from_json(cls, jsn, verbose=False):
        import json
        GLEAMDecoder.decoding_kwargs['filepath'] = os.path.realpath(jsn.name)
        self = json.load(jsn, object_hook=GLEAMDecoder.decode)
        if verbose:
            print(self.__v__)
        return self

    def jsonify(self, name=None, with_hash=True, verbose=False):
        """
        Export instance to JSON

        Args:
            None

        Kwargs:
            name <str> - export a JSON with savename as file name
            with_hash <bool> - append md5 hash to filename, making it truly unique
            verbose <bool> - verbose mode; print command line statements

        Return:
            json <str> - a JSON string output if save is False
            filename <str> - filename with which the JSON file is saved, if save is True
        """
        import json
        jsn = json.dumps(self, cls=GLEAMEncoder, sort_keys=True, indent=4)
        if name:
            filename = self.json_filename(filename=name, with_hash=with_hash)
            with open(filename, 'w') as output:
                json.dump(self, output, cls=GLEAMEncoder, sort_keys=True, indent=4)
            return filename
        if verbose:
            print(jsn)
        return jsn

    def json_filename(self, filename=None, with_hash=True, verbose=False):
        """
        Generate a unique filename for json export

        Args:
            None

        Kwargs:
            filename <str> - filename to which type and md5 hash are appended
            with_hash <bool> - append md5 hash to filename, making it truly unique

        Return:
            filename <str> - unique filename
        """
        finput = filename
        if filename is None:
            filename = ''
            finput = ''
        if len(filename.split('.')) > 1:
            filename = filename.split('.')[:-1]
        else:
            filename = filename.split('.')
        if with_hash:
            if self.__class__.__name__.lower() not in finput:
                filename += ['{}#{}'.format(
                    self.__class__.__name__.lower(), self.encode()[:-3])]
        if 'json' not in filename[-1]:
            filename += ['json']
        filename = '.'.join(filename)
        return filename

    def __str__(self):
        return "SkyF({}@[{:.4f}, {:.4f}])".format(self.band, *self.center)

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
        return ['filename', 'filepath', 'band', 'naxis1', 'naxis2', 'naxis_plus',
                'refval', 'refpx', 'center', 'px2deg', 'px2arcsec', 'megacam_range',
                'field', 'photzp', 'mag_formula', 'roi']

    @property
    def __v__(self):
        """
        Info string for test printing

        Args/Kwargs:
            None

        Return:
            <str> - test of SkyF attributes
        """
        return "\n".join([t.ljust(20)+"\t{}".format(self.__getattribute__(t)) for t in self.tests])

    @property
    def filename(self):
        """
        File name of the .fits file

        Args/Kwargs:
            None

        Return:
            filename <str> - Base name of the file path
        """
        if self.filepath:
            return os.path.basename(self.filepath)

    @staticmethod
    def check_path(filepath, check_ext=True, extensions=['.fits', '.fit', '.fts'],
                   verbose=False):
        """
        Check whether the input file exists and is readable

        Args:
            filepath <str> - path string to the .fits file

        Kwargs:
            check_ext <bool> - check path for .fits extension
            verbose <bool> - verbose mode; print command line statements

        Return:
            filepath <str> - verified path string to the .fits file
        """
        # validate input
        if filepath is None:
            return None
        if not isinstance(filepath, str):
            raise TypeError("Input path needs to be string")
        # expand shortcuts
        if '~' in filepath:
            filepath = os.path.expanduser(filepath)
        if not os.path.isabs(filepath):
            filepath = os.path.abspath(filepath)
        # check if path exists
        if not os.path.exists(filepath):
            try:  # python 3
                FileNotFoundError
            except NameError:  # python 2
                FileNotFoundError = IOError
            raise FileNotFoundError("'{}' does not exist".format(filepath))
        # optionally check extension
        if check_ext and True not in [filepath.endswith(ext) for ext in extensions]:
            raise ValueError('Input file path need to be a .fits file')
        # some verbosity
        if verbose:
            print("Valid input path:\n{}".format(filepath))
        return filepath

    @staticmethod
    def parse_fitsfile(filepath, header=True, header_only=False, verbose=False):
        """
        Parse the .fits file's data and/or header

        Args:
            filepath <str> - path string to the .fits file

        Kwargs:
            header <bool> - parse header of .fits file
            header_only <bool> - parse only header of .fits file, skip data parsing
            verbose <bool> - verbose mode; print command line statements

        Return:
            data <numpy.ndarray>, hdr <dict> - parsed .fits file
        """
        if header_only:
            header = True
        filepath = SkyF.check_path(filepath)
        if filepath is None:
            return None, None
        # get data and header
        data, hdrobj = fits.getdata(filepath, header=True)
        # make a dict out of the header object
        if header:
            hdr = dict(hdrobj)
            keys = sorted(set(hdrobj), key=hdrobj.index)
            for k in keys:  # reformatting comments
                if isinstance(hdr[k], fits.header._HeaderCommentaryCards):
                    comments = [i for i in hdr[k] if i is not '']
                    for i, c in enumerate(comments):
                        c2 = comments[min(i+1, len(comments)-1)]
                        if c2.startswith(" ") and c2.strip()[0].islower():  # mult lines comments
                            comments[i] = c.strip()+" "+c2.strip()
                            del comments[min(i+1, len(comments)-1)]
                        else:
                            comments[i] = c.strip()
                    hdr[k] = comments
        # standardize header dictionary
        for key in SkyF.hdr_keys:
            if key not in hdr.keys():
                hdr[key] = None
        if 'COMMENT' not in hdr.keys():
            hdr['COMMENT'] = []
        # standardize data list/array
        if hasattr(data[0], 'field'):  # data rows are FITS_record > np.ndarray
            standard = []
            for i in range(len(data[0])):
                standard.append(np.asarray(data.field(i)))
            data = np.asarray(standard)
        data = data.squeeze()
        # deal with data if pixeltype is healpix
        if 'PIXTYPE' in hdr and 'HEALPIX' in hdr['PIXTYPE']:
            try:  # import healpy
                from healpy import pixelfunc
            except ImportError:
                raise ImportError(
                    "For this fits file format you need to install healpy\n"
                    + "Use: pip install --user healpy")
            # remap 1D array according to ordering scheme
            mapping = []
            if 'ORDERING' in hdr:
                nside = int(hdr['NSIDE']) if 'NSIDE' in hdr else 1024
                for d in data:
                    if hdr['ORDERING'] == 'NESTED':
                        idx = pixelfunc.ring2nest(nside, np.arange(d.size, dtype=np.int32))
                        mapping.append(d[idx])
                    elif hdr['ORDERING'] == 'RING':
                        idx = pixelfunc.nest2ring(nside, np.arange(d.size, dtype=np.int32))
                        mapping.append(d[idx])
                    else:  # assume NESTED
                        print("Ordering undefined: defaulting back to NESTED ordering scheme")
                        idx = pixelfunc.ring2nest(nside, np.arange(d.size, dtype=np.int32))
                        mapping.append(d[idx])
            if len(mapping) == 1:
                data = mapping[0]
            else:
                data = mapping
        # some verbosity
        if verbose:
            if not header_only:
                print('DATA:')
                print(str(data)+"\n")
            if header:
                print('HEADER:')
                for k in keys:
                    if isinstance(hdr[k], list):
                        print(k+"\n\t"+"\n\t".join(hdr[k]))
                    else:
                        print(k.ljust(20)+str(hdr[k]))
        # select what to return
        if header and header_only:
            return hdr
        elif header:
            return data, hdr
        else:
            return data

    @property
    def band(self):
        """
        Simplified string of the .fits image's band

        Args/Kwargs:
            None

        Return:
            band <str> - simplified string of the image's band
        """
        if self.hdr and self.hdr['FILTER'] is not None:
            return self.hdr['FILTER'].split('.')[0]
        else:
            return ""

    @property
    def naxis1(self):
        """
        Length of axis 1

        Args/Kwargs:
            None

        Return:
            naxis1 <int> - number of pixels along axis 1
        """
        if self.hdr and 'NAXIS1' in self.hdr and self.hdr['NAXIS1'] is not None:
            return self.hdr['NAXIS1']
        elif self.data:
            return self.data.shape[0]
        else:
            return 0

    @property
    def naxis2(self):
        """
        Length of axis 2

        Args/Kwargs:
            None

        Return:
            naxis2 <int> - number of pixels along axis 2
        """
        if self.hdr and 'NAXIS2' in self.hdr and self.hdr['NAXIS2'] is not None:
            return self.hdr['NAXIS2']
        elif self.data:
            return self.data.shape[1]
        else:
            return 0

    @property
    def naxis_plus(self):
        """
        Length of axis 3 and onwards

        Args/Kwargs:
            None

        Return:
            naxis_plus <list(int)> - number of pixels along each additional axis
        """
        naxis = 2
        if self.hdr and 'NAXIS' in self.hdr:
            naxis = self.hdr['NAXIS']
        elif self.hdr:
            naxis = sum(['NAXIS' in k for k in self.hdr.keys()])
        if naxis > 2:
            plus = []
            for i in range(3, naxis+1):
                plus.append(self.hdr['NAXIS{}'.format(i)])
            return plus
        else:
            return None

    @property
    def refval(self):
        """
        Reference pixel from .fits image

        Args/Kwargs:
            None

        Return:
            refval <float,float> - reference pixel coordinates in degrees
        """
        if self.hdr:
            return self.refpx_from_hdr(self.hdr, as_radec=True)

    @refval.setter
    def refval(self, refval):
        """
        Change reference pixel from .fits image by changing the CRVAL1/CRVAL2 entries

        Args:
            refval <float,float> - reference pixel coordinates in degrees

        Kwargs/Return:
            None

        Note:
            - those entry changes will not be saved in the .fits file, only in the dict
        """
        self.hdr['CRVAL1'] = refval[0]
        self.hdr['CRVAL2'] = refval[1]

    @property
    def refpx(self):
        """
        Reference pixel from .fits image

        Args/Kwargs:
            None

        Return:
            refpx <int,int> - reference pixel coordinates in pixels
        """
        if self.hdr:
            return self.refpx_from_hdr(self.hdr, as_radec=False)

    @refpx.setter
    def refpx(self, refpx):
        """
        Change reference pixel values from .fits image by changing the CRPIX1/CRPIX2 entries

        Args:
            refpx <int,int> - reference pixel coordinates in pixels

        Kwargs/Return:
            None

        Note:
            - those entry changes will not be saved in the .fits file, only in the dict
        """
        self.hdr['CRPIX1'] = refpx[0]
        self.hdr['CRPIX2'] = refpx[1]

    @property
    def px2deg(self):
        """
        Pixel scale, i.e. pixel-to-degree conversion factors, for RA and Dec

        Args/Kwargs:
            None

        Return:
            px2deg <float,float> - pixel-to-degrees conversion factor in units of degrees/pixels
        """
        if self.hdr:
            return self.pxscale_from_hdr(self.hdr, as_degrees=True)

    @px2deg.setter
    def px2deg(self, pxscale):
        """
        Change the pixel scale of the .fits file by changing the distortion matrix entries

        Args:
            pxscale <float,float> - pixel-to-degrees conversion factor in units of degrees/pixels

        Kwargs/Return:
            None

        Note:
            - those entry changes will not be saved in the .fits file, only in the dict
        """
        self.hdr['CD1_1'] = -pxscale[0]
        self.hdr['CD1_2'] = 0
        self.hdr['CD2_1'] = 0
        self.hdr['CD2_2'] = pxscale[1]

    @property
    def px2arcsec(self):
        """
        Pixel scale, i.e. pixel-to-arcsec conversion factors, for RA and Dec

        Args/Kwargs:
            None

        Return:
            pxscale <float,float> - pixel-to-arcsec conversion factor in units of arcsecs/pixels
        """
        if self.hdr:
            return self.pxscale_from_hdr(self.hdr, as_degrees=False)

    @px2arcsec.setter
    def px2arcsec(self, pxscale):
        """
        Change the pixel scale of the .fits file by changing the distortion matrix entries

        Args:
            pxscale <float,float> - pixel-to-arcsec conversion factor in units of arcsecs/pixels

        Kwargs/Return:
            None

        Note:
            - those entry changes will not be saved in the .fits file, only in the dict
        """
        if pxscale[0] and pxscale[1]:
            self.hdr['CD1_1'] = -pxscale[0]/3600.
            self.hdr['CD1_2'] = 0
            self.hdr['CD2_1'] = 0
            self.hdr['CD2_2'] = pxscale[1]/3600.

    @property
    def extent(self):
        """
        The extent of the .fits data in arcsec with origin in center (left, bottom, top, right)

        Args/Kwargs:
            None

        Return:
            extent <list(float)> - the map extent
        """
        return [-0.5*self.px2arcsec[0]*self.naxis1, -0.5*self.px2arcsec[1]*self.naxis2,
                0.5*self.px2arcsec[0]*self.naxis1, 0.5*self.px2arcsec[1]*self.naxis2]

    def p2skycoords(self, position, unit='arcsec', relative=True, verbose=False):
        """
        Convert a position into skycoords positions with the skyfs reference pixel information

        Args:
            position <int/float,int/float> - position (relative to lens/center position)

        Kwargs:
            unit <str> - unit of the position input (arcsec, degree, pixel)
            relative <bool> - position relative to lens or center;
                              if False, position is assumed to be absolute (origin at refpx)
            verbose <bool> -  verbose mode; print command line statements

        Return:
            skyc <SkyCoords object> - the position converted into a SkyCoords object
        """
        if unit in ['px', 'pixel', 'pixels']:
            if relative:
                skyc = SkyCoords.from_pixels(*position, **self.center.coordkw)
            else:
                skyc = SkyCoords.from_pixels(*position)
        elif unit in ['arcsec', 'arcsecs', 'arcs']:
            if relative:
                if hasattr(self, 'lens') and self.lens is not None:
                    origin = self.lens.arcsecs
                elif self.center is not None:
                    origin = self.center.arcsecs
                else:
                    origin = [0, 0]
                skyc = SkyCoords.from_arcsec(*[o+p for o, p in zip(origin, position)])
            else:
                skyc = SkyCoords.from_arcsecs(*position)
        else:  # unit is degrees
            if relative:
                if hasattr(self, 'lens') and self.lens is not None:
                    origin = self.lens.radec
                elif self.center is not None:
                    origin = self.center.radec
                else:
                    origin = [0, 0]
                skyc = SkyCoords.from_degrees(*[o+p for o, p in zip(origin, position)])
            else:
                skyc = SkyCoords.from_degrees(*position)
        if verbose:
            print(skyc.__v__)
        return skyc

    def yx2idx(self, y, x, cols=None, verbose=False):
        """
        Index mapping scheme for the 2D plane to 1D indices

        Args:
            y, x <int,int> - the pixel index coordinates for each data point

        Kwargs:
            cols <int> - number of columns of the plane described by the pixel coordinates
            verbose <bool> -  verbose mode; print command line statements

        Return:
            idx <int> - unique 1D index of a pixel
        """
        if cols is None:
            cols = self.data.shape[1]
            # rows = self.data.shape[0]
        idx = y * cols + x
        if verbose:
            print(idx)
        return idx

    def idx2yx(self, idx, cols=None, verbose=False):
        """
        Index mapping scheme for a 1D index to a plane, inverse of yx2idx

        Args:
            idx <int> - unique 1D index of a pixel

        Kwargs:
            cols <int> - number of columns of the plane described by the pixel coordinates
            verbose <bool> -  verbose mode; print command line statements

        Return:
            y, x <int,int> - the pixel index coordinates for the pixel with the provided index
        """
        if cols is None:
            cols = self.data.shape[1]
            # rows = self.data.shape[0]
        y = idx // cols
        x = idx % cols
        if verbose:
            print(y, x)
        return y, x

    def theta(self, position, offset=1e-12, origin=None, verbose=False):
        """
        Transform pixel coordinates into theta vector coordinates with origin in center

        Args:
           position <int,int/SkyCoords object> - the pixel coordinates of the position

        Kwargs:
           origin <int,int/SkyCoords object>
           offset <float> - small offset to avoid having 0 in the center

        Return:
           theta <float,float> - theta vector coordinates in arcsecs
        """
        # position
        if isinstance(position, int):  # 1D pixel index position
            position = self.idx2yx(position)
        if isinstance(position, (tuple, list, np.ndarray)):  # 2D pixel index position
            position = self.p2skycoords(position, unit='pixel')
        # origin
        if origin is None:
            if hasattr(self, 'lens'):
                origin = self.lens
            else:
                origin = self.center
        elif isinstance(origin, (tuple, list, np.ndarray)):
            origin = self.p2skycoords(origin, unit='pixel')
        theta = position.get_shift_to(origin, unit='arcsec', rectangular=True)
        theta = [t + offset for t in theta]
        if verbose:
            print(theta)
        return theta

    @property
    def photzp(self):
        """
        Photometric zero-point

        Args/Kwargs:
            None

        Return:
            photzp <float> - photometric zero-point needed for AB magnitude conversion
        """
        if self.hdr:
            return self.hdr['PHOTZP']

    @photzp.setter
    def photzp(self, photzp):
        """
        Change photometric zero-point by changing the PHOTZP entry

        Args:
            photzp <float> - photometric zero-point needed for AB magnitude conversion

        Kwargs/Return:
            None
        Note:
            - those entry changes will not be saved in the .fits file, only in the dict
        """
        self.hdr['PHOTZP'] = photzp
        self.mag_formula = self.mag_formula_from_hdr(self.hdr, photzp=photzp)

    @property
    def field(self):
        """
        Field signature of the survey (e.g. for CFHTLS: W3+3-2,...)

        Args/Kwargs:
            None

        Return:
            field <str> - Field signature
        """
        if self.hdr and self.hdr['OBJECT'] is not None:
            field = re.sub("\.", "", self.hdr['OBJECT'].upper())
            # standardize, i.e. if field ends with 0, switch ...-0 => ...+0
            if field[-1] == '0' and field[-2] == '-':
                field = field[:-2]+re.sub("\-", "+", field[-2:])
            return field

    @property
    def megacam_range(self):
        """
        Image selection range in the MegaPipe image stacking pipeline

        Args/Kwargs:
            None

        Return:
            selection <(int,int),(int,int)> - lower left and upper right pixel coordinates
        """
        try:
            field_props = MEGACAM_FPROPS[self.field]
        except KeyError:
            print("Properties of that field are unknown [{}]".format(self.field))
            return None
        # get field size
        refx, refy = field_props['X'], field_props['Y']
        refra, refdec = self.refval
        diff_ra, diff_dec = refra-self.center[0], refdec-self.center[1]
        x = int(refx+diff_ra*np.cos(refdec*np.pi/180)/self.px2deg[0])
        y = int(refy-diff_dec/self.px2deg[1])
        return [(math.ceil(x-(self.naxis1-1)/2), math.floor(x+self.naxis1/2)),
                (math.ceil(y-(self.naxis2-1)/2), math.floor(y+self.naxis2/2))]

    @property
    def center(self):
        """
        The approximate center of the .fits file's image

        Args/Kwargs:
            None

        Return:
            center <Skycoords object> - center coordinate of the .fits file's image
        """
        if self.hdr:
            return SkyCoords.from_pixels(self.naxis1/2, self.naxis2/2,
                                         px2arcsec_scale=self.px2arcsec,
                                         reference_pixel=self.refpx, reference_value=self.refval)
        else:
            return SkyCoords.empty()

    @property
    def indices(self):
        """
        A grid of indices

        Args/Kwargs:
            None

        Return:
            indices <np.ndarray> - index array with shape (2+len(naxis_plus), naxis1, naxis2)
        """
        if self.naxis_plus is None:
            idx = np.indices((self.naxis1, self.naxis2))
            idx = np.flip(idx.T, 2)
        else:
            naxes = (self.naxis1, self.naxis2)+self.naxis_plus
            idx = np.indices(naxes)
            idx = np.flip(idx.T, len(naxes))
        return idx

    @property
    def idcs_flat(self):
        """
        A flat array with index positions (e.g. for iterating over indices)

        Args/Kwargs:
            None

        Return:
            idx <np.ndarray> - the flattened indices array
        """
        if self.naxis_plus is None:
            return self.indices.reshape(self.naxis1*self.naxis2, 2)
        else:
            naxes = (self.naxis1, self.naxis2)+self.naxis_plus
            return self.indices.reshape(self.naxis1*self.naxis2, len(naxes))

    @property
    def grid(self):
        """
        A SkyCoords grid with RA,Dec coordinates (in degrees) corresponding to data indices

        Args/Kwargs:
            None

        Return:
            grid <np.ndarray(SkyCoords, dict)> - grid with SkyF coordinates and keyword info
        """
        return np.array(
            [SkyCoords.from_pixels(x, y, px2arcsec_scale=self.px2arcsec,
                                   reference_pixel=self.refpx, reference_value=self.refval)
             for (x, y) in self.idcs_flat],
            dtype=object
        ).reshape(self.naxis1, self.naxis2, 2)

    @property
    def magnitudes(self):
        """
        A magnitude map converted from data

        Args/Kwargs:
            None

        Return:
            mags <np.ndarray> - magnitude map converted from data
        """
        mags = self.mag_formula(self.data)
        # filter out nans
        mags[np.isnan(mags)] = np.max(mags[~np.isnan(mags)])
        return mags

    def total_magnitude(self, radius, center=None, verbose=False):
        """
        Radially integrated total magnitude

        Args:
            radius <int> - integration radius in pixels

        Kwargs:
            center <int,int> - integration center in pixels; if None, the image's center is used
            verbose <bool> -  verbose mode; print command line statements

        Return:
            totmag <float> - radially integratedd total magnitude
        """
        if center is None:
            center = (self.naxis1//2, self.naxis2//2)
        # select integration area
        msk = np.indices(self.data.shape)
        msk[0] -= center[0]
        msk[1] -= center[1]
        msk = np.square(msk)  # square for easier distance comparison
        msk = msk[0, :, :] + msk[1, :, :] < radius*radius
        fluxsum = np.sum(self.data[msk])  # integrated flux
        fluxsum = max(0, fluxsum)
        totmag = self.mag_formula(fluxsum)  # magnitude of integrated flux
        # some verbosity
        if verbose:
            print(totmag)
        return totmag

    def image_f(self, cmap='magma', draw_roi=False):
        """
        An 8-bit PIL.Image of the .fits data

        Args:
            None

        Kwargs:
            cmap <str> - a cmap string from matplotlib.colors.Colormap
            draw_roi <bool> - draw the ROI objects on top of data

        Return:
            f_image <PIL.Image object> - a colorized image object
        """
        from PIL import Image
        cmap = plt.get_cmap(cmap)
        if self.data is not None:
            lower = self.data.min()
            upper = self.data.max()
            img = Image.fromarray(
                np.uint8(255*cmap((self.data-lower)/(upper-lower)))
            )
            if draw_roi:
                img = self.roi.draw_rois(img)
            img = img.transpose(Image.FLIP_TOP_BOTTOM)
            return img
        return self.data

    def cutout(self, window, center=None, verbose=False):
        """
        Cutout a square segment of the .fits file's image

        Args:
            window <int> - window size in pixels of the cutout
            center <int,int> - pixel coordinates of the window's center

        Kwargs:
            verbose <bool> - verbose mode; print command line statements

        Return:
            cutout <np.ndarray> - cutout part of the .fits file's image

        TODO: include use of ROISelector or remove
        """
        if center is None:
            center = (self.naxis1//2, self.naxis2//2)
        if window % 2:
            co = window//2 + 1, window//2
        else:
            co = window//2, window//2
        cutout = self.data[center[1]-co[0]:center[1]+co[1],
                           center[0]-co[0]:center[0]+co[1]]
        # some verbosity
        if verbose:
            print("Cropped data around {} of size {}x{}".format(center, window, window))
            print(cutout)
        return cutout

    @staticmethod
    def flatfield(data, size=0.25, npts=500, verbose=False):
        """
        Calculate signals and variances from random image data cutouts
        using a compensation method for the flat-field effect

        Args:
            data <np.ndarray> - image data array, must be 2D

        Kwargs:
            size <float> - window size as a number between [0, 1]
            npts <int> - number of points with coordinates (variance, signal)
            verbose <bool> - verbose mode; print command line statements

        Return:
            signals, variances <np.ndarrays> - signals and variances for linear regression method
        """
        dims = data.shape
        if len(dims) > 2:
            raise ValueError("Please input the image as a 2D np.ndarray!")
        w, h = int(size*dims[0]), int(size*dims[1])
        rcs = np.c_[np.random.randint(h//2, dims[1]-h//2, (npts)),
                    np.random.randint(w//2, dims[0]-w//2, (npts))]
        s = [(slice(p[0]-h//2, p[0]+h//2), slice(p[1]-w//2, p[1]+w//2)) for p in rcs]
        A = [data[si] for si in s]
        signals = [np.mean(wA) for wA in A]
        r = [A[0]/wB for wB in A]
        B = [wB*rAB for wB, rAB in zip(A, r)]
        AB = [a-b for a, b in zip(A, B)]
        sgs = [np.std(ab) for ab in AB]
        variances = [sg*sg for sg in sgs]
        return np.asarray(signals[1:]), np.asarray(variances[1:])

    def gain(self, **kwargs):
        """
        Calculate an estimate for the gain of the .fits image

        Args:
            None

        Kwargs:
            signals <np.ndarray> - calculate gain from given signals
            variances <np.ndarray> - calculate gain from given variances
            size <float> - window size as a number between [0, 1]
            npts <int> - number of points with coordinates (variance, signal)
            verbose <bool> - verbose mode; print command line statements

        Return:
            gain <float> - an estimate of the gain
            bias <float> - an estimate of other (missing) noise components
        """
        verbose = kwargs.pop('verbose', False)
        signals = kwargs.pop('signals', None)
        variances = kwargs.pop('variances', None)
        if signals is None and variances is None:
            signals, variances = self.flatfield(self.data, **kwargs)
        A = np.vstack([variances, np.ones(len(signals))]).T
        gain, bias = np.linalg.lstsq(A, signals, rcond=None)[0]
        if verbose:
            print("Gain: {}".format(gain))
            print("Bias: {}".format(bias))
        return gain, bias

    def sigma(self, f=1, add_bias=0, flat=False, verbose=False):
        """
        Noise sigma estimation map

        Args:
            None

        Kwargs:
            f <float> - a fudge factor for tuning
            add_bias <float> - other noise components

        Return:
            sigma <np.ndarray> - noise estimates for each data pixel
        """
        sgma = np.sqrt(np.abs(self.data))
        if flat:
            sgma = np.array([sgma[self.idx2yx(i)] for i in range(sgma.size)])
        if add_bias:
            sgma = sgma + add_bias
        return f*sgma

    def sigma2(self, f=1, add_bias=0, add_bias2=0, flat=False, verbose=False):
        """
        Squared noise sigma estimation map

        Args:
            None

        Kwargs:
            f <float> - a fudge factor for tuning

        Return:
            sigma <np.ndarray> - noise estimates for each data pixel
        """
        sgma2 = np.abs(self.data)
        if flat:
            sgma2 = np.array([sgma2[self.idx2yx(i)] for i in range(sgma2.size)])
        if add_bias2:
            return f*sgma2 + add_bias2
        elif add_bias:
            return f*sgma2 + add_bias*add_bias
        else:
            return f*sgma2

    @staticmethod
    def pxscale_from_hdr(hdr, as_degrees=False, verbose=False):
        """
        Get pixel scale from .fits file header (assuming ctype=RA/DEC and cunit=deg)

        Args:
            hdr <dict> - dictionary of the .fits file header

        Kwargs:
            as_degrees <bool> - return result in degrees instead of arcsecs
            verbose <bool> -  verbose mode; print command line statements

        Return:
            cdelt1 <float>, cdelt2 <float> - pixel scales for RA and Dec
        """
        if any([(k not in hdr) or (hdr[k] is None) for k in ['CD1_1', 'CD1_2', 'CD2_1', 'CD2_2']]):
            return [None, None]
        if hdr['CD1_1']//abs(hdr['CD1_1']) > 0:
            print("Warning! In the WCS rotation East is not to the left!")
        cdelt1 = math.sqrt(hdr['CD1_1']*hdr['CD1_1']+hdr['CD2_1']*hdr['CD2_1'])
        cdelt2 = math.sqrt(hdr['CD1_2']*hdr['CD1_2']+hdr['CD2_2']*hdr['CD2_2'])
        # calculate the determinante of the distortion matrix
        det = hdr['CD1_1']*hdr['CD2_2']-hdr['CD1_2']*hdr['CD2_1']
        sign = det//abs(det)  # the sign of the determinante
        # rotation of the coordinate system axes relative to the world coordinates
        crota2 = 0.5*(math.atan2(-hdr['CD2_1'], sign*hdr['CD1_1'])
                      + math.atan2(sign*hdr['CD1_2'], hdr['CD2_2']))
        if abs(crota2) > 1e-9:
            print("Warning! The coordinate system is skewed!")
        # select how to return pixel scale of RA, Dec
        if as_degrees:
            if verbose:
                print([cdelt1, cdelt2])
            return [cdelt1, cdelt2]
        if verbose:
            print([3600*cdelt1, 3600*cdelt2])
        return [3600*cdelt1, 3600*cdelt2]

    @staticmethod
    def crota2_from_hdr(hdr, as_degrees=False, verbose=False):
        """
        Get rotation (of y-axis) and skew of coordinate system from .fits file header
        (assuming ctype=RA/DEC and cunit=deg)

        Args:
            hdr <dict> - dictionary of the .fits file header

        Kwargs:
            as_degrees <bool> - return result in degrees instead of arcsecs
            verbose <bool> -  verbose mode; print command line statements

        Return:
            crota2 <float>, skew <float> - rotation angle and skew of pixel grid
        """
        if any([(k not in hdr) or (hdr[k] is None) for k in ['CD1_1', 'CD1_2', 'CD2_1', 'CD2_2']]):
            return [None, None]
        if hdr['CD1_1']//abs(hdr['CD1_1']) > 0:
            print("Warning! In the WCS rotation East is not to the left!")
        # calculate the determinante of the distortion matrix
        det = hdr['CD1_1']*hdr['CD2_2']-hdr['CD1_2']*hdr['CD2_1']
        sign = det//abs(det)  # the sign of the determinante
        crot1 = math.atan2(-hdr['CD2_1'], sign*hdr['CD1_1'])
        crot2 = math.atan2(sign*hdr['CD1_2'], hdr['CD2_2'])
        crota2 = 0.5*(crot1+crot2)  # average of the rotation angles of both axes
        skew = abs(crot1-crot2)
        # select how to return pixel scale of rotation and skew
        if as_degrees:
            if verbose:
                print([180./math.pi*crota2, 180./math.pi*skew])
            return [180./math.pi*crota2, 180./math.pi*skew]
        if verbose:
            print([crota2, skew])
        return [crota2, skew]

    @staticmethod
    def refpx_from_hdr(hdr, as_radec=False, verbose=False):
        """
        Get reference pixel from .fits file header (assuming ctype=RA/DEC and cunit=deg)

        Args:
            hdr <dict> - dictionary of the .fits file header

        Kwargs:
            as_radec <bool> - return RA,Dec result in degrees instead of pixels
            verbose <bool> -  verbose mode; print command line statements

        Return:
            refx <int/float>, refy <int/float> - reference pixel coordinates
        """
        if any([(k not in hdr) or (hdr[k] is None) for k in ['CRPIX1', 'CRPIX2']]):
            return [None, None]
        refx, refy = hdr['CRPIX1'], hdr['CRPIX2']
        if as_radec:
            refx, refy = hdr['CRVAL1'], hdr['CRVAL2']
        # some verbosity
        if verbose:
            print([refx, refy])
        return [refx, refy]

    @staticmethod
    def mag_formula_from_hdr(hdr, photzp=None, mag_key='AB magnitude', verbose=False):
        """
        Get the magnitude formula from the .fits file header

        Args:
            hdr <dict> - dictionary of the .fits file header
            flux <float> - flux to be converted to

        Kwargs:
            verbose <bool> -  verbose mode; print command line statements

        Return:
            magnitude <func> - magnitude conversion formula
        """
        if photzp is None:
            photzp = 30
        # search the formula in the comments
        formula_match = [c for c in hdr['COMMENT'] if mag_key in c]
        formula_str = "-2.5 * log10(flux) + PHOTZP"
        if formula_match:
            formula_str = formula_match[0].split("=")[-1].strip()
        # flux formatting
        if not any([c in formula_str for c in ['flux', 'FLUX', 'ADU', 'adu']]):
            formula_str = "-2.5 * log10(flux) + PHOTZP"
        else:
            formula_str = re.sub('(?i)flux', 'flux', formula_str)
            formula_str = re.sub('(?i)adu', 'flux', formula_str)
        # photzp formatting
        if not any([c in formula_str for c in ['photzp', 'PHOTZP', 'zp', 'ZP']]):
            formula_str = "-2.5 * log10(flux) + PHOTZP"
        else:
            formula_str = re.sub('(?i)photzp', '{:.24f}', formula_str).format(photzp)
        # log10 formatting
        formula_str = re.sub('(?i)log', 'np.log', formula_str)
        if verbose:
            print(formula_str)

        def formula(flux, verbose=verbose):
            rslt = eval(formula_str)
            if verbose:
                print(rslt)
            return rslt
        formula.__name__ = "mag_formula"
        return formula

    def plot_f(self, fig, ax=None, as_magnitudes=False, scalebar=True,
               colorbar=False, plain=False,
               verbose=False, cmap='magma', reverse_map=False, **kwargs):
        """
        Plot the image on an axis

        Args:
            fig <matplotlib.figure.Figure object> - figure in which the image is to be plotted

        Kwargs:
            ax <matplotlib.axes.Axes object> - option to control on which axis the image is plotted
            as_magnitudes <bool> - if True, plot data as magnitudes
            scalebar <bool> - if True, add scalebar plot (15% of the image's width)
            colorbar <bool> - if True, add colorbar plot
            verbose <bool> -  verbose mode; print command line statements
            kwargs **<dict> - keywords for the imshow function

        Return:
            fig <matplotlib.figure.Figure object> - figure in which the image was plotted
            ax <matplotlib.axes.Axes object> - axis on which the image was plotted
        """
        # check axes
        if ax is None or len(fig.get_axes()) < 1:
            ax = fig.add_subplot(111)
        # handle bad pixels
        cmap = plt.get_cmap(cmap)
        cmap.set_bad('black', alpha=1)
        cmap.set_under('black', alpha=1)
        # plot data
        if as_magnitudes:
            if self.naxis_plus is not None and len(self.data.shape) > 2:
                img = ax.imshow(np.sum(self.magnitudes, axis=0), cmap=cmap, **kwargs)
            else:
                img = ax.imshow(self.magnitudes, cmap=cmap, **kwargs)
        else:
            if self.naxis_plus is not None and len(self.data.shape) > 2:
                d = np.sum(self.data, axis=0)
                img = ax.imshow(d, cmap=cmap, vmax=np.nanmax(d)*0.1, **kwargs)
            else:
                d = self.data
                img = ax.imshow(d, cmap=cmap, **kwargs)
        if colorbar:  # plot colorbar
            clrbar = fig.colorbar(img)
            clrbar.outline.set_visible(False)
        if scalebar and self.px2arcsec[0] is not None:  # plot scalebar
            from matplotlib import patches
            barpos = (0.05*self.naxis1, 0.025*self.naxis2)
            w, h = 0.15*self.naxis1, 0.01*self.naxis2
            scale = self.px2arcsec[0]*w
            rect = patches.Rectangle(barpos, w, h, facecolor='white', edgecolor=None, alpha=0.85)
            ax.add_patch(rect)
            ax.text(barpos[0]+w/4, barpos[1]+2*h, r"$\mathrm{{{:.1f}''}}$".format(scale),
                    color='white', fontsize=16)
            # flip axes
        ax.set_xlim(left=0, right=self.naxis1)
        ax.set_ylim(bottom=0, top=self.naxis2)
        ax.set_aspect('equal')
        # no axis tick labels
        plt.axis('off')
        # plt.tight_layout()
        # some verbosity
        if verbose:
            print(ax)
        return fig, ax

    def show_f(self, savefig=None, **kwargs):
        """
        Plot the image on a new figure

        Args:
            None

        Kwargs:
            as_magnitudes <bool> - if True, plot data as magnitudes
            scalebar <bool> - if True, add scalebar plot (15% of the image's width)
            colorbar <bool> - if True, add colorbar plot
            savefig <str> - save figure in file string instead of showing it
            verbose <bool> -  verbose mode; print command line statements
            kwargs **<dict> - other keywords for the figure and imshow function

        Return:
            fig <matplotlib.figure.Figure object> - figure with the image's plot
            ax <matplotlib.axes.Axes object> - axis on which the image was plotted
        """
        fsize = kwargs.pop('figsize', (8, 6))
        verbose = kwargs.get('verbose', False)
        if fsize is None:
            fsize = (8, 6)
        # open a new figure
        fig = plt.figure(figsize=fsize)
        ax = fig.add_subplot(111)
        fig, ax = self.plot_f(fig, ax, **kwargs)
        if savefig is not None:
            savename = savefig
            if not any([savefig.endswith(ext) for ext in [".pdf", ".jpg", ".png", ".eps"]]):
                savename = savefig + ".pdf"
            plt.savefig(savename)
        else:
            plt.show()
        # some verbosity
        if verbose:
            print(fig)
        return fig, ax

    def show_fullsky(self, cmap='gnuplot2', savefig=None, bg='black',
                     **kwargs):
        """
        Plot the data as a Mollweide-projected fullsky map
        For compatible .fits files, see: https://pla.esac.esa.int/pla/#maps

        Args:
            None

        Kwargs:
            savefig <str> - save figure in file string instead of showing it
            verbose <bool> -  verbose mode; print command line statements
            kwargs **<dict> - other keywords for the figure and mollview function

        Return:
            fig <matplotlib.figure.Figure object> - figure with the image's plot
            ax <matplotlib.axes.Axes object> - axis on which the image was plotted

        Note:
            - Needs healpy module
            - Only compatible with .fits files with HEALPIX pixel type
        """
        try:  # import healpy
            from healpy import mollview
        except ImportError:
            raise ImportError(
                "For this fits file format you need to install healpy\n"
                + "Use: pip install --user healpy")
        if 'HEALPIX' not in self.hdr:
            pass
        fsize = kwargs.pop('figsize', (8, 6))
        verbose = kwargs.get('verbose', False)
        if fsize is None:
            fsize = (8, 6)
        mu, sigma = 1.2*self.data.mean(), self.data.std()
        # open a new figure
        fig = plt.figure(figsize=fsize)
        cmap = plt.get_cmap(cmap)
        cmap.set_bad(bg, alpha=1)
        cmap.set_under(bg, alpha=1)
        mollview(self.data, fig=fig.number, xsize=fsize[0]*300,
                 min=mu-.3*sigma, max=mu,  # +.3*sigma,
                 title="", cbar=False,
                 cmap=cmap)
        ax = fig.axes[0]
        if savefig is not None:
            savename = savefig
            if not any([savefig.endswith(ext) for ext in [".pdf", ".jpg", ".png", ".eps"]]):
                savename = savefig + ".pdf"
            plt.savefig(savename, facecolor=bg)
        else:
            plt.show()
        # some verbosity
        if verbose:
            print(fig)
        return fig, ax


# MAIN FUNCTION ###############################################################
def main(case, args):
    """
    Main function to use SkyF from command line

    Args:
        case <str> - test case
        args <namespace> - namespace of keyword arguments for all functions

    Kwargs:
        end_message <str> - optional message for printing at the end

    Return:
        None
    """
    sp = SkyF(case, px2arcsec=args.scale, refpx=args.refpx, refval=args.refval,
              photzp=args.photzp, verbose=args.verbose)
    if args.hdr:
        print('HEADER:')
        for k in sp.hdr.keys():
            if isinstance(sp.hdr[k], list):
                print(k+"\n\t"+"\n\t".join(sp.hdr[k]))
            else:
                print(k.ljust(20)+str(sp.hdr[k]))
    if args.show or args.savefig is not None and args.fullsky is None:
        sp.show_f(as_magnitudes=args.mags, figsize=args.figsize, savefig=args.savefig,
                  scalebar=args.scalebar, colorbar=args.colorbar, verbose=args.verbose)
    if args.fullsky:
        sp.show_fullsky(figsize=args.figsize, savefig=args.savefig, verbose=args.verbose)


def parse_arguments():
    """
    Parse command line arguments
    """
    from argparse import ArgumentParser, RawTextHelpFormatter
    parser = ArgumentParser(description=__doc__, formatter_class=RawTextHelpFormatter)
    # main args
    parser.add_argument("case", nargs='?',
                        help="Path input to .fits file for skyf to use",
                        default=os.path.join(os.path.abspath(os.path.dirname(__file__)),
                                             'test', 'W3+3-2.U.12907_13034_7446_7573.fits'))
    parser.add_argument("--scale", metavar=("<dx", "dy>"), nargs=2, type=float,
                        help="Pixel-to-arcsec scale for x (-RA) and y (Dec) direction")
    parser.add_argument("--refpx", metavar=("<x", "y>"), nargs=2, type=float,
                        help="Coordinates of the reference pixel")
    parser.add_argument("--refval", metavar=("<ra", "dec>"), nargs=2, type=float,
                        help="Values of the reference pixel")
    parser.add_argument("--photzp", metavar="<zp>", type=float,
                        help="Magnitude zero-point information")

    # misc options
    parser.add_argument("--hdr", dest="hdr", action="store_true",
                        help="Print the hdr", default=False)

    # plotting args
    parser.add_argument("-s", "--show", dest="show", action="store_true",
                        help="Plot and show the .fits file's data",
                        default=False)
    parser.add_argument("--figsize", dest="figsize", metavar=("<w", "h>"), nargs=2, type=float,
                        help="Size of the figure (is multiplied by the default dpi)")
    parser.add_argument("-m", "--magnitudes", dest="mags", action="store_true",
                        help="Plot the .fits file's data in magnitudes",
                        default=False)
    parser.add_argument("--scalebar", dest="scalebar", action="store_true",
                        help="Plot the scalebar in the figure",
                        default=False)
    parser.add_argument("--colorbar", dest="colorbar", action="store_true",
                        help="Plot the colorbar next to the figure",
                        default=False)
    parser.add_argument("--savefig", dest="savefig", metavar="<output-name>", type=str,
                        help="Save the figure in <output-name> instead of showing it")
    parser.add_argument("--fullsky", dest="fullsky", action="store_true",
                        help="Plot the data as a Mollweide-projected fullsky map",
                        default=False)

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
    testdir = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'test')
    no_input = len(sys.argv) <= 1 and testdir in case
    if no_input:
        parser.print_help()
    elif args.test_mode:
        sys.argv = sys.argv[:1]
        from gleam.test.test_skyf import TestSkyF
        TestSkyF.main()
    else:
        main(case, args)
