#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: phdenzel

Gravitational lenses and all their properties
"""
###############################################################################
# Imports
###############################################################################
from gleam.skycoords import SkyCoords
from gleam.skyf import SkyF
from gleam.lensfinder import LensFinder
from gleam.glscfactory import GLSCFactory
from gleam.utils import colors as glmc

import sys
import os
import copy
import math
import warnings
import numpy as np
from scipy import interpolate
from PIL import Image, ImageDraw

warnings.filterwarnings("ignore", category=RuntimeWarning)


__all__ = ['LensObject']


###############################################################################
class LensObject(SkyF):
    """
    An object representing a gravitational lens
    """
    params = SkyF.params + ['lens', 'srcimgs', 'zl', 'zs', 'tdelay', 'tderr',
                            '_light_model', 'stel_mass', 'lens_map']

    def __init__(self, filepath, lens=None, srcimgs=None, zl=None, zs=None,
                 tdelay=None, tderr=None, _light_model=None, stel_mass=None, lens_map=None,
                 auto=False,
                 glscfactory_options={}, finder_options={},
                 verbose=False, **kwargs):
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
            lens <int,int> - overwrite the lens pixel coordinates
            srcimgs <list(int,int)> - overwrite the source image pixel coordinates
            zl <float> - TODO
            zs <float> - TODO
            tdelay <list(float)> - TODO
            tderr <list(float) - TODO
            _light_model <dict/gleam.model object> - a dict or direct input of the light model
            stel_mass <float> - TODO
            lens_map <np.ndarray> - TODO
            auto <bool> - use LensFinder for automatic image recognition (can be unreliable)
            verbose <bool> - verbose mode; print command line statements
            glscfactory_options <dict> - options for the GLSCFactory encompassing the following:
                parameter <dict> - various parameters like redshifts, time delays, etc.;
                                   also contains parameters for the GLASS config generation
                text_file <str> - path to .txt file; shortcuts automatically resolved
                text <list(str)> - alternative to text_file; direct text input
                filter_ <bool> - apply filter from GLSCFactory.key_filter to text information
                reorder <str> - reorder the images relative to ABCD ordered bottom-up
                output <str> - output name of the .gls file
                name <str> - object name in the .gls file (extracted from output by default)
            finder_options <dict> - options for the LensFinder encompassing the following:
                n <int> - number of peak candidates allowed
                min_q <float> - a percentage quotient for the min. peak separation
                sigma <int(,int)> - lower/upper sigma for signal-to-noise estimate
                centroid <int> - use COM positions around a pixel slice of size of centroid
                                 around peak center if centroid > 1

        Return:
            <LensObject object> - standard initializer
        """
        super(LensObject, self).__init__(filepath, **kwargs)
        # a few additional properties like redshifts and time delays
        self.zl = None
        self.zs = None
        self.tdelay = None
        self.tderr = None
        self._light_model = {}
        self.stel_mass = None
        self.lens_map = None
        self._lens = None  # lens position (assume to be in the center for finder)
        self.srcimgs = []  # source image positions
        # GLASS config factory for parsing text and config files
        self.glscfactory = GLSCFactory(lens_object=self, **glscfactory_options)
        self.glscfactory.sync_lens_params(verbose=False)  # load parameters from text file or dict
        # LensFinder for automatic search of lens and source image positions
        self._lens = self.center  # needs lens reference for dummy shifts
        self.finder = LensFinder(self, **finder_options)
        if auto:
            if self.finder.lens_candidate is not None:
                self.lens = self.finder.lens_candidate
            for p in self.finder.source_candidates:
                self.srcimgs.append(p)
        # set lens parameters manually
        self._lens = None
        if lens is not None:
            self.lens = lens
        if srcimgs is not None:
            self.srcimgs = srcimgs
        if zl is not None:
            self.zl = zl
        if zs is not None:
            self.zs = zs
        if tdelay is not None:
            self.tdelay = tdelay
        if tderr is not None:
            self.tderr = tderr
        if _light_model is not None:
            self.light_model = _light_model
        if stel_mass is not None:
            self.stel_mass = stel_mass

        # some verbosity
        if verbose:
            print(self.__v__)

    def __eq__(self, other):
        if isinstance(other, LensObject):
            return \
                super(LensObject, self).__eq__(other) \
                and self.lens == other.lens \
                and self.srcimgs == other.srcimgs \
                and self.zl == other.zl \
                and self.zs == other.zs \
                and self.tdelay == other.tdelay \
                and self.tderr == other.tderr
        else:
            NotImplemented

    def __str__(self):
        return "LensObject({}@[{:.4f}, {:.4f}])".format(self.band, *self.center)

    @property
    def tests(self):
        """
        A list of attributes being tested when calling __v__

        Args/Kwargs:
            None

        Return:
            tests <list(str)> - a list of test variable strings
        """
        return super(LensObject, self).tests \
            + ['lens', 'srcimgs', 'zl', 'zs', 'tdelay', 'tderr', 'light_model', 'stel_mass',
               'glscfactory', 'finder']

    @property
    def lens(self):
        """
        Lens position in the .fits file (as SkyCoords object)

        Args/Kwargs:
            None

        Return:
            lens <SkyCoords object> - lens position
        """
        return self._lens

    @lens.setter
    def lens(self, lens):
        """
        Lens position setter in the .fits file (in pixels)

        Args:
            lens <SkyCoords object | int/float,int/float> - lens position (in pixels)

        Kwargs/Return:
            None
        """
        if isinstance(lens, SkyCoords):
            self._lens = lens
        elif isinstance(lens, (tuple, list)):
            self._lens = SkyCoords.from_pixels(*lens, **self.center.coordkw)

    @property
    def srcimgs_xy(self):
        """
        Source positions in the .fits file (in pixels)

        Args/Kwargs:
            None

        Return:
            sources <list> - list of source positions in pixels
        """
        return [s.xy for s in self.srcimgs]

    @srcimgs_xy.setter
    def srcimgs_xy(self, srcs):
        """
        Source positions setter in the .fits file (in pixels)

        Args:
            srcs <list> - list of tuples of source image positions (in pixels)

        Kwargs/Return:
            None
        """
        self.srcimgs = [SkyCoords.from_pixels(*s, **self.center.coordkw) for s in srcs]

    def p2skycoords(self, position, unit='arcsec', relative=True, verbose=False):
        """
        Convert a position into skycoords positions with the skyfs reference pixel information

        Args:
            position <int/float,int/float> - position (relative to lens position)

        Kwargs:
            unit <str> - unit of the position input (arcsec, degree, pixel)
            relative <bool> - position relative to lens; if False position is assumed absolute
            verbose <bool> -  verbose mode; print command line statements

        Return:
            skyc <SkyCoords object> - the position converted into a SkyCoords object
        """
        if unit in ['pixel', 'pixels']:
            refp = [0, 0]
            refv = [0., 0.]
            if not relative and self.lens is not None:
                refp = self.center.xy  # reference_pixel
                refv = self.center.radec  # reference_value
            elif not relative and self.center is not None:
                refp = self.center.xy  # reference_pixel
                refv = self.center.radec  # reference_value
            p = SkyCoords.pixels2deg(position, px2arcsec_scale=self.center.px2arcsec_scale,
                                     reference_pixel=refp, reference_value=refv)
            if relative:
                p[0] = -p[0]
        elif unit in ['arcsec', 'arcsecs']:
            p = SkyCoords.arcsec2deg(*position)
        elif unit in ['degree', 'degrees']:
            p = position
        else:  # unit = 'degree' by default
            p = position
        if relative:
            if self.lens is not None:
                skyc = self.lens.shift(p, right_increase=True)
            else:
                skyc = self.center.shift(p, right_increase=True)
        else:
            skyc = SkyCoords(*p, **self.center.coordkw)
        if verbose:
            print(skyc.__v__)
        return skyc

    def add_srcimg(self, position, unit='arcsec', relative=True, verbose=False):
        """
        Add a source image position to the list of source images

        Args:
            position <int/float,int/float> - source image position (relative to lens position)

        Kwargs:
            unit <str> - unit of the position input (arcsec, degree, pixel)
            relative <bool> - position relative to lens, if False position is absolute
            verbose <bool> -  verbose mode; print command line statements

        Return:
            None
        """
        srcpos = self.p2skycoords(position, unit=unit, relative=relative)
        self.srcimgs.append(srcpos)
        if verbose:
            print(srcpos.__v__)
        return srcpos

    def src_shifts(self, **kwargs):
        """
        Shifts relative to lens position of all source images

        Args:
            None

        Kwargs:
            unit <str> - unit of the shift (arcsec, degree, pixel)

        Return:
            shifts <list> - list of shifts from lens to source images
        """
        verbose = kwargs.pop('verbose', False)
        if self.lens is not None:
            shifts = [srcpos.get_shift_to(self.lens, **kwargs) for srcpos in self.srcimgs]
        else:
            shifts = [srcpos.get_shift_to(self.center, **kwargs) for srcpos in self.srcimgs]
        if verbose:
            print(shifts)
        return shifts

    @property
    def light_model(self):
        """
        The light model to the .fits data

        Args/Kwargs:
            None

        Return:
            model <gleam.model object> - the light profile model fitting the data
        """
        if not hasattr(self, '_light_model'):
            self._light_model = {}
        models = {}
        for k in self._light_model:
            module_name = 'gleam.model.{}'.format(k)
            cls_name = k.capitalize()
            module = __import__(module_name, fromlist=[k])
            cls = getattr(module, cls_name)
            models[k] = cls(**self._light_model[k])
        return models

    @light_model.setter
    def light_model(self, model):
        """
        Setter of the light models

        Args:
            model <dict/gleam.model object> - dictionary of direcy input of the model

        Kwargs/Return:
            None
        """
        if not hasattr(self, '_light_model'):
            self._light_model = {}
        if isinstance(model, dict):
            self._light_model.update(model)
        else:
            self._light_model[model.__modelname__] = {
                k: v for k, v in zip(model.parameter_keys, model.model_parameters)}

    @property
    def stel_map(self):
        """
        The stellar map from a light model and stellar mass estimate

        Args/Kwargs:
            None

        Return:
            stel_map <np.ndarray> - the stellar map derived from light model
        """
        if self.stel_mass is None or len(self.light_model) < 1:
            return
        if 'sersic' in self.light_model:
            light_model = self.light_model['sersic']
        else:
            model_dict = self.light_model
            model_name = list(model_dict.keys())[0]
            light_model = model_dict[model_name]
        light_model.Ny, light_model.Nx = self.data.shape
        light_model.y, light_model.x = light_model.Ny//2, light_model.Nx//2
        light_model.calc_map()
        if not self.roi._buffer['circle']:
            self.roi.select['circle']((light_model.x, light_model.y), 10)
        else:
            r = self.roi._buffer['circle'][0].radius
            self.roi.select['circle']((light_model.x, light_model.y), r)
        lens_mask = self.roi._masks['circle'][-1]
        mask = lens_mask
        light_norm = light_model.normalize(light_model.map2D, mask=mask)
        return light_norm * self.stel_mass / (self.px2arcsec[0]*self.px2arcsec[1])

    def image_f(self, draw_lens=False, draw_srcimgs=False, **kwargs):
        """
        An 8-bit PIL.Image of the .fits data

        Args:
            None

        Kwargs:
            cmap <str> - a cmap string from matplotlib.colors.Colormap
            draw_lens <bool> - draw the lens position on top of data
            draw_srcimgs <bool> - draw the source image positions on top of data
            draw_roi <bool> - draw the ROI objects on top of data

        Return:
            f_image <PIL.Image object> - a colorized image object
        """
        if self.data is not None:
            img = super(LensObject, self).image_f(**kwargs)
            img = img.transpose(Image.FLIP_TOP_BOTTOM)
            if draw_lens:
                img = self.draw_lens(img)
            if draw_srcimgs:
                img = self.draw_srcimgs(img)
            img = img.transpose(Image.FLIP_TOP_BOTTOM)
            return img
        return self.data

    def draw_lens(self, img, point_size=1, fill=None, outline=None, verbose=False):
        """
        Draw the lens as points on the input image

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
        if self.lens is None:
            return img
        if fill is None:
            fill = glmc.blue
        if outline is None:
            outline = glmc.black
        draw = ImageDraw.Draw(img)
        p = (self.lens.x-point_size, self.lens.y-point_size,
             self.lens.x+point_size, self.lens.y+point_size)
        draw.ellipse(p, fill=fill, outline=outline)
        del draw
        return img

    def draw_srcimgs(self, img, point_size=1, fill=None, outline=None, verbose=False):
        """
        Draw the lens as points on the input image

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
        if len(self.srcimgs) < 1:
            return img
        if fill is None:
            fill = glmc.pink
        if outline is None:
            outline = glmc.black
        draw = ImageDraw.Draw(img)
        pts = [(p.x-point_size, p.y-point_size, p.x+point_size, p.y+point_size)
               for p in self.srcimgs]
        for p in pts:
            draw.ellipse(p, fill=fill, outline=outline)
        del draw
        return img

    def plot_f(self, fig, ax=None, as_magnitudes=False, lens=False, source_images=False,
               label_images=False, sequence=None, scalebar=True, colorbar=False,
               plain=False, verbose=False, cmap='magma', **kwargs):
        """
        Plot the image on an axis

        Args:
            fig <matplotlib.figure.Figure object> - figure in which the image is to be plotted

        Kwargs:
            ax <matplotlib.axes.Axes object> - option to control on which axis the image is plotted
            as_magnitudes <bool> - if True, plot data as magnitudes
            lens <bool> - indicate the lens position as scatter point
            source_images <bool> - indicate the source image positions as scatter points
            label_images <bool> - label the source image positions in sequence
            sequence <str> - sequence of labels for image_labels
            scalebar <bool> - if True, add scalebar plot (15% of the image's width)
            colorbar <bool> - if True, add colorbar plot
            verbose <bool> -  verbose mode; print command line statements
            kwargs **<dict> - keywords for the imshow function

        Return:
            fig <matplotlib.figure.Figure object> - figure in which the image was plotted
            ax <matplotlib.axes.Axes object> - axis on which the image was plotted
        """
        if sequence is None:
            sequence = 'ABCDE'
        fig, ax = super(LensObject, self).plot_f(
            fig, ax, as_magnitudes=as_magnitudes, plain=plain, verbose=False,
            scalebar=scalebar, colorbar=colorbar, cmap=cmap, **kwargs)
        if lens and self.lens is not None:
            ax.scatter(*self.lens.xy, marker='o', s=3**2*math.pi, c=glmc.purpleblue)
        if source_images and self.srcimgs:
            for i, c in enumerate(self.srcimgs):
                ax.scatter(*c.xy, marker='o', s=3**2*math.pi, c=glmc.green)
                if label_images:
                    ax.text(c.x+self.naxis1*0.02, c.y-self.naxis2*0.04,
                            sequence[i],
                            color='white', fontsize=16)
        # some verbosity
        if verbose:
            print(ax)
        return fig, ax


# MAIN FUNCTION ###############################################################
def main(case, args):
    """
    Main function to use LensObject from command line

    Args:
        case <str> - test case
        args <namespace> - namespace of keyword arguments for all functions

    Kwargs:
        end_message <str> - optional message for printing at the end

    Return:
        None
    """
    sp = LensObject(case, px2arcsec=args.scale, refpx=args.refpx, refval=args.refval,
                    lens=args.lens, photzp=args.photzp,
                    auto=args.auto, verbose=args.verbose,
                    finder_options=dict(n=args.n, sigma=args.sigma, min_q=args.min_q),
                    glscfactory_options=dict(text_file=args.text_file, filter_=args.filter_, reorder=args.reorder))
    if args.show or args.show_lens or args.show_sources or args.savefig is not None:
        sp.show_f(as_magnitudes=args.mags, figsize=args.figsize,
                  lens=args.show_lens, source_images=args.show_sources,
                  label_images=args.labels, sequence=args.reorder,
                  scalebar=args.scalebar, colorbar=args.colorbar,
                  savefig=args.savefig, verbose=args.verbose)


def parse_arguments():
    """
    Parse command line arguments
    """
    from argparse import ArgumentParser, RawTextHelpFormatter
    parser = ArgumentParser(description=__doc__, formatter_class=RawTextHelpFormatter)

    # main args
    parser.add_argument("case", nargs='?',
                        help="Path input to .fits file for skyf to use",
                        default=os.path.abspath(os.path.dirname(__file__)) \
                        + '/test/W3+3-2.U.12907_13034_7446_7573.fits')
    parser.add_argument("-a", "--auto", dest="auto", action="store_true",
                        help="Use automatic image recognition (can be unreliable)",
                        default=False)
    parser.add_argument("--scale", metavar=("<dx", "dy>"), nargs=2, type=float,
                        help="Pixel-to-arcsec scale for x (-RA) and y (Dec) direction")
    parser.add_argument("--refpx", metavar=("<x", "y>"), nargs=2, type=float,
                        help="Coordinates of the reference pixel")
    parser.add_argument("--refval", metavar=("<ra", "dec>"), nargs=2, type=float,
                        help="Values of the reference pixel")
    parser.add_argument("--photzp", metavar="<zp>", type=float,
                        help="Magnitude zero-point information")
    parser.add_argument("-l", "--lens", metavar=("<x", "y>"), nargs=2, type=float,
                        help="Lens pixel position")

    # plotting args
    parser.add_argument("-s", "--show", dest="show", action="store_true",
                        help="Plot and show the .fits file's data",
                        default=False)
    parser.add_argument("--figsize", dest="figsize", metavar=("<w", "h>"), nargs=2, type=float,
                        help="Size of the figure (is multiplied by the default dpi)")
    parser.add_argument("-m", "--magnitudes", dest="mags", action="store_true",
                        help="Plot the .fits file's data in magnitudes",
                        default=False)
    parser.add_argument("--show-lens", dest="show_lens", action="store_true",
                        help="Plot the lens position",
                        default=False)
    parser.add_argument("--labels", dest="labels", action="store_true",
                        help="Label the source image positions in sequence",
                        default=False)
    parser.add_argument("--show-srcs", dest="show_sources", action="store_true",
                        help="Plot the source image positions",
                        default=False)
    parser.add_argument("--scalebar", dest="scalebar", action="store_true",
                        help="Plot the scalebar in the figure",
                        default=False)
    parser.add_argument("--colorbar", dest="colorbar", action="store_true",
                        help="Plot the colorbar next to the figure",
                        default=False)
    parser.add_argument("--savefig", dest="savefig", metavar="<output-name>", type=str,
                        help="Save the figure in <output-name> instead of showing it")

    # lensfinder options
    parser.add_argument("-n", "--npeaks", dest="n", metavar="<N>", type=int,
                        help="Number of peaks the automatic image recognition is supposed to find",
                        default=5)
    parser.add_argument("-q", "--min-q", dest="min_q", metavar="<min_q>", type=float,
                        help="A percentage quotient for the minimal peak separation",
                        default=0.1)
    parser.add_argument("--sigma", metavar=("<sx", "sy>"), dest="sigma", nargs=2, type=float,
                        help="Lower/upper sigma factor for signal-to-noise estimation",
                        default=(2, 2))

    # gls config factory args
    parser.add_argument("--text-file", dest="text_file", metavar="<path-to-file>", type=str,
                        help="Path to text file with additional info for glass config generation")
    parser.add_argument("--filter", dest="filter_", action="store_true",
                        help="Use GLSCFactory's additional filter for extracted text info",
                        default=False)
    parser.add_argument("--reorder", dest="reorder", metavar="<abcd-order>", type=str.upper,
                        help="Reorder the image positions relative to ABCD ordered bottom-up")

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
if __name__ == '__main__':
    parser, case, args = parse_arguments()
    no_input = len(sys.argv) <= 1 and os.path.abspath(os.path.dirname(__file__))+'/test/' in case
    if no_input:
        parser.print_help()
    elif args.test_mode:
        sys.argv = sys.argv[:1]
        from gleam.test.test_lensobject import TestLensObject
        TestLensObject.main()
    else:
        main(case, args)
