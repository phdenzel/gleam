#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: phdenzel

Gravitational lenses and all their properties
"""
###############################################################################
# Imports
###############################################################################
import __init__
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

warnings.filterwarnings("ignore", category=RuntimeWarning)


__all__ = ['LensObject']


###############################################################################
class LensObject(SkyF):
    """
    An object representing a gravitational lens
    """
    def __init__(self, filepath, lens=None, srcimgs=None, auto=False,
                 n=5, min_q=0.1, sigma=(4, 4),
                 data=None, text_file=None, text=None, filter_=True,
                 output=None, name=None, reorder=None, verbose=False, **kwargs):
        """
        Initialize parsing of a fits file with a file name

        Args:
            filepath <str> - path to .fits file (shortcuts are automatically resolved)

        Kwargs:
            px2arcsec <float,float> - overwrite the pixel scale in the .fits header (in arcsecs)
            refpx <int,int> - overwrite reference pixel coordinates in .fits header (in pixels)
            refval <float,float> - overwrite reference pixel values in .fits header (in degrees)
            lens <int,int> - overwrite the lens pixel coordinates
            srcimgs <list(int,int)> - overwrite the source image pixel coordinates
            photzp <float> - overwrite photometric zero-point information
            auto <bool> - (finder) use automatic image recognition (can be unreliable)
            n <int> - (finder) number of peak candidates allowed
            min_q <float> - (finder) a percentage quotient for the min. peak separation
            sigma <int(,int)> - (finder) lower/upper sigma for signal-to-noise estimate
            data <dict> - (glscg) data for the GLASS config generation
            text_file <str> - (glscg) path to .txt file; shortcuts automatically resolved
            text <list(str)> - (glscg) alternative to text_file; direct text input
            filter_ <bool> - (glscg) apply filter from GLSCFactory.key_filter to text information
            reorder <str> - (glscg) reorder the image positions relative to ABCD ordered bottom-up
            output <str> - (glscg) output name of the .gls file
            name <str> - (glscg) object name in the .gls file (extracted from output by default)

        Return:
            <LensObject object> - standard initializer
        """
        super(LensObject, self).__init__(filepath, **kwargs)
        # a few additional properties like redshifts and time delays
        self.zl = None
        self.zs = None
        self.tdelay = None
        self.tderr = None
        # assume lens is in the center of the .fits file afterwards if auto is False
        self._lens = None
        self.srcimgs = []  # source image positions

        # initialize glass config factory
        self.glsc_factory = GLSCFactory(lens_object=self, data=data, verbose=False,
                                        text_file=text_file, text=text, filter_=filter_,
                                        output=output, name=name, reorder=reorder)
        # load lens parameters from text file
        self.glsc_factory.sync_lens_params(verbose=False)

        # initialize automatic lens finder
        self._lens = self.center
        self.finder = LensFinder(self, n=n, min_q=min_q, sigma=sigma)
        if auto:
            if self.finder.lens_candidate is not None:
                self._lens = self.finder.lens_candidate
            for p in self.finder.source_candidates:
                self.srcimgs.append(p)

        # set lens and/or source images manually
        if lens is not None:
            self.lens = lens
        if srcimgs is not None:
            self.srcimgs_xy = srcimgs

        # some verbosity
        if verbose:
            print(self.__v__)

    def __copy__(self):
        args = (self.filepath,)
        kwargs = {
            'px2arcsec': self.px2arcsec,
            'refpx': self.refpx,
            'refval': self.refval,
            'photzp': self.photzp,
            'lens': self.lens.xy,
            'srcimgs': self.srcimgs_xy,
            'data': {
                'zl': self.zl,
                'zs': self.zs,
                'tdelay': self.tdelay,
                'tderr': self.tderr
            }
        }
        return LensObject(*args, **kwargs)

    def __deepcopy__(self, memo):
        args = (copy.deepcopy(self.filepath, memo),)
        kwargs = {
            'px2arcsec': copy.deepcopy(self.px2arcsec, memo),
            'refpx': copy.deepcopy(self.refpx, memo),
            'refval': copy.deepcopy(self.refval, memo),
            'photzp': copy.deepcopy(self.photzp, memo),
            'lens': copy.deepcopy(self.lens.xy, memo),
            'srcimgs': copy.deepcopy(self.srcimgs_xy, memo),
            'data': {
                'zl': copy.deepcopy(self.zl, memo),
                'zs': copy.deepcopy(self.zs, memo),
                'tdelay': copy.deepcopy(self.tdelay, memo),
                'tderr': copy.deepcopy(self.tderr, memo)
            }
        }
        return LensObject(*args, **kwargs)

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
            + ['zl', 'zs', 'tdelay', 'tderr', 'lens', 'srcimgs', 'glsc_factory', 'finder']

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
            lens <int/float,int/float> - lens position (in pixels)

        Kwargs/Return:
            None
        """
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
                refp = self.lens.xy  # reference_pixel
                refv = self.lens.radec  # reference_value
            p = SkyCoords.pixels2deg(position, px2arcsec_scale=self.center.px2arcsec_scale,
                                     reference_pixel=refp, reference_value=refv)
            if relative:
                p[0] = -p[0]
        elif unit in ['arcsec', 'arcsecs']:
            p = SkyCoords.arcsec2deg(*position)
        elif unit in ['degree', 'degrees']:
            p = position
        else:
            unit = 'degree'
            p = position
        if relative:
            skyc = self.lens.shift(p, right_increase=True)
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
        shifts = [srcpos.get_shift_to(self.lens, **kwargs) for srcpos in self.srcimgs]
        if verbose:
            print(shifts)
        return shifts

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
                ax.scatter(*c.xy, marker='o', s=3**2*math.pi, c=glmc.neongreen)
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
                    auto=args.auto, n=args.n, sigma=args.sigma, min_q=args.min_q,
                    output=args.config_single or args.config_multi, name=args.name,
                    text_file=args.text_file, filter_=args.filter_, reorder=args.reorder,
                    verbose=args.verbose)
    if args.show or args.show_lens or args.show_sources or args.savefig is not None:
        sp.show_f(as_magnitudes=args.mags, figsize=args.figsize,
                  lens=args.show_lens, source_images=args.show_sources,
                  label_images=args.labels, sequence=args.reorder,
                  scalebar=args.scalebar, colorbar=args.colorbar,
                  savefig=args.savefig, verbose=args.verbose)
    if args.config_single is not None:
        sp.glsc_factory.write(verbose=args.verbose)
    elif args.config_multi is not None:
        sp.glsc_factory.append(last=args.finish_config, verbose=args.verbose)


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
    parser.add_argument("-n", "--npeaks", dest="n", metavar="<N>", type=int,
                        help="Number of peaks the automatic image recognition is supposed to find",
                        default=5)
    parser.add_argument("-q", "--min-q", dest="min_q", metavar="<min_q>", type=float,
                        help="A percentage quotient for the minimal peak separation",
                        default=0.1)
    parser.add_argument("--sigma", metavar=("<sx", "sy>"), dest="sigma", nargs=2, type=float,
                        help="Lower/upper sigma factor for signal-to-noise estimation",
                        default=(2, 2))
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

    # gls config factory args
    parser.add_argument("--single-config", dest="config_single", metavar="<output-name>", type=str,
                        help="Generate a glass config file")
    parser.add_argument("--multi-config", dest="config_multi", metavar="<output-name>", type=str,
                        help="Generate a glass config file in append-mode")
    parser.add_argument("--name", dest="name", metavar="<name>", type=str,
                        help="Name of the lens object in the glass config file")
    parser.add_argument("--finish", dest="finish_config", action="store_true",
                        help="Append and complete the config file with these configs"
                        + " in multi-config mode",
                        default=False)
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
