#!/usr/bin/env python
"""
@author: phdenzel

MultiLens sees same lens in different bands
"""
###############################################################################
# Imports
###############################################################################
from gleam.skyf import SkyF
from gleam.skypatch import SkyPatch
from gleam.lensobject import LensObject
from gleam.utils import colors as glmc

import sys
import os
import copy
import math
import warnings
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=RuntimeWarning)


__all__ = ['MultiLens']


###############################################################################
class MultiLens(SkyPatch):
    """
    Framework for a set of sky patches (.fits files)
    """
    params = ['lens_objects']

    def __init__(self, files, auto=None, glscfactory_options=None, finder_options=None,
                 verbose=False, **kwargs):
        """
        Initialize parsing of a fits file with a directory name

        Args:
            files <str/list(str)> - list of .fits files or directory string containing the files

        Kwargs:
            data <list(list/np.ndarray)> - add the data of the .fits file directly
            hdr <list(dict)> - add the header of the .fits file directly
            px2arcsec <list(float,float)> - overwrite the pixel scale in the .fits header (in arcs)
            refpx <list(int,int)> - overwrite reference pixel coordinates in .fits header (in px)
            refval <list(float,float)> - overwrite reference pixel values in .fits header (in deg)
            photzp <list(float)> - overwrite photometric zero-point information

            lens <list(int,int)> - overwrite the lens pixel coordinates
            srcimgs <list(int,int)> - overwrite the source image pixel coordinates
            auto <list(bool)> - (finder) use automatic image recognition (can be unreliable)
            verbose <bool> - verbose mode; print command line statements

            glscfactory_options <list(dict)> - options for the GLSCFactory:
                parameter <dict> - various parameters like redshifts, time delays, etc.;
                                   also contains parameters for the GLASS config generation
                text_file <str> - path to .txt file; shortcuts automatically resolved
                text <list(str)> - alternative to text_file; direct text input
                filter_ <bool> - apply filter from GLSCFactory.key_filter to text information
                reorder <str> - reorder the images relative to ABCD ordered bottom-up
                output <str> - output name of the .gls file
                name <str> - object name in the .gls file (extracted from output by default)
            finder_options <list(dict)> - options for the LensFinder:
                n <int> - number of peak candidates allowed
                min_q <float> - a percentage quotient for the min. peak separation
                sigma <int(,int)> - lower/upper sigma for signal-to-noise estimate
                centroid <int> - use COM positions around a pixel slice of size of centroid
                                 around peak center if centroid > 1


        Return:
            <MultiLens object> - standard initializer
        """
        super(MultiLens, self).__init__(files, verbose=False,
                                        **{k: kwargs[k] for k in SkyF.params if k in kwargs})
        # keyword defaults
        for k in LensObject.params:
            kwargs.setdefault(k, [None]*self.N)
        if auto is None:
            auto = [False]*self.N
        if glscfactory_options is None:
            glscfactory_options = [{}]*self.N
        if finder_options is None:
            finder_options = [{}]*self.N
        # handle collective keyword inputs
        for k in LensObject.params:
            if not isinstance(kwargs[k], list):
                kwargs[k] = [kwargs[k]]*self.N
        if isinstance(auto, bool):
            auto = [auto]*self.N
        if isinstance(glscfactory_options, dict):
            glscfactory_options = [glscfactory_options]*self.N
        if isinstance(finder_options, dict):
            finder_options = [finder_options]*self.N
        # lensobject instances for all files
        self.lens_objects = [None]*self.N
        for i, f in enumerate(self.filepaths):
            self.lens_objects[i] = LensObject(
                f, glscfactory_options=glscfactory_options[i], finder_options=finder_options[i],
                **{k: kwargs[k][i] for k in kwargs})

        #TODO: unify different lens objects to have uniform information

        if verbose:
            print(self.__v__)

    def __getitem__(self, band):
        """
        Get item from lens_objects either by index or by band string
        """
        if isinstance(band, str):
            try:
                return self.lens_objects[self.bands.index(band)]
            except ValueError:
                raise ValueError('Band not found in SkyPatch object')
        elif isinstance(band, int):
            try:
                return self.lens_objects[band]
            except ValueError:
                raise ValueError("Band index not found in SkyPatch object")
        else:
            raise IndexError

    def __str__(self):
        return "MultiLens({})".format(", ".join(self.bands))

    @property
    def tests(self):
        """
        A list of attributes being tested when calling __v__

        Args/Kwargs:
            None

        Return:
            tests <list(str)> - a list of test variable strings
        """
        return super(MultiLens, self).tests + ['lens_objects', 'lens', 'srcimgs', 'srcimgs_xy']

    @property
    def fs(self):
        """
        Override fs with lens_objects super-instances (if possible)

        Args/Kwargs:
            None

        Return:
            fs <super: LensObject object> - the from lens_objects derived instances
        """
        if hasattr(self, 'lens_objects'):
            return [super(LensObject, l) for l in self.lens_objects]
        else:
            return self._fs

    @fs.setter
    def fs(self, fs):
        """
        Setter for overriden fs

        Args:
            fs <list(SkyF objects)> - dummy SkyF objects to be overridden by lens_objects

        Kwargs/Return:
            None
        """
        self._fs = fs

    @property
    def lens(self):
        """
        Lens positions of all bands

        Args/Kwargs:
            None

        Return:
            lens <list(SkyCoords object)> - list of all lens positions
        """
        return [p.lens for p in self.lens_objects]

    @lens.setter
    def lens(self, position):
        """
        Setter for lens position of all bands

        Args:
            position <int/float,int/float> - pixel position of the lens

        Kwargs/Return:
            None
        """
        for p in self.lens_objects:
            p.lens = position

    @property
    def srcimgs(self):
        """
        Source image positions

        Args/Kwargs:
            None

        Return:
            srcimgs <list(SkyCoords object)> - list of all source image positions
        """
        return [p.srcimgs[:] for p in self.lens_objects]

    @property
    def srcimgs_xy(self):
        """
        Source image positions (in pixels)

        Args/Kwargs:
            None

        Return:
            srcimgs <list(SkyCoords object)> - list of all source image positions
        """
        return [p.srcimgs_xy[:] for p in self.lens_objects]

    @srcimgs_xy.setter
    def srcimgs_xy(self, srcs):
        """
        Source positions setter in the .fits file (in pixels)

        Args:
            srcs <list> - list of tuples of source image positions (in pixels)

        Kwargs/Return:
            None
        """
        for p in self.lens_objects:
            p.srcimgs_xy = srcs

    def add_srcimg(self, position, **kwargs):
        """
        Add a source image position to all bands

        Args:
            position <int/float,int/float> - source image position relative to lens position

        Kwargs:
            unit <str> - unit of the position input (arcsec, degree, pixel)
            relative <bool> - position relative to lens, if False position is absolute
            verbose <bool> -  verbose mode; print command line statements

        Return:
            None
        """
        return [p.add_srcimg(position, **kwargs) for p in self.lens_objects]

    def add_to_patch(self, filepath, index=None, verbose=False, **kwargs):
        """
        Add a file to the patch after initialization

        Args:
            filepath <str> - path to file which is to be added to the patch

        Kwargs:
            index <int> -  list index at which the file is inserted; default -1
            data <list/np.ndarray> - add the data of the .fits file directly
            hdr <dict> - add the header of the .fits file directly
            px2arcsec <float,float> - overwrite the pixel scale in the .fits header (in arcsecs)
            refpx <int,int> - overwrite reference pixel coordinates in .fits header (in pixels)
            refval <float,float> - overwrite reference pixel values in .fits header (in degrees)
            photzp <float> - overwrite photometric zero-point information
            lens <int,int> - overwrite the lens pixel coordinates
            srcimgs <list(int,int)> - overwrite the source image pixel coordinates
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
            filepath <str> - validated filepath which was added to the patch
        """
        filepath = super(MultiLens, self).add_to_patch(
            filepath, index=index, verbose=False,
            **{k: kwargs[k] for k in SkyF.params if k in kwargs})
        if index is None or index == -1:
            self.lens_objects.append(LensObject(filepath, **kwargs))
        else:
            self.lens_objects.insert(index, LensObject(filepath, **kwargs))
        if verbose:
            print(self.__v__)
        return filepath

    def remove_from_patch(self, index, verbose=False):
        """
        Remove a file from the patch after initialization

        Args:
            index <int> - index to be removed

        Kwargs:
            verbose <bool> - verbose mode; print command line statements

        Return:
            rmd_f, rmd_obj <str,LensObject object> - the removed filepath and corresponding object
        """
        rmd_f, rmd_skyfobj = super(MultiLens, self).remove_from_patch(index, verbose=False)
        rmd_obj = self.lens_objects.pop(index)
        if verbose:
            print(self.__v__)
        return rmd_f, rmd_obj

    def reorder_patch(self, new_sequence, verbose=False):
        """
        Rearrange order of the patch after initialization

        Args:
            new_sequence <list(int)> - new order according to current indices

        Kwargs:
            verbose <bool> - verbose mode; print command line statements

        Return:
            None
        """
        super(MultiLens, self).reorder_patch(new_sequence, verbose=False)
        self.lens_objects = [self.lens_objects[i] for i in new_sequence]
        if verbose:
            print(self.__v__)

    def plot_composite(self, fig, ax=None, method='standard', lens=False, source_images=False,
                       colorbar=False, scalebar=True, verbose=False, **kwargs):
        """
        Plot a composite image using the i, r, and g band

        Args:
            fig <matplotlib.figure.Figure object> - figure in which the image is to be plotted

        Kwargs:
            ax <matplotlib.axes.Axes object> - option to control on which axis the image is plotted
            method <str> - method description; chooses the set of weights for the stacking
            lens <bool> - indicate the lens position as scatter point
            source_images <bool> - indicate the source image positions as scatter points
            scalebar <bool> - if True, add scalebar plot (15% of the image's width)
            verbose <bool> -  verbose mode; print command line statements
            kwargs **<dict> - keywords for the imshow function

        Return:
            fig <matplotlib.figure.Figure object> - figure in which the image was plotted
            ax <matplotlib.axes.Axes object> - axis on which the image was plotted
        """
        fig, ax = super(MultiLens, self).plot_composite(fig, ax, method=method,
                                                        colorbar=colorbar, scalebar=scalebar,
                                                        **kwargs)
        if 'i' not in self.bands and 'r' not in self.bands and 'g' not in self.bands:
            return fig, ax
        if lens:
            for p in [self['i'], self['r'], self['g']]:
                ax.scatter(*p.lens.xy, marker='o', s=5**2*math.pi, c=glmc.purpleblue)
        if source_images:
            for p in [self['i'], self['r'], self['g']]:
                for c in p.srcimgs:
                    ax.scatter(*c.xy, marker='o', s=5**2*math.pi, c=glmc.neongreen)
        # some verbosity
        if verbose:
            print(ax)
        return fig, ax


# MAIN FUNCTION ###############################################################
def main(case, args):
    ml = MultiLens(case, px2arcsec=args.scale, refpx=args.refpx, refval=args.refval,
                   lens=args.lens, photzp=args.photzp,
                   auto=args.auto, n=args.n, sigma=args.sigma, min_q=args.min_q,
                   output=args.gen_config, text_file=args.text_file, filter_=args.filter_,
                   verbose=args.verbose)
    if args.show == 'bands' or args.savefig:
        ml.show_patch(as_magnitudes=args.mags, figsize=args.figsize,
                      lens=args.show_lens, source_images=args.show_sources,
                      scalebar=args.scalebar, colorbar=args.colorbar, savefig=args.savefig)
    elif args.show == 'composite' or args.savefig:
        ml.show_composite(figsize=args.figsize, method=args.method,
                          lens=args.show_lens, source_images=args.show_sources,
                          scalebar=args.scalebar, savefig=args.savefig)
    elif args.show == 'both':
        ml.show_patch(as_magnitudes=args.mags, figsize=args.figsize,
                      lens=args.show_lens, source_images=args.show_sources,
                      scalebar=args.scalebar)
        ml.show_composite(figsize=args.figsize, method=args.method,
                          lens=args.show_lens, source_images=args.show_sources,
                          scalebar=args.scalebar)
    if args.gen_config is not None:
        if args.append_config:
            pass  #TODO: once a uniform solution has been found
        else:
            pass  #TODO: once a uniform solution has been found


def parse_arguments():
    from argparse import ArgumentParser, RawTextHelpFormatter
    parser = ArgumentParser(description=__doc__, formatter_class=RawTextHelpFormatter)
    # main args
    parser.add_argument("case", nargs='*',
                        help="Path input to .fits file for skyf to use",
                        default=os.path.abspath(os.path.dirname(__file__))+'/test/')
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
    parser.add_argument("-s", "--show", dest="show", metavar="<plot variant>", type=str,
                        help="Plot and show the .fits file's data as (bands/composite/both)",
                        default=None)
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
    parser.add_argument("--composite-method", dest="method", metavar="<method>", type=str,
                        help="Composite image generation method (standard/brighter/bluer)",
                        default='standard')
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
                        help="reorder the image positions relative to ABCD ordered bottom-up")

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
        from gleam.test.test_multilens import TestMultiLens
        TestMultiLens.main()
    else:
        main(case, args)
