#!/usr/bin/env python
"""
@author: phdenzel

Phase through a region in the sky with SkyPatch
"""
###############################################################################
# Imports
###############################################################################
from gleam.skyf import SkyF
from gleam.utils.rgb_map import lupton_like

import sys
import os
import copy
import math
import warnings
import numpy as np
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=RuntimeWarning)


__all__ = ['SkyPatch']


###############################################################################
class SkyPatch(object):
    """
    Framework for a set of sky patches (.fits files)
    """
    def __init__(self, files, verbose=False, **kwargs):
        """
        Initialize parsing of a fits file with a directory name

        Args:
            files <str/list(str)> - list of .fits files or directory string containing the files

        Kwargs:
            px2arcsec <float,float> - overwrite the pixel scale in the .fits header (in arcsecs)
            refpx <int,int> - overwrite reference pixel coordinates in .fits header (in pixels)
            refval <float,float> - overwrite reference pixel values in .fits header (in degrees)
            photzp <float> - overwrite photometric zero-point information
            verbose <bool> - verbose mode; print command line statements

        Return:
            <SkyPatch object> - standard initializer
        """
        # input as list of files
        if isinstance(files, list):
            self.filepaths = self.find_files(files)
        # single file input
        elif isinstance(files, str) and any([files.endswith(ext) for ext in (
                '.fits', '.fit', '.fts')]):
            self.filepaths = self.find_files([files])
        # input as directory string
        elif isinstance(files, str):
            self.filepaths = self.find_files(files)
        # if there even is such a case
        else:
            self.filepaths = self.find_files(files)
        self.fs = [SkyF(f, **kwargs) for f in self.filepaths]
        if verbose:
            print(self.__v__)

    def __getitem__(self, band):
        """
        Get item from fs either by index or by band string
        """
        if isinstance(band, str):
            try:
                return self.fs[self.bands.index(band)]
            except ValueError:
                raise ValueError('Band not found in SkyPatch object')
        elif isinstance(band, int):
            try:
                return self.fs[band]
            except ValueError:
                raise ValueError("Band index not found in SkyPatch object")
        else:
            raise IndexError

    def __eq__(self, other):
        if isinstance(other, SkyPatch):
            return self.filepaths == other.filepaths
        elif isinstance(other, str):
            return self.filepath == other
        else:
            NotImplemented

    def __copy__(self):
        args = (self.filepaths,)
        kwargs = {
            'px2arcsec': self.fs[0].px2arcsec,
            'refpx': self.fs[0].refpx,
            'refval': self.fs[0].refval,
            'photzp': self.fs[0].photzp
        }
        return SkyPatch(*args, **kwargs)

    def __deepcopy__(self, memo):
        args = (copy.deepcopy(self.filepaths, memo),)
        kwargs = {
            'px2arcsec': copy.deepcopy(self.fs[0].px2arcsec, memo),
            'refpx': copy.deepcopy(self.fs[0].refpx, memo),
            'refval': copy.deepcopy(self.fs[0].refval, memo),
            'photzp': copy.deepcopy(self.fs[0].photzp, memo)
        }
        return SkyPatch(*args, **kwargs)

    def __str__(self):
        return "SkyPatch({})".format(", ".join(self.bands))

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
        return ['N', 'filepaths', 'files', 'fs', 'bands', 'naxis1', 'naxis2']

    @property
    def __v__(self):
        """
        Info string for test printing

        Args/Kwargs:
            None

        Return:
            <str> - test of SkyPatch attributes
        """
        return "\n".join([t.ljust(20)+"\t{}".format(self.__getattribute__(t)) for t in self.tests])

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
    def N(self):
        """
        Number of .fits files/bands

        Args/Kwargs:
            None

        Return:
            N <int> - number of files
        """
        return len(self.filepaths)

    @property
    def files(self):
        """
        File names of the .fits files

        Args/Kwargs:
            None

        Return:
            filenames <list(str)> - Base name of the file paths
        """
        return [os.path.basename(f) for f in self.filepaths]

    @property
    def bands(self):
        """
        Bands of the .fits files

        Args/Kwargs:
            None

        Return:
            bands <list(str)> - list of bands of all fs
        """
        return [nu.band for nu in self.fs]

    @property
    def naxis1(self):
        """
        Length of axis 1 of all .fits files

        Args/Kwargs:
            None

        Return:
            naxis1 <list(int)> - number of pixels along axis 1 of each .fits file
        """
        return [nu.naxis1 for nu in self.fs]

    @property
    def naxis2(self):
        """
        Length of axis 2 of all .fits files

        Args/Kwargs:
            None

        Return:
            naxis2 <list(int)> - number of pixels along axis 2 of each .fits file
        """
        return [nu.naxis2 for nu in self.fs]

    @property
    def data(self):
        """
        Data from the list of .fits files

        Args/Kwargs:
            None

        Return:
            data <list(np.ndarray)> - list of .fits data maps
        """
        return [nu.data for nu in self.fs]

    @property
    def magnitudes(self):
        """
        Magnitude maps converted from the list of .fits file data

        Args/Kwargs:
            None

        Return:
            magnitudes <list(np.ndarray)> - list of converted magnitude maps
        """
        return [nu.magnitudes for nu in self.fs]

    def image_patch(self, cmap='magma'):
        """
        An 8-bit PIL.Image of the .fits data

        Args:
            None

        Kwargs:
            cmap <str> - a cmap string from matplotlib.colors.Colormap

        Return:
            f_images <list(PIL.Image object)> - a colorized image object
        """
        return [f.image_f(cmap=cmap) for f in self.fs]

    @property
    def composite(self):
        """
        Stacked composite map data from the i, g, and r band

        Args/Kwargs:
            None

        Return:
            stack <np.ndarray> - stacked image data array
        """
        if 'i' in self.bands and 'r' in self.bands and 'g' in self.bands:
            return lupton_like(self['i'].data, self['r'].data, self['g'].data, method='standard')
        return None

    @property
    def composite_image(self):
        """
        An 8-bit composite PIL.Image of the stacked .fits data

        Args/Kwargs:
            None

        Return:
            img <>
        """
        from PIL import Image
        data = self.composite
        if data is not None:
            lower = data.min()
            upper = data.max()
            img = Image.fromarray(
                np.uint8(255*(data-lower)/(upper-lower))).transpose(Image.FLIP_TOP_BOTTOM)
            return img
        return data

    @staticmethod
    def find_files(d_o_f, sequence=[".U.", ".G.", ".R.", ".I.", ".I2.", ".Z.",
                                    ".u.", ".g.", ".r.", ".i.", ".i2.", ".z."], verbose=False):
        """
        Select all .fits files in the directory

        Args:
            d_o_f <list(str)/str> - 'directory or file' path to all the .fits files

        Kwargs:
            sequence <list(str)> - sequence of keys after which the files are ordered
            verbose <bool> - verbose mode; print command line statements

        Return:
            filepaths <list(str)> - list of all .fits file paths
        """
        # sorting files after band sequence
        def band_key(f):
            for i, s in enumerate(sequence):
                if s in f:
                    return i

        # single file input
        if isinstance(d_o_f, str):
            if any([d_o_f.endswith(ext) for ext in ('.fits', '.fit', '.fts')]):
                d_o_f = [d_o_f]
            else:
                d_o_f = ['/'.join([d_o_f, f]) for f in os.listdir(d_o_f) if True
                         in [f.endswith(x) for x in ('.fits', '.fit', '.fts')]]
        # input as list of files
        elif isinstance(d_o_f, list) and len(d_o_f) == 1:
            if not any([d_o_f[0].endswith(ext) for ext in ('.fits', '.fit', '.fts')]):
                folder = d_o_f[0]
                d_o_f = ['/'.join([folder, f]) for f in os.listdir(folder) if True
                         in [f.endswith(x) for x in ('.fits', '.fit', '.fts')]]
        # input as directory string
        files = [SkyF.check_path(f, check_ext=False) for f in d_o_f]
        if verbose:
            print("Searching for files with '.fits' extension in {}".format(d_o_f))
        try:
            files = sorted(files, key=band_key)
        except TypeError:
            pass
        # some verbosity
        if verbose:
            print("Files in sorted order:")
            print("\n".join(files))
        return files

    def add_to_patch(self, filepath, index=None, verbose=False, **kwargs):
        """
        Add a file to the patch after initialization

        Args:
            filepath <str> - path to file which is to be added to the patch

        Kwargs:
            index <int> -  list index at which the file is inserted; default -1
            px2arcsec <float,float> - overwrite the pixel scale in the .fits header (in arcsecs)
            refpx <int,int> - overwrite reference pixel coordinates in .fits header (in pixels)
            refval <float,float> - overwrite reference pixel values in .fits header (in degrees)
            photzp <float> - overwrite photometric zero-point information
            verbose <bool> - verbose mode; print command line statements

        Return:
            filepath <str> - validated filepath which was added to the patch
        """
        filepath = SkyF.check_path(filepath, check_ext=False)
        if index is None or index == -1:
            self.filepaths.append(filepath)
            self.fs.append(SkyF(filepath, **kwargs))
        else:
            self.filepaths.insert(index, filepath)
            self.fs.insert(index, SkyF(filepath, **kwargs))
        if verbose:
            print(self.__v__)
        return filepath

    def remove_from_patch(self, index=None, verbose=False):
        """
        Remove a file from the patch after initialization

        Args:
            index <int> - index to be removed

        Kwargs:
            verbose <bool> - verbose mode; print command line statements

        Return:
            rmd_f, rmd_obj <str,SkyF object> - the removed filepath and corresponding object
        """
        if index is None:
            rmd_f = self.filepaths.pop(-1)
            rmd_obj = self.fs.pop(-1)
        else:
            rmd_f = self.filepaths.pop(index)
            rmd_obj = self.fs.pop(index)
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
        if len(new_sequence) == self.N:
            self.filepaths = [self.filepaths[i] for i in new_sequence]
            self.fs = [self.fs[i] for i in new_sequence]
            if verbose:
                print(self.__v__)
        else:
            raise ValueError("New sequence has the wrong length")

    def plot_composite(self, fig, ax=None, method='standard', colorbar=False, scalebar=True,
                       plain=False, verbose=False, **kwargs):
        """
        Plot a composite image using the i, r, and g band

        Args:
            fig <matplotlib.figure.Figure object> - figure in which the image is to be plotted

        Kwargs:
            ax <matplotlib.axes.Axes object> - option to control on which axis the image is plotted
            method <str> - method description; chooses the set of weights for the stacking
            scalebar <bool> - if True, add scalebar plot (15% of the image's width)
            plain <bool> - only plot the image and remove all the rest
            verbose <bool> -  verbose mode; print command line statements
            kwargs **<dict> - keywords for the imshow function

        Return:
            fig <matplotlib.figure.Figure object> - figure in which the image was plotted
            ax <matplotlib.axes.Axes object> - axis on which the image was plotted
        """
        # stack the bands
        if 'i' in self.bands and 'r' in self.bands and 'g' in self.bands:
            stack = lupton_like(self['i'].data, self['r'].data, self['g'].data, method=method)
        else:
            return fig, ax
        # check axes
        if ax is None or len(fig.get_axes()) < 1:
            ax = fig.add_subplot(111)
        # add axes if plain
        if plain:
            fig.clear()
            ax = plt.Axes(fig, [0, 0, 1, 1])
            ax.set_axis_off()
            fig.add_axes(ax)
        # plot the stacked image
        ax.imshow(stack, **kwargs)
        if scalebar:  # plot scalebar
            from matplotlib import patches
            barpos = (0.05*self['i'].naxis1, 0.025*self['i'].naxis2)
            w, h = 0.15*self['i'].naxis1, 0.01*self['i'].naxis2
            scale = self['i'].px2arcsec[0]*w
            rect = patches.Rectangle(barpos, w, h, facecolor='white', edgecolor=None, alpha=0.85)
            ax.add_patch(rect)
            ax.text(barpos[0]+w/4, barpos[1]+2*h, r"$\mathrm{{{:.1f}''}}$".format(scale),
                    color='white', fontsize=16)
        # flip axes
        ax.set_xlim(left=0, right=self['i'].naxis1)
        ax.set_ylim(bottom=0, top=self['i'].naxis2)
        ax.set_aspect('equal')
        # no axis tick labels
        plt.axis('off')
        #plt.tight_layout()
        # some verbosity
        if verbose:
            print(ax)
        return fig, ax

    def show_composite(self, savefig=None, **kwargs):
        """
        Plot and show the composite of the i, g, and r band on a new figure

        Args:
            None

        Kwargs:
            method <str> - method description; chooses the set of weights for the stacking
            scalebar <bool> - if True, add scalebar plot (15% of the image's width)
            savefig <str> - save figure in file string instead of showing it
            verbose <bool> -  verbose mode; print command line statements
            kwargs **<dict> - keywords for the imshow function

        Return:
            fig <matplotlib.figure.Figure object> - figure in which the image was plotted
            ax <matplotlib.axes.Axes object> - axis on which the image was plotted
        """
        verbose = kwargs.get('verbose', False)
        fsize = kwargs.pop('figsize', (8, 6))
        # open a new figure
        fig = plt.figure(figsize=fsize)
        ax = fig.add_subplot(111)
        fig, ax = self.plot_composite(fig, ax, **kwargs)
        if savefig is not None:
            savename = savefig + ".pdf"
            plt.savefig(savename)
        else:
            plt.show()
        # some verbosity
        if verbose:
            print(fig)
        return fig, ax

    def show_patch(self, savefig=None, **kwargs):
        """
        Plot and show all the bands on a new figure

        Args:
            None

        Kwargs:
            as_magnitudes <bool> - if True, plot data as magnitudes
            scalebar <bool> - if True, add scalebar plot (15% of the image's width)
            colorbar <bool> - if True, add colorbar plot
            savefig <str> - save figure in file string instead of showing it
            verbose <bool> -  verbose mode; print command line statements
            kwargs **<dict> - keywords for the imshow function

        Return:
            fig <matplotlib.figure.Figure object> - figure with the image's plots
            axes <list(matplotlib.axes.Axes object)> - axes on which the images were plotted
        """
        verbose = kwargs.get('verbose', False)
        fsize = kwargs.pop('figsize', (12, 8.27))
        # open a new figure
        fig = plt.figure(figsize=fsize)
        axes = []
        for b in range(self.N):
            ax = fig.add_subplot(2-int(self.N == 1), math.ceil(self.N/2.), b+1)
            fig, ax = self.fs[b].plot_f(fig, ax, **kwargs)
            axes.append(ax)
        plt.subplots_adjust(wspace=0.003, hspace=0.003)
        if savefig is not None:
            if not savefig.endswith(".pdf"):
                savefig = savefig + ".pdf"
            plt.savefig(savefig)
        else:
            plt.show()
        # some verbosity
        if verbose:
            print(fig)
        return fig, axes


# MAIN FUNCTION ###############################################################
def main(case, args):
    sp = SkyPatch(case, verbose=args.verbose)
    if args.show == 'bands' or args.savefig is not None:
        sp.show_patch(as_magnitudes=args.mags, figsize=args.figsize,
                      savefig=args.savefig, scalebar=args.scalebar, colorbar=args.colorbar)
    elif args.show == 'composite' or args.savefig is not None:
        sp.show_composite(figsize=args.figsize, method=args.method,
                          savefig=args.savefig, scalebar=args.scalebar)
    elif args.show == 'both':
        sp.show_patch(as_magnitudes=args.mags, figsize=args.figsize, scalebar=args.scalebar)
        sp.show_composite(figsize=args.figsize, method=args.method, scalebar=args.scalebar)


def parse_arguments():
    from argparse import ArgumentParser, RawTextHelpFormatter
    parser = ArgumentParser(description=__doc__, formatter_class=RawTextHelpFormatter)
    # main args
    parser.add_argument("case", nargs='*',
                        help="Path input to .fits file for skyf to use",
                        default=os.path.abspath(os.path.dirname(__file__))+"/test/")
    parser.add_argument("--scale", metavar=("<dx", "dy>"), nargs=2, type=float,
                        help="Pixel-to-arcsec scale for x (-RA) and y (Dec) direction")
    parser.add_argument("--refpx", metavar=("<x", "y>"), nargs=2, type=float,
                        help="Coordinates of the reference pixel")
    parser.add_argument("--refval", metavar=("<ra", "dec>"), nargs=2, type=float,
                        help="Values of the reference pixel")
    parser.add_argument("--photzp", metavar="<zp>", type=float,
                        help="Magnitude zero-point information")

    # plotting args
    parser.add_argument("-s", "--show", dest="show", metavar="<plot variant>", type=str,
                        help="Plot and show the .fits file's data as (bands/composite/both)",
                        default=None)
    parser.add_argument("--figsize", dest="figsize", metavar=("<w", "h>"), nargs=2, type=float,
                        help="Size of the figure (is multiplied by the default dpi)")
    parser.add_argument("-m", "--magnitudes", dest="mags", action="store_true",
                        help="Plot the .fits file's data in magnitudes",
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
        from gleam.test.test_skypatch import TestSkyPatch
        TestSkyPatch.main()
    else:
        main(case, args)
