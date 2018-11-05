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
        self.data = np.array(data) if data is not None else data
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
    def from_skyf(cls, skyf, **kwargs):
        """
        Initialize from a gleam.SkyF instance

        Args:
            skyf <SkyF object> - contains data of a .fits file

        Kwargs:
            verbose <bool> - verbose mode; print command line statements

        Return:
           <ROISelector object> - initializer with SkyF
        """
        return cls(skyf.data, **kwargs)

    @classmethod
    def from_skypatch(cls, skypatch, **kwargs):
        """
        Initialize from a gleam.SkyPatch instance

        Args:
            skypatch <SkyPatch object> - contains data of several .fits files

        Kwargs:
             verbose <bool> - verbose mode; print command line statements

        Return:
           <ROISelector object> - initializer with SkyPatch
        """
        cls(skypatch.data, **kwargs)

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
        from gleam.test.test_roiselector import TestROISelector
        TestROISelector.main(verbosity=1)
