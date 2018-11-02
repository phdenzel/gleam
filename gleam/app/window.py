#!/usr/bin/env python
"""
@author: phdenzel

A window into another univ... well actually OUR Universe
"""
###############################################################################
# Imports
###############################################################################
# import sys
# if sys.version_info.major < 3:
#     import Tkinter as tk
# elif sys.version_info.major == 3:
#     import tkinter as tk
# else:
#     raise ImportError("Could not import Tkinter")
from gleam.app.prototype import FramePrototype, CanvasPrototype


# WINDOW CLASS ################################################################
class Window(FramePrototype):
    """
    Frame window containing a canvas for visualization
    """
    def __init__(self, master, N=None, cell_size=(300, 300), ncols=3,
                 verbose=False, *args, **kwargs):
        """
        Initialize a grid ready to add images to

        Args:
            master <Tk object> - master root of the frame

        Kwargs:
            N <int> - number of grid cells in the canvas (default: lens_patch.N)
            cell_size <int,int> - cell dimensions of the canvas' grid
            ncols <int> - number of columns on the canvas
            verbose <bool> - verbose mode; print command line statements

        Return:
            <Window object> - standard initializer
        """
        FramePrototype.__init__(self, master, *args, **kwargs)
        self.canvas = CanvasPrototype(self, self.number_of_bands, name='canvas',
                                      cell_size=cell_size, ncols=ncols,
                                      borderwidth=0, highlightthickness=0)
        if verbose:
            print(self.__v__)

    @property
    def tests(self):
        """
        A list of attributes being tested when calling __v__

        Args/Kwargs:
            None

        Return:
            tests <list(str)> - a list of test variable strings
        """
        return ['number_of_bands', 'canvas']

    @property
    def number_of_bands(self):
        """
        Number of images (for now use as many as there are in 'lens_patch')

        Args/Kwargs:
            None

        Return:
            N <int> - number of images to be projected onto canvas
        """
        if 'app' in self.env.elements and hasattr(self.env.app, 'lens_patch'):
            return self.env.app.lens_patch.N
        else:
            return 0
