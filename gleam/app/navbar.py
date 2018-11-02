#!/usr/bin/env python
"""
@author: phdenzel

Find your way through the matrix with the Navbar
"""
###############################################################################
# Imports
###############################################################################
from gleam.app.prototype import FramePrototype


# NAVBAR CLASS ################################################################
class Navbar(FramePrototype):
    """
    Frame containing a navbar for generic app operations
    """
    def __init__(self, master, verbose=False, *args, **kwargs):
        """
        Initialize navigation system through the app

        Args:
            master <Tk object> - master root of the frame

        Kwargs:
            verbose <bool> - verbose mode; print command line statements

        Return:
            <Navbar object> - standard initializer
        """
        FramePrototype.__init__(self, master, *args, **kwargs)
