#!/usr/bin/env python
"""
@author: phdenzel

The Toolbar provides what you need, to do what you want
"""
###############################################################################
# Imports
###############################################################################
import __init__
from gleam.app.prototype import FramePrototype


# TOOLBAR CLASS ################################################################
class Toolbar(FramePrototype):
    """
    Frame containing app-specific operations
    """
    def __init__(self, master, verbose=False, *args, **kwargs):
        """
        Args:
            master <Tk object> - master root of the frame

        Kwargs:
            verbose <bool> - verbose mode; print command line statements

        Return:
            <Toolbar object> - standard initializer
        """
        FramePrototype.__init__(self, master, *args, **kwargs)
