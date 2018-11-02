#!/usr/bin/env python
"""
@author: phdenzel

Stay up-to-date with the Statusbar
"""
###############################################################################
# Imports
###############################################################################
import sys
if sys.version_info.major < 3:
    import Tkinter as tk
elif sys.version_info.major == 3:
    import tkinter as tk
else:
    raise ImportError("Could not import Tkinter")
from gleam.app.prototype import FramePrototype


# STATUSBAR CLASS #############################################################
class Statusbar(FramePrototype):
    """
    Frame displaying the logged status
    """
    def __init__(self, master, message='', verbose=False, *args, **kwargs):
        """
        Initialize with first message and link to master

        Args:
            master <Tk object> - master root of the frame

        Kwargs:
            message <str> - initialize Statusbar with a message
            verbose <bool> - verbose mode; print command line statements

        Return:
            <Statusbar object> - standard initializer
        """
        FramePrototype.__init__(self, master, *args, **kwargs)
        self.log_history = []
        self.message = tk.StringVar()
        self.log(message)
        label = tk.Label(self, justify=tk.LEFT, anchor=tk.W, padx=5, textvariable=self.message)
        label.grid(sticky=tk.E)
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
        return ['message', 'log_history']

    def log(self, *message, **kwargs):
        """
        Log a message and display on the Statusbar

        Args:
            message <str> - message to be logged and displayed

        Kwargs:
            verbose <bool> - verbose mode; print command line statements

        Return:
            None
        """
        status = "".join(message)
        verbose = kwargs.pop('verbose', False)
        if verbose:
            print("Status: {:s}".format(status))
        self.message.set(status)
        self.log_history.append(message)
