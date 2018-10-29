#!/usr/bin/env python
"""
@author: phdenzel

A Menubar a la carte
"""
###############################################################################
# Imports
###############################################################################
import sys
if sys.version_info.major < 3:
    import Tkinter as tk
    # import ttk
elif sys.version_info.major == 3:
    import tkinter as tk
    # import tkinter.ttk as ttk
else:
    raise ImportError("Could not import Tkinter")
import __init__
from gleam.app.prototype import FramePrototype


# MENUBAR CLASS ###############################################################
class Menubar(FramePrototype):
    """
    Frame containing a menubar; will depend on the OS' windowing system (x11, win32 or aqua)
    """
    def __init__(self, master, verbose=False, *args, **kwargs):
        """
        Initialize cascading menu containing all menu options

        Args:
            master <Tk object> - master root of the frame

        Kwargs:
            verbose <bool> - verbose mode; print command line statements

        Return:
            <Menubar object> - standard initializer
        """
        FramePrototype.__init__(self, master, *args, **kwargs)
        self.menu = tk.Menu(self, tearoff=0)
        self.menu_labels = {
            'file': {
                'open...': u'\u2303F',
                'save...': u'\u2303S',
                'exit': ''
            },
            'edit': {
                'colormaps': '',
            },
            'view': {
                'status bar': '',
            },
            'help': {
                'help': u'\u2303H',
                'about': ''
            }
        }

        self.file_menu = tk.Menu(self, tearoff=0)
        self.file_menu.add_command(label=self.label('file', 'open...'), command=self.dummy)
        self.file_menu.add_command(label=self.label('file', 'save...'), command=self.dummy)
        self.file_menu.add_command(label=self.label('file', 'exit'), command=self.env.app._on_close)
        self.menu.add_cascade(label=u"File", menu=self.file_menu)
        
        self.edit_menu = tk.Menu(self, tearoff=0)
        #
        self.menu.add_cascade(label=u"Edit", menu=self.edit_menu)
        
        self.view_menu = tk.Menu(self, tearoff=0)
        #
        self.menu.add_cascade(label=u"View", menu=self.view_menu)
        
        self.help_menu = tk.Menu(self, tearoff=0)
        #
        self.menu.add_cascade(label=u"Help", menu=self.help_menu)
        
        self.env.root.configure(menu=self.menu)


    def label(self, menu, command, spacing=3):
        """
        Get a formatted label string for each command label
        """
        left_spacing = max([len(k) for k in self.menu_labels[menu]]) + spacing
        right_spacing = max([len(i) for i in self.menu_labels[menu].items() if i])
        left = command.capitalize().ljust(left_spacing)
        right = self.menu_labels[menu][command].rjust(right_spacing)
        return left+right

    def dummy(self):
        pass
