#!/usr/bin/env python
"""
@author: phdenzel

The Toolbar provides what you need, to do what you want
"""
###############################################################################
# Imports
###############################################################################
import sys
import os
from PIL import Image, ImageTk
if sys.version_info.major < 3:
    import Tkinter as tk
    # import ttk
elif sys.version_info.major == 3:
    import tkinter as tk
    # import tkinter.ttk as ttk
else:
    raise ImportError("Could not import Tkinter")

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
        self.sections = ['data']

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
        return ['sections', 'tools', 'labels', 'buttons', 'spinboxes', 'bindings', 'icons']

    @property
    def labels(self):
        """
        Lists of all labels for each section

        Args/Kwargs:
            None

        Return:
            labels <dict(list(tk.Label object))> - a dictionary of lists containing all labels
                                                   within a section
        """
        return {s: [self.tools[s][k] for k in self.sij(s)
                    if isinstance(self.tools[s][k], tk.Label)]
                for s in self.sections}

    @property
    def buttons(self):
        """
        Lists of all buttons for each section

        Args/Kwargs:
            None

        Return:
            buttons <dict(list(tk.Button object))> - a dictionary of lists containing all buttons
                                                     within a section
        """
        return {s: [self.tools[s][k] for k in self.sij(s)
                    if isinstance(self.tools[s][k], tk.Button)]
                for s in self.sections}

    @property
    def spinboxes(self):
        """
        Lists of all spinboxes for each section

        Args/Kwargs:
            None

        Return:
            spinboxes <dict(list(tk.Spinbox object))> - a dictionary of lists containing
                                                        all spinboxes within a section
        """
        return {s: [self.tools[s][k] for k in self.sij(s)
                    if isinstance(self.tools[s][k], tk.Spinbox)]
                for s in self.sections}

    @property
    def icons(self):
        """
        Buffer for icons and images

        Args/Kwargs:
            None

        Return:
            icons <list(ImageTk.PhotoImage object)> - a list of images to keep in memory
        """
        if not hasattr(self, '_icons'):
            self._icons = []
        return self._icons

    @icons.setter
    def icons(self, icon):
        """
        Append images to buffer

        Args:
            icons <ImageTk.PhotoImage object> - an image to be kept in buffer

        Kwargs/Return:
            None
        """
        if not hasattr(self, '_icons'):
            self._icons = []
        self._icons.append(icon)

    def add_labels(self, section, labels, **kwargs):
        """
        Add labels to a section line by line

        Args:
            section <str> - the section to which the labels are appended in tool settings
            labels <list(list(str/tk.StringVar))> - 2D list specifying label elements to be
                                                    placed on a grid according to the list index

        Kwargs/Return:
            None
        """
        if not hasattr(self, '_tools'):
            self._tools = {s: {} for s in self.sections}
        for s in self.sections:
            self._tools.setdefault(s, {})
        # set kwargs defaults
        defaults = {}
        for k in defaults:
            kwargs.setdefault(k, defaults[k])
        # create tools
        tools = []
        for i, line in enumerate(labels):
            tool_line = []
            for l in line:
                kw = {k: kwargs[k] for k in kwargs}
                if isinstance(l, tk.StringVar):
                    kw.update({'textvariable': l})
                if isinstance(l, str):
                    kw.update({'text': l.capitalize()})
                tool_line.append(tk.Label(self, **kw))
            tools.append(tool_line)
        # order tools
        self.ifocus += 1
        for i, line in enumerate(tools):
            i = i + self.ifocus
            for j, t in enumerate(line):
                self._tools[section][(i, j)] = t
        self.ifocus = i

    def add_buttons(self, section, buttons, **kwargs):
        """
        Add buttons to a section line by line

        Args:
            section <str> - the section to which the buttons are appended in tool settings
            labels <list(list(str))> - 2D list specifying labels for the button elements to be
                                       placed on a grid according to the list index

        Kwargs/Return:
            None
        """
        if not hasattr(self, '_tools'):
            self._tools = {s: {} for s in self.sections}
        for s in self.sections:
            self._tools.setdefault(s, {})
        # set kwargs defaults
        defaults = {'padx': 6, "pady": 2}
        for k in defaults:
            kwargs.setdefault(k, defaults[k])
        # create tools
        tools = []
        for i, line in enumerate(buttons):
            tool_line = []
            for j, b in enumerate(line):
                kw = {k: kwargs[k] for k in kwargs}
                kw.update({'text': b})
                if b.endswith('.png'):
                    root_path = os.path.dirname(os.path.realpath(__file__))
                    image = Image.open(root_path+b)
                    image = image.resize((25, 25))
                    self.icons = ImageTk.PhotoImage(image)
                    kw.update({'image': self.icons[-1]})
                tool_line.append(tk.Button(self, **kw))
            tools.append(tool_line)
        # order tools
        self.ifocus += 1
        for i, line in enumerate(tools):
            i = i + self.ifocus
            for j, t in enumerate(line):
                self._tools[section][(i, j)] = t
        self.ifocus = i

    @property
    def bindings(self):
        """
        Bindings belonging for the buttons, summarized as a dictionary

        Args/Kwargs:
            None

        Return:
            bindings <dict(list(func))> - a list of lists containing menu binding functions
        """
        if not hasattr(self, '_bindings'):
            self._bindings = {s: [self.dummy_func for b in self.buttons[s]]
                              for s in self.sections}
        for s in self.sections:
            self._bindings.setdefault(s, [self.dummy_func for b in self.buttons[s]])
        return self._bindings

    @bindings.setter
    def bindings(self, bindings):
        """
        Set all bindings for each main label

        Args:
            bindings <dict(list)> - contains binding functions for each section

        Kwargs/Return:
            None
        """
        self._bindings = {s: b for s, b in zip(self.sections, bindings)}

    @property
    def tools(self):
        """
        Tool settings for each section

        Args/Kwargs:
            None

        Return:
            tools <dict> - dictionary of an element list for each sections
        """
        if not hasattr(self, '_tools'):
            self._tools = {s: {} for s in self.sections}
        for i, s in enumerate(self.sections):
            self._tools.setdefault(s, {})
        return self._tools

    @tools.setter
    def tools(self, tools):
        """
        Tool settings for each section

        Args:
            tools <dict> - dictionary of an element list for each sections

        Kwargs/Return:
            None
        """
        if not hasattr(self, '_tools'):
            self._tools = {s: {} for s in self.sections}
        for s in self.sections:
            self._tools.setdefault(s, {})
        self._tools.update(tools)

    @property
    def ifocus(self):
        """
        Line index in focus

        Args/Kwargs:
            None

        Return:
            ifocus <int> - line index in focus

        Note:
            - use add_* functions to append to the current tool configuration,
              and move ifocus to a new line in case you want to insert instead of append
        """
        if not hasattr(self, '_ifocus'):
            self._ifocus = -1
        return self._ifocus

    @ifocus.setter
    def ifocus(self, ifocus):
        """
        Set new line index focus

        Args:
            ifocus <int> - line index in focus

        Kwargs/Return:
            None

        Note:
            - use add_* functions to append to the current tool configuration,
              and move ifocus to a new line in case you want to insert instead of append
        """
        self._ifocus = ifocus

    @property
    def ij(self):
        """
        Iterator for the tool grid configuration

        Args/Kwargs:
            None

        Return:
            ij <generator> - ordered row and colum index iterator over the grid
        """
        nrows = max([k[0] for s in self.sections for k in self.tools[s]])+1
        ncols = max([k[1] for s in self.sections for k in self.tools[s]])+1
        for i in range(nrows):
            for j in range(ncols):
                if any([(i, j) in self.tools[s] for s in self.sections]):
                    yield (i, j)

    def sij(self, section):
        """
        Iterator over indices in a section

        Args:
            section <str> - the section over which to iterate over

        Kwargs:
            None

        Return:
            sij <generator> - ordered row and column index iterator over section
        """
        for k in self.ij:
            if k in self.tools[section]:
                yield k

    def rebuild(self):
        """
        Build the toolbar anew (erasing all previous versions, if any existed)

        Args/Kwargs/Return:
            None
        """
        self._tools['sections'] = []
        self.ifocus = -1
        for s in self.sections:
            self.ifocus += 1
            slabel = tk.Label(self, text=s.capitalize())
            self._tools['sections'].append(slabel)
            slabel.grid(row=self.ifocus, column=0, sticky=tk.N+tk.W)
            self.ifocus += 1
            for i, j in self.sij(s):
                t = self.tools[s][(i, j)]
                if isinstance(t, tk.Button):
                    idx = self.buttons[s].index(t)
                    t.configure(command=self.bindings[s][idx])
                i = i + self.ifocus
                if j == 0:
                    t.grid(row=i, column=j, sticky=tk.N+tk.W, padx=(20, 0), pady=5)
                else:
                    t.grid(row=i, column=j, sticky=tk.N+tk.W, padx=(0, 10), pady=5)
            self.ifocus = i

    def dummy_func(self):
        print('DUMMY ACTION')
