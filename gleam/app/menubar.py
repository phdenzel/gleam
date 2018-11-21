#!/usr/bin/env python
"""
@author: phdenzel

A Menubar - a la carte
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
        self.main_labels = ['file', 'edit', 'view', 'help']
        self.menu = self.rebuild()
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
        return ['menu', 'main_labels', 'labels', 'shortcuts', 'bindings', 'menu_settings']

    @property
    def menu_settings(self):
        """
        Menu settings summarized as a dictionary

        Args/Kwargs:
            None

        Return:
            menu_settings <dict> - the entire settings
        """
        return {ml: {l: (self.bindings[ml][l], s)
                     for l, s in zip(self.labels[ml], self.shortcuts[ml])}
                for ml in self.main_labels}

    @property
    def labels(self):
        """
        Labels for each main label

        Args/Kwargs:
            None

        Return:
            labels <dict(list)> - a dictionary of lists containing menu label strings
        """
        if not hasattr(self, '_labels'):
            self._labels = {ml: [] for ml in self.main_labels}
        for ml in self.main_labels:
            self._labels.setdefault(ml, [])
        return self._labels

    @labels.setter
    def labels(self, labels):
        """
        Set all labels for each main label

        Args:
            labels <list(list(str))> - a list of lists containing menu label strings

        Kwargs/Return:
            None
        """
        self._labels = {ml: l for ml, l in zip(self.main_labels, labels)}

    def add_label(self, main_label, label):
        """
        Alternative setter for a single main label

        Args:
            main_label <str> - the main label
            label <str> - the menu label to be added under main menu

        Kwargs/Return:
            None
        """
        if not hasattr(self, '_labels'):
            self._labels = {ml: [] for ml in self.main_labels}
        self._labels[main_label].append(label)

    @property
    def shortcuts(self):
        """
        Shortcuts for each main label

        Args/Kwargs:
            None

        Return:
            shortcuts <dict(list(str))> - a list of lists containing menu label strings
        """
        if not hasattr(self, '_shortcuts'):
            self._shortcuts = {ml: ['']*len(self.labels[ml]) for ml in self.main_labels}
        for ml in self.main_labels:
            while len(self._shortcuts[ml]) != len(self.labels[ml]):
                self._shortcuts[ml].append('')
        return self._shortcuts

    @shortcuts.setter
    def shortcuts(self, shortcuts):
        """
        Set all shortcuts for each main label

        Args:
            shortcuts <list(list(str))> - a list of lists containing menu shortcut strings

        Kwargs/Return:
            None
        """
        self._shortcuts = {ml: s for ml, s in zip(self.main_labels, shortcuts)}

    def add_shortcut(self, label, shortcut):
        """
        Add a shortcut to a menu label

        Args:
            label <str> - label to which the shortcut is added
            shortcut <str> - the shortcut to be added

        Kwargs/Return:
            None
        """
        for ml in self.main_labels:
            while len(self._shortcuts[ml]) != len(self.labels[ml]):
                self._shortcuts[ml].append('')
        for ml in self.main_labels:
            try:
                index = self.labels[ml].index(label)
            except ValueError:
                continue
            else:
                self._shortcuts[ml][index] = shortcut

    @property
    def bindings(self):
        """
        Bindings belonging to the labels, summarized as a dictionary

        Args/Kwargs:
            None

        Return:
            bindings <dict(list(func))> - a list of lists containing menu binding functions
        """
        if not hasattr(self, '_bindings'):
            self._bindings = {ml: {l: self.dummy_func for l in self.labels[ml]}
                              for ml in self.main_labels}
        for ml in self.main_labels:
            self._bindings.setdefault(ml, {l: self.dummy_func for l in self.labels[ml]})
        return self._bindings

    @bindings.setter
    def bindings(self, bindings):
        """
        Set all bindings for each main label

        Args:
            bindings <list(list(func))> - contains binding functions for each section

        Kwargs/Return:
            None
        """
        self._bindings = {ml: {l: b for l, b in zip(self.labels[ml], bindings[i])}
                          for i, ml in enumerate(self.main_labels)}

    def add_binding(self, label, binding):
        """
        Add a binding function to an existing menu label

        Args:
            binding <func> - the binding function to be added

        Kwargs/Return:
            None
        """
        for ml in self.main_labels:
            self._bindings.setdefault(ml, {l: self.dummy_func for l in self.labels[ml]})
        for ml in self.main_labels:
            if label in self._bindings[ml]:
                self._bindings[ml][label] = binding
                break

    def fmtl(self, label, shortcut, spacing=3):
        """
        Formatted labels from label and shortcut

        Args:
            label <str> - label string
            shortcut <str> - shortcut string

        Kwargs:
            spacing <int> - additional spacing between label and shortcut

        Return:
            label <str> - formatted label strings
        """
        left_spacing = max([len(l) for ml in self.main_labels for l in self.labels[ml]]) + spacing
        lb = label.capitalize().ljust(left_spacing) + '\t'
        sc = shortcut
        return lb+sc

    def rebuild(self):
        """
        Build the menu anew (erasing all previous versions, if any existed)

        Args/Kwargs:
            None

        Return:
            menu <tk.Menu object> - the rebuilt menu
        """
        if hasattr(self, 'menu'):
            del self.menu
        self.menu = tk.Menu(self, tearoff=0)
        for ml in self.main_labels:
            self.__setattr__(ml, tk.Menu(self.menu, tearoff=0))
            m = self.__getattribute__(ml)
            for l, s in zip(self.labels[ml], self.shortcuts[ml]):
                m.add_command(label=self.fmtl(l, s), command=self.bindings[ml][l])
            self.menu.add_cascade(label=ml.capitalize(), menu=m)
        self.env.root.configure(menu=self.menu)
        return self.menu

    def dummy_func(self):
        print('DUMMY ACTION')
