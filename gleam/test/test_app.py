#!/usr/bin/env python
"""
@author: phdenzel

Make really, really sure you can...
Climb every peak in search for lens and source candidates
"""
###############################################################################
# Imports
###############################################################################
import __init__
from gleam.gui import App
import os
from gleam.test.utils import UnitTestPrototype


# CLASS TESTING ###############################################################
class TestApp(UnitTestPrototype):

    def setUp(self):
        # arguments and keywords
        self.test_fits = os.path.abspath(os.path.dirname(__file__))
        self.v = {'verbose': 1}
        # __init__ test
        self.root, self.app = App.init(self.test_fits, display_off=True)
        # verbosity
        print("")
        print(self.separator)
        print(self.shortDescription())

    def tearDown(self):
        print("")

    def test_App(self):
        """ # App """
        print(">>> {}".format(self.test_fits))
        app = App(self.root, self.test_fits, display_off=False, **self.v)
        self.assertIsInstance(app, App)
        
    def test_tk(self):
        """ # tcl_library / tk_library """
        print(self.root.tk.exprstring('$tcl_library'))
        print(self.root.tk.exprstring('$tk_library'))


if __name__ == "__main__":
    TestApp.main(verbosity=1)
