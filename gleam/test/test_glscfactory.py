#!/usr/bin/env python
"""
@author: phdenzel

Make really, really sure you can...
Feed the glass config factory with information
"""
###############################################################################
# Imports
###############################################################################
from gleam.lensobject import LensObject
from gleam.glscfactory import GLSCFactory
import os
from gleam.test.utils import UnitTestPrototype


# CLASS TESTING ###############################################################
class TestGLSCFactory(UnitTestPrototype):

    def setUp(self):
        # arguments and keywords
        self.test_fits = os.path.abspath(os.path.dirname(__file__)) \
                         + '/W3+3-2.I.12907_13034_7446_7573.fits'
        self.kwargs = {
            'text_file': os.path.abspath(os.path.dirname(__file__)) + "/test_lensinfo.txt",
        }
        self.v = {'verbose': 1}
        # __init__ test
        self.factory = GLSCFactory(fits_file=self.test_fits, **self.kwargs)
        # verbosity
        self.kwargs.update(self.v)
        print("")
        print(self.separator)
        print(self.shortDescription())

    def tearDown(self):
        print("")

    def test_GLSCFactory(self):
        """ # GLSCFactory """
        print(">>> {}".format(self.test_fits))
        factory = GLSCFactory(fits_file=self.test_fits, **self.kwargs)
        self.assertIsInstance(factory, GLSCFactory)
        lo = LensObject(self.test_fits)
        print(">>> {}".format(lo))
        factory = GLSCFactory(lensobject=lo, **self.kwargs)
        self.assertIsInstance(factory, GLSCFactory)

    def test_text_extract(self):
        """ # text_extract """
        print(">>> {}".format(self.test_fits))
        info = GLSCFactory.text_extract(self.factory.text, **self.v)
        self.assertIsInstance(info, dict)
        self.assertIsNot(info, {})

    def test_lens_extract(self):
        """ # lens_extract """
        print(">>> {}".format(self.test_fits))
        lo = LensObject(self.test_fits)
        info = GLSCFactory.lens_extract(lo, **self.v)
        self.assertIsInstance(info, dict)
        self.assertIsNot(info, {})

    def test_write(self):
        """ # write """
        filename = os.path.abspath(os.path.dirname(__file__))+'/test.gls'
        print(">>> {}".format(filename))
        self.factory.write(filename, **self.v)
        self.assertTrue(os.path.isfile(filename))
        self.assertGreater(os.stat(filename).st_size, 0)
        try:
            os.remove(filename)
        except OSError:
            pass

    def test_append(self):
        """ # append """
        filename = os.path.abspath(os.path.dirname(__file__))+'/test.gls'
        print(">>> {}".format(filename))
        self.factory.append(filename, last=True,  **self.v)
        self.assertTrue(os.path.isfile(filename))
        self.assertGreater(os.stat(filename).st_size, 0)
        try:
            os.remove(filename)
        except OSError:
            pass


if __name__ == "__main__":
    TestGLSCFactory.main(verbosity=1)
