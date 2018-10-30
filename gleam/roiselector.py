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
import __init__

import sys
import os
import copy
import math
import warnings
import numpy as np

warnings.filterwarnings("ignore", category=RuntimeWarning)


__all__ = ['ROISelector']


###############################################################################
class ROISelector(object):
    """
    Selects pixels from fits files
    """
    def __init__(self, data, verbose=False):
        """
        Initialize the selector from a pure data array

        Args:
            data <np.ndarray> - the data on which the selection is based

        Kwargs:
            verbose <bool> - verbose mode; print command line statements

        Return:
           <ROISelector object> - standard initializer (also see other classmethod initializers)
        """
        self.data = data
