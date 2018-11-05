#!/usr/bin/env python
"""
@author: phdenzel

Encoding utilities to help JSON serializing GLEAM objects
"""
###############################################################################
# Imports
###############################################################################
import json
import numpy as np


###############################################################################
class GLEAMEncoder(json.JSONEncoder):
    def default(self, obj):
        if hasattr(obj, '__json__'):
            return obj.__json__
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj.__dict__
        # return json.JSONEncoder.default(self, obj)
