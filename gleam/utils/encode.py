#!/usr/bin/env python
"""
@author: phdenzel

Encoding utilities to help JSON serializing GLEAM objects
"""
###############################################################################
# Imports
###############################################################################
import json
import re
import numpy as np


###############################################################################
class GLEAMEncoder(json.JSONEncoder):
    def default(self, obj):
        """
        Encode the objects to be serializable

        Args:
            obj <object> - object to be serialized

        Kwargs:
            None

        Return:
            obj <object> - serialized object
        """
        if hasattr(obj, '__json__'):
            return obj.__json__
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if 'numpy' in str(type(obj)):  # since numpy types don't have __dict__
            return obj.tolist()
        else:
            return obj.__dict__
        # return json.JSONEncoder.default(self, obj)


class GLEAMDecoder(object):
    decoding_kwargs = {}

    @staticmethod
    def decode(jdict):
        """
        Decoding types requires __type__ signature in json file and
        optionally from_dict implementation

        Args:
            jdict <dict> - dictionary from json loading

        Kwargs:
            None

        Return:
            jcls <__type__ object> - decoded json object

        Note:
            - to be used as object_hook in json.load function
        """
        if '__type__' in jdict:
            jcls_name = jdict.pop('__type__')
            if jcls_name in globals():
                jcls = globals()[jcls_name]
            else:
                if '.' in jcls_name:  # if subclass type
                    jcls_name, jcls_subname = jcls_name.split('.')
                    jmodule = __import__('.'.join(['gleam', jcls_name.lower()]), fromlist=[''])
                    jcls = getattr(jmodule, jcls_name)
                    jcls = getattr(jcls, jcls_subname)
                else:
                    jmodule = __import__('.'.join(['gleam', jcls_name.lower()]), fromlist=[''])
                    jcls = getattr(jmodule, jcls_name)
            if 'from_jdict' in dir(jcls):
                try:
                    return jcls.from_jdict(jdict, **GLEAMDecoder.decoding_kwargs)
                except TypeError:
                    GLEAMDecoder.decoding_kwargs = {}
                    return jcls.from_jdict(jdict, **GLEAMDecoder.decoding_kwargs)
            else:
                return jcls(**jdict)
        return jdict


def an_sort(data, argsort=False):
    """
    Perform an alpha-numeric, natural sort

    Args:
        data <list> - list of strings

    Kwargs:
        None

    Return:
        sorted <list> - the alpha-numerically, naturally sorted list of strings
    """
    def convert(text):
        return int(text) if text.isdigit() else text.lower()

    def an_key(key):
        return [convert(c) for c in re.split('([0-9]+)', key)]
    if argsort:
        return [i[0] for i in sorted(enumerate(data), key=lambda x: an_key(x[1]))]
    return sorted(data, key=an_key)
