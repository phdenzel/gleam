#!/usr/bin/env python
"""
@authot: phdenzel

Make a directory from filepath or path
"""


def mkdir_p(pathname):
    """
    Create a directory as if using 'mkdir -p' on the command line

    Args:
        pathname <str> - create all directories in given path

    Kwargs/Return:
        None
    """
    from os import makedirs, path
    from errno import EEXIST

    try:
        makedirs(pathname)
    except OSError as exc:  # Python > 2.5
        if exc.errno == EEXIST and path.isdir(pathname):
            pass
        else:
            raise
