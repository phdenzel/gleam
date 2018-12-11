#!/usr/bin/env python
"""
@author: phdenzel

Is the GLASS half full or half empty?
"""
###############################################################################
# Imports
###############################################################################
import sys
import os

root = os.path.dirname(os.path.dirname(os.path.abspath(os.path.realpath(__file__))))
libspath = os.path.join(root, 'lib')
if os.path.exists(libspath):
    libs = os.listdir(libspath)[::-1]
    for l in libs:
        lib = os.path.join(libspath, l)
        if lib not in sys.path or not any(['glass' in p for p in sys.path]):
            sys.path.insert(3, lib)

from glass.command import command, Commands
from glass.environment import env, Environment
from glass.exceptions import GLInputError


# Variables ###################################################################
_omp_opts = None


# Functions ###################################################################
def _detect_cpus():
    """
    Detects the number of CPUs on a system.
    From http://codeliberates.blogspot.com/2008/05/detecting-cpuscores-in-python.html
    From http://www.artima.com/weblogs/viewpost.jsp?thread=230001
    """
    import subprocess
    # Linux, Unix and MacOS
    if hasattr(os, "sysconf"):
        if "SC_NPROCESSORS_ONLN" in os.sysconf_names:
            # Linux & Unix:
            ncpus = os.sysconf("SC_NPROCESSORS_ONLN")
            if isinstance(ncpus, int) and ncpus > 0:
                return ncpus
        # OSX
        else:
            return int(subprocess.Popen(
                "sysctl -n hw.ncpu", shell=True, stdout=subprocess.PIPE).communicate()[0])
    # Windows
    if "NUMBER_OF_PROCESSORS" in os.environ:
        ncpus = int(os.environ["NUMBER_OF_PROCESSORS"])
        if ncpus > 0:
            return ncpus
    # Default
    return 1


def _detect_omp():
    global _omp_opts
    if _omp_opts is not None:
        return _omp_opts
    try:
        import weave
        kw = dict(
            extra_compile_args=['-O3', '-fopenmp', '-DWITH_OMP',
                                '-Wall', '-Wno-unused-variable'],
            extra_link_args=['-lgomp'],
            headers=['<omp.h>'])
        weave.inline(' ', **kw)
    except ImportError:
        kw = {}
    _omp_opts = kw
    return kw


@command('Load a glass basis set')
def glass_basis(gls, name, **kwargs):
    """
    Import a glass model basis

    Args:
        name <str> - a module name to be imported

    Kwargs:
        solver <str> - solver to be loaded

    Return:
        None
    """
    gls.basis_options = kwargs
    f = __import__(name, globals(), locals())
    for name, [f, g, help_text] in Commands.glass_command_list.iteritems():
        if name in __builtins__.__dict__:
            message = 'WARNING: Glass command {:s} ({:s}) overrides previous function {:s}'
            print(message.format(name, f, __builtins__.__dict__[name]))
        __builtins__.__dict__[name] = g


def glass_renv():
    """
    Call to set up a standard glass 'read' environment

    Args/Kwargs/Return:
        None
    """
    Environment.global_opts['ncpus_detected'] = _detect_cpus()
    Environment.global_opts['ncpus'] = 1
    Environment.global_opts['omp_opts'] = _detect_omp()
    Environment.global_opts['withgfx'] = True
    Commands.set_env(Environment())
    import glass.glcmds
    import glass.scales
    import glass.plots
    glass_basis('glass.basis.pixels', solver=None)
    exclude_all_priors()
    return glass


def glass_wenv(config):
    """
    Call to set up a standard glass 'write' environment

    Args/Kwargs/Return:
        None
    """
    pass  # TODO


if __name__ == "__main__":
    pass  # TODO: tests
