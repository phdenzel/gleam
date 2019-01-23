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

    Args/Kwargs/Return:
        None
    """
    import subprocess
    # Linux, Unix and MacOS
    if hasattr(os, "sysconf"):
        if "SC_NPROCESSORS_ONLN" in os.sysconf_names:
            # Linux & Unix:
            ncpus = os.sysconf("SC_NPROCESSORS_ONLN")
            if isinstance(ncpus, int) and ncpus > 0:
                return ncpus
        # MacOS
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
    """
    Detects the OpenMP options

    Args/Kwargs/Return:
        None
    """
    global _omp_opts
    if _omp_opts is not None:
        return _omp_opts
    if 'clang' in sys.version:  # probably macOS clang build
        kw = dict(
            extra_compile_args=['-O3', '-DWITH_OMP', '-Wno-unused-variable'])
    else:  # most likely some gcc build
        kw = dict(
            extra_compile_args=['-O3', '-fopenmp', '-DWITH_OMP',
                                '-Wall', '-Wno-unused-variable'],
            extra_link_args=['-lgomp'],
            headers=['<omp.h>'])
    try:
        import weave
    except ImportError:
        kw = {}
    try:
        weave.inline(' ', **kw)
    except weave.build_tools.CompileError:
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
        if isinstance(__builtins__, dict):
            if name in __builtins__:
                message = 'WARNING: Glass command {:s} ({:s}) overrides previous function {:s}'
                print(message.format(name, f, __builtins__[name]))
            __builtins__[name] = g
        else:
            if name in __builtins__.__dict__:
                message = 'WARNING: Glass command {:s} ({:s}) overrides previous function {:s}'
                print(message.format(name, f, __builtins__.__dict__[name]))
            __builtins__.__dict__[name] = g


def glass_renv(**kwargs):
    """
    Call to set up a standard glass 'read' environment

    Args:
        None

    Kwargs:
        threads <int> - run with given number of threads
        no_window <bool> - suppress any graphics

    Return:
        glass <module> - the glass environment as a module
    """
    Environment.global_opts['ncpus_detected'] = _detect_cpus()
    Environment.global_opts['ncpus'] = 1
    Environment.global_opts['omp_opts'] = _detect_omp()
    Environment.global_opts['withgfx'] = True
    Commands.set_env(Environment())
    if 'threads' in kwargs and kwargs['threads'] >= 1:
        Environment.global_opts['ncpus'] = kwargs['threads']
    if 'no_window' in kwargs:
        Environment.global_opts['withgfx'] = kwargs['no_window']
    import glass.glcmds
    import glass.scales
    if Environment.global_opts['withgfx']:
        import glass.plots
    glass_basis('glass.basis.pixels', solver=None)
    glass.basis.pixels.priors.exclude_all_priors()
    return glass


def run(*args, **kwargs):
    """
    Call to run a GLASS config file within a standard glass 'write' environment

    Args:
        args <str/tuple(str)> - input files for GLASS, e.g. '../src/glass/Examples/B1115.gls'

    Kwargs:
        threads <int> - run with given number of threads
        no_window <bool> - suppress any graphics

    Return:
        None
    """
    Environment.global_opts['ncpus_detected'] = _detect_cpus()
    Environment.global_opts['ncpus'] = 1
    Environment.global_opts['omp_opts'] = _detect_omp()
    Environment.global_opts['withgfx'] = True
    Commands.set_env(Environment())
    if 'thread' in kwargs and kwargs['threads'] > 1:
        Environment.global_opts['ncpus'] = kwargs['threads']
    if 'no_window' in kwargs:
        Environment.global_opts['withgfx'] = kwargs['no_window']

    import glass.glcmds
    import glass.scales
    if Environment.global_opts['withgfx']:
        import glass.plots

    with open(args[0], 'r') as f:
        Commands.get_env().input_file = f.read()

    Environment.global_opts['argv'] = args

    try:
        execfile(args[0])
    except GLInputError as e:
        raise e("An error occurred! Figure out whats wrong and give it another go...")


def parse_arguments():
    """
    Parse command line arguments
    """
    from argparse import ArgumentParser, RawTextHelpFormatter
    parser = ArgumentParser(description=__doc__, formatter_class=RawTextHelpFormatter)
    # main args
    parser.add_argument("case", nargs='+',
                        help="Path input to .gls file for running GLASS through glass_interface",
                        default=os.path.join(
                            os.path.abspath(os.path.dirname(os.path.dirname(__file__))),
                            'src', 'glass', 'Examples', 'B1115.gls'))

    # mode args
    parser.add_argument("-t", "--threads", dest="threads", metavar="<threads>", type=int,
                        help="Run GLASS with the given number of threads",
                        default=1)
    parser.add_argument("--nw", "--no-window", dest="no_window", action="store_true",
                        help="Run GLASS in no-window mode",
                        default=False)
    parser.add_argument("-v", "--verbose", dest="verbose", action="store_true",
                        help="Run program in verbose mode",
                        default=False)
    parser.add_argument("--test", "--test-mode", dest="test_mode", action="store_true",
                        help="Run program in testing mode",
                        default=False)

    args = parser.parse_args()
    case = args.case
    delattr(args, 'case')
    return parser, case, args


if __name__ == "__main__":
    parser, case, args = parse_arguments()
    testdir = os.path.join(
        os.path.abspath(os.path.dirname(os.path.dirname(__file__))),
        'src', 'glass', 'Examples')
    no_input = len(sys.argv) <= 1 and testdir in case[0]
    if no_input:
        parser.print_help()
    elif args.test_mode:
        sys.argv = sys.argv[:1]
        # from gleam.test.test_glass_interface import TestGLASSInterface
        # TestGLASSInterface.main()
    else:
        run(*case, **args.__dict__)
