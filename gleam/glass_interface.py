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
import re
import getopt
import numpy as np
# from ctypes import cdll

root = os.path.dirname(os.path.dirname(os.path.abspath(os.path.realpath(__file__))))
# libglpk = os.path.join(root, 'include/glpk/libglpk.so.0')
# if os.path.exists(libglpk):
#     glpk = cdll.LoadLibrary(libglpk)
libspath = os.path.join(root, 'lib')
if os.path.exists(libspath):
    libs = os.listdir(libspath)[::-1]
    for l in libs:
        lib = os.path.join(libspath, l)
        if lib not in sys.path or not any(['glass' in p for p in sys.path]):
            sys.path.insert(3, lib)
try:
    import glass
    from glass.command import command, Commands
    from glass.environment import env, Environment
    from glass.exceptions import GLInputError
except ImportError:
    print("Problem importing GLASS... needs fixing!")
    sys.exit(1)


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


def _detect_omp(force_gcc=False):
    """
    Detects the OpenMP options

    Args/Kwargs/Return:
        None
    """
    global _omp_opts
    # if _omp_opts is not None:
    #     return _omp_opts
    if force_gcc:
        kw = dict(
            extra_compile_args=['-O3', '-fopenmp', '-DWITH_OMP',
                                '-Wall', '-Wno-unused-variable'],
            extra_link_args=['-lgomp'],
            headers=['<omp.h>'])
    elif 'clang' in sys.version:  # probably macOS clang build
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
        print("import weave failed!")
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
        verbose <bool> - if True, warnings will be printed to stdout

    Return:
        None

    Note:
        The argument [gls <glass.Environment object> - the glass state environment]
        is passed to command and is needed in common use
    """
    verbose = kwargs.pop('verbose', False)
    gls.basis_options = kwargs
    f = __import__(name, globals(), locals())
    for name, [f, g, help_text] in Commands.glass_command_list.iteritems():
        if isinstance(__builtins__, dict):
            if name in __builtins__:
                message = 'WARNING: Glass command {:s} ({:s}) overrides previous function {:s}'
                if verbose:
                    print(message.format(name, f, __builtins__[name]))
            __builtins__[name] = g
        else:
            if name in __builtins__.__dict__:
                message = 'WARNING: Glass command {:s} ({:s}) overrides previous function {:s}'
                if verbose:
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
    try:
        _, arglist = getopt.getopt(sys.argv[1:], 't:h:f', ['nw'])
    except getopt.GetoptError:
        arglist = []
    Environment.global_opts['ncpus_detected'] = _detect_cpus()
    Environment.global_opts['ncpus'] = 1
    Environment.global_opts['omp_opts'] = _detect_omp()
    Environment.global_opts['withgfx'] = True
    Environment.global_opts['argv'] = arglist
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


def filter_env(gls, selection):
    """
    Filter a GLASS environment according to a selection

    Args:
        gls <glass.environment object> - the glass state to be filtered
        selection <list(int)> - list of indices used to filter out models

    Kwargs:
        None

    Return:
        envcpy <glass.environment object> - the filtered glass state
    """
    import copy
    envcpy = copy.deepcopy(gls)
    for i in range(len(envcpy.models)-1, -1, -1):
        if i in selection:
            continue
        del envcpy.models[i]
        del envcpy.accepted_models[i]
        del envcpy.solutions[i]
    if 'argv' in gls.global_opts:
        path = os.path.basename(gls.global_opts['argv'][-1])
    elif 'filtered' in gls.meta_info:
        path = gls.meta_info['filtered'][0]
    else:
        path = None
    envcpy.meta_info['filtered'] = (path, len(gls.models), len(envcpy.models))
    return envcpy


def export_state(gls, selection=None, name="filtered.state"):
    """
    Save a filtered state in a new state file

    Args:
        gls <glass.environment object> - state to be exported

    Kwargs:
        selection <list(int)> - list of indices used to filter out models

    Return:
        None
    """
    if selection:
        state = filter_env(gls, selection)
    else:
        state = gls
    state.savestate(name)


def state2txt(gls, **kwargs):
    """
    Export a state model to a .txt file

    Args:
        env <glass.Environment object> - loaded glass state environment

    Kwargs:
        txtdata <np.ndarray> - preset data to write to the txt file
        models <list(glass.model)> - glass input models, other than from the environment
        obj_index <list(int)> - lens model index, in case more than one objects were modelled
        src_index <int> - source model index, in case more than one sources were modelled
        prop <str> - model grid property, e.g. ['kappa'|'arrival'|'potential'|'maginv'|'veldisp']
        savename <str> - path of the exported file
    """
    statename = Environment.global_opts['argv'][0]
    # extract options
    txtdata = kwargs.pop('txtdata', [])
    models = kwargs.pop('models', gls.models)
    obj_index = kwargs.pop('obj_index', 0)
    src_index = kwargs.pop('src_index', 0)
    prop = kwargs.pop('prop', 'kappa')
    savename = kwargs.pop('savename', None)
    verbose = kwargs.pop('verbose', False)
    if isinstance(obj_index, list):
        for obji in obj_index:
            if len(models[0]['obj,data']) <= obji:
                continue
            obj, data = models[0]['obj,data'][obji]
            if isinstance(savename, list):
                savename = savename[0]
            else:
                savename = os.path.join(os.path.dirname(statename),
                                        "{:s}_{}.txt".format(obj.name, prop))
            if verbose:
                print("Object {:2d}: {:s}".format(obji, obj.name))
            txtdta = 1*txtdata
            dta, hdr = model2txt(txtdata=txtdta, models=models, prop=prop,
                                 obj_index=obji, src_index=src_index,
                                 savename=savename)
            if verbose:
                print(hdr)
                print("Saving as {:s}...".format(savename))
            # write to numpy txt file
            np.savetxt(savename, dta, header=hdr)
    elif isinstance(obj_index, int):
        obj, data = models[0]['obj,data'][obj_index]
        savename = savename if savename is not None else os.path.join(
            os.path.dirname(statename), "{:s}_{}.txt".format(obj.name, prop))
        if verbose:
            print("Object {:2d}: {:s}".format(obj_index, obj.name))
        dta, hdr = model2txt(txtdata=txtdata, models=models, prop=prop,
                             obj_index=obj_index, src_index=src_index,
                             savename=savename)
        if verbose:
            print(hdr)
            print("Saving as {:s}...".format(savename))
        # write to numpy txt file
        np.savetxt(savename, dta, header=hdr)


def model2txt(**kwargs):
    """
    Write ensemble models to a np.ndarray and a header string

    Args:
        None

    Kwargs:
        txtdata <np.ndarray> - preset data to write to the txt file
        models <list(glass.model)> - glass input models, other than from the environment
        obj_index <list(int)> - lens model index, in case more than one objects were modelled
        src_index <int> - source model index, in case more than one sources were modelled
        prop <str> - model grid property, e.g. ['kappa'|'arrival'|'potential'|'maginv'|'veldisp']

    Return:
        dta <np.ndarray> - data ready to be written to a txt file
        hdr <str> - header with data info
    """
    from gleam.utils.lensing import DLSDS, dispersion_profile, roche_potential
    txtdata = kwargs.pop('txtdata', [])
    obj_index = kwargs.pop('obj_index', 0)
    src_index = kwargs.pop('src_index', 0)
    models = kwargs.pop('models', [])
    prop = kwargs.pop('prop', 'kappa')
    savename = os.path.basename(kwargs.pop('savename', ""))
    if not txtdata:
        for m in models:
            obj, data = m['obj,data'][obj_index]
            if prop == 'kappa':
                pmap = obj.basis.kappa_grid(data)
                comment = ["", "", ""]
                correction = DLSDS(obj.z, obj.sources[0].z)
            elif prop == 'potential':
                pmap = obj.basis.potential_grid(data)
                comment = ["", "", ""]
                correction = 1
            elif prop == 'arrival':
                pmap = obj.basis.arrival_grid(data)[src_index]
                comment = ["", "", ""]
                correction = 1
            elif prop == 'maginv':
                pmap = obj.basis.maginv_grid(data)[src_index]
                comment = ["", "", ""]
                correction = 1
            elif prop == 'roche':
                gx, gy, pmap = roche_potential(m, obj_index=obj_index,
                                               src_index=src_index)
                comment = ["", "", ""]
                correction = 1
            elif prop == 'veldisp':
                radii, profile = dispersion_profile(m, obj_index=obj_index)
                pmap = np.stack((radii, profile), axis=0)
                # comment will be appended to the header
                comment = [">>> R, dispersion = data[0, :, :]  "
                           + "# e.g. the first model's velocity dispersion",
                           "R [light-seconds], dispersion [c]"]
                correction = 1
            pmap = correction*pmap if correction != 1 else pmap
            txtdata.append(1*pmap)
    if isinstance(txtdata, list):
        txtdata = np.stack(txtdata, axis=0)
    header = "\n".join(
        ("{}".format(savename), "Usage:",
         ">>> import numpy",
         ">>> N_models, N, L = {}, {}, {}".format(txtdata.shape[0],
                                                  txtdata.shape[1],
                                                  txtdata.shape[2]),
         ">>> data = numpy.loadtxt('{}')".format(savename),
         ">>> data = data.reshape(N_models, N, L)"))
    header = "\n".join([header, "\n".join(comment)])
    txtdata = txtdata.reshape((txtdata.size))
    # txtdata = txtdata.reshape((txtdata.shape[0]*txtdata.shape[1], txtdata.shape[2]))
    return txtdata, header


def state2fits(gls, **kwargs):
    """
    Export a state model to a .fits file

    Args:
        env <glass.Environment object> - loaded glass state environment

    Kwargs:
        fitsdata <np.ndarray> - preset data to write to the fits file
        models <list(glass.model)> - glass input models, other than from the environment
        obj_index <list(int)> - lens model index, in case more than one objects were modelled
        src_index <int> - source model index, in case more than one sources were modelled
        prop <str> - model grid property, e.g. ['kappa'|'arrival'|'potential'|'maginv']
        savename <str> - path of the exported file

    Return:
        None
    """
    statename = Environment.global_opts['argv'][0]
    # extract options
    fitsdata = kwargs.pop('fitsdata', [])
    models = kwargs.pop('models', gls.models)
    obj_index = kwargs.pop('obj_index', 0)
    src_index = kwargs.pop('src_index', 0)
    prop = kwargs.pop('prop', 'kappa')
    savename = kwargs.pop('savename', None)
    verbose = kwargs.pop('verbose', False)
    if isinstance(obj_index, list):
        for obji in obj_index:
            if len(models[0]['obj,data']) <= obji:
                continue
            obj, data = models[0]['obj,data'][obji]
            if isinstance(savename, list):
                savename = savename[0]
            else:
                savename = os.path.join(os.path.dirname(statename),
                                        "{:s}_{}.fits".format(obj.name, prop))
            if verbose:
                print("Object {:2d}: {:s}".format(obji, obj.name))
            fitsd = 1*fitsdata
            hdu = model2hdu(fitsdata=fitsd, models=models, prop=prop,
                            obj_index=obji, src_index=src_index,
                            verbose=verbose)
            if verbose:
                print(hdu.header.tostring(sep='\n'))
                print("Saving as {:s}...".format(savename))
            hdu.writeto(savename)
    elif isinstance(obj_index, int):
        obj, data = models[0]['obj,data'][obj_index]
        savename = savename if savename is not None else os.path.join(
            os.path.dirname(statename), "{:s}_{}.fits".format(obj.name, prop))
        if verbose:
            print("Object {:2d}: {:s}".format(obj_index, obj.name))
        hdu = model2hdu(fitsdata=fitsdata, models=models, prop=prop,
                        obj_index=obj_index, src_index=src_index,
                        verbose=verbose)
        if verbose:
            print(hdu.header.tostring(sep='\n'))
            print("Saving as {:s}...".format(savename))
        hdu.writeto(savename)


def model2hdu(**kwargs):
    """
    Write ensemble models to an astropy.fits.PrimaryHDU data structure

    Args:
        None

    Kwargs:
        fitsdata <np.ndarray> - preset data to write to the fits file
        models <list(glass.model)> - glass input models, other than from the environment
        obj_index <list(int)> - lens model index, in case more than one objects were modelled
        src_index <int> - source model index, in case more than one sources were modelled
        prop <str> - model grid property, e.g. ['kappa', 'arrival', 'potential', 'maginv']

    Return:
        hdu <astropy.fits.PrimaryHDU object> - data ready to be written to a fits file
    """
    from astropy import wcs
    from astropy.io import fits
    from gleam.utils.lensing import DLSDS, dispersion_profile, roche_potential
    fitsdata = kwargs.pop('fitsdata', [])
    obj_index = kwargs.pop('obj_index', 0)
    src_index = kwargs.pop('src_index', 0)
    models = kwargs.pop('models', [])
    prop = kwargs.pop('prop', 'kappa')
    verbose = kwargs.pop('verbose', False)

    N_models = len(models)
    pix2deg_alt = None
    if not fitsdata:
        for i, m in enumerate(models):
            if verbose:
                message = "{:4d} / {:4d}\r".format(i+1, N_models)
                sys.stdout.write(message)
                sys.stdout.flush()
            obj, data = m['obj,data'][obj_index]
            if prop == 'kappa':
                pmap = obj.basis.kappa_grid(data)
                correction = DLSDS(obj.z, obj.sources[0].z)
            elif prop == 'potential':
                pmap = obj.basis.potential_grid(data)
                correction = 1
            elif prop == 'arrival':
                pmap = obj.basis.arrival_grid(data)[src_index]
                correction = 1
            elif prop == 'maginv':
                pmap = obj.basis.maginv_grid(data)[src_index]
                correction = 1
            elif prop == 'roche':
                gx, gy, pmap = roche_potential(m, obj_index=obj_index,
                                               src_index=src_index)
                pix2deg_alt = [gx[-1, 0]/(pmap.shape[1]//2)/3600.,
                               gy[-1, 0]/(pmap.shape[0]//2)/3600.]
                correction = 1
            pmap = correction*pmap if correction != 1 else pmap
            fitsdata.append(1*pmap)
    if isinstance(fitsdata, list):
        fitsdata = np.stack(fitsdata, axis=0)
    # set the wcs information
    mapL = 2*obj.basis.mapextent
    pix2deg = [direct*mapL/s/3600 for direct, s in zip([-1, 1], fitsdata.shape[1:])] \
        if pix2deg_alt is None else pix2deg_alt
    w = wcs.WCS(naxis=len(fitsdata.shape)-1)
    w.wcs.ctype = ['deg', 'deg']
    w.wcs.crpix = [d//2 for d in fitsdata.shape[1:]]
    w.wcs.crval = [0, 0]
    w.wcs.cd = np.array([[pix2deg[0], 0], [0, pix2deg[1]]])
    w.wcs.crota = np.array([0, 0])
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    wdict = wcs2hdrdict(w.wcs)
    # build header
    hdu = fits.PrimaryHDU(fitsdata)
    hdu.header.update(wdict)
    hdu.header['COMMENT'] = "The ensemble model's {} grids; {} models along NAXIS3".format(prop, len(models))
    hdu.header['COMMENT'] = "FITS (Flexible Image Transport System) format is defined in 'Astronomy  " \
                            "and Astrophysics', volume 376, page 359; bibcode: 2001A&A...376..359H"
    return hdu


def wcs2hdrdict(wcs, hdr_defaults={}):
    """
    Convert astropy.wcs object into a fits header dictionary

    Note:
        wcs.to_header method annoyingly removes cd_matrix and reset cdelt attributes,
        which is why this functions exists
    """
    import collections
    hdr = collections.OrderedDict()
    hdr['WCSAXES'] = (2, 'Number of World Coordinate System axes')
    hdr['CRPIX1'] = 'Reference pixel for NAXIS1'
    hdr['CRPIX2'] = 'Reference pixel for NAXIS2'
    hdr['CRVAL1'] = '[deg] Coordinate value at reference point'
    hdr['CRVAL2'] = '[deg] Coordinate value at reference point'
    hdr['CTYPE1'] = 'Projection for NAXIS1'
    hdr['CTYPE2'] = 'Projection for NAXIS2'
    hdr['CUNIT1'] = 'Units of coordinate increment and value'
    hdr['CUNIT2'] = 'Units of coordinate increment and value'
    hdr['CROTA2'] = None
    hdr['CD1_1'] = 'Linear projection matrix element'
    hdr['CD1_2'] = 'Linear projection matrix element'
    hdr['CD2_1'] = 'Linear projection matrix element'
    hdr['CD2_2'] = 'Linear projection matrix element'
    hdr.update(hdr_defaults)
    wcs.to_header()
    for k in hdr.keys():
        key = ''.join([ki for ki in k if not ki.isdigit()]).replace('_', '')
        idx = [int(i)-1 for i in re.sub("\D", "", k)]
        if hasattr(wcs, key.lower()):
            prop = wcs.__getattribute__(key.lower())
            if hasattr(prop, '__len__'):
                for i in idx:
                    prop = prop[i]
            if isinstance(hdr[k], (tuple, list)) and len(hdr[k]) == 2:
                continue
            if isinstance(prop, (int, float)):
                hdr[k] = (prop, hdr[k])
            else:
                hdr[k] = (str(prop), hdr[k])
    return hdr


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
