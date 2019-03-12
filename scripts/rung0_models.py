import sys
import os
# import __init__

# Add root to PYTHONPATH
root = os.path.dirname(os.path.abspath(os.path.realpath(__file__)))
# if os.path.exists(root) and root not in sys.path:
#     sys.path.insert(2, root)
#     # print("Adding {} to PYTHONPATH".format(root))

# Add glass to PYTHONPATH
libspath = os.path.join(root, 'lib')
if os.path.exists(libspath):
    libs = os.listdir(libspath)[::-1]
    for l in libs:
        lib = os.path.join(libspath, l)
        if lib not in sys.path or not any(['glass' in p for p in sys.path]):
            sys.path.insert(3, lib)

from gleam.model.spemd import SPEMD
from glass.command import command, Commands
from glass.environment import env, Environment
from glass.exceptions import GLInputError

import numpy as np
import matplotlib.pyplot as plt


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


_omp_opts = None


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
    print(kwargs)
    gls.basis_options = kwargs
    f = __import__(name, globals(), locals())
    print(f, name)
    for name, [f, g, help_text] in Commands.glass_command_list.iteritems():
        print(name)
        if name in __builtins__.__dict__:
            message = 'WARNING: Glass command {:s} ({:s}) overrides previous function {:s}'
            print(message.format(name, f, __builtins__.__dict__[name]))
        __builtins__.__dict__[name] = g


@command
def arrivsurf_plot(gls, model, obj_index, **kwargs):
    obj, data = model['obj,data'][obj_index]
    if not data:
        return

    src_index = kwargs.pop('src_index', 0)
    alpha = kwargs.pop('alpha', 0.05)
    # xlabel = kwargs.pop('xlabel', '$\mathrm{arcsec}$')
    # ylabel = kwargs.pop('ylabel', '$\mathrm{arcsec}$')
    cmap = kwargs.pop('cmap', 'gnuplot2_r')

    # data
    lev = obj.basis.arrival_contour_levels(data)
    arrival_grid = obj.basis.arrival_grid(data)
    R = obj.basis.mapextent
    g = arrival_grid[obj.sources[src_index].index]

    # keyword defaults
    kwargs.setdefault('extent', [-R, R, -R, R])
    kwargs.setdefault('clevels', 50)
    kwargs.setdefault('interpolation', 'nearest')
    kwargs.setdefault('aspect', 'equal')
    kwargs.setdefault('extend', 'neither')
    kwargs.setdefault('origin', 'upper')
    kwargs.setdefault('linestyles', 'solid')
    kwargs.setdefault('linewidths', 2)
    levels = np.linspace(g.min(), g.max(), kwargs['clevels'])

    # actual plotting
    # filled contours
    plt.contourf(g, levels=levels, alpha=alpha, cmap=cmap, **kwargs)
    # contour lines
    plt.contour(g, levels=levels, alpha=1, cmap=cmap, **kwargs)
    # source contour line
    plt.contour(g, lev[src_index], colors='#31343f', **kwargs)
    plt.xlim(-R, R)
    plt.ylim(-R, R)
    # plt.xlabel(xlabel)
    # plt.ylabel(ylabel)
    plt.gca().set_aspect('equal')
    ax.axis('off')


@command
def kappa_contour(gls, model, obj_index, **kwargs):
    obj, data = model['obj,data'][obj_index]
    if not data:
        return

    subtract = kwargs.pop('subtract', 0)
    clevels = kwargs.get('clevels', 30)
    alpha = kwargs.pop('alpha', 1)

    # data
    grid = obj.basis._to_grid(data['kappa']-subtract, 1)
    R = obj.basis.mapextent

    # keyword defaults
    kwargs.setdefault('extent', [-R, R, -R, R])
    kwargs.setdefault('clevels', 30)
    kwargs.setdefault('interpolation', 'nearest')
    kwargs.setdefault('aspect', 'equal')
    kwargs.setdefault('extend', 'neither')
    kwargs.setdefault('origin', 'upper')
    vmin = kwargs.pop('vmin', None)
    vmax = kwargs.pop('vmax', None)
    if vmin is None:
        w = data['kappa'] != 0
        if not np.any(w):
            vmin = -15
            grid += 10**vmin
        else:
            vmin = np.log10(np.amin(data['kappa'][w]))
        kwargs.setdefault('vmin', vmin)
    if vmax is not None:
        kwargs.setdefault('vmax', vmax)
    kwargs.setdefault('levels', np.linspace(vmin, vmax or 1, clevels))

    # actual plotting
    grid = np.log10(grid.copy())
    plt.contour(grid, **kwargs)
    plt.contourf(grid, alpha=alpha, **kwargs)

    # # colorbar
    # rows, cols, _ = plt.gca().get_geometry()
    # x, y = plt.gcf().get_size_inches()
    # pars = plt.gcf().subplotpars
    # figH = (y*(pars.top-pars.bottom) - y*pars.hspace*(rows > 1)) / rows
    # figW = (x*(pars.right-pars.left) - x*pars.wspace*(cols > 1)) / cols
    # plt.colorbar(shrink=figW/figH)

    # labels
    plt.gca().set_aspect('equal')
    ax.axis('off')
    return grid


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

opts = ["/Users/phdenzel/tdlmc/autogen/rung0.state",
        "/Users/phdenzel/tdlmc/autogen/filtered_rung0_dq.state"]
states = [loadstate(f) for f in opts]

for g, o in zip(states, opts):
    for i in range(2):
        if i == 0:
            seed = 'seed3'
        else:
            seed = 'seed4'
        g.make_ensemble_average()
        fig, ax = plt.subplots()
        ax.grid(0)

        g.img_plot(obj_index=i, color='#00d1a4')
        g.arrivsurf_plot(g.ensemble_average, obj_index=i, clevels=50, cmap='gnuplot2_r')
        filename = o.split('/')[-1]
        plt.savefig(''.join(filename.split('.')[:-1])+'_{}_arrival'.format(seed)+'.png',
                    transparent=True)
        print(''.join(filename.split('.')[:-1])+'_{}_arrival'.format(seed)+'.png')
        # plt.show()

    for i in range(2):
        if i == 0:
            seed = 'seed3'
        else:
            seed = 'seed4'
        g.make_ensemble_average()
        fig, ax = plt.subplots()
        ax.grid(0)
        grid = g.kappa_contour(g.ensemble_average, obj_index=i, clevels=30, alpha=0.6,
                               cmap=plt.get_cmap('magma'))
        print(grid.shape)
        plt.savefig(''.join(filename.split('.')[:-1])+'_{}'.format(seed)+'.png', transparent=True)
        print(''.join(filename.split('.')[:-1])+'_{}'.format(seed)+'.png')
        # plt.show()

rung0_seed3 = {
    'Nx': 13,
    'Ny': 13,
    'theta_E': 1.161,
    'q': 0.787,
    'phi_G': 1.605,
    'gamma': 2.044
}
rung0_seed4 = {
    'Nx': 13,
    'Ny': 13,
    'theta_E': 1.19,
    'q': 0.992,
    'phi_G': 2.246,
    'gamma': 2.05
}
rung0 = {'rung0_seed3': SPEMD(**rung0_seed3),
         'rung0_seed4': SPEMD(**rung0_seed4)}
for seed in rung0:
    rung0[seed].calc_map(smooth_center=1)
    fig = rung0[seed].plot_map(log=1, colorbar=0, contours=1, show=0,
                               mask=(grid != -np.inf), vmin=-1.3, vmax=0, alpha=0.55, cmap='magma')
    plt.savefig('{}_model.png'.format(seed), transparent=True)
