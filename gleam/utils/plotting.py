#!/usr/bin/env python
"""
@author: phdenzel

Lensing utility functions for calculating various analytics from lensing observables
"""
###############################################################################
# Imports
###############################################################################
import os
import numpy as np
import scipy.ndimage as ndimage
from scipy.optimize import curve_fit
from matplotlib import patches
from matplotlib import pyplot as plt
from matplotlib.colors import to_rgba
from gleam.utils.lensing import LensModel
from gleam.utils.lensing import DLSDS, kappa_profile, \
    interpolate_profile, find_einstein_radius, find_saddles
from gleam.utils.pdfs import lorentzian, gaussian, pseudovoigt, tukeygh
from gleam.utils.colors import GLEAMcolors, GLEAMcmaps
from gleam.utils.rgb_map import radial_mask
from gleam.utils import units as glu
from gleam.glass_interface import glass_renv
glass = glass_renv()
GLEAMcmaps.register_all()

###############################################################################
def plot_connection(graph, **kwargs):
    """
    Plot a connected graph structure

    Args:
        graph <list(np.ndarray, list)> - list of roots and leaves

    Kwargs:
        TODO
    """
    # defaults
    kwargs.setdefault('marker', 'o')
    kwargs.setdefault('markersize', 4)
    kwargs.setdefault('ls', '-')
    kwargs.setdefault('lw', 1)
    kwargs.setdefault('color', GLEAMcolors.red)
    kwargs.setdefault('label', None)
    kwargs.setdefault('alpha', 1)
    if len(graph) != 2:
        return
    r, leaves = graph
    if isinstance(leaves, (tuple, list)):
        for l in leaves:
            return plot_connection(l, **kwargs)
    if isinstance(leaves, np.ndarray):
        if len(leaves.shape) == 1:
            # plt.plot(*leaves, **kwargs)
            tree, = plt.plot([r[0], leaves[0]], [r[1], leaves[1]], **kwargs)
            return tree
        else:
            for l in leaves.T:
                plt.plot([r[0], l[0]], [r[1], l[1]], **kwargs)


def get_location(position='bottom left'):
    """
    Translate specifically defined corner locations on the image plot

    Args:
        None

    Kwargs:
        position <int/str> - corner location in binary [bottom(0)|top(1) left(0)|right(1)]

    Return:
        loc <tuple/list> - relative position of location index according to asii sketch below

    Note:
       _______
       |2   3|
       |  ?  |
       |0   1|
       -------
    """
    binpos = '00'
    if isinstance(position, str):
        if position == 'center':
            binpos = '100'
        else:
            if position.isdigit():
                binpos = position
            else:
                topbottom, leftright = position.split(' ')
                if topbottom == 'top':
                    binpos = '1' + binpos[1]
                if leftright == 'right':
                    binpos = binpos[0] + '1'
        position = int(binpos, 2)
    if position == 0:
        relx, rely = 0., 0.
    elif position == 1:
        relx, rely = 1., 0.
    elif position == 2:
        relx, rely = 0., 1.
    elif position == 3:
        relx, rely = 1., 1.
    else:
        relx, rely = 0.5, 0.5
    return relx, rely


def square_subplots(fig):
    """
    Square all subplots of a figure

    Args:
        fig <matplotlib.figure.Figure object> - a figure which contains the subplots

    Kwargs/Return:
        None
    """
    ax = fig.axes[0]
    rows, cols = ax.get_subplotspec().get_gridspec().get_geometry()
    l = fig.subplotpars.left
    r = fig.subplotpars.right
    t = fig.subplotpars.top
    b = fig.subplotpars.bottom
    wspace = fig.subplotpars.wspace
    hspace = fig.subplotpars.hspace
    figw, figh = fig.get_size_inches()
    axw = figw*(r-l)/(cols+(cols-1)*wspace)
    axh = figh*(t-b)/(rows+(rows-1)*hspace)
    axs = min(axw, axh)
    w = (1-axs/figw*(cols+(cols-1)*wspace))/2.
    h = (1-axs/figh*(rows+(rows-1)*hspace))/2.
    fig.subplots_adjust(bottom=h, top=1-h, left=w, right=1-w)


def plot_scalebar(R, length=None, unit=r'$^{\prime\prime}$',
                  position='bottom left', origin='center',
                  padding=(0.08, 0.06), barheight=0.03,
                  length_scale=1.,
                  color='white', fontsize=16):
    """
    Add a scalebar to an image plot

    Args:
        R <float> - image scale radius

    Kwargs:
        length <float> - length of the scalebar in the image scale
        unit <str> - unit of the scalebar
        position <int/str> - location of the scalebar [bottom(0)|top(1) left(0)|right(1)]
        origin <int/str> - position of the origin
        padding <tuple/list(float)> - padding relative to the scale radius
        color <str> - text and scalebar color

    Return:
        TODO
    """
    alpha = 0.85
    if color in ['k', 'black', 'grey']:
        alpha = 0.5
    if length is None:
        length = 1.0
    if length // 1 == length:
        lbl = r"{:1.0f}{}".format(length, unit)
    else:
        lbl = r"{:1.1f}{}".format(length, unit)
    wh = np.asarray([length_scale*length, barheight*R])
    w, h = wh
    # positioning scalebar
    position_xy = np.asarray(get_location(position))
    origin_xy = np.asarray(get_location(origin))
    shift_xy = np.asarray(padding)*(origin_xy - position_xy) - 0.5/R*position_xy*wh
    barpos = 2*(position_xy-origin_xy+shift_xy)*R
    rect = patches.Rectangle(barpos, wh[0], wh[1],
                             facecolor=color, edgecolor=color, alpha=alpha,
                             zorder=1000)
    ax = plt.gca()
    ax.add_patch(rect)
    # add scalebar label
    ha = 'left' if position_xy[0] < 1 else 'right'
    va = 'bottom' if position_xy[1] < 1 else 'top'
    updown = 1 if position_xy[1] < 1 else -1
    textshift = np.asarray([wh[0]*(abs(position_xy[0]*8 - 1)/8), 1.5*wh[1]*updown])
    textpos = barpos + textshift
    ax.text(textpos[0], textpos[1], lbl,
            color=color, fontsize=fontsize, va=va, ha=ha, zorder=1000)
    plt.gca().set_aspect('equal')


def plot_labelbox(label, position='bottom left', padding=(0.05, 0.05), color='white',
                  **kwargs):
    """
    Add a box with a label to an image plot

    Args:
        label <str> - label to be added

    Kwargs:
        position <int/str> - location of the label [bottom(0)|top(1) left(0)|right(1)]
        padding <tuple/list(float)> - padding as relative shift
        color <str> - text and labelbox color
        **kwargs <dict> - keyword parameters of matplotlib.text.Text

    Return:
        TODO
    """
    facecolor = color
    facealpha = 0.35
    if color in ['k', 'black', 'grey']:
        facecolor = 'grey'
        facealpha = 0.05
    # defaults
    kwargs.setdefault('family', 'sans-serif')
    kwargs.setdefault('fontname', 'Computer Modern')
    kwargs.setdefault('fontsize', 14)
    kwargs.setdefault('zorder', 1000)
    kwargs.setdefault('bbox', {})
    kwargs['bbox'].setdefault('boxstyle', 'round,pad=0.3')
    kwargs['bbox'].setdefault('facecolor', facecolor)
    kwargs['bbox'].setdefault('edgecolor', None)
    kwargs['bbox'].setdefault('alpha', facealpha)
    ax = plt.gca()
    # positioning labelbox
    position_xy = np.asarray(get_location(position))
    ha = 'left' if position_xy[0] < 1 else 'right'
    va = 'bottom' if position_xy[1] < 1 else 'top'
    leftright = 1 if position_xy[0] < 1 else -1
    updown = 1 if position_xy[1] < 1 else -1
    xy = position_xy + np.array([padding[0]*leftright, padding[1]*updown])
    ax.text(xy[0], xy[1], label, va=va, ha=ha, transform=ax.transAxes,
            color=color, **kwargs)


def plot_annulus(center=(0.5, 0.5), radius=0.5, color='black', **kwargs):
    """
    Add an annotation circle for marking special radii, e.g. image distances

    Args:
        None

    Kwargs:
        center <tuple/list> - center of the annotation circle (transformed coordinates)
        radius <float> - radius of the annotation circle (in transformed units)
        color <str> - text and labelbox color
        **kwargs <dict> - keyword parameters of matplotlib.patches.Circle

    Return:
        c <mpl.patches.Circle object> - the Circle object plotted
    """
    ax = plt.gca()
    kwargs.setdefault('fill', False)
    kwargs.setdefault('facecolor', color)
    kwargs.setdefault('edgecolor', color)
    kwargs.setdefault('alpha', 1)
    kwargs.setdefault('lw', 1)
    kwargs.setdefault('transform', ax.transAxes)
    kwargs.setdefault('zorder', 1000)
    c = plt.Circle(center, radius, **kwargs)
    ax.add_patch(c)
    return c


def plot_annulus_region(center=(0.5, 0.5), radius=0.5, color='white', alpha=0.2, refine=10,
                        **kwargs):
    """
    Add an annotation circle mask for marking special radial regions, e.g. critical densities

    Args:
        None

    Kwargs:
        center <tuple/list> - center of the annotation circle
        radius <float> - radius of the annotation region in normalized units
        alpha <float> - alpha of the region outside the radius
        **kwargs <dict> - keyword parameters of matplotlib.text.Text

    Return:
        TODO
    """
    ax = plt.gca()
    img = plt.gci()
    shape = kwargs.pop('shape', None)
    if shape is None:
        shape = img._A.shape
    if hasattr(img, '_extent'):
        extent = img._extent
    elif hasattr(img, 'extent'):
        extent = img.extent
    else:
        return
    overlay = np.ones((shape[0]*refine, shape[1]*refine))
    msk = radial_mask(overlay, center=center, radius=int(radius*shape[0]*refine))
    cval = to_rgba(color, alpha=alpha)
    overlay = cval * np.ones((shape[0]*refine, shape[1]*refine)+(4,))
    overlay[msk] = 0
    ax.imshow(overlay, extent=extent)
    return overlay


def kappa_map_transf(model, mdl_index=-1, obj_index=0, src_index=0, subcells=1, extent=None,
                     levels=3, delta=0.2, log=True, oversample=True, shift=1e-6, factor=1.):
    """
    Transform data in preparation for plotting

    Args:
        model <LensModel object> - ensemble model as a LensModel object, or tuple of arrays

    Kwargs:
        mdl_index <int> - model index of the LensModel object
        obj_index <int> - object index of the LensModel object
        src_index <int> - source index of the LensModel object
        subcells <int> - number of subcells
        extent <tuple/list> - map extent
        levels <int> - half-number of contours; total number w/ log is 2*N+1
        delta <float> - contour distances
        log <bool> - plot in log scale
        oversample <bool> - oversample the map to show characteristic pixel structure
        shift <float> - shift the grid data by some amount

    Return:
        grid <np.ndarray> - kappa grid oversampled/transformed/scaled according to input settings
        grid_info <tuple> - additional info on grid units, optimal contour levels, and scale
    """
    if isinstance(model, (tuple, list)):
        x, y, grid = model
        R = np.max(np.abs(np.concatenate([x, y])))*(1. + 1./(grid.shape[-1]-1))
        model = LensModel(grid, obj_index=obj_index, src_index=src_index, maprad=R)
    if not isinstance(model, LensModel):
        model = LensModel(model, obj_index=obj_index, src_index=src_index)
    else:
        model.obj_idx = min(obj_index, model.N_obj-1) if hasattr(model, 'N_obj') else obj_index
    grid = model.kappa_grid(model_index=mdl_index, refined=(subcells > 1))
    grid = grid[:] * factor
    grid = grid[:] + shift
    extent = model.extent if extent is None else extent
    # masking if necessary
    msk = grid != 0
    if not np.any(msk):
        vmin = -15
        grid += 10**vmin
    else:
        vmin = np.log10(np.amin(grid[msk]))
    # interpolate
    zoomlvl = 3
    order = 0
    if oversample:
        grid = ndimage.zoom(grid, zoomlvl, order=order)
    grid[grid <= 10**vmin] = 10**vmin
    # contour levels
    clev2 = np.arange(delta, levels*delta, delta)
    clevels = np.concatenate((-clev2[::-1], (0,), clev2))
    if log:
        kappa1 = 0.
        grid = np.log10(grid)
        grid[grid <= clevels[0]] = clevels[0]
        grid[np.isnan(grid)] = np.nanmin(grid)
    else:
        kappa1 = 1.
        clevels = np.concatenate(((0,), 10**clev2))
    return grid, (kappa1, clevels, extent)


def kappa_map_plot(model, mdl_index=-1, obj_index=0, src_index=0, subcells=1,
                   extent=None, origin='upper', shift=0,
                   contours=False, levels=7, delta=0.1, log=True,
                   oversample=True, factor=1.,
                   scalebar=False, label=None, color='white',
                   cmap=GLEAMcmaps.agaveglitch, colorbar=False):
    """
    Plot the convergence map of a lens model with auto-adjusted contour levels

    Args:
        model <LensModel object> - ensemble model as a LensModel object, or tuple of arrays

    Kwargs:
        mdl_index <int> - model index of the LensModel object
        obj_index <int> - object index of the LensModel object
        src_index <int> - source index of the LensModel object
        subcells <int> - number of subcells
        extent <tuple/list> - map extent
        origin <str> - matplotlib.pyplot origin keyword
        contours <bool> - plot contours onto map
        levels <int> - half-number of contours; total number w/ log is 2*N+1
        delta <float> - contour distances
        log <bool> - plot in log scale
        oversample <bool> - oversample the map to show characteristic pixel structure
        scalebar <bool> - add an arcsec scalebar to bottom-left corner instead of axis ticks
        label <str> - add a labelbox to top-left corner
        color <str> - color of labels and annotations
        cmap <str/mpl.cm.ColorMap object> - color map for plotting
        colorbar <bool> - plot colorbar next to convergence map

    Return:
        None
    """
    grid, (kappa1, clevels, extent) = kappa_map_transf(
        model, mdl_index=mdl_index, obj_index=obj_index, src_index=src_index,
        shift=shift, subcells=subcells, extent=extent, levels=levels, log=log,
        delta=delta, oversample=oversample, factor=factor)
    R = max(extent)
    # contours and surface plot
    if contours:
        plt.contour(grid, levels=(kappa1,), colors=['k'],
                    extent=extent, origin=origin)
        plt.contourf(grid, cmap=cmap, antialiased=True,
                     extent=extent, origin=origin, levels=clevels)
        plt.gca().set_aspect('equal')
    else:
        plt.imshow(grid, cmap=cmap, extent=extent, origin=origin)
    if colorbar:
        cbar = plt.colorbar()
        if log:
            lvllbls = ['{:2.1f}'.format(l) if i > 0 else '0'
                       for (i, l) in enumerate(10**cbar._tick_data_values)]
            cbar.ax.set_yticklabels(lvllbls)
    # annotations
    if scalebar:
        plot_scalebar(R, length=1., position='bottom left', origin='center', color=color)
        plt.axis('off')
        plt.gcf().axes[0].get_xaxis().set_visible(False)
        plt.gcf().axes[0].get_yaxis().set_visible(False)
    if label is not None:
        plot_labelbox(label, position='top left', padding=(0.03, 0.03), color=color)


def potential_plot(model, N=None,
                   mdl_index=-1, obj_index=0, src_index=0,
                   log=False, zero_level='center', norm_level='min',
                   contours=True, contours_only=False,
                   levels=20, cmin=np.nanmin, cmax=np.nanmax,
                   draw_images=False, background=None, color='white',
                   cmap=GLEAMcmaps.phoenix,
                   clabels=False, scalebar=False, label=None, colorbar=False,
                   **kwargs):
    """
    Plot the potential map of a lens model with auto-adjusted contour levels

    Args:
        model <LensModel object> - ensemble model as a LensModel object, or tuple of arrays

    Kwargs:
        N <int> - model's grid size in pixels
        mdl_index <int> - model index of the LensModel object
        obj_index <int> - object index of the LensModel object
        src_index <int> - source index of the LensModel object
        log <bool> - plot model in log scale
        zero_level <str> - zero level location, i.e. location to be  0 ['center', 'min', 'max']
        norm_level <str> - norm level location, i.e. location to be -1 ['center', 'min', 'max']
        contours <bool> - plot contours onto map
        contours_only <bool> - only plot contours
        levels <int> - number of contours
        cmin <func> - function determining contour minimum (must accept a single argument)
        cmax <func> - function determining contour maximum (must accept a single argument)
        draw_images <bool> - do/do not plot image
        background <float> - greyscale color for background
        color <str> - color of labels and annotations
        cmap <str/mpl.cm.ColorMap object> - color map for plotting
        clabels <bool> - add labels to the contours
        scalebar <bool> - add an arcsec scalebar to bottom-left corner instead of axis ticks
        label <str> - add a labelbox to top-left corner
        colorbar <bool> - add colorbar next to map

    Return:
        None
    """
    if isinstance(model, LensModel):
        model.obj_idx = min(obj_index, model.N_obj-1)
        N = max(model.data.shape[-1], 36) if N is None else N
        x, y = model.xy_grid(N=N)
        grid = model.potential_grid(model_index=mdl_index, N=N)
    elif isinstance(model, (tuple, list)):
        x, y, grid = model
        R = np.max(np.abs(np.concatenate([x, y])))*(1. + 1./(grid.shape[-1]-1))
        model = LensModel(grid, maprad=R)
    elif isinstance(model, np.ndarray):
        x = y = np.asarray([0, 1])
        grid = model
        model = LensModel(grid, maprad=1)
    else:
        pass
    # get zero level and normalization coordinates
    if zero_level == norm_level:
        norm_level = [e for e in ['max', 'center', 'min'] if e != zero_level][-1]
    if zero_level is None:
        def identity(grid): return 1
        level_shift = identity
    elif zero_level == 'min':
        def mival(grid): return np.nanmin(grid)
        level_shift = mival
    elif zero_level == 'max':
        def maval(grid): return np.nanmax(grid)
        level_shift = maval
    else:  # == 'center'
        def cval(grid): return grid[grid.shape[0]//2, grid.shape[1]//2]
        level_shift = cval
    if norm_level is None:
        def identity(grid): return 1
        norm = identity
    elif norm_level == 'center':
        def cval(grid): return grid[grid.shape[0]//2, grid.shape[1]//2]
        norm = cval
    elif norm_level == 'max':
        def maval(grid): return np.nanmax(grid)
        norm = maval
    else:  # == 'min'
        def mival(grid): return -np.nanmin(grid)
        norm = mival
    # add images
    minima = model.minima
    saddles = model.saddle_points
    maxima = model.maxima
    cmap = plt.get_cmap(cmap) if isinstance(cmap, str) else cmap
    min_clr, sad_clr, max_clr = cmap(.8), cmap(.5), cmap(.2)
    draw_images = 'min, sad, max' if draw_images else ''
    if 'min' in draw_images and len(minima) > 0:
        plt.plot(minima.T[0], minima.T[1], color=min_clr, marker='o', lw=0)
    if 'sad' in draw_images and len(saddles) > 0:
        plt.plot(saddles.T[0], saddles.T[1], color=sad_clr, marker='o', lw=0)
    if 'max' in draw_images and len(maxima) > 0:
        plt.plot(maxima.T[0], maxima.T[1], color=max_clr, marker='o', lw=0)
    if 'center' in draw_images or 'origin' in draw_images:
        plt.plot(0, 0, color=max_clr, marker='o', lw=0)
    # adjust potential level and normalization
    grid = grid - level_shift(grid)
    grid = grid / norm(grid)
    # contour levels
    if log:
        mi, ma = cmin(grid), cmax(grid)
        clevels = np.logspace(np.log10(1), np.log10(1+abs(ma-mi)), levels)
        clevels = clevels - 1 + mi
    else:
        clevels = np.linspace(cmin(grid), cmax(grid), levels)
    if background is not None:
        bg = np.ones(x.shape+(4,)) * to_rgba(background, alpha=0.75)
        plt.imshow(bg, extent=model.extent)
    if contours or contours_only:
        kwargs.setdefault('origin', 'upper')
        kwargs.setdefault('extent', model.extent)
        lw = kwargs.pop('linewidths', 1.25)
        cs = plt.contour(grid, levels=clevels, cmap=cmap,
                         extent=kwargs.get('extent'), origin=kwargs.get('origin'), linewidths=lw)
    if not contours_only:
        kwargs.setdefault('origin', 'upper')
        kwargs.setdefault('extent', model.extent)
        kwargs.setdefault('alpha', 0.2 if contours else 0.8)
        plt.contourf(grid, levels=clevels, cmap=cmap, **kwargs)
    # annotations and amendments
    if colorbar:
        cbar = plt.colorbar()
        cbar.set_alpha(1)
    if clabels and (contours or contours_only):
        plt.clabel(cs, cs.levels[-1:], fmt='%2.1f', inline=True,
                   alpha=0.75, fontsize=10)
        plt.clabel(cs, cs.levels[:1], fmt='%2.1f', inline=True,
                   alpha=0.75, fontsize=10)
    if scalebar:
        plot_scalebar(x.max(), length=1., position='bottom left', origin='center', color=color)
        plt.axis('off')
        plt.gcf().axes[0].get_xaxis().set_visible(False)
        plt.gcf().axes[0].get_yaxis().set_visible(False)
    if label is not None:
        plot_labelbox(label, position='top right', padding=(0.03, 0.03), color=color)
    plt.gca().set_aspect('equal')


def roche_potential_plot(model, N=None,
                         mdl_index=-1, obj_index=0, src_index=0,
                         log=False, zero_level='center', norm_level='min',
                         contours=True, contours_only=False,
                         levels=30, cmin=np.nanmin, cmax=np.nanmax,
                         draw_images=False, background=None, color='white',
                         cmap=GLEAMcmaps.phoenix,
                         clabels=False, scalebar=False, label=None, colorbar=False,
                         fontsize=None,
                         **kwargs):
    """
    Plot the Roche potential data in a standardized manner

    Args:
        model <LensModel object> - ensemble model as a LensModel object, or tuple of arrays

    Kwargs:
        N <int> - model's grid size in pixelsq
        mdl_index <int> - model index of the LensModel object
        obj_index <int> - object index of the LensModel object
        src_index <int> - source index of the LensModel object
        log <bool> - plot model in log scale
        zero_level <str> - zero level location, i.e. location to be  0 ['center', 'min', 'max']
        norm_level <str> - norm level location, i.e. location to be -1 ['center', 'min', 'max']
        contours <bool> - plot contours onto map
        contours_only <bool> - only plot contours
        levels <int> - number of contours
        cmin <func> - function determining contour minimum (must accept a single argument)
        cmax <func> - function determining contour maximum (must accept a single argument)
        draw_images <bool> - do/do not plot image
        background <float> - greyscale color for background
        color <str> - color of labels and annotations
        cmap <str/mpl.cm.ColorMap object> - color map for plotting
        levels <int> - contour level labels
        clabels <bool> - add labels to the contours
        scalebar <bool> - add an arcsec scalebar to bottom-left corner instead of axis ticks
        label <str> - add a labelbox to top-left corner
        colorbar <bool> - add colorbar next to map

    Return:
        None
    """
    if isinstance(model, LensModel):
        model.obj_idx = min(obj_index, model.N_obj-1) if hasattr(model, 'N_obj') else obj_index
        N = max(model.data.shape[-1], 36) if N is None else N
        x, y = model.xy_grid(N=N)
        grid = model.roche_potential_grid(model_index=mdl_index, N=N)
    elif isinstance(model, (tuple, list)):
        x, y, grid = model
        R = np.max(np.abs(np.concatenate([x, y])))*(1. + 1./(grid.shape[-1]-1))
        model = LensModel(grid, maprad=R)
    elif isinstance(model, np.ndarray):
        x = y = np.asarray([-1, 1])
        grid = model
        model = LensModel(grid, maprad=1)
    else:
        pass
    # get zero level and normalization coordinates
    if zero_level == norm_level:
        norm_level = [e for e in ['max', 'center', 'min'] if e != zero_level][-1]
    if zero_level is None:
        def identity(grid): return 1
        level_shift = identity
    elif zero_level == 'min':
        def mival(grid): return np.nanmin(grid)
        level_shift = mival
    elif zero_level == 'max':
        def maval(grid): return np.nanmax(grid)
        level_shift = maval
    else:  # == 'center'
        def cval(grid): return grid[grid.shape[0]//2, grid.shape[1]//2]
        level_shift = cval
    if norm_level is None:
        def identity(grid): return 1
        norm = identity
    elif norm_level == 'center':
        def cval(grid): return grid[grid.shape[0]//2, grid.shape[1]//2]
        norm = cval
    elif norm_level == 'max':
        def maval(grid): return np.nanmax(grid)
        norm = maval
    else:  # == 'min'
        def mival(grid): return -np.nanmin(grid)
        norm = mival
    # add images
    minima = model.minima
    saddles = model.saddle_points
    maxima = model.maxima
    cmap = plt.get_cmap(cmap) if isinstance(cmap, str) else cmap
    min_clr, sad_clr, max_clr = cmap(.8), cmap(.5), cmap(.2)
    draw_images = 'min, sad, max' if draw_images else ''
    if 'min' in draw_images and len(minima) > 0:
        plt.plot(minima.T[0], minima.T[1], color=min_clr, marker='o', lw=0)
    if 'sad' in draw_images and len(saddles) > 0:
        plt.plot(saddles.T[0], saddles.T[1], color=sad_clr, marker='o', lw=0)
    if 'max' in draw_images and len(maxima) > 0:
        plt.plot(maxima.T[0], maxima.T[1], color=max_clr, marker='o', lw=0)
    if 'center' in draw_images or 'origin' in draw_images:
        plt.plot(0, 0, color=max_clr, marker='o', lw=0)
    # adjust potential level and normalization
    grid = grid - level_shift(grid)
    grid = grid / norm(grid)
    # contour levels
    if log:
        mi, ma = cmin(grid), cmax(grid)
        clevels = np.logspace(np.log10(1), np.log10(1+abs(ma-mi)), levels)
        clevels = clevels - 1 + mi
    else:
        clevels = np.linspace(cmin(grid), cmax(grid), levels)
    if background is not None:
        bg = np.ones(x.shape+(4,)) * to_rgba(background, alpha=0.75)
        plt.imshow(bg, extent=model.extent)
    if contours or contours_only:
        kwargs.setdefault('origin', 'upper')
        kwargs.setdefault('extent', model.extent)
        lw = kwargs.pop('linewidths', 1.25)
        cs = plt.contour(grid, levels=clevels, cmap=GLEAMcmaps.reverse(cmap),
                         extent=kwargs.get('extent'), origin=kwargs.get('origin'), linewidths=lw)
    if not contours_only:
        kwargs.setdefault('alpha', 0.2 if contours else 0.1)
        plt.contourf(grid, levels=clevels, cmap=GLEAMcmaps.reverse(cmap), **kwargs)
    # annotations and amendments
    if colorbar:
        cbar = plt.colorbar()
        cbar.set_alpha(1)
    if clabels and (contours or contours_only):
        plt.clabel(cs, cs.levels[-1:], fmt='%2.1f', inline=True,
                   alpha=0.75, fontsize=10)
        plt.clabel(cs, cs.levels[:1], fmt='%2.1f', inline=True,
                   alpha=0.75, fontsize=10)
    if scalebar:
        plot_scalebar(x.max(), length=1., position='bottom left', origin='center', color=color,
                      fontsize=fontsize)
        plt.axis('off')
        plt.gcf().axes[0].get_xaxis().set_visible(False)
        plt.gcf().axes[0].get_yaxis().set_visible(False)
    if label is not None:
        plot_labelbox(label, position='top left', padding=(0.03, 0.03), color=color,
                      fontsize=fontsize)
    plt.gca().set_aspect('equal')


def arrival_time_surface_plot(model, N=None, geofactor=1., psifactor=1.,
                              mdl_index=-1, obj_index=0, src_index=0,
                              cmap=GLEAMcmaps.phoenix, maprad=None, extent=None,
                              draw_images=True, search_saddles=0,
                              contours_only=False, sad_contour_saddles=None,
                              contours=True, levels=60,
                              min_contour_shift=None, sad_contour_shift=None,
                              scalebar=False, label=None, color='black',
                              origin='upper',
                              colorbar=False):
    """
    Plot the arrival-time surface of a GLASS model with auto-adjusted contour levels

    Args:
        model <LensModel object> - ensemble model as a LensModel object, or tuple of arrays

    Kwargs:
        N <int> - model's grid size in pixels
        mdl_index <int> - model index of the LensModel object
        obj_index <int> - object index of the LensModel object
        src_index <int> - source index of the LensModel object
        cmap <str/mpl.cm.ColorMap object> - color map for plotting
        extent <tuple/list> - map extent
        draw_images <bool> - do/do not plot image
        search_saddles <int> - number of saddle-points to look for if unknown
        contours_only <bool> - only plot contours
        contours <bool> - plot contours onto map
        levels <int> - number of contours
        min_contour_shift <float> - shift the contours inwards to avoid contouring edges
        scalebar <bool> - add an arcsec scalebar to bottom-left corner instead of axis ticks
        label <str> - add a labelbox to top-left corner
        color <str> - color of labels and annotations
        colorbar <bool> - add colorbar next to map

    Return:
        None
    """
    if isinstance(model, LensModel):
        model.obj_idx = min(obj_index, model.N_obj-1) if hasattr(model, 'N_obj') else obj_index
        N = max(model.data.shape[-1], 36) if N is None else N
        x, y = model.xy_grid(N=N)
        grid = model.arrival_grid(model_index=mdl_index, N=N,
                                  geofactor=geofactor, psifactor=psifactor)
    elif isinstance(model, (tuple, list)):
        x, y, grid = model
        R = np.max(np.abs(np.concatenate([x, y])))*(1. + 1./(grid.shape[-1]-1))
        model = LensModel(grid, maprad=R)
    elif isinstance(model, np.ndarray):
        x = y = np.asarray([-1, 1])
        grid = model
        model = LensModel(grid, maprad=1)
    else:
        pass
    R = model.maprad if maprad is None else maprad
    extent = model.extent if extent is None else extent
    # add images
    minima = model.minima
    saddles = model.saddle_points
    maxima = model.maxima
    sad_contour_saddles = saddles if sad_contour_saddles is None else sad_contour_saddles
    cmap = plt.get_cmap(cmap) if isinstance(cmap, str) else GLEAMcmaps.reverse(cmap)
    min_clr, sad_clr, max_clr = cmap(.2), cmap(.5), cmap(.8)
    draw_images = 'min, sad, max' if draw_images else ''
    # some saddle point contours
    if not np.any(saddles):
        saddles, isaddles = find_saddles(x, y, grid, n_saddles=search_saddles)
        clev = model.saddle_contour_levels(saddle_points=sad_contour_saddles, maprad=R, N=grid.shape[-1],
                                           geofactor=geofactor,
                                           psifactor=psifactor)
        saddles = np.array([(s[0], -s[1]) for s in saddles])
        clev = sorted([grid[ix, iy] for (ix, iy) in isaddles])
    else:
        clev = model.saddle_contour_levels(saddle_points=sad_contour_saddles, maprad=R, N=grid.shape[-1],
                                           geofactor=geofactor,psifactor=psifactor)
    if sad_contour_shift is not None:
        clev = [c-sad_contour_shift for c in clev]
    # plot extremal image points
    if 'min' in draw_images and len(minima) > 0:
        plt.plot(minima.T[0], minima.T[1], color=min_clr, marker='o', lw=0)
    if 'sad' in draw_images and len(saddles) > 0:
        plt.plot(saddles.T[0], saddles.T[1], color=sad_clr, marker='o', lw=0)
    if 'max' in draw_images and len(maxima) > 0:
        plt.plot(maxima.T[0], maxima.T[1], color=max_clr, marker='o', lw=0)
    if 'center' in draw_images or 'origin' in draw_images:
        plt.plot(0, 0, color=max_clr, marker='o', lw=0)
    # general contours
    mi = np.min(grid)
    ma = np.max(grid)
    if min_contour_shift is not None:
        # start with 0.15
        cshift = min_contour_shift * abs(np.max(grid) - clev[-1])
        ma = clev[-1] + cshift
    if contours:
        plt.contour(grid, np.linspace(mi, ma, levels),
                    extent=extent, origin=origin, cmap=cmap)
    # surface plot
    if not contours_only:
        a = 0.2 if contours else 0.8
        plt.contourf(grid, levels=np.linspace(mi, ma, levels),
                     extent=extent, origin=origin,
                     cmap=cmap, alpha=a, vmin=mi, vmax=ma)
    if colorbar:
        plt.colorbar()
    # critical contours
    plt.contour(grid, clev, extent=extent, origin=origin,
                colors=['k']*max(1, len(clev)), linestyles='-')
    # annotations and amendments
    if scalebar:
        plot_scalebar(R, length=1., position='bottom left', origin='center',
                      color=color)
        plt.axis('off')
        plt.gcf().axes[0].get_xaxis().set_visible(False)
        plt.gcf().axes[0].get_yaxis().set_visible(False)
    if label is not None:
        plot_labelbox(label, position='top left', padding=(0.03, 0.03), color=color)

    plt.gca().set_aspect('equal')


def kappa_profile_plot(model, refined=True,
                       obj_index=-1, maprad=None, pixrad=None,
                       r_shift=0, kappa_factor=1.,
                       kappa1_line=True, einstein_radius_indicator=False,
                       annotation_color='black', label=None,
                       **kwargs):
    """
    Calculate radial kappa profiles for GLASS models and plot them or directly
    plot other model's kappa grids

    Args:
        model <LensModel object> - ensemble model as a LensModel object, or tuple of arrays

    Kwargs:
        obj_index <int> - object index for the LensModel object
        maprad <float> - map radius or physical scale of the profile
        pixrad <int> - pixel radius of the kappa map
        kappa1_line <bool> - indicate <kappa> = 1 with a horizontal line
        einstein_radius_indicator <bool> - indicate notional Einstein radius with vertical line
        annotation_color <str> - color of all custom annotations
        label <str> - add a labelbox to top-left corner and annotate Einstein radius

    Return:
        plot <tuple> - first return argument of mpl.pyplot.plot
        radii <np.ndarray> - data point coordinates on the x axis
        profile <np.ndarray> - data point coordinates on the y axis
    """
    # defaults
    kwargs.setdefault('lw', 1)
    kwargs.setdefault('ls', '-')
    kwargs.setdefault('color', GLEAMcolors.blue)
    kwargs.setdefault('alpha', 1)
    # gather data
    if isinstance(model, np.ndarray):  # kappa grid
        model = LensModel(model, maprad=maprad, pixrad=pixrad)
        radii, profile = kappa_profile(model.data, factor=kappa_factor,
                                       maprad=maprad, pixrad=pixrad, refined=refined)
    elif isinstance(model, dict) and 'obj,data' in model:  # a glass model
        radii, profile = kappa_profile(model, obj_index=obj_index, factor=kappa_factor,
                                       maprad=maprad, pixrad=pixrad, refined=refined)
    elif isinstance(model, LensModel):
        radii, profile = kappa_profile(model, obj_index=obj_index, factor=kappa_factor,
                                       maprad=model.maprad, pixrad=model.pixrad, refined=refined)
    else:  # otherwise assume profile and radii were inputted
        radii, profile = model
        radii = np.asarray(radii)
        profile = np.asarray(profile)
        profile = profile[:] * kappa_factor
    radii = radii + r_shift
    plot, = plt.plot(radii, profile, **kwargs)
    if kappa1_line:
        plt.axhline(1, lw=1, ls='-', color='black', alpha=0.5)
    if einstein_radius_indicator:
        einstein_radius = find_einstein_radius(radii, profile)
        plt.axvline(einstein_radius, lw=1, ls='--', color=annotation_color, alpha=0.5)
        if label is not None:
            ax = plt.gca()
            ax.text(1.025*einstein_radius/np.max(radii), 0.95, r'R$_{\mathsf{E}}$',
                    transform=ax.transAxes, fontsize=14, color=annotation_color)
    if label is not None:
        plot_labelbox(label, position='top right', padding=(0.03, 0.03), color=annotation_color)
    return plot, radii, profile


def kappa_profiles_plot(model, obj_index=0, src_index=0, ensemble_average=True, refined=True,
                        interpolate=None, extent=None, kfactor=1., rfactor=1.,
                        as_range=False, kappa1_line=True, einstein_radius_indicator=True,
                        maprad=None, pixrad=None,
                        hilite_color=GLEAMcolors.red, annotation_color='black',
                        levels=20, cmap=GLEAMcmaps.agaveglitch,
                        adjust_limits=True, label=None, label_axes=False, fontsize=None,
                        **kwargs):
    """
    Plot all kappa profiles of a GLASS ensemble model either as lines or 2D histogram contours

    Args:
        model <LensModel object> - ensemble model as a LensModel object, or tuple of arrays

    Kwargs:
        obj_index <int> - object index for the LensModel object
        src_index <int> - source index for the LensModel object
        ensemble_average <bool> - highlight the ensemble average
        refined <bool> - use kappa model on refined grid instead on toplevel
        as_range <bool> - plot the ensemble as a histogrammed number density plot,
                          contoured using a colormap
        interpolate <int> - interpolate profiles to increase number of data points
        kappa1_line <bool> - indicate <kappa> = 1 with a horizontal line
        einstein_radius_indicator <bool> - indicate the notional Einstein radius
                                           with a vertical line
        maprad <float> - map radius or physical scale of the profile
        pixrad <int> - pixel radius of the kappa map
        hilite_color <str> - highlight color of the Einstein radius line
        annotation_color <str> - color of all custom annotations
        levels <int> - number of contour levels in case of as_range
        cmap <str/mpl.cm.ColorMap object> - color map in case of as_range
        adjust_limits <bool> - adjust limits on x and y axes automatically
        label <str> - annotation label, e.g. name of lens model
        label_axes <bool> - automatically label the axes
        fontsize <int> - fontsize of the axis labels
    """
    # defaults
    kwargs.setdefault('lw', 1)
    kwargs.setdefault('ls', '-')
    kwargs.setdefault('color', GLEAMcolors.blue)
    kwargs.setdefault('alpha', 1)
    verbose = kwargs.pop('verbose', False)
    # data
    if isinstance(model, (tuple, list)):
        model = np.asarray(model)
        model = LensModel(model, pixrad=model.shape[-1])
    elif isinstance(model, np.ndarray):
        model = LensModel(model, pixrad=model.shape[-1])
    if not isinstance(model, LensModel):
        model = LensModel(model, obj_index=obj_index, src_index=src_index)
    else:
        model.obj_idx = min(obj_index, model.N_obj-1) if hasattr(model, 'N_obj') else obj_index
    if refined and hasattr(model, 'data_hires'):
        data = model.data_hires
    elif refined:
        data = model.data
    elif hasattr(model, 'data_toplevel'):
        data = model.data_toplevel
    else:
        data = model.data
    eavg = np.average(data, axis=0)
    maprad = model.maprad if maprad is None else maprad
    pixrad = model.pixrad if pixrad is None else pixrad
    # # setting units (if necessary)
    data = data[:] * kfactor
    eavg = eavg[:] * kfactor
    maprad = maprad * rfactor
    kfactor = 1.
    # # output containers
    plots = []
    profiles = []
    radii = []
    ax = plt.gca()
    for m in data:
        radius, profile = kappa_profile(m, obj_index=model.obj_idx, factor=kfactor,
                                        maprad=maprad, pixrad=pixrad)
        radius = radius[:] * rfactor
        if profile[0] <= 0.2:
            continue
        if interpolate > 1:
            radius, profile = interpolate_profile(radius, profile, Nx=interpolate*len(radius))
        if not as_range:
            plot, radius, profile = kappa_profile_plot((radius, profile), kappa1_line=False,
                                                       annotation_color=annotation_color, **kwargs)
            plots.append(plot)
        profiles.append(profile)
        radii.append(radius)
        ax.set_aspect('auto')
    if as_range:
        radii = [r for r, k in zip(radii, profiles) if k[0] > 0.2]
        profiles = [k for k in profiles if k[0] > 0.2]
        dist = np.asarray([[ri, ki] for r, k in zip(radii, profiles) for ri, ki in zip(r, k)]).T
        H, xedges, yedges = np.histogram2d(dist[0], dist[1],
                                           bins=(interpolate or int(0.1*len(profiles))))
        # H = ndimage.zoom(H, 3)
        dens = H.T
        dens = np.log10(dens + 1)
        levels = np.linspace(dens.min(), dens.max(), levels)
        cs = plt.contourf(dens, levels=levels, cmap=cmap,
                          extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]])
        plots.append(cs)
        ax.set_aspect('auto')
    # radius, profile <- ensemble average
    radius, profile = kappa_profile(eavg, obj_index=model.obj_idx, factor=kfactor,
                                    maprad=maprad, pixrad=pixrad)
    radius = radius[:] * rfactor
    if ensemble_average:
        if interpolate > 1:
            radius, profile = interpolate_profile(radius, profile, Nx=interpolate*len(radius))
        kw = dict(lw=kwargs['lw'], ls=kwargs['ls'], color=hilite_color, alpha=kwargs['alpha'])
        plot, radius, profile = kappa_profile_plot((radius, profile), obj_index=model.obj_idx,
                                                   kappa1_line=False, maprad=maprad, pixrad=pixrad,
                                                   annotation_color=annotation_color, **kw)
        plots.append(plot)
        profiles.append(profile)
        radii.append(radius)
    if kappa1_line:
        plt.axhline(1, lw=1, ls='-', color=annotation_color, alpha=0.5)
    if adjust_limits:
        plt.xlim(left=np.min(radius), right=np.max(radius))
        inner_slope = abs((profile[1]-profile[0])/(radius[1]-radius[0]))
        if inner_slope > 20:
            plt.ylim(bottom=np.min(profile), top=1.2*profile[0])
        else:
            plt.ylim(bottom=np.min(profile), top=np.max(profile))
    if einstein_radius_indicator:
        einstein_radius = find_einstein_radius(radius, profile)
        if verbose:
            print("<R_E> = {} arcsec".format(einstein_radius))
        plt.axvline(einstein_radius, lw=1, ls='--', color=annotation_color, alpha=0.5)
        if label is not None:
            ax.text(0.87*einstein_radius/np.max(radius), 0.9, r'R$_{\mathsf{E}}$',
                    transform=ax.transAxes, fontsize=fontsize, color=annotation_color)
    if label is not None:
        plot_labelbox(label, position='top right', padding=(0.03, 0.04), color=annotation_color,
                      fontsize=fontsize-1)
    fontsize = max(ax.xaxis.label.get_size(), ax.yaxis.label.get_size()) \
        if fontsize is None else fontsize
    if label_axes:
        plt.xlabel(r'R [arcsec]', fontsize=fontsize)
        plt.ylabel(r'$\mathsf{\kappa}_{<\mathsf{R}}$', fontsize=fontsize+4)
    return plots, profiles, radii


def complex_ellipticity_plot(epsilon,
                             scatter=True, samples=-1, ensemble_averages=True,
                             color=None, colors=[GLEAMcolors.blue], alpha=1,
                             marker=None, markers=['o'],
                             markersize=None, markersizes=[4],
                             ls=None, lss=['-'],
                             contours=False, levels=10, cmap=None,
                             origin_marker=True, adjust_limits=True, axlabels=True,
                             label=None, fontsize=None, annotation_color='black',
                             legend=None, colorbar=False):
    """
    Plot the complex ellipticity scatter plot of a number of ensemble models

    Args:
        epsilon <tuple/list> - list of complex epsilon positions

    Kwargs:
        samples <int> - only plot a subsample from the ensemble models' epsilons
        color <str> - marker color for scatter plot
        colors <list(str)> - list of different colors for ensemble models

    Return:
        TODO
    """
    # set defaults
    epsilon = epsilon if isinstance(epsilon, (tuple, list)) else [epsilon]
    N_ensembles = len(epsilon)
    samples = [samples]*N_ensembles if isinstance(samples, int) else samples
    colors = [color]*N_ensembles if color is not None else colors
    markers = [marker]*N_ensembles if marker is not None else markers
    markersizes = [markersize]*N_ensembles if markersize is not None else markersizes
    lss = [ls]*N_ensembles if ls is not None else lss
    if len(colors) != N_ensembles:
        q, r = divmod(N_ensembles, len(colors))
        colors = q*colors + colors[:r]
    if len(markers) != N_ensembles:
        q, r = divmod(N_ensembles, len(markers))
        markers = q*markers + markers[:r]
    if len(markersizes) != N_ensembles:
        q, r = divmod(N_ensembles, len(markersizes))
        markersizes = q*markersizes + markersizes[:r]
    if len(lss) != N_ensembles:
        q, r = divmod(N_ensembles, len(lss))
        lss = q*lss + lss[:r]
    plots = []
    # scatter plot
    if scatter:
        if ensemble_averages:
            ens_avgs = []
            for e in epsilon:
                z = np.mean(e[0]+1j*e[1])
                ens_avgs.append(np.array([z.real, z.imag]))
            ens_avgs = np.asarray(ens_avgs)
            for i, (avg, eps) in enumerate(zip(ens_avgs, epsilon)):
                if np.all(avg == eps):
                    avg = ens_avgs[i-1]
                ensemble_size = eps.shape[-1] if len(eps.shape) > 1 else 1
                e = np.random.choice(ensemble_size, size=min(ensemble_size, samples[i])) \
                    if samples[i] > 0 \
                    else np.arange(ensemble_size)
                eps = eps[:, e] if len(eps.shape) > 1 else eps
                if samples[i] == 0:
                    eps = avg
                tree = plot_connection([avg, eps], color=colors[i], marker=markers[i], ls=lss[i],
                                       markersize=markersizes[i], label=legend, alpha=alpha)
                plots.append(tree)
        else:
            for i, e in enumerate(epsilon):
                plt.scatter(e[0], e[1], c=colors[i], alpha=alpha,
                            s=markersizes[i], marker=markers[i])
    # contours
    if contours:
        for i, e in enumerate(epsilon):
            H, xedges, yedges = np.histogram2d(e[0], e[1])
            extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
            clevels = np.linspace(1, H.max(), levels, dtype=np.int)
            clevels = sorted(list(set(clevels)))
            plt.contour(H.T, levels=clevels, extent=extent, cmap=cmap)
            plt.contourf(H.T, levels=clevels, extent=extent, cmap=cmap, alpha=alpha)
    # plot settings
    if origin_marker:
        plt.axhline(0, lw=0.25, color=annotation_color, alpha=0.5)
        plt.axvline(0, lw=0.25, color=annotation_color, alpha=0.5)
    # annotations and amendments
    if colorbar and cmap is not None:
        plt.colorbar()
    if legend is not None:
        plt.legend(handles=plots)
    if label is not None:
        plot_labelbox(label, position='top right', padding=(0.03, 0.03), color=annotation_color)
    if adjust_limits:
        xlim = 1.1 * max([np.abs(e[0]).max() for e in epsilon])
        ylim = 1.1 * max([np.abs(e[1]).max() for e in epsilon])
        lim = max(xlim, ylim)
        plt.xlim(left=-lim, right=lim)
        plt.ylim(bottom=-lim, top=lim)
    if axlabels:
        plt.xlabel(r'$\mathrm{\mathsf{Re\,\epsilon}}$', fontsize=fontsize)
        plt.ylabel(r'$\mathrm{\mathsf{Im\,\epsilon}}$', fontsize=fontsize)
    return plots


def viewstate_plots(model, obj_index=None, refined=True,
                    title=None, savefig=False, showfig=True, verbose=True):
    """
    Plot a GLEAM-style viewstate

    Args:
        model <LensModel object> - a LensModel object

    Kwargs:
        obj_index <int> - plot only the selected viewstate from the LensModel instead of all
        refined <bool> - use kappa model on the refined grid instead on the toplevel
        title <str> - figure title
        savefig <bool> - save the figure before showing
        showfig <bool> - show the figure after saving
        verbose <bool> - verbose mode; print command line statements

    Return:
        None
    """
    if not isinstance(model, LensModel):
        try:
            model = LensModel(model)
        except:
            print("Input must be a LensModel object from <gleam.utils.lensing>")
            return
    if obj_index is None:
        objlst = range(model.N_obj)
    else:
        objlst = [obj_index]
    for obj in objlst:
        model.obj_idx = obj
        if verbose:
            print(model.obj_name)
            print(model.__v__)
        # automatic data extraction
        fig = plt.figure(figsize=(12, 8))
        # plt.suptitle(mdl.obj_name, ha='left')
        # # Kappa map
        plt.subplot(231)
        plt.title(r'$\kappa$')
        kappa_map_plot(model, obj_index=obj, contours=1, subcells=3*refined)
        # plt.ylabel(r'arcsec')
        # # Arrival surface
        plt.subplot(232)
        plt.title(r'$\tau$')
        arrival_time_surface_plot(model, obj_index=obj)
        # plt.ylabel(r'arcsec')
        # # Roche potential
        plt.subplot(233)
        plt.title(r'$\mathcal{P}$')
        roche_potential_plot(model, obj_index=obj, log=1, draw_images=1)
        # plt.ylabel(r'arcsec')
        # # Kappa profile
        plt.subplot(234)
        plt.title(r'$\kappa_{<R}$')
        if model.N > 1:
            kappa_profiles_plot(model, obj_index=obj, refined=refined, as_range=1, interpolate=200,
                                verbose=verbose)
        else:
            kappa_profile_plot(model, obj_index=obj, refined=refined, einstein_radius_indicator=1)
        # plt.ylabel(r'arcsec')
        # # Potential
        plt.subplot(235)
        plt.title(r'$\psi$')
        potential_plot(model, obj_index=obj, contours=0, draw_images=1)
        # plt.ylabel(r'arcsec')
        # # Shear map
        plt.subplot(236)
        plt.title(r'$\gamma^{\mathrm{ext}}_{\theta}$')
        x, y = model.xy_grid()
        s = model.shear_grid()
        potential_plot((x, y, s), contours=0, draw_images=1)
        # plt.ylabel(r'arcsec')
        if title is not None:
            titlename = title
            if not isinstance(title, str):
                if os.path.splitext(model.filename)[0] == model.obj_name:
                    titlename = model.obj_name
                else:
                    titlename = "{}.{}".format(os.path.splitext(model.filename)[0], model.obj_name)
            plt.suptitle(titlename)
        square_subplots(fig)
        # plt.tight_layout()
        if savefig:
            if os.path.splitext(model.filename)[0] == model.obj_name:
                savename = model.obj_name
            else:
                savename = "{}.{}".format(os.path.splitext(model.filename)[0], model.obj_name)
            plt.savefig('viewstate_{}.pdf'.format(savename))
        if showfig:
            plt.show()


def h0hist_plot(model, units='km/s/Mpc', nbins=30,
                result_label=False, label_pos='left',
                title=None, savefig=False, showfig=True, verbose=True):
    """
    TODO
    """
    if not isinstance(model, LensModel):
        try:
            model = LensModel(model)
        except:
            print("Input must be a LensModel object from <gleam.utils.lensing>")
            return
    if verbose:
        print(model.__v__)
    fig, ax = plt.subplots()
    axc = None
    h0arr = model.H0
    # unit conversions
    if units == 'km/s/Mpc' or units == 'km s^{-1} Mpc^{-1}':
        h0arr = np.array(h0arr)
        hlbl = 'H$_0$'
    elif units == 'aHz':
        h0arr = glu.H02aHz(np.array(h0arr))
        hlbl = 'H$_0$'
        axc = ax.twiny()
        # second unit axis
        def convertaxc(ax):
            x1, x2 = ax.get_xlim()
            axc.set_xlim(glu.aHz2H0(x1), glu.aHz2H0(x2))
            axc.figure.canvas.draw()
        ax.callbacks.connect("xlim_changed", convertaxc)
    elif units == 'Gyrs':
        h0arr = glu.H02Gyrs(np.array(h0arr))
        hlbl = 'H$_0^{{-1}}$'
    elif units == 'GeV/m^3':
        h0arr = glu.H02critdens(np.array(h0arr))
        hlbl = r"$\rho_{{\mathrm{{c}}}}$"
    # calculate/plot statistics and histogram
    q = np.percentile(h0arr, [16, 50, 84])
    n, bins, patches = ax.hist(h0arr, bins=nbins, density=True, rwidth=0.901)
    cm = plt.cm.get_cmap('phoenix')
    ax.axvline(q[1], color=GLEAMcolors.pink, zorder=0, alpha=0.6)
    ax.axvline(q[0], color=GLEAMcolors.red, zorder=0, alpha=0.6)
    ax.axvline(q[2], color=GLEAMcolors.red, zorder=0, alpha=0.6)
    # decide on the coloring of the histogram bars
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    # xdata = bin_centers
    # ydata = n
    # lpopt, lpcov = curve_fit(lorentzian, xdata, ydata)
    # ldist = lorentzian(xdata, lpopt[0], lpopt[1], lpopt[2])
    # gpopt, gpcov = curve_fit(gaussian, xdata, ydata)
    # gdist = gaussian(xdata, gpopt[0], gpopt[1], gpopt[2])
    # vpopt, vpcov = curve_fit(pseudovoigt, xdata, ydata)
    # vdist = pseudovoigt(xdata, vpopt[0], vpopt[1], vpopt[2], vpopt[3], vpopt[4])
    # tpopt, tpcov = curve_fit(tukeygh, xdata, ydata)
    # tdist = tukeygh(xdata, vpopt[0], vpopt[1], vpopt[2], vpopt[3])
    cpf = np.cumsum(n * abs(bins[0]-bins[-1]) / nbins) * 2
    cpf[cpf > 1] = -(cpf[cpf > 1] - 1) + 1
    for c, p in zip(cpf, patches):
        plt.setp(p, 'facecolor', cm(c))
    # add annotations
    plt.rcParams['mathtext.fontset'] = 'stixsans'
    if result_label:
        Hstr = 'H$_0$ = ${:5.1f}^{{{:+4.1f}}}_{{{:+4.1f}}}$'
        q = np.percentile(model.H0, [16, 50, 84])
        if label_pos == 'left':
            lblp = 0.02, 0.85
        elif label_pos == 'right':
            lblp = 0.68, 0.85
        plt.text(lblp[0], lblp[1], Hstr.format(q[1], np.diff(q)[1], -np.diff(q)[0]),
                 fontsize=19, color='black', transform=plt.gca().transAxes)
    # plot styling
    fig.axes[0].get_yaxis().set_visible(False)
    ax.set_xlabel('{} [{}]'.format(hlbl, units))
    if axc is not None:
        axc.set_xlabel('H$_0$ [km/s/Mpc]', fontsize=12)
        plt.setp(axc.get_xticklabels(), fontsize=12)
    plt.tight_layout()
    if savefig:
        savename = os.path.splitext(model.filename)[0]
        plt.savefig('h0hist_{}.pdf'.format(savename))
    if showfig:
        plt.show()
    return fig, (ax, axc)



# INTERACTIVE ##################################################################
class IAColorbar(object):
    def __init__(self, cbar, mappable, verbose=False):
        self.cbar = cbar
        self.mappable = mappable
        self.press = None
        self.cycle = sorted([i for i in dir(plt.cm) if hasattr(getattr(plt.cm, i), 'N')])
        cmap_name = cbar.get_cmap().name
        if cmap_name not in self.cycle:
            self.cycle.append(cmap_name)
        self.index = self.cycle.index(cmap_name)
        self.verbose = verbose

    def connect(self):
        """
        Connect to all the events we need
        """
        self.cidpress = self.cbar.patch.figure.canvas.mpl_connect(
            'button_press_event', self.on_press)
        self.cidrelease = self.cbar.patch.figure.canvas.mpl_connect(
            'button_release_event', self.on_release)
        self.cidmotion = self.cbar.patch.figure.canvas.mpl_connect(
            'motion_notify_event', self.on_motion)
        self.keypress = self.cbar.patch.figure.canvas.mpl_connect(
            'key_press_event', self.key_press)

    def on_press(self, event):
        """
        On button press we will see if the mouse is over us and store some data
        """
        if event.inaxes != self.cbar.ax: return
        self.press = event.x, event.y

    def key_press(self, event):
        if event.key=='down':
            self.index += 1
        elif event.key=='up':
            self.index -= 1
        if self.index < 0:
            self.index = len(self.cycle)
        elif self.index >= len(self.cycle):
            self.index = 0
        cmap = self.cycle[self.index]
        self.cbar.set_cmap(cmap)
        self.cbar.draw_all()
        self.mappable.set_cmap(cmap)
        if self.verbose:
            print(cmap)
        self.cbar.patch.figure.canvas.draw()

    def on_motion(self, event):
        """
        On motion we will move the rect if the mouse is over us
        """
        if self.press is None: return
        if event.inaxes != self.cbar.ax: return
        xprev, yprev = self.press
        # dx = event.x - xprev
        dy = event.y - yprev
        self.press = event.x,event.y
        #print 'x0=%f, xpress=%f, event.xdata=%f, dx=%f, x0+dx=%f'%(x0, xpress, event.xdata, dx, x0+dx)
        scale = self.cbar.norm.vmax - self.cbar.norm.vmin
        perc = 0.03
        if event.button==1:
            self.cbar.norm.vmin -= (perc*scale)*np.sign(dy)
            self.cbar.norm.vmax -= (perc*scale)*np.sign(dy)
        elif event.button==3:
            self.cbar.norm.vmin -= (perc*scale)*np.sign(dy)
            self.cbar.norm.vmax += (perc*scale)*np.sign(dy)
        self.cbar.draw_all()
        self.mappable.set_norm(self.cbar.norm)
        self.cbar.patch.figure.canvas.draw()

    def on_release(self, event):
        """
        On release we reset the press data
        """
        self.press = None
        self.mappable.set_norm(self.cbar.norm)
        self.cbar.patch.figure.canvas.draw()

    def disconnect(self):
        """
        Disconnect all the stored connection ids
        """
        self.cbar.patch.figure.canvas.mpl_disconnect(self.cidpress)
        self.cbar.patch.figure.canvas.mpl_disconnect(self.cidrelease)
        self.cbar.patch.figure.canvas.mpl_disconnect(self.cidmotion)



class IPColorbar(object):
    def __init__(self, cbar, mappable, log=False,
                 lim=None, step=0.1, orientation='horizontal',
                 verbose=False):
        from IPython.display import display
        import ipywidgets as widgets
        self.cbar = cbar
        self.mappable = mappable
        self.figure = self.cbar.patch.figure
        self.verbose = verbose
        # interactive slider
        v_curr = [cbar.norm.vmin, cbar.norm.vmax]
        data = self.mappable.get_array()
        if lim is None:
            lim = [np.min(data), np.max(data)]
            lim[0] = lim[0] - abs(lim[0]) * 0.05
            lim[1] = lim[1] + abs(lim[1]) * 0.05
        if lim[1] <= 2*step:
            step = lim[1] / 100
        self.slider = widgets.FloatRangeSlider(
            value=v_curr, min=lim[0], max=lim[1], step=step,
            description='Colorbar', disabled=False, continuous_update=False,
            orientation=orientation, readout=True, readout_format='.4f',
            layout=widgets.Layout(width='100%'))
        # colormaps
        self.cycle = plt.colormaps()
        cmap_name = cbar.get_cmap().name
        self.index = self.cycle.index(cmap_name)
        self.dropdown = widgets.Dropdown(
            options=self.cycle, value=cmap_name,
            description='Colormap', disabled=False, layout=widgets.Layout(width='100%'))
        self.container = widgets.HBox([self.dropdown, self.slider])
        display(self.container)

    def connect(self):
        """
        Connect to all the events we need
        """
        self.dropdown.observe(self.on_dropdown_change, names='value')
        self.slider.observe(self.on_slider_change, names='value')

    def on_dropdown_change(self, state):
        new_cmap = state['new']
        if self.verbose:
            print(new_cmap)
        self.index = self.cycle.index(new_cmap)
        self.cbar.set_cmap(new_cmap)
        self.cbar.draw_all()
        self.mappable.set_cmap(new_cmap)
        self.figure.canvas.draw()

    def on_slider_change(self, state):
        new_v = state['new']
        if self.verbose:
            print(new_v)
        self.cbar.norm.vmin = new_v[0]
        self.cbar.norm.vmax = new_v[1]
        self.cbar.draw_all()
        self.mappable.set_norm(self.cbar.norm)
        self.figure.canvas.draw()



class IPPointCache(object):
    def __init__(self, mappable, color=GLEAMcolors.purple, height='200px',
                 use_modes=['L', 'I', 'm', 'S', 'M'], verbose=False):
        from IPython.display import display
        import ipywidgets as widgets
        self.mappable = mappable
        self.color = color
        self.figure = self.mappable._axes.patch.figure
        self.xy = []
        self.modes = []
        self.log = []
        self.verbose = verbose
        self.text_display = widgets.Textarea(
            value='', placeholder='Cache', description='Points',
            continuous_update=True, disabled=False,
            layout=widgets.Layout(width='75%', height=height))
        if use_modes:
            self.mode_selector = widgets.Dropdown(
                options=use_modes, value=use_modes[0],
                description='Mode', disabled=False,
                layout=widgets.Layout(width='50%'))
            self.container = widgets.HBox([self.text_display, self.mode_selector])
        else:
            self.mode_selector = None
            self.container = self.text_display
        display(self.container)

    def connect(self):
        """
        Connect to all the events we need
        """
        self.cid = self.figure.canvas.mpl_connect(
            'button_press_event', self.on_click)
        return self.cid

    def on_click(self, event):
        """
        On button press we will store some data in container
        """
        if self.figure.canvas.manager.toolbar.mode == 'zoom rect':
            return
        self.log.append(event)
        self.sync_cache()
        mode = self.mode_selector.value if self.mode_selector else '\r'
        xy = [event.xdata, event.ydata]
        self.modes += [mode]
        self.xy.append(xy)
        if self.mode_selector:
            if self.mode_selector.options[0] in self.modes:
                self.mode_selector.value = self.mode_selector.options[1]
            else:
                self.mode_selector.value = self.mode_selector.options[0]
        text_value = "\n".join([str(c) for c in ["".join(self.modes)]+self.xy])
        self.text_display.value = text_value

    def sync_cache(self):
        if self.text_display.value == '':
            self.xy = []
            self.modes = []
        field = self.text_display.value.split('\n')
        modes = list(field[0])
        text = field[1:]
        deletes = 0
        for i, (m, xy) in enumerate(zip(modes[:], text[:])):
            if m == '' or m == ' ' or xy == '':
                self.modes.pop(i)
                self.xy.pop(i)
                deletes += 1
            else:
                pos_txt = str(xy).replace('[', '').replace(']', '').split(', ')
                self.modes[i-deletes] = m
                self.xy[i-deletes] = [float(pos_txt[0]), float(pos_txt[1])]

    def distances_to(self, index, pixel_scale=1):
        """
        Get all coordinate distances to a selected point
        """
        # defaults
        if index >= len(self.xy):
            return None
        if isinstance(pixel_scale, (int, float)):
            pixel_scale = [pixel_scale, pixel_scale]
        # get point coordinates
        distances = []
        xy = self.xy[:]
        # split into subject/objects
        subject = xy.pop(index)
        objects = xy
        for obj in objects:
            dst = [(obj[0]-subject[0]) * pixel_scale[0],
                   (obj[1]-subject[1]) * pixel_scale[1]]
            distances.append(dst)
        return distances



# EXPERIMENTAL #################################################################
def kappa_ensemble_interactive(gls, obj_index=0, src_index=0, subcells=1,
                               extent=None, levels=3, delta=0.2, log=True,
                               oversample=True,
                               cmap=None, add_slider=True, slider_index=0,
                               savename="kappa_model.html"):
    """
    Create an interactive ensemble image plot with bokeh

    Args:
        gls <np.ndarray/glass state> - the convergence map data to be visualized

    Kwargs:
        obj_index <int> - object index of the GLASS object within the model
        subcells <int> - number of subcells
        extent <tuple/list> - map extent
        levels <int> - (half-)number of contours; total number w/ log is 2*N+1
        log <bool> - plot in log scale
        oversample <bool> - oversample the map to show characteristic pixel structure
        cmap <str/mpl.cm.ColorMap object> - color map for plotting
        add_slider <bool> - show a slider at the bottom of the plot to choose the model
        slider_index <int> - index at with which the slider starts
        savename <str> - filename of the html file
   """
    from bokeh import layouts as bklo
    from bokeh import plotting as bkpl
    from bokeh import palettes as bkpalettes
    from bokeh import models as bkmd
    if isinstance(gls, np.ndarray) and len(gls.shape) == 3:
        ensemble_data = np.array(gls, dtype=np.float32)
        if extent is None:
            extent = [-1, 1, -1, 1]
    else:  # if isinstance(gls, object):
        ensemble_data = []
        for m in gls.models:
            obj, dta = m['obj,data'][obj_index]
            dlsds = DLSDS(obj.z, obj.sources[src_index].z)
            # grid = dlsds * obj.basis._to_grid(dta['kappa'], subcells)
            grid = dlsds * obj.basis.kappa_grid(dta)
            ensemble_data.append(grid)
        ensemble_data = np.array(ensemble_data, dtype=np.float32)
        if extent is None:
            # R = obj.basis.maprad
            R = obj.basis.mapextent
            extent = [-R, R, -R, R]
            # pixrad = obj.basis.pixrad
    # msg = "\n".join(["Ensemble data: {}", "obj idx: {}", "subcells: {}",
    #                  "extent: {}", "levels: {}", "log: {}", "oversample: {}",
    #                  "cmap: {}", "add slider: {}", "slider idx: {}",
    #                  "savename: {}"]).format(
    #                      ensemble_data.shape, obj_index, subcells, extent,
    #                      levels, log, oversample, cmap, add_slider,
    #                      slider_index, savename)
    N = ensemble_data.shape[0]
    ensemble_data = np.array([np.flipud(e) for e in ensemble_data])
    img_data = np.array([
        kappa_map_transf(m, obj_index=obj_index, subcells=subcells, extent=extent,
                         levels=levels, delta=delta, log=log, oversample=oversample)[0]
        for m in ensemble_data], dtype=np.float32)
    x = np.linspace(extent[0], extent[1], img_data.shape[2])
    y = np.linspace(extent[2], extent[3], img_data.shape[1])
    source = bkmd.ColumnDataSource(data=dict(
        images=[img_data],
        image=[img_data[slider_index, :, :]],
        kappa=[ensemble_data[slider_index, :, :]],
        dimensions=[img_data[slider_index, :, :].shape],
        x=[x[0]], y=[y[0]], dw=[abs(x[-1]-x[0])], dh=[abs(y[-1]-y[0])]))
    p = bkpl.figure(tools="box_zoom,wheel_zoom,zoom_in,zoom_out,save,reset",
                    tooltips=[("x", "$x"), ("y", "$y"), ("kappa", "@kappa")],
                    x_range=extent[0:2], y_range=extent[-2:])
    if cmap is None:
        palette = GLEAMcmaps.palette('agaveglitch', 4*levels)
    else:
        if isinstance(cmap, str) and cmap.capitalize() in bkpalettes.all_palettes:
            palette = bkpalettes.all_palettes[cmap.capitalize()][9]
        elif hasattr(GLEAMcmaps, cmap):
            palette = GLEAMcmaps.palette(cmap, 4*levels)
        else:
            palette = cmap

    p.image(source=source, image='image', x='x', y='y', dw='dw', dh='dh',
            palette=palette)
    p.toolbar.logo = None

    slider = bkmd.Slider(start=0, end=(N-1), value=slider_index, step=1,
                         title="Ensemble model")
    callback = bkmd.CustomJS(args=dict(source=source),
                             code="""
        var data = source.data;
        var images = data['images'][0];
        var image = data['image'][0];
        var kappa = data['kappa'][0];
        var L = data['dimensions'][0][0]*data['dimensions'][0][1];
        var f = cb_obj.value;
        var start = f*L;
        var end = start+L;
        image = images.slice(start, end);
        source.data['image'] = [image];
        source.data['kappa'] = [image.map(x => 10**x)];
        source.change.emit();""")
    slider.js_on_change('value', callback)

    layout = bklo.column(p, slider)
    bkpl.output_file(savename, title="Convergence map")
    bkpl.save(layout, filename=savename, title="Convergence map")
    # bkpl.show(layout)
