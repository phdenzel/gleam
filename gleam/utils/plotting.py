#!/usr/bin/env python
"""
@author: phdenzel

Lensing utility functions for calculating various analytics from lensing observables
"""
###############################################################################
# Imports
###############################################################################
import numpy as np
import scipy.ndimage as ndimage
from matplotlib import patches
from matplotlib import pyplot as plt
from matplotlib.colors import to_rgba
from gleam.utils.lensing import DLSDS, kappa_profile, \
    interpolate_profile, find_einstein_radius
from gleam.utils.colors import GLEAMcolors, GLEAMcmaps
from gleam.utils.rgb_map import radial_mask
from gleam.glass_interface import glass_renv
glass = glass_renv()


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


def plot_scalebar(R, length=1., unit=r'$^{\prime\prime}$',
                  position='bottom left', origin='center',
                  padding=(0.08, 0.06), color='white'):
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
    if R > 1.2:
        lbl = r"{:1.0f}{}".format(length, unit)
    else:
        length = 0.1
        lbl = r"{:1.1f}{}".format(length, unit)
    wh = np.asarray([length, 0.03*R])
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
            color=color, fontsize=16, va=va, ha=ha, zorder=1000)
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


def kappa_map_plot(model, obj_index=0, subcells=1, extent=None, origin='upper',
                   contours=False, levels=3, delta=0.2, log=True,
                   oversample=True,
                   scalebar=False, label=None, color='white',
                   cmap='magma', colorbar=False):
    """
    Plot the convergence map of a GLASS model with auto-adjusted contour levels

    Args:
        model <glass.LensModel object> - GLASS ensemble model

    Kwargs:
        obj_index <int> - object index of the GLASS object within the model
        subcells <int> - number of subcells
        extent <tuple/list> - map extent
        contours <bool> - plot contours onto map
        levels <int> - half-number of contours; total number w/ log is 2*N+1
        delta <float> - contour distances
        log <bool> - plot in log scale
        oversample <bool> - oversample the map to show characteristic pixel structure
        scalebar <bool> - add an arcsec scalebar to bottom-left corner instead of axis ticks
        label <str> - add a labelbox to top-left corner
        cmap <str/mpl.cm.ColorMap object> - color map for plotting
        colorbar <bool> - plot colorbar next to convergence map

    Return:
        TODO
    """
    obj, dta = model['obj,data'][obj_index]
    dlsds = DLSDS(obj.z, obj.sources[obj_index].z)
    grid = dlsds * obj.basis._to_grid(dta['kappa'], subcells)
    if extent is None:
        # R = obj.basis.maprad
        R = obj.basis.mapextent
        extent = [-R, R, -R, R]
    # pixrad = obj.basis.pixrad
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
        kappa1 = 0
        grid = np.log10(grid)
        grid[grid < clevels[0]] = clevels[0]
    else:
        kappa1 = 1
        clevels = np.concatenate(((0,), 10**clev2))
    # contours and surface plot
    if contours:
        plt.contour(grid, levels=(kappa1,), colors=['k'],
                    extent=extent, origin=origin)
    plt.contourf(grid, cmap=cmap, antialiased=True,
                 extent=extent, origin=origin, levels=clevels)
    # ax = plt.gca()
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
    if label is not None:
        plot_labelbox(label, position='top left', padding=(0.03, 0.03), color=color)


def kappa_profile_plot(model,
                       obj_index=0, correct_distances=True,
                       kappa1_line=True, einstein_radius_indicator=False,
                       maprad=None,
                       annotation_color='black', label=None,
                       **kwargs):
    """
    Calculate radial kappa profiles for GLASS models and plot them or directly
    plot other model's kappa grids

    Args:
        model <GLASS model dict/np.ndarray> - GLASS model dictionary or some other kappa grid model

    Kwargs:
        obj_index <int> - object index of the GLASS object within the model
        correct_distances <bool> - correct with distance ratios and redshifts
        maprad <float> - map radius or physical scale of the profile

    Return:
        TODO
    """
    # defaults
    kwargs.setdefault('lw', 1)
    kwargs.setdefault('ls', '-')
    kwargs.setdefault('color', GLEAMcolors.blue)
    kwargs.setdefault('alpha', 1)
    # gather data
    if isinstance(model, dict) and 'obj,data' in model:  # a glass model
        radii, profile = kappa_profile(model, obj_index=obj_index,
                                       correct_distances=correct_distances,
                                       maprad=maprad)
    elif isinstance(model, np.ndarray):  # kappa grid
        radii, profile = kappa_profile(model, maprad=maprad)
    else:  # otherwise assume profile and radii were inputted
        radii, profile = model
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


def kappa_profiles_plot(gls, obj_index=0, ensemble_average=True, as_range=False,
                        correct_distances=True, interpolate=None,
                        kappa1_line=True, einstein_radius_indicator=True, maprad=None,
                        hilite_color=GLEAMcolors.red, annotation_color='black',
                        levels=20, cmap=GLEAMcmaps.agaveglitch,
                        adjust_limits=True, label=None,
                        **kwargs):
    """
    Plot all kappa profiles of a GLASS ensemble model either as lines or 2D histogram contours

    Args:
        TODO
    """
    # defaults
    kwargs.setdefault('lw', 1)
    kwargs.setdefault('ls', '-')
    kwargs.setdefault('color', GLEAMcolors.blue)
    kwargs.setdefault('alpha', 1)
    # output containers
    plots = []
    profiles = []
    radii = []
    for m in gls.models:
        radius, profile = kappa_profile(m, obj_index=obj_index,
                                        correct_distances=correct_distances, maprad=maprad)
        if interpolate > 1:
            radius, profile = interpolate_profile(radius, profile, Nx=interpolate*len(radius))
        if not as_range:
            plot, radius, profile = kappa_profile_plot((radius, profile),
                                                       kappa1_line=False,
                                                       annotation_color=annotation_color, **kwargs)
            plots.append(plot)
        profiles.append(profile)
        radii.append(radius)
    if as_range:
        dist = np.asarray([[ri, ki] for r, k in zip(radii, profiles) for ri, ki in zip(r, k)]).T
        H, xedges, yedges = np.histogram2d(dist[0], dist[1],
                                           bins=(interpolate or int(0.1*len(profiles))))
        # H = ndimage.zoom(H, 3)
        dens = H.T
        # levels = np.linspace(max(1, dens.min()), 0.9*dens.max(), levels)
        levels = np.logspace(0, 0.9*np.log10(dens.max()), levels)
        # plt.imshow(dens, vmin=levels[0], vmax=levels[-1], cmap=cmap,
        #            extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
        #            origin='lower', interpolation='gaussian')
        plt.contourf(dens, levels=levels, cmap=cmap,
                     extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]])
    # radius, profile <- ensemble average
    gls.make_ensemble_average()
    radius, profile = kappa_profile(gls.ensemble_average, obj_index=obj_index,
                                    correct_distances=correct_distances)
    if ensemble_average:
        kw = dict(lw=1, ls='-', color=hilite_color, alpha=1)
        plot, radius, profile = kappa_profile_plot((radius, profile), obj_index=obj_index,
                                                   correct_distances=correct_distances,
                                                   kappa1_line=False,
                                                   annotation_color=annotation_color, **kw)
        plots.append(plot)
        profiles.append(profile)
        radii.append(radius)
    if kappa1_line:
        plt.axhline(1, lw=1, ls='-', color='black', alpha=0.5)
    if adjust_limits:
        plt.xlim(left=np.min(radius), right=np.max(radius))
    if einstein_radius_indicator:
        einstein_radius = find_einstein_radius(radius, profile)
        plt.axvline(einstein_radius, lw=1, ls='--', color=annotation_color, alpha=0.5)
        if label is not None:
            ax = plt.gca()
            ax.text(1.025*einstein_radius/np.max(radius), 0.95, r'R$_{\mathsf{E}}$',
                    transform=ax.transAxes, fontsize=14, color=annotation_color)
    if label is not None:
        plot_labelbox(label, position='top right', padding=(0.03, 0.03), color=annotation_color)
    plt.xlabel('R [arcsec]')
    plt.ylabel(r'$\kappa_{<\mathsf{R}}$', fontsize=18)
    return plots, profiles, radii


def roche_potential_plot(data, N=85, log=False, zero_level='center', norm_level='min',
                         contours=True, contours_only=False,
                         levels=50, cmin=np.min, cmax=np.max,
                         cmap=GLEAMcmaps.reverse(GLEAMcmaps.phoenix),
                         background=None,
                         scalebar=False, label=None, colorbar=False, **kwargs):
    """
    Plot the Roche potential data in a standardized manner

    Args:
        x <np.ndarray> - x-coordinate grid
        y <np.ndarray> - y-coordinate grid
        grid <np.ndarray> - Roche potential grid

    Kwargs:
        zero_level <str> - zero level location, i.e. location to be  0 ['center', 'min', 'max']
        norm_level <str> - norm level location, i.e. location to be -1 ['center', 'min', 'max']
        levels <int> - number of contour levels
        cmin <func> - function determining contour minimum (must accept a single argument)
        cmax <func> - function determining contour maximum (must accept a single argument)
        cmap <str/mpl.cm.ColorMap object> - color map for plotting
        colorbar <bool> - add colorbar next to Roche potential map
        scalebar <bool> - add an arcsec scalebar to bottom-left corner instead of axis ticks
        label <str> - add a labelbox to top-left corner

    Return:
        TODO
    """
    if isinstance(data, (tuple, list)):
        x, y, grid = data
    if isinstance(data, dict):
        pass
    # get zero level and normalization coordinates
    if zero_level == norm_level:
        norm_level = [e for e in ['max', 'center', 'min'] if e != zero_level][-1]
    if zero_level is None:
        def identity(grid): return 1
        level_shift = identity
    elif zero_level == 'min':
        def mival(grid): return grid.min()
        level_shift = mival
    elif zero_level == 'max':
        def maval(grid): return grid.max()
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
        def maval(grid): return grid.max()
        norm = maval
    else:  # == 'min'
        def mival(grid): return -grid.min()
        norm = mival
    # adjust potential level and normalization
    grid = grid - level_shift(grid)
    grid = grid / norm(grid)
    # contour levels
    if log:
        mi, ma = cmin(grid), cmax(grid)
        clevels = np.logspace(np.log10(1), np.log10(1+ma-mi), levels)
        clevels = clevels - 1 + mi
    else:
        clevels = np.linspace(cmin(grid), cmax(grid), levels)
    if background is not None:
        bg = np.ones(x.shape+(4,)) * to_rgba(background, alpha=0.75)
        bg_extent = [x.min(), x.max(), y.min(), y.max()]
        plt.imshow(bg, extent=bg_extent)
    if contours or contours_only:
        kwargs.setdefault('origin', 'upper')
        kwargs.setdefault('extent', [x.min(), x.max(), y.min(), y.max()])
        lw = kwargs.pop('linewidths', 0.5)
        plt.contour(grid, levels=clevels, cmap=GLEAMcmaps.reverse(cmap),
                    extent=kwrags.get('extent'), origin=kwargs.get('origin'), linewidths=lw)
    if not contours_only:
        plt.contourf(grid, levels=clevels, cmap=GLEAMcmaps.reverse(cmap), **kwargs)
    # annotations and amendments
    if colorbar:
        cbar = plt.colorbar()
        cbar.set_alpha(1)
    if scalebar:
        plot_scalebar(x.max(), length=1., position='bottom left', origin='center', color='white')
        plt.axis('off')
    if label is not None:
        plot_labelbox(label, position='top right', padding=(0.03, 0.03), color='white')
    plt.gca().set_aspect('equal')


def arrival_time_surface_plot(model, obj_index=0, src_index=0,
                              cmap='magma', extent=None, images_off=False,
                              contours_only=True, contours=True, levels=40,
                              min_contour_shift=0.15,
                              scalebar=False, label=None, color='black',
                              colorbar=False):
    """
    Plot the arrival-time surface of a GLASS model with auto-adjusted contour levels

    Args:
        model <glass.LensModel object> - GLASS ensemble model

    Kwargs:
        obj_index <int> - object index of the GLASS object within the model
        src_index <int> - source index of the GLASS object within the model
        cmap <str/mpl.cm.ColorMap object> - color map for plotting
        extent <tuple/list> - map extent
        images_off <bool> - do not plot image/source positions
        contours <bool> - plot contours onto map
        contours_only <bool> - only plot contours
        levels <int> - number of contours
        scalebar <bool> - add an arcsec scalebar to bottom-left corner instead of axis ticks
        label <str> - add a labelbox to top-left corner
        colorbar <bool> - add colorbar next to convergence map

    Return:
        TODO
    """
    obj, dta = model['obj,data'][obj_index]
    grid = obj.basis.arrival_grid(dta)
    clev = obj.basis.arrival_contour_levels(dta)

    # map extent
    if extent is None:
        # R = obj.basis.maprad
        R = obj.basis.mapextent
        extent = [-R, R, -R, R]

    # images
    min_clr = cmap(.8)
    sad_clr = cmap(.5)
    max_clr = cmap(.2)
    minima = np.array([img._pos for img in obj.sources[src_index].images
                       if img.parity_name == 'min'])
    saddles = np.array([img._pos for img in obj.sources[src_index].images
                        if img.parity_name == 'sad'])
    maxima = np.array([img._pos for img in obj.sources[src_index].images
                       if img.parity_name == 'max'])
    if not images_off:
        if len(minima) > 0:
            plt.plot(minima.T[0], minima.T[1], color=min_clr, marker='o', lw=0)
        if len(saddles) > 0:
            plt.plot(saddles.T[0], saddles.T[1], color=sad_clr, marker='o', lw=0)
        if len(maxima) > 0:
            plt.plot(maxima.T[0], maxima.T[1], color=max_clr, marker='o', lw=0)
    plt.plot(0, 0, color=max_clr, marker='o', lw=0)

    # general contours
    mi = np.min(grid)
    cshift = min_contour_shift * abs(np.max(grid) - clev[src_index][-1])
    ma = clev[src_index][-1] + cshift
    if contours:
        plt.contour(grid[src_index], np.linspace(mi, ma, levels),
                    extent=extent, origin='upper', cmap=cmap)
    # surface plot
    if not contours_only:
        # ma_edge = ma-min_contour_shift*0.25  # shifted a little bit further for aesthetics
        # plt.imshow(grid[src_index], extent=extent, origin='upper',
        #            cmap=cmap, alpha=0.3, interpolation='nearest',
        #            vmin=mi, vmax=ma_edge)
        plt.contourf(grid[src_index], levels=np.linspace(mi, ma, levels),
                     extent=extent, origin='upper',
                     cmap=cmap, alpha=0.3, vmin=mi, vmax=ma)
    if colorbar:
        plt.colorbar()
    # critical contours
    plt.contour(grid[src_index], clev[src_index], extent=extent, origin='upper',
                colors=['k']*len(clev[src_index]), linestyles='-')
    # annotations and amendments
    if scalebar:
        plot_scalebar(R, length=1., position='bottom left', origin='center', color=color)
        plt.axis('off')
    if label is not None:
        plot_labelbox(label, position='top left', padding=(0.03, 0.03), color=color)
    plt.gca().set_aspect('equal')


def arrival_time_hypersurface_plot(model, obj_index=0, src_index=0,
                                   cmap='magma', extent=None, images_off=False,
                                   contours_only=True, contours=True, levels=40,
                                   scalebar=False, label=None, color='black',
                                   colorbar=False):
    """
    Plot the arrival-time surface of a GLASS model with auto-adjusted contour levels

    Args:
        model <glass.LensModel object> - GLASS ensemble model

    Kwargs:
        obj_index <int> - object index of the GLASS object within the model
        src_index <int> - source index of the GLASS object within the model
        cmap <str/mpl.cm.ColorMap object> - color map for plotting
        extent <tuple/list> - map extent
        images_off <bool> - do not plot image/source positions
        contours <bool> - plot contours onto map
        contours_only <bool> - only plot contours
        levels <int> - number of contours
        scalebar <bool> - add an arcsec scalebar to bottom-left corner instead of axis ticks
        label <str> - add a labelbox to top-left corner
        colorbar <bool> - add colorbar next to convergence map

    Return:
        None

    Note:
        best used in an interactive session in order to find best viewing settings
    """
    ax = plt.axes(projection='3d')

    obj, dta = model['obj,data'][obj_index]
    grid = obj.basis.arrival_grid(dta)
    clev = obj.basis.arrival_contour_levels(dta)

    # map extent
    if extent is None:
        # R = obj.basis.maprad
        R = obj.basis.mapextent
        extent = [-R, R, -R, R]
    x = np.linspace(extent[0], extent[1], grid[src_index].shape[1])
    y = np.linspace(extent[2], extent[3], grid[src_index].shape[0])
    X, Y = np.meshgrid(x, y)

    # images
    # min_clr = cmap(.8)
    # sad_clr = cmap(.5)
    # max_clr = cmap(.2)
    # minima = np.array([img._pos for img in obj.sources[src_index].images
    #                    if img.parity_name == 'min'])
    # saddles = np.array([img._pos for img in obj.sources[src_index].images
    #                     if img.parity_name == 'sad'])
    # maxima = np.array([img._pos for img in obj.sources[src_index].images
    #                    if img.parity_name == 'max'])

    # general contours
    mi = np.min(grid[src_index])
    cshift = 0.1 * abs(np.max(grid[src_index]) - clev[src_index][-1])
    ma = clev[src_index][-1] + cshift
    if contours:
        ax.contour3D(X, Y, grid[src_index], np.linspace(mi, ma, levels),
                     cmap=cmap)
        ax.set_zlim(mi, ma)
    if colorbar:
        plt.colorbar()
    # critical contours
    plt.contour(X, Y, grid[src_index], clev[src_index], extent=extent, origin='upper',
                colors=['k']*len(clev[src_index]))

    ax.view_init(70, 35)
    ax.grid(False)


def complex_ellipticity_plot(epsilon,
                             scatter=True, samples=-1, ensemble_averages=True,
                             color=None, colors=[GLEAMcolors.blue], alpha=1,
                             marker=None, markers=['o'],
                             markersize=None, markersizes=[4],
                             ls=None, lss=['-'],
                             contours=False, levels=10, cmap=None,
                             origin_marker=True, adjust_limits=True,
                             label=None, annotation_color='black', legend=None,
                             colorbar=False):
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
            clevels = list(set(clevels))
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
        plt.gca().set_aspect('equal')
    plt.xlabel(r'$\mathrm{\mathsf{Re\,\epsilon}}$')
    plt.ylabel(r'$\mathrm{\mathsf{Im\,\epsilon}}$')
    return plots
