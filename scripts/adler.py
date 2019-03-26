import os
import sys
root = os.path.dirname(os.path.dirname(os.path.abspath(os.path.realpath(__file__))))
sys.path.append(root)
import numpy as np
import time
import pickle
from functools import partial
from pathos.multiprocessing import ProcessingPool as Pool
from astropy.io import fits
import matplotlib.pyplot as plt

from gleam.reconsrc import synth_filter, synth_filter_mp
from gleam.multilens import MultiLens
from gleam.glass_interface import glass_renv
from gleam.utils.encode import an_sort
from gleam.utils.lensing import downsample_model, upsample_model, inertia_tensor, qpm_props, potential_grid, degarr_grid
from gleam.utils.linalg import sigma_product
glass = glass_renv()


class KeyboardInterruptError(Exception):
    pass


def residual_analysis(eagle_model, glass_state, method='e2g', verbose=False):
    """
    Compare the residuals of an EAGLE model to a GLASS model

    Args:
        eagle_model <(np.ndarray, astropy.io.fits.header.Header object)> - EAGLE .fits read output
        glass_state <glass.environment.Environment object> - a GLASS state

    Kwargs:
        method <str> - the method used, e.g. 'e2g' - downsample EAGLE model to match GLASS model
        verbose <bool> - verbose mode; print command line statements

    Return:
        kappa_resids <list> - residual for every ensemble model of the GLASS state
    """
    if verbose:
        print("{} models".format(len(glass_state.models)))
    kappa_resids = []
    if method == 'e2g':
        obj, _ = glass_state.models[0]['obj,data'][0]
        glass_maprad = obj.basis.top_level_cell_size * (obj.basis.pixrad)
        glass_extent = (-glass_maprad, glass_maprad, -glass_maprad, glass_maprad)
        glass_shape = (2*obj.basis.pixrad+1,)*2
        eagle_kappa, eagle_hdr = eagle_model
        eagle_kappa_map = downsample_model(eagle_kappa, glass_extent, glass_shape,
                                           pixel_scale=eagle_hdr['CDELT2']*3600)
        for m in glass_state.models:
            obj, data = m['obj,data'][0]
            glass_kappa_map = obj.basis._to_grid(data['kappa'], 1)
            # residuals
            resid_map = eagle_kappa_map - glass_kappa_map
            resid2_map = resid_map * resid_map
            r = np.sum(resid2_map)
            kappa_resids.append(r)
    elif method == 'g2e':
        eagle_kappa_map, eagle_hdr = eagle_model
        eagle_pixrad = tuple(r//2 for r in eagle_kappa_map.shape)
        eagle_maprad = eagle_pixrad[1]*eagle_hdr['CDELT2']*3600
        extent = [-eagle_maprad, eagle_maprad, -eagle_maprad, eagle_maprad]
        for m in glass_state.models:
            glass_kappa_map = upsample_model(m, extent, eagle_kappa_map.shape)
            # residuals
            resid_map = eagle_kappa_map - glass_kappa_map
            resid2_map = resid_map * resid_map
            r = np.sum(resid2_map)
            kappa_resids.append(r)
    else:
        return None
    if verbose:
        minidx = np.argmin(kappa_resids)
        print("Best fitting model index: {}".format(minidx))
        print("with residual {}".format(kappa_resids[minidx]))
    return kappa_resids


def inertia_analysis(eagle_model, glass_state, method='e2g', activation=None, verbose=False):
    """
    Calculate the tensor of inertia for an EAGLE and GLASS model

    Args:
        eagle_model <(np.ndarray, astropy.io.fits.header.Header object)> - EAGLE .fits read output
        glass_state <glass.environment.Environment object> - a GLASS state

    Kwargs:
        method <str> - the method used, e.g. 'e2g' - downsample EAGLE model to match GLASS model
        activation <float> - a threshold value below which pixel values are ignored
        verbose <bool> - verbose mode; print command line statements

    Return:
        inertias <np.matrix, list> - inertias for the EAGLE and every GLASS ensemble model
    """
    if verbose:
        print("{} models".format(len(glass_state.models)))
    inertias = []
    eagle_kappa_map, eagle_hdr = eagle_model
    eagle_pixel = eagle_hdr['CDELT2']*3600
    if method == 'e2g':
        obj, _ = glass_state.models[0]['obj,data'][0]
        glass_pixel = obj.basis.top_level_cell_size
        glass_maprad = glass_pixel * obj.basis.pixrad
        glass_extent = (-glass_maprad, glass_maprad, -glass_maprad, glass_maprad)
        glass_shape = (2*obj.basis.pixrad+1,)*2
        eagle_kappa_map = downsample_model(eagle_kappa_map, glass_extent, glass_shape,
                                           pixel_scale=eagle_pixel)
        eagle_inertia = inertia_tensor(eagle_kappa_map,
                                       pixel_scale=glass_pixel, activation=activation)
        for m in glass_state.models:
            obj, data = m['obj,data'][0]
            glass_kappa_map = obj.basis._to_grid(data['kappa'], 1)
            i = inertia_tensor(glass_kappa_map,
                               pixel_scale=glass_pixel, activation=activation)
            inertias.append(i)
    elif method == 'g2e':
        eagle_pixrad = tuple(r//2 for r in eagle_kappa_map.shape)
        eagle_maprad = eagle_pixrad[1]*eagle_hdr['CDELT2']*3600
        eagle_extent = [-eagle_maprad, eagle_maprad, -eagle_maprad, eagle_maprad]
        eagle_inertia = inertia_tensor(eagle_kappa_map,
                                       pixel_scale=eagle_pixel, activation=activation)
        for m in glass_state.models:
            glass_kappa_map = upsample_model(m, eagle_extent, eagle_kappa_map.shape)
            i = inertia_tensor(glass_kappa_map, activation=activation)
            inertias.append(i)
    else:
        return None
    if verbose:
        ea, eb, ephi = qpm_props(eagle_inertia)
        print("EAGLE [a, b, phi] = {:4.4f}, {:4.4f}, {:4.4f}".format(ea, eb, ephi*180/np.pi))
        props = [qpm_props(qpm) for qpm in inertias]
        a, b = np.mean([p[0] for p in props]), np.mean([p[1] for p in props])
        phi = np.mean([p[2] for p in props])
        print("GLASS [a, b, phi] = {:4.4f}, {:4.4f}, {:4.4f} avgs".format(a, b, phi*180/np.pi))
    return eagle_inertia, inertias


def potential_analysis(eagle_model, glass_state, method='e2g', N=85, verbose=False):
    """
    Calculate the potential grid of an EAGLE and GLASS model

    Args:
        eagle_model <(np.ndarray, astropy.io.fits.header.Header object)> - EAGLE .fits read output
        glass_state <glass.environment.Environment object> - a GLASS state

    Kwargs:
        method <str> - the method used, e.g. 'e2g' - downsample EAGLE model to match GLASS model
        N <int> - number of pixels sampled along an axis of the potential grid
        verbose <bool> - verbose mode; print command line statements

    Return:
        potentials <np.ndarray, list> - potential grid for EAGLE and every ensemble GLASS model
    """
    N_models = len(glass_state.models)
    if verbose:
        print("{} models".format(N_models))
    potentials = []
    eagle_kappa_map, eagle_hdr = eagle_model
    if method == 'e2g':
        obj, _ = glass_state.models[0]['obj,data'][0]
        eagle_pixel = eagle_hdr['CDELT2']*3600
        glass_maprad = obj.basis.top_level_cell_size * obj.basis.pixrad
        glass_extent = (-glass_maprad, glass_maprad, -glass_maprad, glass_maprad)
        glass_shape = (2*obj.basis.pixrad+1,)*2
        eagle_kappa_map = downsample_model(eagle_kappa_map, glass_extent, glass_shape,
                                           pixel_scale=eagle_pixel)
        eagle_pot = potential_grid(eagle_kappa_map, N, 2*glass_maprad, verbose=0)
        for i, m in enumerate(glass_state.models):
            obj, data = m['obj,data'][0]
            glass_kappa_map = obj.basis._to_grid(data['kappa'], 1)
            gx, gy, pot = potential_grid(glass_kappa_map, N, 2*glass_maprad, verbose=0)
            potentials.append((gx[:], gy[:], pot[:]))
            if verbose:
                message = "{:4d} / {:4d}\r".format(i+1, N_models)
                sys.stdout.write(message)
                sys.stdout.flush()
    elif method == 'g2e':
        eagle_pixrad = tuple(r//2 for r in eagle_kappa_map.shape)
        eagle_maprad = eagle_pixrad[1]*eagle_hdr['CDELT2']*3600
        eagle_extent = [-eagle_maprad, eagle_maprad, -eagle_maprad, eagle_maprad]
        eagle_pot = potential_grid(eagle_kappa_map, N, 2*eagle_maprad, verbose=0)
        for i, m in enumerate(glass_state.models):
            glass_kappa_map = upsample_model(m, eagle_extent, eagle_kappa_map.shape)
            gx, gy, pot = potential_grid(glass_kappa_map, N, 2*eagle_maprad, verbose=0)
            potentials.append((gx[:], gy[:], pot[:]))
            if verbose:
                message = "{:4d} / {:4d}\r".format(i+1, N_models)
                sys.stdout.write(message)
                sys.stdout.flush()
    else:
        return None
    if verbose:
        pass
    return eagle_pot, potentials


def degarr_analysis(eagle_model, glass_state, method='e2g', N=85, verbose=False):
    """
    Calculate the deg. arrival time surface of an EAGLE and GLASS model

    Args:
        eagle_model <(np.ndarray, astropy.io.fits.header.Header object)> - EAGLE .fits read output
        glass_state <glass.environment.Environment object> - a GLASS state

    Kwargs:
        method <str> - the method used, e.g. 'e2g' - downsample EAGLE model to match GLASS model
        N <int> - number of pixels sampled along an axis of the potential grid
        verbose <bool> - verbose mode; print command line statements

    Return:
        degarrs <np.ndarray, list> - deg. arrival surfaces for EAGLE and every ensemble GLASS model
    """
    N_models = len(glass_state.models)
    if verbose:
        print("{} models".format(N_models))
    degarrs = []
    eagle_kappa_map, eagle_hdr = eagle_model
    if method == 'e2g':
        obj, _ = glass_state.models[0]['obj,data'][0]
        eagle_pixel = eagle_hdr['CDELT2']*3600
        glass_maprad = obj.basis.top_level_cell_size * obj.basis.pixrad
        glass_extent = (-glass_maprad, glass_maprad, -glass_maprad, glass_maprad)
        glass_shape = (2*obj.basis.pixrad+1,)*2
        eagle_kappa_map = downsample_model(eagle_kappa_map, glass_extent, glass_shape,
                                           pixel_scale=eagle_pixel)
        egx, egy, eagle_degarr = degarr_grid(eagle_kappa_map, N, 2*glass_maprad, verbose=0)
        for i, m in enumerate(glass_state.models):
            obj, data = m['obj,data'][0]
            glass_kappa_map = obj.basis._to_grid(data['kappa'], 1)
            gx, gy, degarr = degarr_grid(glass_kappa_map, N, 2*glass_maprad, verbose=0)
            degarrs.append(degarr[:])
            if verbose:
                message = "{:4d} / {:4d}\r".format(i+1, N_models)
                sys.stdout.write(message)
                sys.stdout.flush()
    elif method == 'g2e':
        eagle_pixrad = tuple(r//2 for r in eagle_kappa_map.shape)
        eagle_maprad = eagle_pixrad[1]*eagle_hdr['CDELT2']*3600
        eagle_extent = [-eagle_maprad, eagle_maprad, -eagle_maprad, eagle_maprad]
        egx, egy, eagle_degarr = degarr_grid(eagle_kappa_map, N, 2*eagle_maprad, verbose=0)
        for i, m in enumerate(glass_state.models):
            glass_kappa_map = upsample_model(m, eagle_extent, eagle_kappa_map.shape)
            gx, gy, degarr = degarr_grid(glass_kappa_map, N, 2*eagle_maprad, verbose=0)
            degarrs.append(degarr[:])
            if verbose:
                message = "{:4d} / {:4d}\r".format(i+1, N_models)
                sys.stdout.write(message)
                sys.stdout.flush()
    else:
        return None
    if verbose:
        pass
    return egx, egy, eagle_degarr, degarrs


def degarr_single(index, gls, N=85, grid_size=1, N_total=0, verbose=False):
    """
    Helper function to evaluate the deg. arrival time surface of a single glass ensemble model
    within a multiprocessing loop

    Args:
        index <int> - the index of the GLASS ensemble model
        gls_model <glass.Environment object> - all GLASS ensemble models

    Kwargs:
        N <int> - number of pixels sampled along an axis of the potential grid
        grid_size <float> - the length of the grid along the axes of the kappa grid
        N_total <int> - the total size of loop range
        verbose <bool> - verbose mode; print command line statements

    Return:
        degarr <np.ndarray> - the deg. arrival time surface grid
    """
    obj, data = gls.models[index]['obj,data'][0]
    try:
        glass_kappa_map = obj.basis._to_grid(data['kappa'], 1)
        _, _, degarr = degarr_grid(glass_kappa_map, N, grid_size, verbose=0)
        if verbose:
            message = "{:4d} / {:4d}\r".format(index+1, N_total)
            sys.stdout.write(message)
            sys.stdout.flush()
        return degarr[:]
    except KeyboardInterrupt:
        raise KeyboardInterruptError()


def degarr_analysis_mp(eagle_model, glass_state, N=85, nproc=2, verbose=False):
    """
    Calculate the deg. arrival time surface of an EAGLE and GLASS model

    Args:
        eagle_model <(np.ndarray, astropy.io.fits.header.Header object)> - EAGLE .fits read output
        glass_state <glass.environment.Environment object> - a GLASS state

    Kwargs:
        method <str> - the method used, e.g. 'e2g' - downsample EAGLE model to match GLASS model
        N <int> - number of pixels sampled along an axis of the potential grid
        verbose <bool> - verbose mode; print command line statements

    Return:
        degarrs <np.ndarray, list> - deg. arrival surfaces for EAGLE and every ensemble GLASS model
    """
    N_models = len(glass_state.models)
    if verbose:
        print("{} models".format(N_models))
    degarrs = []
    # resample EAGLE model
    eagle_kappa_map, eagle_hdr = eagle_model
    obj, _ = glass_state.models[0]['obj,data'][0]
    eagle_pixel = eagle_hdr['CDELT2']*3600
    glass_maprad = obj.basis.top_level_cell_size * obj.basis.pixrad
    glass_extent = (-glass_maprad, glass_maprad, -glass_maprad, glass_maprad)
    glass_shape = (2*obj.basis.pixrad+1,)*2
    eagle_kappa_map = downsample_model(eagle_kappa_map, glass_extent, glass_shape,
                                       pixel_scale=eagle_pixel)
    egx, egy, eagle_degarr = degarr_grid(eagle_kappa_map, N, 2*glass_maprad, verbose=0)

    pool = Pool(processes=nproc)
    try:
        f = partial(degarr_single, gls=glass_state, N=N, grid_size=2*glass_maprad,
                    N_total=N_models, verbose=verbose)
        degarrs = pool.map(f, range(len(glass_state.models)))
        pool.clear()
    except KeyboardInterrupt:
        pool.terminate()

    if verbose:
        pass
    return egx, egy, eagle_degarr, degarrs


def iprod_analysis(eagle_degarr, ens_degarrs, verbose=False):
    """
    Calculate the inner products of an EAGLE model and each GLASS ensemble model

    Args:
        eagle_degarr <np.ndarray> - the deg. arrival time surface of the EAGLE model
        ens_degarrs <list(np.ndarray)> - the deg. arrival time surfaces of the GLASS ensemble

    Kwargs:
        verbose <bool> - verbose mode; print command line statements

    Return:
        iprods <list> - a list of inner products
    """
    iprods = []
    if verbose:
        print("{} models".format(len(ens_degarrs)))
    for i, d in enumerate(ens_degarrs):
        iprod = sigma_product(eagle_degarr, d)
        iprods.append(iprod)
    return iprods


def synth_loop(keys, jsons, states, save=False, optimized=False, verbose=False):
    """
    Args:
        keys <list(str)> - the keys which correspond to jsons' and states' keys
        jsons <dict> - json dictionary holding a list of filenames for each key
        states <dict> - state dictionary holding a list of filenames for each key

    Kwargs:
        save <bool> - save the filtered states automatically
        verbose <bool> - verbose mode; print command line statements

    Return:
        filtered_states
    """
    synthf = synth_filter
    if optimized:
        synthf = synth_filter_mp
    filtered_states = {}
    for k in keys:
        json = jsons[k][0]
        with open(json) as f:
            ml = MultiLens.from_json(f)
        for sf in states[k]:
            synths, _ = synthf(sf, ml, percentiles=[10, 25, 50], save=save, verbose=verbose)
            filtered_states[sf] = synths
    return filtered_states


def residual_loop(keys, kappa_files, states, method='e2g', verbose=False):
    """
    Args:
        keys <list(str)> - the keys which correspond to kappa_files' and states' keys
        kappa_files <dict> - kappa file dictionary holding a list of filenames for each key
        states <dict> - state dictionary holding a list of filenames for each key

    Kwargs:
        method <str> - the method used, e.g. 'e2g' - downsample EAGLE model to match GLASS model
        verbose <bool> - verbose mode; print command line statements

    Return:
        resids <dict> - a dictionary of residuals
    """
    resids = {}
    for k in keys:
        kappafile = kappa_files[k][0]
        eagle_model = fits.getdata(kappafile, header=True)
        for idx in range(len(states[k])):
            statefile = states[k][idx]
            if verbose:
                print("Comparing {}   vs   {}".format(kappafile, statefile))
            glass_state = glass.glcmds.loadstate(statefile)
            r = residual_analysis(eagle_model, glass_state, method=method, verbose=verbose)
            resids[statefile] = r
    return resids


def inertia_loop(keys, kappa_files, states, method='e2g', activation=None, verbose=False):
    """
    Args:
        keys <list(str)> - the keys which correspond to kappa_files' and states' keys
        kappa_files <dict> - kappa file dictionary holding a list of filenames for each key
        states <dict> - state dictionary holding a list of filenames for each key

    Kwargs:
        method <str> - the method used, e.g. 'e2g' - downsample EAGLE model to match GLASS model
        verbose <bool> - verbose mode; print command line statements

    Return:
        resids <dict> - a dictionary of residuals
    """
    inertias = {}
    for k in keys:
        kappafile = kappa_files[k][0]
        eagle_model = fits.getdata(kappafile, header=True)
        for idx in range(len(states[k])):
            statefile = states[k][idx]
            if verbose:
                print("Comparing {}   vs   {}".format(kappafile, statefile))
            glass_state = glass.glcmds.loadstate(statefile)
            qpms = inertia_analysis(eagle_model, glass_state, method=method,
                                    activation=activation, verbose=verbose)
            inertias[statefile] = qpms
    return inertias


def potential_loop(keys, kappa_files, states, method='e2g', N=85, verbose=False):
    """
    Args:
        keys <list(str)> - the keys which correspond to kappa_files' and states' keys
        kappa_files <dict> - kappa file dictionary holding a list of filenames for each key
        states <dict> - state dictionary holding a list of filenames for each key

    Kwargs:
        method <str> - the method used, e.g. 'e2g' - downsample EAGLE model to match GLASS model
        N <int> - number of pixels sampled along an axis of the potential grid
        verbose <bool> - verbose mode; print command line statements

    Return:
        potential_grids <dict> - a dictionary of potential grids
    """
    potential_grids = {}
    for k in keys:
        kappafile = kappa_files[k][0]
        eagle_model = fits.getdata(kappafile, header=True)
        for idx in range(len(states[k])):
            statefile = states[k][idx]
            if verbose:
                print("Comparing {}   vs   {}".format(kappafile, statefile))
            glass_state = glass.glcmds.loadstate(statefile)
            potentials = potential_analysis(eagle_model, glass_state, method=method,
                                            N=N, verbose=verbose)
            potential_grids[statefile] = potentials
    return potential_grids


def degarr_loop(keys, kappa_files, states, method='e2g', N=85,
                optimized=False, calc_iprod=True, verbose=False):
    """
    Args:
        keys <list(str)> - the keys which correspond to kappa_files' and states' keys
        kappa_files <dict> - kappa file dictionary holding a list of filenames for each key
        states <dict> - state dictionary holding a list of filenames for each key

    Kwargs:
        method <str> - the method used, e.g. 'e2g' - downsample EAGLE model to match GLASS model
        N <int> - number of pixels sampled along an axis of the potential grid
        calc_iprod <bool> - calculate the inner product of the grids immediately
        verbose <bool> - verbose mode; print command line statements

    Return:
        degarr_grids <dict> - a dictionary of deg. arrival surface grids
    """
    degarr_grids = {}
    iprods = {}
    for k in keys:
        kappafile = kappa_files[k][0]
        eagle_model = fits.getdata(kappafile, header=True)
        for idx in range(len(states[k])):
            statefile = states[k][idx]
            if verbose:
                print("Comparing {}   vs   {}".format(kappafile, statefile))
            glass_state = glass.glcmds.loadstate(statefile)
            if optimized:
                gx, gy, eagle_degarr, degarrs = degarr_analysis_mp(
                    eagle_model, glass_state, N=N, verbose=verbose)
            else:
                gx, gy, eagle_degarr, degarrs = degarr_analysis(
                    eagle_model, glass_state, method=method, N=N, verbose=verbose)
            if calc_iprod:
                iprods[statefile] = iprod_analysis(eagle_degarr, degarrs)
            degarr_grids[statefile] = (gx, gy, eagle_degarr, degarrs)
    if calc_iprod:
        return degarr_grids, iprods
    return degarr_grids


if __name__ == "__main__":
    # root directories
    rdir = "/Users/phdenzel/adler"
    jsondir = rdir+"/json/"
    statedir = rdir+"/states/v3/"
    kappadir = rdir+"/kappa/"
    keys = ["H1S0A0B90G0", "H1S1A0B90G0", "H2S1A0B90G0", "H2S2A0B90G0", "H2S7A0B90G0",
            "H3S0A0B90G0", "H3S1A0B90G0", "H4S3A0B0G90", "H10S0A0B90G0", "H13S0A0B90G0",
            "H23S0A0B90G0", "H30S0A0B90G0", "H36S0A0B90G0", "H160S0A90B0G0", "H234S0A0B90G0"]

    # list all files within the directories
    ls_jsons = an_sort([os.path.join(jsondir, f) for f in os.listdir(jsondir)
                        if f.endswith('.json')])
    ls_states = an_sort([os.path.join(statedir, f) for f in os.listdir(statedir)
                         if f.endswith('.state')])
    ls_kappas = an_sort([os.path.join(kappadir, f) for f in os.listdir(kappadir)
                         if f.endswith('.kappa.fits')])

    # file dictionaries
    jsons = {k: [f for f in ls_jsons if k in f] for k in keys}
    filtered_states = {k: [f for f in ls_states
                           if k in f and f.endswith('_filtered.state')] for k in keys}
    ls_states = [f for f in ls_states if not f.endswith('_filtered.state')]
    prefiltered_synthf10 = {k: [f for f in ls_states
                                if k in f and f.endswith('_filtered_synthf10.state')]
                            for k in keys}
    prefiltered_synthf25 = {k: [f for f in ls_states
                                if k in f and f.endswith('_filtered_synthf25.state')]
                            for k in keys}
    prefiltered_synthf50 = {k: [f for f in ls_states
                                if k in f and f.endswith('_filtered_synthf50.state')]
                            for k in keys}
    ls_states = [f for f in ls_states if not (f.endswith('_filtered_synthf10.state')
                                              or f.endswith('_filtered_synthf25.state')
                                              or f.endswith('_filtered_synthf50.state'))]
    synthf10 = {k: [f for f in ls_states
                    if k in f and f.endswith('_synthf10.state')] for k in keys}
    synthf25 = {k: [f for f in ls_states
                    if k in f and f.endswith('_synthf25.state')] for k in keys}
    synthf50 = {k: [f for f in ls_states
                    if k in f and f.endswith('_synthf50.state')] for k in keys}
    ls_states = [f for f in ls_states if not (f.endswith('_synthf10.state')
                                              or f.endswith('_synthf25.state')
                                              or f.endswith('_synthf50.state'))]
    states = {k: [f for f in ls_states if k in f] for k in keys}
    kappa_files = {k: [f for f in ls_kappas if k in f] for k in keys}

    sfiles = states  # filtered_states  # synthf50  # prefiltered_synthf50

    # synth filtering
    if 0:
        kwargs = dict(save=False, optimized=True, verbose=1)
        sfiles = states  # filtererd_states
        synth_filtered_states = synth_loop(keys, jsons, sfiles, **kwargs)
        print(synth_filtered_states)

    # residual analysis
    if 0:
        k = ["H3S0A0B90G0", "H10S0A0B90G0", "H36S0A0B90G0"]
        kwargs = dict(method='e2g', verbose=1)
        # sfiles = synthf50  # prefiltered_synthf50  # states  # filtered_states
        residuals = residual_loop(k, kappa_files, sfiles, **kwargs)
        print(len(residuals))

    # inertia tensor analysis
    if 0:
        k = ["H3S0A0B90G0", "H10S0A0B90G0", "H36S0A0B90G0"]
        kwargs = dict(method='e2g', activation=0.25, verbose=1)
        # sfiles = synthf50  # prefiltered_synthf50  # states  # filtered_states
        qpms = inertia_loop(k, kappa_files, sfiles, **kwargs)
        with open('qpms.pkl', 'wb') as f:
            pickle.dump(qpms, f)
    if 0:
        k = ["H3S0A0B90G0", "H10S0A0B90G0", "H36S0A0B90G0"]
        # sfiles = synthf50  # prefiltered_synthf50  # states  # filtered_states
        with open('qpms.pkl', 'r') as f:
            qpms = pickle.load(f)
        # iprod histograms
        for ki in k:
            files = sfiles[ki]
            for f in files:
                lbl = ['a', 'b', 'phi']
                eq, gq = qpms[f]
                ea, eb, ephi = qpm_props(eq)
                gprops = [qpm_props(q) for q in gq]
                ga, gb, gphi = [p[0] for p in gprops], \
                               [p[1] for p in gprops], \
                               [p[2] for p in gprops]
                name = "".join(os.path.basename(f).split('.')[:-1])
                for i, (eprop, gprop) in enumerate(zip([ea, eb, ephi], [ga, gb, gphi])):
                    plt.hist(gprop, bins=14)
                    plt.axvline(eprop)
                    plt.title(lbl[i]+' '+name)
                    plt.savefig(lbl[i]+' '+name+'_hist.png')
                    plt.close()
                    # plt.show()

    # potential analysis
    if 0:
        k = ["H3S0A0B90G0", "H10S0A0B90G0", "H36S0A0B90G0"]
        kwargs = dict(N=85, verbose=1)
        # sfiles = synthf50  # prefiltered_synthf50  # states  # filtered_states
        gx, gy, potentials = potential_loop(k, kappa_files, sfiles, **kwargs)
        with open('pots.pkl', 'wb') as f:
            pickle.dump(potentials, f)

    # deg. arrival time surface analysis
    if 0:
        k = ["H3S0A0B90G0", "H10S0A0B90G0", "H36S0A0B90G0"]
        kwargs = dict(N=85, calc_iprod=True, optimized=True, verbose=1)
        # sfiles = synthf50  # prefiltered_synthf50  # states  # filtered_states
        degarrs, iprods = degarr_loop(k, kappa_files, sfiles, **kwargs)
        with open('degarrs.pkl', 'wb') as f:
            pickle.dump(degarrs, f)
        with open('iprods.pkl', 'wb') as f:
            pickle.dump(iprods, f)
    if 0:
        k = ["H3S0A0B90G0", "H10S0A0B90G0", "H36S0A0B90G0"]
        # sfiles = synthf50  # prefiltered_synthf50  # states  # filtererd_states
        with open('degarrs.pkl', 'r') as f:
            degarrs = pickle.load(f)
        # with open('iprods.pkl', 'r') as f:
        #     iprods = pickle.load(f)

        # degarr product histograms
        for ki in k:
            files = sfiles[ki]
            for f in files:
                gx, gy, eagle_degarr, glass_degarrs = degarrs[f]
                ip = [sigma_product(eagle_degarr[2], gdegarr[2]) for gdegarr in glass_degarrs]
                name = "".join(os.path.basename(f).split('.')[:-1])
                plt.hist(ip, bins=14)
                plt.title(name)
                plt.savefig('iprod'+' '+name+'_hist.png')
                plt.close()
                # plt.show()

        # best degarr grid plot
        for ki in k:
            files = sfiles[ki]
            for f in files:
                gx, gy, eagle_degarr, glass_degarrs = degarrs[f]
                ip = [sigma_product(eagle_degarr, gdegarr) for gdegarr in glass_degarrs]
                name = "".join(os.path.basename(f).split('.')[:-1])
                minidx = np.argmax(ip)
                da = glass_degarrs[minidx]
                lim = max(abs(da.min()), abs(da.max()))
                plt.contourf(gx, gy, da, levels=np.linspace(da.min(), da.max(), 14),
                             vmin=-lim, vmax=lim, cmap='RdBu')
                plt.colorbar()
                plt.title(name)
                plt.gca().set_aspect('equal')
                plt.savefig('degarr_best'+' '+name+'.png')
                plt.close()
