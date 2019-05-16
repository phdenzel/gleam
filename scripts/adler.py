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

from gleam.reconsrc import ReconSrc, synth_filter, synth_filter_mp
from gleam.multilens import MultiLens
from gleam.glass_interface import glass_renv
from gleam.utils.encode import an_sort
from gleam.utils.lensing import downsample_model, upsample_model, inertia_tensor, qpm_props, potential_grid, degarr_grid
from gleam.utils.linalg import sigma_product
from gleam.utils.makedir import mkdir_p
glass = glass_renv()


class KeyboardInterruptError(Exception):
    pass


def mkdir_structure(keys, root=None):
    """
    Create a directory structure
    """
    for k in keys:
        d = os.path.join(root, k)
        mkdir_p(d)


def chi2_analysis(reconsrc, optimized=False, verbose=False):
    """
    Args:
        reconsrc <ReconSrc object> - loaded object, preferentially with loaded cache

    Kwargs:
        optimized <bool> - run the multiprocessing synth filter
        verbose <bool> - verbose mode; print command line statements

    Return:
        chi2 <list(float)> - non-reduced chi2 values
    """
    synthf = synth_filter
    if optimized:
        synthf = synth_filter_mp
    # Construct the noise map
    lo = reconsrc.lensobject
    signals, variances = lo.flatfield(lo.data, size=0.2)
    gain, _ = lo.gain(signals=signals, variances=variances)
    f = gain
    bias = 0.01*np.max(f * lo.data)
    sgma2 = lo.sigma2(f=f, add_bias=bias)
    dta_noise = np.random.normal(0, 1, size=lo.data.shape)
    dta_noise = dta_noise * np.sqrt(sgma2)
    # recalculate the chi2s
    kwargs = dict(reconsrc=reconsrc, percentiles=[],
                  noise=dta_noise, sigma2=sgma2,
                  reduced=False, return_obj=False, save=False, verbose=verbose)
    _, _, chi2 = synthf(**kwargs)
    return chi2


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
        s = sigma_product(eagle_degarr, d)
        iprods.append(s)
    return iprods


def synth_loop(keys, jsons, states, cached=False, save_state=False, save_obj=False, load_obj=False,
               path=None, optimized=False, verbose=False):
    """
    Args:
        keys <list(str)> - the keys which correspond to jsons' and states' keys
        jsons <dict> - json dictionary holding a list of filenames for each key
        states <dict> - state dictionary holding a list of filenames for each key

    Kwargs:
        save_state <bool> - save the filtered states automatically
        save_obj <bool> - save the ReconSrc object as a pickle file
        load_obj <bool> - load the ReconSrc object from a pickle file
        optimized <bool> - run the multiprocessing synth filter
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
        # generate noise map
        signals, variances = ml[0].flatfield(ml[0].data, size=0.2)
        gain, _ = ml[0].gain(signals=signals, variances=variances)
        f = gain
        bias = 0.01*np.max(f * ml[0].data)
        sgma2 = ml[0].sigma2(f=f, add_bias=bias)
        dta_noise = np.random.normal(0, 1, size=ml[0].data.shape)
        dta_noise = dta_noise * np.sqrt(sgma2)
        for sf in states[k]:
            if verbose:
                print(sf)
            if load_obj:
                loadname = "reconsrc_{}".format(os.path.basename(sf).replace(".state", ".pkl"))
                if path is None:
                    path = ""
                if os.path.exists(path + k):
                    loadname = os.path.join(path, k, loadname)
                elif os.path.exists(path):
                    loadname = os.path.join(path, loadname)
                if os.path.exists(loadname):
                    with open(loadname, 'rb') as f:
                        recon_src = pickle.load(f)
                        if verbose:
                            print(recon_src.__v__)
                else:
                    recon_src = ReconSrc(ml, sf, M=2*ml[0].naxis1, verbose=verbose)
            else:
                recon_src = ReconSrc(ml, sf, M=2*ml[0].naxis1, verbose=verbose)
            if save_obj:
                recon_src = synthf(reconsrc=recon_src, percentiles=[],
                                   noise=dta_noise, sigma2=sgma2,
                                   return_obj=save_obj, save=False, verbose=verbose)
                # save the object
                savename = "reconsrc_{}".format(os.path.basename(sf).replace(".state", ".pkl"))
                if path is None:
                    path = ""
                if os.path.exists(path+k):
                    savename = os.path.join(path, k, savename)
                elif os.path.exists(path):
                    savename = os.path.join(path, savename)
                with open(savename, 'wb') as f:
                    pickle.dump(recon_src, f)
            else:
                recon_src = synthf(reconsrc=recon_src, percentiles=[],
                                   noise=dta_noise, sigma2=sgma2,
                                   return_obj=save_obj, save=False, verbose=verbose)
            if save_state:
                synthf(reconsrc=recon_src, percentiles=[],
                       noise=dta_noise, sigma2=sgma2,
                       save=save_state, verbose=verbose)
    return filtered_states


def cache_loop(keys, jsons, states, path=None, variables=['inv_proj', 'N_nil', 'reproj_d_ij'],
               save_obj=False, verbose=False):
    """
    Args:
        keys <list(str)> - the keys which correspond to jsons' and states' keys
        jsons <dict> - json dictionary holding a list of filenames for each key
        states <dict> - state dictionary holding a list of filenames for each key

    Kwargs:
        save_state <bool> - save the filtered states automatically
        save_obj <bool> - save the ReconSrc object as a pickle file
        load_obj <bool> - load the ReconSrc object from a pickle file
        optimized <bool> - run the multiprocessing synth filter
        verbose <bool> - verbose mode; print command line statements

    Return:
        filtered_states
    """
    for k in keys:
        json = jsons[k][0]
        with open(json) as f:
            ml = MultiLens.from_json(f)
        for sf in states[k]:
            loadname = "reconsrc_{}".format(os.path.basename(sf).replace(".state", ".pkl"))
            if path is None:
                path = ""
            if os.path.exists(path + k):
                loadname = os.path.join(path, k, loadname)
            elif os.path.exists(path):
                loadname = os.path.join(path, loadname)
            if verbose:
                print(loadname)
            if os.path.exists(loadname):
                with open(loadname, 'rb') as f:
                    old_recon_src = pickle.load(f)
            new_recon_src = ReconSrc(ml, sf, M=2*ml[0].naxis1, verbose=verbose)
            new_recon_src.update_cache(old_recon_src._cache, variables=variables)
            # print(new_recon_src._cache)
            if save_obj:
                if os.path.exists(loadname):
                    with open(loadname, 'wb') as f:
                        pickle.dump(new_recon_src, f)


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
    version = "v2/"
    jsondir = rdir+"/json/"
    statedir = rdir+"/states/" + version
    anlysdir = rdir+"/analysis/" + version
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

    # # create a directory structure
    if 0:
        mkdir_structure(keys, root=anlysdir+"states/")

    # # reload cache into new reconsrc objects
    if 0:
        k = ["H3S0A0B90G0", "H10S0A0B90G0", "H36S0A0B90G0"]
        # k = keys
        kwargs = dict(path=anlysdir+"states/", variables=['inv_proj', 'N_nil'], save_obj=True,
                      verbose=1)
        sfiles = states  # filtered_states
        cache_loop(k, jsons, sfiles, **kwargs)

    # # reconsrc synth caching
    if 0:
        k = ["H3S0A0B90G0", "H10S0A0B90G0", "H36S0A0B90G0"]
        # k = ["H1S1A0B90G0", "H2S1A0B90G0", "H2S2A0B90G0", "H2S7A0B90G0",
        #      "H3S1A0B90G0", "H4S3A0B0G90", "H13S0A0B90G0", "H23S0A0B90G0", "H30S0A0B90G0",
        #      "H160S0A90B0G0", "H234S0A0B90G0"]
        # k = keys
        kwargs = dict(save_state=False, save_obj=True, load_obj=True, path=anlysdir+"states/",
                      optimized=True, verbose=1)
        # kwargs = dict(save_state=True, save_obj=False, load_obj=True, path=anlysdir+"states/",
        #               optimized=False, verbose=1)
        sfiles = states  # filtered_states
        synth_filtered_states = synth_loop(k, jsons, sfiles, **kwargs)

    # # chi2 histograms (takes long!!!)
    if 0:
        k = ["H3S0A0B90G0", "H10S0A0B90G0", "H36S0A0B90G0"]
        # k = ["H1S0A0B90G0", "H1S1A0B90G0", "H2S1A0B90G0", "H2S2A0B90G0", "H2S7A0B90G0",
        #      "H3S1A0B90G0", "H4S3A0B0G90", "H13S0A0B90G0", "H23S0A0B90G0", "H30S0A0B90G0",
        #      "H160S0A90B0G0", "H234S0A0B90G0"]
        kwargs = dict(optimized=False, verbose=1)
        sfiles = states  # filtered_states
        path = anlysdir+"states/"
        for ki in k:
            for sf in sfiles[ki]:
                name = os.path.basename(sf).replace(".state", "")
                loadname = "reconsrc_{}.pkl".format(name)
                if path is None:
                    path = ""
                if os.path.exists(path + ki):
                    loadname = os.path.join(path, ki, loadname)
                elif os.path.exists(path):
                    loadname = os.path.join(path, loadname)
                if os.path.exists(loadname):
                    with open(loadname, 'rb') as f:
                        recon_src = pickle.load(f)
                if kwargs.get('verbose', False):
                    print('Loading '+loadname)
                # gather chi2 values
                chi2 = chi2_analysis(recon_src, **kwargs)
                sortedidcs = np.argsort(chi2)
                plt.hist(chi2, bins=20, color='#A0DED2')
                plt.title(name)
                # save the figure
                savename = "chi2_hist_{}.pdf".format(name)
                textname = 'chi2_{}.txt'.format(name)
                if path is None:
                    path = ""
                if os.path.exists(path + ki):
                    savename = os.path.join(path, ki, savename)
                    textname = os.path.join(path, ki, textname)
                elif os.path.exists(path):
                    savename = os.path.join(path, savename)
                    textname = os.path.join(path, textname)
                if kwargs.get('verbose', False):
                    print('Saving '+savename)
                    print('Saving '+textname)
                np.savetxt(textname, np.c_[chi2, sortedidcs], fmt='%12d', delimiter=' ', newline=os.linesep)
                plt.savefig(savename)
                # plt.show()
                plt.close()

    # # kappa diff analysis
    if 0:
        k = ["H3S0A0B90G0", "H10S0A0B90G0", "H36S0A0B90G0"]
        kwargs = dict(method='e2g', verbose=1)
        sfiles = states  # synthf50  # prefiltered_synthf50  # filtered_states
        residuals = residual_loop(k, kappa_files, sfiles, **kwargs)

    # # kappa diff histograms
    if 0:
        k = ["H3S0A0B90G0", "H10S0A0B90G0", "H36S0A0B90G0"]
        kwargs = dict(method='e2g', verbose=1)
        sfiles = states  # synthf50  # prefiltered_synthf50  # filtered_states
        path = anlysdir+"states/"
        residuals = residual_loop(k, kappa_files, sfiles, **kwargs)
        for ki in k:
            print(ki)
            for idx in range(len(states[ki])):
                sf = states[ki][idx]
                name = os.path.basename(sf).replace(".state", "")
                savename = "kappa_diff_hist_{}.pdf".format(name)
                if path is None:
                    path = ""
                if os.path.exists(path + ki):
                    savename = os.path.join(path, ki, savename)
                elif os.path.exists(path):
                    savename = os.path.join(path, savename)
                if kwargs.get('verbose', False):
                    print(savename)
                plt.hist(residuals[sf], color='#386BF1')
                plt.xlim(left=0, right=200)
                plt.ylim(bottom=0, top=375)
                plt.title(name)
                plt.savefig(savename)
                plt.close()

    # # inertia tensor analysis
    if 0:
        k = ["H3S0A0B90G0", "H10S0A0B90G0", "H36S0A0B90G0"]
        kwargs = dict(method='e2g', activation=0.25, verbose=1)
        sfiles = states  # synthf50  # prefiltered_synthf50  # filtered_states
        path = anlysdir+"states/"
        qpms = inertia_loop(k, kappa_files, sfiles, **kwargs)
        if 1:
            savename = 'qpms.pkl'
            if path is None:
                path = ""
            elif os.path.exists(path):
                savename = os.path.join(path, savename)
            with open(savename, 'wb') as f:
                print("Saving "+savename)
                pickle.dump(qpms, f)

    # # inertia histograms
    if 0:
        k = ["H3S0A0B90G0", "H10S0A0B90G0", "H36S0A0B90G0"]
        kwargs = dict(verbose=1)
        sfiles = states  # synthf50  # prefiltered_synthf50  # filtered_states
        path = anlysdir+"states/"
        loadname = 'qpms.pkl'
        if path is None:
            path = ""
        elif os.path.exists(path):
            loadname = os.path.join(path, loadname)
        with open(loadname, 'rb') as f:
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
                name = os.path.basename(f).replace(".state", "")
                for i, (eprop, gprop) in enumerate(zip([ea, eb, ephi], [ga, gb, gphi])):
                    savename = lbl[i]+"_hist_{}.pdf".format(name)
                    if path is None:
                        path = ""
                    if os.path.exists(path + ki):
                        savename = os.path.join(path, ki, savename)
                    elif os.path.exists(path):
                        savename = os.path.join(path, savename)
                    if kwargs.get('verbose', False):
                        print(savename)
                    plt.hist(gprop, bins=14, color='#FF6767')
                    plt.axvline(eprop, color='#F52549')
                    if lbl[i] == 'phi':
                        plt.xlim(left=0, right=2*np.pi)
                    # if lbl[i] == 'a' or lbl[i] == 'b':
                    #     plt.xlim(left=(2*max(eprop, np.max(gprop))-2), right=2)
                    # plt.ylim(bottom=0, top=200)
                    plt.title(name)
                    plt.savefig(savename)
                    plt.close()

    # # potential analysis
    if 0:
        k = ["H3S0A0B90G0", "H10S0A0B90G0", "H36S0A0B90G0"]
        kwargs = dict(N=85, verbose=1)
        sfiles = states  # synthf50  # prefiltered_synthf50  # filtered_states
        path = anlysdir+"states/"
        potentials = potential_loop(k, kappa_files, sfiles, **kwargs)
        if 1:
            savename = 'pots.pkl'
            if path is None:
                path = ""
            elif os.path.exists(path):
                savename = os.path.join(path, savename)
            with open(savename, 'wb') as f:
                print("Saving "+savename)
                pickle.dump(potentials, f)

    # # potential maps
    if 0:
        k = ["H3S0A0B90G0", "H10S0A0B90G0", "H36S0A0B90G0"]
        kwargs = dict(verbose=1)
        sfiles = states  # synthf50  # prefiltered_synthf50  # filtered_states
        path = anlysdir+"states/"
        loadname = 'pots.pkl'
        if path is None:
            path = ""
        elif os.path.exists(path):
            loadname = os.path.join(path, loadname)
        with open(loadname, 'rb') as f:
            pots = pickle.load(f)
        for ki in k:
            print(ki)
            for idx in range(len(states[ki])):
                sf = states[ki][idx]
                print(sf)
                name = os.path.basename(sf).replace(".state", "")
                egx, egy, eagle_map = pots[sf][0]
                gls_models = pots[sf][1]
                gx, gy = [g[0] for g in gls_models], [g[1] for g in gls_models]
                potentials = [g[2] for g in gls_models]
                ens_avg = np.average(potentials, axis=0)
                # start plotting
                fig, axes = plt.subplots(1, 2, figsize=(15, 5))
                pltkw = dict(cmap='magma', vmin=eagle_map.min(), vmax=eagle_map.max(),
                             levels=np.linspace(np.min((ens_avg, eagle_map)), np.max((ens_avg, eagle_map)), 25))
                eimg = axes[0].contourf(egx, egy, eagle_map, **pltkw)
                axes[0].set_title('EAGLE model', fontsize=12)
                gimg = axes[1].contourf(gx[0], gy[0], ens_avg[::-1, :], **pltkw)
                axes[1].set_title('Ensemble average', fontsize=12)
                plt.colorbar(eimg, ax=axes.ravel().tolist(), shrink=0.9)
                axes[0].set_aspect('equal')
                axes[1].set_aspect('equal')
                plt.suptitle(name)
                # save the figure
                savename = "potential_{}.pdf".format(name)
                if path is None:
                    path = ""
                if os.path.exists(path + ki):
                    savename = os.path.join(path, ki, savename)
                elif os.path.exists(path):
                    savename = os.path.join(path, savename)
                if kwargs.get('verbose', False):
                    print(savename)
                plt.savefig(savename)
                # plt.show()
                plt.close()

    # # deg. arrival time surfaces and inner products
    if 0:
        k = ["H3S0A0B90G0", "H10S0A0B90G0", "H36S0A0B90G0"]
        kwargs = dict(N=85, calc_iprod=True, optimized=True, verbose=1)
        sfiles = states  # synthf50  # prefiltered_synthf50  # filtered_states
        path = anlysdir+"states/"
        degarrs, iprods = degarr_loop(k, kappa_files, sfiles, **kwargs)
        if 1:
            savenames = ['degarrs.pkl', 'iprods.pkl']
            for o, savename in zip([degarrs, iprods], ['degarrs.pkl', 'iprods.pkl']):
                if path is None:
                    path = ""
                elif os.path.exists(path):
                    savename = os.path.join(path, savename)
                with open(savename, 'wb') as f:
                    print("Saving " + savename)
                    pickle.dump(o, f)

    # # deg. arrival time surface histograms
    if 0:
        k = ["H3S0A0B90G0", "H10S0A0B90G0", "H36S0A0B90G0"]
        kwargs = dict(verbose=1)
        sfiles = states  # synthf50  # prefiltered_synthf50  # filtererd_states
        path = anlysdir+"states/"
        o = []
        for loadname in ['degarrs.pkl', 'iprods.pkl']:
            if path is None:
                path = ""
            elif os.path.exists(path):
                loadname = os.path.join(path, loadname)
            with open(loadname, 'rb') as f:
                o.append(pickle.load(f))
        degarrs, iprods = o
        for ki in k:
            files = sfiles[ki]
            for sf in files:
                gx, gy, eagle_degarr, glass_degarrs = degarrs[sf]
                # ip = [sigma_product(eagle_degarr, gdegarr) for gdegarr in glass_degarrs]
                ip = iprods[sf]
                name = os.path.basename(sf).replace(".state", "")
                plt.hist(ip, bins=14, color='#4FB09E')
                plt.xlim(left=-1, right=1)
                plt.title(name)
                # save the figure
                savename = "scalar_degarr_{}.pdf".format(name)
                if path is None:
                    path = ""
                if os.path.exists(path + ki):
                    savename = os.path.join(path, ki, savename)
                elif os.path.exists(path):
                    savename = os.path.join(path, savename)
                if kwargs.get('verbose', False):
                    print(savename)
                plt.savefig(savename)
                # plt.show()
                plt.close()

    # # deg. arrival time surface maps
    if 0:
        k = ["H3S0A0B90G0", "H10S0A0B90G0", "H36S0A0B90G0"]
        kwargs = dict(verbose=1)
        sfiles = states  # synthf50  # prefiltered_synthf50  # filtererd_states
        path = anlysdir+"states/"
        loadnames = ['degarrs.pkl', 'iprods.pkl']
        if path is None:
            path = ""
        elif os.path.exists(path):
            loadnames = [os.path.join(path, l) for l in loadnames]
        with open(loadnames[0], 'rb') as f1:
            degarrs = pickle.load(f1)
        with open(loadnames[1], 'rb') as f2:
            iprods = pickle.load(f2)
        for ki in k:
            files = sfiles[ki]
            for sf in files:
                gx, gy, eagle_degarr, glass_degarrs = degarrs[sf]
                # ip = [sigma_product(eagle_degarr, gdegarr) for gdegarr in glass_degarrs]
                ip = iprods[sf]
                sortedidcs = np.argsort(ip)
                name = os.path.basename(sf).replace(".state", "")
                ens_avg = np.average(glass_degarrs, axis=0)
                minidx = sortedidcs[-1]
                min_model = glass_degarrs[minidx]
                maxidx = np.argmin(ip)
                max_model = glass_degarrs[maxidx]
                fig, axes = plt.subplots(2, 2, figsize=(15, 12))
                lim = np.max([np.abs(eagle_degarr), np.abs(ens_avg),
                              np.abs(min_model), np.abs(max_model)])
                pltkw = dict(cmap='RdBu', vmin=-lim, vmax=lim, levels=np.linspace(-lim, lim, 25))
                img0 = axes[0, 0].contourf(gx, gy, eagle_degarr, **pltkw)
                axes[0, 0].set_title('EAGLE model', fontsize=12)
                axes[0, 0].set_aspect('equal')
                img1 = axes[0, 1].contourf(gx, gy, ens_avg, **pltkw)
                axes[0, 1].set_title('Ensemble average', fontsize=12)
                axes[0, 1].set_aspect('equal')
                img2 = axes[1, 0].contourf(gx, gy, min_model, **pltkw)
                axes[1, 0].set_title('Best model', fontsize=12)
                axes[1, 0].set_aspect('equal')
                img3 = axes[1, 1].contourf(gx, gy, max_model, **pltkw)
                axes[1, 1].set_title('Worst model', fontsize=12)
                axes[1, 1].set_aspect('equal')
                plt.colorbar(img3, ax=axes.ravel().tolist(), shrink=0.9)
                plt.suptitle(name)
                # save the figure
                savename = "degarrs_{}.pdf".format(name)
                textname = "iprods_{}.txt".format(name)
                if path is None:
                    path = ""
                if os.path.exists(path + ki):
                    savename = os.path.join(path, ki, savename)
                    textname = os.path.join(path, ki, textname)
                elif os.path.exists(path):
                    savename = os.path.join(path, savename)
                    textname = os.path.join(path, textname)
                if kwargs.get('verbose', False):
                    print(savename)
                    print("Best/worst model of {}: {:4d}/{:4d}".format(name, minidx, maxidx))
                    print(textname)
                np.savetxt(textname, np.c_[ip, sortedidcs], fmt='%4.4g', delimiter=' ', newline=os.linesep)
                plt.savefig(savename)
                # plt.show()
                plt.close()

    # # chi2 vs iprods (takes long!!!)
    if 0:
        k = ["H3S0A0B90G0", "H10S0A0B90G0", "H36S0A0B90G0"]
        kwargs = dict(optimized=True, verbose=1)
        sfiles = states  # filtered_states
        path = anlysdir+"states/"
        loadname = 'iprods.pkl'
        if path is None:
            path = ""
        elif os.path.exists(path):
            loadname = os.path.join(path, loadname)
        with open(loadname, 'rb') as f:
            iprods = pickle.load(f)
        for ki in k:
            for sf in sfiles[ki]:
                name = os.path.basename(sf).replace(".state", "")
                loadname = "reconsrc_{}.pkl".format(name)
                if path is None:
                    path = ""
                if os.path.exists(path + ki):
                    loadname = os.path.join(path, ki, loadname)
                elif os.path.exists(path):
                    loadname = os.path.join(path, loadname)
                if os.path.exists(loadname):
                    with open(loadname, 'rb') as f:
                        recon_src = pickle.load(f)
                if kwargs.get('verbose', False):
                    print('Loading '+loadname)
                # gather chi2 values
                chi2 = chi2_analysis(recon_src, **kwargs)
                ip = iprods[sf]
                # Plot chi2 vs scalar product
                plt.plot(chi2, ip, marker='o', lw=0, color='#756bb1')
                plt.xlabel("chi2")
                plt.ylabel('scalar')
                plt.title(name)
                # save the figure
                savename = "chi2_vs_scalar_{}.pdf".format(name)
                if path is None:
                    path = ""
                if os.path.exists(path + ki):
                    savename = os.path.join(path, ki, savename)
                elif os.path.exists(path):
                    savename = os.path.join(path, savename)
                if kwargs.get('verbose', False):
                    print('Saving '+savename)
                plt.savefig(savename)
                # plt.show()
                plt.close()

    # # data maps
    if 0:
        k = ["H3S0A0B90G0", "H10S0A0B90G0", "H36S0A0B90G0"]
        # k = ["H1S0A0B90G0", "H1S1A0B90G0", "H2S1A0B90G0", "H2S2A0B90G0", "H2S7A0B90G0",
        #      "H3S1A0B90G0", "H4S3A0B0G90", "H13S0A0B90G0", "H23S0A0B90G0", "H30S0A0B90G0",
        #      "H160S0A90B0G0", "H234S0A0B90G0"]
        k = keys
        kwargs = dict(verbose=1)
        sfiles = states  # synthf50  # prefiltered_synthf50  # filtered_states
        path = anlysdir+"states/"
        for ki in k:
            for sf in sfiles[ki]:
                name = os.path.basename(sf).replace(".state", "")
                loadname = "reconsrc_{}.pkl".format(name)
                if path is None:
                    path = ""
                if os.path.exists(path + ki):
                    loadname = os.path.join(path, ki, loadname)
                elif os.path.exists(path):
                    loadname = os.path.join(path, loadname)
                if os.path.exists(loadname):
                    with open(loadname, 'rb') as f:
                        recon_src = pickle.load(f)
                if kwargs.get('verbose', False):
                    print('Loading '+loadname)
                # Ensemble average
                data = recon_src.lensobject.data
                extent = recon_src.lensobject.extent
                extent = [extent[0], extent[2], extent[1], extent[3]]
                plt.imshow(data, cmap='Spectral_r',
                           origin='Lower', extent=extent)
                plt.colorbar()
                plt.title(name)
                # save the figure
                savename = "data_{}.pdf".format(name)
                if path is None:
                    path = ""
                if os.path.exists(path + ki):
                    savename = os.path.join(path, ki, savename)
                elif os.path.exists(path):
                    savename = os.path.join(path, savename)
                if kwargs.get('verbose', False):
                    print('Saving '+savename)
                plt.savefig(savename)
                # plt.show()
                plt.close()

    # source plane map
    if 0:
        k = ["H3S0A0B90G0", "H10S0A0B90G0", "H36S0A0B90G0"]
        # k = ["H1S0A0B90G0", "H1S1A0B90G0", "H2S1A0B90G0", "H2S2A0B90G0", "H2S7A0B90G0",
        #      "H3S1A0B90G0", "H4S3A0B0G90", "H13S0A0B90G0", "H23S0A0B90G0", "H30S0A0B90G0",
        #      "H160S0A90B0G0", "H234S0A0B90G0"]
        k = keys
        kwargs = dict(verbose=1)
        sfiles = states  # synthf50  # prefiltered_synthf50  # filtered_states
        path = anlysdir+"states/"
        for ki in k:
            for sf in sfiles[ki]:
                name = os.path.basename(sf).replace(".state", "")
                loadname = "reconsrc_{}.pkl".format(name)
                if path is None:
                    path = ""
                if os.path.exists(path + ki):
                    loadname = os.path.join(path, ki, loadname)
                elif os.path.exists(path):
                    loadname = os.path.join(path, loadname)
                if os.path.exists(loadname):
                    with open(loadname, 'rb') as f:
                        recon_src = pickle.load(f)
                if kwargs.get('verbose', False):
                    print('Loading '+loadname)
                # noise map
                lo = recon_src.lensobject
                signals, variances = lo.flatfield(lo.data, size=0.2)
                gain, _ = lo.gain(signals=signals, variances=variances)
                f = gain
                bias = 0.01*np.max(f * lo.data)
                sgma2 = lo.sigma2(f=f, add_bias=bias)
                dta_noise = np.random.normal(0, 1, size=lo.data.shape)
                dta_noise = dta_noise * np.sqrt(sgma2)
                # Ensemble average
                recon_src.chmdl(-1)
                r = recon_src.r_antialias
                if r is not None:
                    extent = [-r, r, -r, r]
                else:
                    extent = None
                plt.imshow(recon_src.plane_map(), cmap='Spectral_r',
                           origin='Lower', extent=extent)
                plt.colorbar()
                plt.title(name)
                # save the figure
                savename = "rconsrc_{}.pdf".format(name)
                if path is None:
                    path = ""
                if os.path.exists(path + ki):
                    savename = os.path.join(path, ki, savename)
                elif os.path.exists(path):
                    savename = os.path.join(path, savename)
                if kwargs.get('verbose', False):
                    print('Saving '+savename)
                plt.savefig(savename)
                # plt.show()
                plt.close()

    # synth maps the ensemble averages
    if 0:
        k = ["H3S0A0B90G0", "H10S0A0B90G0", "H36S0A0B90G0"]
        # k = ["H1S0A0B90G0", "H1S1A0B90G0", "H2S1A0B90G0", "H2S2A0B90G0", "H2S7A0B90G0",
        #      "H3S1A0B90G0", "H4S3A0B0G90", "H13S0A0B90G0", "H23S0A0B90G0", "H30S0A0B90G0",
        #      "H160S0A90B0G0", "H234S0A0B90G0"]
        k = keys
        kwargs = dict(verbose=1)
        sfiles = states  # synthf50  # prefiltered_synthf50  # filtered_states
        path = anlysdir+"states/"
        for ki in k:
            for sf in sfiles[ki]:
                name = os.path.basename(sf).replace(".state", "")
                loadname = "reconsrc_{}.pkl".format(name)
                if path is None:
                    path = ""
                if os.path.exists(path + ki):
                    loadname = os.path.join(path, ki, loadname)
                elif os.path.exists(path):
                    loadname = os.path.join(path, loadname)
                if os.path.exists(loadname):
                    with open(loadname, 'rb') as f:
                        recon_src = pickle.load(f)
                if kwargs.get('verbose', False):
                    print('Loading '+loadname)
                # Ensemble average
                recon_src.chmdl(-1)
                extent = recon_src.lensobject.extent
                extent = [extent[0], extent[2], extent[1], extent[3]]
                plt.imshow(recon_src.reproj_map(), cmap='Spectral_r',
                           origin='Lower', extent=extent)
                plt.colorbar()
                plt.title(name)
                # save the figure
                savename = "synth_{}.pdf".format(name)
                if path is None:
                    path = ""
                if os.path.exists(path + ki):
                    savename = os.path.join(path, ki, savename)
                elif os.path.exists(path):
                    savename = os.path.join(path, savename)
                if kwargs.get('verbose', False):
                    print('Saving '+savename)
                plt.savefig(savename)
                # plt.show()
                plt.close()
                
    # # Best/worst chi2/iprods synths
    if 0:
        ki = "H3S0A0B90G0"
        kwargs = dict(verbose=1)
        sf = states[ki][0]   # synthf50  # prefiltered_synthf50  # filtered_states
        chi2_sorted = [4, 36, 1, 2, 96, 6, 5, 35, 34, 0, 38, 71, 97, 39, 3, 72, 94, 75, 23, 40,
                       54, 93, 74, 41, 53, 70, 27, 28, 26, 76, 24, 95, 52, 21, 32, 92, 91, 50, 37,
                       49, 22, 25, 33, 31, 55, 43, 14, 29, 51, 30, 44, 73, 46, 47, 15, 45, 42, 20,
                       17, 19, 48, 16, 79, 77, 18, 60, 59, 58, 62, 61, 57, 78, 56, 82, 83, 81, 80,
                       64, 86, 69, 67, 68, 63, 88, 84, 66, 65, 98, 89, 90, 87, 99, 85, 7, 8, 9, 11,
                       12, 10, 13]
        iprods_sorted = [65, 21, 48, 26, 84, 18, 81, 68, 37, 99, 83, 85, 75, 63, 67, 66, 19, 92,
                         56, 78, 91, 17, 69, 30, 70, 80, 64, 0, 31, 6, 25, 82, 61, 8, 73, 57, 33,
                         16, 32, 9, 10, 20, 76, 55, 87, 13, 59, 5, 24, 29, 93, 90, 41, 74, 22, 27,
                         51, 43, 94, 42, 23, 47, 86, 1, 50, 28, 11, 60, 38, 2, 15, 62, 52, 12, 58,
                         71, 54, 14, 72, 53, 89, 77, 44, 97, 49, 95, 98, 45, 79, 88, 40, 4, 46, 3,
                         7, 96, 39, 35, 34, 36]
        path = anlysdir+"states/"
        name = os.path.basename(sf).replace(".state", "")
        loadname = "reconsrc_{}.pkl".format(name)
        if path is None:
            path = ""
        if os.path.exists(path + ki):
            loadname = os.path.join(path, ki, loadname)
        elif os.path.exists(path):
            loadname = os.path.join(path, loadname)
        if os.path.exists(loadname):
            with open(loadname, 'rb') as f:
                recon_src = pickle.load(f)
        if kwargs.get('verbose', False):
            print('Loading '+loadname)
        extent = recon_src.lensobject.extent
        extent = [extent[0], extent[2], extent[1], extent[3]]
        # chi2 synths
        if 1:
            for i, ichi2 in enumerate(chi2_sorted):
                if 9 < i and i < 90:  # only the edges
                    continue
                label = "chi2_{}".format(i)
                recon_src.chmdl(ichi2)
                plt.imshow(recon_src.reproj_map(), cmap='Spectral_r',
                           origin='Lower', extent=extent)
                plt.colorbar()
                plt.title("Model {}".format(ichi2))
                # save the figure
                savename = "{}_synth.pdf".format(label)
                if path is None:
                    path = ""
                if os.path.exists(path + ki):
                    mkdir_p(path + ki + "/synths")
                    savename = os.path.join(path, ki, "synths", savename)
                elif os.path.exists(path):
                    mkdir_p(path + "/synths")
                    savename = os.path.join(path, savename)
                if kwargs.get('verbose', False):
                    print('Saving '+savename)
                plt.savefig(savename)
                # plt.show()
                plt.close()
        # iprods synths
        if 1:
            for i, iiprods in enumerate(iprods_sorted[::-1]):
                if 9 < i and i < 90:  # only the edges
                    continue
                label = "iprods_{}".format(i)
                recon_src.chmdl(iiprods)
                plt.imshow(recon_src.reproj_map(), cmap='Spectral_r',
                           origin='Lower', extent=extent)
                plt.colorbar()
                plt.title("Model {}".format(iiprods))
                # save the figure
                savename = "{}_synth.pdf".format(label)
                if path is None:
                    path = ""
                if os.path.exists(path + ki):
                    mkdir_p(path + ki + "/synths")
                    savename = os.path.join(path, ki, "synths", savename)
                elif os.path.exists(path):
                    mkdir_p(path + "/synths")
                    savename = os.path.join(path, savename)
                if kwargs.get('verbose', False):
                    print('Saving '+savename)
                plt.savefig(savename)
                # plt.show()
                plt.close()

    # # Best/worst chi2/iprods arival time surfaces
    if 0:
        ki = "H3S0A0B90G0"
        kwargs = dict(verbose=1)
        sf = states[ki][0]   # synthf50  # prefiltered_synthf50  # filtered_states
        chi2_sorted = [4, 36, 1, 2, 96, 6, 5, 35, 34, 0, 38, 71, 97, 39, 3, 72, 94, 75, 23, 40,
                       54, 93, 74, 41, 53, 70, 27, 28, 26, 76, 24, 95, 52, 21, 32, 92, 91, 50, 37,
                       49, 22, 25, 33, 31, 55, 43, 14, 29, 51, 30, 44, 73, 46, 47, 15, 45, 42, 20,
                       17, 19, 48, 16, 79, 77, 18, 60, 59, 58, 62, 61, 57, 78, 56, 82, 83, 81, 80,
                       64, 86, 69, 67, 68, 63, 88, 84, 66, 65, 98, 89, 90, 87, 99, 85, 7, 8, 9, 11,
                       12, 10, 13]
        iprods_sorted = [65, 21, 48, 26, 84, 18, 81, 68, 37, 99, 83, 85, 75, 63, 67, 66, 19, 92,
                         56, 78, 91, 17, 69, 30, 70, 80, 64, 0, 31, 6, 25, 82, 61, 8, 73, 57, 33,
                         16, 32, 9, 10, 20, 76, 55, 87, 13, 59, 5, 24, 29, 93, 90, 41, 74, 22, 27,
                         51, 43, 94, 42, 23, 47, 86, 1, 50, 28, 11, 60, 38, 2, 15, 62, 52, 12, 58,
                         71, 54, 14, 72, 53, 89, 77, 44, 97, 49, 95, 98, 45, 79, 88, 40, 4, 46, 3,
                         7, 96, 39, 35, 34, 36]
        path = anlysdir+"states/"
        name = os.path.basename(sf).replace(".state", "")
        loadname = "reconsrc_{}.pkl".format(name)
        if path is None:
            path = ""
        if os.path.exists(path + ki):
            loadname = os.path.join(path, ki, loadname)
        elif os.path.exists(path):
            loadname = os.path.join(path, loadname)
        if os.path.exists(loadname):
            with open(loadname, 'rb') as f:
                recon_src = pickle.load(f)
        if kwargs.get('verbose', False):
            print('Loading '+loadname)
        # chi2 arrival time surfaces
        if 1:
            for i, ichi2 in enumerate(chi2_sorted):
                if 9 < i and i < 90:  # only the edges
                    continue
                label = "chi2_{}".format(i)
                m = recon_src.gls.models[ichi2]
                recon_src.gls.img_plot(obj_index=0, color='#fe4365')
                recon_src.gls.arrival_plot(m, obj_index=0,
                                           only_contours=True, clevels=75,
                                           colors=['#603dd0'])
                plt.title("Model {}".format(ichi2))
                # save the figure
                savename = "{}_arriv.pdf".format(label)
                if path is None:
                    path = ""
                if os.path.exists(path + ki):
                    mkdir_p(path + ki + "/arrivs")
                    savename = os.path.join(path, ki, "arrivs", savename)
                elif os.path.exists(path):
                    mkdir_p(path + "/arrivs")
                    savename = os.path.join(path, savename)
                if kwargs.get('verbose', False):
                    print('Saving '+savename)
                plt.savefig(savename)
                # plt.show()
                plt.close()
        # iprods arrival time surfaces
        if 1:
            for i, iiprods in enumerate(iprods_sorted[::-1]):
                if 9 < i and i < 90:  # only the edges
                    continue
                label = "iprods_{}".format(i)
                recon_src.gls.models[ichi2]
                m = recon_src.gls.models[ichi2]
                recon_src.gls.img_plot(obj_index=0, color='#fe4365')
                recon_src.gls.arrival_plot(m, obj_index=0,
                                           only_contours=True, clevels=75,
                                           colors=['#603dd0'])
                plt.title("Model {}".format(iiprods))
                # save the figure
                savename = "{}_arriv.pdf".format(label)
                if path is None:
                    path = ""
                if os.path.exists(path + ki):
                    mkdir_p(path + ki + "/arrivs")
                    savename = os.path.join(path, ki, "arrivs", savename)
                elif os.path.exists(path):
                    mkdir_p(path + "/arrivs")
                    savename = os.path.join(path, savename)
                if kwargs.get('verbose', False):
                    print('Saving '+savename)
                plt.savefig(savename)
                # plt.show()
                plt.close()
