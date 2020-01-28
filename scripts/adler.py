import matplotlib
matplotlib.use('Agg')
import os
import sys
root = os.path.dirname(os.path.dirname(os.path.abspath(os.path.realpath(__file__))))
sys.path.append(root)
import numpy as np
import scipy
import pandas as pd
import time
import pickle
from functools import partial
from pathos.multiprocessing import ProcessingPool as Pool
from astropy.io import fits
import matplotlib.pyplot as plt
from gleam.reconsrc import ReconSrc, synth_filter, synth_filter_mp
from gleam.multilens import MultiLens
from gleam.glass_interface import glass_renv, filter_env, export_state
from gleam.utils.encode import an_sort
from gleam.utils.lensing import LensModel, \
    downsample_model, upsample_model, radial_profile, \
    find_einstein_radius, kappa_profile, DLSDS, \
    complex_ellipticity, inertia_tensor, qpm_props, \
    potential_grid, roche_potential_grid, roche_potential
from gleam.utils.plotting import kappa_map_plot, kappa_profile_plot, kappa_profiles_plot, \
    roche_potential_plot, arrival_time_surface_plot, \
    complex_ellipticity_plot, \
    plot_scalebar, plot_labelbox, plot_annulus, plot_annulus_region
from gleam.utils.linalg import sigma_product, sigma_product_map
from gleam.utils.rgb_map import radial_mask
from gleam.utils.makedir import mkdir_p
from gleam.utils.colors import GLEAMcmaps, GLEAMcolors, color_variant
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


def chi2_analysis(reconsrc, cy_opt=False, optimized=False, psf_file=None, reduced=False, verbose=False):
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
    f = 1./(25*gain)
    bias = 0.001*np.max(f * lo.data)
    sgma2 = lo.sigma2(f=f, add_bias=bias)
    dta_noise = np.random.normal(0, 1, size=lo.data.shape)
    dta_noise = dta_noise * np.sqrt(sgma2)
    # recalculate the chi2s
    kwargs = dict(reconsrc=reconsrc, percentiles=[], use_psf=True, psf_file=psf_file,
                  cy_opt=cy_opt, noise=dta_noise, sigma2=sgma2,
                  reduced=reduced, return_obj=False, save=False, verbose=verbose)
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
        eagle_kappa = np.flip(eagle_kappa, 0)
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
        eagle_kappa_map = np.flip(eagle_kappa_map, 0)
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
    eagle_kappa_map = np.flip(eagle_kappa_map, 0)
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
            dlsds = DLSDS(obj.z, obj.sources[0].z)
            glass_kappa_map = dlsds * glass_kappa_map
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
    eagle_kappa_map = np.flip(eagle_kappa_map, 0)
    if method == 'e2g':
        obj, data = glass_state.models[0]['obj,data'][0]
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
            dlsds = DLSDS(obj.z, obj.sources[0].z)
            glass_kappa_map = dlsds * glass_kappa_map
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
    eagle_kappa_map = np.flip(eagle_kappa_map, 0)
    if method == 'e2g':
        obj, _ = glass_state.models[0]['obj,data'][0]
        eagle_pixel = eagle_hdr['CDELT2']*3600
        glass_maprad = obj.basis.top_level_cell_size * obj.basis.pixrad
        glass_extent = (-glass_maprad, glass_maprad, -glass_maprad, glass_maprad)
        glass_shape = (2*obj.basis.pixrad+1,)*2
        eagle_kappa_map = downsample_model(eagle_kappa_map, glass_extent, glass_shape,
                                           pixel_scale=eagle_pixel)
        egx, egy, eagle_degarr = roche_potential_grid(eagle_kappa_map, N, 2*glass_maprad, verbose=0)
        for i, m in enumerate(glass_state.models):
            obj, data = m['obj,data'][0]
            glass_kappa_map = obj.basis._to_grid(data['kappa'], 1)
            dlsds = DLSDS(obj.z, obj.sources[0].z)
            glass_kappa_map = dlsds * glass_kappa_map
            gx, gy, degarr = roche_potential_grid(glass_kappa_map, N, 2*glass_maprad, verbose=0)
            degarrs.append(degarr[:])
            if verbose:
                message = "{:4d} / {:4d}\r".format(i+1, N_models)
                sys.stdout.write(message)
                sys.stdout.flush()
    elif method == 'g2e':
        eagle_pixrad = tuple(r//2 for r in eagle_kappa_map.shape)
        eagle_maprad = eagle_pixrad[1]*eagle_hdr['CDELT2']*3600
        eagle_extent = [-eagle_maprad, eagle_maprad, -eagle_maprad, eagle_maprad]
        egx, egy, eagle_degarr = roche_potential_grid(eagle_kappa_map, N, 2*eagle_maprad, verbose=0)
        for i, m in enumerate(glass_state.models):
            glass_kappa_map = upsample_model(m, eagle_extent, eagle_kappa_map.shape)
            gx, gy, degarr = roche_potential_grid(glass_kappa_map, N, 2*eagle_maprad, verbose=0)
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
        dlsds = DLSDS(obj.z, obj.sources[0].z)
        glass_kappa_map = dlsds * glass_kappa_map
        _, _, degarr = roche_potential_grid(glass_kappa_map, N, grid_size, verbose=0)
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
    eagle_kappa_map = np.flip(eagle_kappa_map, 0)
    obj, _ = glass_state.models[0]['obj,data'][0]
    eagle_pixel = eagle_hdr['CDELT2']*3600
    glass_maprad = obj.basis.top_level_cell_size * obj.basis.pixrad
    glass_extent = (-glass_maprad, glass_maprad, -glass_maprad, glass_maprad)
    glass_shape = (2*obj.basis.pixrad+1,)*2
    eagle_kappa_map = downsample_model(eagle_kappa_map, glass_extent, glass_shape,
                                       pixel_scale=eagle_pixel)
    egx, egy, eagle_degarr = roche_potential_grid(eagle_kappa_map, N, 2*glass_maprad, verbose=0)

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


def synth_loop(keys, jsons, states,
               cached=False, save_state=False, save_obj=False, load_obj=False, psf_file=None,
               use_psf=True, path=None, optimized=False, verbose=False, **kwargs):
    """
    Args:
        keys <list(str)> - the keys which correspond to jsons' and states' keys
        jsons <dict> - json dictionary holding a list of filenames for each key
        states <dict> - state dictionary holding a list of filenames for each key

    Kwargs:
        save_state <bool> - save the filtered states automatically
        save_obj <bool> - save the ReconSrc object as a pickle file
        load_obj <bool> - load the ReconSrc object from a pickle file
        psf_file <str> - path to the psf .fits file
        use_psf <bool> - use the PSF in the reprojection
        cy_opt <bool> - use optimized cython method to construct inverse projection matrix
        path <str> - path to the pickle files to load from
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
        f = 1./(25*gain)
        bias = 0.001*np.max(f * ml[0].data)
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
                if os.path.exists(os.path.join(path, k)):
                    loadname = os.path.join(path, k, loadname)
                elif os.path.exists(path):
                    loadname = os.path.join(path, loadname)
                if os.path.exists(loadname):
                    with open(loadname, 'rb') as f:
                        recon_src = pickle.load(f)
                        if verbose:
                            print(recon_src.__v__)
                else:
                    recon_src = ReconSrc(ml, sf, M=40, verbose=verbose)
            else:
                recon_src = ReconSrc(ml, sf, M=40, verbose=verbose)
            if save_obj:
                recon_src = synthf(reconsrc=recon_src, percentiles=[], psf_file=psf_file,
                                   noise=dta_noise, sigma2=sgma2, use_psf=use_psf,
                                   return_obj=save_obj, save=False, verbose=verbose, **kwargs)
                # save the object
                savename = "reconsrc_{}".format(os.path.basename(sf).replace(".state", ".pkl"))
                if path is None:
                    path = ""
                if os.path.exists(os.path.join(path, k)):
                    savename = os.path.join(path, k, savename)
                elif os.path.exists(path):
                    savename = os.path.join(path, savename)
                with open(savename, 'wb') as f:
                    pickle.dump(recon_src, f)
            elif save_obj < 0:
                recon_src = synthf(reconsrc=recon_src, percentiles=[],
                                   psf_file=psf_file, use_psf=use_psf,
                                   noise=dta_noise, sigma2=sgma2,
                                   return_obj=save_obj, save=False, verbose=verbose, **kwargs)
            if save_state:
                synthf(statefile=sf, reconsrc=recon_src, percentiles=[10, 50],
                       psf_file=psf_file, use_psf=use_psf,
                       noise=dta_noise, sigma2=sgma2,
                       save=save_state, verbose=verbose, **kwargs)
    return filtered_states


def cache_loop(keys, jsons, states, path=None, variables=[
        'inv_proj', 'N_nil', 'r_max', 'M_fullres', 'r_fullres', 'reproj_d_ij'],
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
            if os.path.exists(os.path.join(path, k)):
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
    version = "v6"
    home = os.path.expanduser("~")
    rdir = os.path.join(home, "adler")
    jsondir = os.path.join(rdir, "json")
    statedir = os.path.join(rdir, "states", version)
    lensdir = os.path.join(rdir, "lenses")
    anlysdir = os.path.join(rdir, "analysis", version)
    kappadir = os.path.join(rdir, "kappa")

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
    rochef10 = {k: [f for f in ls_states
                    if k in f and f.endswith('_rochef10.state')] for k in keys}
    rochef25 = {k: [f for f in ls_states
                    if k in f and f.endswith('_rochef25.state')] for k in keys}
    rochef50 = {k: [f for f in ls_states
                    if k in f and f.endswith('_rochef50.state')] for k in keys}
    ls_states = [f for f in ls_states if not (f.endswith('_rochef10.state')
                                              or f.endswith('_rochef25.state')
                                              or f.endswith('_rochef50.state'))]
    rsf10 = {k: [f for f in ls_states
                 if k in f and 'rochef10Xsynthf10_' in f] for k in keys}
    rsf25 = {k: [f for f in ls_states
                 if k in f and 'rochef25Xsynthf25_' in f] for k in keys}
    rsf50 = {k: [f for f in ls_states
                 if k in f and 'rochef50Xsynthf50_' in f] for k in keys}
    ls_states = [f for f in ls_states if not ('rochef10Xsynthf10_' in f
                                              or 'rochef25Xsynthf25_' in f
                                              or 'rochef50Xsynthf50_' in f)]
    states = {k: [f for f in ls_states if k in f] for k in keys}
    # print(states)
    kappa_files = {k: [f for f in ls_kappas if k in f] for k in keys}
    psf_file = os.path.join(lensdir, "psf.fits")

    sfiles = states
    sfiles_str = "states"
    # sfiles = synthf10
    # sfiles_str = "synthf10"
    # sfiles = rochef10
    # sfiles_str = "rochef10"
    # sfiles = rsf25
    # sfiles_str = "rsf25"

    extension = "pdf"

    # loop booleans
    RECONSRC_LOOP    = 0
    CHI2_LOOP        = 0
    K_DIFF_LOOP      = 0
    QUADPM_LOOP      = 0
    ABPHI_LOOP       = 0
    ELLIPTICITY_LOOP = 1
    ELLIPTC_ALL_LOOP = 0
    ROCHE_LOOP       = 0
    ROCHE_HIST_LOOP  = 0
    ROCHE_MAP_LOOP   = 0
    K_PROFILE_LOOP   = 1
    CHI2VSROCHE_LOOP = 1
    DATA_LOOP        = 0
    SOURCE_LOOP      = 0
    SYNTH_LOOP       = 0
    ARRIV_LOOP       = 0
    K_MAP_LOOP       = 0
    K_TRUE_LOOP      = 0
    INDIVIDUAL_LOOP  = 0
    SYNTHS_ONLY      = 0

    ROCHE_DEBUG_LOOP = 0
    TABLE_LOOP       = 0

    # not frequently used loops
    DIR_LOOP       = 0
    SFILTER_LOOP   = 0
    RFILTER_LOOP   = 0
    RSFILTR_LOOP   = 0
    RECACHE_LOOP   = 0
    POTENTIAL_LOOP = 0

    # matplotlib.rcParams['']

    ############################################################################
    # # LOOP OPERATIONS
    # # create a directory structure
    if DIR_LOOP:
        print("### Creating directory structure for analysis files")
        mkdir_structure(keys, root=os.path.join(anlysdir, sfiles_str))

    # # reconsrc synth caching
    if RECONSRC_LOOP:
        print("### Calculating ReconSrc objects and save as pickle files")
        # k = ["H3S0A0B90G0", "H10S0A0B90G0", "H36S0A0B90G0"]
        k = keys
        kwargs = dict(save_state=0, save_obj=1, load_obj=1,
                      path=os.path.join(anlysdir, sfiles_str),
                      method='minres', psf_file=psf_file, use_psf=1,
                      cy_opt=1, optimized=1, nproc=8,
                      from_cache=1, cached=1, save_to_cache=1,
                      stdout_flush=0, verbose=1)
        # kwargs = dict(save_state=1, save_obj=0, load_obj=1, path=anlysdir+"states/",
        #               psf_file=psf_file, use_psf=1, optimized=0, verbose=1)
        # sfiles = states
        synth_filtered_states = synth_loop(k, jsons, sfiles, **kwargs)

    # # reload cache into new reconsrc objects
    if RECACHE_LOOP:
        print("### Reload cache and save as new pickle files")
        # k = ["H3S0A0B90G0", "H10S0A0B90G0", "H36S0A0B90G0"]
        k = keys
        kwargs = dict(path=os.path.join(anlysdir, sfiles_str),
                      variables=['inv_proj', 'N_nil', 'r_max',
                                 'M_fullres', 'r_fullres', 'reproj_d_ij'],
                      save_obj=True, verbose=1)
        # sfiles = states
        cache_loop(k, jsons, sfiles, **kwargs)

    # # reconsrc synth filtering
    if SFILTER_LOOP:
        print("### Filter out bad chi2 and save filtered state files")
        # k = ["H3S0A0B90G0", "H10S0A0B90G0", "H36S0A0B90G0"]
        k = keys
        kwargs = dict(save_state=1, save_obj=0, load_obj=1,
                      path=os.path.join(anlysdir, sfiles_str),
                      method='minres', psf_file=psf_file, use_psf=1,
                      cy_opt=1, optimized=1, nproc=8,
                      from_cache=1, cached=1, save_to_cache=1,
                      stdout_flush=0, verbose=1)
        # kwargs = dict(save_state=1, save_obj=0, load_obj=1, path=anlysdir+"states/",
        #               psf_file=psf_file, use_psf=1, optimized=0, verbose=1)
        # sfiles = states
        synth_filtered_states = synth_loop(k, jsons, sfiles, **kwargs)

    if RFILTER_LOOP:
        print("### Filter out bad Roche scalars and save filtered state files")
        k = keys
        kwargs = dict(verbose=1, percentiles=[10, 50], save=True)
        path = os.path.join(anlysdir, sfiles_str)
        for ki in k:
            files = sfiles[ki]
            for sf in files:
                gls = glass.glcmds.loadstate(sf)
                name = os.path.basename(sf).replace(".state", "")
                print(name)
                textname = "scalarRoche_{}.txt".format(name)
                if os.path.exists(os.path.join(path, ki)):
                    textname = os.path.join(path, ki, textname)
                elif os.path.exists(path):
                    textname = os.path.join(path, textname)
                # since loading text files is faster it has priority
                if os.path.exists(textname):
                    sortedidcs = np.int32(np.loadtxt(textname).T[1])
                else:
                    exit(1)
                N = len(sortedidcs)
                percentiles = kwargs.get('percentiles', [10, 50])
                selected = [sortedidcs[-int(p/100.*N):] for p in percentiles]
                filtered = [filter_env(gls, s) for s in selected]
                if kwargs.get('save', False):
                    dirname = os.path.dirname(sf)
                    basename = ".".join(os.path.basename(sf).split('.')[:-1])
                    saves = [os.path.join(dirname, basename)+'_rochef{}.state'.format(p) for p in percentiles]
                    for f, s in zip(filtered, saves):
                        export_state(f, name=s)

    if RSFILTR_LOOP:
        print("### Filter out bad Roche scalars + bad chi2 and save filtered state files")
        k = keys
        kwargs = dict(verbose=1, percentiles=[10, 25, 50], save=True)
        path = os.path.join(anlysdir, sfiles_str)
        for ki in k:
            files = sfiles[ki]
            for sf in files:
                gls = glass.glcmds.loadstate(sf)
                name = os.path.basename(sf).replace(".state", "")
                print(name)
                textname = "scalarRoche_{}.txt".format(name)
                if os.path.exists(os.path.join(path, ki)):
                    textname = os.path.join(path, ki, textname)
                elif os.path.exists(path):
                    textname = os.path.join(path, textname)
                # since loading text files is faster it has priority
                if os.path.exists(textname):
                    r_sortedidcs = np.int32(np.loadtxt(textname).T[1])
                else:
                    exit(1)
                textname = "chi2_{}.txt".format(name)
                if os.path.exists(os.path.join(path, ki)):
                    textname = os.path.join(path, ki, textname)
                elif os.path.exists(path):
                    textname = os.path.join(path, textname)
                if os.path.exists(textname):
                    s_sortedidcs = np.int32(np.loadtxt(textname).T[1])
                else:
                    exit(1)
                N = len(s_sortedidcs)
                percentiles = kwargs.get('percentiles', [10, 50])
                s_selected = [s_sortedidcs[:int(p/100.*N)] for p in percentiles]
                r_selected = [r_sortedidcs[-int(p/100.*N):] for p in percentiles]
                selected = [sorted(list(set(s).intersection(r))) for s, r in zip(r_selected, s_selected)]
                filtered = [filter_env(gls, s) for s in selected]
                if kwargs.get('save', False):
                    dirname = os.path.dirname(sf)
                    basename = ".".join(os.path.basename(sf).split('.')[:-1])
                    exts = ['_rochef{percent}Xsynthf{percent}_{N}.state'.format(percent=p, N=len(s))
                            for p, s in zip(percentiles, selected)]
                    saves = [os.path.join(dirname, basename)+ext for ext in exts]
                    for f, s in zip(filtered, saves):
                        export_state(f, name=s)

    # # chi2 histograms (takes long!!!)
    if CHI2_LOOP:
        print("### Plotting chi2 histograms")
        # k = ["H3S0A0B90G0", "H10S0A0B90G0", "H36S0A0B90G0"]
        k = keys
        kwargs = dict(optimized=False, psf_file=psf_file, verbose=1)
        # sfiles = states
        path = os.path.join(anlysdir, sfiles_str)
        for ki in k:
            for sf in sfiles[ki]:
                name = os.path.basename(sf).replace(".state", "")
                print(name)
                loadname = "reconsrc_{}.pkl".format(name)
                if path is None:
                    path = ""
                if os.path.exists(os.path.join(path, ki)):
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
                # plt.figure(figsize=(5.68, 5.392))
                plt.hist(chi2, bins=20, color=GLEAMcolors.cyan_light, alpha=0.7, rwidth=0.85)
                plt.grid(axis='y', alpha=0.5)
                plt.xlabel(r'$\chi^{2}$')
                plt.ylabel(r'$\mathrm{\mathsf{N_{models}}}$')
                # plt.title(name)
                plot_labelbox(name, position='top right', color='black')
                plt.tight_layout()
                # save the figure
                savename = "chi2_hist_{}.{}".format(name, extension)
                textname = 'chi2_{}.txt'.format(name)
                if path is None:
                    path = ""
                if os.path.exists(os.path.join(path, ki)):
                    savename = os.path.join(path, ki, savename)
                    textname = os.path.join(path, ki, textname)
                elif os.path.exists(path):
                    savename = os.path.join(path, savename)
                    textname = os.path.join(path, textname)
                if kwargs.get('verbose', False):
                    print('Saving '+savename)
                    print('Saving '+textname)
                np.savetxt(textname, np.c_[chi2, sortedidcs], fmt='%12d', delimiter=' ',
                           newline=os.linesep)
                plt.savefig(savename, dpi=200, transparent=True, bbox_inches='tight', pad_inches=0)
                # plt.show()
                plt.close()

    # # kappa diff histograms
    if K_DIFF_LOOP:
        print("### Plotting kappa residual histograms")
        # k = ["H3S0A0B90G0", "H10S0A0B90G0", "H36S0A0B90G0"]
        k = keys
        kwargs = dict(method='e2g', verbose=1)
        # sfiles = states
        path = os.path.join(anlysdir, sfiles_str)
        residuals = residual_loop(k, kappa_files, sfiles, **kwargs)
        for ki in k:
            print(ki)
            for idx in range(len(states[ki])):
                sf = sfiles[ki][idx]
                name = os.path.basename(sf).replace(".state", "")
                savename = "kappa_diff_hist_{}.{}".format(name, extension)
                if path is None:
                    path = ""
                if os.path.exists(os.path.join(path, ki)):
                    savename = os.path.join(path, ki, savename)
                elif os.path.exists(path):
                    savename = os.path.join(path, savename)
                if kwargs.get('verbose', False):
                    print(savename)
                # plt.figure(figsize=(5.68, 5.392))
                plt.hist(residuals[sf], color='#386BF1', alpha=0.7, rwidth=0.85)
                # plt.xlim(left=0, right=200)
                plt.ylim(bottom=0, top=int(0.3*len(residuals[sf])))
                plt.grid(axis='y', alpha=0.5)
                plt.xlabel(r'$\Delta$')
                plt.ylabel(r'$\mathrm{\mathsf{N_{models}}}$')
                # plt.title(name)
                plot_labelbox(name, position='top right', color='black')
                plt.tight_layout()
                plt.savefig(savename, dpi=200, transparent=True, bbox_inches='tight', pad_inches=0)
                plt.close()

    # # inertia tensor analysis
    if QUADPM_LOOP:
        print("### Calculating inertia tensors")
        # k = ["H3S0A0B90G0", "H10S0A0B90G0", "H36S0A0B90G0"]
        k = keys
        kwargs = dict(method='e2g', activation=0.25, verbose=1)
        # sfiles = states
        path = os.path.join(anlysdir, sfiles_str)
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
    if ABPHI_LOOP:
        print("### Plotting inertia ellipse's parameter histograms")
        # k = ["H3S0A0B90G0", "H10S0A0B90G0", "H36S0A0B90G0"]
        k = keys
        kwargs = dict(verbose=1)
        # sfiles = states
        path = os.path.join(anlysdir, sfiles_str)
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
                ea, eb, ephi = qpm_props(eq, verbose=True)
                # ephi = abs(ephi)
                gprops = [qpm_props(q) for q in gq]
                ga, gb, gphi = [p[0] for p in gprops], \
                               [p[1] for p in gprops], \
                               [p[2] for p in gprops]
                               # [abs(p[2]) for p in gprops]
                name = os.path.basename(f).replace(".state", "")
                for i, (eprop, gprop) in enumerate(zip([ea, eb, ephi], [ga, gb, gphi])):
                    savename = lbl[i]+"_hist_{}.{}".format(name, extension)
                    if path is None:
                        path = ""
                    if os.path.exists(os.path.join(path, ki)):
                        savename = os.path.join(path, ki, savename)
                    elif os.path.exists(path):
                        savename = os.path.join(path, savename)
                    if kwargs.get('verbose', False):
                        print(savename)
                    # plt.figure(figsize=(5.68, 5.392))
                    plt.hist(gprop, bins=14, color=GLEAMcolors.red, alpha=0.7, rwidth=0.85)
                    plt.axvline(eprop, color=color_variant(GLEAMcolors.red, shift=-50))
                    if lbl[i] == 'phi':
                        plt.xlim(left=0, right=np.pi)
                        lbl[i] = '\{}'.format(lbl[i])
                    plt.grid(axis='y', alpha=0.5)
                    plt.xlabel(r'$\mathrm{{\mathsf{{{:s}}}}}$'.format(lbl[i]))
                    plt.ylabel(r'$\mathrm{\mathsf{N_{models}}}$')
                    plot_labelbox(name, position='top left', color='black')
                    plt.tight_layout()
                    # plt.title(name)
                    plt.savefig(savename, dpi=200, transparent=True, bbox_inches='tight', pad_inches=0)
                    plt.close()

    # # create dataframes with various info for true kappa maps
    if TABLE_LOOP:
        print("### Creating pd.DataFrames for csv/tex exports")
        k = keys
        kwargs = dict(verbose=0)
        path = os.path.join(anlysdir, sfiles_str)
        obj_index = 0
        head = ['Image system', 'Visible maxima', 'Tangential arcs', r'R$_{E}$ [arcsec]', 'a [arcsec]', 'b [arcsec]', r'$\phi$ [radians]']
        data = {ki: [None]*len(head) for ki in k}
        r_E_truth = {ki: None for ki in k}
        phi_truth = {ki: None for ki in k}
        # Image number info
        for ki in k:
            files = sfiles[ki]
            for f in files:
                gls = glass.glcmds.loadstate(f)
                gls.make_ensemble_average()
                obj, dta = gls.ensemble_average['obj,data'][obj_index]
                imgs = obj.sources[obj_index].images
                N_min = sum([1 for i in imgs if i.parity == 0])
                N_sad = sum([1 for i in imgs if i.parity == 1])
                N_max = sum([1 for i in imgs if i.parity == 2])
                maxidx = [i for i, img in enumerate(obj.sources[obj_index].images) if img.parity == 2]
                maximum = obj.sources[obj_index].images[maxidx[0]] if maxidx else None
                img_sys = '{}'.format(N_min+N_sad)
                if N_min+N_sad == 4:
                    img_sys = 'quad'
                elif N_min+N_sad == 2:
                    img_sys = 'double'
                is_tangential = 1 if ki in ["H2S7A0B90G0", "H3S1A0B90G0", "H30S0A0B90G0", "H36S0A0B90G0"] else 0
                data[ki][0] = img_sys
                data[ki][1] = N_max
                data[ki][2] = is_tangential
                if kwargs.get('verbose', False):
                    print(img_sys, N_max, is_tangential)
        # R_E
        for ki in k:
            files = sfiles[ki]
            for f in files:
                eagle_model = fits.getdata(kappa_files[ki][0], header=True)
                eagle_kappa_map = eagle_model[0]
                eagle_kappa_map = np.flip(eagle_kappa_map, 0)
                eagle_pixrad = tuple(r//2 for r in eagle_kappa_map.shape)
                eagle_maprad = eagle_pixrad[1]*eagle_model[1]['CDELT2']*3600
                eagle_mapextent = eagle_pixrad[1]*eagle_model[1]['CDELT2']*3600
                radii, profile = kappa_profile(eagle_kappa_map, correct_distances=False,
                                               maprad=eagle_maprad)
                r_E = find_einstein_radius(radii, profile)
                data[ki][3] = "{:.2f}".format(r_E)
                r_E_truth[ki] = r_E
                if kwargs.get('verbose', False):
                    print(r_E)
        # a, b, phi
        loadname = 'qpms.pkl'
        if path is None:
            path = ""
        elif os.path.exists(path):
            loadname = os.path.join(path, loadname)
        with open(loadname, 'rb') as f:
            qpms = pickle.load(f)
        for ki in k:
            files = sfiles[ki]
            for f in files:
                name = os.path.basename(f).replace(".state", "")
                eq, gq = qpms[f]
                ea, eb, ephi = qpm_props(eq, verbose=False)
                phi_truth[ki] = ephi
                data[ki][4] = r'{:.2f}'.format(ea)
                data[ki][5] = r'{:.2f}'.format(eb)
                data[ki][6] = r'{:.2f}'.format(ephi)
                if kwargs.get('verbose', False):
                    print(ea, eb, ephi)
        df = pd.DataFrame.from_dict(data, orient='index', columns=head)
        print(df.to_latex(bold_rows=1, encoding='utf-8', escape=False))
                
    # # create dataframes with various info for ensemble-averaged kappa maps
    if TABLE_LOOP:
        print("### Creating pd.DataFrames for csv/tex exports")
        k = keys
        kwargs = dict(verbose=0)
        path = os.path.join(anlysdir, sfiles_str)
        obj_index = 0
        head = ['Image system', 'Visible maxima', 'Tangential arcs', r'R$_{E}$ [arcsec]', 'a [arcsec]', 'b [arcsec]', r'$\phi$ [radians]']
        data = {ki: [None]*len(head) for ki in k}
        # Image number info
        for ki in k:
            files = sfiles[ki]
            for f in files:
                gls = glass.glcmds.loadstate(f)
                gls.make_ensemble_average()
                obj, dta = gls.ensemble_average['obj,data'][obj_index]
                imgs = obj.sources[obj_index].images
                N_min = sum([1 for i in imgs if i.parity == 0])
                N_sad = sum([1 for i in imgs if i.parity == 1])
                N_max = sum([1 for i in imgs if i.parity == 2])
                maxidx = [i for i, img in enumerate(obj.sources[obj_index].images) if img.parity == 2]
                maximum = obj.sources[obj_index].images[maxidx[0]] if maxidx else None
                img_sys = '{}'.format(N_min+N_sad)
                if N_min+N_sad == 4:
                    img_sys = 'quad'
                elif N_min+N_sad == 2:
                    img_sys = 'double'
                is_tangential = 1 if ki in ["H2S7A0B90G0", "H3S1A0B90G0", "H30S0A0B90G0", "H36S0A0B90G0"] else 0
                data[ki][0] = img_sys
                data[ki][1] = N_max
                data[ki][2] = is_tangential
                if kwargs.get('verbose', False):
                    print(img_sys, N_max, is_tangential)
        # R_E
        for ki in k:
            files = sfiles[ki]
            for f in files:
                gls = glass.glcmds.loadstate(f)
                gls.make_ensemble_average()
                radii = []
                profiles = []
                r_Es = []
                N_models = len(gls.models)
                for i, m in enumerate(gls.models):
                    sys.stdout.write('{:4d}/{:4d}\r'.format(i+1, N_models))
                    sys.stdout.flush()
                    r, p = kappa_profile(m, obj_index=obj_index, correct_distances=True)
                    radii.append(r)
                    profiles.append(p)
                    r_E = find_einstein_radius(r, p)
                    r_Es.append(r_E)
                r_E = np.mean(r_Es)
                r_E_err_internal = np.std(r_Es)
                r_E_err_2truth = np.sqrt((r_E_truth[ki] - r_E)**2)
                pxl = gls.models[0]['obj,data'][0][0].basis.top_level_cell_size
                data[ki][3] = "{:.2f} $\pm$ {:.2f} compare to {:.2f} \% {:5.2f}".format(r_E, r_E_err_2truth, pxl, 100*r_E_err_2truth/r_E)
                if kwargs.get('verbose', False):
                    print(r_E)
        # a, b, phi
        phi_ens = {ki: [] for ki in k}
        loadname = 'qpms.pkl'
        if path is None:
            path = ""
        elif os.path.exists(path):
            loadname = os.path.join(path, loadname)
        with open(loadname, 'rb') as f:
            qpms = pickle.load(f)
        for ki in k:
            files = sfiles[ki]
            for f in files:
                gls = glass.glcmds.loadstate(f)
                for m in gls.models:
                    obj, dta = m['obj,data'][obj_index]
                    dlsds = DLSDS(obj.z, obj.sources[obj_index].z)
                    maprad = obj.basis.top_level_cell_size * obj.basis.pixrad
                    mapextent = obj.basis.top_level_cell_size * (2*obj.basis.pixrad+1)
                    kappa = obj.basis._to_grid(dta['kappa'], 1)
                    kappa_grid = dlsds * kappa
                    ensqpm = inertia_tensor(kappa_grid, pixel_scale=obj.basis.top_level_cell_size, activation=0.25)
                    a, b, phi = qpm_props(ensqpm, verbose=False)
                    phi_ens[ki].append(phi)
                gls.make_ensemble_average()
                obj, dta = gls.ensemble_average['obj,data'][obj_index]
                dlsds = DLSDS(obj.z, obj.sources[obj_index].z)
                maprad = obj.basis.top_level_cell_size * obj.basis.pixrad
                mapextent = obj.basis.top_level_cell_size * (2*obj.basis.pixrad+1)
                kappa = obj.basis._to_grid(dta['kappa'], 1)
                kappa_grid = dlsds * kappa
                ensqpm = inertia_tensor(kappa_grid, pixel_scale=obj.basis.top_level_cell_size, activation=0.25)
                a, b, phi = qpm_props(ensqpm, verbose=False)
                _, gq = qpms[f]
                gprops = [qpm_props(q) for q in gq]
                ga, gb, gphi = [p[0] for p in gprops], \
                               [p[1] for p in gprops], \
                               [p[2] for p in gprops]
                a_ma, a_mi = max(ga), min(ga)
                b_ma, b_mi = max(gb), min(gb)
                phi_ma, phi_mi = max(gphi), min(gphi)
                data[ki][4] = r"{:.2f} $\pm$ {:.2f}".format(a, max(abs(a-a_mi), abs(a-a_ma)))
                data[ki][5] = r'{:.2f} $\pm$ {:.2f}'.format(b, max(abs(b-b_mi), abs(b-b_ma)))
                data[ki][6] = r'{:1.2f} $\pm$ {:1.2f}'.format(phi, max(abs(phi-phi_mi), abs(phi-phi_ma)))
                if kwargs.get('verbose', False):
                    print(a, b, phi)
        df = pd.DataFrame.from_dict(data, orient='index', columns=head)
        print(df.to_latex(bold_rows=1, encoding='utf-8', escape=False))
        doubles = []
        quads = []
        for l, s in zip(df['R$_{E}$ [arcsec]'], df['Image system']):
            r = float(l.split('%')[-1].strip())
            if s.strip() == 'double':
                doubles.append(r)
            else:
                quads.append(r)
        print("AVERAGE R_E error (quads): ", np.average(quads), np.median(quads))
        print("AVERAGE R_E error (dbles): ", np.average(doubles), np.median(doubles))

        print("PHI Errors")
        phi_errs = []
        for ki in k:
            phi = np.median(phi_ens[ki])
            phi = phi if phi > 0.5*np.pi else np.pi - phi
            phi_mean = np.mean(phi_ens[ki])
            phi_mean = phi_mean if phi_mean > 0.5*np.pi else np.pi - phi_mean
            phi_err_internal = 360/(2*np.pi)*np.std(phi_ens[ki])
            # phi_err_internal = phi_err_internal if phi_err_internal < 90 else (180 - phi_err_internal)
            phi_err_2truth = 360/(2*np.pi)*(np.sqrt((phi_truth[ki] - phi_mean)**2))
            # phi_err_2truth = phi_err_2truth if phi_err_2truth < 90 else (180 - phi_err_2truth)
            phi_err = phi_err_2truth
            if phi_err < 100:
                phi_errs.append(phi_err)
            print(phi_mean, phi_err_internal, phi_err_2truth)
        # print(df.to_csv(sep='\t'))
        print(np.average(phi_errs), np.median(phi_errs))

    # # complex ellipticity plots
    if 0 and ELLIPTICITY_LOOP:
        print("### Plotting complex ellipticity plots")
        matplotlib.rcParams['xtick.labelsize'] = 18
        matplotlib.rcParams['ytick.labelsize'] = 18
        matplotlib.rcParams['axes.labelsize'] = 18
        matplotlib.rcParams['axes.titlesize'] = 18
        k = keys
        kwargs = dict(verbose=1)
        # sfiles = states
        path = os.path.join(anlysdir, sfiles_str)
        loadname = 'qpms.pkl'
        if path is None:
            path = ""
        elif os.path.exists(path):
            loadname = os.path.join(path, loadname)
        with open(loadname, 'rb') as f:
            qpms = pickle.load(f)
        # complex ellipticities
        for ki in k:
            files = sfiles[ki]
            for f in files:
                eq, gq = qpms[f]
                ea, eb, ephi = qpm_props(eq)
                gprops = [qpm_props(q) for q in gq]
                ga, gb, gphi = [p[0] for p in gprops], \
                               [p[1] for p in gprops], \
                               [p[2] for p in gprops]
                eagle_epsilon = complex_ellipticity(ea, eb, ephi)
                glass_epsilon = complex_ellipticity(ga, gb, gphi)
                name = os.path.basename(f).replace(".state", "")
                savename = "ellipticity_{}.{}".format(name, extension)
                if path is None:
                    path = ""
                if os.path.exists(os.path.join(path, ki)):
                    savename = os.path.join(path, ki, savename)
                elif os.path.exists(path):
                    savename = os.path.join(path, savename)
                if kwargs.get('verbose', False):
                    print(savename)
                plt.figure(figsize=(5, 5))
                complex_ellipticity_plot([glass_epsilon],
                                         scatter=False, samples=[-1, -1], ensemble_averages=True,
                                         colors=[GLEAMcolors.blue, GLEAMcolors.red],
                                         lss=['--', '--'],
                                         markers=['o', 'o'], markersizes=[2, 6],
                                         contours=True, levels=6,
                                         axlabels=False,  # cmap=GLEAMcmaps.coralglow,
                                         cmap=GLEAMcmaps.reverse(GLEAMcmaps.cyber), alpha=0.8,
                                         origin_marker=False, adjust_limits=False)
                complex_ellipticity_plot([glass_epsilon, eagle_epsilon],
                                         scatter=True, samples=[25, -1], ensemble_averages=True,
                                         colors=[GLEAMcolors.red, GLEAMcolors.red],
                                         lss=['--', '--'],
                                         markers=['o', 'o'], markersizes=[2, 6],
                                         contours=False, levels=6,
                                         cmap=GLEAMcolors.cmap_from_color(GLEAMcolors.blue),
                                         origin_marker=True, adjust_limits=True,
                                         colorbar=False, axlabels=False, fontsize=24,
                                         annotation_color='black')
                plot_labelbox(ki, position='top right', padding=(0.03, 0.03), color='black',
                              fontsize=18)
                plt.gca().set_aspect('equal')
                # plt.gci().colorbar.set_label(r'$\mathrm{\mathsf{N_{models}}}$')
                plt.tight_layout()
                plt.savefig(savename, dpi=500, transparent=True, bbox_inches='tight', pad_inches=0)
                plt.close()

    # # complex ellipticity plots
    if ELLIPTICITY_LOOP:
        print("### Plotting complex ellipticity plots (single figure)")
        matplotlib.rcParams['xtick.labelsize'] = 16
        matplotlib.rcParams['ytick.labelsize'] = 16
        matplotlib.rcParams['axes.labelsize'] = 18
        matplotlib.rcParams['axes.titlesize'] = 18
        k = ["H3S0A0B90G0", "H10S0A0B90G0", "H36S0A0B90G0",
             "H2S2A0B90G0", "H3S1A0B90G0", "H2S1A0B90G0",
             "H160S0A90B0G0", "H4S3A0B0G90", "H30S0A0B90G0",
             "H13S0A0B90G0", "H2S7A0B90G0", "H1S0A0B90G0",
             "H1S1A0B90G0", "H23S0A0B90G0", "H234S0A0B90G0"]
        kwargs = dict(verbose=1)
        # sfiles = states
        path = os.path.join(anlysdir, sfiles_str)
        loadname = 'qpms.pkl'
        if path is None:
            path = ""
        elif os.path.exists(path):
            loadname = os.path.join(path, loadname)
        with open(loadname, 'rb') as f:
            qpms = pickle.load(f)
        # complex ellipticities
        fig, axes = plt.subplots(len(k)//3, 3, sharex=False, sharey=False,
                                 figsize=(6, 9))
        # fig = plt.figure(figsize=(8.27, 11.69))
        # gs = matplotlib.gridspec.GridSpec(len(k)//3, 3)
        # gs.update(wspace=0.025, hspace=0.05)
        for i, ki in enumerate(k):
            files = sfiles[ki]
            for f in files:
                eq, gq = qpms[f]
                ea, eb, ephi = qpm_props(eq)
                gprops = [qpm_props(q) for q in gq]
                ga, gb, gphi = [p[0] for p in gprops], \
                               [p[1] for p in gprops], \
                               [p[2] for p in gprops]
                eagle_epsilon = complex_ellipticity(ea, eb, ephi)
                glass_epsilon = complex_ellipticity(ga, gb, gphi)
                name = os.path.basename(f).replace(".state", "")
                if kwargs.get('verbose', False):
                    print(ki)
                plt.sca(axes[i // 3][i % 3])
                complex_ellipticity_plot([glass_epsilon],
                                         scatter=False, samples=[-1, -1], ensemble_averages=True,
                                         colors=[GLEAMcolors.blue, GLEAMcolors.red],
                                         lss=['--', '--'],
                                         markers=['o', 'o'], markersizes=[1, 5],
                                         contours=True, levels=6,
                                         # cmap=GLEAMcmaps.coralglow,
                                         cmap=GLEAMcmaps.reverse(GLEAMcmaps.cyber), alpha=0.8,
                                         origin_marker=False, adjust_limits=False,
                                         axlabels=False)
                complex_ellipticity_plot([glass_epsilon, eagle_epsilon],
                                         scatter=True, samples=[10, -1], ensemble_averages=True,
                                         colors=[GLEAMcolors.red, GLEAMcolors.red],
                                         lss=[(0, (1, 1)), (0, (1, 1))],
                                         markers=['o', 'o'], markersizes=[1, 5],
                                         contours=False, levels=6,
                                         cmap=GLEAMcolors.cmap_from_color(GLEAMcolors.blue),
                                         origin_marker=True, adjust_limits=False,
                                         colorbar=False,
                                         fontsize=12, annotation_color='black',
                                         axlabels=False)
                plot_labelbox(ki, position='top right', padding=(0.05, 0.075), color='black',
                              fontsize=8)
                axes[i // 3][i % 3].set_yticks(np.linspace(-0.2, 0.2, 5))
                axes[i // 3][i % 3].set_xticks(np.linspace(-0.2, 0.2, 5))
                if (i % 3) == 0:
                    axes[i // 3][i % 3].set_yticklabels(["", "", "0", "", "0.2"])
                    if (i // 3) == 2:
                        axes[i // 3][i % 3].set_ylabel(r'$\mathrm{\mathsf{Im\,\epsilon}}$')
                else:
                    axes[i // 3][i % 3].set_yticklabels(["", "", "", "", ""])
                if (i // 3) > 3:
                    axes[i // 3][i % 3].set_xticklabels(["-0.2", "", "0", "", ""])
                    if (i % 3) == 1:
                        axes[i // 3][i % 3].set_xlabel(r'$\mathrm{\mathsf{Re\,\epsilon}}$')
                else:
                    axes[i // 3][i % 3].set_xticklabels(["", "", "", "", ""])
                lim = 0.225
                plt.xlim(left=-lim, right=lim)
                plt.ylim(bottom=-lim, top=lim)
                axes[i // 3][i % 3].set_aspect('equal')
                # plt.gci().colorbar.set_label(r'$\mathrm{\mathsf{N_{models}}}$')
        # plt.tight_layout()
        axes[4][0].set_yticklabels(["-0.2", "", "0", "", "0.2"])
        axes[4][2].set_xticklabels(["-0.2", "", "0", "", "0.2"])
        fig.subplots_adjust(hspace=0, wspace=0)
        path = os.path.join(anlysdir, sfiles_str)
        savename = os.path.join(path, 'ellipticities.pdf')
        print(savename)
        plt.savefig(savename, dpi=500, transparent=True, bbox_inches='tight', pad_inches=0)
        plt.close()

    # # complex ellipticity summary plot
    if ELLIPTC_ALL_LOOP:
        print("### Plotting complex ellipticity summary plot")
        # k = ["H3S0A0B90G0", "H10S0A0B90G0", "H36S0A0B90G0"]
        k = keys
        kwargs = dict(verbose=1, legend=True, adjust_limits=True)
        # sfiles = states
        path = os.path.join(anlysdir, sfiles_str)
        loadname = 'qpms.pkl'
        if path is None:
            path = ""
        elif os.path.exists(path):
            loadname = os.path.join(path, loadname)
        with open(loadname, 'rb') as f:
            qpms = pickle.load(f)
        # complex ellipticities
        plots = []
        markers = [['o', 'o'], ['2', '2'], ['^', '^'], ['8', '8'], ['s', 's'],
                   ['*', '*'], ['+', '+'], ['x', 'x'], ['d', 'd'], ['v', 'v'],
                   ['h', 'h'], ['1', '1'], ['p', 'p'], ['<', '<'], ['>', '>']]
        labels = []
        for i, ki in enumerate(k):
            files = sfiles[ki]
            for f in files:
                eq, gq = qpms[f]
                ea, eb, ephi = qpm_props(eq)
                gprops = [qpm_props(q) for q in gq]
                ga, gb, gphi = [p[0] for p in gprops], \
                               [p[1] for p in gprops], \
                               [p[2] for p in gprops]
                eagle_epsilon = complex_ellipticity(ea, eb, ephi)
                glass_epsilon = complex_ellipticity(ga, gb, gphi)
                name = os.path.basename(f).replace(".state", "")
                plt.figure(figsize=(5, 5))
                eagle_plot = complex_ellipticity_plot(
                    [glass_epsilon, eagle_epsilon],
                    scatter=True, samples=[0, -1],
                    ensemble_averages=True,
                    colors=[GLEAMcolors.colors[i], GLEAMcolors.colors[i]], alpha=0.8,
                    lss=['--', '--'], markers=markers[i], markersizes=[2, 6],
                    contours=False, levels=6, cmap=None,
                    origin_marker=True, adjust_limits=False,
                    annotation_color='black', legend=ki)
                glass_plot = complex_ellipticity_plot(
                    [glass_epsilon], scatter=False,
                    contours=True, levels=6,
                    cmap=GLEAMcolors.cmap_from_color(GLEAMcolors.colors[i]),
                    alpha=0.5,
                    origin_marker=True, adjust_limits=False,
                    annotation_color='black')
                plots.append(eagle_plot)
                labels.append(ki)
        if kwargs['adjust_limits']:
            ax = plt.gca()
            ax.set_aspect('equal')
            lim = 1.1 * max(np.abs(ax.get_ylim()+ax.get_xlim()))
            plt.xlim(left=-lim, right=lim)
            plt.ylim(bottom=-lim, top=lim)
        leg = None
        if kwargs['legend']:
            leg = plt.legend([p[-1] for p in plots], labels,
                             loc='upper left',
                             fontsize=8, ncol=2, markerscale=0.75,
                             handlelength=0.5, borderpad=0.5,
                             columnspacing=0.5,
                             fancybox=True)
            plt.setp(leg.get_lines(), linewidth=0)
        plt.tight_layout()
        savename = "ellipticity_all.{}".format(extension)
        if path is None:
            path = ""
        if os.path.exists(path):
            savename = os.path.join(path, savename)
        if kwargs.get('verbose', False):
            print(savename)
        plt.savefig(savename, dpi=500, transparent=True, bbox_inches='tight', pad_inches=0)
        if leg is not None:
            leg.remove()
        plt.close()

    # # potential analysis
    if POTENTIAL_LOOP:
        print("### Calculating lensing potentials")
        # k = ["H3S0A0B90G0", "H10S0A0B90G0", "H36S0A0B90G0"]
        k = keys
        kwargs = dict(N=85, verbose=1)
        # sfiles = states
        path = os.path.join(anlysdir, sfiles_str)
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
    if POTENTIAL_LOOP:
        print("### Plotting lensing potential maps")
        # k = ["H3S0A0B90G0", "H10S0A0B90G0", "H36S0A0B90G0"]
        k = keys
        kwargs = dict(verbose=1)
        # sfiles = states
        path = os.path.join(anlysdir, sfiles_str)
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
                pltkw = dict(cmap='magma')
                # vmin=eagle_map.min(), vmax=eagle_map.max(),
                # levels=np.linspace(np.min((ens_avg, eagle_map)),
                #                    np.max((ens_avg, eagle_map)), 25))
                levels = 25
                eimg = axes[0].contourf(egx, egy, eagle_map, levels, **pltkw)
                axes[0].set_title('EAGLE model', fontsize=12)
                # plt.colorbar(eimg, ax=axes[0])
                gimg = axes[1].contourf(gx[0], gy[0], ens_avg[::-1, :], levels, **pltkw)
                axes[1].set_title('Ensemble average', fontsize=12)
                # plt.colorbar(gimg, ax=axes[1])
                # plt.colorbar(eimg, ax=axes.ravel().tolist(), shrink=0.9)
                axes[0].set_aspect('equal')
                axes[1].set_aspect('equal')
                plt.suptitle(name)
                # save the figure
                savename = "potential_{}.{}".format(name, extension)
                if path is None:
                    path = ""
                if os.path.exists(os.path.join(path, ki)):
                    savename = os.path.join(path, ki, savename)
                elif os.path.exists(path):
                    savename = os.path.join(path, savename)
                if kwargs.get('verbose', False):
                    print(savename)
                plt.savefig(savename, dpi=400, transparent=True, bbox_inches='tight', pad_inches=0)
                # plt.show()
                plt.close()

    # # Roche potentials and scalar products
    if ROCHE_LOOP:
        print("### Calculating Roche potentials and scalar products")
        # k = ["H3S0A0B90G0", "H10S0A0B90G0", "H36S0A0B90G0"]
        k = keys
        kwargs = dict(N=85, calc_iprod=True, optimized=True, verbose=1)
        # sfiles = states
        path = os.path.join(anlysdir, sfiles_str)
        degarrs, scalarRoche = degarr_loop(k, kappa_files, sfiles, **kwargs)
        for ki in k:
            files = sfiles[ki]
            for sf in files:
                name = os.path.basename(sf).replace(".state", "")
                textname = "scalarRoche_{}.txt".format(name)
                ip = scalarRoche[sf]
                sortedidcs = np.argsort(ip)
                if path is None:
                    path = ""
                if os.path.exists(os.path.join(path, ki)):
                    textname = os.path.join(path, ki, textname)
                elif os.path.exists(path):
                    textname = os.path.join(path, textname)
                if kwargs.get('verbose', False):
                    print(textname)
                np.savetxt(textname, np.c_[ip, sortedidcs], fmt='%4.4g',
                           delimiter=' ', newline=os.linesep)
        if 1:
            savenames = ['degarrs.pkl', 'scalarRoche.pkl']
            for o, savename in zip([degarrs, scalarRoche], ['degarrs.pkl', 'scalarRoche.pkl']):
                if path is None:
                    path = ""
                elif os.path.exists(path):
                    savename = os.path.join(path, savename)
                with open(savename, 'wb') as f:
                    print("Saving " + savename)
                    pickle.dump(o, f)

    # # Roche potential scalar product histograms
    if ROCHE_HIST_LOOP:
        print("### Plotting Roche potential scalar product histograms")
        # k = ["H3S0A0B90G0", "H10S0A0B90G0", "H36S0A0B90G0"]
        k = keys
        kwargs = dict(verbose=1)
        # sfiles = states
        path = os.path.join(anlysdir, sfiles_str)
        o = []
        for loadname in ['degarrs.pkl', 'scalarRoche.pkl']:
            if path is None:
                path = ""
            elif os.path.exists(path):
                loadname = os.path.join(path, loadname)
            with open(loadname, 'rb') as f:
                o.append(pickle.load(f))
        degarrs, scalarRoche = o
        for ki in k:
            files = sfiles[ki]
            for sf in files:
                gx, gy, eagle_degarr, glass_degarrs = degarrs[sf]
                # ip = [sigma_product(eagle_degarr, gdegarr) for gdegarr in glass_degarrs]
                ip = scalarRoche[sf]
                name = os.path.basename(sf).replace(".state", "")
                # plt.figure(figsize=(5.68, 5.392))
                plt.hist(ip, bins=14, color=GLEAMcolors.cyan_dark, alpha=0.7, rwidth=0.85)
                plt.xlim(left=-1, right=1)
                plot_labelbox(name, position='top left', color='black')
                plt.xlabel(r'$\langle\mathcal{P}, \mathcal{P}_{\mathsf{model}}\rangle$',
                           fontsize=14)
                plt.ylabel(r'$\mathrm{\mathsf{N_{models}}}$')
                plt.tight_layout()
                # save the figure
                savename = "scalarRoche_hist{}.{}".format(name, extension)
                if path is None:
                    path = ""
                if os.path.exists(os.path.join(path, ki)):
                    savename = os.path.join(path, ki, savename)
                elif os.path.exists(path):
                    savename = os.path.join(path, savename)
                if kwargs.get('verbose', False):
                    print(savename)
                plt.savefig(savename, dpi=200, transparent=True, bbox_inches='tight', pad_inches=0)
                # plt.show()
                plt.close()

    # # Roche potential maps
    if ROCHE_MAP_LOOP:
        print("### Plotting Roche potential maps")
        # k = ["H3S0A0B90G0", "H10S0A0B90G0", "H36S0A0B90G0"]
        k = keys
        kwargs = dict(verbose=1)
        # sfiles = states
        path = os.path.join(anlysdir, sfiles_str)
        loadnames = ['degarrs.pkl', 'scalarRoche.pkl']
        if path is None:
            path = ""
        elif os.path.exists(path):
            loadnames = [os.path.join(path, l) for l in loadnames]
        with open(loadnames[0], 'rb') as f1:
            degarrs = pickle.load(f1)
        with open(loadnames[1], 'rb') as f2:
            scalarRoche = pickle.load(f2)
        for ki in k:
            files = sfiles[ki]
            for sf in files:
                gls = glass.glcmds.loadstate(sf)
                gls.make_ensemble_average()
                name = os.path.basename(sf).replace(".state", "")
                # get ordered indices
                textname = "scalarRoche_{}.txt".format(name)
                if path is None:
                    path = ""
                if os.path.exists(os.path.join(path, ki)):
                    textname = os.path.join(path, ki, textname)
                elif os.path.exists(path):
                    textname = os.path.join(path, textname)
                if os.path.exists(textname):
                    sortedidcs = np.int32(np.loadtxt(textname).T[1])
                else:
                    ip = scalarRoche[sf]
                    sortedidcs = np.argsort(ip)
                minidx = sortedidcs[0]
                maxidx = sortedidcs[-1]
                # save target string template
                savename = "Roche_potential_{{}}_{n}.{e}".format("", n=name, e=extension)
                if path is None:
                    path = ""
                if os.path.exists(os.path.join(path, ki)):
                    savename = os.path.join(path, ki, savename)
                elif os.path.exists(path):
                    savename = os.path.join(path, savename)
                if kwargs.get('verbose', False):
                    print("Best/worst model of {}: {:4d}/{:4d}".format(name, maxidx, minidx))
                # get best/worst/ens_avg models
                gx, gy, eagle_degarr, glass_degarrs = degarrs[sf]
                # # ip = [sigma_product(eagle_degarr, gdegarr) for gdegarr in glass_degarrs]
                max_model = glass_degarrs[maxidx]  # best model
                min_model = glass_degarrs[minidx]  # worst model
                _, _, avg_model = roche_potential(gls.ensemble_average, N=85)
                # some defaults
                def five(grid):
                    msk = radial_mask(grid, radius=int(0.8*grid.shape[0]*0.5))
                    return np.max(grid[msk])
                kw = dict(log=1, zero_level='center', cmax=five,
                          cmap=GLEAMcmaps.phoenix, alpha=0.2,
                          background=None, color='black',
                          levels=25, contours=1, linewidths=1.5,
                          contours_only=0,
                          clabels=1, colorbar=0, label=ki, fontsize=22, scalebar=1)
                # SEAGLE model
                plt.figure(figsize=(5, 5))
                roche_potential_plot((gx, gy, eagle_degarr), **kw)
                plot_annulus((0.5, 0.5), 0.7*0.5, color='black', alpha=0.8, lw=2)
                if kwargs.get('verbose', False):
                    print(savename.format('true'))
                plt.tight_layout()
                plt.savefig(savename.format('true'), dpi=500, transparent=True,
                            bbox_inches='tight', pad_inches=0)
                # plt.show()
                plt.close()
                # ensemble average model
                plt.figure(figsize=(5, 5))
                roche_potential_plot((gx, gy, avg_model), **kw)
                plot_annulus((0.5, 0.5), 0.7*0.5, color='black', alpha=0.8, lw=2)
                if kwargs.get('verbose', False):
                    print(savename.format('ens_avg'))
                plt.tight_layout()
                plt.savefig(savename.format('ens_avg'), dpi=500, transparent=True,
                            bbox_inches='tight', pad_inches=0)
                # plt.show()
                plt.close()
                # best model
                plt.figure(figsize=(5, 5))
                roche_potential_plot((gx, gy, max_model), **kw)
                plot_annulus((0.5, 0.5), 0.7*0.5, color='black', alpha=0.8, lw=2)
                if kwargs.get('verbose', False):
                    print(savename.format('best'))
                plt.tight_layout()
                plt.savefig(savename.format('best'), dpi=500, transparent=True,
                            bbox_inches='tight', pad_inches=0)
                # plt.show()
                plt.close()
                # worst model
                plt.figure(figsize=(5, 5))
                roche_potential_plot((gx, gy, min_model), **kw)
                plot_annulus((0.5, 0.5), 0.7*0.5, color='black', alpha=0.8, lw=2)
                if kwargs.get('verbose', False):
                    print(savename.format('worst'))
                plt.tight_layout()
                plt.savefig(savename.format('worst'), dpi=500, transparent=True,
                            bbox_inches='tight', pad_inches=0)
                # plt.show()
                plt.close()

    # # DEBUG: Roche potential
    if ROCHE_DEBUG_LOOP:
        print("### DEBUGGING Roche potential maps")
        # k = ["H3S0A0B90G0", "H10S0A0B90G0", "H36S0A0B90G0"]
        k = keys
        kwargs = dict(verbose=1)
        # sfiles = states
        path = os.path.join(anlysdir, sfiles_str)
        loadnames = ['degarrs.pkl', 'scalarRoche.pkl']
        if path is None:
            path = ""
        elif os.path.exists(path):
            loadnames = [os.path.join(path, l) for l in loadnames]
        # with open(loadnames[0], 'rb') as f1:
        #     degarrs = pickle.load(f1)
        with open(loadnames[1], 'rb') as f2:
            scalarRoche = pickle.load(f2)
        for ki in k:
            print(ki)
            files = sfiles[ki]
            for sf in files:
                gls = glass.glcmds.loadstate(sf)
                gls.make_ensemble_average()
                name = os.path.basename(sf).replace(".state", "")
                print(name)
                # # get ordered indices
                textname = "scalarRoche_{}.txt".format(name)
                if path is None:
                    path = ""
                if os.path.exists(os.path.join(path, ki)):
                    textname = os.path.join(path, ki, textname)
                elif os.path.exists(path):
                    textname = os.path.join(path, textname)
                if os.path.exists(textname):
                    sortedidcs = np.int32(np.loadtxt(textname).T[1])
                else:
                    ip = scalarRoche[sf]
                    sortedidcs = np.argsort(ip)
                firstidx = sortedidcs[0]
                lastidx = sortedidcs[-1]
                # get EAGLE map
                if kappa_files[ki]:
                        eagle_model = fits.getdata(kappa_files[ki][0], header=True)
                else:
                    eagle_model = None
                if kwargs.get('verbose', False):
                    print('Loading '+kappa_files[ki][0])
                eagle_kappa_map = eagle_model[0]
                # resample SEAGLE kappa map
                eagle_pixel = eagle_model[1]['CDELT2']*3600
                obj, dta = gls.ensemble_average['obj,data'][0]
                glass_maprad = obj.basis.top_level_cell_size * obj.basis.pixrad
                glass_extent = (-glass_maprad, glass_maprad, -glass_maprad, glass_maprad)
                glass_shape = (2*obj.basis.pixrad+1,)*2
                eagle_kappa_map = downsample_model(eagle_kappa_map, glass_extent, glass_shape,
                                                   pixel_scale=eagle_pixel)
                print("SEAGLE kappa map: {} @ {} arcsec".format(eagle_kappa_map.shape,
                                                                glass_maprad))
                # calculate Roche potentials
                ex, ey, eagle_roche = roche_potential_grid(eagle_kappa_map, N=85,
                                                           grid_size=2*glass_maprad)
                gx, gy, avg_roche = roche_potential(gls.ensemble_average, N=85)
                for i in [sortedidcs[-1]]:  # the 10 best Roche scalars
                    x, y, roche = roche_potential(gls.models[i], N=85)
                    roche_scalar_map = sigma_product_map(eagle_roche, roche, rmask=True, radius=.7)
                    roche_scalar = np.sum(roche_scalar_map)
                    print("Model {idx}: Roche scalar={scalar}".format(idx=i, scalar=roche_scalar))
                    plt.imshow(roche_scalar_map, cmap='Spectral', origin='Lower',
                               vmin=-np.max(roche_scalar_map),
                               vmax=np.max(roche_scalar_map))
                    # plot_annulus((0.5, 0.5), 0.7*0.5, color='black', alpha=0.2, lw=1)
                    plot_annulus_region((0.5, 0.5), 0.7*0.5, color='white', alpha=0.1)
                    plt.colorbar()
                    plt.show()

    # # kappa profile ensembles
    if 0 and K_PROFILE_LOOP:
        print("### Plotting kappa radial profiles ensembles")
        # k = ["H3S0A0B90G0", "H10S0A0B90G0", "H36S0A0B90G0"]
        k = keys
        kwargs = dict(verbose=1)
        # sfiles = states
        path = os.path.join(anlysdir, sfiles_str)
        for ki in k:
            print(ki)
            for idx in range(len(sfiles[ki])):
                sf = sfiles[ki][idx]
                print(sf)
                name = os.path.basename(sf).replace(".state", "")
                if kappa_files[ki]:
                    eagle_model = fits.getdata(kappa_files[ki][0], header=True)
                    em = LensModel(kappa_files[ki][0])
                else:
                    eagle_model = None
                lm = LensModel(sf)
                # plt.figure(figsize=(5.68, 5.392))
                kappa_profiles_plot(lm, as_range=True, refined=False, ensemble_average=False,
                                    cmap=GLEAMcmaps.agaveglitch,
                                    interpolate=100, levels=40,
                                    adjust_limits=True,
                                    kappa1_line=True, einstein_radius_indicator=True,
                                    annotation_color='white', label_axes=True,
                                    label=ki, fontsize=18)
                eagle_kappa_map = eagle_model[0]
                eagle_kappa_map = np.flip(eagle_kappa_map, 0)
                eagle_pixrad = tuple(r//2 for r in eagle_kappa_map.shape)
                eagle_maprad = eagle_pixrad[1]*eagle_model[1]['CDELT2']*3600
                kappa_profile_plot(em, kappa1_line=False,
                                   maprad=eagle_maprad, color=GLEAMcolors.red, ls='-')
                plt.tight_layout()
                savename = "kappa_profiles_{}.{}".format(name, extension)
                if path is None:
                    path = ""
                if os.path.exists(os.path.join(path, ki)):
                    savename = os.path.join(path, ki, savename)
                elif os.path.exists(path):
                    savename = os.path.join(path, savename)
                if kwargs.get('verbose', False):
                    print('Saving '+savename)
                print(savename)
                plt.savefig(savename, dpi=500, transparent=True, bbox_inches='tight', pad_inches=0)
                plt.close()

    # # kappa profile ensembles
    if K_PROFILE_LOOP:
        print("### Plotting kappa radial profiles ensembles (single figure)")
        matplotlib.rcParams['xtick.labelsize'] = 16
        matplotlib.rcParams['ytick.labelsize'] = 16
        matplotlib.rcParams['axes.labelsize'] = 18
        matplotlib.rcParams['axes.titlesize'] = 18
        k = ["H3S0A0B90G0", "H10S0A0B90G0", "H36S0A0B90G0",
             "H2S2A0B90G0", "H3S1A0B90G0", "H2S1A0B90G0",
             "H160S0A90B0G0", "H4S3A0B0G90", "H30S0A0B90G0",
             "H13S0A0B90G0", "H2S7A0B90G0", "H1S0A0B90G0",
             "H1S1A0B90G0", "H23S0A0B90G0", "H234S0A0B90G0"]
        kwargs = dict(verbose=1)
        # sfiles = states
        path = os.path.join(anlysdir, sfiles_str)
        fig, axes = plt.subplots(len(k)//3, 3, sharex=False, sharey=False,
                                 figsize=(6, 9))
        for i, ki in enumerate(k):
            print(ki)
            for idx in range(len(sfiles[ki])):
                sf = sfiles[ki][idx]
                print(sf)
                name = os.path.basename(sf).replace(".state", "")
                if kappa_files[ki]:
                    eagle_model = fits.getdata(kappa_files[ki][0], header=True)
                    em = LensModel(kappa_files[ki][0])
                else:
                    eagle_model = None
                lm = LensModel(sf)
                plt.sca(axes[i // 3][i % 3])
                extent = [0, 2.5, 0.5, 3]
                eagle_kappa_map = eagle_model[0]
                eagle_kappa_map = np.flip(eagle_kappa_map, 0)
                eagle_pixrad = tuple(r//2 for r in eagle_kappa_map.shape)
                eagle_maprad = eagle_pixrad[1]*eagle_model[1]['CDELT2']*3600
                kappa_profile_plot(em, kappa1_line=False,
                                   maprad=eagle_maprad, color=GLEAMcolors.red, ls='-')
                plots, _, _ = kappa_profiles_plot(lm, as_range=True, cmap=GLEAMcmaps.agaveglitch,
                                                  interpolate=150, levels=40, refined=False,
                                                  ensemble_average=False, label_axes=False,
                                                  adjust_limits=False,
                                                  kappa1_line=True, einstein_radius_indicator=True,
                                                  fontsize=12, annotation_color='white')
                plt.contourf(np.ones((4, 4))*plots[0].levels[0], extent=extent,
                             cmap=GLEAMcmaps.agaveglitch, levels=plots[0].levels, zorder=-99)
                plot_labelbox(ki, position='top right', padding=(0.03, 0.04), color='white',
                              fontsize=8)
                axes[i // 3][i % 3].set_xlim(left=0, right=2.5)
                axes[i // 3][i % 3].set_ylim(bottom=0.75, top=3)
                axes[i // 3][i % 3].set_yticks(np.linspace(extent[2], extent[3], 10))
                axes[i // 3][i % 3].set_xticks(np.linspace(extent[0], extent[1], 6))
                if (i % 3) == 0:
                    axes[i // 3][i % 3].set_yticklabels(
                        ["", "1", "", "1.5", "", "2", "", "2.5", "", ""])
                    if (i // 3) == 2:
                        axes[i // 3][i % 3].set_ylabel(r'$\mathsf{\kappa}_{<\mathsf{R}}$',
                                                       fontsize=22)
                else:
                    axes[i // 3][i % 3].set_yticklabels([""]*10)
                if (i // 3) > 3:
                    axes[i // 3][i % 3].set_xticklabels(["0", "", "1", "", "2", ""])
                    if (i % 3) == 1:
                        axes[i // 3][i % 3].set_xlabel(r'R [arcsec]', fontsize=16)
                else:
                    axes[i // 3][i % 3].set_xticklabels([""]*6)
                #  axes[4][0].set_yticklabels(["0.75", "", "", "1.5", "", "2", "", "2.5", "", ""])
                # axes[4][2].set_xticklabels(["0", "", "1", "", "2", "2.5"])
                axes[i // 3][i % 3].set_frame_on(True)
                axes[i // 3][i % 3].set_facecolor(GLEAMcmaps.agaveglitch(0.1))
        # plt.tight_layout()
        fig.subplots_adjust(hspace=0, wspace=0)
        savename = "kappa_profiles.{}".format(extension)
        savename = os.path.join(path, savename)
        if kwargs.get('verbose', False):
            print('Saving '+savename)
        print(savename)
        plt.savefig(savename, dpi=500, transparent=True, bbox_inches='tight', pad_inches=0)
        plt.close()

    # # chi2 vs scalarRoche
    if 0 and CHI2VSROCHE_LOOP:
        print("### Plotting chi2 vs scalar product scatter")
        # k = keys
        k = ["H3S0A0B90G0", "H10S0A0B90G0", "H36S0A0B90G0",
             "H2S2A0B90G0", "H3S1A0B90G0", "H2S1A0B90G0",
             "H160S0A90B0G0", "H4S3A0B0G90", "H30S0A0B90G0",
             "H13S0A0B90G0", "H2S7A0B90G0", "H1S0A0B90G0",
             "H1S1A0B90G0", "H23S0A0B90G0", "H234S0A0B90G0"]
        kwargs = dict(optimized=True, psf_file=psf_file, reduced=True, verbose=1)
        # sfiles = states
        path = os.path.join(anlysdir, sfiles_str)
        loadname = 'scalarRoche.pkl'
        if path is None:
            path = ""
        elif os.path.exists(path):
            loadname = os.path.join(path, loadname)
        with open(loadname, 'rb') as f:
            scalarRoche = pickle.load(f)
        for ki in k:
            for sf in sfiles[ki]:
                name = os.path.basename(sf).replace(".state", "")
                loadname = "reconsrc_{}.pkl".format(name)
                if path is None:
                    path = ""
                if os.path.exists(os.path.join(path, ki)):
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
                ip = scalarRoche[sf]
                # Plot chi2 vs scalar product
                plt.plot(chi2, ip, marker='o', lw=0, color=GLEAMcolors.blue_marguerite, alpha=0.2)
                plot_labelbox(ki, position='bottom left', padding=(0.04, 0.04), color='black',
                              fontsize=18)
                plt.xlabel(r'$\chi^{2}$', fontsize=18)
                plt.ylabel(r'$\langle\mathcal{P}, \mathcal{P}_{\mathsf{model}}\rangle$',
                           fontsize=18)
                plt.tight_layout()
                # save the figure
                savename = "chi2_VS_scalarRoche_{}.{}".format(name, extension)
                if path is None:
                    path = ""
                if os.path.exists(os.path.join(path, ki)):
                    savename = os.path.join(path, ki, savename)
                elif os.path.exists(path):
                    savename = os.path.join(path, savename)
                if kwargs.get('verbose', False):
                    print('Saving '+savename)
                plt.savefig(savename, dpi=500, transparent=True, bbox_inches='tight', pad_inches=0)
                # plt.show()
                plt.close()

    # # chi2 vs scalarRoche
    if CHI2VSROCHE_LOOP:
        print("### Plotting chi2 vs scalar product scatter (single figure)")
        matplotlib.rcParams['xtick.labelsize'] = 16
        matplotlib.rcParams['ytick.labelsize'] = 16
        matplotlib.rcParams['axes.labelsize'] = 18
        matplotlib.rcParams['axes.titlesize'] = 18
        # k = keys
        k = ["H3S0A0B90G0", "H10S0A0B90G0", "H36S0A0B90G0",
             "H2S2A0B90G0", "H3S1A0B90G0", "H2S1A0B90G0",
             "H160S0A90B0G0", "H4S3A0B0G90", "H30S0A0B90G0",
             "H13S0A0B90G0", "H2S7A0B90G0", "H1S0A0B90G0",
             "H1S1A0B90G0", "H23S0A0B90G0", "H234S0A0B90G0"
        ]
        kwargs = dict(optimized=True, psf_file=psf_file, reduced=True, verbose=1)
        # sfiles = states
        path = os.path.join(anlysdir, sfiles_str)
        loadname = 'chi2vsRocheScalar.pkl'
        if path is None:
            path = ""
        elif os.path.exists(path):
            loadname = os.path.join(path, loadname)
        with open(loadname, 'rb') as f:
            chi2vsRocheScalar = pickle.load(f)
        fig, axes = plt.subplots(len(k)//3, 3, sharex=False, sharey=False,
                                 figsize=(6, 9))
        for i, ki in enumerate(k):
            for sf in sfiles[ki]:
                name = os.path.basename(sf).replace(".state", "")
                print(ki)
                # gather chi2 values
                chi2, ip = chi2vsRocheScalar[ki]
                # Plot chi2 vs scalar product
                plt.sca(axes[i // 3][i % 3])
                # plt.plot(chi2, ip, marker='o', markersize=2, lw=0, color=GLEAMcolors.blue_marguerite, alpha=0.2)
                # plot_labelbox(ki, position='bottom left', padding=(0.04, 0.04), color='black',
                #               fontsize=10)
                plt.hexbin(chi2, ip, gridsize=20, xscale='linear', yscale='linear', extent=[2, 5, -0.25, 1],
                           # cmap=GLEAMcmaps.vilux, vmin=None, vmax=None,
                           cmap=GLEAMcmaps.reverse(GLEAMcmaps.cyberfade), vmin=None, vmax=None,
                           )
                if i in [7, 8, 9, 13]:
                    plot_labelbox(ki, position='top right', padding=(0.03, 0.04), color='black',
                                  fontsize=8)
                else:
                    plot_labelbox(ki, position='bottom left', padding=(0.03, 0.04), color='black',
                                  fontsize=8)
                axes[i // 3][i % 3].set_yticks((-0.25, 0, 0.25, 0.5, 0.75, 1))
                axes[i // 3][i % 3].set_xticks((2, 2.5, 3, 3.5, 4, 4.5, 5))
                if (i % 3) == 0:
                    axes[i // 3][i % 3].set_yticklabels(["", "0", "", "0.5", "", "1"])
                    if (i // 3) == 2:
                        axes[i // 3][i % 3].set_ylabel(
                            r'$\langle\mathcal{P}, \mathcal{P}_{\mathsf{model}}\rangle$',
                            fontsize=18)
                else:
                    axes[i // 3][i % 3].set_yticklabels(["", "", "", "", "", ""])
                if (i // 3) > 3:
                    axes[i // 3][i % 3].set_xticklabels(["2", "", "3", "", "4", "", ""])
                    if (i % 3) == 1:
                        axes[i // 3][i % 3].set_xlabel(r'$\chi^{2}_{\nu}$', fontsize=18)
                else:
                    axes[i // 3][i % 3].set_xticklabels(["", "", "", "", "", "", ""])
                plt.xlim(2, 5)
                plt.ylim(-0.25, 1)
        plt.tight_layout()
        axes[4][2].set_xticklabels(["2", "", "3", "", "4", "", "5"])
        fig.subplots_adjust(hspace=0, wspace=0)
        # picklesave = os.path.join(path, 'chi2vsRocheScalar.pkl')
        # with open(picklesave, 'wb') as f:
        #     pickle.dump(chi2vsRocheScalar, f)
        # save the figure
        savename = "chi2_VS_scalarRoche.{}".format(extension)
        savename = os.path.join(path, savename)
        if kwargs.get('verbose', False):
            print('Saving '+savename)
        plt.savefig(savename, dpi=500, transparent=True, bbox_inches='tight', pad_inches=0)
        # plt.show()
        plt.close()

    # # data maps
    if DATA_LOOP:
        print("### Plotting data maps")
        # k = ["H3S0A0B90G0", "H10S0A0B90G0", "H36S0A0B90G0"]
        k = keys
        kwargs = dict(verbose=1)
        # sfiles = states
        path = os.path.join(anlysdir, sfiles_str)
        for ki in k:
            for sf in sfiles[ki]:
                name = os.path.basename(sf).replace(".state", "")
                loadname = "reconsrc_{}.pkl".format(name)
                if path is None:
                    path = ""
                if os.path.exists(os.path.join(path, ki)):
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
                cmap = GLEAMcmaps.vilux  # Spectral_r
                plt.figure(figsize=(5, 5))
                plt.imshow(data, cmap=cmap, interpolation='bicubic',
                           origin='Lower', extent=extent)
                plot_labelbox(ki, position='top right', padding=(0.04, 0.04))
                plot_scalebar(extent[-1], length=1, position='bottom left', origin='center')
                plt.axis('off')
                plt.gcf().axes[0].get_xaxis().set_visible(False)
                plt.gcf().axes[0].get_yaxis().set_visible(False)
                # plt.colorbar()
                # save the figure
                savename = "data_{}.{}".format(ki, extension)
                if path is None:
                    path = ""
                if os.path.exists(os.path.join(path, ki)):
                    savename = os.path.join(path, ki, savename)
                elif os.path.exists(path):
                    savename = os.path.join(path, savename)
                if kwargs.get('verbose', False):
                    print('Saving '+savename)
                plt.tight_layout()
                plt.savefig(savename, dpi=500, transparent=True, bbox_inches='tight', pad_inches=0)
                # plt.show()
                plt.close()
                break

    # source plane map
    if SOURCE_LOOP:
        print("### Plotting source maps")
        # k = ["H3S0A0B90G0", "H10S0A0B90G0", "H36S0A0B90G0"]
        k = keys
        verbose = 1
        # sfiles = states
        path = os.path.join(anlysdir, sfiles_str)
        for ki in k:
            for sf in sfiles[ki]:
                name = os.path.basename(sf).replace(".state", "")
                loadname = "reconsrc_{}.pkl".format(name)
                if path is None:
                    path = ""
                if os.path.exists(os.path.join(path, ki)):
                    loadname = os.path.join(path, ki, loadname)
                elif os.path.exists(path):
                    loadname = os.path.join(path, loadname)
                if os.path.exists(loadname):
                    with open(loadname, 'rb') as f:
                        recon_src = pickle.load(f)
                if verbose:
                    print('Loading '+loadname)
                # noise map
                lo = recon_src.lensobject
                signals, variances = lo.flatfield(lo.data, size=0.2)
                gain, _ = lo.gain(signals=signals, variances=variances)
                f = 1./(25*gain)
                bias = 0.001*np.max(f * lo.data)
                sgma2 = lo.sigma2(f=f, add_bias=bias)
                dta_noise = np.random.normal(0, 1, size=lo.data.shape)
                dta_noise = dta_noise * np.sqrt(sgma2)
                # Ensemble average
                recon_src.chmdl(-1)
                kwargs = dict(method='minres', use_psf=True, cached=False, sigma2=sgma2)
                src = recon_src.plane_map(**kwargs)
                r = recon_src.r_max
                if r is not None:
                    extent = [-r, r, -r, r]
                else:
                    extent = None
                cmap = GLEAMcmaps.vilux  # 'Spectral_r'
                plt.figure(figsize=(5, 5))
                plt.imshow(src, cmap=cmap,
                           origin='Lower', extent=extent, vmin=0)
                plot_labelbox(ki, position='top right', padding=(0.04, 0.04))
                plot_scalebar(r, length=0.1, position='bottom left', origin='center')
                plt.axis('off')
                plt.gcf().axes[0].get_xaxis().set_visible(False)
                plt.gcf().axes[0].get_yaxis().set_visible(False)
                # plt.colorbar()
                # save the figure
                savename = "rconsrc_{}.{}".format(name, extension)
                if path is None:
                    path = ""
                if os.path.exists(os.path.join(path, ki)):
                    savename = os.path.join(path, ki, savename)
                elif os.path.exists(path):
                    savename = os.path.join(path, savename)
                if verbose:
                    print('Saving '+savename)
                plt.tight_layout()
                plt.savefig(savename, dpi=200, transparent=True, bbox_inches='tight', pad_inches=0)
                # plt.show()
                plt.close()

    # synth map of the ensemble averages
    if SYNTH_LOOP:
        print("### Plotting synthetics of ensemble averages")
        # k = ["H3S0A0B90G0", "H10S0A0B90G0", "H36S0A0B90G0"]
        k = keys
        verbose = 1
        # sfiles = states
        path = os.path.join(anlysdir, sfiles_str)
        for ki in k:
            for sf in sfiles[ki]:
                name = os.path.basename(sf).replace(".state", "")
                loadname = "reconsrc_{}.pkl".format(name)
                if path is None:
                    path = ""
                if os.path.exists(os.path.join(path, ki)):
                    loadname = os.path.join(path, ki, loadname)
                elif os.path.exists(path):
                    loadname = os.path.join(path, loadname)
                if os.path.exists(loadname):
                    with open(loadname, 'rb') as f:
                        recon_src = pickle.load(f)
                if verbose:
                    print('Loading '+loadname)
                # noise map
                lo = recon_src.lensobject
                signals, variances = lo.flatfield(lo.data, size=0.2)
                gain, _ = lo.gain(signals=signals, variances=variances)
                f = 1./(25*gain)
                bias = 0.001*np.max(f * lo.data)
                sgma2 = lo.sigma2(f=f, add_bias=bias)
                dta_noise = np.random.normal(0, 1, size=lo.data.shape)
                dta_noise = dta_noise * np.sqrt(sgma2)
                # Ensemble average
                recon_src.chmdl(-1)
                kwargs = dict(method='minres', use_psf=True, cached=False,
                              from_cache=False, sigma2=sgma2)
                synth = recon_src.reproj_map(**kwargs)
                extent = recon_src.lensobject.extent
                extent = [extent[0], extent[2], extent[1], extent[3]]
                cmap = GLEAMcmaps.vilux  # 'Spectral_r'
                plt.figure(figsize=(5, 5))
                plt.imshow(synth, cmap=cmap, interpolation='bicubic',
                           origin='Lower', extent=extent,
                           vmin=0, vmax=recon_src.lensobject.data.max())
                plot_labelbox(ki, position='top right', padding=(0.04, 0.04))
                plot_scalebar(extent[-1], length=1, position='bottom left', origin='center')
                plt.axis('off')
                plt.gcf().axes[0].get_xaxis().set_visible(False)
                plt.gcf().axes[0].get_yaxis().set_visible(False)
                # plt.colorbar()
                # save the figure
                savename = "synth_{}.{}".format(name, extension)
                if path is None:
                    path = ""
                if os.path.exists(os.path.join(path, ki)):
                    savename = os.path.join(path, ki, savename)
                elif os.path.exists(path):
                    savename = os.path.join(path, savename)
                if verbose:
                    print('Saving '+savename)
                plt.tight_layout()
                plt.savefig(savename, dpi=500, transparent=True, bbox_inches='tight', pad_inches=0)
                # plt.show()
                plt.close()

    # arrival-time surface of the ensemble averages
    if ARRIV_LOOP:
        print("### Plotting arrival-time surface maps of the ensemble averages")
        # k = ["H3S0A0B90G0", "H10S0A0B90G0", "H36S0A0B90G0"]
        k = keys
        verbose = 1
        # sfiles = states
        path = os.path.join(anlysdir, sfiles_str)
        for ki in k:
            for sf in sfiles[ki]:
                name = os.path.basename(sf).replace(".state", "")
                if verbose:
                    print('Loading '+sf)
                gls = glass.glcmds.loadstate(sf)
                gls.make_ensemble_average()
                m = gls.ensemble_average
                cmap = GLEAMcmaps.reverse(GLEAMcmaps.aquaria, set_under='white', set_over='white')
                plt.figure(figsize=(5, 5))
                arrival_time_surface_plot(m, cmap=cmap, images_off=0,
                                          contours_only=0, contours=1, levels=25,
                                          min_contour_shift=0.1,
                                          scalebar=True, label=ki, color='black',
                                          colorbar=0)
                # save the figure
                savename = "arriv_{}.{}".format(name, extension)
                if path is None:
                    path = ""
                if os.path.exists(os.path.join(path, ki)):
                    savename = os.path.join(path, ki, savename)
                elif os.path.exists(path):
                    savename = os.path.join(path, savename)
                if verbose:
                    print('Saving '+savename)
                plt.tight_layout()
                plt.savefig(savename, dpi=500, transparent=True, bbox_inches='tight', pad_inches=0)
                # plt.show()
                plt.close()

    # kappa map of the ensemble averages
    if K_MAP_LOOP:
        print("### Plotting kappa maps of the ensemble averages")
        # k = ["H3S0A0B90G0", "H10S0A0B90G0", "H36S0A0B90G0"]
        k = keys
        verbose = 1
        # sfiles = states
        path = os.path.join(anlysdir, sfiles_str)
        for ki in k:
            for sf in sfiles[ki]:
                name = os.path.basename(sf).replace(".state", "")
                if verbose:
                    print('Loading '+sf)
                gls = glass.glcmds.loadstate(sf)
                gls.make_ensemble_average()
                m = gls.ensemble_average
                plt.figure(figsize=(5, 5))
                kappa_map_plot(m, subcells=1, contours=1, colorbar=0, log=1,
                               oversample=True,
                               scalebar=True, label=ki,
                               cmap=GLEAMcmaps.agaveglitch)
                # save the figure
                savename = "kappa_{}.{}".format(name, extension)
                if path is None:
                    path = ""
                if os.path.exists(os.path.join(path, ki)):
                    savename = os.path.join(path, ki, savename)
                elif os.path.exists(path):
                    savename = os.path.join(path, savename)
                if verbose:
                    print('Saving '+savename)
                plt.tight_layout()
                plt.savefig(savename, dpi=500, transparent=True, bbox_inches='tight', pad_inches=0)
                # plt.show()
                plt.close()

    # kappa maps of the EAGLE models
    if K_TRUE_LOOP:
        print("### Plotting true SEAGLE kappa maps")
        # k = ["H3S0A0B90G0", "H10S0A0B90G0", "H36S0A0B90G0"]
        k = keys
        verbose = 1
        # sfiles = states
        path = os.path.join(anlysdir, sfiles_str)
        for ki in k:
            if ki in sfiles and not sfiles[ki]:
                continue
            if kappa_files[ki]:
                fk = kappa_files[ki][0]
            eagle_model = fits.getdata(fk, header=True)
            eagle_kappa, eagle_hdr = eagle_model
            # eagle_kappa = np.flip(eagle_kappa, 0)
            eagle_pixrad = tuple(r//2 for r in eagle_kappa.shape)
            eagle_maprad = eagle_pixrad[1]*eagle_hdr['CDELT2']*3600
            # plot in GLASS style
            grid = eagle_kappa
            levels, delta = 6, 0.2
            gls_maprad = []
            for sf in sfiles[ki]:
                name = os.path.basename(sf).replace(".state", "")
                gls = glass.glcmds.loadstate(sf)
                gls.make_ensemble_average()
                mapextent = gls.ensemble_average['obj,data'][0][0].basis.mapextent
                gls_maprad.append(mapextent)
            maprad = max(gls_maprad)
            extent = [-maprad, maprad, -maprad, maprad]
            X, Y = np.meshgrid(np.linspace(-eagle_maprad, eagle_maprad, grid.shape[1]),
                               np.linspace(-eagle_maprad, eagle_maprad, grid.shape[0]))
            # masking if necessary
            msk = grid > 0
            if not np.any(msk):
                vmin = -15
                grid += 10**vmin
            else:
                vmin = np.log10(np.amin(grid[msk]))
            # interpolate
            if 0:
                grid = scipy.ndimage.zoom(grid, 3, order=0)
            grid[grid <= 10**vmin] = 10**vmin
            # contour levels
            clev2 = np.arange(delta, levels*delta, delta)
            clevels = np.concatenate((-clev2[::-1], (0,), clev2))
            # plot in log
            kappa1 = 0
            grid = np.log10(grid)
            grid[grid < clevels[0]] = clevels[0]+1e-6
            # plt.figure(figsize=(5, 5))
            plt.contourf(X, Y, grid, cmap=GLEAMcmaps.agaveglitch, antialiased=True,
                         extent=extent, origin='upper', levels=clevels)
            # colorbar
            # ax = plt.gca()
            # cbar = plt.colorbar()
            # lvllbls = ['{:2.1f}'.format(l) if i > 0 else '0'
            #            for (i, l) in enumerate(10**cbar._tick_data_values)]
            # cbar.ax.set_yticklabels(lvllbls)
            # single contour around kappa = 1
            plt.contour(X, Y, grid, levels=(kappa1,), colors=['k'],
                        extent=extent, origin='upper')
            plt.xlim(left=-maprad, right=maprad)
            plt.ylim(bottom=-maprad, top=maprad)
            # scale bar and label
            plot_scalebar(maprad, length=1, position='bottom left', origin='center')
            plt.axis('off')
            plt.gcf().axes[0].get_xaxis().set_visible(False)
            plt.gcf().axes[0].get_yaxis().set_visible(False)
            plot_labelbox(ki, position='top left', padding=(0.03, 0.03))
            # save figure
            savename = "kappa_true_{}.{}".format(ki, extension)
            if path is None:
                path = ""
            if os.path.exists(os.path.join(path, ki)):
                savename = os.path.join(path, ki, savename)
            elif os.path.exists(path):
                savename = os.path.join(path, savename)
            if verbose:
                print('Saving '+savename)
            plt.tight_layout()
            plt.savefig(savename, dpi=500, transparent=True, bbox_inches='tight', pad_inches=0)
            plt.close()

    # # Individual model plots
    if INDIVIDUAL_LOOP:
        print("### Plotting chi2 synthetics, arrival-time surfaces, kappa maps "
              + "of all individual models")
        verbose = True
        k = keys
        # sfiles = states
        for ki in k:
            for sf in sfiles[ki]:
                if verbose:
                    print(sf)

                # # Best/worst synths
                if 1 or SYNTHS_ONLY:
                    name = os.path.basename(sf).replace(".state", "")
                    path = os.path.join(anlysdir, sfiles_str)
                    loadname = "reconsrc_{}.pkl".format(name)
                    if path is None:
                        path = ""
                    if os.path.exists(os.path.join(path, ki)):
                        loadname = os.path.join(path, ki, loadname)
                    elif os.path.exists(path):
                        loadname = os.path.join(path, loadname)
                    if os.path.exists(loadname):
                        with open(loadname, 'rb') as f:
                            recon_src = pickle.load(f)
                    if verbose:
                        print('Loading '+loadname)
                    # noise map
                    lo = recon_src.lensobject
                    signals, variances = lo.flatfield(lo.data, size=0.2)
                    gain, _ = lo.gain(signals=signals, variances=variances)
                    f = 1./(25*gain)
                    bias = 0.001*np.max(f * lo.data)
                    sgma2 = lo.sigma2(f=f, add_bias=bias)
                    dta_noise = np.random.normal(0, 1, size=lo.data.shape)
                    dta_noise = dta_noise * np.sqrt(sgma2)
                    extent = recon_src.lensobject.extent
                    extent = [extent[0], extent[2], extent[1], extent[3]]
                    kwargs = dict(method='minres', use_psf=True, cached=True,
                                  from_cache=True, sigma2=sgma2)
                    # chi2 ordered synths
                    if 1:
                        textname = os.path.join(path, ki, "chi2_{}.txt".format(name))
                        chi2_sorted = np.int32(np.loadtxt(textname).T[1])
                        for i, ichi2 in enumerate(chi2_sorted):
                            label = "chi2_{:04d}".format(i)
                            recon_src.chmdl(ichi2)
                            synth = recon_src.reproj_map(**kwargs)
                            # plt.figure(figsize=(5, 5))
                            plt.imshow(synth, cmap=GLEAMcmaps.vilux,
                                       interpolation='bicubic',
                                       origin='Lower', extent=extent,
                                       vmin=0, vmax=recon_src.lensobject.data.max())
                            plot_scalebar(extent[-1], length=1,
                                          position='bottom left', origin='center')
                            plt.axis('off')
                            plt.gcf().axes[0].get_xaxis().set_visible(False)
                            plt.gcf().axes[0].get_yaxis().set_visible(False)
                            plot_labelbox(ki, position='top left', padding=(0.04, 0.04))
                            # plt.colorbar()
                            # save the figure
                            savename = "{}_synth_{}.{}".format(label, ichi2, extension)
                            if path is None:
                                path = ""
                            if os.path.exists(os.path.join(path, ki)):
                                sig = name.split('.')[-1]
                                if sig == name:
                                    sig = 'orig'
                                mkdir_p(os.path.join(path, ki, 'synths', sig))
                                savename = os.path.join(path, ki, "synths", sig, savename)
                            elif os.path.exists(path):
                                sig = name.split('.')[-1]
                                if sig == name:
                                    sig = 'orig'
                                mkdir_p(os.path.join(path, "synths", sig))
                                savename = os.path.join(path, sig, savename)
                            if verbose:
                                print('Saving '+savename)
                            plt.tight_layout()
                            plt.savefig(savename, dpi=500, transparent=True, bbox_inches='tight', pad_inches=0)
                            # plt.show()
                            plt.close()
                    # Roche scalar ordered synths
                    if 1:
                        textname = os.path.join(path, ki, "scalarRoche_{}.txt".format(name))
                        scalarRoche_sorted = np.int32(np.loadtxt(textname).T[1])
                        for i, idegarr in enumerate(scalarRoche_sorted[::-1]):
                            label = "scalarRoche_{:04d}".format(i)
                            recon_src.chmdl(idegarr)
                            synth = recon_src.reproj_map(**kwargs)
                            # plt.figure(figsize=(5, 5))
                            plt.imshow(synth, cmap=GLEAMcmaps.vilux,
                                       interpolation='bicubic',
                                       origin='Lower', extent=extent,
                                       vmin=0, vmax=recon_src.lensobject.data.max())
                            plot_scalebar(extent[-1], length=1,
                                          position='bottom left', origin='center')
                            plt.axis('off')
                            plt.gcf().axes[0].get_xaxis().set_visible(False)
                            plt.gcf().axes[0].get_yaxis().set_visible(False)
                            plot_labelbox(ki, position='top left', padding=(0.04, 0.04))
                            # plt.colorbar()
                            # save the figure
                            savename = "{}_synth_{}.{}".format(label, idegarr, extension)
                            if path is None:
                                path = ""
                            if os.path.exists(os.path.join(path, ki)):
                                sig = name.split('.')[-1]
                                if sig == name:
                                    sig = 'orig'
                                mkdir_p(os.path.join(path, ki, 'synths', sig))
                                savename = os.path.join(path, ki, "synths", sig, savename)
                            elif os.path.exists(path):
                                sig = name.split('.')[-1]
                                if sig == name:
                                    sig = 'orig'
                                mkdir_p(os.path.join(path, "synths", sig))
                                savename = os.path.join(path, sig, savename)
                            if verbose:
                                print('Saving '+savename)
                            plt.tight_layout()
                            plt.savefig(savename, dpi=200, transparent=True, bbox_inches='tight', pad_inches=0)
                            # plt.show()
                            plt.close()

                # # Best/worst arrival time surfaces
                if 1 and not SYNTHS_ONLY:
                    name = os.path.basename(sf).replace(".state", "")
                    path = os.path.join(anlysdir, sfiles_str)
                    loadname = "reconsrc_{}.pkl".format(name)
                    if path is None:
                        path = ""
                    if os.path.exists(os.path.join(path, ki)):
                        loadname = os.path.join(path, ki, loadname)
                    elif os.path.exists(path):
                        loadname = os.path.join(path, loadname)
                    if os.path.exists(loadname):
                        with open(loadname, 'rb') as f:
                            recon_src = pickle.load(f)
                    if verbose:
                        print('Loading '+loadname)
                    # chi2 ordered arrival time surfaces
                    if 1:
                        textname = os.path.join(path, ki, "chi2_{}.txt".format(name))
                        chi2_sorted = np.int32(np.loadtxt(textname).T[1])
                        for i, ichi2 in enumerate(chi2_sorted):
                            label = "chi2_{:04d}".format(i)
                            m = recon_src.gls.models[ichi2]
                            cmap = GLEAMcmaps.reverse(GLEAMcmaps.aquaria,
                                                      set_under='white', set_over='white')
                            # plt.figure(figsize=(5, 5))
                            arrival_time_surface_plot(m, cmap=cmap, images_off=0,
                                                      contours_only=0, contours=1, levels=25,
                                                      min_contour_shift=0.1,
                                                      scalebar=True, label=ki, color='black',
                                                      colorbar=0)
                            # save the figure
                            savename = "{}_arriv_{}.{}".format(label, ichi2, extension)
                            if path is None:
                                path = ""
                            if os.path.exists(os.path.join(path, ki)):
                                sig = name.split('.')[-1]
                                if sig == name:
                                    sig = 'orig'
                                mkdir_p(os.path.join(path, ki, 'arrivs', sig))
                                savename = os.path.join(path, ki, "arrivs", sig, savename)
                            elif os.path.exists(path):
                                sig = name.split('.')[-1]
                                if sig == name:
                                    sig = 'orig'
                                mkdir_p(os.path.join(path, 'arrivs', sig))
                                savename = os.path.join(path, sig, savename)
                            if verbose:
                                print('Saving '+savename)
                            plt.tight_layout()
                            plt.savefig(savename, dpi=200, transparent=True, bbox_inches='tight', pad_inches=0)
                            # plt.show()
                            plt.close()
                    # Roche scalar ordered arrival time surface
                    if 1:
                        textname = os.path.join(path, ki, "scalarRoche_{}.txt".format(name))
                        scalarRoche_sorted = np.int32(np.loadtxt(textname).T[1])
                        for i, idegarr in enumerate(scalarRoche_sorted[::-1]):
                            label = "scalarRoche_{:04d}".format(i)
                            m = recon_src.gls.models[idegarr]
                            cmap = GLEAMcmaps.reverse(GLEAMcmaps.aquaria,
                                                      set_under='white', set_over='white')
                            # plt.figure(figsize=(5, 5))
                            arrival_time_surface_plot(m, cmap=cmap, images_off=0,
                                                      contours_only=0, contours=1, levels=25,
                                                      min_contour_shift=0.1,
                                                      scalebar=True, label=ki, color='black',
                                                      colorbar=0)
                            # save the figure
                            savename = "{}_arriv_{}.{}".format(label, idegarr, extension)
                            if path is None:
                                path = ""
                            if os.path.exists(os.path.join(path, ki)):
                                sig = name.split('.')[-1]
                                if sig == name:
                                    sig = 'orig'
                                mkdir_p(os.path.join(path, ki, 'arrivs', sig))
                                savename = os.path.join(path, ki, "arrivs", sig, savename)
                            elif os.path.exists(path):
                                sig = name.split('.')[-1]
                                if sig == name:
                                    sig = 'orig'
                                mkdir_p(os.path.join(path, 'arrivs', sig))
                                savename = os.path.join(path, sig, savename)
                            if verbose:
                                print('Saving '+savename)
                            plt.tight_layout()
                            plt.savefig(savename, dpi=200, transparent=True, bbox_inches='tight', pad_inches=0)
                            # plt.show()
                            plt.close()

                # # Best/worst kappa maps
                if 1 and not SYNTHS_ONLY:
                    path = os.path.join(anlysdir, sfiles_str)
                    name = os.path.basename(sf).replace(".state", "")
                    loadname = "reconsrc_{}.pkl".format(name)
                    if path is None:
                        path = ""
                    if os.path.exists(os.path.join(path, ki)):
                        loadname = os.path.join(path, ki, loadname)
                    elif os.path.exists(path):
                        loadname = os.path.join(path, loadname)
                    if os.path.exists(loadname):
                        with open(loadname, 'rb') as f:
                            recon_src = pickle.load(f)
                    if verbose:
                        print('Loading '+loadname)
                    # chi2 kappa maps
                    if 1:
                        textname = os.path.join(path, ki, "chi2_{}.txt".format(name))
                        chi2_sorted = np.int32(np.loadtxt(textname).T[1])
                        for i, ichi2 in enumerate(chi2_sorted):
                            label = "chi2_{:04d}".format(i)
                            m = recon_src.gls.models[ichi2]
                            # plt.figure(figsize=(5, 5))
                            kappa_map_plot(m, subcells=1, contours=1, colorbar=0, log=1,
                                           oversample=True,
                                           scalebar=True, label=ki,
                                           cmap=GLEAMcmaps.agaveglitch)
                            # save the figure
                            savename = "{}_kappa_{}.{}".format(label, ichi2, extension)
                            if path is None:
                                path = ""
                            if os.path.exists(os.path.join(path, ki)):
                                sig = name.split('.')[-1]
                                if sig == name:
                                    sig = 'orig'
                                mkdir_p(os.path.join(path, ki, 'kappas', sig))
                                savename = os.path.join(path, ki, "kappas", sig, savename)
                            elif os.path.exists(path):
                                sig = name.split('.')[-1]
                                if sig == name:
                                    sig = 'orig'
                                mkdir_p(os.path.join(path, "kappas", sig))
                                savename = os.path.join(path, sig, savename)
                            if verbose:
                                print('Saving '+savename)
                            plt.tight_layout()
                            plt.savefig(savename, dpi=200, transparent=True, bbox_inches='tight', pad_inches=0)
                            # plt.show()
                            plt.close()
                    # Roche scalar ordered arrival time surface
                    if 1:
                        textname = os.path.join(path, ki, "scalarRoche_{}.txt".format(name))
                        scalarRoche_sorted = np.int32(np.loadtxt(textname).T[1])
                        for i, idegarr in enumerate(scalarRoche_sorted[::-1]):
                            label = "scalarRoche_{:04d}".format(i)
                            m = recon_src.gls.models[idegarr]
                            # plt.figure(figsize=(5, 5))
                            kappa_map_plot(m, subcells=1, contours=1, colorbar=0, log=1,
                                           oversample=True,
                                           scalebar=True, label=ki,
                                           cmap=GLEAMcmaps.agaveglitch)
                            # save the figure
                            savename = "{}_kappa_{}.{}".format(label, idegarr, extension)
                            if path is None:
                                path = ""
                            if os.path.exists(os.path.join(path, ki)):
                                sig = name.split('.')[-1]
                                if sig == name:
                                    sig = 'orig'
                                mkdir_p(os.path.join(path, ki, 'kappas', sig))
                                savename = os.path.join(path, ki, "kappas", sig, savename)
                            elif os.path.exists(path):
                                sig = name.split('.')[-1]
                                if sig == name:
                                    sig = 'orig'
                                mkdir_p(os.path.join(path, "kappas", sig))
                                savename = os.path.join(path, sig, savename)
                            if verbose:
                                print('Saving '+savename)
                            plt.tight_layout()
                            plt.savefig(savename, dpi=200, transparent=True, bbox_inches='tight', pad_inches=0)
                            # plt.show()
                            plt.close()

                # # Best/worst kappa profiles
                if 1 and not SYNTHS_ONLY:
                    path = os.path.join(anlysdir, sfiles_str)
                    name = os.path.basename(sf).replace(".state", "")
                    loadname = "reconsrc_{}.pkl".format(name)
                    if path is None:
                        path = ""
                    if os.path.exists(os.path.join(path, ki)):
                        loadname = os.path.join(path, ki, loadname)
                    elif os.path.exists(path):
                        loadname = os.path.join(path, loadname)
                    if os.path.exists(loadname):
                        with open(loadname, 'rb') as f:
                            recon_src = pickle.load(f)
                    if verbose:
                        print('Loading '+loadname)
                    if kappa_files[ki]:
                        eagle_model = fits.getdata(kappa_files[ki][0], header=True)
                    else:
                        eagle_model = None
                    if verbose:
                        print('Loading '+kappa_files[ki][0])
                    eagle_kappa_map = eagle_model[0]
                    eagle_kappa_map = np.flip(eagle_kappa_map, 0)
                    eagle_pixrad = tuple(r//2 for r in eagle_kappa_map.shape)
                    eagle_maprad = eagle_pixrad[1]*eagle_model[1]['CDELT2']*3600
                    # chi2 kappa maps
                    if 1:
                        textname = os.path.join(path, ki, "chi2_{}.txt".format(name))
                        chi2_sorted = np.int32(np.loadtxt(textname).T[1])
                        for i, ichi2 in enumerate(chi2_sorted):
                            label = "chi2_{:04d}".format(i)
                            m = recon_src.gls.models[ichi2]
                            # plt.figure(figsize=(5, 5))
                            kappa_profile_plot(m, correct_distances=True,
                                               kappa1_line=True, einstein_radius_indicator=True,
                                               label=ki, color=GLEAMcolors.blue, ls='-')
                            kappa_profile_plot(eagle_kappa_map, correct_distances=False,
                                               kappa1_line=False, einstein_radius_indicator=False,
                                               maprad=eagle_maprad, color=GLEAMcolors.red, ls='-')
                            plt.tight_layout()
                            # save the figure
                            savename = "{}_kappa_profile_{}.{}".format(label, ichi2, extension)
                            if path is None:
                                path = ""
                            if os.path.exists(os.path.join(path, ki)):
                                sig = name.split('.')[-1]
                                if sig == name:
                                    sig = 'orig'
                                mkdir_p(os.path.join(path, ki, 'kappa_profiles', sig))
                                savename = os.path.join(path, ki, "kappa_profiles", sig, savename)
                            elif os.path.exists(path):
                                sig = name.split('.')[-1]
                                if sig == name:
                                    sig = 'orig'
                                mkdir_p(os.path.join(path, "kappa_profiles", sig))
                                savename = os.path.join(path, sig, savename)
                            if verbose:
                                print('Saving '+savename)
                            plt.savefig(savename, dpi=200, transparent=True, bbox_inches='tight', pad_inches=0)
                            # plt.show()
                            plt.close()
                    # Roche scalar ordered arrival time surface
                    if 1:
                        textname = os.path.join(path, ki, "scalarRoche_{}.txt".format(name))
                        scalarRoche_sorted = np.int32(np.loadtxt(textname).T[1])
                        for i, idegarr in enumerate(scalarRoche_sorted[::-1]):
                            label = "scalarRoche_{:04d}".format(i)
                            m = recon_src.gls.models[idegarr]
                            # plt.figure(figsize=(5, 5))
                            kappa_profile_plot(m, correct_distances=True,
                                               kappa1_line=True, einstein_radius_indicator=True,
                                               label=ki, color=GLEAMcolors.blue, ls='-')
                            kappa_profile_plot(eagle_kappa_map, correct_distances=False,
                                               kappa1_line=False, einstein_radius_indicator=False,
                                               maprad=eagle_maprad, color=GLEAMcolors.red, ls='-')
                            plt.tight_layout()
                            # save the figure
                            savename = "{}_kappa_profile_{}.{}".format(label, idegarr, extension)
                            if path is None:
                                path = ""
                            if os.path.exists(os.path.join(path, ki)):
                                sig = name.split('.')[-1]
                                if sig == name:
                                    sig = 'orig'
                                mkdir_p(os.path.join(path, ki, 'kappa_profiles', sig))
                                savename = os.path.join(path, ki, "kappa_profiles", sig, savename)
                            elif os.path.exists(path):
                                sig = name.split('.')[-1]
                                if sig == name:
                                    sig = 'orig'
                                mkdir_p(os.path.join(path, "kappa_profiles", sig))
                                savename = os.path.join(path, sig, savename)
                            if verbose:
                                print('Saving '+savename)
                            plt.savefig(savename, dpi=200, transparent=True, bbox_inches='tight', pad_inches=0)
                            # plt.show()
                            plt.close()
