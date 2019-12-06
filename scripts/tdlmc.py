import matplotlib
# matplotlib.use('Agg')
import os
import sys
import fnmatch
import pickle
from pathos.multiprocessing import ProcessingPool as Pool
import numpy as np
root = os.path.dirname(os.path.dirname(os.path.abspath(os.path.realpath(__file__))))
sys.path.append(root)
import matplotlib.pyplot as plt
from astropy.io import fits
from gleam.reconsrc import ReconSrc, synth_filter, synth_filter_mp
from gleam.multilens import MultiLens
from gleam.utils.encode import an_sort
from gleam.utils.lensing import LensModel, downsample_model, find_einstein_radius
from gleam.utils.linalg import sigma_product
from gleam.utils.rgb_map import radial_mask
from gleam.glass_interface import glass_renv
from gleam.utils import plotting as gplt
from gleam.utils.colors import GLEAMcmaps as gcm
from gleam.utils.colors import GLEAMcolors as gcs
glass = glass_renv()


class KeyboardInterruptError(Exception):
    pass


def get_infofile(key, data_dir, ext='.txt'):
    """
    Args:
        key <str> - identifier key to distinguish from other filenames
        data_dir <str> - the directory in which to search for the info file

    Kwargs:
        ext <str> - file extension which the info filename is matched with

    Return:
        infofile <str> - the path to the info file
    """
    rung = key.split('_')[-1]
    for dirpath, dirnames, filenames in os.walk(data_dir):
        for filename in [f for f in filenames if f.endswith(ext) and rung in dirpath]:
            infofile = os.path.join(dirpath, filename)
    return infofile


def synth_debug(keys, jsons, states, mdl_index=-1, cy_opt=False,
                psf_files=None, use_psf=False, cmap=gcm.vilux,
                savefig=False, showfig=True, verbose=False):
    """
    Args:
        keys <list(str)> - the keys which correspond to jsons' and states' keys
        jsons <dict> - json dictionary holding a list of filenames for each key
        states <dict> - state dictionary holding a list of filenames for each key

    Kwargs:
        obj_index <int> - selected model
        cy_opt <bool> - use optimized cython method to construct inverse projection matrix
        psf_files <dict> - paths to the psf .fits files as a dictionary using <keys>
        use_psf <bool> - use the PSF in the reprojection
        verbose <bool> - verbose mode; print command line statements

    Return:
        filtered_states
    """
    if psf_files is None:
        use_psf = False
    for k in keys:
        json = jsons[k][0] if jsons[k] else None
        psf_file = psf_files[k]
        sigma2, shdr = fits.getdata(sgms[k], header=True)
        if json:
            with open(json) as f:
                ml = MultiLens.from_json(f)
        else:
            ml = MultiLens(ftsf[k])  # ftsf must be defined before execution!
        for sf in states[k]:
            sf_root = os.path.dirname(sf)
            pkl_name = os.path.join(sf_root, "reconsrc_{}.pkl".format(k))
            if os.path.exists(pkl_name):
                with open(pkl_name, 'rb') as f:
                    recon_src = pickle.load(f)
            else:
                recon_src = ReconSrc(ml, sf, M=ml.naxis1[0], M_fullres=2*ml.naxis1[0],
                                     mask_keys=['circle'], verbose=False)
                recon_src.calc_psf(psf_file, cy_opt=cy_opt)
            recon_src.chmdl(mdl_index)
            if verbose:
                print(k)
                print(recon_src.__v__)
            # calculate maps on source and lens plane
            recon_src.calc_psf(psf_file, cy_opt=cy_opt)
            recon_src.chmdl(mdl_index)
            img = recon_src.d_ij(flat=False)
            kw = dict(use_mask=1, sigma2=sigma2)  # , cached=1
            chi2 = recon_src.reproj_chi2(output_all=1, use_psf=use_psf, **kw)
            src = recon_src.plane_map(cached=1, **kw)
            src_ext = [-recon_src.r_max, recon_src.r_max]*2
            synth = recon_src.reproj_map(from_cache=1, cached=1, **kw)
            resids = recon_src.residual_map(from_cache=1, cached=1, **kw)
            print("Chi2 : {} / Total: {} ".format(chi2[1], chi2[0]))
            # look at the light mapping plots
            fig = plt.figure(figsize=(12, 8))
            plt.subplot(231)
            plt.title('data')
            plt.imshow(img, cmap=cmap, extent=ml[0].extent, vmin=0, vmax=img.max())
            gplt.plot_scalebar(max(ml[0].extent), length=2, origin='center')
            plt.axis('off')
            plt.subplot(232)
            plt.title('synthetic')
            plt.imshow(synth, cmap=cmap, extent=ml[0].extent, vmin=0, vmax=img.max())
            gplt.plot_scalebar(max(ml[0].extent), length=2, origin='center')
            plt.axis('off')
            plt.subplot(233)
            plt.title('residual')
            plt.imshow(resids, cmap=cmap, extent=ml[0].extent, vmin=0, vmax=img.max())
            gplt.plot_scalebar(max(ml[0].extent), length=2, origin='center')
            plt.axis('off')
            plt.subplot(234)
            plt.title('noise x 100')
            plt.imshow(sigma2*100, cmap=cmap, extent=ml[0].extent, vmin=0, vmax=img.max())
            gplt.plot_scalebar(max(ml[0].extent), length=2, origin='center')
            plt.axis('off')
            plt.subplot(235)
            plt.title('source plane')
            plt.imshow(src, cmap=cmap, extent=src_ext, vmin=0, vmax=img.max())
            gplt.plot_scalebar(max(src_ext), length=0.5, origin='center')
            plt.axis('off')
            plt.subplot(236)
            plt.text(0.1, 0.5, k, fontsize=20, transform=plt.gca().transAxes)
            plt.axis('off')
            gplt.square_subplots(fig)
            if savefig:
                savename = os.path.join(os.path.dirname(sf), "synth.maps_{}.pdf".format(k))
                plt.savefig(savename)
            if showfig:
                plt.show()
            plt.close(fig)


def synth_loop(keys, jsons, states, optimized=False, cy_opt=False,
               psf_files=None, use_psf=False, overwrite=False,
               verbose=False):
    """
    Args:
        keys <list(str)> - the keys which correspond to jsons' and states' keys
        jsons <dict> - json dictionary holding a list of filenames for each key
        states <dict> - state dictionary holding a list of filenames for each key

    Kwargs:
        optimized <bool> - run the multiprocessing synth filter
        cy_opt <bool> - use optimized cython method to construct inverse projection matrix
        psf_files <dict> - paths to the psf .fits files as a dictionary using <keys>
        use_psf <bool> - use the PSF in the reprojection
        verbose <bool> - verbose mode; print command line statements

    Return:
        filtered_states
    """
    if psf_files is None:
        use_psf = False
    synthf = synth_filter
    if optimized:
        synthf = synth_filter_mp
    for k in keys:
        json = jsons[k][0] if jsons[k] else None
        psf_file = psf_files[k]
        add_noise = 0
        sigma2, shdr = fits.getdata(sgms[k], header=True)
        if json:
            with open(json) as f:
                ml = MultiLens.from_json(f)
        else:
            ml = MultiLens(ftsf[k])  # ftsf must be defined before execution!
        for sf in states[k]:
            pkl_name = os.path.join(os.path.dirname(sf), "reconsrc_{}.pkl".format(k))
            if os.path.exists(pkl_name) and not overwrite:
                continue
            if verbose:
                print(k)
                print(sf)
            recon_src = ReconSrc(ml, sf, M=ml.naxis1[0], M_fullres=2*ml.naxis1[0],
                                 mask_keys=['circle'], verbose=verbose)
            recon_src.calc_psf(psf_file, cy_opt=cy_opt)
            recon_src = synthf(reconsrc=recon_src, percentiles=[], psf_file=psf_file,
                               noise=add_noise, sigma2=sigma2, use_psf=use_psf,
                               nonzero_only=0, use_mask=1,
                               return_obj=True, save=False, verbose=verbose)
            with open(pkl_name, 'wb') as f:
                pickle.dump(recon_src, f)


def load_chi2(keys, jsons, states, mdl_index=0, cy_opt=False,
              psf_files=None, use_psf=False, cmap=gcm.vilux,
              verbose=False):
    """
    Args:
        keys <list(str)> - the keys which correspond to jsons' and states' keys
        states <dict> - state dictionary holding a list of filenames for each key

    Kwargs:
        verbose <bool> - verbose mode; print command line statements

    Return:
        filtered_states
    """
    """
    Args:
        keys <list(str)> - the keys which correspond to jsons' and states' keys
        jsons <dict> - json dictionary holding a list of filenames for each key
        states <dict> - state dictionary holding a list of filenames for each key

    Kwargs:
        obj_index <int> - selected model
        cy_opt <bool> - use optimized cython method to construct inverse projection matrix
        psf_files <dict> - paths to the psf .fits files as a dictionary using <keys>
        use_psf <bool> - use the PSF in the reprojection
        verbose <bool> - verbose mode; print command line statements

    Return:
        filtered_states
    """
    if psf_files is None:
        use_psf = False
    chi2 = {}
    for k in keys:
        json = jsons[k][0] if jsons[k] else None
        psf_file = psf_files[k]
        sigma2, shdr = fits.getdata(sgms[k], header=True)
        if json:
            with open(json) as f:
                ml = MultiLens.from_json(f)
        else:
            ml = MultiLens(ftsf[k])  # ftsf must be defined before execution!
        for sf in states[k]:
            sf_root = os.path.dirname(sf)
            pkl_name = os.path.join(sf_root, "reconsrc_{}.pkl".format(k))
            if os.path.exists(pkl_name):
                with open(pkl_name, 'rb') as f:
                    recon_src = pickle.load(f)
            else:
                recon_src = ReconSrc(ml, sf, M=ml.naxis1[0], M_fullres=2*ml.naxis1[0],
                                     verbose=verbose)
                recon_src.calc_psf(psf_file, cy_opt=cy_opt)
            recon_src.chmdl(mdl_index)
            if verbose:
                print(k)
                print(recon_src.__v__)
            chi2[k] = []
            N_mdls = len(recon_src.gls.models)
            for i in range(N_mdls):
                recon_src.chmdl(i)
                c = recon_src.reproj_chi2(sigma2=sigma2, output_all=1,
                                          from_cache=1, cached=1,
                                          use_psf=use_psf, use_mask=1)
                chi2[k].append(c)
                if verbose:
                    msg = "\r{} / {}: {}".format(i+1, N_mdls, c)
                    sys.stdout.write(msg)
                    sys.stdout.flush()
            if verbose:
                print("")
    return chi2


def eval_from_header(filename, comment='#', exec_str='>>>',
                     matches=['N_models', 'N', 'L']):
    """
    Extract exe information from text file header

    Args:
        filename <str> - path of the text file

    Kwargs:
        comment <str> - comment sign which designates the header
        exec_str <str> - exec sequence which is interpreted if there is a match
        matches <list(str)> - match strings to search for

    Return:
        variables <list> - list of variables
    """
    with open(filename, 'rb') as f:
        for line in f.readlines():
            if not line.startswith(comment):
                break
            line = line.replace(comment, '').strip()
            if exec_str not in line:
                continue
            line = line.replace(exec_str, '').strip()
            if any([m in line for m in matches]):
                variables = eval(line.split('=')[-1])
                return variables


if __name__ == "__main__":
    ############################################################################
    # # SEARCH AND SETUP FILES
    # root directories
    home = os.path.expanduser("~")
    rdir = os.path.join(home, "tdlmc")
    jsondir = os.path.join(rdir, "json")
    statedir = os.path.join(rdir, "states")
    solsdir = os.path.join(rdir, "rung3_open_box/deflector_map")
    dtadir = os.path.join(rdir, "data")
    txtdir = os.path.join(rdir, "txt")
    ftsdir = os.path.join(rdir, "fits")
    rung = "rung3"
    # file root names
    keys = [
        "rung0_seed3", "rung0_seed4",
        "rung1_seed101", "rung1_seed102", "rung1_seed103", "rung1_seed104",
        "rung1_seed105", "rung1_seed107", "rung1_seed108", "rung1_seed109",
        "rung1_seed110", "rung1_seed111", "rung1_seed113", "rung1_seed114",
        "rung1_seed115", "rung1_seed116", "rung1_seed117", "rung1_seed118",
        "rung2_seed119", "rung2_seed120", "rung2_seed121", "rung2_seed122",
        "rung2_seed123", "rung2_seed124", "rung2_seed125", "rung2_seed126",
        "rung2_seed127", "rung2_seed128", "rung2_seed129", "rung2_seed130",
        "rung2_seed131", "rung2_seed132", "rung2_seed133", "rung2_seed134",
        "rung3_seed135", "rung3_seed136", "rung3_seed137", "rung3_seed138",
        "rung3_seed139", "rung3_seed140", "rung3_seed141", "rung3_seed142",
        "rung3_seed143", "rung3_seed144", "rung3_seed145", "rung3_seed146",
        "rung3_seed147", "rung3_seed148", "rung3_seed149", "rung3_seed150"]
    multi_keys = ["rung0",
                  "rung1_seeds1to4", "rung1_seeds5to9", "rung1_seeds10to14", "rung1_seeds15to18",
                  "rung2_A1quad", "rung2_A2quad", "rung2_Adouble", "rung2_Bquad",
                  "rung3Cdoubles", "rung3Cquads", "rung3D1quads", "rung3D2quads"]
    sol_keys = ['f160w-seed135', 'f160w-seed136', 'f160w-seed137', 'f160w-seed138',
                'f160w-seed139', 'f160w-seed140', 'f160w-seed141', 'f160w-seed142',
                'f160w-seed143', 'f160w-seed144', 'f160w-seed145', 'f160w-seed146',
                'f160w-seed147', 'f160w-seed148', 'f160w-seed149', 'f160w-seed150']
    infotext = "lens_info_for_Good_team.txt"
    # find TDLMC data files
    dtaftsf = {}
    psfs = {}
    sgms = {}
    for root, dirnames, filenames in os.walk(dtadir):
        for filename in fnmatch.filter(filenames, 'lens-image.fits'):
            for k in keys:
                r, s = k.split('_')
                if r in root and s in root:
                    break
            dtaftsf[k] = os.path.join(root, filename)
        for filename in fnmatch.filter(filenames, 'psf.fits'):
            for k in keys:
                r, s = k.split('_')
                if r in root and s in root:
                    break
            psfs[k] = os.path.join(root, filename)
        for filename in fnmatch.filter(filenames, 'noise_map.fits'):
            for k in keys:
                r, s = k.split('_')
                if r in root and s in root:
                    break
            sgms[k] = os.path.join(root, filename)
    # find state files
    ls_jsons = an_sort([os.path.join(jsondir, f) for f in os.listdir(jsondir)
                        if f.endswith('.json')])
    jsons = {k: [f for f in ls_jsons if k in f] for k in keys}
    ls_states = an_sort([os.path.join(statedir, f) for f in os.listdir(statedir)
                         if f.endswith('.state')])
    synthf10 = {k: [f for f in ls_states
                    if k in f and f.endswith('_synthf10.state')] for k in keys}
    synthf25 = {k: [f for f in ls_states
                    if k in f and f.endswith('_synthf25.state')] for k in keys}
    synthf50 = {k: [f for f in ls_states
                    if k in f and f.endswith('_synthf50.state')] for k in keys}
    ls_states = [f for f in ls_states if not (f.endswith('_synthf10.state')
                                              or f.endswith('_synthf25.state')
                                              or f.endswith('_synthf50.state'))]
    states = {k: [f for f in ls_states if k in f] for k in keys+multi_keys}
    # find exported fits files
    ls_fts = an_sort([os.path.join(ftsdir, f) for f in os.listdir(ftsdir)
                      if f.endswith('.fits')])
    kftsf = {k: [f for f in ls_fts if k in f and 'kappa' in f] for k in keys}
    pftsf = {k: [f for f in ls_fts if k in f and 'potential' in f] for k in keys}
    tftsf = {k: [f for f in ls_fts if k in f and 'arrival' in f] for k in keys}
    rftsf = {k: [f for f in ls_fts if k in f and 'roche' in f] for k in keys}
    ls_txts = an_sort([os.path.join(txtdir, f) for f in os.listdir(txtdir)
                       if f.endswith('.txt')])
    txts = {k: [f for f in ls_txts
                if k in f and f.endswith('_veldisp.txt')] for k in keys}
    # find solutions
    ls_sols = an_sort([
        os.path.join(root, filename) for root, dirnames, filenames in os.walk(solsdir)
        for filename in fnmatch.filter(filenames, 'phi*.fits')])
    sols = {k: [f for f in ls_sols
                if k in f and (f.endswith('phi11.fits') or f.endswith('phi22.fits'))]
            for k in sol_keys}
    # filter out selected rung
    if rung is not None:
        # TDLMC data
        keys = [k for k in keys if rung in k]
        multi_keys = [k for k in multi_keys if rung in k]
        dtaftsf = {k: dtaftsf[k] for k in keys}
        psfs = {k: psfs[k] for k in keys}
        sgms = {k: sgms[k] for k in keys}
        # states
        jsons = {k: jsons[k] for k in keys}
        synthf10 = {k: synthf10[k] for k in keys}
        synthf25 = {k: synthf25[k] for k in keys}
        synthf50 = {k: synthf50[k] for k in keys}
        states = {k: states[k] for k in keys+multi_keys}
        # exported formats
        kftsf = {k: kftsf[k] for k in keys}
        pftsf = {k: pftsf[k] for k in keys}
        tftsf = {k: tftsf[k] for k in keys}
        rftsf = {k: rftsf[k] for k in keys}
        txts = {k: txts[k] for k in keys}

    ############################################################################

    # # JSON UPDATING
    if 0:  # json updating with info text files
        for k in keys:
            # get info file
            info = get_infofile(k, dtadir, ext=infotext)
            # load json
            jfile = jsons[k][0]
            with open(jfile) as j:
                ml = MultiLens.from_json(j)
            # load info
            for m in ml:
                m.glscfactory.text = ml[0].glscfactory.read(info)
                m.glscfactory.sync_lens_params()
            # re-save json
            ml.jsonify(name=jfile, with_hash=False)

    # # LOOP OPERATIONS
    if 0:  # SYNTH CACHE loop
        kwargs = dict(optimized=1, psf_files=psfs, use_psf=1, cy_opt=0, verbose=1)
        synth_loop(keys, jsons, states, **kwargs)

    if 0:  # SRC MAPPING PLOT loop
        kwargs = dict(psf_files=psfs, use_psf=1, cy_opt=0, verbose=1)
        synth_debug(keys, jsons, states, mdl_index=-1, savefig=1, showfig=0, **kwargs)

    if 0:  # BEST 10 SYNTH MAPS loop
        kwargs = dict(psf_files=psfs, use_psf=1, cy_opt=0, verbose=1)
        N = 25
        chi2s = load_chi2(keys, jsons, states, mdl_index=0, **kwargs)
        for k in keys:
            chi2 = [c[1] for c in chi2s[k]]
            chi2_idcs = np.argsort(np.abs(np.array(chi2)-1))

            json = jsons[k][0] if jsons[k] else None
            psf_file = psfs[k]
            sigma2, shdr = fits.getdata(sgms[k], header=True)
            if json:
                with open(json) as f:
                    ml = MultiLens.from_json(f)
            for sf in states[k]:
                sf_root = os.path.dirname(sf)
                pkl_name = os.path.join(sf_root, "reconsrc_{}.pkl".format(k))
                if os.path.exists(pkl_name):
                    with open(pkl_name, 'rb') as f:
                        recon_src = pickle.load(f)
                else:
                    recon_src = ReconSrc(ml, sf, M=ml.naxis1[0], M_fullres=2*ml.naxis1[0],
                                         verbose=kwargs.get('verbose', 0))
                recon_src.calc_psf(psf_file, cy_opt=kwargs.get('cy_opt', 0))
                if kwargs.get('verbose', 0):
                    print(k)
                    print(recon_src.__v__)
                img = recon_src.d_ij(flat=False)
                for i, mi in enumerate(chi2_idcs[:10]):
                    recon_src.chmdl(mi)
                    synth = recon_src.reproj_map(from_cache=1, cached=1, use_mask=1, **kwargs)
                    plt.imshow(synth, cmap=gcm.vilux, extent=ml[0].extent, vmin=0, vmax=img.max())
                    gplt.plot_scalebar(max(ml[0].extent), length=2, origin='center',
                                       padding=(0.55, 0.40))
                    # src = recon_src.plane_map(use_psf=1, cached=1, sigma2=sigma2, use_mask=1)
                    # src_ext = [-recon_src.r_max, recon_src.r_max]*2
                    # plt.imshow(src, cmap=gcm.vilux, extent=src_ext, vmin=0, vmax=img.max())
                    # gplt.plot_scalebar(max(src_ext), length=0.25, origin='center',
                    #                    padding=(0.25, 0.2))
                    plt.axis('off')
                    plt.tight_layout()
                    savename = os.path.join(
                        os.path.dirname(sf), "synth.map.{}_{}.pdf".format(i, k))
                    print(savename)

                    # plt.axis('off')
                    # plt.tight_layout()
                    # savename = os.path.join(
                    #     os.path.dirname(sf), "synth.src.{}_{}.pdf".format(i, k))
                    # print(savename)

                    plt.savefig(savename, transparent=True)
                    # plt.show()
                    plt.close()

    if 0:  # CHI2 HIST loop
        kwargs = dict(psf_files=psfs, use_psf=1, cy_opt=0, verbose=1)
        N = 25
        chi2 = load_chi2(keys, jsons, states, mdl_index=0, **kwargs)
        for k in keys:
            savename = os.path.join(statedir, "synth.chi2_{}.pdf".format(k))
            # plot reduced chi2
            chi2red = [c[1] for c in chi2[k]]
            y, bins, patches = plt.hist(chi2red, N, rwidth=0.7)
            # calculate non-reduced chi2 for coloring
            cmap = gcm.cozytime
            chi2tot = [c[0] for c in chi2[k]]
            Y, X = np.histogram(chi2tot, N, density=1)
            col = Y*(X.max()-X.min())/N
            col = (col-col.min()) / (col.max()-col.min())
            for c, p in zip(col, patches):
                plt.setp(p, 'facecolor', cmap(c))
            plt.title(k)
            plt.xlabel(r'$\chi^{2}$')
            plt.ylabel(r'$\mathrm{\mathsf{N_{models}}}$')
            plt.tight_layout()
            plt.savefig(savename)
            plt.close()
            # plt.show()

    if 0:  # PLOT LENS DATA individually
        for k in keys:
            print(k)
            hdu = fits.open(dtaftsf[k])
            dta = hdu[0].data
            R = (dta.shape[0]//2)*0.08
            extent = [-R, R]*2
            plt.imshow(dta, cmap=gcm.vilux, extent=extent, vmin=0, vmax=dta.max())
            gplt.plot_scalebar(R, length=2, origin='center', padding=(0.55, 0.40))
            plt.axis('off')
            # plt.show()
            plt.savefig("lens.data_{}.pdf".format(k), transparent=True)
            plt.close()

    if 0:  # PLOT VIEWSTATE from states loop
        for k in keys:
            sf = states[k][0]
            lm = LensModel(sf)
            print(lm)
            gplt.viewstate_plots(lm, refined=False, title=True, showfig=False, savefig=True)
        for k in multi_keys:
            sf = states[k][0]
            lm = LensModel(sf)
            print(lm)
            gplt.viewstate_plots(lm, refined=False, title=True, showfig=False, savefig=True)

    if 0:  # PLOT VIEWSTATE from solutions loop
        pixel_scale = 0.13/4/4
        for k in [key for key in keys if 'rung3' in key]:
            print(k)
            lm = LensModel(states[k][0])
            ksol = [key for key in sol_keys if lm.obj_name.split('_')[-1] in key][0]
            # load solution
            fsol = sols[ksol]
            hdus = fits.open(fsol[0]), fits.open(fsol[1])
            data = hdus[0][0].data, hdus[1][0].data
            hdrs = hdus[0][0].header, hdus[1][0].header
            kappa_fr = 0.5 * (data[0] + data[1])
            # params
            pars_sol = dict(
                pixrad=kappa_fr.shape[0]//2,
                maprad=kappa_fr.shape[0]//2 * pixel_scale)
            # downsample solution
            kappa = downsample_model(kappa_fr, lm.extent, lm.kappa_grid(refined=True).shape,
                                     pixel_scale=pixel_scale)
            lm_sol = LensModel(np.flipud(kappa), filename=ksol+'.sol',
                               maprad=lm.maprad, pixrad=lm.pixrad)
            gplt.viewstate_plots(lm_sol, showfig=False, savefig=False)
            savename = "sol.viewstate_{}.pdf".format(k)
            print(savename)
            plt.savefig(savename)
            # plt.show()
            plt.close()

    if 0:  # BEST 10 KAPPA MAPS loop
        kwargs = dict(psf_files=psfs, use_psf=1, cy_opt=0, verbose=1)
        chi2s = load_chi2(keys, jsons, states, mdl_index=0, **kwargs)
        for k in keys:
            chi2 = [c[1] for c in chi2s[k]]
            chi2_idcs = np.argsort(np.abs(np.array(chi2)-1))
            lm = LensModel(states[k][0])
            for i, mi in enumerate(chi2_idcs[:10]):
                gplt.kappa_map_plot(lm, mdl_index=mi, contours=1, subcells=3)
                gplt.plot_scalebar(lm.maprad, length=1, origin='center', padding=(0.45, 0.30))
                plt.axis('off')
                plt.savefig("kappa.map.{}_{}.pdf".format(i, k), transparent=True)
                # plt.show()
                plt.close()

    if 0:  # PLOT SINGLE KAPPA MAPS from solutions loop
        pixel_scale = 0.13/4/4
        for k in [key for key in keys if 'rung3' in key]:
            print(k)
            lm = LensModel(states[k][0])
            ksol = [key for key in sol_keys if lm.obj_name.split('_')[-1] in key][0]
            # load solution
            fsol = sols[ksol]
            hdus = fits.open(fsol[0]), fits.open(fsol[1])
            data = hdus[0][0].data, hdus[1][0].data
            hdrs = hdus[0][0].header, hdus[1][0].header
            kappa_fullres = 0.5 * (data[0] + data[1])
            # params
            pars_sol = dict(
                pixrad=kappa_fullres.shape[0]//2,
                maprad=kappa_fullres.shape[0]//2 * pixel_scale)
            # downsample solution
            kappa = downsample_model(kappa_fullres, lm.extent, lm.kappa_grid(refined=True).shape,
                                     pixel_scale=pixel_scale)

            lm_sol = LensModel(np.flipud(kappa), filename=ksol+'.sol',
                               maprad=lm.maprad, pixrad=lm.pixrad)

            plt.figure(figsize=(12, 7))
            plt.subplot(121)
            gplt.kappa_map_plot(lm_sol, contours=1)
            plt.subplot(122)
            gplt.kappa_map_plot(lm, contours=1, subcells=3)
            plt.suptitle(k)
            plt.tight_layout()
            savename = "compare.kappa_{}.pdf".format(k)
            plt.savefig(savename)
            # plt.show()
            plt.close()

    if 0:  # PLOT MULTI KAPPA MAPS from solutions loop
        pixel_scale = 0.13/4/4
        for k in multi_keys:
            print(k)
            lm = LensModel(states[k][0])
            for obji in range(lm.N_obj):
                lm.obj_idx = obji
                ksol = [key for key in sol_keys if lm.obj_name.split('_')[-1] in key][0]
                # load solution
                fsol = sols[ksol]
                hdus = fits.open(fsol[0]), fits.open(fsol[1])
                data = hdus[0][0].data, hdus[1][0].data
                hdrs = hdus[0][0].header, hdus[1][0].header
                kappa_fr = 0.5 * (data[0] + data[1])
                # params
                pars_sol = dict(
                    pixrad=kappa_fr.shape[0]//2, maprad=kappa_fr.shape[0]//2 * pixel_scale)
                # downsample solution
                kappa = downsample_model(kappa_fr, lm.extent, lm.kappa_grid(refined=True).shape,
                                         pixel_scale=pixel_scale)

                lm_sol = LensModel(np.flip(kappa, 0), filename=ksol+'.sol',
                                   maprad=lm.maprad, pixrad=lm.pixrad)

                plt.figure(figsize=(12, 7))
                plt.subplot(121)
                gplt.kappa_map_plot(lm_sol, contours=1)
                plt.subplot(122)
                gplt.kappa_map_plot(lm, obj_index=obji, contours=1, subcells=3)
                plt.suptitle(k)
                plt.tight_layout()
                savename = "compare.kappa_{}.{}.pdf".format(k, lm.obj_name)
                print(savename)
                plt.savefig(savename)
                # plt.show()
                plt.close()

    if 0:  # PLOT SINGLE KAPPA PROFILES from solutions loop
        pixel_scale = 0.13/4/4
        for k in [key for key in keys if 'rung3' in key]:
            print(k)
            lm = LensModel(states[k][0])
            ksol = [key for key in sol_keys if lm.obj_name.split('_')[-1] in key][0]
            # load solution
            fsol = sols[ksol]
            hdus = fits.open(fsol[0]), fits.open(fsol[1])
            data = hdus[0][0].data, hdus[1][0].data
            hdrs = hdus[0][0].header, hdus[1][0].header
            kappa_fr = 0.5 * (data[0] + data[1])
            # params
            pars_sol = dict(
                pixrad=kappa_fr.shape[0]//2,
                maprad=kappa_fr.shape[0]//2 * pixel_scale)
            # downsample solution
            kappa = downsample_model(kappa_fr, lm.extent, lm.kappa_grid(refined=True).shape,
                                     pixel_scale=pixel_scale)
            lm_sol = LensModel(np.flipud(kappa), filename=ksol+'.sol',
                               maprad=lm.maprad, pixrad=lm.pixrad)
            with plt.style.context('dark_background'):
                plt.figure(figsize=(8, 8))
                solplts, _, _ = gplt.kappa_profile_plot(lm_sol, color=gcs.red, lw=2, r_shift=0.08)
                plots, _, _ = gplt.kappa_profiles_plot(lm, obj_index=0, refined=0,
                                                       interpolate=500, as_range=1,
                                                       hilite_color=gcs.blue,
                                                       adjust_limits=1,
                                                       lw=2, label_axes=1)
                # plt.title(k)
                plt.legend((solplts, plots[-1]), ('truth', 'avg. model'))
                plt.tight_layout()
            savename = "compare.profiles_{}.pdf".format(k)
            print(savename)
            plt.savefig(savename, transparent=True)
            # plt.show()
            plt.close()

    if 0:  # PLOT MULTI KAPPA PROFILES from solutions loop
        pixel_scale = 0.13/4/4
        for k in multi_keys:
            print(k)
            lm = LensModel(states[k][0])
            for obji in range(lm.N_obj):
                lm.obj_idx = obji
                ksol = [key for key in sol_keys if lm.obj_name.split('_')[-1] in key][0]
                # load solution
                fsol = sols[ksol]
                hdus = fits.open(fsol[0]), fits.open(fsol[1])
                data = hdus[0][0].data, hdus[1][0].data
                hdrs = hdus[0][0].header, hdus[1][0].header
                kappa_fr = 0.5 * (data[0] + data[1])
                # params
                pars_sol = dict(
                    pixrad=kappa_fr.shape[0]//2, maprad=kappa_fr.shape[0]//2 * pixel_scale)
                # downsample solution
                kappa = downsample_model(kappa_fr, lm.extent, lm.kappa_grid(refined=True).shape,
                                         pixel_scale=pixel_scale)
                lm_sol = LensModel(np.flip(kappa, 0), filename=ksol+'.sol',
                                   maprad=lm.maprad, pixrad=lm.pixrad)
                plt.figure(figsize=(8, 8))
                solplts, solradii, solprof = gplt.kappa_profile_plot(lm_sol, color=gcs.red, lw=2,
                                                                     r_shift=0.08)
                plots, profs, radii = gplt.kappa_profiles_plot(lm, obj_index=obji, refined=0,
                                                               interpolate=500, as_range=1,
                                                               hilite_color=gcs.blue,
                                                               adjust_limits=1,
                                                               lw=2, label_axes=1)
                plt.title("{}.{}".format(k, lm.obj_name))
                plt.legend((solplts, plots[-1]), ('truth', 'avg. model'))
                plt.tight_layout()
                savename = "compare.profiles_{}.{}.pdf".format(k, lm.obj_name)
                print(savename)
                plt.savefig(savename)
                # plt.show()
                plt.close()

    if 0:  # SAVE EINSTEIN RADII from SINGLE in txt solutions loop
        verbose = True
        pixel_scale = 0.13/4/4
        for k in keys:
            print(k)
            lm = LensModel(states[k][0])
            ksol = [key for key in sol_keys if lm.obj_name.split('_')[-1] in key][0]
            # load solution
            fsol = sols[ksol]
            hdus = fits.open(fsol[0]), fits.open(fsol[1])
            data = hdus[0][0].data, hdus[1][0].data
            hdrs = hdus[0][0].header, hdus[1][0].header
            kappa_fr = 0.5 * (data[0] + data[1])
            # params
            pars_sol = dict(
                pixrad=kappa_fr.shape[0]//2, maprad=kappa_fr.shape[0]//2 * pixel_scale)
            # downsample solution
            kappa = downsample_model(kappa_fr, lm.extent, lm.kappa_grid(refined=True).shape,
                                     pixel_scale=pixel_scale)
            lm_sol = LensModel(np.flip(kappa, 0), filename=ksol+'.sol',
                               maprad=lm.maprad, pixrad=lm.pixrad)
            solplts, solradii, solprof = gplt.kappa_profile_plot(lm_sol, color=gcs.red, lw=2,
                                                                 r_shift=0.08)
            plots, profs, radii = gplt.kappa_profiles_plot(lm, obj_index=0, refined=0,
                                                           interpolate=500, as_range=1,
                                                           hilite_color=gcs.blue,
                                                           adjust_limits=1,
                                                           lw=2)
            re_sol = find_einstein_radius(solradii, solprof)
            re_mdl = [find_einstein_radius(r, p) for r, p in zip(radii, profs)]
            re_mdl.append(re_sol)
            re_mdl = np.asarray(re_mdl)
            if verbose:
                print("R_E (model): {} | min: {}; max: {}".format(
                    re_mdl[-2], min(re_mdl[:-2]), max(re_mdl[:-2])))
                print("R_E (solutions): {}".format(re_mdl[-1]))
            savename = "compare.RE_{}.txt".format(k)
            header = "\n".join(
                ("{}".format(savename),
                 "Notional Einstein radii (in arcsec) "
                 + "of all {} lens models, ".format(len(re_mdl)-2)
                 + "ensemble average model, and truth",
                 "Usage:",
                 ">>> import numpy as np",
                 ">>> data = np.loadtxt('{}')".format(savename),
                 ">>> R_E_models = data[:-2]",
                 ">>> R_E_ensavg = data[-2]",
                 ">>> R_E_truth  = data[-1]"))
            np.savetxt(savename, re_mdl, header=header)

    if 0:  # SAVE EINSTEIN RADII from MULTI in txt solutions loop
        verbose = True
        pixel_scale = 0.13/4/4
        for k in multi_keys:
            print(k)
            lm = LensModel(states[k][0])
            for obji in range(lm.N_obj):
                lm.obj_idx = obji
                ksol = [key for key in sol_keys if lm.obj_name.split('_')[-1] in key][0]
                # load solution
                fsol = sols[ksol]
                hdus = fits.open(fsol[0]), fits.open(fsol[1])
                data = hdus[0][0].data, hdus[1][0].data
                hdrs = hdus[0][0].header, hdus[1][0].header
                kappa_fr = 0.5 * (data[0] + data[1])
                # params
                pars_sol = dict(
                    pixrad=kappa_fr.shape[0]//2, maprad=kappa_fr.shape[0]//2 * pixel_scale)
                # downsample solution
                kappa = downsample_model(kappa_fr, lm.extent, lm.kappa_grid(refined=True).shape,
                                         pixel_scale=pixel_scale)
                lm_sol = LensModel(np.flip(kappa, 0), filename=ksol+'.sol',
                                   maprad=lm.maprad, pixrad=lm.pixrad)
                solplts, solradii, solprof = gplt.kappa_profile_plot(lm_sol, color=gcs.red, lw=2,
                                                                     r_shift=0.08)
                plots, profs, radii = gplt.kappa_profiles_plot(lm, obj_index=obji, refined=0,
                                                               interpolate=500, as_range=1,
                                                               hilite_color=gcs.blue,
                                                               adjust_limits=1,
                                                               lw=2)
                re_sol = find_einstein_radius(solradii, solprof)
                re_mdl = [find_einstein_radius(r, p) for r, p in zip(radii, profs)]
                re_mdl.append(re_sol)
                re_mdl = np.asarray(re_mdl)
                if verbose:
                    print("R_E (model): {} | min: {}; max: {}".format(
                        re_mdl[-2], min(re_mdl[:-2]), max(re_mdl[:-2])))
                    print("R_E (solutions): {}".format(re_mdl[-1]))
                savename = "compare.RE_{}.{}.txt".format(k, lm.obj_name)
                header = "\n".join(
                    ("{}".format(savename),
                     "Notional Einstein radii (in arcsec) "
                     + "of all {} lens models, ".format(len(re_mdl)-2)
                     + "ensemble average model, and truth",
                     "Usage:",
                     ">>> import numpy as np",
                     ">>> data = np.loadtxt('{}')".format(savename),
                     ">>> R_E_models = data[:-2]",
                     ">>> R_E_ensavg = data[-2]",
                     ">>> R_E_truth  = data[-1]"))
                np.savetxt(savename, re_mdl, header=header)

    if 0:  # LOAD and HIST EINSTEIN RADII txt files solution loop
        verbose = True
        for txt_re in [
                'compare.RE_rung3_seed135.txt', 'compare.RE_rung3_seed136.txt',
                'compare.RE_rung3_seed137.txt', 'compare.RE_rung3_seed138.txt',
                'compare.RE_rung3_seed139.txt', 'compare.RE_rung3_seed140.txt',
                'compare.RE_rung3_seed141.txt', 'compare.RE_rung3_seed142.txt',
                'compare.RE_rung3_seed143.txt', 'compare.RE_rung3_seed144.txt',
                'compare.RE_rung3_seed145.txt', 'compare.RE_rung3_seed146.txt',
                'compare.RE_rung3_seed147.txt', 'compare.RE_rung3_seed148.txt',
                'compare.RE_rung3_seed149.txt', 'compare.RE_rung3_seed150.txt',
                'compare.RE_rung3Cdoubles.rung3_seed138.txt',
                'compare.RE_rung3Cdoubles.rung3_seed142.txt',
                'compare.RE_rung3Cdoubles.rung3_seed146.txt',
                'compare.RE_rung3Cdoubles.rung3_seed150.txt',
                'compare.RE_rung3Cquads.rung3_seed137.txt',
                'compare.RE_rung3Cquads.rung3_seed139.txt',
                'compare.RE_rung3Cquads.rung3_seed141.txt',
                'compare.RE_rung3Cquads.rung3_seed149.txt',
                'compare.RE_rung3D1quads.rung3_seed135.txt',
                'compare.RE_rung3D1quads.rung3_seed136.txt',
                'compare.RE_rung3D1quads.rung3_seed140.txt',
                'compare.RE_rung3D1quads.rung3_seed143.txt',
                'compare.RE_rung3D2quads.rung3_seed144.txt',
                'compare.RE_rung3D2quads.rung3_seed145.txt',
                'compare.RE_rung3D2quads.rung3_seed147.txt',
                'compare.RE_rung3D2quads.rung3_seed148.txt']:
            if os.path.exists(txt_re):
                N = 20
                data = np.loadtxt(txt_re)
                R_E_models = data[:-2]
                R_E_ensavg = data[-2]
                R_E_truth = data[-1]
                mainc = gcm.agaveglitch(0.3)
                eavgc = gcs.blue_marguerite
                truec = gcm.agaveglitch(0.8)
                y, bins, patches = plt.hist(R_E_models, N, rwidth=0.5, color=mainc)
                eavgplt = plt.axvline(R_E_ensavg, color=eavgc, lw=3)
                trueplt = plt.axvline(R_E_truth, color=truec, lw=3)
                plt.title(txt_re.replace('compare.RE_', '').replace('.txt', ''))
                plt.xlabel(r'R$_{\mathrm{\mathsf{E}}}$ [arcsec]')
                plt.ylabel(r'$\mathrm{\mathsf{N_{models}}}$')
                plt.legend((trueplt, eavgplt), ('truth', 'avg. model'),
                           handlelength=1, fontsize=13)
                plt.tight_layout()
                # with plt.style.context('dark_background'):
                #     plt.rcParams['axes.facecolor'] = '#1A2E54'
                #     plt.rcParams['savefig.facecolor'] = '#1A2E54'
                #     plt.rcParams['savefig.edgecolor'] = '#1A2E54'
                #     y, bins, patches = plt.hist(R_E_models, N, rwidth=0.5, color=mainc)
                #     eavgplt = plt.axvline(R_E_ensavg, color=eavgc, lw=3)
                #     trueplt = plt.axvline(R_E_truth, color=truec, lw=3)
                #     # plt.title(txt_re.replace('compare.RE_', '').replace('.txt', ''))
                #     plt.xlabel(r'R$_{\mathrm{\mathsf{E}}}$ [arcsec]')
                #     plt.ylabel(r'$\mathrm{\mathsf{N_{models}}}$')
                #     plt.legend((trueplt, eavgplt), ('truth', 'avg. model'),
                #                handlelength=1, fontsize=13)
                #     plt.tight_layout()
                savename = txt_re.replace('.txt', '.pdf')
                print(savename)
                plt.savefig(savename)
                # plt.savefig(savename)
                # plt.show()
                plt.close()

    if 0:  # ARRIVAL CONTOUR PLOTS 
        # kwargs = dict(psf_files=psfs, use_psf=1, cy_opt=0, verbose=1)
        # chi2s = load_chi2(keys, jsons, states, mdl_index=0, **kwargs)
        for k in keys:
            # chi2 = [c[1] for c in chi2s[k]]
            # chi2_idcs = np.argsort(np.abs(np.array(chi2)-1))
            print(k)
            lm = LensModel(states[k][0])
            gplt.arrival_time_surface_plot(lm, min_contour_shift=0.1,
                                           cmap=gcm.agaveglitch,
                                           scalebar=1)
            plt.savefig("arrival.surface_{}.pdf".format(k), transparent=True)
            # plt.show()
            plt.close()

    if 0:  # ROCHE CONTOUR PLOTS 
        # kwargs = dict(psf_files=psfs, use_psf=1, cy_opt=0, verbose=1)
        # chi2s = load_chi2(keys, jsons, states, mdl_index=0, **kwargs)
        for k in keys:
            # chi2 = [c[1] for c in chi2s[k]]
            # chi2_idcs = np.argsort(np.abs(np.array(chi2)-1))
            print(k)
            lm = LensModel(states[k][0])
            def max_val(grid):
                msk = radial_mask(grid, radius=int(0.75*grid.shape[0]*0.5))
                return np.max(grid[msk])
            gplt.roche_potential_plot(lm, log=1, zero_level='center', cmax=max_val,
                                      background=None, levels=30,
                                      contours_only=0,
                                      cmap=gcm.agaveglitch,
                                      scalebar=1, color='black')
            plt.savefig("roche.potential_{}.pdf".format(k), transparent=True)
            # plt.show()
            plt.close()

    if 0:  # ROCHE POTENTIAL SINGLE ANALYSIS loop
        pixel_scale = 0.13/4/4
        for k in [key for key in keys if 'rung3' in key][5:]:
            print(k)
            lm = LensModel(states[k][0])
            ksol = [key for key in sol_keys if lm.obj_name.split('_')[-1] in key][0]
            # load solution
            fsol = sols[ksol]
            hdus = fits.open(fsol[0]), fits.open(fsol[1])
            data = hdus[0][0].data, hdus[1][0].data
            hdrs = hdus[0][0].header, hdus[1][0].header
            kappa_fr = 0.5 * (data[0] + data[1])
            # params
            pars_sol = dict(
                pixrad=kappa_fr.shape[0]//2,
                maprad=kappa_fr.shape[0]//2 * pixel_scale)
            # downsample solution
            kappa = downsample_model(kappa_fr, lm.extent, lm.kappa_grid(refined=True).shape,
                                     pixel_scale=pixel_scale)
            lm_sol = LensModel(np.flipud(kappa), filename=ksol+'.sol',
                               maprad=lm.maprad, pixrad=lm.pixrad)

            rsol = lm_sol.roche_potential_grid(model_index=0, N=None)

            def roche_calc(mi):
                try:
                    roche_pot = lm.roche_potential_grid(model_index=mi)
                except KeyboardInterrupt:
                    raise KeyboardInterruptError()
                msg = "\r{} / {}".format(mi+1, lm.N)
                sys.stdout.write('\r')
                sys.stdout.write(msg)
                sys.stdout.flush()
                return roche_pot[:]

            pool = Pool(processes=2)
            r_mdls = pool.map(roche_calc, range(lm.N))

            savename = "roche.maps_{}.pkl".format(k)
            with open(savename, 'wb') as f:
                pickle.dump(r_mdls, f)
            print(savename)

    if 1:
        pixel_scale = 0.13/4/4
        for k in [key for key in keys if 'rung3' in key]:
            print(k)
            pkl_file = "roche.maps_{}.pkl".format(k)
            if os.path.exists(pkl_file):
                with open(pkl_file, 'rb') as f:
                    r_mdls = pickle.load(f)
            else:
                continue

            lm = LensModel(states[k][0])
            ksol = [key for key in sol_keys if lm.obj_name.split('_')[-1] in key][0]
            # load solution
            fsol = sols[ksol]
            hdus = fits.open(fsol[0]), fits.open(fsol[1])
            data = hdus[0][0].data, hdus[1][0].data
            hdrs = hdus[0][0].header, hdus[1][0].header
            kappa_fr = 0.5 * (data[0] + data[1])
            # params
            pars_sol = dict(
                pixrad=kappa_fr.shape[0]//2,
                maprad=kappa_fr.shape[0]//2 * pixel_scale)
            # downsample solution
            kappa = downsample_model(kappa_fr, lm.extent, lm.kappa_grid(refined=True).shape,
                                     pixel_scale=pixel_scale)
            lm_sol = LensModel(np.flipud(kappa), filename=ksol+'.sol',
                               maprad=lm.maprad, pixrad=lm.pixrad)

            rsol = lm_sol.roche_potential_grid(model_index=0, N=None)

            roche_scalars = [None]*lm.N
            for midx in range(len(r_mdls)):
                roche_scalars[midx] = sigma_product(rsol, r_mdls[midx])

            Y, X, patches = plt.hist(roche_scalars, 30, rwidth=0.7)
            plt.ylabel(r'$\mathrm{\mathsf{N_{models}}}$')
            plt.xlabel(r'$\langle\mathcal{P}, \mathcal{P}_{\mathsf{model}}\rangle$')
            cscheme = (X - X.min()) / (X.max()-X.min())
            for x, p in zip(cscheme, patches):
                plt.setp(p, 'facecolor', gcm.agaveglitch(x))
            plt.tight_layout()
            savename = "compare.roche_{}.pdf".format(k)
            print(savename)
            plt.savefig(savename)
            # plt.show()
            plt.close()

    if 0:  # PLOT NICE H0 HIST loop
        obj_index = 0
        true_H0 = 65.413
        for k in keys+multi_keys:
            print(k)
            lm = LensModel(states[k][0])
            values = [[], [], []]
            for m in lm.env.models:
                obj, data = m['obj,data'][obj_index]
                if data.has_key('H0'):
                    values[m.get('accepted', 2)].append(data['H0'])
            not_acc, acc, notag = values
            H0s = np.asarray(acc)
            # with plt.style.context('dark_background'):
            Y, X, patches = plt.hist(H0s, 50, rwidth=1)
            diffs = (1. / np.abs(X - true_H0)**(0.001))
            diffs = (diffs - diffs.min()) / (diffs.max() - diffs.min())
            for d, p in zip(diffs, patches):
                plt.setp(p, 'facecolor', gcm.agaveglitch(d))
            plt.axvline(true_H0, lw=3, color=gcs.red, label='truth')
            plt.xlabel(r'$\mathrm{\mathsf{H_{0}}}$')
            plt.ylabel(r'$\mathrm{\mathsf{N_{models}}}$')
            plt.legend(handlelength=1)
            plt.tight_layout()
            plt.savefig("H0.hist_{}.pdf".format(k), transparent=True)
            plt.close()
