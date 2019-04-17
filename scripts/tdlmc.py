import os
import sys
root = os.path.dirname(os.path.dirname(os.path.abspath(os.path.realpath(__file__))))
sys.path.append(root)
import numpy as np
from functools import partial
from pathos.multiprocessing import ProcessingPool as Pool
import matplotlib.pyplot as plt

from gleam.reconsrc import ReconSrc, synth_filter, synth_filter_mp
from gleam.multilens import MultiLens
from gleam.utils.encode import an_sort
from gleam.utils.linalg import sigma_product
from gleam.glass_interface import glass_renv
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


def synth_loop(keys, jsons, states, save=False, optimized=False, verbose=False):
    """
    Args:
        keys <list(str)> - the keys which correspond to jsons' and states' keys
        jsons <dict> - json dictionary holding a list of filenames for each key
        states <dict> - state dictionary holding a list of filenames for each key

    Kwargs:
        save <bool> - save the filtered states automatically
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
        for sf in states[k]:
            synths, _, _ = synthf(sf, ml, percentiles=[10, 25, 50], save=save, verbose=verbose)
            filtered_states[sf] = synths
    return filtered_states


if __name__ == "__main__":
    # root directories
    home = os.path.expanduser("~")
    rdir = home+"/tdlmc"
    jsondir = rdir+"/json/"
    statedir = rdir+"/states/"

    dtadir = rdir+"/data/"
    infoext = "lens_info_for_Good_team.txt"

    keys = ["rung2_seed119", "rung2_seed120", "rung2_seed121", "rung2_seed122",
            "rung2_seed123", "rung2_seed124", "rung2_seed125", "rung2_seed126",
            "rung2_seed127", "rung2_seed128", "rung2_seed129", "rung2_seed130",
            "rung2_seed131", "rung2_seed132", "rung2_seed133", "rung2_seed134"]
    # list all files within the directories
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
    states = {k: [f for f in ls_states if k in f] for k in keys}

    if 1:  # synth filtering
        kwargs = dict(save=True, optimized=True, verbose=True)
        synth_filtered_states = synth_loop(keys, jsons, states, **kwargs)

    if 0:  # json updating with info text files
        for k in keys:
            # get info file
            info = get_infofile(k, dtadir, ext=infoext)
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

    if 0:  # some plots for single seed
        k = "rung2_seed119"
        f = states[k][0]

        info = get_infofile(k, dtadir, ext=infoext)

        with open(jsons[k][0]) as j:
            ml = MultiLens.from_json(j)
        recon_src = ReconSrc(ml, f, M=20, verbose=True)

        dta = recon_src.d_ij(flat=False)
        kw = dict(vmin=dta.min(), vmax=dta.max())
        plt.imshow(dta, **kw)
        plt.colorbar()
        plt.axis('off')
        plt.show()

        plt.imshow(recon_src.plane_map(), **kw)
        plt.colorbar()
        plt.axis('off')
        plt.show()

        plt.imshow(recon_src.reproj_map(), **kw)
        plt.colorbar()
        plt.axis('off')
        plt.show()

        plt.imshow(recon_src.residual_map())
        plt.colorbar()
        plt.axis('off')
        plt.show()
