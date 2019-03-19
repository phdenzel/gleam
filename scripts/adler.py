import os
import sys
root = os.path.dirname(os.path.dirname(os.path.abspath(os.path.realpath(__file__))))
sys.path.append(root)
import re
import numpy as np
from scipy import interpolate
from astropy.io import fits
import matplotlib.pyplot as plt

from gleam.multilens import MultiLens
from gleam.reconsrc import ReconSrc
from gleam.glass_interface import glass_renv, filter_env, export_state
glass = glass_renv()


def an_sorted(data):
    """
    Perform an alpha-numeric, natural sort

    Args:
        data <list> - list of strings

    Kwargs:
        None

    Return:
        sorted <list> - the alpha-numerically, naturally sorted list of strings
    """
    def convert(text):
        return int(text) if text.isdigit() else text.lower()

    def an_key(key):
        return [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(data, key=an_key)


def synth_filter(statefile, gleamobject, percentiles=[10, 25, 50], verbose=False):
    """
    Filter a GLASS state file using GLEAM's source reconstruction feature

    Args:
        statefile <str> - filename of a GLASS state
        gleamobject <GLEAM object> - a GLEAM object instance with .fits file's data

    Kwargs:
        percentiles <list(float)> - percentages the filter retains

    Return:
        filtered_states <list(glass.Environment object)> 
    """
    if verbose:
        print(state)
    recon_src = ReconSrc(gleamobject, statefile, M=20, verbose=verbose)

    residuals = []
    for i in range(len(recon_src.models)):
        recon_src.chmdl(i)
        delta = recon_src.reproj_residual()
        residuals.append(residual)

    if verbose:
        print("Number of residual models: {}".format(len(residuals)))

    # TODO


    return percentiles


def resample_eaglemodel(eagle_model, extent, shape, verbose=False):
    """
    Resample (usually downsample) an EAGLE model's kappa grid to match the specified scale and size

    Args:
        eagle_model <tuple(np.ndarray, header object)> - EAGLE model with (data, hdr)
        extent <tuple/list> - extent of the output
        shape <tuple/list> - shape of the output

    Kwargs:
        verbose <bool> - verbose mode; print command line statements

    Return:
        kappa_resmap <np.ndarray> - resampled kappa grid
    """
    kappa_map = eagle_model[0]
    hdr = eagle_model[1]
    print()

    pixrad = tuple(r//2 for r in kappa_map.shape)
    maprad = pixrad[1]*hdr['CDELT2']*3600

    if verbose:
        print(hdr)
        print("Kappa map: {}".format(kappa_map.shape))
        print("Pixrad {}".format(pixrad))
        print("Maprad {}".format(maprad))

    xmdl = np.linspace(-maprad, maprad, kappa_map.shape[0])
    ymdl = np.linspace(-maprad, maprad, kappa_map.shape[1])
    newx = np.linspace(extent[0], extent[1], shape[0])
    newy = np.linspace(extent[2], extent[3], shape[1])

    rescale = interpolate.interp2d(xmdl, ymdl, kappa_map)
    kappa_resmap = rescale(newx, newy)
    kappa_resmap[kappa_resmap < 0] = 0

    return kappa_resmap


def resample_glassmodel(gls_model, extent, shape, verbose=False):
    """
    Resample (usually upsample) a GLASS model's kappa grid to match the specified scales and size

    Args:
        gls_model <glass.LensModel object> - GLASS ensemble model
        extent <tuple/list> - extent of the output
        shape <tuple/list> - shape of the output

    Kwargs:
        verbose <bool> - verbose mode; print command line statements

    Return:
        kappa_resmap <np.ndarray> - resampled kappa grid
    """
    obj, data = gls_model['obj,data'][0]
    kappa_map = obj.basis._to_grid(data['kappa'], 1)
    pixrad = obj.basis.pixrad
    maprad = obj.basis.top_level_cell_size * (obj.basis.pixrad)
    mapextent = (-obj.basis.top_level_cell_size * (obj.basis.pixrad+0.5),
                 obj.basis.top_level_cell_size * (obj.basis.pixrad+0.5))
    cell_size = obj.basis.top_level_cell_size

    if verbose:
        print(obj)
        print("Kappa map: {}".format(kappa_map.shape))
        print("Pixrad {}".format(pixrad))
        print("Maprad {}".format(maprad))
        print("Mapextent {}".format(mapextent))
        print("Cellsize {}".format(cell_size))

    xmdl = np.linspace(-maprad, maprad, kappa_map.shape[0])
    ymdl = np.linspace(-maprad, maprad, kappa_map.shape[1])
    Xmdl, Ymdl = np.meshgrid(xmdl, ymdl)
    xnew = np.linspace(extent[0], extent[1], shape[0])
    ynew = np.linspace(extent[2], extent[3], shape[1])
    Xnew, Ynew = np.meshgrid(xnew, ynew)

    rescale = interpolate.Rbf(Xmdl, Ymdl, kappa_map)
    kappa_resmap = rescale(Xnew, Ynew)
    kappa_resmap[kappa_resmap < 0] = 0

    return kappa_resmap


if __name__ == "__main__":
    # root directories
    rdir = "/Users/phdenzel/adler"
    jsondir = rdir+"/json/"
    statedir = rdir+"/states/v2/"
    kappadir = rdir+"/kappa/"
    keys = ["H1S0A0B90G0", "H1S1A0B90G0", "H2S1A0B90G0", "H2S2A0B90G0", "H2S7A0B90G0",
            "H3S0A0B90G0", "H3S1A0B90G0", "H4S3A0B0G90", "H10S0A0B90G0", "H13S0A0B90G0",
            "H23S0A0B90G0", "H30S0A0B90G0", "H36S0A0B90G0", "H160S0A90B0G0", "H234S0A0B90G0"]

    # list all files within the directories
    ls_jsons = an_sorted([os.path.join(jsondir, f) for f in os.listdir(jsondir)
                          if f.endswith('.json')])
    ls_states = an_sorted([os.path.join(statedir, f) for f in os.listdir(statedir)
                           if f.endswith('.state')])
    ls_kappas = an_sorted([os.path.join(kappadir, f) for f in os.listdir(kappadir)
                           if f.endswith('.kappa.fits')])

    # file dictionaries
    jsons = {k: [f for f in ls_jsons if k in f] for k in keys}
    filtered_states = {k: [f for f in ls_states
                           if k in f and f.endswith('_filtered.state')] for k in keys}
    ls_states = [f for f in ls_states if not f.endswith('_filtered.state')]
    prefiltered_fsynth10 = {k: [f for f in ls_states
                                if k in f and f.endswith('_filtered_synthf10.state')]
                            for k in keys}
    prefiltered_fsynth25 = {k: [f for f in ls_states
                                if k in f and f.endswith('_filtered_synthf25.state')]
                            for k in keys}
    prefiltered_fsynth50 = {k: [f for f in ls_states
                                if k in f and f.endswith('_filtered_synthf50.state')]
                            for k in keys}
    ls_states = [f for f in ls_states if not (f.endswith('_filtered_synthf10.state')
                                              or f.endswith('_filtered_synthf10.state')
                                              or f.endswith('_filtered_synthf10.state'))]
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

    for k in keys:
        json = jsons[k][0]
        with open(json) as f:
            ml = MultiLens.from_json(f)
        for state in filtered_states[k]:
            synth10, synth25, synth50 = synth_filter(state, ml, percentiles=[10, 25, 50])
            print(synth10)
