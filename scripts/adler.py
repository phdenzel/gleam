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
                            if k in f and f.endswith('_filtered_synthf10.state')] for k in keys}
prefiltered_fsynth25 = {k: [f for f in ls_states
                            if k in f and f.endswith('_filtered_synthf25.state')] for k in keys}
prefiltered_fsynth50 = {k: [f for f in ls_states
                            if k in f and f.endswith('_filtered_synthf50.state')] for k in keys}
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

