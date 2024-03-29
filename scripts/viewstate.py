"""
@author: phdenzel

Create a GLEAM viewstate of a lens model from GLASS, PixeLens, etc.

Usage:
    python viewstate.py gls.state

"""
import sys
import os
import numpy as np
from matplotlib import pyplot as plt
gleam_root = "/Users/phdenzel/gleam"
sys.path.append(gleam_root)
from gleam.glass_interface import glass_renv
from gleam.utils.lensing import LensModel
from gleam.utils.plotting import viewstate_plots
glass = glass_renv()


if __name__ == "__main__":
    # saving input states in other formats
    opts = glass.environment.Environment.global_opts['argv']
    models = (LensModel(f) for f in opts)

    # general plot settings
    plt.rcParams.update({'axes.labelsize':  16,
                         'xtick.labelsize': 14,
                         'ytick.labelsize': 14})

    for mdl in models:
        print("Processing lens model {}...".format(mdl))
        viewstate_plots(mdl, savefig=True, verbose=True)
