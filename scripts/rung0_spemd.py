import sys
import os

from gleam.model.spemd import SPEMD

import numpy as np
import matplotlib.pyplot as plt


rung0_seed3 = {
    'Nx': 13,
    'Ny': 13,
    'theta_E': 1.161,
    'q': 0.787,
    'phi_G': 1.605,
    'gamma': 2.044
}
rung0_seed4 = {
    'Nx': 13,
    'Ny': 13,
    'theta_E': 1.19,
    'q': 0.992,
    'phi_G': 2.246,
    'gamma': 2.05
}
rung0 = {'rung0_seed3': SPEMD(**rung0_seed3),
         'rung0_seed4': SPEMD(**rung0_seed4)}
for seed in rung0:
    rung0[seed].calc_map(smooth_center=1)
    fig = rung0[seed].plot_map(log=1, colorbar=0, contours=1, show=1,
                               mask=None, vmin=-1.3, vmax=0,
                               alpha=0.55, cmap='magma')
    # plt.savefig('{}_model.png'.format(seed), transparent=True)
