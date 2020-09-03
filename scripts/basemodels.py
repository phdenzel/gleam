import sys
import os
import numpy as np
import matplotlib as mpl
mpl.rcParams['text.usetex'] = True
mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = ['DejaVu Sans']
mpl.rcParams['font.size'] = 17
import matplotlib.pyplot as plt
from gleam.starsampler import StarSampler
from gleam.utils.colors import GLEAMcolors

# StarSampler
bm = StarSampler.read_basemodels()

fig, ax = plt.subplots(figsize=(7, 7))
lss = ['-']*4 + ['--']*4 + [':']*4
clrs = GLEAMcolors.pl5
clrs = clrs[:4]*3

for i, sed in enumerate(bm['sed']):
    # ax.plot(bm['w'], sed, c=clrs[i], lw=1, ls=lss[i])
    ax.plot(bm['w'], 1e4*sed, c=clrs[i], lw=1, ls=lss[i])

fs = 24
# linestyle references for legend
plt.plot(-1, -1, color=clrs[0], ls='-', lw=2, label=r"$-0.3\leq\log(\frac{t}{Gyr})<0.0$")
plt.plot(-1, -1, color=clrs[1], ls='-', lw=2, label=r"$0.0\leq\log(\frac{t}{Gyr})<0.3$")
plt.plot(-1, -1, color=clrs[2], ls='-', lw=2, label=r"$0.3\leq\log(\frac{t}{Gyr})<0.7$")
plt.plot(-1, -1, color=clrs[3], ls='-', lw=2, label=r"$0.7\leq\log(\frac{t}{Gyr})<1.1$")
ax.plot(-1, -1, color='black', ls='-', lw=2, label="$Z/H=-0.5$")
ax.plot(-1, -1, color='black', ls='--', lw=2, label="$Z/H=0.0$")
ax.plot(-1, -1, color='black', ls=':', lw=2, label="$Z/H=0.3$")
lg = plt.legend(fontsize=fs-4, numpoints=1, fancybox=True, handlelength=1, borderpad=0.2, labelspacing=0.3, ncol=1)
plt.ylabel(r'$F_{\lambda}$', fontsize=fs+2)
plt.xlabel(r'$\lambda [\AA]$', fontsize=fs)

ax.set_xlim(1000, 14000)
# ax.set_ylim(0, 0.000415)
ax.set_ylim(0, 4.15)
ax.set_xticks([1500, 4000, 6500, 9000, 11500])
ax.tick_params(axis='both', which='major', labelsize=fs+4)
plt.tight_layout()

plt.savefig("base_models.pdf", transparent=True, bbox_inches='tight', pad_inches=0.1)
plt.close()
