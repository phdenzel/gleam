import numpy as np
from matplotlib import pyplot as plt
from gleam.multilens import MultiLens
from gleam.starsampler import StarSampler
from gleam.roiselector import ROISelector
from gleam.glass_interface import glass_renv
from gleam.utils import colors as glmc
glass = glass_renv()

# read in state files
state_files = [
    "data/Caba22/caba22.state",
    "data/Caba39/caba39.state",
    "data/MoreSA28/sa28.state",
    "data/MoreSA59/sa59.state",
    "data/MoreSA121/sa121.state",
    "data/Para1024/para1024.state",
    # "data/Para1037/para1037.state",
    "data/Para1079/para1079.state",
    "data/Para1106/para1106.state",
    "data/Para2109/para2109.state",
    "data/Para2169/para2169.state",
    # "data/SW05/sw05.state",
    "data/SW06/sw06.state"
]
states = [loadstate(f) for f in state_files]

json_files = [
    "data/Caba22.multilens#a061574e76f9a7d1a7bbbd7a0a61e.json",
    "data/Caba39.multilens#7c20b67b2e57c890d283ab7ba35e4.json",
    "data/MoreSA28.multilens#45d308e03681414d48a2e8c6504cf.json",
    "data/MoreSA59.multilens#e27bed3796282edbb2f4add2b0928.json",
    "data/MoreSA121.multilens#f8401b8d6046f4bb22b714032be1e.json",
    "data/Para1024.multilens#7d27cc987b89ad1b9d03828a8bc2d.json",
    # "data/Para1037.multilens#0d34f904ecd3e5c38035b43483295.json",
    "data/Para1079.multilens#81a0a92146af752f4404c0380fdf6.json",
    "data/Para1106.multilens#843e6153eea1cdb0c8d6c6c89eba2.json",
    "data/Para2109.multilens#586f69cabe4f43ef74751c4820fa3.json",
    "data/Para2169.multilens#1cf88bd132faf8931a02f3076b591.json",
    # "data/SW05.multilens#4ebca5f7da0763d2a8aae675763a8.json",
    "data/SW06.multilens#dbfc84ca1d42c30410d812fc1dab2.json"
]
models = []
for jsn in json_files:
    with open(jsn, 'r') as f:
        ml = MultiLens.from_json(f)
        models.append(ml)

# get lens and stel maps in units of Msun/arcsec^2
lens_maps = []
stel_maps = []
px2arcsecs = []
for gls, ml in zip(states, models):
    gls.make_ensemble_average()
    glsobj, glsdata = gls.ensemble_average['obj,data'][0]
    px2arcsec = (glsobj.basis.top_level_cell_size / glsobj.basis.subdivision)
    # arcsec2kpc = glass.scales.convert('arcsec to kpc', 1, glsobj.dL, glsdata['nu'])
    kappa = glass.scales.convert('kappa to Msun/arcsec^2', 1, glsobj.dL, glsdata['nu'])
    glsextent = np.array([-1, -1, 1, 1]) * glsobj.basis.mapextent
    lens_map = glsobj.basis.kappa_grid(glsdata) * kappa
    stel_map = ml['i'].stel_map
    stel_map = StarSampler.resample_map(stel_map, ml['i'].extent, lens_map.shape, glsextent)
    print(ml[0].filename.split('.')[0])
    lens_maps.append(lens_map)
    stel_maps.append(stel_map)
    px2arcsecs.append(px2arcsec)
    # plt.style.use('dark_background')
    # plt.imshow(stel_map, origin='lower', cmap='magma',
    #            extent=[glsextent[0], glsextent[2], glsextent[1], glsextent[3]])
    # plt.colorbar(label=r'$M_{\odot}/\mathrm{arcsec}^{2}$')
    # plt.xlabel(r'$\mathrm{arcsec}$')
    # plt.ylabel(r'$\mathrm{arcsec}$')
    # plt.tight_layout()
    # plt.savefig('SW06_light_model.png', transparent=True)

    Mtot = ROISelector.r_integrate(lens_map)*px2arcsec**2
    print("Total mass: {:e}".format(Mtot))
    Mstel = ROISelector.r_integrate(stel_map)*px2arcsec**2
    print("Stellar mass: {:e}".format(Mstel))

# get half light radii
halflight_R = []
for px2arcsec, stel_map in zip(px2arcsecs, stel_maps):
    Mstel = ROISelector.r_integrate(stel_map)*px2arcsec**2
    prof = ROISelector.cumr_profile(stel_map)*px2arcsec**2
    idxR = (np.abs(prof - (0.5*Mstel))).argmin()
    r = np.linspace(0, px2arcsec*0.5*stel_map.shape[0], 0.5*stel_map.shape[0])[idxR]
    halflight_R.append(r)

# frames in units of halflight radii
upto = 4    # halflight radii
nframes = 200  # number of movie frames
frames = np.linspace(0.1, upto, nframes)
N_models = len(lens_maps)

stel_mass_frames = []
lens_mass_frames = []
stel_prof_frames = []
lens_prof_frames = []
for f in range(nframes):
    stel_mass_frames.append([ROISelector.r_integrate(
        stel_maps[i], R=halflight_R[i]*frames[f]/px2arcsecs[i])*px2arcsecs[i]**2
                             for i in range(N_models)])
    lens_mass_frames.append([ROISelector.r_integrate(
        lens_maps[i], R=halflight_R[i]*frames[f]/px2arcsecs[i])*px2arcsecs[i]**2
                             for i in range(N_models)])
    stel_prof_frames.append([ROISelector.cumr_profile(
        stel_maps[i], radii=halflight_R[i]*frames[:min(f+1, nframes-1)]/px2arcsecs[i])
                             * px2arcsecs[i]**2
                             for i in range(N_models)])
    lens_prof_frames.append([ROISelector.cumr_profile(
        lens_maps[i], radii=halflight_R[i]*frames[:min(f+1, nframes-1)]/px2arcsecs[i])
                             * px2arcsecs[i]**2
                             for i in range(N_models)])
for f in range(nframes):
    for i in range(N_models):
        plt.loglog(lens_mass_frames[f][i], stel_mass_frames[f][i], linewidth=0,
                   marker=">", markersize=4, color=glmc.colors[i],
                   label=models[i][0].filename.split('.')[0])
        plt.legend(loc=2, fontsize=8, numpoints=1, markerscale=2, ncol=2)
        if len(lens_prof_frames[f][i]) >= 30 and len(lens_prof_frames[f][i]) >= 30:
            plt.loglog(lens_prof_frames[f][i][-30:], stel_prof_frames[f][i][-30:], linewidth=1,
                       color=glmc.colors[i])
        else:
            plt.loglog(lens_prof_frames[f][i], stel_prof_frames[f][i], linewidth=1,
                       color=glmc.colors[i])
    plt.xlim(1e10, 2.74e13)
    plt.ylim(1e8, 1e12)
    plt.xlabel(r'$\mathrm{M_{lens}} [\mathrm{M}_{\odot}]$')
    plt.ylabel(r'$\mathrm{M_{stel}} [\mathrm{M}_{\odot}]$')
    plt.tight_layout()
    plt.savefig("frame{:04d}.png".format(f), transparent=False)
    plt.clf()
for i in range(N_models):
    plt.loglog(lens_mass_frames[-1][i], stel_mass_frames[-1][i], linewidth=0,
               marker=">", markersize=4, color=glmc.colors[i],
               label=models[i][0].filename.split('.')[0])
    plt.legend(loc=2, fontsize=8, numpoints=1, markerscale=1, ncol=2)
    plt.loglog(lens_prof_frames[-1][i], stel_prof_frames[-1][i], linewidth=1,
               color=glmc.colors[i])
plt.xlim(1e10, 2.74e13)
plt.ylim(1e8, 1e12)
plt.xlabel(r'$\mathrm{M_{lens}} [\mathrm{M}_{\odot}]$')
plt.ylabel(r'$\mathrm{M_{stel}} [\mathrm{M}_{\odot}]$')
plt.tight_layout()
for i in range(nframes, nframes+51):
    plt.savefig("frame{:04d}.png".format(i), transparent=False)
plt.close()
