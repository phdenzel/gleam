from gleam.multilens import MultiLens
from gleam.lightsampler import LightSampler

# Caba22    - done
# Caba39    - good
# MoreSA28  - done
# MoreSA59  - good
# MoreSA121 - done
# Para1024  - done
# Para1037  - good
# Para1079  - done
# Para1106  - okay
# Para2109  - good
# Para2169  - done
# SW05      - done
# SW06      - good

jsonfile = "data/Caba22.multilens#a061574e76f9a7d1a7bbbd7a0a61e.json"
# jsonfile = "data/Caba39.multilens#7c20b67b2e57c890d283ab7ba35e4.json"
# jsonfile = "data/MoreSA28.multilens#45d308e03681414d48a2e8c6504cf.json"
# jsonfile = "data/MoreSA59.multilens#e27bed3796282edbb2f4add2b0928.json"
# jsonfile = "data/MoreSA121.multilens#f8401b8d6046f4bb22b714032be1e.json"
# jsonfile = "data/Para1024.multilens#7d27cc987b89ad1b9d03828a8bc2d.json"
# jsonfile = "data/Para1037.multilens#0d34f904ecd3e5c38035b43483295.json"
# jsonfile = "data/Para1079.multilens#81a0a92146af752f4404c0380fdf6.json"
# jsonfile = "data/Para1106.multilens#843e6153eea1cdb0c8d6c6c89eba2.json"
# jsonfile = "data/Para2109.multilens#586f69cabe4f43ef74751c4820fa3.json"
# jsonfile = "data/Para2169.multilens#1cf88bd132faf8931a02f3076b591.json"
# jsonfile = "data/SW05.multilens#4ebca5f7da0763d2a8aae675763a8.json"
# jsonfile = "data/SW06.multilens#dbfc84ca1d42c30410d812fc1dab2.json"
with open(jsonfile, 'r') as f:
    ml = MultiLens.from_json(f, verbose=1)
print("\n")
sampler = LightSampler.from_gleamobj(ml, verbose=1)
sampler.parameters = {'x': 65, 'y': 71}
sampler.fixed = 'x', 'y', 'c_0'
# sampler.fixed = ['c_0']
# kwargs = {'n_walkers': 500, 'burn_in': 50, 'mcmc_steps': 300}
kwargs = {'n_walkers': 200, 'burn_in': 50, 'mcmc_steps': 200}
sampler.run(**kwargs)
sampler.parspace_plot(show=1)
sampler.ensemble_average(verbose=0)
ml['i'].light_model = sampler.model
sampler.model.plot_map(show=1)
sampler.plot_residuals(show=1)
print(ml['i'].__v__)
print(sampler.model.__v__)
# ml.jsonify(name=jsonfile, with_hash=False)
