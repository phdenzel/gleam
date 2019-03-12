from gleam.multilens import MultiLens
from gleam.redshiftsampler import RedshiftSampler

# Caba22    - no source - gls check - 0.74
# Caba39    - both      - gls check - 0.61
# MoreSA28  - no source - gls check - 0.68
# MoreSA59  - no source - gls check - 0.56
# MoreSA121 - no source - gls check - 0.56
# Para1024  - no source - gls check - 0.48
# Para1037  - both      - gls check - 0.32
# Para1079  - no source - gls check - 0.40
# Para1106  - both      - gls check - 0.36
# Para2109  - no source - gls check - 0.74
# Para2169  - no source - gls check - 0.49
# SW05      - no source - gls check - 0.62
# SW06      - no source - gls check - 0.36

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
sampler = RedshiftSampler.from_gleamobj(ml, verbose=1)
for i in range(sampler.image_magnitudes.shape[-1]):
    sampler.add2bpz_cat(sampler.image_magnitudes[:, i])
    z, zmin, zmax = sampler.run_bpz(verbose=True)
sampler.add2bpz_cat(sampler.lens_magnitudes)
z, zmin, zmax = sampler.run_bpz(verbose=True)
# sampler.plot_probs()
for l in ml:
    l.zl = z
print(ml['i'].__v__)
# ml.jsonify(name=jsonfile, with_hash=False)
