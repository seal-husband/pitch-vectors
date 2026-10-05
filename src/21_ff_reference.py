"""
One internal standard for everybody: the four-seam, real or converted.

Peaks were read from each pitcher's PRIMARY fastball. That is a different standard for a
sinker pitcher and a four-seam pitcher, and on the arm side it wrecks comparison -- a
changeup sits (−5.9, −10.9 in, −8.0 mph) from a four-seam but (+0.8, −6.2, −8.6) from a
sinker, and the difference is the reference, not the changeup.

The fix is the chemist's: one standard, with a conversion factor when a sample lacks it.

  pitcher throws FF (≥ 5% and ≥ 30 pitches)   reference = his FF
  otherwise                                    reference = his primary fastball + the league
                                               offset (FF − that type), measured on pitchers
                                               who throw both

Every peak is re-expressed from this reference; the peak that IS the four-seam (nearest the
reference, labelled FF) is dropped. A sinker then becomes an ordinary arm-side peak.

Writes output/peaks_ff_2026.parquet: the peak table in the four-seam frame.

Usage:  python src/21_ff_reference.py
"""
import sys, os
from paths import SNAP
import numpy as np, pandas as pd
sys.stdout.reconfigure(encoding="utf-8")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

raw = pd.read_parquet(SNAP[2026],
                      columns=["game_type", "pitcher", "pitch_type", "p_throws", "pfx_x", "pfx_z", "release_speed"])
raw = raw[raw.game_type == "R"].dropna().copy()
for c in ["pfx_x", "pfx_z", "release_speed"]:
    raw[c] = raw[c].astype(float)
raw["hx"] = np.where(raw.p_throws.eq("L"), -raw.pfx_x, raw.pfx_x) * 12
raw["vz"] = raw.pfx_z * 12
raw["vs"] = raw.release_speed * .63
Pk = pd.read_parquet(os.path.join(ROOT, "output", "peaks_2026.parquet"))
raw = raw[raw.pitcher.isin(set(Pk.pitcher))]

g = raw[raw.pitch_type.isin(["FF", "SI", "FC"])].groupby(["pitcher", "pitch_type"])
fb = g[["hx", "vz", "vs"]].mean()
fb["n"] = g.size()
tot = raw.groupby("pitcher").size()
fb["share"] = fb.n / tot.reindex(fb.index.get_level_values(0)).values
fb = fb.reset_index()
has = fb[(fb.n >= 30) & (fb.share >= .05)]

# league conversion: FF − SI and FF − FC on pitchers who throw both
W = has.pivot(index="pitcher", columns="pitch_type", values=["hx", "vz", "vs"])
off = {}
for t in ["SI", "FC"]:
    both = W.dropna(subset=[("hx", "FF"), ("hx", t)])
    off[t] = np.array([(both[(k, "FF")] - both[(k, t)]).median() for k in ["hx", "vz", "vs"]])
    print("換算 FF − %s（両方投げる %d 人の中央値）: 横 %+.1f in, 縦 %+.1f in, 球速 %+.1f mph"
          % (t, len(both), off[t][0], off[t][1], off[t][2] / .63))

ref, kind = {}, {}
for p, gg in has.groupby("pitcher"):
    ff = gg[gg.pitch_type == "FF"]
    if len(ff):
        ref[p] = ff[["hx", "vz", "vs"]].values[0]; kind[p] = "FF"
    else:
        top = gg.sort_values("n").iloc[-1]
        ref[p] = top[["hx", "vz", "vs"]].values.astype(float) + off[top.pitch_type]; kind[p] = "換算(%s)" % top.pitch_type
prim = {}
for p, gg in raw[raw.pitch_type.isin(["FF", "SI", "FC"])].groupby("pitcher"):
    t = gg.pitch_type.value_counts().index[0]
    prim[p] = gg[gg.pitch_type == t][["hx", "vz", "vs"]].mean().values
print("基準の内訳:", pd.Series(kind).value_counts().to_dict(), " 基準なし:", len(set(Pk.pitcher) - set(ref)))

Pk = Pk[Pk.pitcher.isin(ref.keys()) & Pk.pitcher.isin(prim.keys())].copy()
shift = np.array([prim[p] - ref[p] for p in Pk.pitcher])        # old frame → FF frame
Pk["fx"] = Pk.dhx + shift[:, 0]
Pk["fz"] = Pk.dvz + shift[:, 1]
Pk["fmph"] = Pk.dmph + shift[:, 2] / .63
Pk["ref_kind"] = Pk.pitcher.map(kind)
# drop the four-seam's own peak
Pk["r_ff"] = np.sqrt(Pk.fx ** 2 + Pk.fz ** 2 + (Pk.fmph * .63) ** 2)
ffpk = Pk[Pk.label == "FF"].groupby("pitcher").r_ff.idxmin()
Pk = Pk.drop(index=ffpk.values)
print("フォーシーム基準のピーク %d（FF のピーク %d を除く）" % (len(Pk), len(ffpk)))
print(Pk.groupby("label")[["fx", "fz", "fmph"]].median().round(1)
      .loc[["SI", "CH", "FS", "FC", "SL", "ST", "SV", "CU", "KC"]].to_string())
Pk.to_parquet(os.path.join(ROOT, "output", "peaks_ff_2026.parquet"), index=False)
