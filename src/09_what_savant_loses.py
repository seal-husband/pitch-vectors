"""
What does a Savant movement chart lose about speed -- and what would a plot on the plane keep?

Projecting a pitch onto the common plane throws away exactly its NORMAL component. But on
the plane, speed is already a function of position (08: speed = a·縦 + b·横 + const), so
the movement chart plus that equation carries the along-plane part of speed for free.
What the chart truly loses is therefore:

    (1) the pitcher's height along the normal  -- how hard he throws, one number
    (2) each type's off-plane distance         -- speed its movement does not explain

This file measures how big each part is, and whether (2) is a stable trait or noise.

Usage:  python src/09_what_savant_loses.py
"""
import sys, os, importlib.util
import numpy as np, pandas as pd
sys.stdout.reconfigure(encoding="utf-8")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
spec = importlib.util.spec_from_file_location("p8", os.path.join(ROOT, "src", "08_the_plane.py"))
p8 = importlib.util.module_from_spec(spec); spec.loader.exec_module(p8)

z = np.load(os.path.join(ROOT, "output", "plane_2026.npz"))
n, S = z["normal"], float(z["s"])
a, b = p8.natural(n)
d, v0 = p8.load(2026)
d["mph"] = d.release_speed
d["pred"] = a * d.y + b * d.x                      # along-plane speed, up to a pitcher constant
d["half"] = d.groupby(["pitcher", "pitch_type"]).cumcount() % 2

m = p8.type_means(d)
m = m.merge(d.groupby(["pitcher", "pitch_type"])[["mph", "pred"]].mean().reset_index(),
            on=["pitcher", "pitch_type"])
tot = m.mph.var()
height = m.groupby("pitcher").mph.transform("mean") - m.groupby("pitcher").pred.transform("mean")
resid = m.mph - m.pred - height
within = m.mph - m.groupby("pitcher").mph.transform("mean")
print("球種平均の球速（%d投手×球種 = %d）の分散の内訳" % (m.pitcher.nunique(), len(m)))
print("   全体の分散                      %.1f mph²（SD %.2f）" % (tot, np.sqrt(tot)))
print("   投手内（球種間）                %.1f" % within.var())
print("     うち変化量＋平面の式で決まる  %.1f   → 投手内の %.0f%%" %
      (within.var() - resid.var(), 100 * (1 - resid.var() / within.var())))
print("     平面から外れる分（残差）      %.1f   SD %.2f mph" % (resid.var(), resid.std()))
print("   投手間（平面の高さ）            %.1f   SD %.2f mph" % (height.groupby(m.pitcher).first().var(),
                                                       height.groupby(m.pitcher).first().std()))

m["res"] = resid
print("\n残差を球種で集計（mph、＋＝変化量のわりに速い）")
g = m.groupby("pitch_type").res
print(pd.DataFrame({"中央": g.median(), "四分位25": g.quantile(.25), "四分位75": g.quantile(.75),
                    "n": g.size()}).round(2).reindex(["FF","SI","FC","SL","ST","SV","CU","KC","CH","FS"]).dropna().to_string())
eta = 1 - m.groupby("pitch_type").res.transform(lambda s: s - s.mean()).var() / resid.var()
print("   残差の分散のうち球種ラベルで説明できる割合 %.0f%%" % (100 * eta))

# is the off-plane distance a trait?  split each pitcher-type by alternate pitches
h = d[d.set_index(["pitcher", "pitch_type"]).index.isin(m.set_index(["pitcher", "pitch_type"]).index)]
hm = h.groupby(["pitcher", "pitch_type", "half"])[["mph", "pred"]].mean().unstack("half")
r_ = []
for k in (0, 1):
    e = hm[("mph", k)] - hm[("pred", k)]
    r_.append(e - e.groupby(level=0).transform("mean"))
r = np.corrcoef(r_[0], r_[1])[0, 1]
print("\n残差の信頼性（一球おきの折半）  r = %.3f   Spearman-Brown %.3f" % (r, 2 * r / (1 + r)))
