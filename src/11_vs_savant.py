"""
Savant's way vs the plane's way of saying "this pitch is fast".

  Savant   Δ = speed − league mean speed of pitches with the SAME LABEL
  plane    speed = c (the pitcher's height) + a·縦 + b·横 (what the movement implies) + r

Savant's Δ is therefore  (c − league c) + (movement term − its league mean for the label) + r.
This file measures how big each part is, whether the parts repeat across a pitcher's
types (redundancy), and where the two readings disagree.

Usage:  python src/11_vs_savant.py
"""
import sys, os, json
import numpy as np, pandas as pd
sys.stdout.reconfigure(encoding="utf-8")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
s = open(os.path.join(ROOT, "web", "pitchers_data.js"), encoding="utf-8").read()
D = json.loads(s[len("window.PITCHERS="):-1])
A, B = D["a"], D["b"]
rows = [dict(pid=p["id"], name=p["name"], c=p["c"], k=len(p["types"]), **t) for p in D["pitchers"] for t in p["types"]]
m = pd.DataFrame(rows)
m = m[m.k >= 3]
m["mov"] = A * m.y + B * m.x
m["sav"] = m.v - m.groupby("t").v.transform("mean")                       # Savant's Δ
parts = {"腕（c − 平均）": m.c - m.c.mean(),
         "変化量（同じラベル内での差）": m.mov - m.groupby("t").mov.transform("mean"),
         "平面からのずれ r": m.r - m.groupby("t").r.transform("mean")}
print("■ 1. Savant の Δ（同ラベル平均との差）を三つに分ける   投手×球種 n=%d" % len(m))
tot = m.sav.var()
print("   Δ の分散 %.2f mph²（SD %.2f）" % (tot, np.sqrt(tot)))
for k, v in parts.items():
    print("   %-26s 分散 %.2f  共分散込みの寄与 %4.0f%%" % (k, v.var(), 100 * np.cov(v, m.sav)[0, 1] / tot))

print("\n■ 2. 一人の投手の中で、球種どうしの値は重複しているか（投手ごとの相関）")
def pair_corr(col):
    w = m.pivot_table(index="pid", columns="t", values=col)
    out = []
    for a_, b_ in [("FF", "SL"), ("FF", "CH"), ("FF", "CU"), ("SL", "CH"), ("SI", "SL")]:
        ok = w[[a_, b_]].dropna()
        out.append("%s-%s %+.2f" % (a_, b_, np.corrcoef(ok[a_], ok[b_])[0, 1]))
    return "  ".join(out)
print("   Savant Δ      " + pair_corr("sav"))
print("   平面のずれ r  " + pair_corr("r"))

print("\n■ 3. 読みが食い違う球種（|Δ| と |r| の差が大きい、腕の高さを除いて）")
m["sav_noarm"] = m.sav - (m.c - m.c.mean())
m["gap"] = m.sav_noarm - m.r
for t in ["SL", "CU", "CH", "ST"]:
    g = m[m.t == t]
    lo, hi = g.nsmallest(1, "gap").iloc[0], g.nlargest(1, "gap").iloc[0]
    for q, lab in ((lo, "Savant は遅い／平面は並"), (hi, "Savant は速い／平面は並")):
        print("   %-3s %-22s %s  球速 %.1f  横%+.1f 縦%+.1f  Δ(腕除く)%+.1f  r%+.1f"
              % (t, q["name"], lab, q.v, q.x, q.y, q.sav_noarm, q.r))
print("   同ラベル内での、腕を除いた Δ と r の相関:",
      "  ".join("%s %.2f" % (t, np.corrcoef(g.sav_noarm, g.r)[0, 1]) for t, g in m.groupby("t") if len(g) > 50))
