"""
Data for the two-panel map: every pitcher placed by the four numbers of 23.

  arm panel    x = 腕側の横, y = 腕側の縦     (inches per pitch thrown)
  glove panel  x = グラブ側の横, y = グラブ側の縦

Also written, to read a point without going back to the data: how much of each side the
pitcher throws and the side's mean position (the four numbers = usage × mean position),
his labelled pitch mix for reference, and his nearest pitchers in the four numbers.

The four numbers come from fourparts.py: arm side / glove side is decided per peak (found
without labels in 18), and the vertical part counts movement up as well as down, so a
submariner's rising slider is counted (it used to fall outside both sides).

Writes web/four_data.js.

Usage:  python src/24_four_map_data.py
"""
import sys, os, json
from paths import SNAP
import fourparts
import numpy as np, pandas as pd
sys.stdout.reconfigure(encoding="utf-8")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MIN_PITCHES = 400

raw = pd.read_parquet(SNAP[2026],
                      columns=["game_type", "pitcher", "player_name", "pitch_type", "p_throws", "stand", "pfx_x", "pfx_z",
                               "game_pk", "at_bat_number", "pitch_number"])
raw = raw[raw.game_type == "R"].dropna().copy()
tot = raw.groupby("pitcher").size()
raw = raw[raw.pitcher.isin(tot[tot >= MIN_PITCHES].index)]
tot = raw.groupby("pitcher").size()
for c in ["pfx_x", "pfx_z"]:
    raw[c] = raw[c].astype(float)
raw["hx"] = np.where(raw.p_throws.eq("L"), -raw.pfx_x, raw.pfx_x) * 12
raw["vz"] = raw.pfx_z * 12

# four-seam reference, converted from SI / FC when the pitcher has no four-seam (as in 21)
fb = raw[raw.pitch_type.isin(["FF", "SI", "FC"])].groupby(["pitcher", "pitch_type"])[["hx", "vz"]].agg(["mean", "size"])
fb.columns = ["hx", "n1", "vz", "n"]
fb = fb.drop(columns="n1").reset_index()
fb["share"] = fb.n / tot.reindex(fb.pitcher).values
has = fb[(fb.n >= 30) & (fb.share >= .05)]
W = has.pivot(index="pitcher", columns="pitch_type", values=["hx", "vz"])
off = {}
for t in ["SI", "FC"]:
    b = W.dropna(subset=[("hx", "FF"), ("hx", t)])
    off[t] = np.array([(b[(k, "FF")] - b[(k, t)]).median() for k in ["hx", "vz"]])
ref, kind = {}, {}
for p, g in has.groupby("pitcher"):
    ff = g[g.pitch_type == "FF"]
    if len(ff):
        ref[p] = ff[["hx", "vz"]].values[0]; kind[p] = "FF"
    else:
        top = g.sort_values("n").iloc[-1]
        ref[p] = top[["hx", "vz"]].values.astype(float) + off[top.pitch_type]; kind[p] = "換算(" + top.pitch_type + ")"
raw = raw[raw.pitcher.isin(ref.keys())].copy()
R = np.array([ref[p] for p in raw.pitcher])
raw["fx"], raw["fz"] = raw.hx - R[:, 0], raw.vz - R[:, 1]
raw = fourparts.attach_peaks(raw)
nonff = raw[raw.pitch_type != "FF"].copy()

F = fourparts.four_numbers(raw)
F["ou"] = 0.                                       # no "other" side any more: every pitch has a side
# mean position on each side = the part divided by that side's share
for part, use, col in (("ah", "au", "amx"), ("av", "au", "amz"), ("gh", "gu", "gmx"), ("gv", "gu", "gmz")):
    F[col] = np.where(F[use] > 0, F[part] / F[use].where(F[use] > 0), np.nan)
X = F[["ah", "av", "gh", "gv"]].values
D = np.linalg.norm(X[:, None] - X[None], axis=2)
info = raw.groupby("pitcher").agg(name=("player_name", "first"), hand=("p_throws", "first"))
mix = raw.groupby("pitcher").pitch_type.value_counts(normalize=True)

out = []
pids = list(F.index)
for i, p in enumerate(pids):
    r = F.loc[p]
    m = mix[p]
    out.append({"id": int(p), "name": info.loc[p, "name"], "hand": info.loc[p, "hand"], "ref": kind[p],
                "v": [round(float(x), 2) for x in X[i]],
                "use": [round(float(r.au), 3), round(float(r.gu), 3), round(float(r.ou), 3),
                        round(float((raw[raw.pitcher == p].pitch_type == "FF").mean()), 3)],
                "mean": [None if pd.isna(x) else round(float(x), 1) for x in (r.amx, r.amz, r.gmx, r.gmz)],
                "mix": [[t, round(float(s), 3)] for t, s in m[m >= .02].items()],
                "nn": [[int(j), round(float(D[i, j]), 2)] for j in np.argsort(D[i])[1:9]]})
path = os.path.join(ROOT, "web", "four_data.js")
with open(path, "w", encoding="utf-8") as f:
    f.write("window.FOUR=" + json.dumps({"pitchers": out}, ensure_ascii=False, separators=(",", ":")) + ";")
print("%d投手  wrote %s (%.0f KB)" % (len(out), path, os.path.getsize(path) / 1024))
print(F[["ah", "av", "gh", "gv"]].describe().round(2).to_string())
