"""
Per-pitcher data for the movement chart that keeps its speed.

For every pitcher: the usual movement chart (横, 縦 in inches, lefties mirrored), plus
what 09 showed the chart loses --

    height   where his arsenal sits along the common normal, as an mph constant c
             (the median over his types, so one outlying pitch does not move it):
             on the plane, speed = a·縦 + b·横 + c
    resid    per type: actual mean speed − (a·縦 + b·横 + c), in mph

and, for context, where each resid falls among all pitchers' pitches of the same type.

Two views are drawn side by side, sharing the vertical axis like a technical drawing:

    front  (横, 縦)    the usual movement chart
    side   (球速, 縦)   the plane seen from the side; speed and 縦 are the pair the plane
                       ties together (0.48 mph per inch), so each arsenal falls on a
                       sloped line, and a hard thrower's line sits further right.

Also kept, for other uses:

    face   (p1, p2)  the arsenal seen straight down the normal. p1 is the 横 axis laid
                     into the plane, p2 = n × p1 points up; both mix in a little speed.
    side   (p1, eq)  the plane seen edge-on. eq = speed − a·縦 − b·横 = what the pitch's
                     speed would be with no movement at all. A pitcher's arsenal is a flat
                     line here, at height c; a hard thrower's line simply sits higher.

Writes web/pitchers_data.js as window.PITCHERS (the artifact sandbox cannot fetch).

Usage:  python src/10_pitcher_cards_data.py
"""
import sys, os, json, importlib.util
import numpy as np, pandas as pd
sys.stdout.reconfigure(encoding="utf-8")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
spec = importlib.util.spec_from_file_location("sph", os.path.join(ROOT, "src", "03_sphere.py"))
sph = importlib.util.module_from_spec(spec); spec.loader.exec_module(sph)

YEAR = 2026
MIN_SHARE, MIN_N = .03, 30
DOTS = 60
SEED = 11

z = np.load(os.path.join(ROOT, "output", "plane_%d.npz" % YEAR))
n, S = z["normal"], float(z["s"])
A, B = -n[1] / (n[2] * S), -n[0] / (n[2] * S)        # mph per inch of 縦, of 横
V0 = float(z["v0"])
F1 = np.array([1., 0., 0.]) - n[0] * n; F1 /= np.linalg.norm(F1)   # 横, laid into the plane
F2 = np.cross(n, F1); F2 *= np.sign(F2[1])                          # up, in the plane

d = pd.read_parquet(sph.SNAP[YEAR], columns=["game_type", "pitcher", "player_name", "pitch_type",
                    "p_throws", "pfx_x", "pfx_z", "release_speed"])
d = d[(d.game_type == "R") & d.pitch_type.isin(sph.TYPES)].dropna().copy()
cnt = d.groupby("pitcher").size()
d = d[d.pitcher.isin(cnt[cnt >= sph.MIN_PITCHES].index)]
for c in ["pfx_x", "pfx_z", "release_speed"]:
    d[c] = d[c].astype(float)
d["x"] = np.where(d.p_throws.eq("L"), -d.pfx_x, d.pfx_x) * 12
d["y"] = d.pfx_z * 12
Wd = np.c_[d.x, d.y, (d.release_speed - V0) * S]
d["p1"], d["p2"] = Wd @ F1, Wd @ F2
d["eq"] = d.release_speed - (A * d.y + B * d.x)

g = d.groupby(["pitcher", "pitch_type"])
m = g[["x", "y", "release_speed"]].mean()
m["n"] = g.size()
m["share"] = m.n / m.groupby(level=0).n.transform("sum")
m = m[(m.share >= MIN_SHARE) & (m.n >= MIN_N)].reset_index()
m["pred0"] = A * m.y + B * m.x
W = np.c_[m.x, m.y, (m.release_speed - V0) * S]
m["p1"], m["p2"] = W @ F1, W @ F2
m["eq"] = m.release_speed - m.pred0                  # speed with the movement taken out
# median, not mean: one far-off pitch (a very slow changeup) must not drag the others
m["c"] = (m.release_speed - m.pred0).groupby(m.pitcher).transform("median")
m["resid"] = m.release_speed - m.pred0 - m.c
m["pct"] = m.groupby("pitch_type").resid.rank(pct=True)
m["peers"] = m.groupby("pitch_type").resid.transform("size")

rng = np.random.default_rng(SEED)
info = d.groupby("pitcher").agg(name=("player_name", "first"), hand=("p_throws", "first"),
                                total=("pitch_type", "size"))
code = {t: i for i, t in enumerate(sph.TYPES)}
out = []
for pid, mt in m.groupby("pitcher"):
    dp = d[(d.pitcher == pid) & d.pitch_type.isin(mt.pitch_type)]
    take = dp.sample(min(DOTS * len(mt), len(dp)), random_state=int(rng.integers(1 << 30)))
    fast = mt[mt.pitch_type.isin(sph.FAST)]
    top = fast.sort_values("n").iloc[-1] if len(fast) else None
    out.append({
        "id": int(pid), "name": info.loc[pid, "name"], "hand": info.loc[pid, "hand"],
        "total": int(info.loc[pid, "total"]), "c": round(float(mt.c.iloc[0]), 2),
        "fb": None if top is None else [top.pitch_type, round(float(top.release_speed), 1)],
        "types": [{"t": r.pitch_type, "x": round(r.x, 2), "y": round(r.y, 2),
                   "p1": round(r.p1, 2), "p2": round(r.p2, 2), "eq": round(r.eq, 2),
                   "v": round(r.release_speed, 1), "n": int(r.n), "share": round(r.share, 3),
                   "r": round(r.resid, 2), "pct": round(r.pct, 3), "peers": int(r.peers)}
                  for r in mt.sort_values("share", ascending=False).itertuples()],
        "dots": {"t": [code[t] for t in take.pitch_type],
                 "x": np.round(take.x.values, 1).tolist(), "y": np.round(take.y.values, 1).tolist(),
                 "v": np.round(take.release_speed.values, 1).tolist()},
    })
out.sort(key=lambda p: p["name"])
league = {"t": [code[t] for t in m.pitch_type], "x": np.round(m.x.values, 2).tolist(),
          "y": np.round(m.y.values, 2).tolist(), "v": np.round(m.release_speed.values, 2).tolist()}
cs = m.groupby("pitcher").c.first()
data = {"year": YEAR, "a": round(A, 4), "b": round(B, 4), "types": sph.TYPES,
        "v0": round(V0, 2), "s": S, "n": np.round(n, 4).tolist(),
        "f1": np.round(F1, 4).tolist(), "f2": np.round(F2, 4).tolist(), "league": league,
        "cq": np.round(np.quantile(cs, [.1, .25, .5, .75, .9]), 2).tolist(),
        "residSD": round(float(m.resid.std()), 2), "pitchers": out}
path = os.path.join(ROOT, "web", "pitchers_data.js")
with open(path, "w", encoding="utf-8") as f:
    f.write("window.PITCHERS=" + json.dumps(data, ensure_ascii=False, separators=(",", ":")) + ";")
print("F1 (横 %+.3f, 縦 %+.3f, 速度 %+.3f)   F2 (横 %+.3f, 縦 %+.3f, 速度 %+.3f)" % (*F1, *F2))
print("%d投手  a=%.3f b=%.3f  wrote %s (%.0f KB)" % (len(out), A, B, path, os.path.getsize(path) / 1024))
ch = m[m.pitch_type == "CH"].sort_values("resid")
print("CH 残差の両端:", ch.head(3)[["pitcher", "resid"]].values.tolist(), ch.tail(3)[["pitcher", "resid"]].values.tolist())
print(info.loc[[int(ch.pitcher.iloc[0]), int(ch.pitcher.iloc[-1])], "name"].tolist())
