"""
Raw movement, or movement projected onto the Pitch Plane -- does the map care?

The map's coordinates use each pitch's movement from the pitcher's four-seam, (fx, fz),
and no speed. That squashes the 3-D vector (Δ横, Δ縦, Δ球速 × 0.63) along the SPEED axis.
Projecting onto the Pitch Plane instead squashes it along the plane's NORMAL, which leans
17.5° from the speed axis:

    in-plane 横  p1 = F1 · v      F1 ≈ (+0.996, +0.027, −0.090)
    in-plane 縦  p2 = F2 · v      F2 ≈ ( 0.000, +0.958, +0.285)

Both throw away the plane's height (the pitcher's arm strength -- and relative to his own
four-seam it cancels anyway) and the off-plane part (a changeup's slowness beyond what its
movement implies). They differ only by the in-plane share of speed, which enters p2 with
weight ~0.29: a pitch 8 mph slower than the four-seam reads ~1.4 in lower.

This file rebuilds the four numbers and the map from (p1, p2) and compares.

Usage:  python src/31_projection_vs_raw.py
"""
import sys, os, importlib.util, builtins
import numpy as np, pandas as pd
sys.stdout.reconfigure(encoding="utf-8")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
from paths import SNAP
import fourparts

_p = builtins.print; builtins.print = lambda *a, **k: None
spec = importlib.util.spec_from_file_location("m26", os.path.join(ROOT, "src", "26_three_axes_data.py"))
m26 = importlib.util.module_from_spec(spec); spec.loader.exec_module(m26)
builtins.print = _p
raw, info, to_map = m26.raw.copy(), m26.info, m26.to_map

z = np.load(os.path.join(ROOT, "output", "plane_2026.npz"))
n, S = z["normal"], float(z["s"])
F1 = np.array([1., 0., 0.]) - n[0] * n; F1 /= np.linalg.norm(F1)
F2 = np.cross(n, F1); F2 *= np.sign(F2[1])
print("F1 (%+.3f, %+.3f, %+.3f)   F2 (%+.3f, %+.3f, %+.3f)" % (*F1, *F2))

# speed relative to the pitcher's four-seam reference (real FF, or the SI/FC conversion as in 24)
sp = pd.read_parquet(SNAP[2026], columns=["game_pk", "at_bat_number", "pitch_number", "pitcher", "release_speed"])
key = ["game_pk", "at_bat_number", "pitch_number", "pitcher"]
for c in key:
    sp[c] = sp[c].astype("int64"); raw[c] = raw[c].astype("int64")
raw = raw.merge(sp, on=key, how="left").set_index(raw.index)
raw["release_speed"] = raw.release_speed.astype(float)
ffv = raw[raw.pitch_type == "FF"].groupby("pitcher").release_speed.mean()
fb = raw[raw.pitch_type.isin(["SI", "FC"])].groupby(["pitcher", "pitch_type"]).release_speed.agg(["mean", "size"]).reset_index()
both = raw[raw.pitch_type.isin(["FF", "SI", "FC"])].groupby(["pitcher", "pitch_type"]).release_speed.mean().unstack()
off = {t: (both.FF - both[t]).dropna().median() for t in ["SI", "FC"]}
vref = {}
for p in raw.pitcher.unique():
    if p in ffv.index and m26.kind[p] == "FF":
        vref[p] = ffv[p]
    else:
        t = m26.kind[p][3:5]
        vref[p] = raw[(raw.pitcher == p) & (raw.pitch_type == t)].release_speed.mean() + off[t]
raw["fv"] = (raw.release_speed - raw.pitcher.map(vref)) * S
V = raw[["fx", "fz", "fv"]].values
proj = raw.copy()
proj["fx"], proj["fz"] = V @ F1, V @ F2
print("速度差の中央値（腕側の球 / グラブ側の球）: %+.1f / %+.1f mph"
      % (raw[raw.side == "arm"].fv.median() / S, raw[raw.side == "glove"].fv.median() / S))

A, B = fourparts.four_numbers(raw), fourparts.four_numbers(proj)
M = {}
for k, Fn in (("raw", A), ("proj", B)):
    x, y, zz = to_map(Fn.ah, Fn.av, Fn.gh, Fn.gv, Fn.au, Fn.gu)
    M[k] = pd.DataFrame({"x": x, "y": y, "z": zz})
R, Q = M["raw"], M["proj"].loc[M["raw"].index]
print("\n■ 地図の3軸の一致（データの順：x=腕側 落とす÷流す、y=グラブ側 落とす÷曲げる、z=腕側÷グラブ側）")
for c in "xyz":
    print("   %s  相関 %.3f   平均の差 %+.3f（SD %.2f）" % (c, np.corrcoef(R[c], Q[c])[0, 1], (Q[c] - R[c]).mean(), R[c].std()))
Ra, Qa = R.values, Q.values
Dr = np.linalg.norm(Ra[:, None] - Ra[None], axis=2); Dq = np.linalg.norm(Qa[:, None] - Qa[None], axis=2)
k = 10
nr, nq = np.argsort(Dr, 1)[:, 1:k + 1], np.argsort(Dq, 1)[:, 1:k + 1]
ov = np.array([len(set(nr[i]) & set(nq[i])) / k for i in range(len(Ra))])
mv = np.linalg.norm(Qa - Ra, axis=1)
print("   近い10人の一致 中央 %.0f%%   位置の移動 中央 %.3f（全組の距離の中央 %.2f）"
      % (100 * np.median(ov), np.median(mv), np.median(Dr[np.triu_indices(len(Ra), 1)])))
names = info.name.reindex(R.index)
top = pd.DataFrame({"name": names, "move": mv, "x_raw": R.x, "x_proj": Q.x, "z_raw": R.z, "z_proj": Q.z}).sort_values("move", ascending=False)
print(top.head(8).round(2).to_string(index=False))

# ---------------------------------------------------------------- vs Stuff stand-in and xwOBA
from scipy import stats
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_predict, KFold
Sx = pd.read_csv(os.path.join(ROOT, "output", "pp_pitchers_2026.csv"), index_col=0)     # stuff, xwoba (13)
kf = KFold(10, shuffle=True, random_state=0)
print("\n■ 代用Stuff・xwOBA との関係（100打席以上、29 と同じ比べ方）")
lab = {"x": "x 腕側 落とす÷流す", "y": "y グラブ側 落とす÷曲げる", "z": "z 腕側÷グラブ側"}
for k in ("raw", "proj"):
    J = M[k].join(Sx[["stuff", "xwoba"]], how="inner").dropna()
    print("   %s（%d投手）" % ("変化量そのまま（球速を無視）" if k == "raw" else "PP に投影", len(J)))
    for c in "xyz":
        r1, p1 = stats.spearmanr(J[c], J.stuff); r2, p2 = stats.spearmanr(J[c], J.xwoba)
        print("      %-20s × 代用Stuff %+.3f (p=%.2f)   × xwOBA %+.3f (p=%.2f)" % (lab[c], r1, p1, r2, p2))
    for tgt in ("stuff", "xwoba"):
        X_, t = J[["x", "y", "z"]].values, J[tgt].values
        lin = LinearRegression().fit(X_, t).score(X_, t)
        pr = cross_val_predict(HistGradientBoostingRegressor(max_iter=200, learning_rate=.05, min_samples_leaf=20,
                                                             random_state=0), X_, t, cv=kf)
        cv = 1 - ((t - pr) ** 2).sum() / ((t - t.mean()) ** 2).sum()
        print("      3軸で %-6s を説明：線形 R² %.3f ／ 交差検証 R² %+.3f" % ("代用Stuff" if tgt == "stuff" else "xwOBA", lin, cv))
