"""
Does ignoring the changeup/splitter's slowness change the map?

The map uses each pitch's movement from the pitcher's four-seam (fx, fz) and no speed. Speed
is mostly redundant -- the Pitch Plane predicts it from movement -- except for the pitcher's
plane height (arm strength, cancelled by the four-seam origin anyway) and each pitch's
distance off the plane, r. Changeups and splitters sit ~3 mph below it. That r is what the
map throws away; this file puts it back and compares.

    r   = (speed − four-seam speed) − (A·fz + B·fx)       mph; < 0: slower than its movement says

Two ways back in:

  drop    a pitch slower than the plane says spends longer in the air and falls further:
          S = 0.63 in per mph (02). So fz' = fz + S·r, and the map is rebuilt from (fx, fz').
          The changeup's slowness becomes extra arm-side drop.
  axis    a fourth coordinate, the arm side's usage-weighted mean r, scaled to the map's spread.
          A null fourth axis (r shuffled between pitchers) shows how much any added
          dimension moves the neighbours on its own.

For each: how far the map moves (axis correlations, 10-nearest-neighbour overlap, the four
named pitchers), and whether what came back relates to the Stuff stand-in or xwOBA.

Usage:  python src/33_offplane_check.py
"""
import sys, os, importlib.util, builtins
import numpy as np, pandas as pd
from scipy import stats
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_predict, KFold
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
A, B = -n[1] / (n[2] * S), -n[0] / (n[2] * S)

# speed relative to the four-seam reference (real FF, or the SI/FC conversion), as in 31
sp = pd.read_parquet(SNAP[2026], columns=["game_pk", "at_bat_number", "pitch_number", "pitcher", "release_speed"])
key = ["game_pk", "at_bat_number", "pitch_number", "pitcher"]
for c in key:
    sp[c] = sp[c].astype("int64"); raw[c] = raw[c].astype("int64")
raw = raw.merge(sp, on=key, how="left").set_index(raw.index)
raw["release_speed"] = raw.release_speed.astype(float)
both = raw[raw.pitch_type.isin(["FF", "SI", "FC"])].groupby(["pitcher", "pitch_type"]).release_speed.mean().unstack()
off = {t: (both.FF - both[t]).dropna().median() for t in ["SI", "FC"]}
vref = {}
for p in raw.pitcher.unique():
    k = m26.kind[p]
    vref[p] = both.loc[p, "FF"] if k == "FF" else both.loc[p, k[3:5]] + off[k[3:5]]
raw["dv"] = raw.release_speed - raw.pitcher.map(vref)
raw["r"] = raw.dv - (A * raw.fz + B * raw.fx)
raw = raw[raw.r.notna()]

nf = raw[raw.pitch_type != "FF"]
print("PP からのずれ r の中央値（自分のフォーシーム基準、mph）")
for t in ["SI", "FC", "SL", "ST", "CU", "CH", "FS"]:
    print("   %-3s %+.1f" % (t, nf[nf.pitch_type == t].r.median()), end="")
print("\n   腕側 %+.1f   グラブ側 %+.1f" % (nf[nf.side == "arm"].r.median(), nf[nf.side == "glove"].r.median()))


def as_map(F):
    x, y, zz = to_map(F.ah, F.av, F.gh, F.gv, F.au, F.gu)
    return pd.DataFrame({"x": x, "y": y, "z": zz}, index=F.index)


F0 = fourparts.four_numbers(raw)
M0 = as_map(F0)
F1 = fourparts.four_numbers(raw.assign(fz=raw.fz + S * raw.r))
M1 = as_map(F1).loc[M0.index]
arm = nf[nf.side == "arm"]
dr = arm.groupby("pitcher").r.mean().reindex(M0.index).fillna(0)      # pitchers with no arm side: 0
scale = M0.std().mean() / dr.std()
M2 = M0.assign(w=(dr - dr.mean()) * scale)
rng = np.random.default_rng(33)
M3 = M0.assign(w=rng.permutation(M2.w.values))


def knn(M, k=10):
    X = M.values
    D = np.linalg.norm(X[:, None] - X[None], axis=2)
    return np.argsort(D, 1)[:, 1:k + 1], D


N0, D0 = knn(M0)
NAMES = ["Yamamoto, Yoshinobu", "Misiorowski, Jacob", "Sale, Chris", "Sánchez, Cristopher"]
ids = {v: k for k, v in info.name.items()}
pos = {p: i for i, p in enumerate(M0.index)}
print("\n■ 地図はどれだけ動くか（データの順：x=腕側 落とす÷流す、y=グラブ側 落とす÷曲げる、z=腕側÷グラブ側）")
for lab, M in (("落下に換算して足す", M1), ("4本目の軸として足す", M2), ("対照：4本目をでたらめに", M3)):
    Nk, Dk = knn(M)
    ov = np.array([len(set(N0[i]) & set(Nk[i])) / 10 for i in range(len(M0))])
    cor = "  ".join("%s %.3f" % (c, np.corrcoef(M0[c], M[c])[0, 1]) for c in "xyz")
    print("   %-16s 3軸の相関 %s   近い10人の一致 中央 %.0f%%" % (lab, cor, 100 * np.median(ov)))
print("   4人の位置（今 → 落下に換算）")
for nm in NAMES:
    a, b = M0.loc[ids[nm]], M1.loc[ids[nm]]
    print("      %-22s x %+.2f→%+.2f  y %+.2f→%+.2f  z %+.2f→%+.2f   腕側の平均ずれ %+.1f mph"
          % (nm, a.x, b.x, a.y, b.y, a.z, b.z, dr[ids[nm]]))
N1, _ = knn(M1)
for nm in ("Yamamoto, Yoshinobu", "Sánchez, Cristopher"):
    i = pos[ids[nm]]
    print("   %s の近い5人  今: %s\n   %s              換算: %s" % (nm[:8], ", ".join(info.name[M0.index[j]] for j in N0[i][:5]),
                                                            " " * 8, ", ".join(info.name[M0.index[j]] for j in N1[i][:5])))

# ---------------------------------------------------------------- what came back: performance?
Sx = pd.read_csv(os.path.join(ROOT, "output", "pp_pitchers_2026.csv"), index_col=0)
kf = KFold(10, shuffle=True, random_state=0)
print("\n■ 戻した情報は成績と関係するか（100打席以上、Spearman）")
J = M0.join(M1, rsuffix="1").assign(dr=dr).join(Sx[["stuff", "xwoba"]], how="inner").dropna()
print("   投手 %d" % len(J))
for c, lab in (("dr", "腕側の平均ずれ r"), ("x", "x 腕側 落とす÷流す（今）"), ("x1", "x 腕側 落とす÷流す（換算）")):
    r1, p1 = stats.spearmanr(J[c], J.stuff); r2, p2 = stats.spearmanr(J[c], J.xwoba)
    print("   %-24s × 代用Stuff %+.3f (p=%.3f)   × xwOBA %+.3f (p=%.3f)" % (lab, r1, p1, r2, p2))


def cv(cols, tgt):
    X_, t = J[cols].values, J[tgt].values
    lin = LinearRegression().fit(X_, t).score(X_, t)
    pr = cross_val_predict(HistGradientBoostingRegressor(max_iter=200, learning_rate=.05, min_samples_leaf=20,
                                                         random_state=0), X_, t, cv=kf)
    return lin, 1 - ((t - pr) ** 2).sum() / ((t - t.mean()) ** 2).sum()


for tgt, lab in (("stuff", "代用Stuff"), ("xwoba", "xwOBA")):
    for cols, cl in ((["x", "y", "z"], "今の3軸"), (["x1", "y1", "z1"], "換算した3軸"), (["x", "y", "z", "dr"], "今の3軸＋ずれ")):
        lin, c = cv(cols, tgt)
        print("   %-6s を %-10s で説明：線形 R² %.3f ／ 交差検証 R² %+.3f" % (lab, cl, lin, c))
