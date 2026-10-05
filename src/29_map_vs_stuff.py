"""
Is the map free of Stuff?

The map (26: three log-ratio coordinates of how non-four-seam movement is distributed)
was built to leave out what Stuff+ measures: speed, the pitcher's plane height, and the
total amount of movement. This file checks that it worked.

Stuff+ itself is not public; the stand-in from 13 is used (physics-only model of run value,
cross-validated by pitcher, averaged over a pitcher's pitches, usage-weighted). Positive
controls -- quantities Stuff should see -- show whether the stand-in is measuring anything.

  1. rank correlation of each map axis with the stand-in, and with xwOBA
  2. how much of the stand-in the three axes predict together: linear R², and a gradient-
     boosted model under 10-fold cross-validation (catches non-linear dependence)
  3. the same for the controls: four-seam speed, plane height c, total movement, and the
     four numbers before they were turned into ratios
  4. what was left out, on the same yardstick: C and four-seam speed against the stand-in
     and xwOBA; draws output/figures/29_c_vs_stuff.png (C beside the map's strongest axis)

Usage:  python src/29_map_vs_stuff.py
"""
import sys, os, importlib.util, builtins
from paths import SNAP
import numpy as np, pandas as pd
from scipy import stats
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_predict, KFold
sys.stdout.reconfigure(encoding="utf-8")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

_p = builtins.print; builtins.print = lambda *a, **k: None
spec = importlib.util.spec_from_file_location("m26", os.path.join(ROOT, "src", "26_three_axes_data.py"))
m26 = importlib.util.module_from_spec(spec); spec.loader.exec_module(m26)
builtins.print = _p
F = m26.F.copy()                                   # xa, yg, z (log-ratio map) + the four numbers
S = pd.read_csv(os.path.join(ROOT, "output", "pp_pitchers_2026.csv"), index_col=0)   # stuff, c, xwoba (13)
raw = m26.raw
ffv = raw[raw.pitch_type == "FF"].groupby("pitcher").release_speed.mean() if "release_speed" in raw else None
if ffv is None:
    sc = pd.read_parquet(SNAP[2026],
                         columns=["game_type", "pitcher", "pitch_type", "release_speed"])
    sc = sc[(sc.game_type == "R") & sc.pitch_type.isin(["FF", "SI", "FC"])].dropna()
    top = sc.groupby("pitcher").pitch_type.agg(lambda s: s.value_counts().index[0])
    ffv = sc[sc.pitch_type.values == sc.pitcher.map(top).values].groupby("pitcher").release_speed.mean().astype(float)
D = F.join(S[["stuff", "c", "xwoba"]], how="inner").join(ffv.rename("fbv"), how="inner").dropna(subset=["stuff", "xwoba", "fbv"])
D["total"] = np.hypot(D.ah, D.av) + np.hypot(D.gh, D.gv)
print("投手 %d（地図・代用Stuff・xwOBA・速球の球速がそろう投手、100打席以上）" % len(D))

print("\n■ 1. 地図の各軸と代用Stuff・xwOBAの順位相関（Spearman）")
for c, lab in (("xa", "x 腕側 落とす÷流す"), ("yg", "y グラブ側 落とす÷曲げる"), ("z", "z 腕側÷グラブ側")):
    r1, p1 = stats.spearmanr(D[c], D.stuff); r2, p2 = stats.spearmanr(D[c], D.xwoba)
    print("   %-22s × 代用Stuff %+.2f (p=%.2f)    × xwOBA %+.2f (p=%.2f)" % (lab, r1, p1, r2, p2))

kf = KFold(10, shuffle=True, random_state=0)


def explain(cols, y):
    X = D[cols].values; t = D[y].values
    lin = LinearRegression().fit(X, t).score(X, t)
    pred = cross_val_predict(HistGradientBoostingRegressor(max_iter=200, learning_rate=.05, min_samples_leaf=20,
                                                           random_state=0), X, t, cv=kf)
    cv = 1 - ((t - pred) ** 2).sum() / ((t - t.mean()) ** 2).sum()
    return lin, cv


print("\n■ 2–3. 代用Stuff をどれだけ説明できるか（線形 R² ／ 非線形・10分割交差検証 R²）")
rows = [("地図の3軸（x, y, z）", ["xa", "yg", "z"]),
        ("  比にする前の4つの数", ["ah", "av", "gh", "gv"]),
        ("  変化の総量（地図から外したもの）", ["total"]),
        ("対照：主な速球の球速", ["fbv"]),
        ("対照：PPの高さ c", ["c"]),
        ("対照：速球の球速 ＋ 地図の3軸", ["fbv", "xa", "yg", "z"])]
for lab, cols in rows:
    lin, cv = explain(cols, "stuff")
    print("   %-30s %.3f ／ %+.3f" % (lab, lin, cv))
print("   （交差検証 R² が 0 以下なら、平均を当てるより良い予測はできていない）")

print("\n■ 参考：xwOBA を説明できるか（同じ形式）")
for lab, cols in rows[:1] + rows[3:4]:
    lin, cv = explain(cols, "xwoba")
    print("   %-30s %.3f ／ %+.3f" % (lab, lin, cv))

print("\n■ 4. 外した量の側：C と速球の球速（Spearman）")
for c, lab in (("c", "PPの高さ C"), ("fbv", "主な速球の球速")):
    r1, p1 = stats.spearmanr(D[c], D.stuff); r2, p2 = stats.spearmanr(D[c], D.xwoba)
    print("   %-14s × 代用Stuff %+.2f (p=%.0e)    × xwOBA %+.2f (p=%.0e)" % (lab, r1, p1, r2, p2))
print("   C × 速球の球速 %+.2f" % stats.spearmanr(D.c, D.fbv)[0])

# C and the map's strongest axis against the stand-in, same vertical scale
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.family": ["Yu Gothic", "Meiryo", "MS Gothic", "sans-serif"], "font.size": 10})
fig, axs = plt.subplots(1, 2, figsize=(8, 4.4), dpi=160, sharey=True)
HL = {"Misiorowski": ("#eb6834", "Misiorowski"), "Sale, Chris": ("#1baf7a", "Sale"),
      "Yamamoto": ("#2a78d6", "Yamamoto"), "Sánchez, Cristopher": ("#4a3aa7", "Sánchez")}
names = m26.info.name.reindex(D.index)
for ax, (col, xl) in zip(axs, (("c", "PP の高さ C（mph）"), ("yg", "グラブ側：落とす／曲げる（地図の軸）"))):
    ax.scatter(D[col], D.stuff, s=9, color="#9a9994", alpha=.6, linewidths=0)
    b, a = np.polyfit(D[col].values, D.stuff.values, 1)
    xs = np.linspace(D[col].min(), D[col].max(), 2)
    ax.plot(xs, a + b * xs, color="#3d3c39", lw=1.2)
    for k, (cl, lab) in HL.items():
        i = names.index[names.str.startswith(k)][0]
        ax.scatter([D.loc[i, col]], [D.loc[i, "stuff"]], s=40, color=cl, edgecolors="white", linewidths=1, zorder=5)
        ax.annotate(lab, (D.loc[i, col], D.loc[i, "stuff"]), xytext=(5, 4), textcoords="offset points", fontsize=8.5)
    rho, p = stats.spearmanr(D[col], D.stuff)
    ax.set_title("順位相関 %+.2f（p = %.0e）" % (rho, p), fontsize=10, loc="left")
    ax.set_xlabel(xl); ax.spines[["top", "right"]].set_visible(False)
axs[0].set_ylabel("代用Stuff（100球あたりの失点抑止、予測値）\n↑ 球質が良い", linespacing=1.4)
fig.suptitle("C と代用Stuff、地図の軸と代用Stuff（%d投手、100打席以上）" % len(D), x=0.02, ha="left",
             fontsize=12, fontweight="bold")
fig.tight_layout()
out = os.path.join(ROOT, "output", "figures", "29_c_vs_stuff.png")
fig.savefig(out, facecolor="white")
print("   ", out)
