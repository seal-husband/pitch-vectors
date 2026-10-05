"""
Are chosen pitchers cleanly separated on the 3-D map?

For each named pitcher: his (x, y, z) and its sampling uncertainty -- his pitches are
resampled with replacement (the four-seam reference held fixed) and the coordinates
recomputed. Then, for every pair:

  distance        in the map's three log-ratio coordinates (one unit)
  percentile      where that distance falls among all pairs of the 477 pitchers
  separation      distance ÷ the combined bootstrap spread along the line joining them;
                  above ~4 the two clouds of resamples do not touch

Draws the three 2-D views with the league in grey and the named pitchers' resample clouds.

Usage:  python src/27_separation_check.py "Yamamoto, Yoshinobu" "Misiorowski, Jacob" "Sale, Chris" "Sánchez, Cristopher"
"""
import sys, os, importlib.util, itertools
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
spec = importlib.util.spec_from_file_location("m26", os.path.join(ROOT, "src", "26_three_axes_data.py"))
m26 = importlib.util.module_from_spec(spec); spec.loader.exec_module(m26)
sys.stdout.reconfigure(encoding="utf-8")
plt.rcParams["font.family"] = ["Yu Gothic", "Meiryo", "MS Gothic", "sans-serif"]
N_BOOT = 1000
rng = np.random.default_rng(27)
names = sys.argv[1:] or ["Yamamoto, Yoshinobu", "Misiorowski, Jacob", "Sale, Chris", "Sánchez, Cristopher"]

F, info, raw = m26.F, m26.info, m26.raw
name_to_id = {v: k for k, v in info.name.items()}
ALL = F[["xa", "yg", "z"]].values
MU, SD = np.zeros(3), np.ones(3)             # log-ratio axes share one unit: no standardising
ALL = (ALL - MU) / SD
allpair = np.linalg.norm(ALL[:, None] - ALL[None], axis=2)[np.triu_indices(len(ALL), 1)]


def coords(fx, fz, isff, unit, glove_unit):
    """The map's three coordinates from one sample of a pitcher's pitches (as fourparts.py:
    parts summed per peak, horizontal toward the peak's side, vertical up or down)."""
    n = len(fx); m = ~isff; U = len(glove_unit)
    sx = np.bincount(unit[m], weights=fx[m], minlength=U)
    sz = np.bincount(unit[m], weights=fz[m], minlength=U)
    h = np.where(glove_unit, np.clip(sx, 0, None), np.clip(-sx, 0, None)); v = np.abs(sz)
    ah, av = h[~glove_unit].sum() / n, v[~glove_unit].sum() / n
    gh, gv = h[glove_unit].sum() / n, v[glove_unit].sum() / n
    k = np.bincount(unit[m], minlength=U)
    au, gu = k[~glove_unit].sum() / n, k[glove_unit].sum() / n
    return np.array(m26.to_map(ah, av, gh, gv, au, gu))


boot, point = {}, {}
for nm in names:
    p = name_to_id[nm]
    g = raw[raw.pitcher == p]
    fx, fz, isff = g.fx.values, g.fz.values, (g.pitch_type == "FF").values
    key = np.where(g.peak.notna(), g.peak.fillna(-1).astype(int).astype(str), "p" + g.index.astype(str))
    unit, uniq = pd.factorize(key)
    glove_unit = np.array([(g.side.values[unit == u] == "glove")[0] for u in range(len(uniq))])
    point[nm] = coords(fx, fz, isff, unit, glove_unit)
    B = np.empty((N_BOOT, 3))
    for b in range(N_BOOT):
        i = rng.integers(0, len(g), len(g))
        B[b] = coords(fx[i], fz[i], isff[i], unit[i], glove_unit)
    boot[nm] = B
    lo, hi = np.quantile(B, [.025, .975], axis=0)
    assert np.allclose(point[nm], F.loc[p, ["xa", "yg", "z"]].values, atol=1e-6)
    print("%-22s x %+5.2f [%+5.2f, %+5.2f]  y %+5.2f [%+5.2f, %+5.2f]  z %+5.2f [%+5.2f, %+5.2f]   %d球"
          % (nm, point[nm][0], lo[0], hi[0], point[nm][1], lo[1], hi[1], point[nm][2], lo[2], hi[2], len(g)))

print("\n■ 組ごとの隔たり")
print("   %-44s %7s %10s %8s" % ("", "距離", "全組の中で", "分離度"))
for a, b in itertools.combinations(names, 2):
    pa, pb = (point[a] - MU) / SD, (point[b] - MU) / SD
    d = np.linalg.norm(pa - pb)
    u = (pa - pb) / d
    sd = np.sqrt(np.var(((boot[a] - MU) / SD) @ u) + np.var(((boot[b] - MU) / SD) @ u))
    pct = 100 * (allpair < d).mean()
    print("   %-20s ⇔ %-20s %6.2f   上位 %4.1f%%  %8.1f" % (a, b, d, 100 - pct, d / sd))
print("   （全%d投手の組の距離：中央 %.2f、上位10%%の境 %.2f）" % (len(ALL), np.median(allpair), np.quantile(allpair, .9)))

print("\n■ それぞれの最近傍（全投手の中で）")
for nm in names:
    p = name_to_id[nm]
    i = list(F.index).index(p)
    d = np.linalg.norm(ALL - ALL[i], axis=1)
    order = np.argsort(d)[1:4]
    print("   %-22s %s" % (nm, ", ".join("%s(%.2f)" % (info.name[F.index[j]], d[j]) for j in order)))

# ---- figure
C = ["#b8372b", "#2f64a3", "#2f8a4c", "#9a5bb5", "#c9822b", "#555555"]
views = [("xa", "yg", "x 腕側：ln(落とす/流す)/√2", "y グラブ側：ln(落とす/曲げる)/√2"),
         ("xa", "z", "x 腕側：ln(落とす/流す)/√2", "z ln(腕側/グラブ側)/√2"),
         ("yg", "z", "y グラブ側：ln(落とす/曲げる)/√2", "z ln(腕側/グラブ側)/√2")]
idx = {"xa": 0, "yg": 1, "z": 2}
fig, axs = plt.subplots(1, 3, figsize=(16, 5.6))
for ax, (a, b, la, lb) in zip(axs, views):
    ax.scatter(F[a], F[b], s=9, c="#b9c1ca", edgecolors="none")
    for k, nm in enumerate(names):
        B = boot[nm]
        ax.scatter(B[:, idx[a]], B[:, idx[b]], s=2, c=C[k], alpha=.12, edgecolors="none")
        ax.scatter([point[nm][idx[a]]], [point[nm][idx[b]]], s=70, c=C[k], edgecolors="white", linewidths=1, zorder=5)
        ax.annotate(nm.split(",")[0], (point[nm][idx[a]], point[nm][idx[b]]), xytext=(6, 5),
                    textcoords="offset points", fontsize=10, color=C[k], fontweight="bold")
    ax.set_xlabel(la); ax.set_ylabel(lb)
    ax.axhline(0, color="#e2e6ea", lw=.8, zorder=0); ax.axvline(0, color="#e2e6ea", lw=.8, zorder=0)
    for s_ in ["top", "right"]:
        ax.spines[s_].set_visible(False)
fig.suptitle("3D地図の3つの投影：灰色＝全477投手、色つきの霧＝各投手の投球を再標本化した位置（%d回）" % N_BOOT,
             fontsize=11, x=.01, ha="left")
fig.tight_layout()
out = os.path.join(ROOT, "output", "figures", "27_separation.png")
fig.savefig(out, dpi=120); print(out)
