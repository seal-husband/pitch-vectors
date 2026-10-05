"""
From four numbers to two: what each side's elongated cloud is made of.

On the map each side looks like one streak running away from the origin (arm side to the
lower left, glove side to the lower right). A streak through the origin means the main
variation is MAGNITUDE -- how much of that side a pitcher throws, times how far -- and the
direction (vertical vs horizontal preference) is what is left across it.

The candidate reductions, one number per side:

  ratio      angle atan2(縦, 横): which way that side's pitches go, independent of amount
  diff       縦 − 横
  amount     length of the (横, 縦) vector: how much that side is relied on
  pca        the side's first principal component

Each is judged by split-half reliability (alternate pitches), by how much of the side's
variance it carries, and by how well the resulting 2-D map keeps the 4-D neighbourhoods.

Usage:  python src/25_projection.py
"""
import sys, os, importlib.util
import numpy as np, pandas as pd
from scipy import stats
sys.stdout.reconfigure(encoding="utf-8")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
spec = importlib.util.spec_from_file_location("m24", os.path.join(ROOT, "src", "24_four_map_data.py"))
m24 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m24)                           # builds nonff, tot, F (the four numbers); prints its summary
import fourparts
pitches, F = m24.raw.copy(), m24.F.copy()          # every pitch, with its peak and side
pitches["half"] = pitches.groupby("pitcher").cumcount() % 2
halves = [fourparts.four_numbers(pitches[pitches.half == h])[["ah", "av", "gh", "gv", "au", "gu"]] for h in (0, 1)]
X = F[["ah", "av", "gh", "gv"]]
sb = lambda r: 2 * r / (1 + r)

print("■ 1. 各側の雲の形（投手 %d）" % len(X))
for s, (h, v) in (("腕側", ("ah", "av")), ("グラブ側", ("gh", "gv"))):
    Z = X[[h, v]].values
    val, vec = np.linalg.eigh(np.cov(Z.T))
    main = vec[:, 1] * np.sign(vec[:, 1].sum())
    ang = np.degrees(np.arctan2(main[1], main[0]))
    # does the main axis pass through the origin? compare the mean's direction with the axis
    mean_ang = np.degrees(np.arctan2(Z[:, 1].mean(), Z[:, 0].mean()))
    print("   %-4s 主軸の向き %.0f°（横=0°, 縦=90°）  平均の向き %.0f°  主軸の分散 %.0f%%"
          % (s, ang, mean_ang, 100 * val[1] / val.sum()))

print("\n■ 2. 一側を1つの数にする候補：折半の再現性、その側の分散をどれだけ運ぶか")


def reduce(df, kind, side):
    h, v = (df.ah, df.av) if side == "arm" else (df.gh, df.gv)
    if kind == "ratio":
        return pd.Series(np.degrees(np.arctan2(v, h)), index=df.index)
    if kind == "diff":
        return v - h
    if kind == "amount":
        return np.hypot(h, v)
    Z = X[["ah", "av"]] if side == "arm" else X[["gh", "gv"]]
    val, vec = np.linalg.eigh(np.cov(Z.values.T)); w = vec[:, 1] * np.sign(vec[:, 1].sum())
    return h * w[0] + v * w[1]


res = {}
for side in ("arm", "glove"):
    Zs = X[["ah", "av"]] if side == "arm" else X[["gh", "gv"]]
    for kind in ("ratio", "diff", "amount", "pca"):
        full = reduce(X, kind, side)
        a, b = reduce(halves[0], kind, side), reduce(halves[1], kind, side)
        ok = full.notna() & a.notna() & b.notna() & np.isfinite(full)
        rel = sb(stats.pearsonr(a[ok], b[ok])[0])
        # share of the side's 2-D variance carried: R² of regressing (h, v) on the number
        from sklearn.linear_model import LinearRegression
        f = full[ok].values.reshape(-1, 1)
        r2 = LinearRegression().fit(f, Zs[ok].values).score(f, Zs[ok].values)
        res[(side, kind)] = full
        print("   %-6s %-7s 再現性 %.3f   その側の2次元の分散の保持 %.2f" % (side, kind, rel, r2))
    # reliability of the ratio for pitchers who throw little of that side
    use = F.au if side == "arm" else F.gu
    full = reduce(X, "ratio", side); a, b = reduce(halves[0], "ratio", side), reduce(halves[1], "ratio", side)
    for lo, hi in ((0, .1), (.1, .2), (.2, .4), (.4, 1.1)):
        m = (use >= lo) & (use < hi) & a.notna() & b.notna()
        if m.sum() > 15:
            print("        比（向き）の再現性  その側の使用率 %2.0f〜%2.0f%%: %.3f (n=%d)"
                  % (100 * lo, 100 * min(hi, 1), sb(stats.pearsonr(a[m], b[m])[0]), m.sum()))

print("\n■ 3. 2次元の地図にしたとき、4次元の近傍（k=10）をどれだけ保つか")
Xv = X.values
D4 = np.linalg.norm(Xv[:, None] - Xv[None], axis=2)


def knn_keep(Y, k=10):
    Y = (Y - Y.mean(0)) / Y.std(0)
    D2 = np.linalg.norm(Y[:, None] - Y[None], axis=2)
    a = np.argsort(D4, 1)[:, 1:k + 1]; b = np.argsort(D2, 1)[:, 1:k + 1]
    return np.mean([len(set(a[i]) & set(b[i])) / k for i in range(len(Y))])


Xc = Xv - Xv.mean(0); val, vec = np.linalg.eigh(np.cov(Xc.T))
for lab, Y in [("比（腕側）× 比（グラブ側）", np.c_[res[("arm", "ratio")], res[("glove", "ratio")]]),
               ("差（腕側）× 差（グラブ側）", np.c_[res[("arm", "diff")], res[("glove", "diff")]]),
               ("量（腕側）× 量（グラブ側）", np.c_[res[("arm", "amount")], res[("glove", "amount")]]),
               ("4次元の主成分2つ", Xc @ vec[:, [3, 2]])]:
    ok = np.isfinite(Y).all(1)
    Yf = np.where(ok[:, None], Y, np.nanmean(Y, 0))
    print("   %-24s 近傍の保持 %.2f   （参考：無作為 %.2f）" % (lab, knn_keep(Yf), 10 / len(Y)))
print("   4次元の主成分：分散 %s  PC1 %s  PC2 %s" % (np.round(val[::-1] / val.sum(), 2),
      np.round(vec[:, 3], 2), np.round(vec[:, 2], 2)))

print("\n■ 4. 主成分に近い、読める2軸の組み合わせ")
amt_a, amt_g = res[("arm", "amount")].values, res[("glove", "amount")].values
dif_a, dif_g = res[("arm", "diff")].values, res[("glove", "diff")].values
rat_a, rat_g = res[("arm", "ratio")].values, res[("glove", "ratio")].values
Z = Xc @ vec[:, [3, 2]]
for lab, Y in [("グラブ側の量 − 腕側の量 × グラブ側の差（縦−横）", np.c_[amt_g - amt_a, dif_g]),
               ("グラブ側の量 − 腕側の量 × グラブ側の比（向き）", np.c_[amt_g - amt_a, rat_g]),
               ("腕側の量 × グラブ側の量", np.c_[amt_a, amt_g])]:
    ok = np.isfinite(Y).all(1)
    Yf = np.where(ok[:, None], Y, np.nanmean(Y, 0))
    r1 = abs(np.corrcoef(Yf[:, 0], Z[:, 0])[0, 1]); r2 = abs(np.corrcoef(Yf[:, 1], Z[:, 1])[0, 1])
    print("   %-36s 近傍の保持 %.2f   PC1との相関 %.2f  PC2との相関 %.2f" % (lab, knn_keep(Yf), r1, r2))
