"""
Difference or ratio? Treat the four numbers as a composition.

The four numbers (arm run, arm drop, glove sweep, glove drop; inches per pitch) are
positive parts of one whole: how a pitcher distributes the movement of everything that
is not his four-seam. Compositional data analysis (Aitchison) says such parts are compared
by RATIOS, and its standard coordinates are isometric log-ratios. For these four parts,
with the natural partition {arm | glove}, the three ilr balances are

    b1 = ln(arm drop / arm run) / √2
    b2 = ln(glove drop / glove sweep) / √2
    b3 = ln(arm amount / glove amount) · 1/√2    (amount = geometric mean of the side's parts)

-- the three axes chosen on the map, all as ratios, in one geometry. The total (how much
non-four-seam movement there is at all) is the composition's size and is left out by design.

The flaw that made ratios fail for x and y -- a side barely thrown is set by a handful of
pitches -- is compositional analysis's zero problem, and its standard fix is a small
pseudo-amount δ added to every part before taking ratios (Bayesian-multiplicative zero
replacement, in effect shrinkage toward "no preference" for sides with little data).

Compared here with the current map (x, y differences; z angle), for several δ:
split-half reliability overall and for pitchers with under 10% arm-side pitches, and the
separation of four named pitchers.

Usage:  python src/28_logratio.py
"""
import sys, os, importlib.util, builtins, itertools
import numpy as np, pandas as pd
sys.stdout.reconfigure(encoding="utf-8")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_p = builtins.print; builtins.print = lambda *a, **k: None
spec = importlib.util.spec_from_file_location("m25", os.path.join(ROOT, "src", "25_projection.py"))
m25 = importlib.util.module_from_spec(spec); spec.loader.exec_module(m25)
builtins.print = _p
X, F, halves = m25.X, m25.F, m25.halves
info = m25.m24.info
ids = {v: k for k, v in info.name.items()}
NAMES = ["Yamamoto, Yoshinobu", "Misiorowski, Jacob", "Sale, Chris", "Sánchez, Cristopher"]
sb = lambda r: 2 * r / (1 + r)


def current(D):
    aa, ag = np.hypot(D.ah, D.av), np.hypot(D.gh, D.gv)
    return pd.DataFrame({"x": D.av - D.ah, "y": D.gv - D.gh, "z": np.degrees(np.arctan2(aa, ag))}, index=D.index)


def ilr(D, delta):
    P = D[["ah", "av", "gh", "gv"]].clip(lower=0) + delta
    return pd.DataFrame({"x": np.log(P.av / P.ah) / np.sqrt(2),
                         "y": np.log(P.gv / P.gh) / np.sqrt(2),
                         "z": np.log(np.sqrt(P.ah * P.av) / np.sqrt(P.gh * P.gv)) / np.sqrt(2)}, index=D.index)


low = F.au < .10
print("投手 %d（腕側 1割未満 %d）" % (len(X), low.sum()))
print("   %-22s %22s %28s" % ("", "再現性（x / y / z）", "腕側1割未満での x / z"))
rows = [("現行（差・差・角度）", current)] + [("対数比 δ=%.2f" % d, (lambda D, d=d: ilr(D, d))) for d in (.05, .1, .25, .5, 1.0)]
for lab, f in rows:
    A, H0, H1 = f(X), f(halves[0]), f(halves[1])
    r = [sb(np.corrcoef(H0[c], H1[c])[0, 1]) for c in "xyz"]
    rl = [sb(np.corrcoef(H0[c][low], H1[c][low])[0, 1]) for c in "xz"]
    print("   %-22s %.3f / %.3f / %.3f        %.3f / %.3f" % (lab, *r, *rl))

print("\n■ 4人の組ごとの隔たり（3軸を標準化、全組の中で「これより近い組の割合」）")
for lab, f in [rows[0], rows[3]]:
    A = f(X); Z = ((A - A.mean()) / A.std()).values
    allp = np.linalg.norm(Z[:, None] - Z[None], axis=2)[np.triu_indices(len(Z), 1)]
    idx = {p: i for i, p in enumerate(A.index)}
    out = []
    for a, b in itertools.combinations(NAMES, 2):
        d = np.linalg.norm(Z[idx[ids[a]]] - Z[idx[ids[b]]])
        out.append("%s⇔%s %2.0f%%" % (a[:4], b[:4], 100 * (allp < d).mean()))
    print("   %-20s %s" % (lab, "  ".join(out)))

A = ilr(X, .25)
print("\n■ δ=0.25 の対数比座標（4人と、腕側を投げない投手の位置）")
for nm in NAMES:
    r = A.loc[ids[nm]]
    print("   %-22s x %+5.2f  y %+5.2f  z %+5.2f" % (nm, r.x, r.y, r.z))
z0 = F.au == 0
print("   腕側ゼロの投手 %d 人：x は全員 %+.2f、z は %+.2f 〜 %+.2f（グラブ側の量で並ぶ）"
      % (z0.sum(), A.x[z0].iloc[0], A.z[z0].min(), A.z[z0].max()))
print("   全体の範囲 x %+.2f〜%+.2f  y %+.2f〜%+.2f  z %+.2f〜%+.2f"
      % (A.x.min(), A.x.max(), A.y.min(), A.y.max(), A.z.min(), A.z.max()))

# ---------------------------------------------------------------- the usage axis (2026-10-05)
# z became the plain share ratio ln((au + ε) / (gu + ε)) / √2. How much does ε matter?
print("\n■ z を球数の比にしたとき（擬似量 ε を各側の割合に足す）")
G, H0, H1 = m25.F, halves[0], halves[1]
old = ilr(X, .25).z
for e in (.005, .01, .02, .05, .1):
    zf = lambda D: np.log((D.au + e) / (D.gu + e)) / np.sqrt(2)
    r = sb(np.corrcoef(zf(H0), zf(H1))[0, 1]); rl = sb(np.corrcoef(zf(H0)[low], zf(H1)[low])[0, 1])
    print("   ε=%.3f  再現性 %.3f（腕側1割未満 %.3f）  旧 z との相関 %.3f  範囲 %+.2f〜%+.2f"
          % (e, r, rl, np.corrcoef(zf(G), old)[0, 1], zf(G).min(), zf(G).max()))
