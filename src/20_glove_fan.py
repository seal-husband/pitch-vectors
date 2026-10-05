"""
The glove-side continent in two coordinates, and pitchers compared on it.

Peaks of the glove-side continent (19: FC, SL, ST, SV, CU, KC form one connected region,
separated from the arm side by a real valley), read from the pitcher's primary fastball:

    θ  direction of the break away from the fastball: 0° = straight sideways (sweep),
       90° = straight down (depth)
    ρ  how far from the fastball, inches (in the movement plane; speed follows ρ and θ
       along the Pitch Plane, so it is not a third coordinate here)

Membership is geometric -- Δ横 ≥ 2 in and Δ縦 ≤ 2 in -- then cleaned of the few peaks whose
dominant label is an arm-side or fastball type (changeups of sinker-primary pitchers land
just glove-side of their sinker).

A pitcher's glove-side repertoire is his set of peaks with weights = share of HIS glove-side
pitches (so the comparison is of shape; how much he throws glove-side is kept separately).
Distance between two pitchers = Earth Mover's Distance between their weighted peak sets,
in inches: the least total inch-weight needed to move one repertoire onto the other.

Writes output/glove_data.js (the page it fed was superseded by 22_two_fans.py).

Usage:  python src/20_glove_fan.py
"""
import sys, os, json
from paths import SNAP
import numpy as np, pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.manifold import MDS
sys.stdout.reconfigure(encoding="utf-8")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UNITS = 60                               # weight resolution for the EMD
NOT_GLOVE = {"CH", "FS", "FO", "SI", "FF", "FA"}

Pk = pd.read_parquet(os.path.join(ROOT, "output", "peaks_2026.parquet"))
Pk["r0"] = np.sqrt(Pk.dhx ** 2 + Pk.dvz ** 2 + (Pk.dmph * .63) ** 2)
Pk = Pk.drop(index=Pk.groupby("pitcher").r0.idxmin().values)
G = Pk[(Pk.dhx >= 2) & (Pk.dvz <= 2) & ~Pk.label.isin(NOT_GLOVE)].copy()
G["theta"] = np.degrees(np.arctan2(-G.dvz, G.dhx))
G["rho"] = np.hypot(G.dhx, G.dvz)
G["gshare"] = G.share / G.groupby("pitcher").share.transform("sum")
glove_use = G.groupby("pitcher").share.sum()
print("グラブ側のピーク %d、投手 %d（グラブ側を持たない投手 %d）"
      % (len(G), G.pitcher.nunique(), Pk.pitcher.nunique() - G.pitcher.nunique()))
print("ピーク数/投手:", G.groupby("pitcher").size().value_counts().sort_index().to_dict())

pids = np.array(sorted(G.pitcher.unique()))
sets = {}
for p, g in G.groupby("pitcher"):
    u = np.round(g.gshare.values * UNITS).astype(int)
    u[np.argmax(u)] += UNITS - u.sum()
    sets[p] = np.repeat(g[["dhx", "dvz"]].values, u, axis=0)


def emd(a, b):
    C = np.linalg.norm(a[:, None] - b[None], axis=2)
    i, j = linear_sum_assignment(C)
    return C[i, j].mean()


N = len(pids)
Dm = np.zeros((N, N))
for i in range(N):
    for j in range(i + 1, N):
        Dm[i, j] = Dm[j, i] = emd(sets[pids[i]], sets[pids[j]])
print("EMD 中央 %.1f in   最近傍 中央 %.1f in" % (np.median(Dm[np.triu_indices(N, 1)]),
      np.median(np.sort(Dm, 1)[:, 1])))

xy = MDS(2, dissimilarity="precomputed", random_state=0, n_init=4, normalized_stress="auto").fit_transform(Dm)
# orient the map: x ~ mean θ (sweep → depth), y ~ mean ρ
mt = G.groupby("pitcher").apply(lambda g: np.average(g.theta, weights=g.gshare), include_groups=False).loc[pids].values
mr = G.groupby("pitcher").apply(lambda g: np.average(g.rho, weights=g.gshare), include_groups=False).loc[pids].values
for k, ref_ in ((0, mt), (1, mr)):
    if np.corrcoef(xy[:, k], ref_)[0, 1] < 0:
        xy[:, k] *= -1
print("地図の軸と平均 θ・平均 ρ の相関: x×θ %.2f  y×ρ %.2f  x×ρ %.2f  y×θ %.2f"
      % (np.corrcoef(xy[:, 0], mt)[0, 1], np.corrcoef(xy[:, 1], mr)[0, 1],
         np.corrcoef(xy[:, 0], mr)[0, 1], np.corrcoef(xy[:, 1], mt)[0, 1]))
stress_rank = np.corrcoef(Dm[np.triu_indices(N, 1)],
                          np.linalg.norm(xy[:, None] - xy[None], axis=2)[np.triu_indices(N, 1)])[0, 1]
print("地図上の距離と EMD の相関 %.2f" % stress_rank)

names = Pk.groupby("pitcher").name.first()
hand = pd.read_parquet(SNAP[2026],
                       columns=["pitcher", "p_throws"]).groupby("pitcher").p_throws.first()
out = {"league": {"x": np.round(G.dhx, 2).tolist(), "y": np.round(G.dvz, 2).tolist(),
                  "t": G.label.tolist(), "w": np.round(G.gshare, 3).tolist()},
       "pitchers": []}
for i, p in enumerate(pids):
    g = G[G.pitcher == p].sort_values("gshare", ascending=False)
    nn = np.argsort(Dm[i])[1:9]
    out["pitchers"].append({
        "id": int(p), "name": names[p], "hand": hand.get(p, "R"),
        "glove": round(float(glove_use[p]), 3), "mx": round(float(xy[i, 0]), 3), "my": round(float(xy[i, 1]), 3),
        "peaks": [{"x": round(r.dhx, 2), "y": round(r.dvz, 2), "th": round(r.theta, 1), "rho": round(r.rho, 1),
                   "mph": round(r.dmph, 1), "w": round(r.gshare, 3), "share": round(r.share, 3),
                   "t": r.label, "labels": r.labels} for r in g.itertuples()],
        "nn": [[int(j), round(float(Dm[i, j]), 2)] for j in nn]})
path = os.path.join(ROOT, "output", "glove_data.js")   # superseded by 22; kept out of web/
with open(path, "w", encoding="utf-8") as f:
    f.write("window.GLOVE=" + json.dumps(out, ensure_ascii=False, separators=(",", ":")) + ";")
print("wrote", path, "%.0f KB" % (os.path.getsize(path) / 1024))
for nm in ["Misiorowski, Jacob", "Sale, Chris", "Glasnow, Tyler", "Sugano, Tomoyuki"]:
    rec = next((q for q in out["pitchers"] if q["name"] == nm), None)
    if rec:
        print("  %-20s peaks %s  近い投手: %s" % (nm, [(q["t"], q["th"], q["rho"], q["w"]) for q in rec["peaks"]],
              ", ".join("%s(%.1f)" % (out["pitchers"][j]["name"], dd) for j, dd in rec["nn"][:4])))
