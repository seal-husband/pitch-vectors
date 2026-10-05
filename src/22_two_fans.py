"""
Two fans, one standard: the whole arsenal read from the four-seam.

From the peak table in the four-seam frame (21):

  glove fan   FC SL ST SV CU KC      θ = direction of break away from the four-seam
                                     (0° sideways to the glove side, 90° straight down),
                                     ρ = distance, inches. Speed follows θ and ρ along the
                                     Pitch Plane, so two coordinates suffice.
  arm fan     SI CH FS FO            θ = direction (0° straight to the arm side, 90° down),
                                     Δv = mph slower than the four-seam. Here speed is the
                                     coordinate that matters: a sinker and a "running"
                                     changeup can share a direction and differ only in
                                     speed; the polar pair separated the arm-side types at
                                     silhouette +0.02 against +0.25 for (θ, Δv).

Membership is by position (glove: Δ横 ≥ 2 in, arm: Δ横 < 2 in, both Δ縦 ≤ 2 in), cleaned
by the peak's dominant label where a pitch crosses over. Peaks elsewhere (a second
fastball riding above the four-seam, say) stay in the whole-arsenal comparison only.

Distances: Earth Mover's Distance in the physical frame (Δ横, Δ縦, Δ球速 × 0.63 inches),
  side    peaks of one fan, weights normalised within the fan (shape of that side)
  whole   every peak, with the four-seam itself as a peak at the origin, weights = share
          of all pitches (so usage counts too)

Writes web/fans_data.js.

Usage:  python src/22_two_fans.py
"""
import sys, os, json
from paths import SNAP
import numpy as np, pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.manifold import MDS
sys.stdout.reconfigure(encoding="utf-8")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UNITS = 60
GLOVE_T = {"FC", "SL", "ST", "SV", "CU", "KC", "CS"}
ARM_T = {"SI", "CH", "FS", "FO"}

Pk = pd.read_parquet(os.path.join(ROOT, "output", "peaks_ff_2026.parquet"))
Pk["fvs"] = Pk.fmph * .63
low = Pk.fz <= 2
Pk["side"] = np.where(low & (Pk.fx >= 2) & Pk.label.isin(GLOVE_T), "glove",
             np.where(low & (Pk.fx < 2) & Pk.label.isin(ARM_T), "arm", "other"))
Pk.loc[Pk.side == "glove", "theta"] = np.degrees(np.arctan2(-Pk.fz, Pk.fx))
Pk.loc[Pk.side == "arm", "theta"] = np.degrees(np.arctan2(-Pk.fz, -Pk.fx))
Pk["rho"] = np.hypot(Pk.fx, Pk.fz)
print("ピーク: グラブ側 %d / 腕側 %d / その他 %d" % tuple((Pk.side == s).sum() for s in ["glove", "arm", "other"]))
print("その他の中身:", Pk[Pk.side == "other"].label.value_counts().head(6).to_dict())
for s, cols in (("glove", ["theta", "rho"]), ("arm", ["theta", "fmph"])):
    print("■ %s" % s)
    print(Pk[Pk.side == s].groupby("label")[cols].median().round(1).to_string())

pids = np.array(sorted(Pk.pitcher.unique()))
ff_share = (1 - Pk.groupby("pitcher").share.sum()).clip(lower=0)
kind = Pk.groupby("pitcher").ref_kind.first()


def units(w):
    u = np.round(np.asarray(w) / np.sum(w) * UNITS).astype(int)
    u[np.argmax(u)] += UNITS - u.sum()
    return u


def point_set(p, side):
    g = Pk[Pk.pitcher == p]
    if side != "whole":
        g = g[g.side == side]
        if g.empty:
            return None
        return np.repeat(g[["fx", "fz", "fvs"]].values, units(g.share), axis=0)
    pts, w = g[["fx", "fz", "fvs"]].values, g.share.values
    if kind[p] == "FF" and ff_share[p] > 0:
        pts = np.vstack([pts, [0, 0, 0]]); w = np.r_[w, ff_share[p]]
    return np.repeat(pts, units(w), axis=0)


def emd(a, b):
    C = np.linalg.norm(a[:, None] - b[None], axis=2)
    i, j = linear_sum_assignment(C)
    return C[i, j].mean()


res = {}
for side in ["glove", "arm", "whole"]:
    sets = {p: point_set(p, side) for p in pids}
    have = np.array([p for p in pids if sets[p] is not None])
    N = len(have)
    Dm = np.zeros((N, N))
    for i in range(N):
        for j in range(i + 1, N):
            Dm[i, j] = Dm[j, i] = emd(sets[have[i]], sets[have[j]])
    res[side] = (have, Dm)
    print("%-5s 投手 %d   EMD 中央 %.1f in   最近傍 中央 %.1f in"
          % (side, N, np.median(Dm[np.triu_indices(N, 1)]), np.median(np.sort(Dm, 1)[:, 1])))

have, Dw = res["whole"]
xy = MDS(2, dissimilarity="precomputed", random_state=0, n_init=4, normalized_stress="auto").fit_transform(Dw)
fit = np.corrcoef(Dw[np.triu_indices(len(have), 1)],
                  np.linalg.norm(xy[:, None] - xy[None], axis=2)[np.triu_indices(len(have), 1)])[0, 1]
print("持ち球全体の地図：地図上の距離と EMD の相関 %.2f" % fit)
# name the map's axes by what they correlate with
glove_use = Pk[Pk.side == "glove"].groupby("pitcher").share.sum().reindex(have).fillna(0).values
arm_use = Pk[Pk.side == "arm"].groupby("pitcher").share.sum().reindex(have).fillna(0).values
ffs = ff_share.reindex(have).fillna(0).values
for k in range(2):
    print("  地図の軸%d × グラブ側の割合 %+.2f  腕側の割合 %+.2f  FFの割合 %+.2f"
          % (k + 1, np.corrcoef(xy[:, k], glove_use)[0, 1], np.corrcoef(xy[:, k], arm_use)[0, 1],
             np.corrcoef(xy[:, k], ffs)[0, 1]))
if np.corrcoef(xy[:, 0], glove_use)[0, 1] < 0:
    xy[:, 0] *= -1
if np.corrcoef(xy[:, 1], arm_use)[0, 1] < 0:
    xy[:, 1] *= -1

hand = pd.read_parquet(SNAP[2026],
                       columns=["pitcher", "p_throws"]).groupby("pitcher").p_throws.first()
idx = {side: {p: i for i, p in enumerate(res[side][0])} for side in res}
out = {"league": {s: Pk[Pk.side == s][["fx", "fz", "fmph", "theta", "rho", "share", "label"]]
                  .round(2).to_dict(orient="list") for s in ["glove", "arm"]}, "pitchers": []}
wi = {p: i for i, p in enumerate(have)}
for p in have:
    g = Pk[Pk.pitcher == p].sort_values("share", ascending=False)
    rec = {"id": int(p), "name": g.name.iloc[0], "hand": hand.get(p, "R"), "ref": kind[p],
           "ff": round(float(ff_share[p]), 3) if kind[p] == "FF" else None,
           "mx": round(float(xy[wi[p], 0]), 3), "my": round(float(xy[wi[p], 1]), 3),
           "peaks": [{"side": r.side, "t": r.label, "labels": r.labels, "fx": round(r.fx, 2), "fz": round(r.fz, 2),
                      "mph": round(r.fmph, 1), "th": None if np.isnan(r.theta) else round(r.theta, 1),
                      "rho": round(r.rho, 1), "share": round(r.share, 3)} for r in g.itertuples()],
           "nn": {}}
    for side in ["whole", "glove", "arm"]:
        if p in idx[side]:
            i = idx[side][p]; Dm = res[side][1]
            nn = np.argsort(Dm[i])[1:9]
            rec["nn"][side] = [[int(res[side][0][j]), round(float(Dm[i, j]), 2)] for j in nn]
    out["pitchers"].append(rec)
path = os.path.join(ROOT, "web", "fans_data.js")
with open(path, "w", encoding="utf-8") as f:
    f.write("window.FANS=" + json.dumps(out, ensure_ascii=False, separators=(",", ":")) + ";")
print("wrote", path, "%.0f KB" % (os.path.getsize(path) / 1024))
names = {r["id"]: r["name"] for r in out["pitchers"]}
for nm in ["Sale, Chris", "Misiorowski, Jacob", "Sugano, Tomoyuki", "Skubal, Tarik"]:
    r = next(q for q in out["pitchers"] if q["name"] == nm)
    print("  %-20s arm: %s" % (nm, [(q["t"], q["th"], q["mph"]) for q in r["peaks"] if q["side"] == "arm"]))
    for side in ["whole", "arm"]:
        if side in r["nn"]:
            print("      近い（%s）: %s" % (side, ", ".join("%s(%.1f)" % (names[j], d) for j, d in r["nn"][side][:4])))
