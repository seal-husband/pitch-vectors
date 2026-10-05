"""
Everything the sphere page draws, computed once and written as a JS file.

The page cannot fetch (artifact sandbox), so the data travels as `window.SPHERE=...`
loaded by a <script src>. Nothing here is new analysis -- it re-derives the numbers
03 printed, in shapes a plot can use:

  cloud     a random sample of pitches, as true vectors (inches) and map coordinates
  raster    cap density on a pixel grid of the whole-sphere Lambert disk
  modes     the density maxima, with what is inside them
  paths     the great circles 03 walked between modes, projected, with their profiles
  lengths   |v| histograms for the pairs a sphere cannot tell apart
  angles    the angle matrix between type mean directions

The map is ONE disk holding the WHOLE sphere: the Lambert projection is equal-area all
the way to the antipode, which becomes the rim (radius 2). The centre is picked to put as
many pitches as possible inside the inner hemisphere (radius sqrt 2).

Usage:  python src/04_viz_data.py [season]
"""
from __future__ import annotations
import sys
sys.stdout.reconfigure(encoding="utf-8")
import json
import os
import importlib.util

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_spec = importlib.util.spec_from_file_location("sph", os.path.join(ROOT, "src", "03_sphere.py"))
sph = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sph)

SHOW = ["FF", "SI", "FC", "SL", "ST", "SV", "CU", "KC", "CH", "FS"]
N_CLOUD = 16000
RASTER = 170
SEED = 3


def basis(c):
    """Map frame: x follows the 横 axis as closely as the centre allows."""
    c = c / np.linalg.norm(c)
    e1 = np.array([1., 0., 0.]) - c[0] * c
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(c, e1)
    if e2[1] < 0:                   # keep 縦 pointing up on the page
        e2 = -e2
    return c, e1, e2


def project(U, c, e1, e2):
    cos = np.clip(U @ c, -1, 1)
    r = 2 * np.sin(np.arccos(cos) / 2)
    ang = np.arctan2(U @ e2, U @ e1)
    return np.c_[r * np.cos(ang), r * np.sin(ang)]


def unproject(xy, c, e1, e2):
    r = np.hypot(xy[:, 0], xy[:, 1])
    th = 2 * np.arcsin(np.clip(r / 2, 0, 1))
    ph = np.arctan2(xy[:, 1], xy[:, 0])
    return (np.cos(th)[:, None] * c
            + np.sin(th)[:, None] * (np.cos(ph)[:, None] * e1 + np.sin(ph)[:, None] * e2))


def main():
    year = int(sys.argv[1]) if len(sys.argv) > 1 else 2026
    rng = np.random.default_rng(SEED)
    d = sph.load(year)
    U, L, v0 = sph.vectors_E(d)
    pt = d.pitch_type.values
    V = U * L[:, None]

    take = rng.choice(len(U), min(90000, len(U)), replace=False)
    Us = U[take].astype(np.float32)
    cos_r = 1 - 2 * sph.CAP_FRACTION

    # ---- centre: the direction that keeps the most pitches in the inner hemisphere
    grid = sph.fibonacci_sphere(2000)
    inside = (grid.astype(np.float32) @ Us.T >= 0).mean(1)
    c, e1, e2 = basis(grid[np.argmax(inside)])
    print("中心 (%+.2f, %+.2f, %+.2f)  内側の半球に %.1f%%" % (*c, 100 * inside.max()))

    # ---- density raster over the whole disk
    ax = np.linspace(-2, 2, RASTER)
    X, Y = np.meshgrid(ax, ax)
    xy = np.c_[X.ravel(), Y.ravel()]
    ok = np.hypot(xy[:, 0], xy[:, 1]) <= 2
    D = np.full(len(xy), np.nan)
    D[ok] = sph.cap_density(unproject(xy[ok], c, e1, e2).astype(np.float32), Us, cos_r)
    peak = np.nanmax(D)
    D = D / peak

    # ---- modes (same rule as 03)
    g4 = sph.fibonacci_sphere(4000)
    dens = sph.cap_density(g4.astype(np.float32), Us, cos_r)
    NEIGH = np.cos(np.radians(25))
    modes = []
    for i in np.argsort(dens)[::-1]:
        if all(g4[i] @ g4[j] < NEIGH for j in modes):
            modes.append(i)
        if len(modes) >= 12:
            break
    modes = [i for i in modes if dens[i] >= .2 * dens[modes[0]]]
    mode_out = []
    for k, i in enumerate(modes):
        near = pt[take][Us @ g4[i].astype(np.float32) >= cos_r]
        lab = pd.Series(near).value_counts(normalize=True)
        p = project(g4[i][None], c, e1, e2)[0]
        mode_out.append({"k": k + 1, "x": round(p[0], 4), "y": round(p[1], 4),
                         "rel": round(dens[i] / dens[modes[0]], 3),
                         "mix": [[t, round(v, 3)] for t, v in lab.head(3).items()]})

    # ---- great-circle paths between the top four modes, with profiles
    paths = []
    top = modes[:4]
    for a in range(len(top)):
        for b in range(a + 1, len(top)):
            ga, gb = g4[top[a]], g4[top[b]]
            ts = np.linspace(0, 1, 61)
            P = np.array([(1 - t) * ga + t * gb for t in ts])
            P /= np.linalg.norm(P, axis=1, keepdims=True)
            dp = sph.cap_density(P.astype(np.float32), Us, cos_r)
            ref = min(dp[0], dp[-1])
            xyP = project(P, c, e1, e2)
            paths.append({"a": a + 1, "b": b + 1,
                          "sep": round(float(np.degrees(np.arccos(np.clip(ga @ gb, -1, 1)))), 1),
                          "dip": round(float(dp[4:-4].min() / ref), 3),
                          "x": np.round(xyP[:, 0], 4).tolist(),
                          "y": np.round(xyP[:, 1], 4).tolist(),
                          "prof": np.round(dp / ref, 3).tolist()})

    # ---- type centroids
    cent = {}
    for t in SHOW:
        m = pt == t
        u = U[m].mean(0); u /= np.linalg.norm(u)
        p = project(u[None], c, e1, e2)[0]
        cent[t] = {"u": np.round(u, 4).tolist(), "x": round(p[0], 4), "y": round(p[1], 4),
                   "L": round(float(np.median(L[m])), 2), "n": int(m.sum())}
    ang = [[round(float(np.degrees(np.arccos(np.clip(np.dot(cent[a]["u"], cent[b]["u"]), -1, 1)))), 1)
            for b in SHOW] for a in SHOW]

    # ---- length histograms
    edges = np.arange(0, 30.5, .5)
    lengths = {}
    for t in SHOW:
        h, _ = np.histogram(L[pt == t], bins=edges)
        lengths[t] = np.round(h / h.sum() / .5, 4).tolist()

    # ---- cloud sample
    keep = np.flatnonzero(np.isin(pt, SHOW))
    s = rng.choice(keep, min(N_CLOUD, len(keep)), replace=False)
    mp = project(U[s], c, e1, e2)
    code = {t: i for i, t in enumerate(SHOW)}
    cloud = {"t": [code[t] for t in pt[s]],
             "x": np.round(V[s, 0], 1).tolist(), "y": np.round(V[s, 2], 1).tolist(),
             "z": np.round(V[s, 1], 1).tolist(),
             "mx": np.round(mp[:, 0], 3).tolist(), "my": np.round(mp[:, 1], 3).tolist()}

    # ---- where the three physical axes land on the map
    axes = {}
    for nm, vec in (("横+", [1, 0, 0]), ("横−", [-1, 0, 0]), ("縦+", [0, 1, 0]),
                    ("縦−", [0, -1, 0]), ("速+", [0, 0, 1]), ("遅", [0, 0, -1])):
        p = project(np.array(vec, float)[None], c, e1, e2)[0]
        axes[nm] = [round(p[0], 4), round(p[1], 4)]

    # ---- the common plane from 08, if it has been fitted
    plane = None
    pf = os.path.join(ROOT, "output", "plane_%d.npz" % year)
    if os.path.exists(pf):
        z = np.load(pf)
        plane = {"n": np.round(z["normal"], 4).tolist(), "h": round(float(z["height"]), 2),
                 "e1": np.round(z["axes"][:, 0], 4).tolist(),
                 "e2": np.round(z["axes"][:, 1], 4).tolist(),
                 "sd": np.round(np.sqrt(z["var"]), 2).tolist()}

    out = {"year": year, "plane": plane, "pitchers": int(d.pitcher.nunique()), "pitches": int(len(d)),
           "v0": round(float(v0), 1), "s": sph.S_IN_PER_MPH,
           "capDeg": round(float(np.degrees(np.arccos(cos_r))), 1),
           "centre": np.round(c, 4).tolist(), "innerShare": round(float(inside.max()), 4),
           "types": SHOW, "raster": {"n": RASTER,
                                     "z": [None if np.isnan(v) else round(float(v), 3) for v in D]},
           "modes": mode_out, "paths": paths, "cent": cent, "angles": ang,
           "lenEdges": edges.tolist(), "lengths": lengths, "cloud": cloud, "axes": axes}
    web = os.path.join(ROOT, "web")
    os.makedirs(web, exist_ok=True)
    with open(os.path.join(web, "sphere_data.js"), "w", encoding="utf-8") as f:
        f.write("window.SPHERE=" + json.dumps(out, ensure_ascii=False, separators=(",", ":")) + ";")
    print("wrote web/sphere_data.js  (%.0f KB)"
          % (os.path.getsize(os.path.join(web, "sphere_data.js")) / 1024))
    for p in paths:
        print("  %d⇔%d  %.1f°  谷 %.2f" % (p["a"], p["b"], p["sep"], p["dip"]))


if __name__ == "__main__":
    main()
