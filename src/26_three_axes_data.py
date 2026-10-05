"""
Data for the 3-D map: the four numbers as a composition, in log-ratio coordinates.

The four numbers (arm run, arm drop, glove sweep, glove drop; inches per pitch) are the
parts of one whole -- how a pitcher distributes the movement of everything that is not his
four-seam. Compared as a composition (Aitchison), with the partition {arm | glove}, the
three isometric log-ratio balances are

    x = ln(arm drop / arm run) / √2            + drops more than it runs
    y = ln(glove drop / glove sweep) / √2      + drops more than it sweeps
    z = ln(arm share / glove share) / √2       + throws more arm-side pitches
        share = that side's pitches ÷ all pitches (the four-seam counted in neither)

x and y are pure shape: usage cancels inside a side. z was first the ilr balance between the
sides, ln(g(arm) / g(glove)) with g the geometric mean of a side's two parts -- usage × movement.
That mixed in how far each side's pitches move from the four-seam and how balanced their two
parts are: Tim Hill, 83% arm-side, sat in the middle because his sinker moves little from the
(converted) four-seam and barely drops. z is now usage alone, as the article describes it
(2026-10-05). The two versions correlate 0.93 and neither relates to the Stuff stand-in.

All three are natural logs of ratios over √2, so the map can be drawn at one scale and
neighbours found by plain Euclidean distance. The composition's size -- how much non-four-seam
movement there is at all -- is left out on purpose, as the pitcher's plane height was: the
map describes how movement is distributed, not how much.

History of the choice (25, 28): x, y were first angles (unstable for a side barely thrown),
then differences (stable but mixing amount into direction); z was a difference, then an
angle. Log-ratios settle it in one geometry. Their zero problem -- a side not thrown -- is
handled the standard way, a small pseudo-amount DELTA added to every part before the ratio;
split-half reliability is then 0.995 / 0.998 / 0.994, and 0.97 for pitchers with under 10%
arm-side pitches, flat for DELTA from 0.05 to 1.0.

Three views, because a pitcher's mix depends on who is batting (changeups go to the
opposite side): all batters, vs right-handed batters, vs left-handed batters. Each view is
computed from that view's pitches only; a pitcher needs MIN_VIEW pitches in it to appear.
The four-seam reference and each peak's side are the pitcher's whole-season ones in
every view.

Writes web/three_data.js.

Usage:  python src/26_three_axes_data.py
"""
import sys, os, json, importlib.util
import numpy as np, pandas as pd
import fourparts
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
spec = importlib.util.spec_from_file_location("m24", os.path.join(ROOT, "src", "24_four_map_data.py"))
m24 = importlib.util.module_from_spec(spec); spec.loader.exec_module(m24)
F, info, kind, mix, raw = m24.F, m24.info, m24.kind, m24.mix, m24.raw

DELTA = .25                      # inches per pitch added to each part before the shape ratios
EPS = .02                        # share added to each side before the usage ratio (28: reliability vs EPS)


def to_map(ah, av, gh, gv, au, gu, delta=DELTA, eps=EPS):
    """The map's three log-ratios from the four numbers and the two sides' shares (arrays or scalars)."""
    ah, av, gh, gv = (np.clip(v, 0, None) + delta for v in (ah, av, gh, gv))
    return (np.log(av / ah) / np.sqrt(2), np.log(gv / gh) / np.sqrt(2),
            np.log((au + eps) / (gu + eps)) / np.sqrt(2))


F["xa"], F["yg"], F["z"] = to_map(F.ah, F.av, F.gh, F.gv, F.au, F.gu)
F["amt_a"] = np.hypot(F.ah, F.av)
F["amt_g"] = np.hypot(F.gh, F.gv)
MIN_VIEW = 150                   # pitches a pitcher needs in a batter-hand view to be placed in it


# the four numbers come from fourparts.py (side decided per peak, vertical = up or down);
# raw from 24 already carries each pitch's peak and side
four_numbers = fourparts.four_numbers


views = {"all": raw, "R": raw[raw.stand == "R"], "L": raw[raw.stand == "L"]}
V = {}
for k, sub in views.items():
    f = four_numbers(sub)
    f = f[f.n >= MIN_VIEW]
    f["x"], f["y"], f["z"] = to_map(f.ah, f.av, f.gh, f.gv, f.au, f.gu)
    mixv = sub.groupby("pitcher").pitch_type.value_counts(normalize=True)
    V[k] = (f, mixv)
    print("%-3s %d投手  x %+.2f±%.2f  y %+.2f±%.2f  z %+.2f±%.2f"
          % (k, len(f), f.x.mean(), f.x.std(), f.y.mean(), f.y.std(), f.z.mean(), f.z.std()))
assert np.allclose(V["all"][0].loc[F.index, "x"].values, F.xa.values)

out = []
for p in F.index:
    rec = {"id": int(p), "name": info.loc[p, "name"], "hand": info.loc[p, "hand"], "ref": kind[p], "v": {}}
    for k, (f, mixv) in V.items():
        if p not in f.index:
            continue
        r = f.loc[p]; m = mixv[p]
        rec["v"][k] = {"x": round(float(r.x), 3), "y": round(float(r.y), 3), "z": round(float(r.z), 3),
                       "use": [round(float(r.au), 3), round(float(r.gu), 3), round(float(r.ffu), 3)],
                       "n": int(r.n), "mix": [[t, round(float(v), 3)] for t, v in m[m >= .02].items()]}
    out.append(rec)
path = os.path.join(ROOT, "web", "three_data.js")
with open(path, "w", encoding="utf-8") as fh:
    fh.write("window.THREE=" + json.dumps({"pitchers": out, "delta": DELTA, "eps": EPS, "minView": MIN_VIEW},
                                         ensure_ascii=False, separators=(",", ":")) + ";")
print("%d投手  wrote %s (%.0f KB)" % (len(out), path, os.path.getsize(path) / 1024))
