"""
The four numbers, computed one way everywhere (24, 25, 26, 27).

For every pitch other than the four-seam, with (fx, fz) its movement from the pitcher's
four-seam (converted from his sinker or cutter if he throws none):

  side   decided per PEAK, not per pitch: a peak (18_peak_table.py, found without labels)
         is glove side when its mean fx is at least GLOVE_X inches toward the glove side,
         arm side otherwise. Deciding pitch by pitch split a pitch type lying near the
         boundary in two (Tim Hill's sinker, whose four-seam sits almost on top of it).
         Pitches outside the peak table (knuckleballs, eephus...) fall back to their own fx.
  parts  per peak, horizontal = how far it goes toward its side (Σ fx, zero if the peak
         sits the other way), vertical = how far it moves up OR down (|Σ fz|).
         Counting only drop, and leaving out pitches that rise above the four-seam, made
         a submariner's rising slider vanish from the map (Tyler Rogers: 24% of pitches).

Each part is summed over the pitcher's peaks and divided by ALL his pitches, so it reads
"inches per pitch thrown".
"""
import os
import numpy as np, pandas as pd
from paths import OUTPUT

GLOVE_X = 2.0
KEY = ["game_pk", "at_bat_number", "pitch_number", "pitcher"]


def attach_peaks(raw, path=os.path.join(OUTPUT, "peak_assign_2026.parquet")):
    """Add column 'peak' (NaN for pitches not in the peak table) and 'side'."""
    A = pd.read_parquet(path)
    r = raw.copy()
    for c in KEY:
        r[c] = r[c].astype("int64")
    r = r.merge(A, on=KEY, how="left")
    r.index = raw.index
    g = r[(r.pitch_type != "FF") & r.peak.notna()]
    cen = g.groupby(["pitcher", "peak"]).fx.mean()
    side = pd.Series(np.where(cen >= GLOVE_X, "glove", "arm"), index=cen.index)
    r["side"] = [side.get((p, k), None) if k == k else None for p, k in zip(r.pitcher, r.peak)]
    lone = r.side.isna()
    r.loc[lone, "side"] = np.where(r.fx[lone] >= GLOVE_X, "glove", "arm")
    return r


def four_numbers(sub):
    """ah, av, gh, gv (inches per pitch), au, gu (shares), n, ffu from pitches carrying fx, fz, side, peak."""
    tot = sub.groupby("pitcher").size()
    g = sub[sub.pitch_type != "FF"].copy()
    # one unit per peak; a pitch with no peak is its own unit
    g["unit"] = np.where(g.peak.notna(), g.peak.fillna(-1).astype(int).astype(str), "p" + g.index.astype(str))
    u = g.groupby(["pitcher", "unit"]).agg(side=("side", "first"), sx=("fx", "sum"), sz=("fz", "sum"), k=("fx", "size"))
    u["h"] = np.where(u.side == "glove", u.sx.clip(lower=0), (-u.sx).clip(lower=0))
    u["v"] = u.sz.abs()
    rows = {}
    for p, w in u.groupby(level=0):
        n = tot[p]; a, gl = w[w.side == "arm"], w[w.side == "glove"]
        rows[p] = dict(ah=a.h.sum() / n, av=a.v.sum() / n, gh=gl.h.sum() / n, gv=gl.v.sum() / n,
                       au=a.k.sum() / n, gu=gl.k.sum() / n, n=n)
    out = pd.DataFrame(rows).T
    ffu = (sub.pitch_type == "FF").groupby(sub.pitcher).mean()
    out["ffu"] = ffu.reindex(out.index).fillna(0)
    out["n"] = out.n.astype(int)
    return out
