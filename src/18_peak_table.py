"""
Peak table: pick each pitcher's peaks without labels, then line them up across pitchers.

The metabolomics answer to peaks that shift from sample to sample: detect and integrate
peaks in each sample first, align them afterwards. Here a sample is a pitcher, the
spectrum is his pitches on the Pitch Plane read from his own primary fastball
(16_mcr: the best internal standard), and a peak is a pitch he actually throws.

Peaks are picked in 3-D, not on the plane: a first version on the 2-D Pitch Plane merged
changeups into sinkers (Lodolo: FF, SI and CH as one peak), because what separates them is
mostly speed, and most of that speed difference lies along the plane's normal.

  space     (Δ横, Δ縦, Δ球速 × 0.63) from the pitcher's primary fastball, inches
  modes     mean shift with a Gaussian kernel of bandwidth BW: every mode of the density
  merging   saddles are taken on the segment between modes; regions are joined in order
            of falling saddle, and two regions stay apart when the saddle is at or below
            VALLEY × the lower of their two summits (persistence; a first version compared
            against single modes and let a small mode chain FF, SI and SL together)
  area      each pitch belongs to its mode's peak; peaks holding fewer than MIN_SHARE of
            the pitcher's pitches are folded into the nearest kept peak

Checks: agreement with Statcast labels (adjusted Rand, peak count vs label count), and
split-half reproducibility (alternate pitches) of the peak count.

Writes output/peaks_2026.parquet -- one row per (pitcher, peak).

Usage:  python src/18_peak_table.py [BW] [VALLEY]
"""
from __future__ import annotations
import sys, os
from paths import SNAP
import numpy as np, pandas as pd
from scipy.ndimage import gaussian_filter
from sklearn.metrics import adjusted_rand_score
from sklearn.cluster import MeanShift
sys.stdout.reconfigure(encoding="utf-8")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

BW = float(sys.argv[1]) if len(sys.argv) > 1 else 1.5
VALLEY = float(sys.argv[2]) if len(sys.argv) > 2 else .7
STEP = .5
MIN_SHARE, MIN_N = .03, 30
PAD = 4.


def kde(pts, at, bw):
    d2 = ((at[:, None, :] - pts[None, :, :]) ** 2).sum(-1)
    return np.exp(-d2 / (2 * bw * bw)).sum(1)


def pick(xy, bw=None, valley=None):
    """Return a peak label per point (0..k-1)."""
    bw = BW if bw is None else bw
    valley = VALLEY if valley is None else valley
    ms = MeanShift(bandwidth=bw * 2, bin_seeding=True, min_bin_freq=5, cluster_all=True).fit(xy)
    cen, lab = ms.cluster_centers_, ms.labels_.copy()
    k = len(cen)
    h = kde(xy, cen, bw)
    # saddle between every pair of modes = lowest density on the segment joining them
    ts = np.linspace(0, 1, 25)[1:-1]
    pairs = []
    for a in range(k):
        for b in range(a + 1, k):
            seg = cen[a][None] + ts[:, None] * (cen[b] - cen[a])[None]
            pairs.append((min(kde(xy, seg, bw).min(), h[a], h[b]), a, b))
    # persistence: join regions in order of falling saddle; two regions stay apart when
    # the saddle is at or below VALLEY x the lower of their two summits. Processing by
    # falling saddle stops a small mode from chaining two big peaks together.
    root = list(range(k)); top = h.copy()
    def find(a):
        while root[a] != a:
            a = root[a]
        return a
    for sd, a, b in sorted(pairs, reverse=True):
        ra, rb = find(a), find(b)
        if ra == rb:
            continue
        if sd > valley * min(top[ra], top[rb]):
            hi_, lo_ = (ra, rb) if top[ra] >= top[rb] else (rb, ra)
            root[lo_] = hi_
    pl = np.array([find(c) for c in lab])
    # fold small peaks into the nearest large one
    u, cnt = np.unique(pl, return_counts=True)
    small = u[(cnt < MIN_N) | (cnt < MIN_SHARE * len(pl))]
    big = u[~np.isin(u, small)]
    if len(big) and len(small):
        cenb = np.array([xy[pl == b].mean(0) for b in big])
        m = np.isin(pl, small)
        pl[m] = big[np.linalg.norm(xy[m][:, None] - cenb[None], axis=2).argmin(1)]
    u = np.unique(pl)
    return np.searchsorted(u, pl)


def main():
    d = pd.read_parquet(os.path.join(ROOT, "output", "mcr_pitches_2026.parquet"))
    raw = pd.read_parquet(SNAP[2026],
                          columns=["game_type", "pitcher", "pitch_type", "p_throws", "pfx_x", "pfx_z", "release_speed",
                                   "game_pk", "at_bat_number", "pitch_number"])
    raw = raw[(raw.game_type == "R") & raw.pitch_type.isin(set(d.pitch_type))].dropna()
    raw = raw[raw.pitcher.isin(set(d.pitcher))].reset_index(drop=True)
    assert len(raw) == len(d) and (raw.pitch_type.values == d.pitch_type.values).all()
    d = d.reset_index(drop=True)
    d["hx"] = np.where(raw.p_throws.eq("L"), -raw.pfx_x.astype(float), raw.pfx_x.astype(float)) * 12
    d["vz"] = raw.pfx_z.astype(float) * 12
    d["vs"] = raw.release_speed.astype(float) * .63
    f = d[d.pitch_type.isin(["FF", "SI", "FC"])]
    top = f.groupby("pitcher").pitch_type.agg(lambda s_: s_.value_counts().index[0])
    ref = f[f.pitch_type.values == f.pitcher.map(top).values].groupby("pitcher")[["hx", "vz", "vs"]].mean()
    for c in ["hx", "vz", "vs"]:
        d["d" + c] = d[c] - d.pitcher.map(ref[c])
    print("帯域幅 %.1f in、谷の基準 %.2f（鞍点がピークの %.0f%% 以下なら別ピーク）"
          % (BW, VALLEY, 100 * VALLEY))
    rows, stats_, assign = [], [], []
    for pid, g in d.groupby("pitcher", sort=False):
        xy = g[["dhx", "dvz", "dvs"]].values
        lab = pick(xy)
        assign.append(pd.DataFrame({"row": g.index.values, "peak": lab}))
        vc = g.pitch_type.value_counts()
        ntype = int(((vc / len(g) >= MIN_SHARE) & (vc >= MIN_N)).sum())
        ha = pick(xy[g.half.values == 0]); hb = pick(xy[g.half.values == 1])
        stats_.append((pid, lab.max() + 1, ntype, adjusted_rand_score(g.pitch_type, lab),
                       ha.max() + 1, hb.max() + 1))
        for k in range(lab.max() + 1):
            m = lab == k
            t = g.pitch_type[m].value_counts(normalize=True)
            cov = np.cov(xy[m].T)
            rows.append(dict(pitcher=pid, name=g.player_name.iloc[0], peak=k, n=int(m.sum()),
                             share=m.mean(), dhx=xy[m, 0].mean(), dvz=xy[m, 1].mean(),
                             dmph=xy[m, 2].mean() / .63, ur=g.ur[m].mean(), wr=g.wr[m].mean(),
                             u=g.u[m].mean(), w=g.w[m].mean(),
                             sd_major=np.sqrt(np.linalg.eigvalsh(cov)[-1]),
                             sd_minor=np.sqrt(np.linalg.eigvalsh(cov)[0]),
                             label=t.index[0], purity=t.iloc[0],
                             labels=" ".join("%s%.0f" % (a, 100 * b) for a, b in t.head(3).items())))
    St = pd.DataFrame(stats_, columns=["pitcher", "peaks", "types", "ari", "ha", "hb"])
    Pk = pd.DataFrame(rows)
    print("\n■ ピーク検出の成績（%d投手、%dピーク）" % (len(St), len(Pk)))
    print("   ピーク数 中央 %d（平均 %.1f）   ラベル上の球種数 中央 %d（平均 %.1f）"
          % (St.peaks.median(), St.peaks.mean(), St.types.median(), St.types.mean()))
    diff = St.peaks - St.types
    print("   ピーク数 − 球種数:  " + "  ".join("%+d: %.0f%%" % (k, 100 * v) for k, v in
          diff.clip(-3, 3).value_counts(normalize=True).sort_index().items()))
    print("   ラベルとの一致（ARI）中央 %.2f   四分位 %.2f 〜 %.2f" % (St.ari.median(), *St.ari.quantile([.25, .75])))
    print("   ピークの純度（最多ラベルの割合）中央 %.2f   0.8以上のピーク %.0f%%"
          % (Pk.purity.median(), 100 * (Pk.purity >= .8).mean()))
    print("   折半の再現：二つの半分でピーク数が一致 %.0f%%、±1以内 %.0f%%"
          % (100 * (St.ha == St.hb).mean(), 100 * ((St.ha - St.hb).abs() <= 1).mean()))
    print("\n■ ピークが複数のラベルを抱える例（純度の低い順、使用率10%以上）")
    print(Pk[Pk.share >= .1].nsmallest(10, "purity")[["name", "share", "dhx", "dvz", "dmph", "labels"]]
          .round(2).to_string(index=False))
    print("\n■ 一つのラベルが複数のピークに割れる例")
    sp_ = Pk.groupby(["pitcher", "label"]).size()
    many = sp_[sp_ >= 2].reset_index()
    print("   同じ投手で最多ラベルが同じピークが2つ以上: %d件（投手 %d人）　ラベル別: %s"
          % (len(many), many.pitcher.nunique(), many.label.value_counts().head(6).to_dict()))
    tag = "" if (BW, VALLEY) == (1.5, .7) else "_bw%.1f_v%.2f" % (BW, VALLEY)
    Pk.to_parquet(os.path.join(ROOT, "output", "peaks_2026%s.parquet" % tag), index=False)
    St.to_csv(os.path.join(ROOT, "output", "peak_stats_2026%s.csv" % tag), index=False)
    # which peak each pitch belongs to, keyed by the pitch -- the four numbers (fourparts.py)
    # decide arm side / glove side per peak, so a pitch type is never split by the boundary
    A = pd.concat(assign, ignore_index=True)
    keys = raw.loc[A.row.values, ["game_pk", "at_bat_number", "pitch_number", "pitcher"]].reset_index(drop=True)
    keys["peak"] = A.peak.values
    for c in ["game_pk", "at_bat_number", "pitch_number", "pitcher"]:
        keys[c] = keys[c].astype("int64")
    keys.to_parquet(os.path.join(ROOT, "output", "peak_assign_2026%s.parquet" % tag), index=False)
    print("wrote output/peak_assign_2026%s.parquet (%d pitches)" % (tag, len(keys)))


if __name__ == "__main__":
    main()
