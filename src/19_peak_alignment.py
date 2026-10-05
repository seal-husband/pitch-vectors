"""
Align the peak table: do pitchers' peaks fall into a few places, or fill a continuum?

A pitch type that is a real species should behave like a compound in a chemical-shift
table: every pitcher's peak for it lands in about the same place, and the places are
separated by empty space. A continuum fills the space between.

The earlier test (01, 03) found no valleys between named types among single PITCHES. That
test cannot tell apart two explanations: (a) the types really are a continuum, or (b) each
pitcher's spread and the pitcher-to-pitcher shifts smear discrete types together. Peaks
remove the within-pitcher spread (each peak is one point), so if (b) is true, valleys
should open up at the peak level.

Space: (Δ横, Δ縦, Δ球速 × 0.63) from the pitcher's primary fastball, inches. The peak
nearest the origin (the fastball itself) is left out.

Valley depth along the segment between two label centroids, 1.0 = no valley, compared for
pitch-level density and peak-level density at the same kernel width, with a bootstrap over
pitchers for the peak level.

Usage:  python src/19_peak_alignment.py [MIN_PURITY]
"""
import sys, os
from paths import SNAP
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.stdout.reconfigure(encoding="utf-8")
plt.rcParams["font.family"] = ["Yu Gothic", "Meiryo", "MS Gothic", "sans-serif"]
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COL = {"FF": "#D22D49", "SI": "#FE9D00", "FC": "#933F2C", "SL": "#EEE716", "ST": "#DDB33A",
       "SV": "#93AFD4", "CU": "#00D1ED", "KC": "#6236CD", "CS": "#0068FF", "CH": "#1DBE3A",
       "FS": "#3BACAC", "FO": "#55CCAB", "FA": "#888888", "KN": "#3C44CD"}
BW = 2.0
N_BOOT = 200
rng = np.random.default_rng(19)

Pk = pd.read_parquet(os.path.join(ROOT, "output", "peaks_2026.parquet"))
Pk["dvs"] = Pk.dmph * .63
Pk["r0"] = np.sqrt(Pk.dhx ** 2 + Pk.dvz ** 2 + Pk.dvs ** 2)
ref = Pk.groupby("pitcher").r0.idxmin()
Q = Pk.drop(index=ref.values).copy()
# robustness: a peak that merged several labelled pitches sits BETWEEN them and could fill a
# valley by construction; MIN_PURITY > 0 keeps only peaks dominated by one label
MIN_PURITY = float(sys.argv[1]) if len(sys.argv) > 1 else 0.
Q = Q[Q.purity >= MIN_PURITY]
print("ピーク %d（自分の速球のピーク %d を除く、純度 %.2f 以上）、投手 %d"
      % (len(Q), len(ref), MIN_PURITY, Q.pitcher.nunique()))

# pitch-level cloud in the same frame, for comparison: a sample of pitches, with each
# pitcher's primary fastball left out as the reference peak is left out above
d = pd.read_parquet(os.path.join(ROOT, "output", "mcr_pitches_2026.parquet"))
raw = pd.read_parquet(SNAP[2026],
                      columns=["game_type", "pitcher", "pitch_type", "p_throws", "pfx_x", "pfx_z", "release_speed"])
raw = raw[(raw.game_type == "R") & raw.pitch_type.isin(set(d.pitch_type))].dropna()
raw = raw[raw.pitcher.isin(set(d.pitcher))].reset_index(drop=True)
d = d.reset_index(drop=True)
d["hx"] = np.where(raw.p_throws.eq("L"), -raw.pfx_x.astype(float), raw.pfx_x.astype(float)) * 12
d["vz"] = raw.pfx_z.astype(float) * 12
d["vs"] = raw.release_speed.astype(float) * .63
f = d[d.pitch_type.isin(["FF", "SI", "FC"])]
top = f.groupby("pitcher").pitch_type.agg(lambda s: s.value_counts().index[0])
refm = f[f.pitch_type.values == f.pitcher.map(top).values].groupby("pitcher")[["hx", "vz", "vs"]].mean()
for c in ["hx", "vz", "vs"]:
    d["d" + c] = d[c] - d.pitcher.map(refm[c])
d = d[d.pitch_type.values != d.pitcher.map(top).values]
d = d.sample(60000, random_state=1)


def kde(pts, at, bw=BW):
    out = np.zeros(len(at))
    for i in range(0, len(pts), 5000):
        p = pts[i:i + 5000]
        out += np.exp(-((at[:, None, :] - p[None]) ** 2).sum(-1) / (2 * bw * bw)).sum(1)
    return out


def valley(pts, a, b):
    ts = np.linspace(0, 1, 41)
    seg = a[None] + ts[:, None] * (b - a)[None]
    f_ = kde(pts, seg)
    return f_[3:-3].min() / min(f_[:4].max(), f_[-4:].max())


P3 = Q[["dhx", "dvz", "dvs"]].values
D3 = d[["dhx", "dvz", "dvs"]].values
cent = {t: Q.loc[Q.label == t, ["dhx", "dvz", "dvs"]].mean().values for t in
        ["FC", "SL", "ST", "SV", "CU", "KC", "CH", "FS", "SI", "FF"] if (Q.label == t).sum() >= 10}
pairs = [("SL", "CU"), ("SL", "ST"), ("ST", "CU"), ("SL", "FC"), ("SV", "CU"), ("SV", "SL"),
         ("CH", "FS"), ("CU", "KC"), ("SI", "CH"), ("FC", "FF"),
         ("CH", "SL"), ("SI", "SL"), ("FS", "CU"), ("CH", "FC")]
print("\n■ 名前つき球種の重心を結ぶ線上の谷（1.0＝谷なし、小さいほど深い谷）  カーネル幅 %.1f in" % BW)
print("   %-8s %6s %8s %20s" % ("", "一球", "ピーク", "ピーク 95%区間"))
pitch_ids = Q.pitcher.unique()
groups = {p: g[["dhx", "dvz", "dvs"]].values for p, g in Q.groupby("pitcher")}
for a, b in pairs:
    if a not in cent or b not in cent:
        continue
    v_pitch = valley(D3, cent[a], cent[b])
    v_peak = valley(P3, cent[a], cent[b])
    bs = []
    for _ in range(N_BOOT):
        pick = rng.choice(pitch_ids, len(pitch_ids))
        bs.append(valley(np.vstack([groups[p] for p in pick]), cent[a], cent[b]))
    lo, hi = np.quantile(bs, [.025, .975])
    print("   %-3s⇔%-3s %6.2f %8.2f      [%.2f, %.2f]" % (a, b, v_pitch, v_peak, lo, hi))

# ---- figure: peak positions, front (Δ横, Δ縦) and side (Δ球速, Δ縦)
fig, axs = plt.subplots(1, 2, figsize=(14, 6.6), sharey=True)
for ax, (xk, xl) in zip(axs, [("dhx", "速球との横変化の差 in（− 腕側 ／ ＋ グラブ側）"), ("dmph", "速球との球速差 mph")]):
    for t, g in Q.groupby("label"):
        ax.scatter(g[xk], g.dvz, s=12 + 160 * g.share, c=COL.get(t, "#888"), edgecolors="#18202b",
                   linewidths=.3, alpha=.8, label="%s (%d)" % (t, len(g)))
    ax.axhline(0, color="#ccc", lw=.6); ax.axvline(0, color="#ccc", lw=.6)
    ax.set_xlabel(xl, fontsize=10)
    for s_ in ["top", "right"]:
        ax.spines[s_].set_visible(False)
axs[0].set_ylabel("速球との縦変化の差 in", fontsize=10)
h_, l_ = axs[0].get_legend_handles_labels()
order = np.argsort([-int(x.split("(")[1][:-1]) for x in l_])
axs[1].legend([h_[i] for i in order], [l_[i] for i in order], fontsize=8, frameon=False, loc="lower right", ncol=2)
fig.suptitle("ピークテーブル：全投手のピーク位置（自分の主な速球を原点、色＝ピーク内で最多の Statcast ラベル、大きさ＝使用率）",
             fontsize=11, x=.01, ha="left")
fig.tight_layout()
out = os.path.join(ROOT, "output", "figures", "19_peak_alignment%s.png" % ("" if MIN_PURITY == 0 else "_pure"))
fig.savefig(out, dpi=120); print(out)
