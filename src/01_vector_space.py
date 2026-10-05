"""
Is a pitch type a place, or a box someone drew?

Every pitch becomes a vector from the pitcher's own fastball:

    Δ横   inches sideways of his fastball  (mirrored so a righty frame holds for all)
    Δ縦   inches below his fastball
    Δ速度 mph slower than his fastball

Then three questions, in the order that can actually settle them:

  1. Between two named types, is there a valley in the density? A label with no valley
     under it is a convenience, not a seam in nature.
  2. Where does the slurve sit? If it is genuinely intermediate its centroid lies between
     slider and curve AND its own spread overlaps both, rather than forming a third lump.
  3. Do the changeup and the splitter point the same way? Same box in the old map; the
     claim here is that one is mostly slower and the other mostly lower.

Nothing is standardised in this file. The three components are reported in their own
units, because how to make inches and mph commensurate is the decision this project
exists to make, and dividing by a standard deviation would quietly make it already.

Usage:  python src/01_vector_space.py [season]
"""
from __future__ import annotations
import sys
sys.stdout.reconfigure(encoding="utf-8")
import os

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
from paths import SNAP
MIN_PITCHES = 400
FAST = ["FF", "SI", "FC", "FA"]          # the reference the hitter times off
JP = {"FF": "フォーシーム", "SI": "シンカー", "FC": "カッター", "FA": "速球(不明)",
      "SL": "スライダー", "ST": "スイーパー", "SV": "スラーブ",
      "CU": "カーブ", "KC": "ナックルカーブ", "CS": "スローカーブ",
      "CH": "チェンジ", "FS": "スプリット", "FO": "フォーク",
      "KN": "ナックル", "EP": "イーファス"}
AX = ["Δ横", "Δ縦", "Δ速度"]


def load(year):
    d = pd.read_parquet(SNAP[year], columns=[
        "game_type", "pitcher", "player_name", "pitch_type", "p_throws",
        "pfx_x", "pfx_z", "release_speed"])
    d = d[(d.game_type == "R") & d.pitch_type.isin(JP)
          & d.pfx_x.notna() & d.pfx_z.notna() & d.release_speed.notna()].copy()
    n = d.groupby("pitcher").size()
    d = d[d.pitcher.isin(n[n >= MIN_PITCHES].index)]
    d["hx"] = np.where(d.p_throws.eq("L"), -d.pfx_x, d.pfx_x) * 12
    d["vz"] = d.pfx_z * 12
    return d


def vectors(d):
    """Each pitch minus its own pitcher's fastball."""
    f = d[d.pitch_type.isin(FAST)]
    ref = f.groupby("pitcher")[["hx", "vz", "release_speed"]].mean()
    ref.columns = ["rh", "rv", "rs"]
    d = d.join(ref, on="pitcher")
    d = d[d.rh.notna()]
    # float64 throughout: parquet hands back nullable extension arrays, and those refuse
    # to take part in the matrix arithmetic further down.
    d["Δ横"] = (d.hx - d.rh).astype(float)
    d["Δ縦"] = (d.vz - d.rv).astype(float)
    d["Δ速度"] = (d.release_speed - d.rs).astype(float)
    return d


def valley(x, lo, hi, bins=60):
    """Smallest density between the two modes, as a share of the smaller mode.

    1.0 means no valley at all -- the two lumps are one lump. Near 0 means a clean gap.
    """
    x = x[(x >= lo) & (x <= hi)]
    h, edges = np.histogram(x, bins=bins, range=(lo, hi))
    h = np.convolve(h, np.ones(5) / 5, mode="same")          # light smoothing
    if h.max() <= 0:
        return np.nan, np.nan
    peak_l = np.argmax(h[:bins // 2])
    peak_r = bins // 2 + np.argmax(h[bins // 2:])
    if peak_r <= peak_l:
        return np.nan, np.nan
    dip = h[peak_l:peak_r + 1].min()
    mid = edges[peak_l:peak_r + 1][np.argmin(h[peak_l:peak_r + 1])]
    return dip / min(h[peak_l], h[peak_r]), mid


def main():
    pd.set_option("display.width", 220)
    year = int(sys.argv[1]) if len(sys.argv) > 1 else 2026
    d = vectors(load(year))
    print("%d年 %d投手 / %d球（自分の速球を持つ投手のみ）"
          % (year, d.pitcher.nunique(), len(d)))

    print("\n■ 1. 球種ごとの平均ベクトルと、その散らばり")
    print("   %-3s %-10s %7s %7s %7s   %7s %7s %7s  %8s"
          % ("", "", "Δ横", "Δ縦", "Δ速度", "SD横", "SD縦", "SD速", "球数"))
    order = ["FF", "SI", "FC", "SL", "ST", "SV", "CU", "KC", "CS", "CH", "FS", "FO"]
    for t in order:
        g = d[d.pitch_type == t]
        if len(g) < 200:
            continue
        print("   %-3s %-10s %+7.1f %+7.1f %+7.1f   %7.1f %7.1f %7.1f  %8d"
              % (t, JP[t], g["Δ横"].mean(), g["Δ縦"].mean(), g["Δ速度"].mean(),
                 g["Δ横"].std(), g["Δ縦"].std(), g["Δ速度"].std(), len(g)))

    print("\n■ 2. 箱の間に谷はあるか（二つの群の重心を結ぶ軸に射影）")
    pairs = [("SL", "CU"), ("SL", "ST"), ("ST", "SV"), ("SV", "CU"),
             ("CH", "FS"), ("SL", "FC"), ("CU", "KC")]
    for a, b in pairs:
        ga, gb = d[d.pitch_type == a], d[d.pitch_type == b]
        if len(ga) < 200 or len(gb) < 200:
            continue
        ca = ga[AX].mean().values
        cb = gb[AX].mean().values
        u = cb - ca
        L = np.linalg.norm(u)
        u = u / L
        both = d[d.pitch_type.isin([a, b])]
        proj = (both[AX].values - ca) @ u          # 0 = a の重心, L = b の重心
        v, mid = valley(proj, -.35 * L, 1.35 * L)
        print("   %-3s ⇔ %-3s  重心間 %5.1f   谷の深さ %s   （1.0=谷なし）"
              % (a, b, L, "%.2f" % v if v == v else "  ―"))

    print("\n■ 3. スラーブは本当に中間か（SL→CU 軸への射影）")
    cs = d[d.pitch_type == "SL"][AX].mean().values
    cc = d[d.pitch_type == "CU"][AX].mean().values
    u = (cc - cs) / np.linalg.norm(cc - cs)
    for t in ["SL", "ST", "SV", "CU", "KC"]:
        g = d[d.pitch_type == t]
        if len(g) < 200:
            continue
        p = (g[AX].values - cs) @ u / np.linalg.norm(cc - cs)
        print("   %-3s %-10s 位置 %+.2f （SL=0, CU=1）  四分位 %+.2f / %+.2f  n=%d"
              % (t, JP[t], p.mean(), np.quantile(p, .25), np.quantile(p, .75), len(g)))

    print("\n■ 4. チェンジとスプリットは同じ向きか")
    for t in ["CH", "FS", "FO"]:
        g = d[d.pitch_type == t]
        if len(g) < 200:
            continue
        c = g[AX].mean().values
        ang = np.degrees(np.arctan2(-c[1], -c[2]))   # 落差 vs 減速の比
        print("   %-3s %-10s Δ横 %+5.1f in  Δ縦 %+5.1f in  Δ速度 %+5.1f mph"
              "   落差/減速 = %.2f in per mph"
              % (t, JP[t], c[0], c[1], c[2], (-c[1]) / (-c[2])))
    print("   （チェンジが「遅らせる球」、スプリットが「落とす球」なら比が食い違うはず）")

    out = os.path.join(ROOT, "output")
    os.makedirs(out, exist_ok=True)
    cols = ["pitcher", "player_name", "p_throws", "pitch_type"] + AX
    d[cols].to_parquet(os.path.join(out, "vectors_%d.parquet" % year), index=False)
    print("\nwrote output/vectors_%d.parquet （%d球）" % (year, len(d)))


if __name__ == "__main__":
    main()
