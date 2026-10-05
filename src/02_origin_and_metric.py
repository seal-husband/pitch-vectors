"""
Before the sphere: two choices that the sphere cannot make for you.

Putting a pitch on a unit sphere says "the direction is the thing". That only means
something once two prior questions are answered, and neither is a detail.

  1. WHERE IS THE ORIGIN. Two of the three axes have a physical zero -- pfx_x = pfx_z = 0
     is a real pitch, the one a spinless ball would throw. Velocity has no zero. So the
     origin is a choice, not a discovery, and a bad choice parks a crowd of pitches on
     top of it, where direction is pure noise.

  2. WHAT IS AN INCH WORTH IN MPH. The direction of (Δ横, Δ縦, Δ速度) is NOT scale-free:
     stretch the velocity axis and every direction on the sphere moves. The spherical
     framing does not dodge the units problem, it makes the units the whole map.

For (2) this file does not reach for a standard deviation. The trajectory columns are
there, so the conversion can be derived: being 1 mph slower buys extra flight time, and
gravity turns extra time into extra drop. That is a real number of inches, computed from
each pitch's own trajectory rather than from the spread of the league.

Then the question the sphere raises on its own: a sphere carries direction only. Is the
LENGTH redundant given the direction, or is it a fourth thing that needs its own channel?

Usage:  python src/02_origin_and_metric.py [season]
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
FAST = ["FF", "SI", "FC", "FA"]
TYPES = ["FF", "SI", "FC", "FA", "SL", "ST", "SV", "CU", "KC", "CS", "CH", "FS", "FO"]
OFF = ["SL", "ST", "SV", "CU", "KC", "CS", "CH", "FS", "FO"]
Y0, YPLATE = 50.0, 17.0 / 12.0          # Statcast reports the state at y = 50 ft


def load(year):
    d = pd.read_parquet(SNAP[year], columns=[
        "game_type", "pitcher", "player_name", "pitch_type", "p_throws",
        "pfx_x", "pfx_z", "release_speed", "vy0", "az", "vz0"])
    d = d[(d.game_type == "R") & d.pitch_type.isin(TYPES)
          & d.pfx_x.notna() & d.pfx_z.notna() & d.release_speed.notna()
          & d.vy0.notna() & d.az.notna()].copy()
    n = d.groupby("pitcher").size()
    d = d[d.pitcher.isin(n[n >= MIN_PITCHES].index)]
    for c in ["pfx_x", "pfx_z", "release_speed", "vy0", "az", "vz0"]:
        d[c] = d[c].astype(float)
    d["hx"] = np.where(d.p_throws.eq("L"), -d.pfx_x, d.pfx_x) * 12
    d["vz"] = d.pfx_z * 12
    return d


def drop_per_mph(d):
    """Inches of extra drop at the plate bought by being 1 mph slower.

    Not a guess and not a standard deviation. Flight time comes from each pitch's own
    y-motion; d(time)/d(speed) is differentiated from it; gravity plus the ball's own
    vertical velocity turn that extra time into extra fall.
    """
    t = (Y0 - YPLATE) / np.abs(d.vy0)                  # flight time, s
    vz_plate = d.vz0 + d.az * t                        # vertical speed at the plate, ft/s
    # slower by 1 mph -> dt = (distance / v^2) * 1.4667 ft/s per mph
    dt = (Y0 - YPLATE) / (d.vy0 ** 2) * 1.4667
    return t, (np.abs(vz_plate) * dt) * 12             # inches per mph


def unit(V):
    n = np.linalg.norm(V, axis=1, keepdims=True)
    n[n < 1e-9] = 1e-9
    return V / n, n[:, 0]


def main():
    pd.set_option("display.width", 220)
    year = int(sys.argv[1]) if len(sys.argv) > 1 else 2026
    d = load(year)
    print("%d年 %d投手 / %d球" % (year, d.pitcher.nunique(), len(d)))

    # ---------------------------------------------------------------- 1. the metric
    t, dz = drop_per_mph(d)
    print("\n■ 1. 1 mph 遅いことは、何インチの落差に相当するか（各球の軌道から）")
    print("   到達時間      中央 %.3f 秒" % t.median())
    print("   1mphあたりの追加時間 中央 %.2f ms" % (1000 * ((Y0 - YPLATE) /
          (d.vy0 ** 2) * 1.4667).median()))
    print("   **物理換算 s = %.2f インチ/mph**（四分位 %.2f 〜 %.2f）"
          % (dz.median(), dz.quantile(.25), dz.quantile(.75)))
    off = d[d.pitch_type.isin(OFF)]
    sd_ratio = off.vz.std() / off.release_speed.std()
    print("   参考：標準偏差比で決めるなら s = %.2f インチ/mph" % sd_ratio)
    print("   → 二つは %.1f 倍ちがう。どちらを採るかで球面上の位置が動く。"
          % (sd_ratio / dz.median()))

    # ---------------------------------------------------------------- 2. the origin
    print("\n■ 2. 原点の候補と、その原点に貼りつく球（＝方向が定義できない球）")
    fam = d[d.pitch_type.isin(FAST)]
    ownF = fam.groupby("pitcher")[["hx", "vz", "release_speed"]].mean()
    ownFF = (d[d.pitch_type == "FF"].groupby("pitcher")
             [["hx", "vz", "release_speed"]].mean())
    ownAll = d.groupby("pitcher")[["hx", "vz", "release_speed"]].mean()
    lgF = fam[["hx", "vz", "release_speed"]].mean()
    s = float(dz.median())

    cands = {
        "A 自分の速球（FF/SI/FC）": ownF,
        "B 自分のフォーシームだけ": ownFF,
        "C リーグの速球平均（全員共通）": None,
        "D 自分の全球の重心": ownAll,
        "E 物理ゼロ（無回転）＋リーグ速球の球速": None,
    }
    print("   %-26s %9s %9s %9s" % ("", "|v|中央", "|v|<3in", "変化球で<3in"))
    for nm, ref in cands.items():
        if nm.startswith("C"):
            R = np.tile(lgF.values, (len(d), 1))
        elif nm.startswith("E"):
            R = np.c_[np.zeros(len(d)), np.zeros(len(d)),
                      np.full(len(d), lgF.release_speed)]
        else:
            j = d.join(ref, on="pitcher", rsuffix="_r")
            R = j[["hx_r", "vz_r", "release_speed_r"]].values
        V = np.c_[d.hx.values - R[:, 0], d.vz.values - R[:, 1],
                  (d.release_speed.values - R[:, 2]) * s]
        _, L = unit(V)
        ok = ~np.isnan(L)
        isoff = d.pitch_type.isin(OFF).values & ok
        print("   %-26s %9.1f %8.1f%% %11.1f%%"
              % (nm, np.nanmedian(L[ok]), 100 * np.mean(L[ok] < 3),
                 100 * np.mean(L[isoff] < 3)))
    print("   （原点に近い球は方向がノイズになる。少ないほど原点として健全）")

    # ---------------------------------------------------------------- 3. length
    print("\n■ 3. 長さは方向から予測できるか（＝球面だけで足りるか）")
    j = d.join(ownF, on="pitcher", rsuffix="_r")
    V = np.c_[j.hx - j.hx_r, j.vz - j.vz_r, (j.release_speed - j.release_speed_r) * s]
    U, L = unit(V)
    keep = (L > 1) & d.pitch_type.isin(OFF).values
    U, L = U[keep], L[keep]
    X = np.c_[U, U[:, 0] ** 2, U[:, 1] ** 2, U[:, 2] ** 2,
              U[:, 0] * U[:, 1], U[:, 0] * U[:, 2], U[:, 1] * U[:, 2],
              np.ones(len(U))]
    y = np.log(L)
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    r2 = 1 - ((y - X @ b) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    print("   log|v| を方向（2次まで）で回帰：R² = %.3f  n=%d" % (r2, len(U)))
    print("   |v| 中央 %.1f in（四分位 %.1f 〜 %.1f）" % (np.median(L),
          np.quantile(L, .25), np.quantile(L, .75)))

    # ---------------------------------------------------------------- 4. sensitivity
    print("\n■ 4. 単位の選び方で、方向はどれだけ動くか")
    for s2 in [0.5 * s, s, 2 * s, sd_ratio]:
        V2 = np.c_[j.hx - j.hx_r, j.vz - j.vz_r,
                   (j.release_speed - j.release_speed_r) * s2]
        U2, _ = unit(V2)
        cos = np.clip((U * U2[keep]).sum(1), -1, 1)
        ang = np.degrees(np.arccos(cos))
        tag = "物理換算" if abs(s2 - s) < 1e-9 else ("SD比" if abs(s2 - sd_ratio) < 1e-9
                                                 else "")
        print("   s=%.2f in/mph %-8s 物理換算からのずれ 中央 %5.1f°  上位10%% %5.1f°"
              % (s2, tag, np.median(ang), np.quantile(ang, .9)))


if __name__ == "__main__":
    main()
