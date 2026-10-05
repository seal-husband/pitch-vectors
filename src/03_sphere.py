"""
The arsenal on a sphere, counted honestly.

Origin E: the physical zero. A pitch's vector is

    (横変化, 縦変化, (球速 − リーグ速球平均) × 0.63)     all in inches

pfx = 0 is a real pitch -- the one a spinless ball throws -- so two of the three axes
need no convention. The third gets its zero from the league's fastballs and its scale
from physics: 0.63 inches of extra gravity drop per mph of slowness, derived in 02.

Nothing here privileges the fastball. It is not the origin; it has a direction like
every other pitch, and it appears on the sphere alongside them.

Density is counted in caps of FIXED SOLID ANGLE, not in cells of a projection. That is
the whole reason for caring about equal area: a window that is the same size everywhere
on the sphere gives counts that mean the same thing everywhere. The Lambert azimuthal
equal-area projection is then only how the answer is drawn.

Asks: how many modes does the density have, where are they, and is there a valley
between them -- the same question 01 asked along straight lines, now on the sphere where
it belongs.

Usage:  python src/03_sphere.py [season]
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
S_IN_PER_MPH = 0.63                 # from 02; gravity's price for being slow
FAST = ["FF", "SI", "FC", "FA"]
TYPES = ["FF", "SI", "FC", "FA", "SL", "ST", "SV", "CU", "KC", "CS", "CH", "FS", "FO"]
JP = {"FF": "フォーシーム", "SI": "シンカー", "FC": "カッター", "FA": "速球(不明)",
      "SL": "スライダー", "ST": "スイーパー", "SV": "スラーブ",
      "CU": "カーブ", "KC": "ナックルカーブ", "CS": "スローカーブ",
      "CH": "チェンジ", "FS": "スプリット", "FO": "フォーク"}
CAP_FRACTION = .01                  # counting window = 1% of the sphere
SEED = 3


def load(year):
    d = pd.read_parquet(SNAP[year], columns=[
        "game_type", "pitcher", "player_name", "pitch_type", "p_throws",
        "pfx_x", "pfx_z", "release_speed"])
    d = d[(d.game_type == "R") & d.pitch_type.isin(TYPES)
          & d.pfx_x.notna() & d.pfx_z.notna() & d.release_speed.notna()].copy()
    n = d.groupby("pitcher").size()
    d = d[d.pitcher.isin(n[n >= MIN_PITCHES].index)]
    for c in ["pfx_x", "pfx_z", "release_speed"]:
        d[c] = d[c].astype(float)
    d["hx"] = np.where(d.p_throws.eq("L"), -d.pfx_x, d.pfx_x) * 12
    d["vz"] = d.pfx_z * 12
    return d


def vectors_E(d):
    """Origin E: physical zero for break, league fastball speed for velocity."""
    v0 = d[d.pitch_type.isin(FAST)].release_speed.mean()
    V = np.c_[d.hx.values, d.vz.values,
              (d.release_speed.values - v0) * S_IN_PER_MPH]
    L = np.linalg.norm(V, axis=1)
    return V / L[:, None], L, v0


def fibonacci_sphere(n):
    """Roughly equal-area grid of directions -- no pole pile-up."""
    i = np.arange(n) + .5
    phi = np.arccos(1 - 2 * i / n)
    theta = np.pi * (1 + 5 ** .5) * i
    return np.c_[np.cos(theta) * np.sin(phi), np.sin(theta) * np.sin(phi), np.cos(phi)]


def cap_density(grid, U, cos_r, chunk=400):
    """Pitches per fixed-solid-angle cap, at each grid direction."""
    out = np.zeros(len(grid))
    for a in range(0, len(grid), chunk):
        g = grid[a:a + chunk]
        out[a:a + chunk] = (g @ U.T >= cos_r).sum(1)
    return out


def lambert(U, centre):
    """Lambert azimuthal equal-area about `centre`. Returns disk coords, radius <= 2."""
    z = centre / np.linalg.norm(centre)
    a = np.array([0., 0., 1.])
    if abs(z @ a) > .95:
        a = np.array([1., 0., 0.])
    e1 = np.cross(a, z); e1 /= np.linalg.norm(e1)
    e2 = np.cross(z, e1)
    cos = np.clip(U @ z, -1, 1)
    r = 2 * np.sin(np.arccos(cos) / 2)           # the equal-area radius
    ang = np.arctan2(U @ e2, U @ e1)
    return np.c_[r * np.cos(ang), r * np.sin(ang)], np.degrees(np.arccos(cos))


def main():
    pd.set_option("display.width", 220)
    rng = np.random.default_rng(SEED)
    year = int(sys.argv[1]) if len(sys.argv) > 1 else 2026
    d = load(year)
    U, L, v0 = vectors_E(d)
    print("%d年 %d投手 / %d球　リーグ速球平均 %.1f mph、s=%.2f in/mph"
          % (year, d.pitcher.nunique(), len(d), v0, S_IN_PER_MPH))
    print("   |v| 中央 %.1f in（四分位 %.1f 〜 %.1f）  最小 %.2f"
          % (np.median(L), np.quantile(L, .25), np.quantile(L, .75), L.min()))
    print("   |v| < 3 in の球（方向が不安定）: %d (%.2f%%)"
          % ((L < 3).sum(), 100 * (L < 3).mean()))

    print("\n■ 1. 球種ごとの平均方向（原点E、単位ベクトル）")
    cen = {}
    for t in ["FF", "SI", "FC", "SL", "ST", "SV", "CU", "KC", "CH", "FS"]:
        m = (d.pitch_type == t).values
        if m.sum() < 200:
            continue
        c = U[m].mean(0)
        cen[t] = c / np.linalg.norm(c)
        print("   %-3s %-10s (%+.3f, %+.3f, %+.3f)  n=%6d  |v|中央 %5.1f in"
              % (t, JP[t], *cen[t], m.sum(), np.median(L[m])))

    mean_dir = U.mean(0); mean_dir /= np.linalg.norm(mean_dir)
    _, ang_mean = lambert(U, mean_dir)
    _, ang_ff = lambert(U, cen["FF"])
    print("\n■ 2. どこを投影中心にすれば片半球に収まるか")
    for nm, a in (("全球の平均方向", ang_mean), ("フォーシームの平均方向", ang_ff)):
        print("   %-16s 90°以内 %5.1f%%   中央 %5.1f°   95%%点 %5.1f°"
              % (nm, 100 * (a <= 90).mean(), np.median(a), np.quantile(a, .95)))

    # ---- density in caps of fixed solid angle --------------------------------
    take = rng.choice(len(U), min(90000, len(U)), replace=False)
    Us = U[take]
    grid = fibonacci_sphere(4000)
    cos_r = 1 - 2 * CAP_FRACTION
    dens = cap_density(grid, Us, cos_r)
    dens = dens / dens.sum()
    print("\n■ 3. 立体角一定の窓（球面の %.0f%%、半径 %.1f°）で数えた密度"
          % (100 * CAP_FRACTION, np.degrees(np.arccos(cos_r))))

    # local maxima: higher than every grid direction within NEIGH degrees
    NEIGH = np.cos(np.radians(25))
    order = np.argsort(dens)[::-1]
    modes = []
    for i in order:
        if dens[i] <= 0:
            break
        if all(grid[i] @ grid[j] < NEIGH for j in modes):
            modes.append(i)
        if len(modes) >= 12:
            break
    print("   密度の極大（25°以内で最大のもの）:")
    for i in modes:
        if dens[i] < .2 * dens[modes[0]]:
            continue
        near = d.pitch_type.values[take][Us @ grid[i] >= cos_r]
        lab = pd.Series(near).value_counts(normalize=True)
        who = "  ".join("%s%.0f%%" % (k, 100 * v) for k, v in lab.head(3).items())
        print("      密度 %5.1f%%（最大比） 方向 (%+.2f, %+.2f, %+.2f)   %s"
              % (100 * dens[i] / dens[modes[0]], *grid[i], who))

    print("\n■ 4. 極大どうしの間に谷はあるか（大円上を歩いて密度を見る）")
    top = modes[:4]
    for a in range(len(top)):
        for b in range(a + 1, len(top)):
            ga, gb = grid[top[a]], grid[top[b]]
            sep = np.degrees(np.arccos(np.clip(ga @ gb, -1, 1)))
            ts = np.linspace(0, 1, 41)
            path = np.array([(1 - t) * ga + t * gb for t in ts])
            path /= np.linalg.norm(path, axis=1, keepdims=True)
            dp = cap_density(path, Us, cos_r)
            dip = dp[3:-3].min() / min(dp[0], dp[-1])
            print("      極大%d ⇔ 極大%d  離角 %5.1f°   谷の深さ %.2f"
                  % (a + 1, b + 1, sep, dip))
    print("   （1.0=谷なし。0に近いほど二つは別の塊）")

    out = os.path.join(ROOT, "output")
    os.makedirs(out, exist_ok=True)
    np.savez(os.path.join(out, "sphere_%d.npz" % year),
             U=U.astype(np.float32), L=L.astype(np.float32),
             pitch_type=d.pitch_type.values.astype("U2"),
             grid=grid, dens=dens, mean_dir=mean_dir)
    print("\nwrote output/sphere_%d.npz" % year)


if __name__ == "__main__":
    main()
