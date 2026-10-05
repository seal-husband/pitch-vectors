"""
What is the common plane?   speed ≈ 0.58 mph × (inches of vertical break) + const

Two readings compete, and the trajectory columns can separate them.

  A. ARTIFACT OF FLIGHT TIME. pfx is a displacement: the same spin force moves a slow
     pitch further because it is in the air longer. If the plane is this, it should
     weaken when movement is replaced by the spin ACCELERATION (ax, az + g), which does
     not know how long the ball flew.

  B. A COUPLING IN THE HAND. Backspin needs the fingers behind the ball, where they also
     push it fastest; topspin needs the hand turned over the top, which costs speed.
     If so the relation lives in the direction of spin force, and it should also appear
     INSIDE one pitch type, pitch to pitch, where no grip change is involved.

Usage:  python src/07_what_the_plane_is.py
"""
import sys, os, importlib.util
import numpy as np, pandas as pd
sys.stdout.reconfigure(encoding="utf-8")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
spec = importlib.util.spec_from_file_location("sph", os.path.join(ROOT, "src", "03_sphere.py"))
sph = importlib.util.module_from_spec(spec); spec.loader.exec_module(sph)
G = 32.174

d = pd.read_parquet(sph.SNAP[2026], columns=["game_type", "pitcher", "pitch_type", "p_throws",
                    "pfx_x", "pfx_z", "release_speed", "ax", "az"])
d = d[(d.game_type == "R") & d.pitch_type.isin(sph.TYPES)].dropna().copy()
n = d.groupby("pitcher").size(); d = d[d.pitcher.isin(n[n >= 400].index)]
for c in ["pfx_x", "pfx_z", "release_speed", "ax", "az"]:
    d[c] = d[c].astype(float)
Lh = d.p_throws.eq("L")
d["hx"] = np.where(Lh, -d.pfx_x, d.pfx_x) * 12
d["vz"] = d.pfx_z * 12
d["ahx"] = np.where(Lh, -d.ax, d.ax)            # ft/s², spin (+drag) sideways
d["avz"] = d.az + G                              # ft/s², spin lift

def fit(y, X):
    X = np.c_[X, np.ones(len(X))]
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    r = y - X @ b
    return b, 1 - r.var() / y.var()

# ---------------------------------------------------------------- A
print("■ A. 変位（pfx）を加速度に替えても平面は残るか")
print("   投手ごとに、球種平均の球速を 縦・横 で回帰（4球種以上）")
out = []
for pid, g in d.groupby("pitcher"):
    vc = g.pitch_type.value_counts(); ok = vc[(vc / len(g) >= .03) & (vc >= 30)].index
    if len(ok) < 4: continue
    m = g[g.pitch_type.isin(ok)].groupby("pitch_type")[["release_speed", "hx", "vz", "ahx", "avz"]].mean()
    b1, r1 = fit(m.release_speed.values, m[["vz", "hx"]].values)
    b2, r2 = fit(m.release_speed.values, m[["avz", "ahx"]].values)
    out.append((r1, r2, b1[0], b2[0]))
O = np.array(out)
print("   変位（in）      R² 中央 %.2f   縦の傾き 中央 %.2f mph/in" % (np.median(O[:, 0]), np.median(O[:, 2])))
print("   加速度（ft/s²） R² 中央 %.2f   縦の傾き 中央 %.2f mph per ft/s²" % (np.median(O[:, 1]), np.median(O[:, 3])))
print("   n=%d投手" % len(O))

# ---------------------------------------------------------------- B
print("\n■ B. 同じ球種の中でも、一球ごとに「浮くほど速い」か（投手×球種ごとに中心化）")
print("   加速度で測る（滞空時間の影響を除く）。傾き = 揚力 1 ft/s² あたりの mph")
rows = []
for t in ["FF", "SI", "FC", "SL", "ST", "CU", "CH", "FS"]:
    g = d[d.pitch_type == t]
    k = g.groupby("pitcher").release_speed.transform("size") >= 100
    g = g[k]
    dv = g.release_speed - g.groupby("pitcher").release_speed.transform("mean")
    da = g.avz - g.groupby("pitcher").avz.transform("mean")
    dh = g.ahx - g.groupby("pitcher").ahx.transform("mean")
    b, r2 = fit(dv.values, np.c_[da.values, dh.values])
    rows.append((t, b[0], b[1], r2, len(g)))
    print("   %-3s  縦 %+.3f  横 %+.3f mph per ft/s²   R² %.3f   n=%d" % rows[-1])

# the across-type slope in the same currency, for comparison
lg = d.groupby(["pitcher", "pitch_type"])[["release_speed", "avz", "ahx"]].mean().reset_index()
lg = lg[lg.pitch_type.isin(["FF", "SI", "FC", "SL", "ST", "CU", "CH", "FS"])]
dv = lg.release_speed - lg.groupby("pitcher").release_speed.transform("mean")
da = lg.avz - lg.groupby("pitcher").avz.transform("mean")
dh = lg.ahx - lg.groupby("pitcher").ahx.transform("mean")
b, r2 = fit(dv.values, np.c_[da.values, dh.values])
print("   参考：球種の間（投手内）  縦 %+.3f  横 %+.3f  R² %.2f" % (b[0], b[1], r2))

# ---------------------------------------------------------------- the changeup, in this currency
print("\n■ C. 球種の間の関係から外れる量（投手内の回帰の残差, mph）")
lg["res"] = dv - np.c_[da, dh, np.ones(len(lg))] @ np.r_[b[:2], 0] + 0
lg["res"] -= lg.res.mean()
print(lg.groupby("pitch_type").res.median().round(2).reindex(["FF","SI","FC","SL","ST","CU","CH","FS"]).to_string())
