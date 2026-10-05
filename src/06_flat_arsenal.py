"""
The arsenal is flat -- but the plane does not pass through origin E.

05 showed a plane THROUGH E fits poorly (15.5 deg RMS). This file fits the free plane
per pitcher and asks the questions again:

  1. How flat, against a fake arsenal built from other pitchers' type means?
  2. How much does the plane vary between pitchers, and what orients it (arm angle proxy
     from the release point, shoulder assumed at 5.0 ft)?
  3. Which types leave their own pitcher's plane (leave-one-out), in mph of speed that
     the pitch's movement does not account for?

Part 1-2 was first run as a scratch script; this is that script, kept.

Usage:  python src/06_flat_arsenal.py
"""
import sys, importlib.util, numpy as np, pandas as pd
from scipy import stats
sys.stdout.reconfigure(encoding="utf-8")
spec = importlib.util.spec_from_file_location("pl", "src/05_plane.py"); pl = importlib.util.module_from_spec(spec); spec.loader.exec_module(pl)
sph = pl.sph
rng = np.random.default_rng(7)
d = pl.load(2026); U, L, v0 = sph.vectors_E(d); V = U*L[:,None]; d["i"]=np.arange(len(d))
# map ring: angle of type centroids from the map centre
c = np.array([-0.2, 0.25, -0.95]); c/=np.linalg.norm(c)
print("型の平均方向と地図中心の角度 / 速度成分の割合")
for t in ["FF","SI","FC","SL","ST","SV","CU","KC","CH","FS"]:
    m = d.pitch_type.values==t; u = pl.unit(V[m].mean(0))
    mv = V[m].mean(0)
    print("  %-3s %5.1f°   速度成分/長さ %+.2f   横縦の長さ %.1f  速度成分 %+.1f in" % (t, np.degrees(np.arccos(u@c)), mv[2]/np.linalg.norm(mv), np.hypot(mv[0],mv[1]), mv[2]))
ars = {}
for pid, g in d.groupby("pitcher"):
    vc = g.pitch_type.value_counts(); ok = vc[(vc/len(g)>=.03)&(vc>=30)].index
    if len(ok) < 4: continue
    ars[pid] = [(t, V[g.i.values[g.pitch_type.values==t]].mean(0)) for t in ok]
pool = {}
for rows in ars.values():
    for t,v in rows: pool.setdefault(t,[]).append(v)
def aff_rms(Vp):
    n,_ = pl.normal_affine(Vp); r = (Vp-Vp.mean(0))@n
    return np.sqrt(np.mean(r**2)), n     # inches
rec=[]
for pid, rows in ars.items():
    Vp = np.array([v for _,v in rows]); own, n = aff_rms(Vp)
    fk = np.median([aff_rms(np.array([pool[t][rng.integers(len(pool[t]))] for t,_ in rows]))[0] for _ in range(30)])
    g = d[d.pitcher==pid]
    if n[2] < 0: n = -n
    h = abs(Vp.mean(0) @ n)     # distance of the plane from origin E
    rec.append(dict(k=len(rows), own=own, fake=fk, nx=n[0], ny=n[1], nz=n[2], h=h,
                    spread=np.sqrt(((Vp-Vp.mean(0))**2).sum(1).mean()),
                    arm=np.degrees(np.arctan2(g.release_pos_z.median()-5.0, g.rx.abs().median())),
                    ffdir=np.nan))
R = pd.DataFrame(rec)
print("\n自由な平面（投手ごと、球種4以上 n=%d）: 平面からのずれ RMS（インチ）" % len(R))
for k,s in [(4,R[R.k==4]),(5,R[R.k>=5])]:
    print("  球種%s%d n=%3d  自分 %.2f in   偽の持ち球 %.2f in   自分<偽 %.0f%%   持ち球の広がり %.1f in"
          % ("" if k==4 else "≥",k,len(s),s.own.median(),s.fake.median(),100*(s.own<s.fake).mean(),s.spread.median()))
print("  平面の原点Eからの距離 中央 %.1f in（持ち球の広がり %.1f in）" % (R.h.median(), R.spread.median()))
# orientation: normal direction
N = R[["nx","ny","nz"]].values
print("\n法線の平均 (横 %+.2f, 縦 %+.2f, 速度 %+.2f)  法線の散らばり（平均からの角度 中央 %.0f°）"
      % (*pl.unit(N.mean(0)), np.median(np.degrees(np.arccos(np.clip(N@pl.unit(N.mean(0)),-1,1))))))
# tilt of the plane's trace on the movement chart
tr = np.cross(N, [0,0,1.]); tr = tr*np.sign(tr[:,1:2]+1e-12)
R["trace"] = np.degrees(np.arctan2(tr[:,1], -tr[:,0]))
R["lean"] = np.degrees(np.arccos(np.abs(N[:,2])))   # 0 = plane perpendicular to speed axis
for nm in ["trace","lean","nx","ny","nz"]:
    r,p = stats.spearmanr(R[nm], R.arm); print("  %-5s × 腕の角度  Spearman %+.2f (p=%.1g)" % (nm,r,p))
q = pd.qcut(R.arm,4)
print(R.groupby(q,observed=True)[["trace","lean","nx","ny","nz"]].median().round(2).to_string())
R.to_csv("output/planes_affine_2026.csv", index=False)

# ---------------- part 3: who leaves the plane
S = sph.S_IN_PER_MPH
res=[]; inplane=[]
for pid, g in d.groupby("pitcher"):
    vc = g.pitch_type.value_counts(); ok = vc[(vc/len(g)>=.03)&(vc>=30)].index
    if len(ok) < 4: continue
    T = list(ok); Vp = np.array([V[g.i.values[g.pitch_type.values==t]].mean(0) for t in T])
    arm = np.degrees(np.arctan2(g.release_pos_z.median()-5.0, g.rx.abs().median()))
    for j,t in enumerate(T):
        m = np.arange(len(T))!=j
        n,c = pl.normal_affine(Vp[m]);  n = n*np.sign(n[2])
        res.append((t, (Vp[j]-c)@n/(n[2]*S), pid))   # mph faster than the plane allows
    # FF movement direction in the chart vs arm angle (the known relation, as a check)
    if "FF" in T or "SI" in T:
        f = Vp[T.index("FF" if "FF" in T else "SI")]
        inplane.append((arm, np.degrees(np.arctan2(f[1], -f[0]))))
R = pd.DataFrame(res, columns=["t","mph","pitcher"])
print("自分の平面（その球種を除いて当て直す）からの速度のずれ  +＝変化量のわりに速い、−＝遅い（mph）")
print(R.groupby("t").mph.describe(percentiles=[.25,.5,.75])[["count","25%","50%","75%"]].round(2)
      .loc[["FF","SI","FC","SL","ST","SV","CU","KC","CH","FS"]].to_string())
A = np.array(inplane); r,p = stats.spearmanr(A[:,0],A[:,1])
print("\n速球の変化方向（変化量図上、0°=腕側、90°=真上） × 腕の角度: Spearman %+.2f (p=%.1g) n=%d" % (r,p,len(A)))
# league plane equation in readable units
P = pd.read_csv("output/planes_affine_2026.csv")
n = pl.unit(P[["nx","ny","nz"]].values.mean(0))
print("\n共通平面: 速度差[mph] ≈ %+.2f × 縦変化[in] %+.2f × 横変化[in] + 定数" % (-n[1]/(n[2]*S), -n[0]/(n[2]*S)))
