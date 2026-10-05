"""
Does leaving the Pitch Plane pay?

  per pitch type   r (mph off the PP)  vs  a Stuff+ stand-in, whiff rate, run value
  per pitcher      how far his arsenal strays from the PP (usage-weighted |r|)  vs  xwOBA

Stuff+ itself is not public. The stand-in follows its definition, as baseball-atlas did:
predict each pitch's run value from its PHYSICS ONLY -- speed, movement, spin, extension,
release point, the gap from the pitcher's own fastball, and its type; no location, no
count -- cross-validated by pitcher, then averaged over a pitcher's pitches of that type.

What r is, before reading anything into it: given a pitch's own speed and movement, r
differs between pitchers only through c, the height of the pitcher's plane. So r is a
velocity differential from the rest of HIS arsenal, corrected for movement. The test that
matters is therefore whether r predicts outcomes BEYOND the pitch's own physics.

Usage:  python src/13_pp_vs_outcomes.py
"""
import sys, os, importlib.util
import numpy as np, pandas as pd
from scipy import stats
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import LinearRegression
sys.stdout.reconfigure(encoding="utf-8")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
spec = importlib.util.spec_from_file_location("sph", os.path.join(ROOT, "src", "03_sphere.py"))
sph = importlib.util.module_from_spec(spec); spec.loader.exec_module(sph)

YEAR = 2026
MIN_SHARE, MIN_N = .03, 30
SWING = {"swinging_strike", "swinging_strike_blocked", "foul", "foul_tip", "hit_into_play",
         "foul_bunt", "missed_bunt", "bunt_foul_tip"}
WHIFF = {"swinging_strike", "swinging_strike_blocked", "missed_bunt"}
SHOW = ["FF", "SI", "FC", "SL", "ST", "CU", "KC", "CH", "FS"]

z = np.load(os.path.join(ROOT, "output", "plane_%d.npz" % YEAR))
n_, S = z["normal"], float(z["s"])
A, B = -n_[1] / (n_[2] * S), -n_[0] / (n_[2] * S)

cols = ["game_type", "pitcher", "player_name", "pitch_type", "p_throws", "pfx_x", "pfx_z",
        "release_speed", "release_spin_rate", "release_extension", "release_pos_x", "release_pos_z",
        "delta_run_exp", "description", "events", "estimated_woba_using_speedangle",
        "woba_value", "woba_denom"]
d = pd.read_parquet(sph.SNAP[YEAR], columns=cols)
d = d[(d.game_type == "R") & d.pitch_type.isin(sph.TYPES)]
d = d.dropna(subset=["pfx_x", "pfx_z", "release_speed"]).copy()
cnt = d.groupby("pitcher").size(); d = d[d.pitcher.isin(cnt[cnt >= sph.MIN_PITCHES].index)]
for c in ["pfx_x", "pfx_z", "release_speed", "release_spin_rate", "release_extension",
          "release_pos_x", "release_pos_z", "delta_run_exp", "estimated_woba_using_speedangle",
          "woba_value", "woba_denom"]:
    d[c] = d[c].astype(float)
L = d.p_throws.eq("L")
d["x"] = np.where(L, -d.pfx_x, d.pfx_x) * 12
d["y"] = d.pfx_z * 12
d["side"] = np.where(L, -d.release_pos_x, d.release_pos_x)

# ---- PP residual per pitcher × type (same rule as 10)
g = d.groupby(["pitcher", "pitch_type"])
m = g[["x", "y", "release_speed"]].mean()
m["n"] = g.size(); m["share"] = m.n / m.groupby(level=0).n.transform("sum")
m = m[(m.share >= MIN_SHARE) & (m.n >= MIN_N)].reset_index()
m["eq"] = m.release_speed - (A * m.y + B * m.x)
m["c"] = m.groupby("pitcher")["eq"].transform("median")
m["r"] = m["eq"] - m.c
m = m[m.groupby("pitcher").pitch_type.transform("size") >= 3]

# ---- own primary fastball, for the gap features
fb = (d[d.pitch_type.isin(["FF", "SI", "FC"])].groupby(["pitcher", "pitch_type"])
      [["release_speed", "x", "y"]].agg(["mean", "size"]))
fb.columns = ["v", "_n1", "x", "_n2", "y", "n"]
fb = fb.reset_index().sort_values("n").groupby("pitcher").last()[["v", "x", "y"]]
d = d.join(fb.add_prefix("fb_"), on="pitcher")
d["dv"], d["dx"], d["dy"] = d.release_speed - d.fb_v, d.x - d.fb_x, d.y - d.fb_y

# ---- Stuff+ stand-in: physics only -> run value, CV by pitcher
F = ["release_speed", "x", "y", "release_spin_rate", "release_extension", "release_pos_z",
     "side", "dv", "dx", "dy", "tcode"]
d["tcode"] = d.pitch_type.map({t: i for i, t in enumerate(sph.TYPES)})
ok = d[F[:-1]].notna().all(1) & d.delta_run_exp.notna()
X, yv, grp = d.loc[ok, F].values, -d.loc[ok, "delta_run_exp"].values, d.loc[ok, "pitcher"].values
pred = np.full(len(X), np.nan)
for tr, te in GroupKFold(5).split(X, yv, grp):
    mdl = HistGradientBoostingRegressor(max_iter=300, learning_rate=.05, min_samples_leaf=400,
                                        categorical_features=[len(F) - 1], random_state=0)
    mdl.fit(X[tr], yv[tr]); pred[te] = mdl.predict(X[te])
d.loc[ok, "stuff"] = pred * 100                      # runs saved per 100 pitches, physics only
print("代用Stuff：一球の得点価値を物理量だけで予測（投手単位CV）  相関 %.3f  n=%d"
      % (np.corrcoef(pred, yv)[0, 1], ok.sum()))

# ---- outcomes per pitcher × type
d["swing"] = d.description.isin(SWING)
d["whiff"] = d.description.isin(WHIFF)
d["pa_x"] = np.where(d.woba_denom == 1, d.estimated_woba_using_speedangle.fillna(d.woba_value), np.nan)
o = d.groupby(["pitcher", "pitch_type"]).agg(stuff=("stuff", "mean"), swings=("swing", "sum"),
      whiffs=("whiff", "sum"), rv=("delta_run_exp", "sum"), npt=("delta_run_exp", "size")).reset_index()
o["whiff"] = o.whiffs / o.swings.replace(0, np.nan)
o["rv100"] = -100 * o.rv / o.npt
m = m.merge(o, on=["pitcher", "pitch_type"])

# ---------------------------------------------------------------- 1. per type
print("\n■ 1. 球種ごと：PPからのずれ r と、代用Stuff・空振り率・得点価値（投手間 Spearman）")
print("   %-3s %5s   %8s %8s %8s   | 自分の物理量を統制した上での r の追加説明（空振り率 ΔR²）"
      % ("", "n", "Stuff", "空振り", "RV/100"))
for t in SHOW:
    s = m[(m.pitch_type == t) & (m.swings >= 30)]
    if len(s) < 40:
        continue
    rs = [stats.spearmanr(s.r, s[k])[0] for k in ["stuff", "whiff", "rv100"]]
    base = s[["release_speed", "x", "y"]].values
    w = s.whiff.values
    r0 = LinearRegression().fit(base, w).score(base, w)
    r1 = LinearRegression().fit(np.c_[base, s.r], w).score(np.c_[base, s.r], w)
    print("   %-3s %5d   %+8.2f %+8.2f %+8.2f   | R² %.3f → %.3f（%+.3f）"
          % (t, len(s), *rs, r0, r1, r1 - r0))

# ---------------------------------------------------------------- 2. per pitcher
print("\n■ 2. 投手ごと：持ち球が PP からどれだけ離れているか vs xwOBA")
pa = d[d.woba_denom == 1].groupby("pitcher").agg(xwoba=("pa_x", "mean"), pa=("pa_x", "size"))
m["absr"] = m.r.abs()
p = m.groupby("pitcher").apply(lambda g: pd.Series({
    "stray": np.average(g.absr, weights=g.share),                  # usage-weighted |r|
    "stray_off": np.average(g.absr[~g.pitch_type.isin(sph.FAST)], weights=g.share[~g.pitch_type.isin(sph.FAST)])
                 if (~g.pitch_type.isin(sph.FAST)).any() else np.nan,
    "c": g.c.iloc[0], "stuff": np.average(g.stuff, weights=g.share)}), include_groups=False).join(pa)
p = p[p.pa >= 100].dropna()
print("   投手 n=%d（100打席以上）" % len(p))
for k, lab in [("stray", "PP離れ（全球種、使用率加重 |r|）"), ("stray_off", "PP離れ（変化球のみ）"),
               ("c", "PPの高さ c"), ("stuff", "代用Stuff（使用率加重）")]:
    rho, pv = stats.spearmanr(p[k], p.xwoba)
    print("   %-30s × xwOBA  Spearman %+.2f  (p=%.2g)" % (lab, rho, pv))
Xb = p[["c", "stuff"]].values
r0 = LinearRegression().fit(Xb, p.xwoba).score(Xb, p.xwoba)
r1 = LinearRegression().fit(np.c_[Xb, p.stray], p.xwoba).score(np.c_[Xb, p.stray], p.xwoba)
b = LinearRegression().fit(np.c_[Xb, p.stray], p.xwoba).coef_[2]
print("   xwOBA ~ c + Stuff の R² %.3f → + PP離れ %.3f（%+.3f）  PP離れの係数 %+.4f / mph"
      % (r0, r1, r1 - r0, b))
m.to_csv(os.path.join(ROOT, "output", "pp_outcomes_%d.csv" % YEAR), index=False)
p.to_csv(os.path.join(ROOT, "output", "pp_pitchers_%d.csv" % YEAR))

# ---------------------------------------------------------------- 3. is r just the velocity gap?
print("\n■ 3. r の効果は「速球との球速差」を言い換えただけか（空振り率、ΔR²）")
gap = d.groupby(["pitcher", "pitch_type"])[["dv", "dx", "dy"]].mean().reset_index()
mm = m.merge(gap, on=["pitcher", "pitch_type"])
for t in ["FC", "SL", "ST", "CH", "FS"]:
    s = mm[(mm.pitch_type == t) & (mm.swings >= 30)].dropna(subset=["dv", "whiff"])
    w = s.whiff.values
    def R(cols):
        X_ = s[cols].values
        return LinearRegression().fit(X_, w).score(X_, w)
    b0 = R(["release_speed", "x", "y"])
    b1 = R(["release_speed", "x", "y", "dv", "dx", "dy"])
    b2 = R(["release_speed", "x", "y", "dv", "dx", "dy", "r"])
    print("   %-3s 自分の物理量 %.3f → ＋速球との差 %.3f → ＋r %.3f（r の上積み %+.3f）"
          % (t, b0, b1, b2, b2 - b1))
