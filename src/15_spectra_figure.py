"""
What a pitcher's "spectra" look like: his pitches projected on each axis, stacked by type.

Usage:  python src/15_spectra_figure.py "Glasnow, Tyler" "Sale, Chris"
"""
import sys, os, importlib.util
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.stdout.reconfigure(encoding="utf-8")
plt.rcParams["font.family"] = ["Yu Gothic", "Meiryo", "MS Gothic", "sans-serif"]
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
spec = importlib.util.spec_from_file_location("sph", os.path.join(ROOT, "src", "03_sphere.py"))
sph = importlib.util.module_from_spec(spec); spec.loader.exec_module(sph)
COL = {"FF": "#D22D49", "SI": "#FE9D00", "FC": "#933F2C", "SL": "#EEE716", "ST": "#DDB33A",
       "SV": "#93AFD4", "CU": "#00D1ED", "KC": "#6236CD", "CS": "#0068FF", "CH": "#1DBE3A",
       "FS": "#3BACAC", "FO": "#55CCAB", "FA": "#888888"}
z = np.load(os.path.join(ROOT, "output", "plane_2026.npz"))
E1, S, V0 = z["axes"][:, 0], float(z["s"]), float(z["v0"])

names = sys.argv[1:] or ["Glasnow, Tyler", "Sale, Chris"]
d = pd.read_parquet(sph.SNAP[2026], columns=["game_type", "player_name", "pitch_type", "p_throws",
                    "pfx_x", "pfx_z", "release_speed"])
d = d[(d.game_type == "R") & d.player_name.isin(names) & d.pitch_type.isin(sph.TYPES)].dropna().copy()
for c in ["pfx_x", "pfx_z", "release_speed"]:
    d[c] = d[c].astype(float)
d["横変化 (in)"] = np.where(d.p_throws.eq("L"), -d.pfx_x, d.pfx_x) * 12
d["縦変化 (in)"] = d.pfx_z * 12
d["球速 (mph)"] = d.release_speed
d["PP第1軸 (in)"] = np.c_[d["横変化 (in)"], d["縦変化 (in)"], (d.release_speed - V0) * S] @ E1
AXES = ["球速 (mph)", "横変化 (in)", "縦変化 (in)", "PP第1軸 (in)"]

fig, axs = plt.subplots(len(names), 4, figsize=(15, 3.1 * len(names)), squeeze=False)
for i, nm in enumerate(names):
    g = d[d.player_name == nm]
    vc = g.pitch_type.value_counts()
    T = [t for t in vc.index if vc[t] / len(g) >= .03]
    for j, a in enumerate(AXES):
        ax = axs[i, j]
        lo, hi = np.quantile(g[a], [.002, .998]); bins = np.linspace(lo - 1, hi + 1, 70)
        ax.hist([g[g.pitch_type == t][a] for t in T], bins=bins, stacked=True,
                color=[COL.get(t, "#888") for t in T], label=T, edgecolor="none")
        ax.hist(g[g.pitch_type.isin(T)][a], bins=bins, histtype="step", color="#18202b", lw=1)
        ax.set_yticks([]); ax.set_xlabel(a, fontsize=9)
        for s in ["top", "right", "left"]:
            ax.spines[s].set_visible(False)
        if j == 0:
            ax.set_title(nm, loc="left", fontsize=11, fontweight="bold")
        if j == 3:
            ax.legend(fontsize=8, frameon=False, loc="upper left")
fig.suptitle("持ち球のスペクトル：各軸への射影（色は球種、黒線は合計＝観測されるスペクトル）", fontsize=11, x=.01, ha="left")
fig.tight_layout()
out = os.path.join(ROOT, "output", "figures", "15_spectra.png")
fig.savefig(out, dpi=130); print(out)
