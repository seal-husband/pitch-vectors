"""
Look at the resolved components: where they sit, which Statcast labels fall in them.

Each pitch is assigned to the component with the largest share of the reconstructed
density at its location (W_pk · H_k(bin) / Σ_k W_pk · H_k(bin)), then cross-tabulated
against its Statcast label.

Usage:  python src/17_mcr_components.py abs 8
"""
import sys, os, importlib.util
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.stdout.reconfigure(encoding="utf-8")
plt.rcParams["font.family"] = ["Yu Gothic", "Meiryo", "MS Gothic", "sans-serif"]
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
spec = importlib.util.spec_from_file_location("mcr", os.path.join(ROOT, "src", "16_mcr.py"))
mcr = importlib.util.module_from_spec(spec); spec.loader.exec_module(mcr)

tag = sys.argv[1] if len(sys.argv) > 1 else "abs"
K = int(sys.argv[2]) if len(sys.argv) > 2 else 8
a, b = ("u", "w") if tag == "abs" else ("ur", "wr")
d = pd.read_parquet(os.path.join(ROOT, "output", "mcr_pitches_%d.parquet" % mcr.YEAR))
edges = mcr.grid_for(d, a, b)
X, pids = mcr.spectra(d, a, b, edges)
W, H = mcr.resolve(X, K)
nx, ny = len(edges[0]) - 1, len(edges[1]) - 1

# order components by usage
order = np.argsort(-W.sum(0)); W, H = W[:, order], H[order]

# assign each pitch
ix = np.clip(np.digitize(d[a], edges[0]) - 1, 0, nx - 1)
iy = np.clip(np.digitize(d[b], edges[1]) - 1, 0, ny - 1)
row = pd.Series(np.arange(len(pids)), index=pids)[d.pitcher].values
contrib = W[row] * H[:, ix * ny + iy].T                  # pitches × K
d["comp"] = contrib.argmax(1) + 1
d["conf"] = contrib.max(1) / contrib.sum(1)
T = ["FF", "SI", "FC", "SL", "ST", "SV", "CU", "KC", "CS", "CH", "FS", "FO"]
ct = pd.crosstab(d.comp, d.pitch_type)[[t for t in T if t in set(d.pitch_type)]]
share = W.sum(0) / W.sum()
print("%s  K=%d" % (tag, K))
print("■ 成分ごとの中身（行＝成分、各成分の中の球種ラベルの割合 %）")
tab = (100 * ct.div(ct.sum(1), axis=0)).round(0).astype(int)
tab.insert(0, "使用率%", (100 * share).round(1))
tab.insert(1, "確信度", d.groupby("comp").conf.median().round(2))
print(tab.to_string())
print("\n■ 球種ラベルごとに、どの成分へ行ったか（列＝ラベル、%）")
print((100 * ct.div(ct.sum(0), axis=1)).round(0).astype(int).to_string())
print("\n一球の割り当ての確信度 中央 %.2f   0.6未満 %.0f%%" % (d.conf.median(), 100 * (d.conf < .6).mean()))

# figure: component spectra
cols = min(K, 5); rows_ = int(np.ceil(K / cols))
fig, axs = plt.subplots(rows_, cols, figsize=(3.2 * cols, 3.3 * rows_), squeeze=False)
ext = [edges[0][0], edges[0][-1], edges[1][0], edges[1][-1]]
bg = X.mean(0).reshape(nx, ny).T
for k in range(rows_ * cols):
    ax = axs.flat[k]
    if k >= K:
        ax.axis("off"); continue
    ax.contour(bg, levels=5, extent=ext, colors="#c8ced6", linewidths=.6, origin="lower")
    ax.imshow(H[k].reshape(nx, ny).T, origin="lower", extent=ext, cmap="magma_r", aspect="equal")
    top = tab.iloc[k, 2:].sort_values(ascending=False).head(3)
    ax.set_title("成分%d  %.0f%%\n%s" % (k + 1, 100 * share[k], "  ".join("%s%d" % (t, v) for t, v in top.items() if v > 0)),
                 fontsize=9, loc="left")
    ax.axhline(0, color="#ddd", lw=.5); ax.axvline(0, color="#ddd", lw=.5)
    ax.tick_params(labelsize=7)
fig.suptitle("多変量スペクトル分解の純成分（%s, K=%d）  横＝面内の横（腕側←→グラブ側）、縦＝面内の縦" %
             ("PP上の絶対位置" if tag == "abs" else "自分の速球が原点", K), fontsize=10, x=.01, ha="left")
fig.tight_layout()
out = os.path.join(ROOT, "output", "figures", "17_mcr_%s_K%d.png" % (tag, K))
fig.savefig(out, dpi=120); print(out)
np.savez(os.path.join(ROOT, "output", "mcr_%s_K%d.npz" % (tag, K)), W=W, H=H, pids=pids,
         e0=edges[0], e1=edges[1])
