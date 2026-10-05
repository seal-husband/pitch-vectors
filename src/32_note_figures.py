"""
The four static figures for the note article (投手空間.docx).

  note_1.png  every pitch of 2026 as (横変化, 縦変化, 球速), lefties mirrored, Savant colours
  note_2.png  the same cloud edge-on to the Pitch Plane: the league PP and Misiorowski's,
              with where every pitcher's PP sits (its height c)
  note_3.png  edge-on again, every pitcher's PP moved to the league height: CH/FS sit
              below it; how far each kind of pitch sits from its own pitcher's PP
  note_4.png  the pitcher space of three.html, with Yamamoto, Misiorowski, Sale, Sánchez

Edge-on views look along a direction lying in the plane, in the coordinates the plane was
fitted in (横, 縦, 球速 × 0.63); the 3-D box is drawn to that same scale, so the plane is a line.

Reads data/statcast_2026.parquet, output/plane_2026.npz, web/pitchers_data.js (each pitcher's c),
web/three_data.js (the map). Writes output/figures/note/note_1..4.png, 1280 px wide.

Usage:  python src/32_note_figures.py
"""
import sys, os, json
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d import proj3d
sys.stdout.reconfigure(encoding="utf-8")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
from paths import SNAP

OUT = os.path.join(ROOT, "output", "figures", "note")
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"font.family": ["Yu Gothic", "Meiryo", "MS Gothic", "sans-serif"],
                     "font.size": 10, "axes.edgecolor": "#8a8a86", "axes.labelcolor": "#2b2b29",
                     "xtick.color": "#52514e", "ytick.color": "#52514e", "axes.titlesize": 11})
W, H, DPI = 8.0, 5.0, 160                      # 1280 × 800 px
INK, MUTED, GREY = "#1f1f1d", "#6b6a66", "#b9b8b3"
SOURCE = "データ：MLB Statcast（Baseball Savant）2026年レギュラーシーズン　右投手の向きに揃え、左投手は左右反転"

COL = {"FF": "#D22D49", "SI": "#FE9D00", "FC": "#933F2C", "FA": "#D22D49", "SL": "#EEE716", "ST": "#DDB33A",
       "SV": "#93AFD4", "CU": "#00D1ED", "KC": "#6236CD", "CS": "#0068FF", "CH": "#1DBE3A", "FS": "#3BACAC",
       "FO": "#55CCAB"}
JA = {"FF": "フォーシーム", "SI": "シンカー", "FC": "カッター", "SL": "スライダー", "ST": "スイーパー",
      "SV": "スラーブ", "CU": "カーブ", "KC": "ナックルカーブ", "CS": "スローカーブ", "CH": "チェンジアップ",
      "FS": "スプリット", "FO": "フォーク"}
TYPES = ["FF", "SI", "FC", "FA", "SL", "ST", "SV", "CU", "KC", "CS", "CH", "FS", "FO"]
LEG_TYPES = ["FF", "SI", "FC", "SL", "ST", "SV", "CU", "KC", "CH", "FS"]
MIN_PITCHES = 400
SEED = 32


def js(name):
    s = open(os.path.join(ROOT, "web", name), encoding="utf-8").read()
    return json.loads(s[s.index("=") + 1:].rstrip().rstrip(";"))


z = np.load(os.path.join(ROOT, "output", "plane_2026.npz"))
n, S = z["normal"], float(z["s"])
A, B = -n[1] / (n[2] * S), -n[0] / (n[2] * S)                       # mph per inch of 縦, of 横
F1 = np.array([1., 0., 0.]) - n[0] * n; F1 /= np.linalg.norm(F1)
F2 = np.cross(n, F1); F2 *= np.sign(F2[1])
PD = js("pitchers_data.js")
CP = {p["id"]: p["c"] for p in PD["pitchers"]}
C_MED = float(np.median(list(CP.values())))

d = pd.read_parquet(SNAP[2026], columns=["game_type", "pitcher", "player_name", "pitch_type", "p_throws",
                                         "pfx_x", "pfx_z", "release_speed"])
d = d[(d.game_type == "R") & d.pitch_type.isin(TYPES)].dropna().copy()
cnt = d.groupby("pitcher").size()
d = d[d.pitcher.isin(cnt[cnt >= MIN_PITCHES].index)]
for c in ["pfx_x", "pfx_z", "release_speed"]:
    d[c] = d[c].astype(float)
d["x"] = np.where(d.p_throws.eq("L"), -d.pfx_x, d.pfx_x) * 12      # + = グラブ側
d["y"] = d.pfx_z * 12
d["v"] = d.release_speed
d["c"] = d.pitcher.map(CP)
d = d[d.c.notna()]
d["r"] = d.v - (A * d.y + B * d.x + d.c)                            # mph off his own PP
rng = np.random.default_rng(SEED)
bg = d.iloc[rng.choice(len(d), 30000, replace=False)]
MISI = d[d.player_name.str.startswith("Misiorowski")]
print("投球 %d（投手 %d）  PP: 球速 = %.3f×縦 %+.3f×横 + c   c の中央 %.1f" % (len(d), d.pitcher.nunique(), A, B, C_MED))

XL, YL, ZL = (-24, 24), (-24, 24), (62, 106)


def box(ax, zoom=1.0):
    ax.set_xlim(*XL); ax.set_ylim(*YL); ax.set_zlim(*ZL)
    ax.set_box_aspect((XL[1] - XL[0], YL[1] - YL[0], (ZL[1] - ZL[0]) * S), zoom=zoom)
    ax.set_proj_type("ortho")
    ax.set_xlabel("横変化（インチ）\n← 腕側　　グラブ側 →", labelpad=6, linespacing=1.4)
    ax.set_ylabel("縦変化（インチ）", labelpad=4)
    ax.set_zlabel("球速（mph）", labelpad=4)
    for a in (ax.xaxis, ax.yaxis, ax.zaxis):
        a.set_pane_color((0.97, 0.97, 0.96, 1)); a._axinfo["grid"]["color"] = (0.86, 0.86, 0.84, 1)
        a._axinfo["grid"]["linewidth"] = 0.6
    ax.set_xticks([-20, -10, 0, 10, 20]); ax.set_yticks([-20, -10, 0, 10, 20]); ax.set_zticks([70, 80, 90, 100])
    ax.tick_params(labelsize=8, pad=0)


def look(ax, direction):
    """Point the camera from `direction`, given in plane-fit coordinates (横, 縦, 球速×s)."""
    v = direction / np.linalg.norm(direction)
    ax.view_init(elev=np.degrees(np.arcsin(v[2])), azim=np.degrees(np.arctan2(v[1], v[0])))


def side_axes(ax):
    """Seen edge-on the 横 axis runs into the page: drop its ticks and say so under the box."""
    ax.set_xticks([]); ax.set_xlabel("")
    ax.figure.text(0.36, 0.235, "横変化の向きから見た図（横変化は奥行き方向）", ha="center", va="top",
                   fontsize=8.5, color=MUTED)


def plane(ax, c, label, color, ls="-", va="bottom"):
    """The plane's outline over 横, 縦 in [-20, 20]; edge-on it is a line. Label at its left end."""
    q = np.array([[-20, -20], [20, -20], [20, 20], [-20, 20], [-20, -20]], float)
    zz = A * q[:, 1] + B * q[:, 0] + c
    ax.plot(q[:, 0], q[:, 1], zz, color=color, lw=1.6, ls=ls, zorder=5)
    sx = proj3d.proj_transform(q[:4, 0], q[:4, 1], zz[:4], ax.get_proj())[0]
    k = int(np.argmin(sx))
    ax.text(q[k, 0], q[k, 1], zz[k] + (1 if va == "bottom" else -5), label, fontsize=9, color=INK,
            ha="left", va=va, zorder=6, bbox=dict(fc="white", ec="none", alpha=.8, pad=1.5))


def type_legend(fig, types, loc, **kw):
    h = [Line2D([], [], ls="", marker="o", ms=6, mfc=COL[t], mec="none", label=JA[t]) for t in types]
    return fig.legend(handles=h, loc=loc, frameon=False, fontsize=8.5, handletextpad=0.2, **kw)


def finish(fig, name, title, sub):
    fig.text(0.03, 0.965, title, fontsize=14, fontweight="bold", color=INK, va="top")
    fig.text(0.03, 0.905, sub, fontsize=9.5, color=MUTED, va="top")
    fig.text(0.03, 0.02, SOURCE, fontsize=7.5, color=MUTED)
    p = os.path.join(OUT, name)
    fig.savefig(p, dpi=DPI, facecolor="white")
    plt.close(fig)
    print("   ", p)


# ---------------------------------------------------------------- 1  the whole cloud
fig = plt.figure(figsize=(W, H))
ax = fig.add_axes([0.0, 0.05, 0.80, 0.84], projection="3d")
box(ax, zoom=1.15)
o = bg.sample(frac=1, random_state=SEED)
ax.scatter(o.x, o.y, o.v, c=o.pitch_type.map(COL), s=1.6, alpha=.45, linewidths=0, depthshade=False)
ax.view_init(elev=14, azim=-62)
type_legend(fig, LEG_TYPES, "center right", bbox_to_anchor=(0.985, 0.5), ncol=1)
finish(fig, "note_1.png", "2026年の全投球を、変化量と球速の3次元に並べる",
       "1点が1球（30,000球を無作為に表示）。色は Baseball Savant の球種の色")

# ---------------------------------------------------------------- 2  league PP and Misiorowski
EDGE = np.cos(np.radians(20)) * F1 + np.sin(np.radians(20)) * F2      # in the plane, ~1° above it: PP seen edge-on
fig = plt.figure(figsize=(W, H))
ax = fig.add_axes([0.04, 0.12, 0.64, 0.76], projection="3d")
box(ax, zoom=1.1)
look(ax, EDGE)
side_axes(ax)
ax.scatter(bg.x, bg.y, bg.v, c=GREY, s=1.2, alpha=.35, linewidths=0, depthshade=False)
c_m = CP[MISI.pitcher.iloc[0]]
m = MISI.sample(frac=1, random_state=SEED)
ax.scatter(m.x, m.y, m.v, c=m.pitch_type.map(COL), s=5, alpha=.85, linewidths=0, depthshade=False)
plane(ax, C_MED, "リーグ中央の PP（C = %.1f）" % C_MED, "#3d3c39", va="top")
plane(ax, c_m, "Misiorowski の PP（C = %.1f）" % c_m, "#D22D49", ls=(0, (4, 2)))
fig.text(0.04, 0.083, "Misiorowski の球：", fontsize=8.5, color=INK, va="center")
type_legend(fig, [t for t in LEG_TYPES if (MISI.pitch_type == t).mean() >= .01], "center left",
            bbox_to_anchor=(0.17, 0.083), ncol=6, columnspacing=.8)

ax2 = fig.add_axes([0.74, 0.24, 0.23, 0.56])
cs = np.array(list(CP.values()))
ax2.hist(cs, bins=np.arange(np.floor(cs.min()), np.ceil(cs.max()) + .5, .5), color=GREY, edgecolor="white", linewidth=.6)
ax2.axvline(C_MED, color="#52514e", lw=1.4)
ax2.axvline(c_m, color="#D22D49", lw=2)
ax2.text(C_MED - .3, ax2.get_ylim()[1] * .97, "中央\n%.1f" % C_MED, ha="right", va="top", fontsize=8, color=INK)
ax2.text(c_m - .3, ax2.get_ylim()[1] * .6, "Misiorowski\n%.1f" % c_m, ha="right", va="top", fontsize=8, color=INK)
ax2.set_xlabel("PP の高さ C（mph）", fontsize=9)
ax2.set_ylabel("投手数", fontsize=9)
ax2.set_title("投手 %d 人の PP の高さ" % len(cs), fontsize=9.5, loc="left", color=INK)
ax2.spines[["top", "right"]].set_visible(False); ax2.tick_params(labelsize=8)
finish(fig, "note_2.png", "PP を真横から見ると、Misiorowski の球は平面ごと上にある",
       "PP：球速 ≒ %.2f×縦変化 − %.2f×横変化 + C。投手ごとに違うのは高さ C だけ（腕の振りの強さ）" % (A, -B))

# ---------------------------------------------------------------- 3  CH/FS below the plane
SLOW = ["CH", "FS", "FO"]
fig = plt.figure(figsize=(W, H))
ax = fig.add_axes([0.04, 0.12, 0.64, 0.76], projection="3d")
box(ax, zoom=1.1)
look(ax, EDGE)
side_axes(ax)
al = bg.assign(v=bg.v - (bg.c - C_MED))
rest, slow = al[~al.pitch_type.isin(SLOW)], al[al.pitch_type.isin(SLOW)]
ax.scatter(rest.x, rest.y, rest.v, c=GREY, s=1.2, alpha=.35, linewidths=0, depthshade=False)
ax.scatter(slow.x, slow.y, slow.v, c=slow.pitch_type.map(COL), s=2.4, alpha=.6, linewidths=0, depthshade=False)
plane(ax, C_MED, "PP（高さを全投手でリーグ中央に揃えた）", "#3d3c39")
type_legend(fig, ["CH", "FS", "FO"], "center left", bbox_to_anchor=(0.04, 0.083), ncol=3)

ax2 = fig.add_axes([0.765, 0.19, 0.215, 0.60])
groups = [("フォーシーム\nシンカー", ["FF", "SI", "FA"], "#D22D49"),
          ("グラブ側の変化球\n（FC含む）", ["FC", "SL", "ST", "SV", "CU", "KC", "CS"], "#DDB33A"),
          ("チェンジアップ\nスプリット", SLOW, "#1DBE3A")][::-1]
bins = np.arange(-9, 6.01, .25)
for i, (lab, ts, col) in enumerate(groups):
    r = d[d.pitch_type.isin(ts)].r
    h, e = np.histogram(r, bins=bins, density=True)
    ax2.fill_between(e[:-1], i + h / h.max() * .85, i, step="post", color=col, alpha=.75, linewidth=0)
    med = r.median()
    ax2.plot([med, med], [i, i + .9], color=INK, lw=1)
    ax2.text(5.8, i + .45, "%+.1f" % med, ha="right", va="center", fontsize=8.5, color=INK)
ax2.axvline(0, color="#52514e", lw=.8, ls=(0, (3, 2)))
ax2.set_yticks([i + .4 for i in range(3)]); ax2.set_yticklabels([g[0] for g in groups], fontsize=8)
ax2.set_xlabel("自分の PP からのずれ（mph）\n← 遅い　　速い →", fontsize=9, linespacing=1.4)
ax2.set_title("分布（黒線＝中央値）", fontsize=9.5, loc="left", color=INK)
ax2.set_xlim(-9, 6); ax2.set_ylim(-.1, 3)
ax2.spines[["top", "right", "left"]].set_visible(False); ax2.tick_params(labelsize=8, left=False)
finish(fig, "note_3.png", "チェンジアップとスプリットは、PP より少し下（遅い側）にいる",
       "変化量から予想される球速より約 3 mph 遅い。回転を殺して落とす球だからと考えられる")

# ---------------------------------------------------------------- 4  the pitcher space
TD = js("three_data.js")
P = [p for p in TD["pitchers"] if "all" in p["v"]]
X = np.array([p["v"]["all"]["z"] for p in P])      # 腕側／グラブ側
Y = np.array([p["v"]["all"]["y"] for p in P])      # グラブ側 落とす／曲げる
Z = np.array([p["v"]["all"]["x"] for p in P])      # 腕側 落とす／流す
FOUR = [("Yamamoto", "Yamamoto", "#2a78d6", "o", (.12, .18, "left", "bottom")),
        ("Misiorowski", "Misiorowski", "#eb6834", "^", (-.15, 0, "right", "center")),
        ("Sale, Chris", "Sale", "#1baf7a", "s", (-.15, .15, "right", "bottom")),
        ("Sánchez, Cristopher", "Sánchez", "#4a3aa7", "D", (.1, -.2, "left", "top"))]
fig = plt.figure(figsize=(W, H))
ax = fig.add_axes([0.0, 0.07, 0.76, 0.82], projection="3d")
lim = [(np.floor(a.min() * 2) / 2 - .1, np.ceil(a.max() * 2) / 2 + .1) for a in (X, Y, Z)]
ax.set_xlim(*lim[0]); ax.set_ylim(*lim[1]); ax.set_zlim(*lim[2])
ax.set_box_aspect([l[1] - l[0] for l in lim], zoom=.95)
ax.set_proj_type("ortho")
ax.scatter(X, Y, Z, c=GREY, s=7, alpha=.55, linewidths=0, depthshade=False)
for key, ja, col, mk, (dx, dz, ha, va) in FOUR:
    i = next(k for k, p in enumerate(P) if p["name"].startswith(key))
    ax.plot([X[i], X[i]], [Y[i], Y[i]], [lim[2][0], Z[i]], color=col, lw=.9, alpha=.7)
    ax.scatter([X[i]], [Y[i]], [lim[2][0]], color=col, s=10, alpha=.5, depthshade=False)
    ax.scatter([X[i]], [Y[i]], [Z[i]], color=col, marker=mk, s=70, edgecolors="white", linewidths=1.2,
               depthshade=False, zorder=10)
    ax.text(X[i] + dx, Y[i], Z[i] + dz, ja, fontsize=9.5, color=INK, ha=ha, va=va, zorder=11)
    print("    %-26s x %+.2f  y %+.2f  z %+.2f" % (ja, X[i], Y[i], Z[i]))
ax.set_xlabel("腕側／グラブ側（球数）\n← グラブ側が多い　腕側が多い →", labelpad=6, linespacing=1.4)
ax.set_ylabel("グラブ側：落とす／曲げる", labelpad=6)
ax.set_zlabel("腕側：落とす／流す", labelpad=4)
for a in (ax.xaxis, ax.yaxis, ax.zaxis):
    a.set_pane_color((0.97, 0.97, 0.96, 1)); a._axinfo["grid"]["color"] = (0.86, 0.86, 0.84, 1)
    a._axinfo["grid"]["linewidth"] = 0.6
ax.tick_params(labelsize=8, pad=0)
ax.set_yticks(np.arange(np.ceil(lim[1][0]), lim[1][1], 1.0))
ax.view_init(elev=20, azim=-55)
h = [Line2D([], [], ls="", marker=mk, ms=8, mfc=col, mec="white", label=ja) for _, ja, col, mk, _o in FOUR]
h.append(Line2D([], [], ls="", marker="o", ms=5, mfc=GREY, mec="none", label="その他の投手（%d人）" % (len(P) - 4)))
fig.legend(handles=h, loc="center right", bbox_to_anchor=(0.99, 0.5), frameon=False, fontsize=8.5, handletextpad=0.3)
finish(fig, "note_4.png", "投手空間：持ち球の配分で投手を並べる",
       "x は腕側とグラブ側の球数の比、y・z は各側の縦と横の変化量の比（どれも対数）。球速と PP の高さは使っていない")
