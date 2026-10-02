# -*- coding: utf-8 -*-
"""Figures for Paper 3 (IEEE two-column). Serif text to match the body (the Fig. 1 diagram is
sans-serif); validated categorical palette (Okabe-Ito subset: blue, orange, green) with
markers/line styles as secondary encoding; sequential single-hue ramps for magnitude; no dual
axes; recessive grids."""
import json, os, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties, findfont, get_font
from matplotlib.patches import FancyArrowPatch, Rectangle, Ellipse, Polygon
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AN, PA = os.path.join(ROOT, "results"), os.path.join(ROOT, "out")
AN = os.environ.get("FIG_AN", AN)
PA = os.environ.get("FIG_PA", PA)
INK, MUTED, GRID = "#1f1f1f", "#5f6368", "#d9d9d9"
BLUE, ORANGE, GREEN = "#0072B2", "#E69F00", "#009E73"
SEQ = ["#9ecae1", "#4292c6", "#08519c"]          # one hue, light -> dark (ordinal)
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "stix", "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7, "axes.edgecolor": MUTED,
    "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED, "axes.linewidth": 0.6,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.02, "pdf.fonttype": 42})


PREVIEW = os.environ.get("FIG_PREVIEW", "")
os.makedirs(PA, exist_ok=True)


def save(fig, name):
    fig.savefig(os.path.join(PA, name + ".pdf"))
    if PREVIEW:
        fig.savefig(os.path.join(PREVIEW, name + ".png"), dpi=220)
    plt.close(fig)


def clean(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="y", color=GRID, lw=0.5)
    ax.set_axisbelow(True)


# ------------------------------------------------------------------ Fig. 1: architecture
# Drawn 1:1 at its printed size (full text width x 99.4 pt, the height of the earlier box
# diagram), so every font prints at its nominal size (>= 6.5 pt, subscripts aside): two lanes
# (data host | analysis server), six stages with one icon each, and the a_i path that bypasses
# grouping. Numbers come from the same sources as make_numbers.py.
ARCH_W, ARCH_H = 516.0, 99.4                                  # pt
RED, DRED, SLATE, LGRAY, RULE = "#e3311d", "#a3200f", "#56636f", "#c4c4c4", "#b0b0b0"
HOUR_RECORDS_M, HOUR_GB, HOUR_SOURCES_K = 20, 11, 148         # one-hour probe


def fig_arch():
    sans = ["Segoe UI", "Segoe UI Symbol", "Arial", "DejaVu Sans"]
    rc = {"font.family": sans, "mathtext.fontset": "custom", "mathtext.rm": "Segoe UI:semibold",
          "mathtext.it": "Segoe UI:italic:semibold", "mathtext.bf": "Segoe UI:bold",
          "mathtext.fallback": "stixsans", "savefig.bbox": None}
    rs = os.path.join(AN, "red_sizes.json")
    mb = round(float(np.median(json.load(open(rs)))) / 1e6) if os.path.exists(rs) else 17
    lat = []
    for v in ("plain", "rich", "kb"):
        p = os.path.join(AN, "openjev_sim_%s.json" % v)
        lat += json.load(open(p))["latency_s"] if os.path.exists(p) else []
    sec = "%.2f s" % float(np.median(lat)) if lat else "0.29 s"
    W, gap, gap_d = (ARCH_W - 1.2 - 4 * 11.0 - 18.0) / 6, 11.0, 18.0
    xs = [0.6 + k * (W + gap) + (gap_d - gap) * (k >= 2) for k in range(6)]
    yb, yt, xd = 16.6, 86.6, (xs[1] + W + xs[2]) / 2           # box bottom/top, lane divider
    labels = ["SENSOR", "REDUCTION", "01 · PROFILING", "02 · INFERENCE", "03 · ORCHESTRATION",
              "04 · AI DECISION"]
    titles = [["Merit ORION /13"], ["Pre-processing"], ["Behavior +", "fingerprint"],
              ["Blocks →", "HDBSCAN-ε"], ["Two-null test"], ["Typed questions"]]
    details = [["~%d M records/h" % HOUR_RECORDS_M, "%d GB · ≤3 pkt samples" % HOUR_GB],
               ["%d× smaller" % round(HOUR_GB * 1e3 / mb, -1), "~%d k sources/h" % HOUR_SOURCES_K],
               [r"$x_i \in \mathrm{ℝ}^d \cdot f_i \cdot \pi_i \cdot a_i$"],
               [r"groups ($\pi$, $f$, mode)", "residue set aside"],
               [r"exact Monte Carlo $p$", r"$\max(p_{\rm rot}, p_{\rm pop})$ → BH"],
               ["service · tool ·", "coordination · strategy"]]
    with plt.rc_context(rc):
        fig = plt.figure(figsize=(ARCH_W / 72, ARCH_H / 72), dpi=72)
        ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off")
        ax.set_xlim(0, ARCH_W); ax.set_ylim(0, ARCH_H)          # 1 data unit = 1 pt

        def caps(x, y, s, size, weight, color, track, maxw):
            """Letter-spaced capitals; the tracking tightens until the label fits maxw."""
            f = get_font(findfont(FontProperties(family=sans, weight=weight)))   # first installed
            f.set_size(size, 72)
            adv = [f.load_char(ord(c)).linearHoriAdvance / 65536 for c in s]
            if sum(adv) > maxw:
                print("fig_arch: %r is %.1f pt too wide" % (s, sum(adv) - maxw))
            track = max(0.0, min(track, (maxw - sum(adv)) / max(1, len(s) - 1)))
            for c, a in zip(s, adv):
                if c != " ":
                    ax.text(x, y, c, fontsize=size, fontweight=weight, color=color, va="baseline")
                x += a + track

        def dot(x, y, d, c):
            ax.add_patch(Ellipse((x, y), d, d, fc=c, ec="none"))

        def bar(x, y, w, h, c):
            ax.add_patch(Rectangle((x, y), w, h, fc=c, ec="none"))

        def arrow(a, b, c, lw, ms):
            ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=ms, lw=lw, color=c,
                                         shrinkA=0, shrinkB=0))

        def ic_sensor(x, y0, y1):                     # dark addresses; a few are hit
            for r, row in enumerate(("000000", "001010", "010001")):
                for c, v in enumerate(row):
                    dot(x + 1.5 + 9.0 * c, y1 - 2.0 - 4.5 * r, 2.6 if v == "1" else 2.1,
                        RED if v == "1" else LGRAY)

        def ic_reduce(x, y0, y1):                     # many records in, one stream out
            ym = (y0 + y1) / 2
            for j in range(5):
                ax.plot([x, x + 13], [y1 - 1.5 - 2.5 * j] * 2, color=LGRAY, lw=1.1,
                        solid_capstyle="butt")
            ax.add_patch(Polygon([(x + 16, y1), (x + 31, ym + 2.2), (x + 31, ym - 2.2), (x + 16, y0)],
                                 closed=True, fc="none", ec=INK, lw=0.9))
            arrow((x + 31, ym), (x + 43, ym), RED, 1.1, 5.5)

        def ic_profile(x, y0, y1):                    # activity series + header barcode
            zz = [(0, .35), (4, .35), (6.5, .08), (9, .9), (11.5, .15), (14, .62), (16.2, .38),
                  (18.4, .62), (20.5, .45)]
            ax.plot([x + a for a, _ in zz], [y0 + b * (y1 - y0) for _, b in zz], color=INK, lw=0.9)
            xb = x + 26
            for w, c in ((1.7, INK), (.8, INK), (1.7, INK), (.8, INK), (1.3, INK), (.6, INK),
                         (1.7, INK), (.8, RED), (1.7, RED)):
                bar(xb, y0, w, y1 - y0, c); xb += w + 1.1

        def ic_infer(x, y0, y1):                      # two campaigns amid background
            for a, b in ((1, 10), (4, 2.5), (9.5, 1.2), (52, 11), (55.5, 3.5)):
                dot(x + a, y0 + b, 2.0, LGRAY)
            for cx, cy, c in ((22, 8.6, INK), (36, 4.4, RED)):
                ax.add_patch(Ellipse((x + cx, y0 + cy), 19, 8.4, fc="none", ec=c, lw=0.9,
                                     ls=(0, (1, 1.1))))
                for a, b in ((-4.5, 0), (0, 2.1), (4.5, 0), (0, -2.1)):
                    dot(x + cx + a, y0 + cy + b, 2.4, c)

        def ic_orch(x, y0, y1):                       # aligned activity vs. shifted surrogates
            yl = y0 + 5.2
            ax.plot([x, x + 56], [yl, yl], color=INK, lw=0.8, dashes=(2.2, 1.3))
            for xa in (x + 13, x + 34):
                for j in range(3):
                    bar(xa, yl + 1.6 + 2.4 * j, 10, 1.4, RED)
            for a, b, dy in ((1, 8, 2.4), (22, 31, 2.4), (47, 55, 2.4), (5, 12, 4.8), (16, 25, 4.8),
                             (37, 45, 4.8)):
                bar(x + a, yl - dy - 0.7, b - a, 1.4, LGRAY)

        def ic_ai(x, y0, y1):                         # per-option probabilities
            y1 -= 0.8
            bar(x, y1 - 2.4, 34, 2.4, RED)
            ax.text(x + 36.5, y1 - 1.2, "0.91", fontsize=6.5, fontweight="bold", color=INK,
                    va="center")
            for j, w in enumerate((13, 9, 5)):
                bar(x, y1 - 2.4 - 3.6 * (j + 1), w, 2.2, LGRAY)

        icons = (ic_sensor, ic_reduce, ic_profile, ic_infer, ic_orch, ic_ai)
        for s, a, b in (("DATA HOST — AT THE TELESCOPE", 0, 1),
                        ("ANALYSIS SERVER — CPU · HOURLY · ALG. 1", 2, 4), ("GPU · PER CAMPAIGN", 5, 5)):
            caps(xs[a], 92.6, s, 6.5, "bold", INK, 0.55, xs[b] + W - xs[a])
            ax.plot([xs[a], xs[b] + W], [89.6, 89.6], color=RULE, lw=0.6, solid_capstyle="butt")
        ax.plot([xd, xd], [1.0, ARCH_H - 1.0], color="#b9b9b9", lw=0.7, dashes=(3, 2.2))
        for k, x in enumerate(xs):
            ax.add_patch(Rectangle((x, yb), W, yt - yb, fc="white", ec=INK, lw=0.9))
            caps(x + 5, 77.6, labels[k], 6.5, "normal", SLATE if k < 2 else RED, 0.4, W - 10)
            icons[k](x + 5, 60.5, 73.5)
            for j, t in enumerate(titles[k]):
                ax.text(x + 5, 51.5 - 9.6 * j, t, fontsize=8, fontweight="bold", color=INK,
                        va="baseline")
            ax.plot([x + 5, x + W - 5], [37.0, 37.0], color=RULE, lw=0.6)
            for j, t in enumerate(details[k]):
                ax.text(x + 5, 29.2 - 8.0 * j, t, fontsize=6.5, fontweight="semibold", color=DRED,
                        va="baseline")
            if k < 5:
                arrow((x + W + 0.5, (yb + yt) / 2), (xs[k + 1] - 0.6, (yb + yt) / 2), INK, 0.9, 6.5)
        xa, xb, yp = xs[2] + W / 2, xs[4] + W / 2, 7.5           # a_i bypasses grouping
        ax.plot([xa, xa, xb, xb], [yb, yp, yp, yb - 3.4], color=RED, lw=0.9, dashes=(3, 2))
        arrow((xb, yb - 3.8), (xb, yb), RED, 0.9, 6)
        ax.text((xa + xb) / 2, yp, r"$a_i$ withheld from grouping", fontsize=6.5,
                fontweight="semibold", color=DRED, ha="center", va="center",
                bbox=dict(fc="white", ec="none", pad=1.2))
        ax.text(xd - 3, yp, "~%d MB/h" % mb, fontsize=6.5, fontweight="semibold", color=MUTED,
                ha="right", va="center")
        ax.text(xs[5] + 5, yp, "%s each" % sec, fontsize=6.5, fontweight="semibold", color=MUTED,
                va="center")
        fig.canvas.draw()                                         # overflow check (1 px = 1 pt)
        ren = fig.canvas.get_renderer()
        for t in ax.texts:
            x, y = t.get_position()
            k = max(i for i in range(6) if xs[i] <= x + 0.01) if x >= xs[0] else 0
            if yb < y < yt and t.get_window_extent(ren).x1 > xs[k] + W - 3.0:
                print("fig_arch: %r overflows box %d by %.1f pt" % (
                    t.get_text(), k, t.get_window_extent(ren).x1 - xs[k] - W + 3.0))
        save(fig, "fig_arch")


# ------------------------------------------------------------------ Fig. 2: sensitivity
def fig_sens():
    r = pd.DataFrame(json.load(open(os.path.join(AN, "sim_results.json"))))
    m = r.groupby(["name", "variant"]).ari.agg(["mean", "std"]).reset_index()
    fig, (a, b) = plt.subplots(1, 2, figsize=(3.5, 1.52), gridspec_kw={"width_ratios": [1.05, 1]})
    rates = [3, 12, 60]
    for n, col, mk in zip([25, 100, 400], SEQ, ["o", "s", "D"]):
        ys = [m[(m.name == "sweep_r%d_n%d" % (rt, n)) & (m.variant == "ours")]["mean"].iat[0]
              for rt in rates]
        es = [m[(m.name == "sweep_r%d_n%d" % (rt, n)) & (m.variant == "ours")]["std"].iat[0]
              for rt in rates]
        a.errorbar(rates, ys, yerr=es, color=col, marker=mk, ms=4, lw=1.4, capsize=2,
                   label="%d bots" % n)
    a.set_xscale("log"); a.set_xticks(rates); a.set_xticklabels([str(x) for x in rates])
    a.set_xlabel("per-bot rate (pps)"); a.set_ylabel("ARI"); a.set_ylim(0.1, 1.0)
    a.legend(frameon=False, loc="lower center", ncol=3, handlelength=1.0, columnspacing=0.8,
             handletextpad=0.3, borderaxespad=0.05, fontsize=6.3)
    a.set_title("(a) rate and campaign size", loc="left"); clean(a)
    labels = ["/13", "/16", "/20", "4", "8", "16"]
    names = ["scale_/13", "scale_/16", "scale_/20", "conc_K4", "conc_K8", "conc_K16"]
    vals = [m[(m.name == nm) & (m.variant == "ours")]["mean"].iat[0] for nm in names]
    errs = [m[(m.name == nm) & (m.variant == "ours")]["std"].iat[0] for nm in names]
    xs = [0, 1, 2, 3.9, 4.9, 5.9]
    b.bar(xs, vals, yerr=errs, width=0.72, color=BLUE, ecolor=MUTED, capsize=2, lw=0)
    for x, v, e in zip(xs, vals, errs):
        b.text(x, v + e + 0.03, "%.2f" % v, ha="center", fontsize=6.3, color=INK)
    b.set_xticks(xs); b.set_xticklabels(labels, fontsize=6.5); b.set_ylim(0, 1.12)
    b.text(1, -0.36, "telescope", ha="center", fontsize=6.5, color=MUTED,
           transform=b.get_xaxis_transform())
    b.text(4.9, -0.36, "campaigns/port", ha="center", fontsize=6.5, color=MUTED,
           transform=b.get_xaxis_transform())
    b.set_title("(b) stress tests (ARI)", loc="left"); clean(b)
    fig.tight_layout(w_pad=0.8)
    save(fig, "fig_sens")


# ------------------------------------------------------------------ Fig. 4: week
def fig_week():
    p = os.path.join(AN, "week_results.json")
    if not os.path.exists(p):
        print("skip fig_week (no week_results.json)"); return
    w = json.load(open(p))
    L = pd.DataFrame(w["landscape"])
    po = os.path.join(AN, "week_orch.json")     # largest ORCHESTRATED campaigns (Table III)
    act = json.load(open(po if os.path.exists(po) else os.path.join(AN, "week_activity.json")))
    fig, (a, b) = plt.subplots(1, 2, figsize=(3.5, 1.55),
                               gridspec_kw={"width_ratios": [1, 1.25]})
    for flag, col, mk, lab in ((True, BLUE, "o", "orchestrated"), (False, ORANGE, "^", "other")):
        s_ = np.sort(L[L.orchestrated == flag].bots.to_numpy())[::-1]
        if len(s_):
            y = np.arange(1, len(s_) + 1) / len(s_)
            a.loglog(s_, y, color=col, marker=mk, ms=2.5, lw=1.0,
                     markevery=max(1, len(s_) // 10), label=lab)
    a.set_xlabel("bots per campaign"); a.set_ylabel("CCDF")
    a.legend(frameon=False, loc="lower left", fontsize=6.2, handlelength=1.4,
             borderaxespad=0.1)
    a.set_title("(a) campaign sizes", loc="left"); clean(a)
    M = np.array(act["matrix"], dtype=float)
    b.imshow(M / np.maximum(M.max(1, keepdims=True), 1), aspect="auto", cmap="Blues",
             interpolation="nearest", vmin=0, vmax=1)
    labs = [l.split()[0] for l in act["labels"]]
    b.set_yticks(range(len(labs))); b.set_yticklabels(labs, fontsize=6)
    H = M.shape[1]
    b.set_xticks(np.arange(12, H, 24)); b.set_xticklabels([str(24 + d) for d in range(H // 24)],
                                                          fontsize=6.5)
    for x in range(24, H, 24):
        b.axvline(x - 0.5, color="white", lw=0.6)
    b.set_xlabel("day of September 2026 (UTC)")
    b.set_title("(b) ten largest orchestrated", loc="left")
    for sp in b.spines.values():
        sp.set_visible(False)
    b.tick_params(length=0, pad=1.5)
    fig.tight_layout(w_pad=0.6)
    save(fig, "fig_week")


# ------------------------------------------------------------------ Fig.: case study
def fig_case():
    p = os.path.join(AN, "week_case.json")
    if not os.path.exists(p):
        print("skip fig_case (no week_case.json)"); return
    panels = json.load(open(p))["panels"]
    fig, axes = plt.subplots(1, len(panels), figsize=(3.5, 1.4), sharey=False)
    from matplotlib.colors import ListedColormap
    cmap = ListedColormap(["#ffffff", BLUE])
    titles = {"orchestrated": "orchestrated", "surrogate": "rotation null",
              "peers": "population null", "co-tooled": "co-tooled"}
    for k, (ax, pn) in enumerate(zip(np.atleast_1d(axes), panels)):
        M = np.array([[int(ch) for ch in r] for r in pn["rows"]], dtype=float)
        H = M.shape[1]
        ax.imshow(M, aspect="auto", cmap=cmap, interpolation="nearest", vmin=0, vmax=1)
        for x in range(24, H, 24):
            ax.axvline(x - 0.5, color=GRID, lw=0.4)
        ax.set_xticks(np.arange(12, H, 48))
        ax.set_xticklabels([str(24 + d) for d in range(0, H // 24, 2)], fontsize=5.8)
        ax.tick_params(length=0, pad=1.5)
        ax.set_yticks([])
        sub = "$T$=%.2f" % pn["T"] + ("" if pn["z"] is None else ", $z$=%.0f" % pn["z"])
        ax.set_title("(%s) %s\n%s" % ("abcd"[k], titles.get(pn["kind"], pn["kind"]), sub),
                     loc="left", fontsize=6.2, linespacing=1.1)
        for s in ax.spines.values():
            s.set_color(MUTED); s.set_linewidth(0.4)
        if k == 0:
            ax.set_ylabel("bots (by onset)", fontsize=6.6)
    fig.text(0.5, -0.02, "day of September 2026 (UTC)", ha="center", fontsize=6.6, color=INK)
    fig.tight_layout(w_pad=0.6)
    save(fig, "fig_case")


# ------------------------------------------------------------------ Fig. 5: real time
def fig_rt():
    p = os.path.join(AN, "stream_0930.json")
    if not os.path.exists(p):
        print("skip fig_rt (no stream_0930.json)"); return
    s = pd.DataFrame(json.load(open(p)))
    if "warm" in s:
        s = s[s.warm != True].reset_index(drop=True)
    fig, (a, b) = plt.subplots(2, 1, figsize=(3.5, 1.62), sharex=True,
                               gridspec_kw={"height_ratios": [1.4, 1], "hspace": 0.45})
    h = (s.hour.to_numpy() % 24)
    ing = (s.t_read + s.t_ingest).to_numpy(); pro = s.t_profile.to_numpy()
    inf = s.t_infer.to_numpy()
    kw = dict(width=0.8, edgecolor="white", linewidth=0.4)
    a.bar(h, ing, color=BLUE, label="ingest", **kw)
    a.bar(h, pro, bottom=ing, color=ORANGE, label="profiles", **kw)
    a.bar(h, inf, bottom=ing + pro, color=GREEN, label="inference + test", **kw)
    a.set_ylabel("seconds"); a.set_ylim(0, 1.3 * float((ing + pro + inf).max()))
    a.legend(frameon=False, ncol=3, loc="upper left", fontsize=6.3, handlelength=1.0,
             columnspacing=0.8, borderaxespad=0.1)
    a.set_title("(a) processing time per hourly batch", loc="left"); clean(a)
    scan = s.campaigns - s.residue if "residue" in s else s.campaigns   # residue is not scanning
    b.plot(h, scan, color=BLUE, lw=1.2, marker="o", ms=2.3, label="scanning campaigns")
    b.plot(h, s.orchestrated, color=ORANGE, lw=1.2, marker="^", ms=2.3, ls="--",
           label="orchestrated")
    b.set_ylim(0, 1.5 * float(scan.max()))
    b.legend(frameon=False, ncol=2, loc="upper left", fontsize=6.3, handlelength=1.6,
             borderaxespad=0.1)
    b.set_ylabel("count"); b.set_xlabel("hour of Sep. 30 (UTC)")
    b.set_xticks(range(0, 24, 3))
    b.set_title("(b) campaigns reported after each batch", loc="left"); clean(b)
    save(fig, "fig_rt")


if __name__ == "__main__":
    only = sys.argv[1:]
    for f in (fig_arch, fig_sens, fig_week, fig_case, fig_rt):
        if not only or f.__name__ in only:
            f()
    print("figures written")
