# -*- coding: utf-8 -*-
"""Figures for Paper 3 (IEEE two-column). Serif text to match the body; validated categorical
palette (Okabe-Ito subset: blue, orange, green) with markers/line styles as secondary encoding;
sequential single-hue ramps for magnitude; no dual axes; recessive grids."""
import json, os, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
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
def fig_arch():
    fig, ax = plt.subplots(figsize=(7.16, 1.32))
    ax.axis("off"); ax.set_xlim(0, 100); ax.set_ylim(-0.4, 18.6)
    txts = ["Merit ORION /13\nhourly flow records\n(~20M records/h)",
            "Pre-processing\nprobe filter, per-\nsource aggregates,\npacket samples",
            "Profiling engine\nrandomness, trend,\ndispersion tests\n+ header fingerprint",
            "Inference engine\nfingerprint blocks\n+ HDBSCAN-$\\varepsilon$\n+ residue check",
            "Orchestration engine\nrotation + peer nulls\nMonte Carlo $p$, BH",
            "AI decision engine\n(openjev): service,\ntool, strategy"]
    w, h, y, gap = 14.2, 14.0, 3.0, 2.76
    xs = [0.6 + i * (w + gap) for i in range(len(txts))]
    for k, (x, txt) in enumerate(zip(xs, txts)):
        fc = "#eef3f8" if k < 2 else "#f4f4f4"
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.2,rounding_size=1.0",
                                    fc=fc, ec=MUTED, lw=0.7))
        ax.text(x + w / 2, y + h / 2, txt, ha="center", va="center", fontsize=6.6, color=INK,
                linespacing=1.15)
    for x, x2 in zip(xs[:-1], xs[1:]):
        ax.add_patch(FancyArrowPatch((x + w + 0.35, y + h / 2), (x2 - 0.35, y + h / 2),
                                     arrowstyle="-|>", mutation_scale=7, lw=0.8, color=INK))
    ax.text((xs[0] + xs[1] + w) / 2, 0.5, "data host (at the telescope)", ha="center",
            fontsize=6.5, color=MUTED)
    ax.text((xs[2] + xs[5] + w) / 2, 0.5, "analysis server (CPU + one GPU)", ha="center",
            fontsize=6.5, color=MUTED)
    rs = os.path.join(AN, "red_sizes.json")
    mb = round(float(np.median(json.load(open(rs)))) / 1e6) if os.path.exists(rs) else 17
    ax.text(xs[1] + w + gap / 2, y + h + 0.9, "~%d MB/h" % mb, ha="center", fontsize=6.3,
            color=MUTED, style="italic")
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
