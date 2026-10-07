# -*- coding: utf-8 -*-
"""Supplementary report on the 2011 sipscan (docs/sipscan.tex): figures (PDF), every number as a
LaTeX macro (docs/sipscan_numbers.tex) and the tables (docs/sipscan_tab_*.tex), from aggregate
results only: sipscan_timeline.json, sipscan_check.json, sipscan_week_<peers>.json,
sipscan_stream_<peers>.json, rev_nores.json, week_results.json, rev_tzneg.json, rev_negp.json.
Then: latexmk -pdf docs/sipscan.tex"""
import datetime, json, os, sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AN = os.environ.get("FIG_AN", os.path.join(ROOT, "results"))
OUT = os.environ.get("DOC_OUT", os.path.join(ROOT, "docs"))
PREVIEW = os.environ.get("FIG_PREVIEW", "")
INK, MUTED, GRID = "#1f1f1f", "#5f6368", "#d9d9d9"
BLUE, ORANGE, GREEN = "#0072B2", "#E69F00", "#009E73"
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "stix", "font.size": 9, "axes.labelsize": 9, "axes.titlesize": 9,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8, "axes.edgecolor": MUTED,
    "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED, "axes.linewidth": 0.6,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.04, "pdf.fonttype": 42})
DAY0 = datetime.datetime(2011, 1, 31)
DAY0_TS = 1296432000                                       # Jan 31, 2011, 00:00 UTC
LANDMARK_H = {"onset": -96, "steady": 72, "restart": 240}  # Thursday landmarks, h since Jan 31


def load(name):
    p = os.path.join(AN, name)
    return json.load(open(p)) if os.path.exists(p) else None


def save(fig, name):
    os.makedirs(OUT, exist_ok=True)
    fig.savefig(os.path.join(OUT, name + ".pdf"), metadata={"CreationDate": None})
    if PREVIEW:
        fig.savefig(os.path.join(PREVIEW, name + ".png"), dpi=160)
    plt.close(fig)


def clean(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="y", color=GRID, lw=0.5)
    ax.set_axisbelow(True)


# ------------------------------------------------------------------ phases of the scan
def phases(T):
    """Hours since Jan 31, 00:00 UTC, from the hourly bots per /13 slice: onset (first packet),
    decline (first hour of day 6 below 0.9 of the same hour a day earlier), restart (first hour
    after day 7 above ten times the idle median), stop (end of the last such hour)."""
    y = np.array(T["bots_per_slice_mean"])
    onset = (T["first"] - DAY0_TS) / 3600.0
    dec = next(h for h in range(6 * 24, 7 * 24) if y[h] < 0.9 * y[h - 24])
    idle = float(np.median(y[8 * 24:11 * 24]))
    on = [h for h in range(7 * 24, len(y)) if y[h] > 10 * idle]
    return {"onset": onset, "decline": dec, "restart": on[0], "stop": on[-1] + 1,
            "idle_lo": float(y[8 * 24:on[0]].min()), "idle_hi": float(y[8 * 24:on[0]].max()),
            "peak": float(y.max())}


def events(ax, P, label=True, y=1.0):
    for name in ("onset", "decline", "restart", "stop"):
        h = P[name]
        ax.axvline(h, color=MUTED, lw=0.6, ls=(0, (3, 2)), zorder=0)
        if label:                       # restart and stop are a day apart: restart to the left
            ax.text(h - 2 if name == "restart" else h + 2, y, name,
                    transform=ax.get_xaxis_transform(), ha="right" if name == "restart" else "left",
                    va="top", fontsize=8, color=MUTED)


def day_axis(ax, lo, hi, every):
    ticks = np.arange(lo, hi + 1)
    ax.set_xticks(ticks * 24)
    ax.set_xticklabels([(DAY0 + datetime.timedelta(days=int(d))).strftime("%b %d")
                        if d % every == 0 else "" for d in ticks])
    ax.set_xlim(lo * 24, hi * 24)


def timeline_panel(ax, T, P):
    y = np.array(T["bots_per_slice_mean"])
    x = np.arange(len(y)) + 0.5
    ax.fill_between(x, y, step="mid", color=BLUE, alpha=0.18, lw=0)
    ax.plot(x, y, color=BLUE, lw=0.9, drawstyle="steps-mid")
    ax.set_ylabel("bots per hour\nin a /13 slice")
    ax.set_ylim(0, y.max() * 1.2)
    clean(ax)
    events(ax, P)


# ------------------------------------------------------------------ figures
def fig_timeline(m=100):
    T = load("sipscan_timeline.json")
    P = phases(T)
    W = {p: load("sipscan_week_%s.json" % p) for p in ("scan", "all")}
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(6.4, 4.2), sharex=True,
                                 gridspec_kw={"height_ratios": [1.0, 1.1], "hspace": 0.2})
    timeline_panel(a1, T, P)
    xs = np.arange(8) * 24 + 84                        # centre of each 7-day window
    series = [("scan", "flag", BLUE, "o", "-", "two-null, residue-free peers"),
              ("all", "flag", GREEN, "s", "-", "two-null, as deployed"),
              ("scan", "flag_rot", ORANGE, "^", "--", "rotation null alone")]
    for p, key, col, mk, ls, lab in series:
        r = {x["w"]: x[key] for x in W[p]["by_window_size"] if x["m"] == m}
        a2.plot(xs, [100 * r.get(w, np.nan) for w in range(8)], color=col, marker=mk, ms=4,
                lw=1.1, ls=ls, label=lab)
    for key, col in (("flag", BLUE), ("flag_rot", ORANGE)):
        c = {x["m"]: x[key] for x in W["scan"]["controls_by_size"]}
        a2.axhline(100 * c[m], color=col, lw=0.8, ls=(0, (1, 1.5)))
    a2.text(0.995, 100 * {x["m"]: x["flag_rot"] for x in W["scan"]["controls_by_size"]}[m] + 2,
            "random groups of ORION scanners", transform=a2.get_yaxis_transform(), ha="right",
            va="bottom", fontsize=7.5, color=MUTED)
    for w in range(8):
        a2.plot([w * 24, w * 24 + 168], [-7 - 2.2 * w] * 2, color=MUTED, lw=0.8,
                solid_capstyle="butt")
    a2.set_ylim(-26, 104)
    a2.set_yticks([0, 25, 50, 75, 100])
    a2.set_ylabel("sub-campaigns flagged (%%)\n(m = %d bots, weekly windows)" % m)
    clean(a2)
    events(a2, P, label=False)
    a2.legend(loc="lower center", frameon=False, handlelength=2.2, ncol=3,
              bbox_to_anchor=(0.5, 0.995), columnspacing=1.2)
    day_axis(a2, 0, 14, 2)
    a2.set_xlabel("2011 (UTC); bars below: the eight 7-day windows, points at their centres")
    save(fig, "sipscan_timeline")


def stream_series(S, tag, key):
    """Share of the 32 slices whose campaign (tag: full | m30) is flagged at each update, per
    landmark window, on the common clock (hours since Jan 31, 00:00 UTC)."""
    out = {}
    rows = [r for r in S["rows"] if tag + "_p" in r]
    for ev, h0 in LANDMARK_H.items():
        hs = sorted({r["h"] for r in rows if r["ev"] == ev})
        out[ev] = (np.array([h0 + h + 1 for h in hs]),
                   np.array([np.mean([r[tag + "_" + key] for r in rows
                                      if r["ev"] == ev and r["h"] == h]) for h in hs]))
    return out


def fig_stream(tag="m30"):
    T = load("sipscan_timeline.json")
    P = phases(T)
    S = {p: load("sipscan_stream_%s.json" % p) for p in ("scan", "all")}
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(6.4, 3.9), sharex=True,
                                 gridspec_kw={"height_ratios": [0.8, 1.2], "hspace": 0.2})
    timeline_panel(a1, T, P)
    series = [("scan", "flag", BLUE, "-", "two-null, residue-free peers"),
              ("all", "flag", GREEN, "-", "two-null, as deployed"),
              ("scan", "flag_rot", ORANGE, "--", "rotation null alone")]
    for p, key, col, ls, lab in series:
        for k, (hx, v) in enumerate(stream_series(S[p], tag, key).values()):
            a2.plot(hx, 100 * v, color=col, lw=1.1, ls=ls, label=lab if k == 0 else None,
                    drawstyle="steps-post")
    for h0 in LANDMARK_H.values():
        a2.axvspan(h0, h0 + 47, color=GRID, alpha=0.45, lw=0)
    a2.text(LANDMARK_H["steady"] + 23.5, 14, "no decision\nbefore two\nwhole days", ha="center",
            va="center", fontsize=7, color=MUTED)
    a2.set_ylim(-3, 103)
    a2.set_yticks([0, 25, 50, 75, 100])
    a2.set_ylabel("slices whose 30-bot\nsub-campaign is flagged (%)")
    clean(a2)
    events(a2, P, label=False)
    a2.legend(loc="lower center", frameon=False, handlelength=2.2, ncol=3,
              bbox_to_anchor=(0.5, 0.995), columnspacing=1.2)
    day_axis(a2, -4, 17, 3)
    a2.set_xlim(-96, 408)
    a2.set_xlabel("2011 (UTC), hourly updates; weekly windows start on Thursdays (Jan 27, Feb 3, "
                  "Feb 10)")
    save(fig, "sipscan_stream")


# ------------------------------------------------------------------ replay statistics
def onset_stats(S):
    """First update after the onset with a campaign (>= 10 profiled bots) in some slice: how many
    slices hold one and how many are flagged; first update with all 32 flagged; last update at
    which the whole campaign is testable; 30-bot sub-campaigns' flagged share to the window's end."""
    R = [r for r in S["rows"] if r["ev"] == "onset"]
    h1 = min(r["h"] for r in R if r["n"] >= 10)
    first = [r for r in R if r["h"] == h1]
    by_h = {}
    for r in R:
        by_h.setdefault(r["h"], []).append(r)
    all_flag = min(h for h, X in by_h.items()
                   if len(X) == 32 and all(r["n"] >= 10 and r["full_flag"] for r in X))
    last_test = max(h for h, X in by_h.items() if any(r["full_testable"] for r in X))
    return {"h1": h1, "camps_first": sum(r["n"] >= 10 for r in first),
            "flag_first": sum(r["n"] >= 10 and r["full_flag"] for r in first),
            "h_all": all_flag, "last_testable_h": last_test,
            "m30_window": float(np.mean([r["m30_flag"] for r in R if "m30_p" in r]))}


def restart_stats(S):
    R = [r for r in S["rows"] if r["ev"] == "restart"]
    h1 = min(r["h"] for r in R)
    first = [r for r in R if r["h"] == h1]
    return {"h1": h1, "m30_first": sum(r.get("m30_flag", False) for r in first),
            "slices": len(first), "full_testable_first": sum(r["full_testable"] for r in first)}


def steady_stats(S):
    R = [r for r in S["rows"] if r["ev"] == "steady" and "m30_p" in r]
    by = {}
    for r in R:
        by.setdefault(r["h"], []).append(r["m30_flag_rot"])
    return {"flagged": int(sum(r["m30_flag"] for r in R)), "updates": len(by),
            "slice_updates": len(R), "rot_max": float(max(np.mean(v) for v in by.values()))}


def deployed_stats(S):
    """Replay with all port-5060 peers: most slices flagged after the onset and when; the update
    at which ORION's residue event enters the peer pool; restart updates flagged before and
    after it."""
    on = [r for r in S["rows"] if r["ev"] == "onset"]
    by = {}
    for r in on:
        by.setdefault(r["h"], []).append(r["n"] >= 10 and r["full_flag"])
    h_max = max(by, key=lambda h: (sum(by[h]), -h))
    rs = [r for r in S["rows"] if r["ev"] == "restart"]
    burst = min(r["h"] for r in rs if r["peers"] > 10 * min(x["peers"] for x in rs))
    return {"onset_max": sum(by[h_max]), "onset_max_h": h_max, "burst_h": burst,
            "restart_after_burst": int(sum(r.get("m30_flag", False) for r in rs
                                           if r["h"] >= burst)),
            "restart_before_burst": float(np.mean([r["m30_flag"] for r in rs
                                                   if r["h"] < burst and "m30_p" in r]))}


# ------------------------------------------------------------------ LaTeX outputs
def num(x):
    return "{:,}".format(int(round(x))).replace(",", "{,}")


def pct(x, d=0):
    return ("%." + str(d) + "f") % (100 * x)


def at(h):
    """Hours since Jan 31, 00:00 -> '23:00 on Jan 31'."""
    d = DAY0 + datetime.timedelta(hours=float(h))
    return "%s on %s %d" % (d.strftime("%H:%M"), d.strftime("%b"), d.day)


def window_label(w):
    d = DAY0 + datetime.timedelta(days=w)
    return "%s %d" % (d.strftime("%b"), d.day)


def window_phases(w, P):
    lo, hi = 24 * w, 24 * w + 168
    out = []
    for name, (a, b) in (("onset", (P["onset"], P["onset"])),
                         ("steady", (P["onset"] + 1, P["decline"])),
                         ("decline", (P["decline"], P["decline"] + 24)),
                         ("idle", (P["decline"] + 24, P["restart"])),
                         ("restart", (P["restart"], P["restart"])),
                         ("stop", (P["stop"], P["stop"]))):
        if a < hi and b >= lo:
            out.append(name)
    return ", ".join(out)


def tex(path, lines):
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, path), "w", encoding="utf-8", newline="\n") as fh:
        fh.write("% generated by code/sipscan_doc.py -- do not edit\n" + "\n".join(lines) + "\n")


def tables(Ws, Wa, P, N, P0, T0, NG):
    rows = [r"\begin{tabular}{@{}llrrrrrrrrr@{}}", r"\toprule",
            r" & & \multicolumn{3}{c}{$m=30$} & \multicolumn{3}{c}{$m=100$} & "
            r"\multicolumn{3}{c}{$m=1{,}000$} \\",
            r"\cmidrule(lr){3-5}\cmidrule(lr){6-8}\cmidrule(l){9-11}",
            r"Window from & Holds & RF & Dep. & Rot. & RF & Dep. & Rot. & RF & Dep. & Rot. \\",
            r"\midrule"]
    for w in range(8):
        cells = []
        for m in (30, 100, 1000):
            a = next(x for x in Ws["by_window_size"] if x["w"] == w and x["m"] == m)
            b = next(x for x in Wa["by_window_size"] if x["w"] == w and x["m"] == m)
            cells += [pct(a["flag"]), pct(b["flag"]), pct(a["flag_rot"])]
        rows.append("%s & %s & %s \\\\" % (window_label(w), window_phases(w, P), " & ".join(cells)))
    rows += [r"\bottomrule", r"\end{tabular}"]
    tex("sipscan_tab_windows.tex", rows)

    rows = [r"\begin{tabular}{@{}rrrrrrr@{}}", r"\toprule",
            r" & \multicolumn{3}{c}{sipscan} & "
            r"\multicolumn{3}{c}{random groups} \\",
            r"\cmidrule(lr){2-4}\cmidrule(l){5-7}",
            r"$m$ & RF & Dep. & Rot. & RF & Dep. & Rot. \\", r"\midrule"]
    for m in (10, 30, 100, 300, 1000):
        s = next(x for x in Ws["by_size"] if x["m"] == m)
        d = next(x for x in Wa["by_size"] if x["m"] == m)
        c = next(x for x in Ws["controls_by_size"] if x["m"] == m)
        e = next(x for x in Wa["controls_by_size"] if x["m"] == m)
        rows.append("%s & %s & %s & %s & %s & %s & %s \\\\" % (
            num(m), pct(s["flag"], 1), pct(d["flag"], 1), pct(s["flag_rot"], 1),
            pct(c["flag"], 1), pct(e["flag"], 1), pct(c["flag_rot"], 1)))
    rows.append(r"\midrule")
    rows.append(r"groups per row & \multicolumn{3}{c}{%s} & \multicolumn{3}{c}{%s} \\" % (
        num(Ws["by_size"][0]["groups"]), num(Ws["controls_by_size"][0]["groups"])))
    rows += [r"\bottomrule", r"\end{tabular}"]
    tex("sipscan_tab_sizes.tex", rows)

    nc, tz = N["negctl_port"], N["negctl_country"]
    rows = [r"\begin{tabular}{@{}lrr@{}}", r"\toprule",
            r" & Paper (as deployed) & Residue-free peers \\", r"\midrule",
            r"Testable campaigns & %s & %s \\" % (num(N["testable_week"]), num(N["testable"])),
            r"Orchestrated (BH, $q=0.05$) & %s & %s \\" % (num(N["orchestrated_week"]),
                                                          num(N["orchestrated_nores"])),
            r"\quad flagged by both & %s & %s \\" % (num(N["both"]), num(N["both"])),
            r"\quad TCP / UDP & %s / %s & %s / %s \\" % (
                num(N["by_proto_week"]["1"]), num(N["by_proto_week"]["2"]),
                num(N["by_proto_nores"]["1"]), num(N["by_proto_nores"]["2"])),
            r"Same-port control: flagged / groups & %s / %s & %s / %s \\" % (
                num(P0["flagged"]), num(P0["pseudo_campaigns"]), num(nc["flagged"]),
                num(nc["pseudo_campaigns"])),
            r"\quad uncorrected $p<0.05$ / $p<0.01$ & %s\%% / %s\%% & %s\%% / %s\%% \\" % (
                pct(NG["raw_p_lt_0.05"], 1), pct(NG["raw_p_lt_0.01"], 1),
                pct(nc["raw_p_lt_0.05"], 1), pct(nc["raw_p_lt_0.01"], 1)),
            r"Same-country control: flagged / groups & %s / %s & %s / %s \\" % (
                num(round(T0["fpr"] * T0["pseudo_campaigns"])), num(T0["pseudo_campaigns"]),
                num(tz["flagged"]), num(tz["pseudo_campaigns"])),
            r"\quad uncorrected $p<0.05$ / $p<0.01$ & %s\%% / %s\%% & %s\%% / %s\%% \\" % (
                pct(T0["raw_p_lt_0.05"], 1), pct(T0["raw_p_lt_0.01"], 1),
                pct(tz["raw_p_lt_0.05"], 1), pct(tz["raw_p_lt_0.01"], 1)),
            r"\bottomrule", r"\end{tabular}"]
    tex("sipscan_tab_nores.tex", rows)


def numbers(T, Ck, Ws, Wa, Ss, Sa, N, P0, T0, P):
    M = {}
    M["SipPackets"] = num(T["total_packets"])
    M["SipBots"] = num(T["total_bots"])
    M["SipCountries"] = num(T["countries"])
    M["SipOnsetTime"] = at(P["onset"])
    M["SipDeclineTime"] = at(P["decline"])
    M["SipRestartTime"] = at(P["restart"])
    M["SipStopTime"] = at(P["stop"])
    M["SipIdleLo"], M["SipIdleHi"] = num(P["idle_lo"]), num(P["idle_hi"])
    M["SipPeakSlice"] = num(P["peak"])
    M["SipSteadyDays"] = "%.1f" % ((P["decline"] - P["onset"]) / 24)
    M["SipIdleDays"] = "%.1f" % ((P["restart"] - P["decline"]) / 24)
    M["SipRestartHours"] = num(P["stop"] - P["restart"])
    M["SipCheckBots"], M["SipCheckDmode"] = num(Ck["bots"]), num(Ck["dmode"])
    prof = np.array(Ws["profiled_bots"])
    M["SipSlices"], M["SipWindows"] = num(prof.shape[1]), num(prof.shape[0])
    M["SipSliceWindows"] = num(prof.size)
    M["SipProfMin"] = num(np.median(prof, 1).min())
    M["SipProfMax"] = num(np.median(prof, 1).max())
    M["SipPerSize"] = num(Ws["by_size"][0]["groups"])
    M["SipDraws"] = num(Ws["by_size"][0]["groups"] / prof.size)
    M["SipPerCell"] = num(Ws["by_size"][0]["groups"] / prof.shape[0])
    M["SipCtlPerSize"] = num(Ws["controls_by_size"][0]["groups"])
    M["SipCtlTotal"] = num(sum(x["groups"] for x in Ws["controls_by_size"]))
    o5 = Ws["orion_5060"]
    M["OrionFive"], M["OrionResidue"] = num(o5["sources"]), num(o5["residue"])
    M["OrionResiduePct"] = pct(o5["residue"] / o5["sources"])
    M["OrionTopDest"], M["OrionOneHour"] = num(o5["top_dest_sources"]), num(o5["single_hour_max"])
    M["OrionScanners"] = num(Ws["orion_5060_sources"])
    M["OrionScanCountries"] = num(len(Ws["orion_5060_countries_ge40"]))
    M["OrionScanCountryList"] = ", ".join(Ws["orion_5060_countries_ge40"])
    M["OrionAllCountries"] = num(len(Wa["orion_5060_countries_ge40"]))
    M["SipFallbackPct"] = pct(np.mean([x["fallback"] for x in Ws["by_size"]]))
    M["SipFallbackPctDep"] = pct(np.mean([x["fallback"] for x in Wa["by_size"]]))
    M["SipCountryCovPct"] = pct(Ws["sip_share_cc_ge40_orion"], 1)
    M["SipTau"], M["SipTauRot"] = "%.4f" % Ws["bh_threshold"], "%.4f" % Ws["bh_threshold_rot"]
    M["WeekTested"] = num(Ws["week_tested"])
    w7 = {x["m"]: x for x in Ws["by_window_size"] if x["w"] == 7}
    w6 = {x["m"]: x for x in Ws["by_window_size"] if x["w"] == 6}
    M["SipSevenThirty"], M["SipSevenHundred"] = pct(w7[30]["flag"]), pct(w7[100]["flag"])
    M["SipSevenThousand"] = pct(w7[1000]["flag"])
    M["SipSixThirty"], M["SipSixThousand"] = pct(w6[30]["flag"]), pct(w6[1000]["flag"])
    early = [x["flag"] for x in Ws["by_window_size"] if x["w"] <= 3]
    M["SipEarlyMax"] = pct(max(early))
    M["SipRotEarlyMin"] = pct(min(x["flag_rot"] for x in Ws["by_window_size"]
                                  if x["w"] <= 1 and x["m"] >= 30))
    M["SipDeployedMax"] = pct(max(x["flag"] for x in Wa["by_window_size"]))
    ck = sum(round(x["flag"] * x["tested"]) for x in Ws["controls_by_size"])
    ca = sum(round(x["flag"] * x["tested"]) for x in Wa["controls_by_size"])
    cn = sum(x["tested"] for x in Ws["controls_by_size"])
    M["SipCtlFlag"], M["SipCtlFlagPct"] = num(ck), pct(ck / cn, 1)
    M["SipCtlFlagDep"], M["SipCtlFlagDepPct"] = num(ca), pct(ca / cn, 1)
    M["SipCtlRotMax"] = pct(max(x["flag_rot"] for x in Ws["controls_by_size"]))
    M["SipCtlOwnBH"] = num(Ws["controls_bh_own"]["all"]["flagged"])
    M["SipCtlOwnBHDep"] = num(Wa["controls_bh_own"]["all"]["flagged"])
    M["SipCtlOwnBHRot"] = num(Ws["controls_bh_own"]["all"]["flagged_rot"])
    ob, oa = Ws["controls_bh_own"]["all"]["flagged"], Wa["controls_bh_own"]["all"]["flagged"]
    M["SipCtlOwnBHText"] = ("none is flagged with either peer set" if ob == oa == 0 else
                            "%s are flagged with residue-free peers and %s as deployed"
                            % (num(ob), num(oa)))
    M["SipCtlNactMed"] = num(np.median([x["nact_med"] for x in Ws["controls_by_size"]]))
    M["SipNactMed"] = num(np.median([x["nact_med"] for x in Ws["by_size"]]))
    F = Ws["full"]
    M["SipFullTestable"] = num(sum(f["testable"] for f in F))
    fr = {w: np.mean([f["p_rot"] <= Ws["bh_threshold_rot"] for f in F if f["w"] == w])
          for w in range(8)}
    fp = {w: np.mean([f["p_pop_waived"] <= Ws["bh_threshold"] for f in F if f["w"] == w])
          for w in range(8)}
    M["SipFullRotFirst"], M["SipFullRotLast"] = pct(fr[0]), pct(fr[7])
    M["SipFullRotFrom"] = window_label(min(w for w in range(8) if fr[w] >= 0.99))
    M["SipFullPopFirst"], M["SipFullPopLast"] = pct(fp[0]), pct(fp[7])
    o, r, st = onset_stats(Ss), restart_stats(Ss), steady_stats(Ss)
    M["SipOnsetFirstUpdate"] = at(LANDMARK_H["onset"] + o["h1"] + 1)
    M["SipOnsetCamps"], M["SipOnsetFlag"] = num(o["camps_first"]), num(o["flag_first"])
    M["SipOnsetAll"] = at(LANDMARK_H["onset"] + o["h_all"] + 1)
    M["SipOnsetLastTestable"] = at(LANDMARK_H["onset"] + o["last_testable_h"] + 1)
    M["SipOnsetSubPct"] = pct(o["m30_window"])
    M["SipRestartFirstUpdate"] = at(LANDMARK_H["restart"] + r["h1"] + 1)
    M["SipRestartDelay"] = num(LANDMARK_H["restart"] + r["h1"] + 1 - P["restart"])
    M["SipRestartFirst"], M["SipRestartSlices"] = num(r["m30_first"]), num(r["slices"])
    M["SipRestartFullTestable"] = num(r["full_testable_first"])
    M["SipSteadyFlagged"], M["SipSteadyUpdates"] = num(st["flagged"]), num(st["updates"])
    M["SipSteadySliceUpdates"] = num(st["slice_updates"])
    M["SipSteadyRotMax"] = pct(st["rot_max"])
    d = deployed_stats(Sa)
    M["SipDepOnsetMax"] = num(d["onset_max"])
    M["SipDepOnsetAt"] = at(LANDMARK_H["onset"] + d["onset_max_h"] + 1)
    M["SipDepBurst"] = at(LANDMARK_H["restart"] + d["burst_h"] + 1)
    M["SipDepRestartAfter"] = num(d["restart_after_burst"])
    M["SipDepRestartAfterText"] = ("never after it" if d["restart_after_burst"] == 0 else
                                   "in %s slice-updates after it" % num(d["restart_after_burst"]))
    M["SipDepRestartBefore"] = pct(d["restart_before_burst"])
    M["SipReplayUpdates"] = num(len({(x["ev"], x["h"]) for x in Ss["rows"]}))
    M["WeekOrch"], M["NoresOrch"] = num(N["orchestrated_week"]), num(N["orchestrated_nores"])
    M["NoresBoth"], M["NoresDrop"], M["NoresGain"] = (num(N["both"]), num(N["week_only"]),
                                                      num(N["nores_only"]))
    M["NoresAdd"] = num(N["orchestrated_nores"] - N["orchestrated_week"])
    M["NoresTestable"], M["WeekTestable"] = num(N["testable"]), num(N["testable_week"])
    M["NoresNegN"], M["NoresNegFlag"] = (num(N["negctl_port"]["pseudo_campaigns"]),
                                         num(N["negctl_port"]["flagged"]))
    M["NoresTzN"], M["NoresTzFlag"] = (num(N["negctl_country"]["pseudo_campaigns"]),
                                       num(N["negctl_country"]["flagged"]))
    M["PaperNegN"], M["PaperTzN"] = num(P0["pseudo_campaigns"]), num(T0["pseudo_campaigns"])
    lines = ["\\newcommand{\\%s}{%s}" % (k, v) for k, v in sorted(M.items())]
    tex("sipscan_numbers.tex", lines)
    return M


def report():
    T, Ck = load("sipscan_timeline.json"), load("sipscan_check.json")
    Ws, Wa = load("sipscan_week_scan.json"), load("sipscan_week_all.json")
    Ss, Sa = load("sipscan_stream_scan.json"), load("sipscan_stream_all.json")
    N = load("rev_nores.json")
    P0, T0 = load("week_results.json")["negative_control"], load("rev_tzneg.json")
    NG = load("rev_negp.json")
    P = phases(T)
    tables(Ws, Wa, P, N, P0, T0, NG)
    M = numbers(T, Ck, Ws, Wa, Ss, Sa, N, P0, T0, P)
    print("macros:", len(M))


if __name__ == "__main__":
    for f in (sys.argv[1:] or ["timeline", "stream", "report"]):
        (report if f == "report" else globals()["fig_" + f])()
