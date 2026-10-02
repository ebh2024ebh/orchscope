#!/usr/bin/env python3
"""Write a SYNTHETIC data.js for the offline campaign-replay demo.

Standard library only. Typical use (python is invoked as `py` on Windows):

    py make_sample_data.py                 # writes data.js next to this script
    py make_sample_data.py --seed 7        # a different synthetic week
    py make_sample_data.py --out other.js

Every number this script writes is invented for UI development. The output
follows the schema the real pipeline will export (documented at the top of
app.js), but nothing here is a telescope measurement. Destinations are
relative offsets inside the monitored block, never IP addresses. The output
is deterministic for a given seed.
"""

from __future__ import annotations

import argparse
import bisect
import json
import math
import os
import random

HOURS = 168
WINDOW_START = "2026-09-24T00:00:00Z"   # Thu 24 Sep 2026, 00:00 UTC
BLOCK_SIZE = 2 ** 19                    # a /13 darknet
MAX_POINTS = 400
SEV_LEVELS = ["can wait", "this week", "today", "right now"]

TOOL_LABELS = ["Mirai-family", "Gafgyt", "Mozi", "ZMap", "Masscan", "Nmap",
               "Research (Censys/Shodan)", "Sality-like", "Custom", "Unknown"]
SERVICE_LABELS = ["Telnet", "SSH", "HTTP", "HTTPS/TLS", "SMB", "SIP",
                  "multi-service", "other"]
CLASS_LABELS = ["IoT-botnet propagation", "research scanning",
                "vulnerability sweep", "service discovery",
                "misclassified backscatter"]

STRATEGY_NAMES = {
    "uniform": "uniform random permutation",
    "seq_inc": "sequential increasing",
    "seq_dec": "sequential decreasing",
    "nonuniform": "non-uniform permutation",
}

# Country pools (ISO 3166-1 alpha-2 -> relative weight).
POOLS = {
    "iot": {"BR": 14, "CN": 12, "IN": 10, "VN": 8, "RU": 6, "IR": 5, "TW": 5,
            "KR": 4, "EG": 4, "TR": 4, "ID": 4, "MX": 3, "AR": 3, "TH": 3,
            "US": 3, "CO": 2, "PK": 2, "UA": 2, "RO": 2, "IT": 2, "ZA": 2,
            "PH": 2, "BD": 2, "MA": 1, "DZ": 1, "EC": 1, "PE": 1, "CL": 1},
    "mozi": {"CN": 45, "IN": 30, "RU": 3, "BR": 3, "AL": 2, "VN": 2, "KR": 2,
             "TW": 2, "US": 1},
    "research": {"US": 60, "NL": 12, "DE": 10, "GB": 8, "SG": 4, "CA": 3,
                 "JP": 2},
    "hosting": {"US": 18, "NL": 12, "DE": 10, "RU": 9, "SG": 7, "HK": 6,
                "CN": 6, "FR": 5, "GB": 5, "UA": 4, "BG": 3, "SC": 2, "PA": 2,
                "RO": 2},
    "cloud": {"CN": 22, "US": 14, "SG": 7, "DE": 6, "KR": 6, "IN": 6, "BR": 5,
              "HK": 5, "FR": 4, "NL": 4, "RU": 4, "VN": 4, "JP": 3, "ID": 3,
              "GB": 3},
    "windows": {"CN": 15, "IN": 12, "RU": 10, "VN": 9, "ID": 7, "BR": 6,
                "TW": 5, "PK": 5, "EG": 4, "TR": 4, "US": 4, "IR": 4, "TH": 3,
                "PH": 3, "MY": 3, "UA": 2},
    "sality": {"TR": 10, "IN": 10, "VN": 9, "BR": 8, "RU": 8, "EG": 7, "ID": 7,
               "TH": 6, "IR": 5, "CN": 5, "PK": 5, "RO": 4, "MX": 4, "PH": 4,
               "IT": 3},
    "mixed": {"US": 12, "CN": 10, "DE": 6, "NL": 6, "RU": 6, "FR": 5, "GB": 5,
              "BR": 5, "IN": 5, "KR": 4, "JP": 4, "SG": 4, "VN": 3, "CA": 3},
}


def hop_range(r: random.Random) -> str:
    lo = r.randint(7, 12)
    return f"{lo}\u2013{lo + r.randint(5, 10)} hops"


def fp(ttl, ipid, win, opts, seq, sport):
    return {"TTL": ttl, "IP ID": ipid, "TCP window": win, "TCP options": opts,
            "Seq number": seq, "Source port": sport}


# Header fingerprints per family (plausible, not measured).
def fp_mirai(r):
    return fp(f"64 initial, {hop_range(r)}", "random", "random per probe",
              "none", "= destination IP", "random")


def fp_mozi(r):
    return fp(f"64 initial, {hop_range(r)}", "random", "random per probe",
              "none", "random", "random")


def fp_gafgyt(r):
    return fp(f"64 initial, {hop_range(r)}", "incremental per host",
              r.choice(["5840", "14600"]), "MSS, SACK, TS, NOP, WS",
              "kernel ISN (random)", "ephemeral, sequential")


def fp_sality(r):
    return fp(f"128 initial (Windows), {hop_range(r)}", "incremental (global)",
              "n/a (UDP)", "n/a (UDP)", "n/a (UDP)", "fixed 5060 per bot")


def fp_zmap(r):
    return fp(f"255 initial, {hop_range(r)}", "54321 (ZMap constant)", "65535",
              r.choice(["none", "MSS 1460"]), "per-target validation hash",
              "range 32768\u201361000")


def fp_masscan(r):
    return fp(f"255 initial, {hop_range(r)}", "derived from target (masscan)",
              "1024", "none", "SYN cookie (target hash)", "fixed per scanner")


def fp_linux(r):
    return fp(f"64 initial (Linux), {hop_range(r)}", "per-destination counter",
              r.choice(["64240", "29200"]), "MSS, SACK, TS, NOP, WS",
              "kernel ISN (random)", "ephemeral 32768\u201360999")


def fp_windows(r):
    return fp(f"128 initial (Windows), {hop_range(r)}", "incremental (global)",
              r.choice(["8192", "64240"]), "MSS, NOP, WS, NOP, NOP, SACK",
              "kernel ISN (random)", "ephemeral 49152\u201365535")


def fp_custom_http(r):
    return fp(f"64 initial, {hop_range(r)}", "zero", "29200", "MSS 1460",
              "incremental per probe", "fixed per bot")


def fp_custom_sip(r):
    return fp(f"64 initial, {hop_range(r)}", "zero", "n/a (UDP)", "n/a (UDP)",
              "n/a (UDP)", "fixed 5060")


def fp_custom_multi(r):
    return fp(f"64 initial, {hop_range(r)}", "random",
              r.choice(["1024", "2048"]), "MSS 1460", "random",
              "fixed per scan")


def fp_custom_seq(r):
    return fp(f"64 initial, {hop_range(r)}", "zero", "14600", "none",
              "incremental per probe", "fixed per bot")


def fp_nmap(r):
    return fp(f"{r.choice(['37', '44', '52', '59'])} initial (randomized), "
              f"{hop_range(r)}", "random", "1024", "MSS 1460", "random",
              "fixed per scan")


def fp_unknown(r):
    return fp("mixed (64 / 128 initial)", "mixed", "varies", "varies",
              "random", "varies")


# family -> attributes. sev = base weights over SEV_LEVELS.
FAMILIES = {
    "mirai": dict(tool="Mirai-family IoT malware", fp=fp_mirai, pph=(6, 60),
                  pool="iot", asn=(0.15, 0.35), oj_tool="Mirai-family",
                  oj_tool_p=(0.80, 0.95), oj_tool_conf={"Mozi": 2, "Gafgyt": 2,
                                                       "Custom": 0.5},
                  oj_cls="IoT-botnet propagation", oj_cls_p=(0.72, 0.92),
                  oj_cls_conf={"vulnerability sweep": 2},
                  sev=(0.08, 0.27, 0.42, 0.23)),
    "mozi": dict(tool="Mozi-family P2P IoT malware", fp=fp_mozi, pph=(4, 30),
                 pool="mozi", asn=(0.12, 0.3), oj_tool="Mozi",
                 oj_tool_p=(0.55, 0.8), oj_tool_conf={"Mirai-family": 3,
                                                      "Gafgyt": 1},
                 oj_cls="IoT-botnet propagation", oj_cls_p=(0.7, 0.9),
                 oj_cls_conf={"vulnerability sweep": 2},
                 sev=(0.1, 0.32, 0.4, 0.18)),
    "gafgyt": dict(tool="Gafgyt-family IoT malware", fp=fp_gafgyt, pph=(3, 25),
                   pool="iot", asn=(0.15, 0.35), oj_tool="Gafgyt",
                   oj_tool_p=(0.5, 0.78), oj_tool_conf={"Mirai-family": 2.5,
                                                        "Mozi": 1},
                   oj_cls="IoT-botnet propagation", oj_cls_p=(0.65, 0.88),
                   oj_cls_conf={"vulnerability sweep": 2},
                   sev=(0.1, 0.32, 0.4, 0.18)),
    "sality": dict(tool="Sality-like P2P botnet (SIP scanning)", fp=fp_sality,
                   pph=(4.5, 6.5), pool="sality", asn=(0.3, 0.5),
                   oj_tool="Sality-like", oj_tool_p=(0.55, 0.78),
                   oj_tool_conf={"Custom": 1.5, "Unknown": 1},
                   oj_cls="service discovery", oj_cls_p=(0.5, 0.7),
                   oj_cls_conf={"vulnerability sweep": 3},
                   sev=(0.12, 0.45, 0.33, 0.1)),
    "zmap_res": dict(tool="ZMap (sharded research scanning)", fp=fp_zmap,
                     pph=(2000, 40000), pool="research", asn_abs=(1, 3),
                     oj_tool="ZMap", oj_tool_p=(0.55, 0.75),
                     oj_tool_conf={"Research (Censys/Shodan)": 4,
                                   "Masscan": 0.6},
                     oj_cls="research scanning", oj_cls_p=(0.75, 0.93),
                     oj_cls_conf={"service discovery": 2.5},
                     sev=(0.72, 0.2, 0.06, 0.02)),
    "zmap_ind": dict(tool="ZMap (independent scanners)", fp=fp_zmap,
                     pph=(800, 15000), pool="mixed", asn=(0.4, 0.8),
                     oj_tool="ZMap", oj_tool_p=(0.6, 0.85),
                     oj_tool_conf={"Research (Censys/Shodan)": 2,
                                   "Masscan": 1},
                     oj_cls="service discovery", oj_cls_p=(0.45, 0.7),
                     oj_cls_conf={"research scanning": 3},
                     sev=(0.55, 0.3, 0.12, 0.03)),
    "masscan_svc": dict(tool="Masscan (sharded scanning service)",
                        fp=fp_masscan, pph=(800, 20000), pool="hosting",
                        asn_abs=(1, 4), oj_tool="Masscan",
                        oj_tool_p=(0.6, 0.85),
                        oj_tool_conf={"ZMap": 1.5,
                                      "Research (Censys/Shodan)": 1.5},
                        oj_cls="service discovery", oj_cls_p=(0.5, 0.75),
                        oj_cls_conf={"research scanning": 2,
                                     "vulnerability sweep": 1.5},
                        sev=(0.35, 0.4, 0.2, 0.05)),
    "masscan_ind": dict(tool="Masscan (independent scanners)", fp=fp_masscan,
                        pph=(500, 12000), pool="hosting", asn=(0.4, 0.8),
                        oj_tool="Masscan", oj_tool_p=(0.6, 0.85),
                        oj_tool_conf={"ZMap": 1.5, "Nmap": 0.8},
                        oj_cls="service discovery", oj_cls_p=(0.5, 0.75),
                        oj_cls_conf={"vulnerability sweep": 2},
                        sev=(0.4, 0.38, 0.18, 0.04)),
    "ssh_bot": dict(tool="SSH brute-force botnet (Linux TCP stack)",
                    fp=fp_linux, pph=(2, 15), pool="cloud", asn=(0.25, 0.5),
                    oj_tool="Custom", oj_tool_p=(0.4, 0.6),
                    oj_tool_conf={"Unknown": 2.5, "Mirai-family": 0.6},
                    oj_cls="vulnerability sweep", oj_cls_p=(0.55, 0.8),
                    oj_cls_conf={"IoT-botnet propagation": 2.5},
                    sev=(0.08, 0.4, 0.38, 0.14)),
    "os_linux": dict(tool="Linux TCP stack (unattributed)", fp=fp_linux,
                     pph=(2, 20), pool="cloud", asn=(0.3, 0.6),
                     oj_tool="Unknown", oj_tool_p=(0.4, 0.6),
                     oj_tool_conf={"Custom": 2.5, "Nmap": 1},
                     oj_cls="service discovery", oj_cls_p=(0.45, 0.65),
                     oj_cls_conf={"vulnerability sweep": 3},
                     sev=(0.3, 0.42, 0.22, 0.06)),
    "smb_worm": dict(tool="SMB worm (Windows TCP stack)", fp=fp_windows,
                     pph=(1.5, 12), pool="windows", asn=(0.2, 0.45),
                     oj_tool="Unknown", oj_tool_p=(0.4, 0.58),
                     oj_tool_conf={"Custom": 2.5, "Sality-like": 1},
                     oj_cls="vulnerability sweep", oj_cls_p=(0.6, 0.85),
                     oj_cls_conf={"IoT-botnet propagation": 1.5},
                     sev=(0.04, 0.16, 0.42, 0.38)),
    "custom_http": dict(tool="Custom HTTP exploit scanner", fp=fp_custom_http,
                        pph=(20, 300), pool="hosting", asn=(0.2, 0.5),
                        oj_tool="Custom", oj_tool_p=(0.45, 0.7),
                        oj_tool_conf={"Unknown": 2, "Masscan": 0.8},
                        oj_cls="vulnerability sweep", oj_cls_p=(0.6, 0.85),
                        oj_cls_conf={"service discovery": 2},
                        sev=(0.05, 0.2, 0.43, 0.32)),
    "custom_sip": dict(tool="Custom SIP scanner", fp=fp_custom_sip,
                       pph=(10, 80), pool="hosting", asn=(0.2, 0.5),
                       oj_tool="Custom", oj_tool_p=(0.4, 0.6),
                       oj_tool_conf={"Sality-like": 3, "Unknown": 1},
                       oj_cls="service discovery", oj_cls_p=(0.5, 0.72),
                       oj_cls_conf={"vulnerability sweep": 3},
                       sev=(0.18, 0.45, 0.3, 0.07)),
    "custom_multi": dict(tool="Custom vertical scanner", fp=fp_custom_multi,
                         pph=(100, 2000), pool="hosting", asn=(0.2, 0.5),
                         oj_tool="Custom", oj_tool_p=(0.4, 0.6),
                         oj_tool_conf={"Nmap": 2, "Masscan": 1.5},
                         oj_cls="service discovery", oj_cls_p=(0.55, 0.8),
                         oj_cls_conf={"vulnerability sweep": 2},
                         sev=(0.2, 0.42, 0.3, 0.08)),
    "custom_seq": dict(tool="Custom sequential sweeper", fp=fp_custom_seq,
                       pph=(5, 50), pool="iot", asn=(0.2, 0.45),
                       oj_tool="Custom", oj_tool_p=(0.35, 0.55),
                       oj_tool_conf={"Gafgyt": 1.5, "Unknown": 2},
                       oj_cls="IoT-botnet propagation", oj_cls_p=(0.4, 0.6),
                       oj_cls_conf={"service discovery": 3},
                       sev=(0.2, 0.42, 0.3, 0.08)),
    "nmap": dict(tool="Nmap SYN scans (independent)", fp=fp_nmap,
                 pph=(30, 400), pool="mixed", asn=(0.5, 0.9), oj_tool="Nmap",
                 oj_tool_p=(0.5, 0.75), oj_tool_conf={"Masscan": 1.5,
                                                      "Custom": 1},
                 oj_cls="service discovery", oj_cls_p=(0.55, 0.8),
                 oj_cls_conf={"research scanning": 2},
                 sev=(0.45, 0.38, 0.14, 0.03)),
    "unknown": dict(tool="Unattributed (mixed fingerprints)", fp=fp_unknown,
                    pph=(2, 60), pool="mixed", asn=(0.4, 0.8),
                    oj_tool="Unknown", oj_tool_p=(0.45, 0.65),
                    oj_tool_conf={"Custom": 2.5, "Mirai-family": 0.8},
                    oj_cls="service discovery", oj_cls_p=(0.35, 0.55),
                    oj_cls_conf={"misclassified backscatter": 3.5},
                    sev=(0.4, 0.4, 0.16, 0.04)),
}

# family, port, proto, bots, orchestrated, strategy, activity pattern,
# optional override for openjev P(coordinated) (model disagrees with the test)
CAST = [
    ("mirai", "23+2323", "TCP", 3000, True, "uniform", "always", None),
    ("mirai", "23", "TCP", 2200, True, "uniform", "waves", None),
    ("mirai", "23+2323", "TCP", 1500, True, "uniform", "ramp", None),
    ("mirai", "8080", "TCP", 800, True, "uniform", "burst", None),
    ("mozi", "23", "TCP", 600, True, "nonuniform", "always", None),
    ("gafgyt", "23", "TCP", 400, True, "seq_inc", "waves", None),
    ("sality", "5060", "UDP", 850, True, "seq_dec", "always", None),
    ("zmap_res", "443", "TCP", 24, True, "uniform", "daily", None),
    ("zmap_res", "many ports", "TCP", 64, True, "uniform", "daily", None),
    ("masscan_svc", "445", "TCP", 32, True, "uniform", "waves", None),
    ("masscan_svc", "many ports", "TCP", 48, True, "uniform", "always", None),
    ("ssh_bot", "22", "TCP", 950, True, "nonuniform", "always", None),
    ("ssh_bot", "22", "TCP", 300, True, "seq_inc", "burst", 0.41),
    ("custom_http", "80", "TCP", 180, True, "seq_inc", "burst", None),
    ("custom_http", "8080", "TCP", 120, True, "seq_dec", "waves", None),
    ("mirai", "23+2323", "TCP", 250, True, "uniform", "burst", None),
    ("gafgyt", "80", "TCP", 350, True, "uniform", "decay", None),
    ("masscan_svc", "80", "TCP", 20, True, "uniform", "daily", None),
    ("zmap_res", "22", "TCP", 40, True, "uniform", "daily", None),
    ("mozi", "8080", "TCP", 220, True, "nonuniform", "ramp", None),
    ("custom_sip", "5060", "UDP", 140, True, "uniform", "waves", None),
    ("custom_multi", "many ports", "TCP", 90, True, "nonuniform", "burst",
     0.47),
    ("mirai", "23", "TCP", 700, True, "uniform", "waves", None),
    ("custom_http", "443", "TCP", 60, True, "seq_inc", "burst", None),
    # independent clusters: shared features, but no coordination signal
    ("smb_worm", "445", "TCP", 1200, False, "uniform", "always", None),
    ("zmap_ind", "443", "TCP", 30, False, "uniform", "always", None),
    ("masscan_ind", "many ports", "TCP", 60, False, "uniform", "always", None),
    ("os_linux", "22", "TCP", 400, False, "uniform", "always", None),
    ("nmap", "80", "TCP", 45, False, "seq_inc", "waves", None),
    ("unknown", "8080", "TCP", 85, False, "uniform", "always", None),
    ("mirai", "23", "TCP", 500, False, "uniform", "always", 0.62),
    ("smb_worm", "445", "TCP", 150, False, "uniform", "decay", None),
    ("zmap_ind", "80", "TCP", 25, False, "uniform", "waves", None),
    ("custom_sip", "5060", "UDP", 70, False, "uniform", "always", None),
    ("masscan_ind", "443", "TCP", 35, False, "uniform", "waves", None),
    ("gafgyt", "2323", "TCP", 260, False, "uniform", "always", None),
    ("os_linux", "8080", "TCP", 110, False, "uniform", "waves", None),
    ("unknown", "many ports", "TCP", 50, False, "uniform", "burst", None),
    ("nmap", "22", "TCP", 20, False, "seq_dec", "burst", None),
    ("custom_seq", "23", "TCP", 95, False, "seq_dec", "waves", None),
]


# ---------------------------------------------------------------- helpers --

def log_uniform(r, lo, hi):
    return math.exp(r.uniform(math.log(lo), math.log(hi)))


def dirichlet(r, alphas):
    xs = [r.gammavariate(max(a, 1e-3), 1.0) for a in alphas]
    s = sum(xs) or 1.0
    return [x / s for x in xs]


def clamp(x, lo, hi):
    return lo if x < lo else hi if x > hi else x


def finalize(probs):
    """Round to 3 decimals, keep every value >= 0.001, sum exactly to 1."""
    items = {k: max(0.001, round(v, 3)) for k, v in probs.items()}
    top = max(items, key=items.get)
    items[top] = round(items[top] + (1.0 - sum(items.values())), 3)
    return items


def make_dist(r, labels, true_label, p_true, confusers=None):
    confusers = confusers or {}
    others = [lab for lab in labels if lab != true_label]
    weights = [r.gammavariate(0.5, 1.0) * 0.35
               + confusers.get(lab, 0.0) * r.uniform(0.6, 1.4)
               for lab in others]
    total = sum(weights) or 1.0
    probs = {}
    for lab in labels:
        if lab == true_label:
            probs[lab] = p_true
        else:
            probs[lab] = (1.0 - p_true) * weights[others.index(lab)] / total
    return finalize(probs)


def service_for(port):
    if port in ("23", "2323", "23+2323"):
        return "Telnet", {"other": 1.5, "multi-service": 0.5}
    if port == "22":
        return "SSH", {"other": 1, "Telnet": 0.8}
    if port in ("80", "8080"):
        return "HTTP", {"HTTPS/TLS": 1.5, "other": 1.2}
    if port == "443":
        return "HTTPS/TLS", {"HTTP": 2}
    if port == "445":
        return "SMB", {"other": 1}
    if port == "5060":
        return "SIP", {"other": 1.5}
    return "multi-service", {"HTTP": 1.5, "other": 1.5}


def make_severity(r, base, bots, orchestrated):
    size = clamp(math.log10(max(bots, 1)) / 3.5, 0.0, 1.0)
    w = [base[0] * (1.25 - 0.6 * size), base[1],
         base[2] * (0.8 + 0.5 * size), base[3] * (0.6 + 0.9 * size)]
    if not orchestrated:
        w = [w[0] * 1.5, w[1] * 1.2, w[2] * 0.75, w[3] * 0.55]
    p = dirichlet(r, [max(0.03, x) * 28 for x in w])
    return finalize(dict(zip(SEV_LEVELS, p)))


# --------------------------------------------------------- activity shapes --

def pattern_always(r):
    level = r.uniform(0.3, 0.6)
    amp = r.uniform(0.05, 0.25)
    phase = r.uniform(0, 24)
    start = 0 if r.random() < 0.7 else r.randint(1, 30)
    end = HOURS - 1 if r.random() < 0.8 else r.randint(120, HOURS - 1)
    out = [0.0] * HOURS
    for h in range(start, end + 1):
        diurnal = 1 + amp * math.sin(2 * math.pi * (h - phase) / 24)
        out[h] = max(0.0, level * diurnal * (1 + 0.08 * r.gauss(0, 1)))
    return out


def pattern_waves(r):
    k = r.randint(2, 5)
    waves = [(r.uniform(6, HOURS - 8), r.uniform(3, 12), r.uniform(0.4, 0.9))
             for _ in range(k)]
    out = [0.0] * HOURS
    for h in range(HOURS):
        v = max(a * math.exp(-0.5 * ((h - c) / w) ** 2) for c, w, a in waves)
        out[h] = v if v >= 0.03 else 0.0
    return out


def pattern_ramp(r):
    start = r.randint(0, 60)
    top, mid, k = r.uniform(0.5, 0.8), r.uniform(12, 40), r.uniform(4, 10)
    out = [0.0] * HOURS
    for h in range(start, HOURS):
        v = top / (1 + math.exp(-((h - start) - mid) / k))
        out[h] = v * (1 + 0.06 * r.gauss(0, 1)) if v >= 0.02 else 0.0
    return out


def pattern_burst(r):
    dur = r.randint(5, 16)
    start = r.randint(8, HOURS - dur - 2)
    peak = r.uniform(0.55, 0.9)
    out = [0.0] * HOURS
    for i in range(dur):
        rise = min(1.0, (i + 1) / 2.0)
        fall = min(1.0, (dur - i) / 3.0)
        out[start + i] = peak * rise * fall * (1 + 0.05 * r.gauss(0, 1))
    return out


def pattern_daily(r):
    hod, dur = r.randint(0, 23), r.randint(2, 6)
    out = [0.0] * HOURS
    for day in range(HOURS // 24 + 1):
        if r.random() < 0.12:          # a skipped day now and then
            continue
        for i in range(dur):
            h = day * 24 + hod + i
            if 0 <= h < HOURS:
                out[h] = r.uniform(0.75, 1.0)
    return out


def pattern_decay(r):
    start, tau, top = r.randint(0, 40), r.uniform(15, 50), r.uniform(0.55, 0.8)
    out = [0.0] * HOURS
    for h in range(start, HOURS):
        v = top * math.exp(-(h - start) / tau)
        out[h] = v * (1 + 0.06 * r.gauss(0, 1)) if v >= 0.02 else 0.0
    return out


PATTERNS = {"always": pattern_always, "waves": pattern_waves,
            "ramp": pattern_ramp, "burst": pattern_burst,
            "daily": pattern_daily, "decay": pattern_decay}


def to_activity(r, frac, bots):
    act = []
    for f in frac:
        if f <= 0:
            act.append(0)
            continue
        n = int(round(f * bots * (1 + 0.04 * r.gauss(0, 1))))
        act.append(clamp(n, 1, bots))
    if not any(act):
        act[r.randrange(HOURS)] = 1
    return act


# ------------------------------------------------------------------ points --

def sample_times(r, act, n):
    cum, total = [], 0
    for a in act:
        total += a
        cum.append(total)
    times = []
    for _ in range(n):
        h = min(bisect.bisect_right(cum, r.uniform(0, total)), HOURS - 1)
        while act[h] == 0 and h > 0:   # guard against float edge cases
            h -= 1
        times.append(h + r.random())
    return sorted(times)


def progress_fn(act):
    cum = [0.0]
    for a in act:
        cum.append(cum[-1] + a)
    total = cum[-1] or 1.0

    def prog(t):
        h = min(int(t), HOURS - 1)
        return (cum[h] + act[h] * (t - h)) / total
    return prog


def make_points(r, strategy, act, n, sweep, orchestrated):
    ts = sample_times(r, act, n)
    pts = []
    if strategy == "uniform":
        pts = [(t, r.random()) for t in ts]
    elif strategy == "nonuniform":
        k = r.randint(2, 3)
        centers = []
        while len(centers) < k:
            c = r.uniform(0.06, 0.94)
            if all(abs(c - o) > 0.18 for o in centers):
                centers.append(c)
        sig = [r.uniform(0.006, 0.022) for _ in range(k)]
        wts = dirichlet(r, [3.0] * k)
        for t in ts:
            if r.random() < 0.08:
                u = r.random()
            else:
                j = r.choices(range(k), weights=wts)[0]
                u = r.gauss(centers[j], sig[j])
            pts.append((t, clamp(u, 0.0, 0.9999)))
    elif orchestrated:
        # bots partition the block into shards and sweep them in lock-step
        shards = r.choice([2, 3, 4, 4, 6, 8])
        prog = progress_fn(act)
        offset = r.random() * 0.25
        for t in ts:
            k = r.randrange(shards)
            f = (offset + prog(t) * sweep) % 1.0
            if strategy == "seq_dec":
                f = 1.0 - f
            u = (k + f) / shards + r.gauss(0, 0.0012)
            pts.append((t, clamp(u, 0.0, 0.9999)))
    else:
        # several unrelated sequential scanners, each on its own clock and
        # only probing while the campaign is active
        first = next(h for h, a in enumerate(act) if a)
        last = HOURS - 1 - next(h for h, a in enumerate(reversed(act)) if a)
        span = max(2.0, last + 1.0 - first)
        sweepers = []
        for t_start in sample_times(r, act, r.randint(3, 6)):
            length = span * r.uniform(0.25, 0.7)
            t0 = min(t_start, last + 1.0 - length) if length < span else first
            sweepers.append((max(first, t0), length, r.random(),
                             r.uniform(0.25, 0.9) / length))
        for _ in range(n):
            t0, length, u0, rate = r.choice(sweepers)
            for _attempt in range(25):   # reject samples in silent hours
                t = t0 + r.random() * length
                if act[min(int(t), HOURS - 1)]:
                    break
            step = rate * (t - t0)
            u = (u0 + step) % 1.0 if strategy == "seq_inc" else (u0 - step) % 1.0
            pts.append((t, clamp(u + r.gauss(0, 0.001), 0.0, 0.9999)))
        pts.sort()
    return [[round(t, 2), round(u, 4)] for t, u in pts]


# ----------------------------------------------------------------- country --

def make_countries(r, pool_name, bots):
    pool = POOLS[pool_name]
    if pool_name == "research":
        k = r.randint(1, 3)
    else:
        k = int(round(math.sqrt(bots) * r.uniform(0.35, 0.7)))
        k = clamp(k, 2, len(pool))
    k = min(k, bots)
    names, weights = list(pool), [pool[c] for c in pool]
    chosen = []
    for _ in range(k):                   # weighted sampling w/o replacement
        i = r.choices(range(len(names)), weights=weights)[0]
        chosen.append(names.pop(i))
        weights.pop(i)
    shares = dirichlet(r, [pool[c] * 0.8 + 0.3 for c in chosen])
    raw = [s * (bots - k) for s in shares]
    counts = [1 + int(x) for x in raw]   # every chosen country gets >= 1 bot
    rest = bots - sum(counts)
    order = sorted(range(k), key=lambda i: raw[i] - int(raw[i]), reverse=True)
    for i in order[:rest]:
        counts[i] += 1
    pairs = sorted(zip(chosen, counts), key=lambda p: -p[1])
    return {c: n for c, n in pairs}


# ------------------------------------------------------------------- build --

def build_campaign(r, spec):
    family, port, proto, bots, orch, strategy, pattern, coord_override = spec
    fam = FAMILIES[family]
    act = to_activity(r, PATTERNS[pattern](r), bots)
    bot_hours = float(sum(act))
    first_h = next(h for h, a in enumerate(act) if a)
    last_h = HOURS - 1 - next(h for h, a in enumerate(reversed(act)) if a)

    if strategy.startswith("seq"):
        # a sequential sweep should cross its slice 0.7-2.2 times in the window
        pph = r.uniform(0.7, 2.2) * BLOCK_SIZE / bot_hours
    else:
        pph = log_uniform(r, *fam["pph"])
    probes = bot_hours * pph
    if strategy.startswith("seq"):
        coverage = min(0.999, probes / BLOCK_SIZE)
    elif strategy == "nonuniform":
        band = r.uniform(0.12, 0.3)
        coverage = (band * (1 - math.exp(-0.92 * probes / (BLOCK_SIZE * band)))
                    + (1 - band) * (1 - math.exp(-0.08 * probes
                                                  / (BLOCK_SIZE * (1 - band)))))
    elif family in ("zmap_res", "masscan_svc"):
        coverage = min(0.999, probes / BLOCK_SIZE)     # sharded, no overlap
    else:
        coverage = 1 - math.exp(-probes / BLOCK_SIZE)
    coverage = clamp(coverage, 0.0005, 0.999)

    n_pts = int(60 + 340 * min(1.0, math.log10(probes + 1) / 6.3))
    n_pts = clamp(n_pts, 40, MAX_POINTS)
    points = make_points(r, strategy, act, n_pts, probes / BLOCK_SIZE, orch)

    # detection latency of the hourly online engine (0 = first batch)
    if orch:
        latency = r.randint(0, 2) if bots >= 500 else r.randint(1, 4) \
            if bots >= 100 else r.randint(2, 7)
    else:
        latency = r.randint(2, 6) if bots >= 300 else r.randint(3, 10)
    if first_h == 0:
        latency = max(latency, 1)        # engine warm-up at window start
    latency = min(latency, max(1, (last_h - first_h) // 2 + 1))
    first_detected_h = min(HOURS - 1, first_h + latency)

    if orch:
        size = clamp((math.log10(bots) + 0.5) / 4.0, 0.0, 1.0)
        z = 3.2 + 36.5 * r.uniform(0.05, 1.0) * size
    else:
        z = r.uniform(-1.0, 2.0)

    countries = make_countries(r, fam["pool"], bots)
    if "asn_abs" in fam:
        asns = r.randint(*fam["asn_abs"])
    else:
        asns = int(bots * r.uniform(*fam["asn"])) + 1
    asns = clamp(max(asns, min(len(countries), bots)), 1, bots)

    service, service_conf = service_for(port)
    coordinated = coord_override if coord_override is not None else (
        r.uniform(0.74, 0.985) if orch else r.uniform(0.03, 0.32))
    openjev = {
        "tool": make_dist(r, TOOL_LABELS, fam["oj_tool"],
                          r.uniform(*fam["oj_tool_p"]), fam["oj_tool_conf"]),
        "service": make_dist(r, SERVICE_LABELS, service, r.uniform(0.78, 0.96),
                             service_conf),
        "cls": make_dist(r, CLASS_LABELS, fam["oj_cls"],
                         r.uniform(*fam["oj_cls_p"]), fam["oj_cls_conf"]),
        "coordinated": round(coordinated, 3),
        "severity": make_severity(r, fam["sev"], bots, orch),
    }

    return {
        "id": 0,                          # assigned after sorting
        "bots": bots,
        "port": port,
        "proto": proto,
        "tool": fam["tool"],
        "fingerprint": fam["fp"](r),
        "strategy": STRATEGY_NAMES[strategy],
        "orchestrated": orch,
        "z": round(z, 1),
        "pph_med": round(pph, 1) if pph < 100 else float(round(pph)),
        "coverage": round(coverage, 4),
        "first_h": first_h,
        "last_h": last_h,
        "first_detected_h": first_detected_h,
        "countries": countries,
        "asns": asns,
        "activity": act,
        "points": points,
        "openjev": openjev,
    }


def build_hourly(r, campaigns):
    rows = []
    for h in range(HOURS):
        hod = h % 24                      # window starts at 00:00 UTC
        diurnal = (1 + 0.09 * math.sin(2 * math.pi * (hod - 9) / 24)
                   + 0.03 * math.sin(4 * math.pi * (hod - 3) / 24))
        trend = 1 + 0.025 * h / HOURS
        records = 2.0e7 * diurnal * trend * (1 + 0.022 * r.gauss(0, 1))
        sources = 2.45e5 * (1 + 0.11 * math.sin(2 * math.pi * (hod - 13) / 24)) \
            * (1 + 0.018 * r.gauss(0, 1))
        sources += sum(c["activity"][h] for c in campaigns)
        if 98 <= h <= 101:               # backscatter burst from a DoS victim
            records *= 1.28 - 0.06 * (h - 98)
            sources *= 1.03
        rows.append({"h": h, "records": int(records), "sources": int(sources)})
    return rows


def build(seed):
    r = random.Random(seed)
    campaigns = [build_campaign(r, spec) for spec in CAST]
    campaigns.sort(key=lambda c: (c["first_detected_h"], -c["bots"]))
    for i, c in enumerate(campaigns, start=1):
        c["id"] = i
    meta = {
        "title": "Real-time inference of orchestrated scanning campaigns",
        "telescope": "Synthetic /13 darknet (development stand-in for Merit ORION)",
        "window_start": WINDOW_START,
        "hours": HOURS,
        "n_profiles": int(r.uniform(3.6e5, 4.4e5)),
        "n_campaigns": len(campaigns),
        "n_orchestrated": sum(1 for c in campaigns if c["orchestrated"]),
        "anonymized": True,
        "synthetic": True,
        "note": ("SYNTHETIC development sample written by make_sample_data.py "
                 f"(seed {seed}). Campaigns, counts and model outputs are "
                 "invented for UI testing; they are not telescope measurements. "
                 "Destinations are relative offsets inside the monitored block; "
                 "no IP addresses are included."),
    }
    return {"meta": meta, "hourly": build_hourly(r, campaigns),
            "campaigns": campaigns}


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seed", type=int, default=2027)
    ap.add_argument("--out", default=os.path.join(here, "data.js"))
    args = ap.parse_args()

    data = build(args.seed)
    body = json.dumps(data, separators=(",", ":"), ensure_ascii=True)
    with open(args.out, "w", encoding="ascii", newline="\n") as fh:
        fh.write("window.CAMPAIGN_DATA = ")
        fh.write(body)
        fh.write(";\n")

    m = data["meta"]
    sizes = sorted(c["bots"] for c in data["campaigns"])
    print(f"wrote {args.out} ({os.path.getsize(args.out):,} bytes)")
    print(f"campaigns={m['n_campaigns']} orchestrated={m['n_orchestrated']} "
          f"bots={sum(sizes):,} (min {sizes[0]}, max {sizes[-1]}) "
          f"points={sum(len(c['points']) for c in data['campaigns']):,}")


if __name__ == "__main__":
    main()
