#!/usr/bin/env python3
"""Export the anonymized replay dataset for the offline demo (demo/data.js) from the ORION
week results.

Nothing that identifies a source leaves this script: no addresses, ASNs, organizations, or
host names. Campaigns are summarized by counts, header-field classes, country tallies, hourly
activity, and probe destinations expressed as offsets relative to the monitored block."""
import collections, datetime, glob, json, os, random, sys

import numpy as np
import pandas as pd

import engine as E
import openjev_judge as J

H = 168
TOP = 80             # largest campaigns exported (openjev labels exist for the 60 largest)
TOP_ORCH = 40        # plus the largest orchestrated campaigns (openjev labels for 24)
MAX_POINTS = 400
MIN_SIZE = 10
W0 = datetime.datetime(2026, 9, 24)
STRAT = {"uperm": "uniform", "seq_fwd": "seq_inc", "seq_rev": "seq_dec", "perm": "nonuniform"}
PROTO = {1: "TCP", 2: "UDP", 4: "ICMP"}
SEV = ["can wait", "this week", "today", "right now"]
D0 = 591396864          # first address of the monitored /13 (offsets only are exported)


def tool_label(fpc, research):
    f = fpc.split("|")
    if "sD" in f:
        return "Mirai-family"
    if "i54321" in f:
        return "ZMap"
    if "w1024" in f and "o-" in f:
        return "Masscan"
    if research >= 0.5:
        return "Research scanner"
    return "Unknown"


def mode(series):
    vc = series.astype(str).value_counts()
    return vc.index[0] if len(vc) else ""


def fingerprint(sub, fpc):
    f = fpc.split("|")
    tcp = f[0] != "udp"
    w = next((x[1:] for x in f if x.startswith("w")), "")
    return {"TTL": mode(sub.ttl), "IP ID": J.IPID.get(mode(sub.ipid), mode(sub.ipid)),
            "TCP window": ("varies" if w in ("x", "") else w) if tcp else "-",
            "TCP options": J.opts_words(mode(sub.opts)) if tcp else "-",
            "Seq number": ("equals destination" if "sD" in f else
                           ("zero" if "sZ" in f else "other")) if tcp else "-",
            "Source port": J.SPORT.get(mode(sub.sportp), mode(sub.sportp))}


def main(out_path="demo_data.js"):
    res = json.load(open("week_results.json"))
    wo = json.load(open("week_openjev.json"))
    df = E.load_profiles(sorted(glob.glob("prof/week_*.tsv")))
    labels = np.load("week_results_labels.npy")
    L = pd.DataFrame(res["landscape"]).sort_values("bots", ascending=False).reset_index(drop=True)
    # the largest campaigns overall plus the largest orchestrated ones (the paper's subject)
    orch = L[L.orchestrated].head(TOP_ORCH)
    L = pd.concat([L.head(TOP), orch]).drop_duplicates("id").sort_values(
        "bots", ascending=False).reset_index(drop=True)
    oj = {r["id"]: r for r in wo["rows"]}
    try:
        oj.update({r["id"]: r for r in json.load(open("week_orch.json"))["rows"]})
    except OSError:
        pass
    camps, member_of = [], {}
    for k, c in L.iterrows():
        mem = np.where(labels == c["id"])[0]
        sub = df.iloc[mem]
        bits = [int(b, 16) if b else 0 for b in sub.hbits]
        act = [int(sum((b >> h) & 1 for b in bits)) for h in range(H)]
        firsts = np.sort(sub.hfirst.to_numpy())
        det = int(firsts[min(MIN_SIZE, len(firsts)) - 1]) if len(firsts) else None
        cc = sub.cc.replace("", "??").value_counts().head(12)
        rec = {"id": k + 1, "bots": int(len(mem)),
               "port": c["portsig"].lstrip("p").replace("multi", "many").replace("+", "/"),
               "proto": PROTO.get(int(c["proto"]), "mixed"),
               "tool": tool_label(c["fpc"], float(c["research"])),
               "strategy": STRAT.get(c["strategy"]),
               # untestable campaigns (too few peers on their port): orchestration undetermined
               "orchestrated": bool(c["orchestrated"]) if c.get("testable", True) else None,
               "z": round(float(c["z"]), 2) if c.get("testable", True) else None,
               "pph_med": round(float(c["pph_med"]), 2),
               "coverage": round(min(1.0, float(sub.ud.sum()) / E.DSIZE), 4),
               "first_h": int(c["hfirst_min"]), "last_h": int(c["hlast_max"]),
               "first_detected_h": det, "asns": int(c["n_asn"]),
               "fingerprint": fingerprint(sub, c["fpc"]),
               "countries": {str(a): int(b) for a, b in cc.items()},
               "activity": act, "points": []}
        r = oj.get(int(c["id"]))
        if r:
            sev = r.get("p_sev") or {}
            rec["openjev"] = {"tool": r.get("p_tool") or {}, "service": r.get("p_service") or {},
                              "cls": r.get("p_cls") or {}, "coordinated": r.get("coord_p"),
                              "severity": {s: float(sev.get(s, 0.0)) for s in SEV}}
        camps.append(rec)
        for s in sub.src:
            member_of[s] = k
    # destination offsets (relative to the monitored block) from the reduced samples
    rng = random.Random(7)
    seen = collections.Counter()
    hourly = []
    for path in sorted(glob.glob("red/*.tsv.zst") + glob.glob("red/*.tsv")):
        n_src = n_rec = 0
        for f in E.iter_rows([path]):
            if len(f) < 25:
                continue
            n_src += 1; n_rec += int(f[3])
            k = member_of.get(f[1])
            if k is None:
                continue
            hb = E.batch_hour(f[0], W0)
            if not 0 <= hb < H:
                continue
            for smp in E.parse_samples(f[24]):
                off = (smp[3] - D0) / E.DSIZE
                if not 0.0 <= off < 1.0:
                    continue
                pt = [round(hb + (smp[0] % 3600) / 3600.0, 3), round(off, 5)]
                seen[k] += 1
                pts = camps[k]["points"]
                if len(pts) < MAX_POINTS:
                    pts.append(pt)
                else:
                    j = rng.randrange(seen[k])
                    if j < MAX_POINTS:
                        pts[j] = pt
        label = os.path.basename(path).split(".tsv")[0]
        hourly.append({"h": E.batch_hour(label, W0), "records": n_rec, "sources": n_src})
    for c in camps:
        c["points"].sort()
    data = {"meta": {
        "title": "Real-time inference of orchestrated scanning campaigns",
        "telescope": "Merit ORION /13 network telescope (anonymized replay)",
        "window_start": W0.strftime("%Y-%m-%dT%H:%M:%SZ"), "hours": H,
        "n_profiles": int(res["n_profiles"]), "n_campaigns": int(res["ours"]["campaigns"]),
        "n_orchestrated": int(res["ours"]["orchestrated"]), "anonymized": True,
        "note": ("One week of Merit ORION traffic (Sep 24-30, 2026), reduced to campaign-level "
                 "summaries of the %d largest campaigns. No source addresses, ASNs, "
                 "organizations, or host names are included; probe destinations are offsets "
                 "inside the monitored block. Hourly counts cover probing records." % len(camps))},
        "hourly": sorted(hourly, key=lambda r: r["h"]), "campaigns": camps}
    with open(out_path, "w") as fh:
        fh.write("window.CAMPAIGN_DATA = ")
        json.dump(data, fh, separators=(",", ":"))
        fh.write(";\n")
    print("campaigns", len(camps), "points", sum(len(c["points"]) for c in camps),
          "hours", len(hourly))


if __name__ == "__main__":
    main(*sys.argv[1:])
