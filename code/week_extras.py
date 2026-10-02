#!/usr/bin/env python3
"""Post-analysis for the ORION week (OrchScope): activity matrix of the largest campaigns,
top-campaign signatures, and openjev characterization of real campaigns, with playbook tool
attribution scored against ORION's withheld scanner-tool flags."""
import collections, glob, json, subprocess

import numpy as np
import pandas as pd

import engine as E
import openjev_judge as J

FLAG_TOOL = {"mirai": J.TOOLS[0], "zmap": J.TOOLS[1], "masscan": J.TOOLS[2]}


def fp_short(fpc):
    f = fpc.split("|")
    if f[0] == "udp":
        head = J.ascii_head(f[4][2:]) if len(f) > 4 else ""
        return "UDP, payload '%s'" % head.strip(".")[:6] if head.strip(".") else "UDP"
    tags = []
    if "sD" in f:
        tags.append("seq=dst")
    if "i54321" in f:
        tags.append("IPID 54321")
    w = [x for x in f if x.startswith("w")]
    if w and w[0] not in ("wx", "w"):
        tags.append("win " + w[0][1:])
    o = [x for x in f if x.startswith("o")]
    if o and o[0] in ("o-",):
        tags.append("no opts")
    t = [x for x in f[1:] if x.startswith("t") and x[1:].isdigit()]
    if t:
        tags.append("TTL " + t[0][1:])
    return ", ".join(tags[:3])


def main():
    res = json.load(open("week_results.json"))
    df = E.load_profiles(sorted(glob.glob("prof/week_*.tsv")))
    labels = np.load("week_results_labels.npy")
    assert len(labels) == len(df)
    L = pd.DataFrame(res["landscape"]).sort_values("bots", ascending=False).reset_index(drop=True)
    H = 168
    mat, labs = [], []
    for k, c in L.head(10).iterrows():
        mem = np.where(labels == c["id"])[0]
        bits = [int(b, 16) if b else 0 for b in df.hbits.iloc[mem]]
        mat.append([sum((b >> h) & 1 for b in bits) for h in range(H)])
        labs.append("#%d %s (%s)" % (k + 1, c["portsig"].lstrip("p").replace("multi", "many"),
                                     fp_short(c["fpc"])))
    json.dump({"matrix": mat, "labels": labs,
               "days": ["Sep %d" % d for d in range(24, 31)]}, open("week_activity.json", "w"))

    # openjev on the largest real campaigns (playbook-conditioned) vs withheld ORION flags
    rows = []
    for k, c in L.head(60).iterrows():
        mem = np.where(labels == c["id"])[0]
        sig = E.signature(df, mem)
        ans, dt = J.ask(J.describe(sig, rich=True, kb=True), J.QUESTIONS)
        pt, ps, pc = J.probs(ans, "tool"), J.probs(ans, "service"), J.probs(ans, "cls")
        sev = ans.get("answers", {}).get("severity", {}).get("score")
        coord = J.probs(ans, "coordinated").get("yes")
        flag = c["flag"] if c["flag_share"] >= 0.5 else "none"
        rows.append({"rank": k + 1, "id": int(c["id"]), "bots": int(c["bots"]),
                     "portsig": c["portsig"], "fp": fp_short(c["fpc"]), "fpc": c["fpc"],
                     "strategy": c["strategy"], "orchestrated": bool(c["orchestrated"]),
                     "testable": bool(c.get("testable", True)),
                     "z": c["z"], "n_cc": int(c["n_cc"]), "n_asn": int(c["n_asn"]),
                     "top_cc": c["top_cc"], "flag": flag, "flag_share": c["flag_share"],
                     "research": c["research"], "pph_med": c["pph_med"],
                     "tool": max(pt, key=pt.get) if pt else None,
                     "tool_p": max(pt.values()) if pt else None,
                     "service": max(ps, key=ps.get) if ps else None,
                     "cls": max(pc, key=pc.get) if pc else None,
                     "severity": sev, "coord_p": coord, "latency": dt,
                     "p_tool": pt, "p_service": ps, "p_cls": pc,
                     "p_sev": J.probs(ans, "severity")})
    lab = [r for r in rows if r["flag"] in FLAG_TOOL]
    acc = float(np.mean([r["tool"] == FLAG_TOOL[r["flag"]] for r in lab])) if lab else None
    # total probing flow records (sum of per-source record counts over all reduced files)
    tot = 0
    for f in sorted(glob.glob("red/*.tsv.zst")):
        out = subprocess.run("zstd -dc %s | awk -F'\\t' '{s+=$4} END {print s+0}'" % f,
                             shell=True, capture_output=True, text=True).stdout.strip()
        tot += int(float(out or 0))
    json.dump({"rows": rows, "tool_vs_flags_acc": acc, "n_flag_labeled": len(lab),
               "probe_records": tot}, open("week_openjev.json", "w"), default=str)
    print(json.dumps({"tool_vs_flags_acc": acc, "n_flag_labeled": len(lab),
                      "probe_records": tot}))


if __name__ == "__main__":
    main()
