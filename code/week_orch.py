#!/usr/bin/env python3
"""Largest orchestrated campaigns of the ORION week (OrchScope): signature rows with openjev labels
(playbook-conditioned) and the hour-by-hour activity of the ten largest, for Table III / Fig. 3."""
import glob, json

import numpy as np
import pandas as pd

import engine as E
import openjev_judge as J
from week_extras import fp_short

H = 168


def main(top=24):
    res = json.load(open("week_results.json"))
    df = E.load_profiles(sorted(glob.glob("prof/week_*.tsv")))
    labels = np.load("week_results_labels.npy")
    L = pd.DataFrame(res["landscape"])
    O = L[L.orchestrated].sort_values("bots", ascending=False).reset_index(drop=True).head(top)
    rows, mat, labs = [], [], []
    for k, c in O.iterrows():
        mem = np.where(labels == c["id"])[0]
        sig = E.signature(df, mem)
        ans, dt = J.ask(J.describe(sig, rich=True, kb=True), J.QUESTIONS)
        pt, ps, pc = J.probs(ans, "tool"), J.probs(ans, "service"), J.probs(ans, "cls")
        bits = [int(b, 16) if b else 0 for b in df.hbits.iloc[mem]]
        act = [int(sum((b >> h) & 1 for b in bits)) for h in range(H)]
        rows.append({"rank": k + 1, "id": int(c["id"]), "bots": int(c["bots"]),
                     "portsig": c["portsig"], "fp": fp_short(c["fpc"]), "fpc": c["fpc"],
                     "strategy": c["strategy"], "orchestrated": True, "testable": True,
                     "z": float(c["z"]), "n_cc": int(c["n_cc"]), "n_asn": int(c["n_asn"]),
                     "flag": c["flag"] if c["flag_share"] >= 0.5 else "none",
                     "tool": max(pt, key=pt.get) if pt else None,
                     "service": max(ps, key=ps.get) if ps else None,
                     "cls": max(pc, key=pc.get) if pc else None,
                     "coord_p": J.probs(ans, "coordinated").get("yes"),
                     "p_tool": pt, "p_service": ps, "p_cls": pc,
                     "p_sev": J.probs(ans, "severity"), "activity": act, "latency": dt})
        if k < 10:
            mat.append(act)
            labs.append("#%d" % (k + 1))
    json.dump({"rows": rows, "matrix": mat, "labels": labs,
               "days": ["Sep %d" % d for d in range(24, 31)]}, open("week_orch.json", "w"))
    print(json.dumps([{k: r[k] for k in ("rank", "bots", "portsig", "fp", "strategy", "z",
                                          "service", "tool", "flag")} for r in rows[:12]],
                     indent=0))


if __name__ == "__main__":
    main()
