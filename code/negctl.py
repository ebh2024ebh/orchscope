#!/usr/bin/env python3
"""Negative-control diagnostic (OrchScope). For a sample of inferred campaigns and each peer key
(port signature, primary port): the share of campaigns that are testable (port offers >= 5x
non-member peers), and the BH-flag rate of pseudo-campaigns drawn from the port population
excluding the campaign's members, under the rotation null alone and under the two-null test."""
import glob, json, sys

import numpy as np
import pandas as pd

import engine as E


def main(out="negctl.json", n_max=6000, B=100, seed=11):
    df = E.load_profiles(sorted(glob.glob("prof/week_*.tsv")))
    labels = np.load("week_results_labels.npy")
    L = pd.DataFrame(json.load(open("week_results.json"))["landscape"])
    if "residue" in L:
        L = L[~L.residue]                       # residue groups are never tested
    dmode = E.pd_numeric(df["dmode"]) if "dmode" in df.columns else None
    if len(L) > int(n_max):
        L = L.sample(int(n_max), random_state=seed)
    bits_all = [int(b, 16) if b else 0 for b in df.hbits]
    res = {}
    for key in ((sys.argv[2],) if len(sys.argv) > 2 else ("portsig", "top1")):
        rng = np.random.default_rng(seed)
        P = E.Pops(df, bits_all, key=key)
        n_all = n_ok = 0
        pr, pc = [], []
        for _, c in L.iterrows():
            mem = np.where(labels == c["id"])[0]
            if len(mem) < 10:
                continue
            n_all += 1
            port = E.campaign_port(P, mem)
            if not E.testable(P, mem, port):
                continue
            n_ok += 1
            pool = np.setdiff1d(P.port_all[port], mem, assume_unique=True)
            pseudo = rng.choice(pool, len(mem), replace=False)
            if E.convergence(pseudo, dmode)[1]:
                continue
            r = E.sync_test(pseudo, None, bits_all, 168, P, rng, B=int(B))
            pr.append(r["p_rot"]); pc.append(r["p"])
        pr, pc = np.array(pr), np.array(pc)
        res[key] = {"campaigns": n_all, "testable": n_ok,
                    "testable_frac": n_ok / max(n_all, 1),
                    "fpr_rot": float(E.bh(pr).mean()) if len(pr) else None,
                    "fpr_both": float(E.bh(pc).mean()) if len(pc) else None,
                    "raw_rot": float((pr < 0.05).mean()) if len(pr) else None,
                    "raw_both": float((pc < 0.05).mean()) if len(pc) else None}
        print(key, json.dumps(res[key]), flush=True)
    json.dump(res, open(out, "w"), indent=1)


if __name__ == "__main__":
    main(*sys.argv[1:2])
