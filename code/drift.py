#!/usr/bin/env python3
"""Week-scale non-stationarity of telescope populations (OrchScope): source churn across days and
day-to-day drift of each port's active population. Motivates the population null."""
import glob, json, sys

import numpy as np
import pandas as pd


def main(out="week_drift.json", min_source_days=2000):
    cols = ["portsig", "hbits"]
    df = pd.concat([pd.read_csv(p, sep="\t", usecols=cols, dtype={"hbits": str, "portsig": str},
                                keep_default_na=False)
                    for p in sorted(glob.glob("prof/week_*.tsv"))], ignore_index=True)
    D = np.zeros((len(df), 7), dtype=np.int8)
    for k, b in enumerate(df.hbits):
        v = int(b, 16) if b else 0
        for d in range(7):
            D[k, d] = 1 if (v >> (24 * d)) & 0xFFFFFF else 0
    nd = D.sum(1)
    nextday = [float((D[:, d] & D[:, d + 1]).sum() / max(D[:, d].sum(), 1)) for d in range(6)]
    g = pd.DataFrame(D).groupby(df.portsig.values).sum()
    g = g[g.sum(axis=1) >= min_source_days]
    cv = g.std(axis=1) / g.mean(axis=1)
    peak = g.max(axis=1) / g.mean(axis=1)
    res = {"profiles": int(len(df)), "one_day_frac": float((nd == 1).mean()),
           "all_week_frac": float((nd == 7).mean()), "next_day_persistence": nextday,
           "sources_per_day": D.sum(0).tolist(), "ports": int(len(g)),
           "port_cv_median": float(cv.median()), "port_cv_share_gt_0.2": float((cv > 0.2).mean()),
           "port_peak_over_mean_p90": float(peak.quantile(0.9)),
           "port_peak_over_mean_median": float(peak.median())}
    json.dump(res, open(out, "w"))
    print(json.dumps(res))


if __name__ == "__main__":
    main(*sys.argv[1:])
