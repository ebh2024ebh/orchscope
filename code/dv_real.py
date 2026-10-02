#!/usr/bin/env python3
"""DarkVec versus the pipeline on one real ORION day (OrchScope).

  prep : write the day's reduced flow rows as DarkVec input            -> dv_real/rows.tsv
  eval : score DarkVec's clusters and the pipeline's one-day campaigns against ORION's withheld
         tool tags (homogeneity over tagged sources; unassigned sources are singletons)
                                                                        -> dv_real.json
"""
import argparse, glob, json, os

import numpy as np

import engine as E
from analyze_week import flag_label, weak_label_scores

DAY = "2026-09-30"


def prep(outdir):
    os.makedirs(outdir, exist_ok=True)
    import datetime
    w0 = datetime.datetime.strptime(DAY, "%Y-%m-%d")
    n = 0
    with open(os.path.join(outdir, "rows.tsv"), "w") as fh:
        for p in sorted(glob.glob("red/%s.*.tsv.zst" % DAY)):
            for f in E.iter_rows([p]):
                if len(f) < 25:
                    continue
                fh.write("\t".join([str(E.batch_hour(f[0], w0)), f[1], f[5], f[11] or "0",
                                    f[12] or "0", f[6]]) + "\n")
                n += 1
    print("rows", n)


def evaluate(outdir, out, labels="labels.tsv"):
    df = E.load_profiles(sorted(glob.glob("prof/day6_*.tsv")))      # Sep 30 profiles
    flags = flag_label(df)
    ours = np.load("camp_day_6_labels.npy")
    assert len(ours) == len(df)
    lab = {}
    with open(os.path.join(outdir, labels)) as fh:
        for line in fh:
            k, v = line.rstrip("\n").split("\t")
            lab[k] = int(v)
    dv = np.array([lab.get(s, -1) for s in df.src], dtype=np.int64)
    res = {"profiles": int(len(df)), "darkvec_vocab": len(lab),
           "ours": weak_label_scores(ours, flags), "darkvec": weak_label_scores(dv, flags)}
    for k, lb in (("ours", ours), ("darkvec", dv)):
        ids, cnt = np.unique(lb[lb >= 0], return_counts=True)
        res[k].update({"clusters": int(len(ids)), "covered": float((lb >= 0).mean()),
                       "median_size": float(np.median(cnt)) if len(cnt) else 0.0,
                       "max_size": int(cnt.max()) if len(cnt) else 0})
    # do DarkVec clusters mix header fingerprints (machinery) that our campaigns keep apart?
    fpc = df.fpc.astype(str).to_numpy()
    for k, lb in (("ours", ours), ("darkvec", dv)):
        pur = []
        for c in np.unique(lb[lb >= 0]):
            m = lb == c
            _, cnt = np.unique(fpc[m], return_counts=True)
            pur.append(cnt.max() / m.sum())
        res[k]["fp_purity_median"] = float(np.median(pur)) if pur else 0.0
    json.dump(res, open(out, "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["prep", "eval"])
    ap.add_argument("--dir", default="dv_real")
    ap.add_argument("--out", default="dv_real.json")
    ap.add_argument("--labels", default="labels.tsv")
    a = ap.parse_args()
    prep(a.dir) if a.cmd == "prep" else evaluate(a.dir, a.out, a.labels)
