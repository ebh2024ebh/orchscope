#!/usr/bin/env python3
"""DarkVec on the default simulation configuration (OrchScope).

  prep : for each seed, write the reduced flow rows (synthetic campaigns plus the same real
         background sources the pipeline sees) as DarkVec input  -> dv/rows_<seed>.tsv
  eval : read DarkVec labels (dv/labels_<seed>.tsv) and score them with the simulator's
         metrics on exactly the profiles the pipeline is scored on  -> dv_results.json
"""
import argparse, collections, glob, json, os

import numpy as np
import pandas as pd

import engine as E
import simulate as S


def bg_profiles(path):
    return E.load_profiles([path])


def prep(seeds, outdir, bg_path, red_glob):
    BG = bg_profiles(bg_path)
    specs = S.mix(8, 100, 12, None)
    os.makedirs(outdir, exist_ok=True)
    per_seed, need = {}, set()
    for s in seeds:
        syn, truth, rows = S.synth_profiles(specs, s, return_rows=True)
        bg_src = set(BG[BG.portsig.isin(set(syn.portsig))].src)
        need |= bg_src
        per_seed[s] = (rows, bg_src)
    bg_rows = collections.defaultdict(list)
    for p in sorted(glob.glob(red_glob)):
        for f in E.iter_rows([p]):
            if len(f) >= 25 and f[1] in need:
                bg_rows[f[1]].append(f)
    for s, (rows, bg_src) in per_seed.items():
        with open(os.path.join(outdir, "rows_%d.tsv" % s), "w") as fh:
            for f in list(rows) + [r for src in bg_src for r in bg_rows.get(src, [])]:
                fh.write("\t".join([str(E.batch_hour(f[0], S.W0)), f[1], f[5], f[11] or "0",
                                    f[12] or "0", f[6]]) + "\n")
        print("seed", s, "rows written", flush=True)


def evaluate(seeds, outdir, bg_path, out):
    BG = bg_profiles(bg_path)
    specs = S.mix(8, 100, 12, None)
    res = []
    for s in seeds:
        syn, truth = S.synth_profiles(specs, s)
        bg = BG[BG.portsig.isin(set(syn.portsig))]
        df = pd.concat([bg, syn], ignore_index=True)
        lab = {}
        with open(os.path.join(outdir, "labels_%d.tsv" % s)) as fh:
            for line in fh:
                k, v = line.rstrip("\n").split("\t")
                lab[k] = int(v)
        labels = np.array([lab.get(x, -1) for x in df.src], dtype=np.int64)
        m = S.evaluate(df, labels, truth, None)
        m.update({"name": "abl", "seed": s, "scale": 1.0, "variant": "darkvec",
                  "n_campaigns": int(len(set(labels[labels >= 0])))})
        res.append(m)
        print(s, {k: round(v, 3) for k, v in m.items() if isinstance(v, float)}, flush=True)
    json.dump(res, open(out, "w"), indent=1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["prep", "eval"])
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--dir", default="dv")
    ap.add_argument("--bg", default="prof_bg24.tsv")
    ap.add_argument("--red", default="red/2026-09-24.*.tsv.zst")
    ap.add_argument("--out", default="dv_results.json")
    a = ap.parse_args()
    seeds = range(1, a.seeds + 1)
    if a.cmd == "prep":
        prep(seeds, a.dir, a.bg, a.red)
    else:
        evaluate(seeds, a.dir, a.bg, a.out)
