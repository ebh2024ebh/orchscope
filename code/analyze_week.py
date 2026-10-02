#!/usr/bin/env python3
"""Week-scale analysis (OrchScope): campaign inference and the orchestration test on one
week of ORION data, the campaign landscape, weak-label validation against ORION's tool
flags, and the real-traffic negative control."""
import argparse, collections, glob, json, time

import numpy as np
import pandas as pd

import engine as E


def flag_label(df):
    lab = np.full(len(df), "none", dtype=object)
    lab[df.mif.to_numpy() > 0.5] = "mirai"
    lab[df.zmf.to_numpy() > 0.5] = "zmap"
    lab[df.msf.to_numpy() > 0.5] = "masscan"
    return lab


def weak_label_scores(labels, flags):
    """Homogeneity/completeness/V-measure of campaigns w.r.t. ORION tool flags, restricted to
    flagged sources (unassigned sources become singletons)."""
    from sklearn.metrics import homogeneity_completeness_v_measure
    m = flags != "none"
    y, p = flags[m], labels[m].copy()
    nxt = (p.max() + 1) if len(p) else 0
    for i in np.where(p < 0)[0]:
        p[i] = nxt; nxt += 1
    h, c, v = homogeneity_completeness_v_measure(y, p)
    covered = float((labels[m] >= 0).mean()) if m.any() else float("nan")
    return {"homogeneity": h, "completeness": c, "v_measure": v, "flagged": int(m.sum()),
            "flagged_covered": covered}


def landscape(df, labels, camps):
    rows = []
    for c in camps:
        mem = np.where(labels == c["id"])[0]
        sub = df.iloc[mem]
        flags = collections.Counter(flag_label(sub))
        rows.append({
            "id": c["id"], "bots": int(len(mem)), "portsig": sub.portsig.mode().iat[0],
            "proto": int(sub.pm.mode().iat[0]), "fpc": sub.fpc.mode().iat[0],
            "strategy": sub.strategy.mode().iat[0], "orchestrated": bool(c.get("orchestrated")),
            "conv": float(c.get("conv", 0.0)), "residue": bool(c.get("residue", False)),
            "p": float(c.get("p", 1.0)), "p_rot": float(c.get("p_rot", 1.0)),
            "p_pop": float(c.get("p_pop", 1.0)),
            "z": float(c.get("z", 0.0)), "coact": float(c.get("coact", 0.0)),
            "z_rot": float(c.get("z_rot", 0.0)), "z_pop": float(c.get("z_pop", 0.0)),
            "orchestrated_rot": bool(c.get("orchestrated_rot")),
            "testable": bool(c.get("testable")),
            "pph_med": float(sub.pph.median()), "ud_sum": int(sub.ud.sum()),
            "span_med": float(sub.span.median()), "hfirst_min": int(sub.hfirst.min()),
            "hlast_max": int(sub.hlast.max()), "n_cc": int(sub.cc.nunique()),
            "n_asn": int(sub.asn.nunique()), "flag": flags.most_common(1)[0][0],
            "flag_share": flags.most_common(1)[0][1] / len(mem),
            "research": float(sub.research.mean()),
            "top_cc": sub.cc.value_counts().head(3).to_dict(),
            "top_org": sub.org.value_counts().head(2).to_dict(),
        })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--profiles", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--H", type=int, default=168)
    ap.add_argument("--B", type=int, default=200)
    ap.add_argument("--min-size", type=int, default=10)
    a = ap.parse_args()
    t0 = time.time()
    df = E.load_profiles(sorted(glob.glob(a.profiles)))
    t_load = time.time() - t0
    flags = flag_label(df)
    res = {"n_profiles": int(len(df)), "t_load": t_load,
           "flag_counts": dict(collections.Counter(flags))}

    t0 = time.time()
    labels, camps = E.enhanced_infer(df, a.H, min_size=a.min_size, B=a.B)
    res["t_ours"] = time.time() - t0
    res["ours"] = weak_label_scores(labels, flags)
    res["ours"].update({"campaigns": len(camps), "covered": int((labels >= 0).sum()),
                        "orchestrated": int(sum(c.get("orchestrated", False) for c in camps)),
                        "orchestrated_rot": int(sum(c.get("orchestrated_rot", False)
                                                    for c in camps)),
                        "testable": int(sum(c.get("testable", False) for c in camps)),
                        "residue": int(sum(c.get("residue", False) for c in camps)),
                        "residue_sources": int(sum(len(c["members"]) for c in camps
                                                   if c.get("residue", False)))})

    for v in ("fingerprint", "strategy", "policy"):
        l2, c2 = E.enhanced_infer(df, a.H, min_size=a.min_size, drop=(v,), use_test=False)
        res["ours_minus_" + v] = weak_label_scores(l2, flags)
        res["ours_minus_" + v]["campaigns"] = len(c2)

    res["negative_control"] = E.negative_control(df, a.H, camps, np.random.default_rng(11),
                                                 B=max(60, a.B // 2))
    res["landscape"] = landscape(df, labels, camps)
    res["strategy_mix"] = dict(collections.Counter(df.strategy))
    np.save(a.out.replace(".json", "_labels.npy"), labels)
    pd.DataFrame({"src": df.src, "camp": labels}).to_csv(a.out.replace(".json", "_assign.tsv"),
                                                         sep="\t", index=False)
    with open(a.out, "w") as fh:
        json.dump(res, fh, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    print(json.dumps({k: v for k, v in res.items() if k not in ("landscape",)}, indent=1,
                     default=str))


if __name__ == "__main__":
    main()
