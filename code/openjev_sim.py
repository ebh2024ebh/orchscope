#!/usr/bin/env python3
"""Evaluate the openjev decision layer on ground-truth synthetic campaigns (OrchScope).

Each true campaign is summarized (fingerprint + behavior + activity profile; never tool labels)
and openjev answers typed questions; strategy is read from a rendered destination-vs-time plot.
Compared against the statistical strategy classifier and the synchrony test."""
import collections, itertools, json, sys, time

import numpy as np

import engine as E
import openjev_judge as J
import simulate as S

TOOL_TRUTH = {"mirai": J.TOOLS[0], "zmap": J.TOOLS[1], "masscan": J.TOOLS[2], "linux": J.TOOLS[4],
              "winsyn": J.TOOLS[4], "sipudp": J.TOOLS[5]}
SERVICE_TRUTH = {"mirai": J.SERVICES[0], "zmap": J.SERVICES[2], "masscan": J.SERVICES[2],
                 "linux": J.SERVICES[1], "winsyn": J.SERVICES[3], "sipudp": J.SERVICES[4]}


def activity_line(bits, H):
    frac = [sum((b >> h) & 1 for b in bits) / len(bits) for h in range(H)]
    return "Fraction of these sources active in each hour of the day (hours 0-%d): %s." % (
        H - 1, ", ".join("%.2f" % f for f in frac))


def top(d):
    k = max(d, key=d.get)
    return k, d[k]


def main(out_path, seeds=(1, 2, 3), rich=True, kb=False):
    rows, lat = [], []
    for seed in seeds:
        specs = [{"tool": t, "bots": 60, "rate": r, "strategy": st, "orchestrated": o,
                  "window": (0, 24)}
                 for (t, st, o), r in zip(itertools.product(list(S.TOOLS), S.STRATS if hasattr(
                     S, "STRATS") else ["uperm", "seq_fwd", "seq_rev", "perm"], [True, False]),
                     itertools.cycle([12, 60]))]
        df, truth, acc = S.synth_profiles(specs, seed=seed, return_acc=True)
        idx = {s: i for i, s in enumerate(df.src)}
        by = collections.defaultdict(list)
        for s, (cid, st, orch) in truth.items():
            if s in idx:
                by[cid].append(idx[s])
        bits_all = [int(b, 16) if b else 0 for b in df.hbits]
        rng = np.random.default_rng(seed)
        for cid, mem in by.items():
            if len(mem) < 5:
                continue
            spec = specs[cid]
            sig = E.signature(df, np.array(mem))
            _, p_test, z = E.coordination_test(np.array(mem), bits_all, 24, rng, B=150)
            st_stat = collections.Counter(s for s in df.strategy.iloc[mem] if s != "undet")
            st_stat = st_stat.most_common(1)[0][0] if st_stat else "undet"
            state = J.describe(sig, rich=rich, kb=kb) + "\n" + activity_line(
                [bits_all[i] for i in mem], 24)
            ans, dt = J.ask(state, J.QUESTIONS)
            lat.append(dt)
            pts = []
            for i in mem:
                a = acc[df.src.iat[i]]
                pts += [(s[0], s[3] - S.D0) for s in a.sam if 0 <= s[3] - S.D0 < S.DSIZE]
            ans_img, dt2 = J.ask({"task": "Classify the scanning strategy in the image.",
                                  "image": J.strategy_png(pts)}, J.STRAT_Q)
            lat.append(dt2)
            r = {"seed": seed, "cid": cid, "tool": spec["tool"], "strategy": spec["strategy"],
                 "orchestrated": spec["orchestrated"], "n": len(mem), "z_test": z,
                 "p_test": p_test, "strategy_stat": st_stat, "raw": ans, "raw_img": ans_img}
            for q in ("tool", "service", "cls", "coordinated", "severity"):
                r["p_" + q] = J.probs(ans, q)
            r["p_strategy_img"] = J.probs(ans_img, "strategy")
            rows.append(r)
            print(cid, spec["tool"], spec["strategy"], spec["orchestrated"], "->",
                  top(r["p_tool"]) if isinstance(r["p_tool"], dict) and r["p_tool"] else r["p_tool"],
                  flush=True)
    res = {"rows": rows, "latency_s": lat}
    ok = [r for r in rows if isinstance(r["p_tool"], dict) and r["p_tool"]]
    if ok:
        res["tool_acc"] = float(np.mean([top(r["p_tool"])[0] == TOOL_TRUTH[r["tool"]] for r in ok]))
        res["service_acc"] = float(np.mean([top(r["p_service"])[0] == SERVICE_TRUTH[r["tool"]]
                                            for r in ok]))
        km = dict(zip(J.STRATS, J.STRAT_KEYS))
        img = [r for r in ok if isinstance(r["p_strategy_img"], dict) and r["p_strategy_img"]]
        res["strategy_img_acc"] = float(np.mean([km.get(top(r["p_strategy_img"])[0]) ==
                                                 r["strategy"] for r in img])) if img else None
        res["strategy_stat_acc"] = float(np.mean([r["strategy_stat"] == r["strategy"]
                                                  for r in ok]))
        from sklearn.metrics import roc_auc_score
        y = [int(r["orchestrated"]) for r in ok]
        pc = [r["p_coordinated"].get("yes", 0.5)
              if isinstance(r["p_coordinated"], dict) else 0.5 for r in ok]
        res["coord_auc_openjev"] = float(roc_auc_score(y, pc)) if len(set(y)) > 1 else None
        res["coord_auc_test"] = float(roc_auc_score(y, [r["z_test"] for r in ok])) \
            if len(set(y)) > 1 else None
        res["latency_median_s"] = float(np.median(lat))
    with open(out_path, "w") as fh:
        json.dump(res, fh, default=str)
    print(json.dumps({k: v for k, v in res.items() if k not in ("rows", "latency_s")}, indent=1))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "openjev_sim.json",
         rich=(len(sys.argv) < 3 or sys.argv[2] != "plain"),
         kb=(len(sys.argv) >= 3 and sys.argv[2] == "kb"))
