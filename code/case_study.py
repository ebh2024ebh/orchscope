#!/usr/bin/env python3
"""Case study for Paper 3: week-long activity rasters of the most significant orchestrated
campaign, one day-shift null surrogate of it, and a co-tooled (non-orchestrated) campaign on the
same service. Only hour-by-bot activity bits are exported; no source addresses."""
import argparse, glob, json

import numpy as np
import pandas as pd

import engine as E
from week_extras import fp_short

H = 168


def bits_matrix(hexes):
    rows = []
    for b in hexes:
        v = int(b, 16) if b else 0
        rows.append([(v >> h) & 1 for h in range(H)])
    return np.array(rows, dtype=np.uint8)


def order_rows(M):
    first = np.where(M.any(1), M.argmax(1), H)
    return M[np.lexsort((-M.sum(1), first))]


def coact(M, rng, npairs=4000):
    n = len(M)
    if n < 2:
        return 0.0
    i = rng.integers(0, n, npairs); j = rng.integers(0, n, npairs)
    k = i != j
    a, b = M[i[k]].astype(bool), M[j[k]].astype(bool)
    inter = (a & b).sum(1); union = (a | b).sum(1)
    ok = union > 0
    return float((inter[ok] / union[ok]).mean()) if ok.any() else 0.0


def surrogate(M, rng):
    days = max(1, H // 24)
    out = np.empty_like(M)
    for r in range(len(M)):
        out[r] = np.roll(M[r], 24 * int(rng.integers(0, days)) if days > 1
                         else int(rng.integers(0, H)))
    return out


def pack(M):
    return ["".join(map(str, r)) for r in M]


def main():
    global H
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="week_results.json")
    ap.add_argument("--profiles", default="prof/week_*.tsv")
    ap.add_argument("--out", default="week_case.json")
    ap.add_argument("--H", type=int, default=168)
    ap.add_argument("--min-bots", type=int, default=60)
    a = ap.parse_args()
    H = a.H
    rng = np.random.default_rng(5)
    res = json.load(open(a.results))
    df = E.load_profiles(sorted(glob.glob(a.profiles)))
    labels = np.load(a.results.replace(".json", "_labels.npy"))
    L = pd.DataFrame(res["landscape"])
    bits_all = [int(b, 16) if b else 0 for b in df.hbits]
    P = E.Pops(df, bits_all)
    big = L[L.bots >= a.min_bots]
    orch = big[big.orchestrated].sort_values("z", ascending=False)
    if orch.empty:
        print("no orchestrated campaign"); return
    out = {"panels": []}
    o = orch.iloc[0]
    tested = big[big.testable] if "testable" in big else big
    same = tested[(~tested.orchestrated) & (tested.portsig == o.portsig)].sort_values("bots", ascending=False)
    if same.empty:
        same = tested[(~tested.orchestrated) & (tested.proto == o.proto)].sort_values("bots", ascending=False)
    picks = [("orchestrated", o)] + ([("co-tooled", same.iloc[0])] if len(same) else [])
    for kind, c in picks:
        mem = np.where(labels == c["id"])[0]
        if len(mem) > 120:
            mem = rng.choice(mem, 120, replace=False)
        M = order_rows(bits_matrix(df.hbits.iloc[mem]))
        panel = {"kind": kind, "bots": int(c["bots"]), "z": float(c["z"]),
                 "port": c["portsig"].lstrip("p").replace("multi", "many"),
                 "fp": fp_short(c["fpc"]), "strategy": c["strategy"],
                 "T": coact(M, rng), "rows": pack(M)}
        out["panels"].append(panel)
        if kind == "orchestrated":
            S = order_rows(surrogate(M, rng))
            out["panels"].append({"kind": "surrogate", "bots": int(c["bots"]), "z": None,
                                  "port": panel["port"], "fp": panel["fp"],
                                  "strategy": c["strategy"], "T": coact(S, rng),
                                  "rows": pack(S)})
            # population null: activity-matched peers on the same port (never members), of the
            # member's country when PEER_CC and the country has enough peers on the port
            full = np.where(labels == c["id"])[0]
            P.is_mem[full] = True
            peers = []
            for i in mem:
                pool = (E.peer_pool_cc(P, str(c["portsig"]), P.cc[i], int(P.pbin[i])) if E.PEER_CC
                        else E.peer_pool(P, str(c["portsig"]), int(P.pbin[i])))
                for _ in range(50):
                    j = int(pool[rng.integers(len(pool))])
                    if not P.is_mem[j]:
                        break
                peers.append(j)
            P.is_mem[full] = False
            Q = order_rows(bits_matrix(df.hbits.iloc[peers]))
            out["panels"].append({"kind": "peers", "bots": int(c["bots"]), "z": None,
                                  "port": panel["port"], "fp": panel["fp"],
                                  "strategy": c["strategy"], "T": coact(Q, rng),
                                  "rows": pack(Q)})
    json.dump(out, open(a.out, "w"))
    print(json.dumps([{k: v for k, v in p.items() if k != "rows"} for p in out["panels"]]))


if __name__ == "__main__":
    main()
