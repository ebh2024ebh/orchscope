#!/usr/bin/env python3
"""Pipeline-level negative control (OrchScope), run in the work directory next to the week profiles.

  resample : every source's active hours are redrawn, without replacement, from its own number of
             active hours and its primary port's hourly activity profile (a Gumbel top-k draw), so
             each source keeps its activity density and its port's day-to-day drift, while any
             coordination within a group is removed. OrchScope then runs end to end on the
             resampled week (clustering reads profile features only, so the campaigns are the
             week's own), and BH counts the campaigns still declared orchestrated.
  fallback : the week's BH decisions with every campaign whose members mostly lack 40
             same-country peers (port-only population null) marked undetermined.
                                                               -> pipectl.json (both sections)
"""
import collections, glob, json, os, sys, time

import numpy as np
import pandas as pd

import engine as E

H = 168
OUT = "pipectl.json"


def load_out():
    return json.load(open(OUT)) if os.path.exists(OUT) else {}


def save_out(key, val):
    d = load_out()
    d[key] = val
    json.dump(d, open(OUT, "w"), indent=1)


def bits_matrix(hbits):
    """hex activity bitmaps -> (n, H) bool matrix; column h is bit h (hour h of the window)."""
    W = E.pack_bits([int(b, 16) if b else 0 for b in hbits])            # (n, 3) uint64
    return np.unpackbits(W.view(np.uint8), axis=1, bitorder="little")[:, :H].astype(bool)


def to_hex(M):
    """(n, H) bool matrix -> hex bitmaps in the profiles' encoding ('' for no activity)."""
    pad = np.zeros((M.shape[0], 192), dtype=np.uint8)
    pad[:, :H] = M
    W = np.packbits(pad, axis=1, bitorder="little").view(np.uint64)     # (n, 3)
    return ["%x" % (int(a) | (int(b) << 64) | (int(c) << 128)) if (a or b or c) else ""
            for a, b, c in W]


def cmd_resample(seed=17, chunk=200000):
    t0 = time.time()
    df = E.load_profiles(sorted(glob.glob("prof/week_*.tsv")))
    rng = np.random.default_rng(seed)
    ports = df[E.PEER_KEY].astype(str).to_numpy()
    new_hex = np.empty(len(df), dtype=object)
    n_sources = 0
    for port, idx in pd.Series(ports).groupby(ports).indices.items():
        idx = np.asarray(idx)
        prof = np.zeros(H)
        for s in range(0, len(idx), chunk):                              # the port's hourly profile
            prof += bits_matrix(df.hbits.iloc[idx[s:s + chunk]]).sum(0)
        logw = np.log(prof + 1e-3)
        for s in range(0, len(idx), chunk):
            sub = idx[s:s + chunk]
            M = bits_matrix(df.hbits.iloc[sub])
            k = M.sum(1)
            keys = logw[None, :] + rng.gumbel(size=M.shape)              # Gumbel top-k = weighted
            kth = -np.sort(-keys, axis=1)[np.arange(len(sub)), np.maximum(k, 1) - 1]
            R = (keys >= kth[:, None]) & (k[:, None] > 0)                # draw w/o replacement
            new_hex[sub] = to_hex(R)
            n_sources += len(sub)
    df["hbits"] = list(new_hex)
    t_res = time.time() - t0
    labels, camps = E.enhanced_infer(df, H, min_size=10, B=200)
    week_labels = np.load("week_results_labels.npy")
    test = [c for c in camps if not c.get("residue") and c.get("testable")]
    res = {"sources": n_sources, "seed": seed, "seconds_resample": round(t_res),
           "seconds": round(time.time() - t0),
           "same_campaigns_as_week": bool(np.array_equal(labels, week_labels)),
           "campaigns": len(camps), "residue": int(sum(bool(c.get("residue")) for c in camps)),
           "testable": len(test),
           "orchestrated": int(sum(bool(c.get("orchestrated")) for c in test)),
           "orchestrated_rot": int(sum(bool(c.get("orchestrated_rot")) for c in test)),
           "raw_p_lt_0.05": float(np.mean([c["p"] < 0.05 for c in test])) if test else None,
           "raw_rot_p_lt_0.05": float(np.mean([c["p_rot"] < 0.05 for c in test])) if test else None}
    print(json.dumps(res), flush=True)
    save_out("resample", res)


def cmd_fallback(min_pool=40):
    import revision_checks as R
    R._init()
    L, labels, P = R.G["L"], R.G["labels"], R.G["P"]
    major = set()
    for r in L.itertuples():
        if r.residue or not r.testable:
            continue
        mem = np.where(labels == r.id)[0]
        port = E.campaign_port(P, mem)
        own = collections.Counter(zip(P.ports[mem], P.cc[mem], P.pbin[mem].tolist()))
        own_cc = collections.Counter(zip(P.ports[mem], P.cc[mem]))
        n_fb = 0
        for (c, b), k in collections.Counter(zip(P.cc[mem], P.pbin[mem].tolist())).items():
            ok = any(len(P.port_cc_bin.get((port, c, b + d), ())) - own[(port, c, b + d)] >= min_pool
                     for d in (0, -1, 1))
            ok = ok or len(P.port_cc_all.get((port, c), ())) - own_cc[(port, c)] >= min_pool
            n_fb += 0 if ok else k
        if n_fb > len(mem) / 2:
            major.add(int(r.id))
    T = L[~L.residue]
    p = np.where(T.id.isin(major), 1.0, T.p.to_numpy(float))
    sig = E.bh(p)
    res = {"majority_fallback_testable": len(major),
           "majority_fallback_orchestrated": int(T[T.id.isin(major)].orchestrated.sum()),
           "orchestrated_week": int(T.orchestrated.sum()),
           "orchestrated_fallback_undetermined": int(sig.sum())}
    print(json.dumps(res), flush=True)
    save_out("fallback", res)


if __name__ == "__main__":
    {"resample": cmd_resample, "fallback": cmd_fallback}[sys.argv[1]]()
