#!/usr/bin/env python3
"""Revision checks (OrchScope), run in the work directory.

  orch   : what are the orchestrated campaigns? protocol, UDP payload class, destination spread,
           source /24 / AS / country concentration, known research scanners; time from a
           campaign's onset until it has m members (first hourly update that can test it)
                                                                         -> rev_orch.json
  tzneg  : hard negative on real traffic: pseudo-campaigns drawn from the same port AND the same
           country (shared time zone) as each testable campaign, excluding its members, through
           the identical two-null test + BH                              -> rev_tzneg.json
  pemp   : p-value robustness: every campaign re-tested with B=1000 surrogates; BH decisions from
           the normal-tail p-values vs. exact empirical p-values (1+#{null>=T})/(B+1), compared
           with the reported B=200 decisions                             -> rev_pemp.json
  negp   : the week's same-port negative control re-run exactly (same campaigns and order, seed
           and B as analyze_week.py), keeping every p-value: unadjusted rejection rates, so
           "BH flags none" is not an artifact of the Monte Carlo p-value floor -> rev_negp.json
  refine : both negative controls replayed exactly; every p < 0.01 re-tested with 10^4 fresh
           surrogates per null, then BH re-applied                       -> rev_refine.json
"""
import argparse, collections, glob, json, os, sys, time
from multiprocessing import Pool

import numpy as np
import pandas as pd
from scipy import stats

import engine as E

H = 168
M_MIN = 10


def load():
    df = E.load_profiles(sorted(glob.glob("prof/week_*.tsv")))
    labels = np.load("week_results_labels.npy")
    L = pd.DataFrame(json.load(open("week_results.json"))["landscape"])
    assert len(labels) == len(df)
    return df, labels, L


# ----------------------------------------------------------------------------- orch
def payload_class(ph, port):
    ph = (ph or "").lower()
    if port == "53":
        return "dns"
    if not ph:
        return "empty"
    b = bytes.fromhex(ph[: len(ph) // 2 * 2]) if all(c in "0123456789abcdef" for c in ph) else b""
    if b.startswith(b"d1:") or b.startswith(b"d2:"):
        return "bittorrent-dht"
    if len(b) >= 2 and b[0] in (0x01, 0x11, 0x21, 0x31, 0x41) and b[1] in (0x00, 0x01, 0x02):
        return "utp"
    if b[:1] in (b"\xe4", b"\xe5"):
        return "kad"
    if len(b) >= 8 and b[4:8] == b"\x21\x12\xa4\x42":
        return "stun"
    if b[:4] in (b"OPTI", b"REGI", b"INVI", b"SIP/"):
        return "sip"
    if b[:4] == b"M-SE":
        return "ssdp"
    if all(32 <= x < 127 for x in b) and len(b) >= 4:
        return "ascii-other"
    return "binary-other"


def cmd_orch(out):
    df, labels, L = load()
    src24 = df.src.astype(str).str.rsplit(".", n=1).str[0].to_numpy()
    rows = []
    for _, c in L.iterrows():
        mem = np.where(labels == c["id"])[0]
        if len(mem) < M_MIN:
            continue
        sub = df.iloc[mem]
        port = str(sub.top1.mode().iat[0])
        pcs = collections.Counter(payload_class(p, port) for p in sub.phead.astype(str))
        hf = np.sort(sub.hfirst.to_numpy().astype(int))
        rows.append({
            "id": int(c["id"]), "bots": int(len(mem)), "orch": bool(c["orchestrated"]),
            "testable": bool(c["testable"]), "proto": int(c["proto"]), "port": port,
            "payload": pcs.most_common(1)[0][0], "payload_share": pcs.most_common(1)[0][1] / len(mem),
            "ud_med": float(sub.ud.median()), "ud24_med": float(sub.ud24.median()),
            "frac_ud1": float((sub.ud <= 1).mean()),
            "n_src24": int(len(set(src24[mem]))), "n_asn": int(c["n_asn"]), "n_cc": int(c["n_cc"]),
            "research": float(c["research"]), "span_med": float(c["span_med"]),
            "h1": int(hf[0]), "h_m": int(hf[M_MIN - 1]), "dur": int(c["hlast_max"] - c["hfirst_min"] + 1),
            "z": float(c["z"]), "top_cc": c["top_cc"], "top_org": c["top_org"],
            "residue": bool(c.get("residue", False)), "conv": float(c.get("conv", 0.0)),
            "dmode_top": int(collections.Counter(E.pd_numeric(sub.dmode)).most_common(1)[0][0])
            if "dmode" in sub else -1})
    R = pd.DataFrame(rows)

    def summ(X):
        if not len(X):
            return {}
        w = X.bots
        udp = X[X.proto == 2]
        return {
            "n": int(len(X)), "bots": int(w.sum()),
            "tcp": float((X.proto == 1).mean()), "udp": float((X.proto == 2).mean()),
            "other": float((~X.proto.isin([1, 2])).mean()),
            "tcp_bots": float(w[X.proto == 1].sum() / w.sum()), "udp_bots": float(w[X.proto == 2].sum() / w.sum()),
            "udp_payload": {k: float(v) for k, v in udp.payload.value_counts(normalize=True).items()},
            "udp_payload_bots": {k: float(udp.bots[udp.payload == k].sum() / max(udp.bots.sum(), 1))
                                 for k in udp.payload.unique()},
            "ud_med_median": float(X.ud_med.median()), "ud24_med_median": float(X.ud24_med.median()),
            "spread_ge3": float((X.ud_med >= 3).mean()), "single_addr": float((X.ud_med <= 1).mean()),
            "udp_spread_ge3": float((udp.ud_med >= 3).mean()) if len(udp) else None,
            "udp_single_addr": float((udp.ud_med <= 1).mean()) if len(udp) else None,
            "one_src24": float((X.n_src24 == 1).mean()), "one_asn": float((X.n_asn == 1).mean()),
            "one_cc": float((X.n_cc == 1).mean()), "research_half": float((X.research >= 0.5).mean()),
            "span_med_median": float(X.span_med.median()),
        }

    O = R[R.orch]
    lat = O[O.h1 >= 24]                         # onset after the first day (no left censoring)
    d = (lat.h_m - lat.h1).to_numpy()
    res = {"orchestrated": summ(O), "not_orchestrated_testable": summ(R[(~R.orch) & R.testable]),
           "all": summ(R), "residue": summ(R[R.residue]), "scanning": summ(R[~R.residue]),
           "residue_share": float(R.residue.mean()), "residue_sources": int(R.bots[R.residue].sum()),
           "conv_quantiles": {q: float(R.conv.quantile(q)) for q in (0.1, 0.25, 0.5, 0.75, 0.9)},
           "time_to_m": {"n": int(len(d)), "median_h": float(np.median(d)) if len(d) else None,
                         "p75_h": float(np.percentile(d, 75)) if len(d) else None,
                         "p90_h": float(np.percentile(d, 90)) if len(d) else None,
                         "share_same_hour": float((d == 0).mean()) if len(d) else None},
           "top": O.sort_values("bots", ascending=False).head(16).to_dict("records"),
           "top_tcp": O[O.proto == 1].sort_values("bots", ascending=False).head(8).to_dict("records")}
    json.dump(res, open(out, "w"), indent=1, default=str)
    print(json.dumps({k: res[k] for k in ("orchestrated", "scanning", "residue", "residue_share",
                                          "conv_quantiles", "time_to_m")},
                     indent=1, default=str))


# ----------------------------------------------------------------------------- tzneg
def tz_control(df, labels, L, B=100, seed=13):
    """Same-port, same-country pseudo-campaigns through the two-null test (draw order fixed)."""
    bits_all = [int(b, 16) if b else 0 for b in df.hbits]
    P = E.Pops(df, bits_all, key=E.PEER_KEY)
    cc = df.cc.astype(str).to_numpy()
    rng = np.random.default_rng(seed)
    pv, pv_rot, mems, skipped, skipped_res = [], [], [], 0, 0
    dmode = E.pd_numeric(df["dmode"]) if "dmode" in df.columns else None
    keep = L.testable & ~(L["residue"] if "residue" in L else False)
    for _, c in L[keep].iterrows():
        mem = np.where(labels == c["id"])[0]
        port = E.campaign_port(P, mem)
        top = collections.Counter(cc[mem]).most_common(1)[0][0]
        pool = np.setdiff1d(P.port_all[port], mem, assume_unique=True)
        pool = pool[cc[pool] == top]
        if len(pool) < len(mem):
            skipped += 1
            continue
        pseudo = rng.choice(pool, len(mem), replace=False)
        if E.convergence(pseudo, dmode)[1]:
            skipped_res += 1
            continue
        r = E.sync_test(pseudo, None, bits_all, H, P, rng, B=B)
        pv.append(r["p"]); pv_rot.append(r["p_rot"]); mems.append(pseudo)
    return pv, pv_rot, mems, skipped, skipped_res


def cmd_tzneg(out, B=100, seed=13):
    df, labels, L = load()
    t0 = time.time()
    pv, pv_rot, _, skipped, skipped_res = tz_control(df, labels, L, B=B, seed=seed)
    sig, sig_rot = E.bh(pv), E.bh(pv_rot)
    res = {"pseudo_campaigns": len(pv), "skipped_small_country_pool": skipped,
           "skipped_residue": skipped_res,
           "fpr": float(sig.mean()) if pv else None, "fpr_rot": float(sig_rot.mean()) if pv else None,
           "raw_p_lt_0.05": float(np.mean(np.array(pv) < 0.05)) if pv else None,
           "seconds": round(time.time() - t0)}
    res.update(raw_rates(pv, pv_rot))
    print(json.dumps(res))
    res["p"], res["p_rot"] = [round(float(x), 6) for x in pv], [round(float(x), 6) for x in pv_rot]
    json.dump(res, open(out, "w"))


def raw_rates(pv, pv_rot):
    """Unadjusted rejection rates and how many p-values sit at the Monte Carlo floor 1/(B2+1)."""
    p, pr = np.asarray(pv, float), np.asarray(pv_rot, float)
    floor = 1.0 / (E.MC_B2 + 1) + 1e-12
    out = {}
    for name, x in (("", p), ("rot_", pr)):
        for a in (0.05, 0.01):
            out["raw_%sp_lt_%s" % (name, a)] = float(np.mean(x < a)) if len(x) else None
        out["%sfloor_n" % name] = int((x <= floor).sum())
    return out


# ----------------------------------------------------------------------------- negp
def cmd_negp(out, B=100):
    df, labels, L = load()
    camps = [{"id": int(r.id), "members": np.where(labels == r.id)[0], "residue": bool(r.residue)}
             for r in L.itertuples()]                   # same campaigns, same order as the week run
    t0 = time.time()
    r = E.negative_control(df, H, camps, np.random.default_rng(11), B=B, keep_p=True)
    pv, pv_rot = r.pop("p"), r.pop("p_rot")
    r.pop("members")
    r.update(raw_rates(pv, pv_rot))
    wk = json.load(open("week_results.json"))["negative_control"]
    r["reproduces_week_run"] = bool(all(r[k] == wk[k] for k in ("pseudo_campaigns", "flagged",
                                                                "flagged_rot"))
                                    and abs(r["raw_p_lt_0.05"] - wk["raw_p_lt_0.05"]) < 1e-12)
    r["seconds"] = round(time.time() - t0)
    print(json.dumps(r))
    r["p"], r["p_rot"] = [round(float(x), 6) for x in pv], [round(float(x), 6) for x in pv_rot]
    json.dump(r, open(out, "w"))


# ----------------------------------------------------------------------------- refine
_R = {}


def _refine_one(args):
    k, mem = args
    r = E.sync_test(np.asarray(mem), None, _R["bits"], H, _R["P"], np.random.default_rng(500000 + k),
                    B=_R["B2"])
    return k, r["p"], r["p_rot"]


def refine(pv, pv_rot, mems, B2, cut, procs, B1):
    """Re-tests every pseudo-campaign with p < cut with B2 fresh surrogates per null (one stage,
    floor 1/(B2+1)) and re-applies BH; the other p-values are kept. B1: the pipeline's budget."""
    idx = [k for k, p in enumerate(pv) if p < cut]
    with Pool(procs) as pool:
        out = pool.map(_refine_one, [(k, mems[k]) for k in idx])
    p2, pr2 = np.array(pv, float), np.array(pv_rot, float)
    for k, p, pr in out:
        p2[k], pr2[k] = p, pr
    return {"n": len(pv), "refined": len(idx), "B2": B2,
            "flagged_before": int(E.bh(pv).sum()), "flagged_after": int(E.bh(p2).sum()),
            "floor_before": int((np.asarray(pv) <= 1.0 / (B1 + 1) + 1e-12).sum()),
            "floor_after": int((p2 <= 1.0 / (B2 + 1) + 1e-12).sum()),
            "raw_p_lt_0.01_after": float(np.mean(p2 < 0.01)), "min_p_after": float(p2.min())}


def cmd_refine(out, B2=10000, cut=0.01, procs=40):
    """Both negative controls replayed exactly (same draws as negp / tzneg), then their small
    p-values sharpened with B2 surrogates: does BH still flag none without the 1/1001 floor?"""
    df, labels, L = load()
    t0 = time.time()
    camps = [{"id": int(r.id), "members": np.where(labels == r.id)[0], "residue": bool(r.residue)}
             for r in L.itertuples()]
    r = E.negative_control(df, H, camps, np.random.default_rng(11), B=100, keep_p=True)
    tz = tz_control(df, labels, L, B=100, seed=13)
    _R.update(bits=[int(b, 16) if b else 0 for b in df.hbits], B2=B2)
    _R["P"] = E.Pops(df, _R["bits"], key=E.PEER_KEY)
    b1, E.MC_B2 = E.MC_B2, B2                      # one stage of B2 surrogates (no screening)
    res = {"same_port": refine(r["p"], r["p_rot"], r["members"], B2, cut, procs, b1),
           "same_country": refine(tz[0], tz[1], tz[2], B2, cut, procs, b1),
           "seconds": round(time.time() - t0)}
    print(json.dumps(res))
    json.dump(res, open(out, "w"), indent=1)


# ----------------------------------------------------------------------------- pemp
G = {}


def _init():
    df, labels, L = load()
    bits_all = [int(b, 16) if b else 0 for b in df.hbits]
    G.update(df=df, labels=labels, L=L, bits=bits_all, P=E.Pops(df, bits_all, key=E.PEER_KEY))


def rot_null(members, bits_all, rng, B, local):
    unit, steps, mask = 24, H // 24, (1 << H) - 1
    need, I, J = local
    R = np.stack([E.pack_bits([E.rot(bits_all[i] & mask, (s * unit) % H, H, mask)
                               for s in range(steps)]) for i in members[need]])
    t = float(E.coact_vec(R[None, :, 0, :], I, J)[0])
    S = rng.integers(0, steps, size=(B, len(need)))
    return t, E.coact_vec(R[np.arange(len(need))[None, :], S], I, J)


def pop_null(members, port, P, rng, B, local, min_pool=40):
    need, I, J = local
    mem_need = members[need]
    t = float(E.coact_vec(P.packed[mem_need][None], I, J)[0])
    P.is_mem[members] = True
    try:
        bins = P.pbin[mem_need]
        draws = np.empty((B, len(need)), dtype=np.int64)
        for b in np.unique(bins):
            cols = np.where(bins == b)[0]
            pool = E.peer_pool(P, port, int(b), min_pool)
            d = pool[rng.integers(0, len(pool), size=(B, len(cols)))]
            for _ in range(8):
                bad = P.is_mem[d]
                if not bad.any():
                    break
                d[bad] = pool[rng.integers(0, len(pool), size=int(bad.sum()))]
            draws[:, cols] = d
    finally:
        P.is_mem[members] = False
    return t, E.coact_vec(P.packed[draws], I, J)


def p_both(t, null):
    sd = null.std()
    pn = float(stats.norm.sf((t - null.mean()) / sd)) if sd > 1e-9 else 1.0
    pe = float((1 + np.sum(null >= t)) / (len(null) + 1))
    return pn, pe


def _one(args):
    cid, B = args
    labels, P, bits = G["labels"], G["P"], G["bits"]
    mem = np.where(labels == cid)[0]
    rng = np.random.default_rng(1000 + cid)
    local = E.local_pairs(len(mem), rng)
    if len(local[1]) == 0:
        return cid, 1.0, 1.0, 1.0, 1.0
    t, nr = rot_null(mem, bits, rng, B, local)
    rn, re_ = p_both(t, nr)
    port = E.campaign_port(P, mem)
    if E.testable(P, mem, port):
        t2, npop = pop_null(mem, port, P, rng, B, local)
        pn, pe = p_both(t2, npop)
    else:
        pn = pe = 1.0
    return cid, rn, re_, pn, pe


def cmd_pemp(out, B=1000, procs=40):
    _init()                       # load once; forked workers share it copy-on-write
    G.pop("df")
    Lr = G["L"]
    if "residue" in Lr:
        Lr = Lr[~Lr.residue]                    # residue groups are never tested
    ids = [int(i) for i in Lr.id]
    t0 = time.time()
    with Pool(procs) as pool:
        res = pool.map(_one, [(i, B) for i in ids], chunksize=64)
    res.sort()
    rn = np.array([r[1] for r in res]); re_ = np.array([r[2] for r in res])
    pn = np.array([r[3] for r in res]); pe = np.array([r[4] for r in res])
    d_norm = E.bh(np.maximum(rn, pn)); d_emp = E.bh(np.maximum(re_, pe))
    L = G["L"].set_index("id").loc[[r[0] for r in res]]
    d200 = L.orchestrated.to_numpy().astype(bool)
    agree = lambda a, b: float((a == b).mean())
    out_ = {"B": B, "campaigns": len(res), "seconds": round(time.time() - t0),
            "orch_pipeline": int(d200.sum()), "orch_B1000_normal": int(d_norm.sum()),
            "orch_B1000_empirical": int(d_emp.sum()),
            "agree_pipeline_vs_B1000_normal": agree(d200, d_norm),
            "agree_pipeline_vs_B1000_empirical": agree(d200, d_emp),
            "agree_normal_vs_empirical_B1000": agree(d_norm, d_emp),
            "pipeline_orch_confirmed_by_B1000_empirical": float(d_emp[d200].mean()) if d200.any() else None,
            "jaccard_pipeline_B1000_empirical": float((d200 & d_emp).sum() / max((d200 | d_emp).sum(), 1))}
    json.dump(out_, open(out, "w"), indent=1)
    print(json.dumps(out_))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["orch", "tzneg", "pemp", "negp", "refine"])
    ap.add_argument("--out")
    ap.add_argument("--B", type=int)
    ap.add_argument("--procs", type=int, default=40)
    a = ap.parse_args()
    if a.cmd == "orch":
        cmd_orch(a.out or "rev_orch.json")
    elif a.cmd == "tzneg":
        cmd_tzneg(a.out or "rev_tzneg.json", B=a.B or 100)
    elif a.cmd == "negp":
        cmd_negp(a.out or "rev_negp.json", B=a.B or 100)
    elif a.cmd == "refine":
        cmd_refine(a.out or "rev_refine.json", B2=a.B or 10000, procs=a.procs)
    else:
        cmd_pemp(a.out or "rev_pemp.json", B=a.B or 1000, procs=a.procs)
