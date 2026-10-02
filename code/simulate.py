#!/usr/bin/env python3
"""ORION-faithful campaign simulator with ground truth (OrchScope).

Synthetic campaigns are emitted as raw ORION per-flow JSON records (base64 packet samples with
tool-consistent headers), reduced by the production reducer, profiled by the production engine,
and injected next to *real* background profiles from the same hours. Ground truth: bot ->
campaign, campaign strategy, and whether the campaign is orchestrated (synchronized schedule)
or independent (same machinery, independent per-bot schedules).
"""
import argparse, base64, collections, datetime, itertools, json, math, os, struct, sys
from multiprocessing import Pool

import numpy as np

import camp_reduce
import engine

D0 = 591396864
DSIZE = 1 << 19
P_HIT = 2.0 ** -13            # /13 share of IPv4: a uniform /0 scanner hits it w.p. 2^-13
W0 = datetime.datetime(2026, 9, 24)
W0_EPOCH = int(W0.replace(tzinfo=datetime.timezone.utc).timestamp())

OPT_MSS = b"\x02\x04\x05\xb4"
OPT_LINUX = b"\x02\x04\x05\xb4\x04\x02\x08\x0a\x00\x00\x00\x01\x00\x00\x00\x00\x01\x03\x03\x07"
OPT_WIN = b"\x02\x04\x05\xb4\x01\x03\x03\x08\x01\x01\x04\x02"
TOOLS = {
    "mirai":   dict(proto=6, ttl=64, ipid="rand", win="rand", opts=b"", seq="dst", sport="rand",
                    df=0, flag="Mirai"),
    "zmap":    dict(proto=6, ttl=255, ipid="z54321", win=65535, opts=OPT_MSS, seq="rand",
                    sport="narrow", df=0, flag="Zmap"),
    "masscan": dict(proto=6, ttl=255, ipid="rand", win=1024, opts=b"", seq="rand", sport="fixed",
                    df=0, flag="Masscan"),
    "linux":   dict(proto=6, ttl=64, ipid="zero", win=64240, opts=OPT_LINUX, seq="rand",
                    sport="rand", df=1, flag=None),
    "winsyn":  dict(proto=6, ttl=128, ipid="inc", win=8192, opts=OPT_WIN, seq="rand", sport="rand",
                    df=1, flag=None),
    "sipudp":  dict(proto=17, ttl=128, ipid="inc", sport="rand", df=0,
                    payload=b"OPTIONS sip:100@", flag=None),
}
PORTS = {"mirai": [23, 2323], "zmap": [443], "masscan": [8080], "linux": [22], "winsyn": [445],
         "sipudp": [5060]}


def ip_str(x):
    return ".".join(str((x >> s) & 255) for s in (24, 16, 8, 0))


def craft(tool, bot, dst, dport, rng):
    t = TOOLS[tool]
    if t["ipid"] == "z54321":
        ipid = 54321
    elif t["ipid"] == "zero":
        ipid = 0
    elif t["ipid"] == "inc":
        bot["ipid"] = (bot["ipid"] + int(rng.integers(1, 40))) & 0xffff
        ipid = bot["ipid"]
    else:
        ipid = int(rng.integers(0, 65536))
    if t["sport"] == "fixed":
        sport = bot["sport"]
    elif t["sport"] == "narrow":
        sport = bot["sport"] + int(rng.integers(0, 8))
    else:
        sport = int(rng.integers(1024, 65536))
    if t["proto"] == 6:
        seq = dst if t["seq"] == "dst" else int(rng.integers(0, 2 ** 32))
        win = int(rng.integers(1, 65536)) if t["win"] == "rand" else t["win"]
        opts = t["opts"] + b"\x00" * ((4 - len(t["opts"]) % 4) % 4)
        l4 = struct.pack("!HHIIBBHHH", sport, dport, seq, 0, ((20 + len(opts)) // 4) << 4, 0x02,
                         win, 0, 0) + opts
    else:
        p = t["payload"]
        l4 = struct.pack("!HHHH", sport, dport, 8 + len(p), 0) + p
    ttl = max(1, t["ttl"] - bot["hops"])
    ip = struct.pack("!BBHHHBBH4s4s", 0x45, 0, 20 + len(l4), ipid, 0x4000 if t["df"] else 0, ttl,
                     t["proto"], 0, bot["src"].to_bytes(4, "big"), dst.to_bytes(4, "big"))
    return base64.b64encode(b"\xaa\xbb\xcc\xdd\xee\xff\x11\x22\x33\x44\x55\x66\x08\x00" + ip + l4
                            ).decode()


def iso(ts):
    return datetime.datetime.fromtimestamp(ts, datetime.timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%S.%f") + "Z"


def schedule(spec, rng, H):
    """Campaign-level active-hour set: 1-3 waves inside the window (circular placement)."""
    lo, hi = spec["window"]
    span = hi - lo
    hours = set()
    for _ in range(int(rng.integers(1, 4))):
        a = int(rng.integers(0, span))
        hours.update(lo + (a + i) % span for i in range(int(rng.integers(3, 10))))
    return sorted(hours)


def gen_campaign(cid, spec, rng, H, scale, src_base):
    """Yields (hour, json_line) and returns ground truth for each bot."""
    tool, n, rate, strat = spec["tool"], spec["bots"], spec["rate"], spec["strategy"]
    ports = spec.get("ports") or PORTS[tool]
    pw = np.array([0.9] + [0.1 / (len(ports) - 1)] * (len(ports) - 1)) if len(ports) > 1 else \
        np.array([1.0])
    sched = schedule(spec, rng, H)
    blocks = rng.choice(8, 2, replace=False)               # for non-uniform permutation
    lines = collections.defaultdict(list)
    truth = {}
    asn_pool = rng.integers(1000, 400000, size=max(3, n // 4))
    cc_pool = ["BR", "US", "CN", "IN", "VN", "RU", "ID", "TW", "KR", "MX"]
    lam = rate * 3600 * P_HIT * scale
    for b in range(n):
        src = src_base + cid * 4096 + b
        bot = {"src": src, "hops": int(rng.integers(4, 26)), "ipid": int(rng.integers(0, 65536)),
               "sport": int(rng.integers(32768, 61000))}
        if spec["orchestrated"]:
            act = [(h + int(rng.integers(-1, 2))) % H for h in sched if rng.random() > 0.1]
        else:                       # independent: own contiguous runs, circular placement
            lo, hi = spec["window"]
            span = hi - lo
            act = set()
            for _ in range(int(rng.integers(1, 4))):
                a = int(rng.integers(0, span))
                act.update(lo + (a + i) % span for i in range(int(rng.integers(3, 10))))
            act = sorted(act)
        act = sorted(set(h for h in act if 0 <= h < H))
        hits = []
        for h in act:
            kk = int(rng.poisson(lam))
            ts = np.sort(rng.uniform(0, 3600, kk))
            hits.extend((h, W0_EPOCH + h * 3600 + float(t)) for t in ts)
        if not hits:
            truth[ip_str(src)] = (cid, strat, spec["orchestrated"])
            continue
        m = len(hits)
        if strat in ("seq_fwd", "seq_rev"):
            step = max(1, int(DSIZE * 0.8 / m))
            start = int(rng.integers(0, max(1, DSIZE - step * m)))
            offs = [start + i * step + int(rng.integers(0, max(1, step // 4))) for i in range(m)]
            if strat == "seq_rev":
                offs = offs[::-1]
        elif strat == "perm":
            offs = [int(rng.choice(blocks)) * 65536 + int(rng.integers(0, 65536)) for _ in range(m)]
        else:
            offs = [int(x) for x in rng.integers(0, DSIZE, m)]
        rec = collections.defaultdict(list)
        for (h, t), off in zip(hits, offs):
            dport = int(rng.choice(ports, p=pw))
            rec[(h, dport)].append((t, D0 + off % DSIZE))
        asn = int(rng.choice(asn_pool)); cc = str(rng.choice(cc_pool))
        flag = TOOLS[tool]["flag"]
        for (h, dport), pk in rec.items():
            ds = [d for _, d in pk]
            r = {"SourceIP": ip_str(src), "Port": dport,
                 "Traffic": 11 if TOOLS[tool]["proto"] == 6 else 16, "First": iso(pk[0][0]),
                 "Last": iso(pk[-1][0]), "Packets": len(pk), "Bytes": 60 * len(pk),
                 "UniqueDests": len(set(ds)), "UniqueDest24s": len(set(d >> 8 for d in ds)),
                 "Country": cc, "ASN": asn, "Org": "SIM-AS%d" % asn, "RDNS": [],
                 "Zmap": flag == "Zmap", "Masscan": flag == "Masscan", "Mirai": flag == "Mirai",
                 "Samples": [craft(tool, bot, d, dport, rng) for d in ds[:3]],
                 "TCP": "TCPSYN" if TOOLS[tool]["proto"] == 6 else "", "ICMP": ""}
            lines[h].append(json.dumps(r))
        truth[ip_str(src)] = (cid, strat, spec["orchestrated"])
    return lines, truth


def label_of(h):
    return (W0 + datetime.timedelta(hours=h)).strftime("%Y-%m-%d.%H")


def synth_profiles(specs, seed, H=24, scale=1.0, return_acc=False, return_rows=False):
    rng = np.random.default_rng(seed)
    all_lines = collections.defaultdict(list)
    truth = {}
    src_base = (240 << 24) + (seed % 64) * (1 << 18)       # reserved 240/4: no collisions
    for cid, spec in enumerate(specs):
        lines, tr = gen_campaign(cid, spec, rng, H, scale, src_base)
        for h, ls in lines.items():
            all_lines[h].extend(ls)
        truth.update(tr)
    rows = []
    for h in sorted(all_lines):
        rows.extend(r.rstrip("\n").split("\t") for r in camp_reduce.reduce_lines(all_lines[h],
                                                                                 label_of(h)))
    acc, _ = engine.accumulate_rows(iter(rows), W0)
    prof = [engine.profile(s, a, D0, H) for s, a in acc.items() if a.ud >= 2 or len(a.sam) >= 2]
    import pandas as pd
    df = pd.DataFrame(prof, columns=engine.PROF_COLS)
    if return_rows:
        return df, truth, rows
    if return_acc:
        return df, truth, acc
    return df, truth

# ----------------------------------------------------------------------------- metrics


def evaluate(df, labels, truth, camps=None):
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
    idx = {s: i for i, s in enumerate(df.src)}
    syn = [s for s in truth if s in idx]
    if not syn:
        return {}
    tl = np.array([truth[s][0] for s in syn])
    pl = np.array([labels[idx[s]] for s in syn])
    nxt = pl.max() + 1 if len(pl) else 0
    pl2 = pl.copy()
    for i in range(len(pl2)):
        if pl2[i] < 0:
            pl2[i] = nxt; nxt += 1
    out = {"ari": float(adjusted_rand_score(tl, pl2)),
           "nmi": float(normalized_mutual_info_score(tl, pl2))}
    size_full = collections.Counter(labels[labels >= 0])
    tsize = collections.Counter(tl)
    pair = collections.Counter(zip(tl, pl))
    prec = rec = 0.0
    for t, p in zip(tl, pl):
        if p < 0:
            prec += 1.0; rec += 1.0 / tsize[t]
        else:
            prec += pair[(t, p)] / size_full[p]
            rec += pair[(t, p)] / tsize[t]
    P, R = prec / len(tl), rec / len(tl)
    out.update({"b3_p": P, "b3_r": R, "b3_f1": 2 * P * R / (P + R) if P + R else 0.0})
    recovered = {}
    for t in tsize:
        best = None
        for (tt, p), c in pair.items():
            if tt != t or p < 0:
                continue
            if c >= 0.5 * tsize[t] and c >= 0.5 * size_full[p]:
                best = p
        recovered[t] = best
    out["campaign_recall"] = float(np.mean([v is not None for v in recovered.values()]))
    if camps is not None:
        orch = {c["id"]: c.get("orchestrated", False) for c in camps}
        orch_rot = {c["id"]: c.get("orchestrated_rot", False) for c in camps}
        zz = {c["id"]: c.get("z") for c in camps}
        tp = fp = pos = neg = tpr_ = fpr_ = 0
        scores = []
        for t, p in recovered.items():
            is_orch = next(truth[s][2] for s in syn if truth[s][0] == t)
            if p is None:
                continue
            if is_orch:
                pos += 1; tp += orch.get(p, False); tpr_ += orch_rot.get(p, False)
            else:
                neg += 1; fp += orch.get(p, False); fpr_ += orch_rot.get(p, False)
            if zz.get(p) is not None:
                scores.append([int(bool(is_orch)), float(zz[p])])
        out.update({"orch_tpr": tp / pos if pos else float("nan"),
                    "orch_fpr": fp / neg if neg else float("nan"),
                    "orch_tpr_rot": tpr_ / pos if pos else float("nan"),
                    "orch_fpr_rot": fpr_ / neg if neg else float("nan"),
                    "orch_scores": scores})
    st = [(truth[s][1], df.strategy.iat[idx[s]]) for s in syn]
    out["strategy_acc"] = float(np.mean([a == b for a, b in st]))
    det = [(a, b) for a, b in st if b != "undet"]
    out["strategy_acc_det"] = float(np.mean([a == b for a, b in det])) if det else float("nan")
    out["n_syn"] = len(syn)
    return out

# ----------------------------------------------------------------------------- scenarios


def mix(K, n, rate, rng, same_port=None, window=(0, 24)):
    tools = list(TOOLS)
    strats = ["uperm", "seq_fwd", "seq_rev", "perm"]
    specs = []
    for k in range(K):
        tool = tools[k % len(tools)]
        sp = {"tool": tool, "bots": n, "rate": rate, "strategy": strats[(k // 2) % 4],
              "orchestrated": k % 2 == 0, "window": window}
        if same_port is not None:
            sp["ports"] = [same_port]
            if TOOLS[tool]["proto"] != 6:
                sp["tool"] = "linux"
        specs.append(sp)
    return specs


BG = None


def _init(bg_path):
    global BG
    BG = engine.load_profiles([bg_path])


def run_one(job):
    import pandas as pd
    name, specs, seed, scale, variant = job
    syn, truth = synth_profiles(specs, seed, scale=scale)
    bg = BG[BG.portsig.isin(set(syn.portsig))]          # only the relevant port populations
    df = pd.concat([bg, syn], ignore_index=True)
    H = 24
    kw = dict(min_size=10, B=200, seed=seed)
    eps0, frac0 = engine.EPS, engine.MCS_FRAC
    if variant.startswith("eps"):                       # sensitivity: merge distance
        engine.EPS = float(variant[3:])
    elif variant.startswith("mcs"):                     # sensitivity: minimum mode fraction
        engine.MCS_FRAC = float(variant[3:])
    elif variant.startswith("B") and variant[1:].isdigit():
        kw["B"] = int(variant[1:])
    if variant == "ours" or variant.startswith(("eps", "mcs")) or (
            variant.startswith("B") and variant[1:].isdigit()):
        labels, camps = engine.enhanced_infer(df, H, **kw)
    elif variant.startswith("-"):
        v = variant[1:]
        if v == "hdbscan":
            labels, camps = engine.enhanced_infer(df, H, use_hdb=False, **kw)
        elif v == "test":
            labels, camps = engine.enhanced_infer(df, H, use_test=False, **kw)
        else:
            labels, camps = engine.enhanced_infer(df, H, drop=(v,), **kw)
    elif variant == "+sync":
        labels, camps = engine.enhanced_infer(df, H, use_sync=True, **kw)
    elif variant == "porttime":
        key = df.portsig.astype(str) + "|" + df.hfirst.astype(str)
        codes, uniq = pd.factorize(key)
        cnt = np.bincount(codes)
        labels = np.where(cnt[codes] >= 10, codes, -1)
        camps = None
    engine.EPS, engine.MCS_FRAC = eps0, frac0
    m = evaluate(df, labels, truth, camps)
    m.update({"name": name, "seed": seed, "scale": scale, "variant": variant,
              "n_campaigns": int(len(set(labels[labels >= 0])))})
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bg", required=True, help="background profiles (24h, real)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--procs", type=int, default=16)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--only-abl", action="store_true", help="default configuration only")
    ap.add_argument("--hp", action="store_true", help="add hyper-parameter variants")
    a = ap.parse_args()
    jobs = []
    seeds = range(1, a.seeds + 1)
    rates = [3, 12, 60] if not a.quick else [12]
    bots = [25, 100, 400] if not a.quick else [100]
    for s in seeds:
        rng = np.random.default_rng(s)
        if a.only_abl:
            rates_, bots_ = [], []
        for r in (rates if not a.only_abl else []):
            for n in bots:
                jobs.append(("sweep_r%d_n%d" % (r, n), mix(8, n, r, rng), s, 1.0, "ours"))
        for scale, nm in (((1.0, "/13"), (1 / 8, "/16"), (1 / 128, "/20")) if not a.only_abl
                          else ()):
            jobs.append(("scale_%s" % nm, mix(8, 100, 60, rng), s, scale, "ours"))
        for K in (((1, 4, 8, 16) if not a.quick else (8,)) if not a.only_abl else ()):
            jobs.append(("conc_K%d" % K, mix(K, 100, 30, rng, same_port=23), s, 1.0, "ours"))
        base = mix(8, 100, 12, rng)
        hp = ["eps0.25", "eps0.5", "eps1.0", "eps1.5", "mcs0.05", "mcs0.2", "B100", "B400"]
        for v in ["ours", "porttime", "-fingerprint", "-temporal", "-rate", "-coverage",
                  "-dispersion", "-strategy", "-hdbscan", "+sync", "-test"] + (hp if a.hp else []):
            jobs.append(("abl", base, s, 1.0, v))
    with Pool(a.procs, initializer=_init, initargs=(a.bg,)) as pool:
        res = pool.map(run_one, jobs, chunksize=1)
    with open(a.out, "w") as fh:
        json.dump(res, fh, indent=1)
    print("jobs", len(res))


if __name__ == "__main__":
    main()
