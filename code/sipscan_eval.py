#!/usr/bin/env python3
"""Independent ground truth for the orchestration test (Paper 3, auxiliary evaluation): the 2011
sipscan, a /0 SIP scan by the Sality botnet whose probes the UCSD Network Telescope captured
(CAIDA, "The CAIDA UCSD Network Telescope on the Sipscan Dataset"; Dainotti et al., IMC 2012).
Run in the work directory next to the week profiles (prof/week_*.tsv, week_results.json).

  prep   : sipscan/sipscan.release_dataset.gz -> sipscan/sipscan.npz (per-packet arrays). The
           dataset holds every sipscan probe that reached the telescope /8 (time, anonymized source,
           country, ASN, destination with its first octet replaced, source port) and no header
           fields, so OrchScope's tool fingerprint, and hence its blocking, cannot be evaluated;
           only the orchestration test is.
  check  : the bot profiles built here equal the production reducer + profiler's (one slice).
  week   : ORION-sized views. The /8 splits into 32 /13 slices, each the size of the ORION
           telescope; a 7-day window starts at 00:00 UTC on each of Jan 31 - Feb 7. A bot is
           profiled once it has >= 2 packets in the slice (the profiler keeps sources with >= 2
           destinations or packet samples). Sub-campaigns of m sipscan bots, and same-size groups
           of ORION's own port-5060 sources (negative controls), go through the two-null test
           against ORION's port-5060 population of its week (Sep 24-30, 2026). A group is flagged
           if Benjamini-Hochberg (q = 0.05) would flag it among the week's 10,398 tested campaigns.
                                                                  -> sipscan_week_<peers>.json
  stream : hourly replay with the deployment's weekly landmark (Thursday 00:00 UTC, as for ORION)
           and trailing whole days, for the windows holding the onset, the steady scan and its
           decline, and the restart; the campaign at each update is every sipscan bot profiled
           in the slice so far (and a random sub-campaign of 30 of them).
                                                                  -> sipscan_stream_<peers>.json
  --peers all : the population null draws from ORION's whole port-5060 population, as deployed;
  --peers scan: only from its sources outside residue groups. In ORION's week, 18,793 of the 21,067
           port-5060 sources sit in residue groups (one event: 18,893 sources whose modal dark
           destination is the same address, 12,738 of them active in one hour), so "all" pits any
           campaign against peers far more co-active than scanners.
Only aggregates leave the analysis server (CAIDA AUA; Merit data-use agreement).
"""
import argparse, base64, collections, datetime, glob, gzip, json, os, re, struct, time
from multiprocessing import Pool

import numpy as np
import pandas as pd

import engine as E

DIR = "sipscan"
UTC = datetime.timezone.utc
DAY0 = datetime.datetime(2011, 1, 31, tzinfo=UTC)          # the scan starts Jan 31, 21:07 UTC
NWIN = 8                                                    # weekly windows from Jan 31 .. Feb 7
H = 168
SBITS = 19                                                  # /13 slice of the /8: 2^19 addresses
NSLICE = 1 << (24 - SBITS)
PORT = 5060
SIZES = (10, 30, 100, 300, 1000)
K_SIP = 5                                                   # sub-campaigns per window, slice, size
K_CTL = 320                                                 # negative controls per size
Q = 0.05
SEED = 2011
D0 = 591396864                                              # ORION's base address (profiles)


# ----------------------------------------------------------------------------- prep
def ip_int(s):
    a, b, c, d = s.split(".")
    return (int(a) << 24) | (int(b) << 16) | (int(c) << 8) | int(d)


def cmd_prep(src=os.path.join(DIR, "sipscan.release_dataset.gz"),
             out=os.path.join(DIR, "sipscan.npz")):
    """The ASN field can hold a multi-origin or AS-set value with commas: split rows from both ends."""
    t, sid, cc, asn, dst, sport, ccs = [], [], [], [], [], [], {}
    lead = re.compile(r"\d+")
    with gzip.open(src, "rt", encoding="ascii", errors="replace") as fh:
        head = fh.readline().strip().split(",")
        assert head == ["time", "src_id", "cntry", "lat", "long", "asn", "dst_ip", "src_port"], head
        for line in fh:
            f = line.rstrip("\n").split(",")
            t.append(float(f[0])); sid.append(ip_int(f[1]))
            cc.append(ccs.setdefault(f[2], len(ccs)))
            m = lead.match(f[5]); asn.append(int(m.group()) if m else 0)
            dst.append(ip_int(f[-2]) & 0xFFFFFF); sport.append(int(f[-1]))
    t = np.array(t); o = np.argsort(t, kind="stable")
    np.savez(out, t=t[o], src=np.array(sid, np.int32)[o], cc=np.array(cc, np.int16)[o],
             asn=np.array(asn, np.int32)[o], dst24=np.array(dst, np.int32)[o],
             sport=np.array(sport, np.int32)[o], cc_names=np.array(sorted(ccs, key=ccs.get)))
    print("packets", len(t), "sources", len(np.unique(sid)), "t", t.min(), t.max())


D = {}


def load_sip():
    if not D:
        Z = np.load(os.path.join(DIR, "sipscan.npz"))
        D.update({k: Z[k] for k in Z.files})
        D["slice"] = (D["dst24"] >> SBITS).astype(np.int16)
    return D


# ----------------------------------------------------------------------------- bot profiles
def bots(s, t0, nh):
    """Profiled sipscan bots of /13 slice s in [t0, t0 + nh hours), as the test sees them: active-hour
    bitmap (hour 0 = t0; nh <= 192), active hours, country, modal destination (residue rule), and
    the time each became profiled (its second packet). Returns (frame, bitmaps) or None."""
    S = load_sip()
    lo = np.searchsorted(S["t"], t0)
    hi = np.searchsorted(S["t"], t0 + nh * 3600)
    idx = lo + np.where(S["slice"][lo:hi] == s)[0]
    if len(idx) == 0:
        return None
    t, src = S["t"][idx], S["src"][idx]
    hr = ((t - t0) // 3600).astype(np.int64)
    u, inv, cnt = np.unique(src, return_inverse=True, return_counts=True)
    keep = cnt >= 2
    if not keep.any():
        return None
    W = np.zeros((len(u), 3), np.uint64)
    np.bitwise_or.at(W, (inv, hr >> 6), np.left_shift(np.uint64(1), (hr & 63).astype(np.uint64)))
    first = np.full(len(u), len(idx)); np.minimum.at(first, inv, np.arange(len(idx)))
    rest = np.ones(len(idx), bool); rest[first] = False
    second = np.full(len(u), np.inf); np.minimum.at(second, inv[rest], t[rest])
    off = (S["dst24"][idx] & ((1 << SBITS) - 1)).astype(np.int64)
    key = (inv.astype(np.int64) << SBITS) | off
    uk, kfirst, kc = np.unique(key, return_index=True, return_counts=True)
    kb = uk >> SBITS
    o = np.lexsort((kfirst, -kc, kb))                         # most packets, then earliest
    pick = o[np.unique(kb[o], return_index=True)[1]]
    dmode = np.full(len(u), -1, np.int64); dmode[kb[pick]] = uk[pick] & ((1 << SBITS) - 1)
    bits = [int(a) | (int(b) << 64) | (int(c) << 128) for a, b, c in W[keep]]
    F = pd.DataFrame({
        "src": ["sip%d" % x for x in u[keep]], "top1": PORT, "portsig": "p%d" % PORT,
        "nact": np.bitwise_count(W[keep]).sum(1).astype(int), "hbits": ["%x" % b for b in bits],
        "cc": S["cc_names"][S["cc"][idx][first[keep]]].astype(str), "dmode": dmode[keep],
        "pk": cnt[keep], "t_prof": second[keep]})
    return F, bits


def orion5060():
    """ORION's port-5060 sources of the week, with `residue`: the source belongs to a group the
    week's run labeled residue (members converging on one dark address)."""
    cols = ["src", "top1", "portsig", "nact", "hbits", "cc", "dmode", "pk"]
    frames = [pd.read_csv(p, sep="\t", usecols=cols, dtype={"src": str, "cc": str, "hbits": str,
                                                              "portsig": str},
                          keep_default_na=False, na_values=["nan"])
              for p in sorted(glob.glob("prof/week_*.tsv"))]
    df = pd.concat(frames, ignore_index=True)
    labels = np.load("week_results_labels.npy")
    assert len(labels) == len(df)
    L = pd.DataFrame(json.load(open("week_results.json"))["landscape"])
    res_ids = set(L.id[L.residue].astype(int))
    keep = (df.top1 == PORT).to_numpy()
    df = df[keep].reset_index(drop=True)
    df["residue"] = [int(x) in res_ids for x in labels[keep]]
    df["dmode"] = E.pd_numeric(df.dmode)
    return df


def week_family():
    L = pd.DataFrame(json.load(open("week_results.json"))["landscape"])
    L = L[~L.residue]
    return L.p.to_numpy(float), L.p_rot.to_numpy(float)


def flagged(fam, p):
    return bool(E.bh(np.append(fam, p), q=Q)[-1])


def bh_threshold(fam):
    """Largest p that BH would flag when added to the family (bisection on the monotone rule)."""
    lo, hi = 0.0, Q
    for _ in range(60):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if flagged(fam, mid) else (lo, mid)
    return lo


# ----------------------------------------------------------------------------- week
G = {}


def setup(peers):
    """peers = "all": ORION's whole port-5060 population, as the deployed engine uses it;
    "scan": only its sources outside residue groups (the port's scanners). Negative controls are
    always drawn from non-residue sources: random groups of residue sources are residue themselves
    and never tested."""
    load_sip()
    O = orion5060()
    if peers == "scan":
        O = O[~O.residue].reset_index(drop=True)
    G.update(orion=O, obits=[int(b, 16) if b else 0 for b in O.hbits],
             n_cc=O.cc.value_counts().to_dict(), ctl_pool=np.where(~O.residue.to_numpy())[0],
             peers=peers)
    G["oP"] = E.Pops(O, G["obits"], key="top1")


def test_members(df, bits, members, rng, own, P=None):
    """Two-null test of one group. own: True if the members are ORION sources themselves (they are
    then not their own country peers)."""
    conv, res = E.convergence(members, df.dmode.to_numpy())
    if res:
        return {"residue": True, "conv": conv}
    P = P if P is not None else E.Pops(df, bits, key="top1")
    r = E.sync_test(members, None, bits, H, P, rng, B=200)
    cc = df.cc.to_numpy()[members]
    mc = collections.Counter(cc)
    fb = float(np.mean([G["n_cc"].get(c, 0) - (mc[c] if own else 0) < 40 for c in cc]))
    r.update({"residue": False, "conv": conv, "fallback": fb,
              "nact_med": float(np.median(df.nact.to_numpy()[members]))})
    return r


def job_ws(a):
    """All tests of one window and slice: K_SIP sub-campaigns per size, and the full campaign."""
    w, s = a
    x = bots(s, DAY0.timestamp() + w * 86400, H)
    if x is None:
        return [], None, 0, {}
    F, fb = x
    O, ob = G["orion"], G["obits"]
    R = []
    for m in SIZES:
        if len(F) < m:
            continue
        for k in range(K_SIP):
            rng = np.random.default_rng([SEED, 1, w, s, m, k])
            pick = rng.choice(len(F), m, replace=False)
            df = pd.concat([O, F.iloc[pick]], ignore_index=True)
            r = test_members(df, ob + [fb[i] for i in pick], np.arange(len(O), len(df)), rng, False)
            r.update({"w": w, "s": s, "m": m, "k": k})
            R.append(r)
    rng = np.random.default_rng([SEED, 3, w, s])
    df = pd.concat([O, F], ignore_index=True)
    bits = ob + fb
    mem = np.arange(len(O), len(df))
    P = E.Pops(df, bits, key="top1")
    port = E.campaign_port(P, mem)
    local = E.local_pairs(len(mem), rng)
    _, p_rot, z_rot = E.coordination_test(mem, bits, H, rng, B=200, local=local)
    _, p_pop, z_pop = E.population_test(mem, port, P, rng, B=200, local=local)  # testability waived
    full = {"w": w, "s": s, "n": int(len(mem)), "testable": bool(E.testable(P, mem, port)),
            "p_rot": p_rot, "z_rot": z_rot, "p_pop_waived": p_pop, "z_pop_waived": z_pop}
    return R, full, int(len(F)), dict(collections.Counter(F.cc))


def job_ctl(a):
    m, k = a
    rng = np.random.default_rng([SEED, 2, m, k])
    O = G["orion"]
    r = test_members(O, G["obits"], rng.choice(G["ctl_pool"], m, replace=False), rng, True,
                     P=G["oP"])
    r.update({"m": m, "k": k})
    return r


def summarize(R, fam, fam_rot, keys):
    out = []
    df = pd.DataFrame([r for r in R if r is not None])
    for g, X in df.groupby(keys):
        T = X[~X.residue.astype(bool)]
        p, pr = T.p.to_numpy(float), T.p_rot.to_numpy(float)
        out.append({**dict(zip(keys, g if isinstance(g, tuple) else (g,))),
                    "groups": int(len(X)), "residue": int(X.residue.astype(bool).sum()),
                    "tested": int(len(T)), "testable": float(T.testable.astype(bool).mean()),
                    "flag": float(np.mean([flagged(fam, x) for x in p])),
                    "flag_rot": float(np.mean([flagged(fam_rot, x) for x in pr])),
                    "raw05": float((p < 0.05).mean()), "raw01": float((p < 0.01).mean()),
                    "raw05_rot": float((pr < 0.05).mean()), "raw01_rot": float((pr < 0.01).mean()),
                    "p_med": float(np.median(p)), "p_rot_med": float(np.median(pr)),
                    "fallback": float(T.fallback.mean()), "nact_med": float(T.nact_med.median())})
    return out


def residue_event():
    """ORION's port-5060 population of the week: sources sharing the most common modal dark
    destination, and the most sources whose only active hour is the same hour."""
    O = orion5060()
    dm = O.dmode.to_numpy()
    _, c = np.unique(dm[dm >= 0], return_counts=True)
    one = [int(b, 16).bit_length() - 1 for b, n in zip(O.hbits, O.nact) if n == 1 and b]
    _, h = np.unique(one, return_counts=True)
    return {"sources": int(len(O)), "residue": int(O.residue.sum()),
            "top_dest_sources": int(c.max()), "single_hour_max": int(h.max())}


def controls_bh(C):
    """BH (q = 0.05) applied to the negative controls' own p-values, as the paper's real-traffic
    negative control does (per size and pooled), with the p-values themselves."""
    C = [r for r in C if r is not None and not r["residue"]]
    out = {}
    for name, X in [("m%d" % m, [r for r in C if r["m"] == m]) for m in SIZES] + [("all", C)]:
        p, pr = [r["p"] for r in X], [r["p_rot"] for r in X]
        out[name] = {"groups": len(X), "flagged": int(E.bh(p, q=Q).sum()),
                     "flagged_rot": int(E.bh(pr, q=Q).sum())}
        if name != "all":
            out[name]["p"] = [round(float(x), 6) for x in p]
            out[name]["p_rot"] = [round(float(x), 6) for x in pr]
    return out


def cmd_week(peers, procs=48):
    out = "sipscan_week_%s.json" % peers
    t0 = time.time()
    setup(peers)
    fam, fam_rot = week_family()
    O = G["orion"]
    with Pool(procs) as pool:
        WS = pool.map(job_ws, [(w, s) for w in range(NWIN) for s in range(NSLICE)], chunksize=1)
        C = pool.map(job_ctl, [(m, k) for m in SIZES for k in range(K_CTL)], chunksize=4)
    print("tests done", round(time.time() - t0), "s", flush=True)
    R = [r for x in WS for r in x[0]]
    full = [x[1] for x in WS if x[1] is not None]
    sipcc = collections.Counter()
    for x in WS:
        sipcc.update(x[3])
    nb = sum(sipcc.values())
    res = {"peers": peers, "orion_5060_sources": int(len(O)),
           "orion_5060_residue_sources": int(O.residue.sum()),
           "orion_5060_countries_ge40": sorted(c for c, v in G["n_cc"].items() if v >= 40),
           "bh_threshold": bh_threshold(fam), "bh_threshold_rot": bh_threshold(fam_rot),
           "week_tested": int(len(fam)),
           "profiled_bots": [[WS[w * NSLICE + s][2] for s in range(NSLICE)] for w in range(NWIN)],
           "sip_top_cc": [(c, v / nb) for c, v in sipcc.most_common(12)],
           "sip_share_cc_ge40_orion": float(sum(v for c, v in sipcc.items()
                                                if G["n_cc"].get(c, 0) >= 40) / nb),
           "by_window_size": summarize(R, fam, fam_rot, ["w", "m"]),
           "by_size": summarize(R, fam, fam_rot, ["m"]),
           "controls_by_size": summarize(C, fam, fam_rot, ["m"]),
           "controls_bh_own": controls_bh(C), "orion_5060": residue_event(),
           "full": full, "seconds": round(time.time() - t0)}
    json.dump(res, open(out, "w"), indent=1, default=float)
    for k in ("by_size", "controls_by_size"):
        print(k)
        for r in res[k]:
            print(json.dumps(r))
    print("full-scale testable:", sum(f["testable"] for f in full), "of", len(full))


# ----------------------------------------------------------------------------- stream
# Weekly landmarks on Thursdays, 00:00 UTC, as for the ORION week (Thursday, Sep 24, 2026): the
# onset (Jan 31, 21:07) falls on day 4 of the first window, the restart (Feb 11, ~14:00) on day 1
# of the third; the second window holds the steady scan and its decline (Feb 6 from ~13:00).
LANDMARKS = {"onset": datetime.datetime(2011, 1, 27, tzinfo=UTC),
             "steady": datetime.datetime(2011, 2, 3, tzinfo=UTC),
             "restart": datetime.datetime(2011, 2, 10, tzinfo=UTC)}
M_STREAM = 30


def job_stream(a):
    """Hourly updates of one landmark window at one slice. At the end of hour h the campaign is
    every sipscan bot profiled in the slice so far (and a random sub-campaign of M_STREAM of them);
    the test sees the trailing whole days, and the peers are ORION's port-5060 sources seen by the
    same hour of ORION's week (also from a Thursday landmark)."""
    ev, s = a
    L0 = LANDMARKS[ev].timestamp()
    O, ob = G["orion"], G["obits"]
    out = []
    for h in range(47, H):                                     # decisions from two whole days
        n = h + 1
        days = n // 24
        off, Hv = n - 24 * days, 24 * days
        x = bots(s, L0, n)
        if x is None:
            continue
        F, fb = x
        vis = (1 << n) - 1
        pb = [b & vis for b in ob]
        keep = np.array([b != 0 for b in pb])
        Pdf = O[keep].assign(nact=[b.bit_count() for b, k in zip(pb, keep) if k])
        pbits = [b >> off for b, k in zip(pb, keep) if k]
        row = {"ev": ev, "s": s, "h": h, "n": int(len(F)), "peers": int(len(Pdf))}
        for j, tag in enumerate(("full", "m%d" % M_STREAM)):
            rng = np.random.default_rng([SEED, 4, list(LANDMARKS).index(ev), s, h, j])
            if j == 0:
                pick = np.arange(len(F))
            elif len(F) >= M_STREAM:
                pick = rng.choice(len(F), M_STREAM, replace=False)
            else:
                continue
            df = pd.concat([Pdf, F.iloc[pick]], ignore_index=True)
            bits = pbits + [fb[i] >> off for i in pick]
            mem = np.arange(len(Pdf), len(df))
            P = E.Pops(df, bits, key="top1")
            r = E.sync_test(mem, None, bits, Hv, P, rng, B=200)
            row.update({tag + "_" + k: r[k] for k in ("p", "p_rot", "p_pop", "testable")})
        out.append(row)
    return out


def cmd_stream(peers, procs=48):
    out = "sipscan_stream_%s.json" % peers
    t0 = time.time()
    setup(peers)
    fam, fam_rot = week_family()
    tau, tau_rot = bh_threshold(fam), bh_threshold(fam_rot)
    with Pool(procs) as pool:
        R = pool.map(job_stream, [(ev, s) for ev in LANDMARKS for s in range(NSLICE)], chunksize=1)
    R = [r for x in R for r in x]
    for r in R:
        for tag in ("full", "m%d" % M_STREAM):
            if tag + "_p" in r:
                r[tag + "_flag"] = bool(r[tag + "_testable"] and r[tag + "_p"] <= tau)
                r[tag + "_flag_rot"] = bool(r[tag + "_p_rot"] <= tau_rot)
    json.dump({"tau": tau, "tau_rot": tau_rot,
               "landmarks": {k: v.isoformat() for k, v in LANDMARKS.items()},
               "rows": R, "seconds": round(time.time() - t0)}, open(out, "w"), default=float)
    df = pd.DataFrame(R)
    for ev, X in df.groupby("ev"):
        g = X.groupby("h").agg(n=("n", "median"), full=("full_flag", "mean"),
                               full_testable=("full_testable", "mean"),
                               full_rot=("full_flag_rot", "mean"), m30=("m30_flag", "mean"),
                               m30_rot=("m30_flag_rot", "mean"))
        print(ev)
        print(g.to_string())


# ----------------------------------------------------------------------------- check
def cmd_check(w=0, s=0, max_bots=3000):
    """Rebuild the profiles of up to max_bots bots of one slice-window through the production path
    (ORION JSON records -> camp_reduce -> engine.accumulate_rows -> engine.profile; one record per
    bot, hour and port with its first three packets as samples, as in simulate.py) and compare the
    fields the test uses. The samples carry the real destination and source port; the header
    fields the dataset lacks are zero, so the header-derived columns are not compared."""
    import camp_reduce
    S = load_sip()
    t0 = DAY0.timestamp() + w * 86400
    F, _ = bots(s, t0, H)
    F = F.head(max_bots)
    want = np.array(sorted(int(x[3:]) for x in F.src))
    lo, hi = np.searchsorted(S["t"], t0), np.searchsorted(S["t"], t0 + H * 3600)
    idx = lo + np.where((S["slice"][lo:hi] == s) & np.isin(S["src"][lo:hi], want))[0]
    w0 = datetime.datetime.fromtimestamp(t0, UTC).replace(tzinfo=None)
    rec = collections.defaultdict(list)
    for i in idx:
        rec[(int((S["t"][i] - t0) // 3600), int(S["src"][i]))].append(i)

    def iso(x):
        return datetime.datetime.fromtimestamp(x, UTC).strftime("%Y-%m-%dT%H:%M:%S.%fZ")
    lines = collections.defaultdict(list)
    for (h, sid), ii in rec.items():
        ds = [D0 + int(S["dst24"][i] & ((1 << SBITS) - 1)) for i in ii]
        smp = []
        for i, d in list(zip(ii, ds))[:3]:
            ip = struct.pack("!BBHHHBBH4s4s", 0x45, 0, 28, 0, 0, 0, 17, 0, sid.to_bytes(4, "big"),
                             d.to_bytes(4, "big"))
            udp = struct.pack("!HHHH", int(S["sport"][i]), PORT, 8, 0)
            smp.append(base64.b64encode(b"\0" * 12 + b"\x08\x00" + ip + udp).decode())
        lines[h].append(json.dumps({"SourceIP": "sip%d" % sid, "Port": PORT, "Traffic": 16,
                                    "First": iso(S["t"][ii[0]]), "Last": iso(S["t"][ii[-1]]),
                                    "Packets": len(ii), "Bytes": 0, "UniqueDests": len(set(ds)),
                                    "UniqueDest24s": len(set(d >> 8 for d in ds)),
                                    "Country": str(S["cc_names"][S["cc"][ii[0]]]), "ASN": 0,
                                    "Samples": smp}))
    rows = []
    for h in sorted(lines):
        label = (w0 + datetime.timedelta(hours=h)).strftime("%Y-%m-%d.%H")
        rows += [r.rstrip("\n").split("\t") for r in camp_reduce.reduce_lines(lines[h], label)]
    acc, _ = E.accumulate_rows(iter(rows), w0)
    prof = {k: E.profile(k, a, D0, H) for k, a in acc.items() if a.ud >= 2 or len(a.sam) >= 2}
    col = {c: i for i, c in enumerate(E.PROF_COLS)}
    ok = collections.Counter(bots=len(F), production_profiled=len(prof))
    for r in F.itertuples():
        p = prof.get(r.src)
        if p is None:
            continue
        for c in ("hbits", "nact", "cc", "top1", "dmode"):
            ok[c] += int(str(p[col[c]]) == str(getattr(r, c)))
    print(json.dumps(ok))
    json.dump(ok, open("sipscan_check.json", "w"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["prep", "check", "week", "stream"])
    ap.add_argument("--procs", type=int, default=48)
    ap.add_argument("--peers", choices=["all", "scan"], default="all")
    a = ap.parse_args()
    if a.cmd == "prep":
        cmd_prep()
    elif a.cmd == "check":
        cmd_check()
    elif a.cmd == "week":
        cmd_week(a.peers, procs=a.procs)
    else:
        cmd_stream(a.peers, procs=a.procs)


if __name__ == "__main__":
    main()
