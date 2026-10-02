#!/usr/bin/env python3
"""OrchScope engine: per-source profiles, campaign inference, and the orchestration test.

Stages (CLI)
  features : reduced source-hour rows (camp_reduce.py) -> one profile per source
  infer    : campaign inference (fingerprint blocks + HDBSCAN-eps + residue check) and the
             two-null orchestration test (exact Monte Carlo p-values, intersection-union, BH)

Profiles adapt classical tests (Wald-Wolfowitz runs, Poisson rate CI, Mann-Kendall,
chi-square uniformity, relative uncertainty) to telescope flow records with sparse packet
samples, and add packet-header fingerprints recovered from the per-record samples.
"""
import argparse, collections, datetime, glob, io, json, math, os, random, sys, zlib

import numpy as np
from scipy import stats

DBITS = 19
DSIZE = 1 << DBITS
EPS = 0.75  # HDBSCAN cluster_selection_epsilon (standardized units): do not split closer modes
MCS_FRAC = 0.1  # minimum mode size as a fraction of the block (HDBSCAN min_cluster_size)
KNOWN_WIN = {"512", "1024", "2048", "3072", "4096", "5840", "8192", "14600", "16384", "29200",
             "32768", "42340", "64240", "65228", "65535"}
RESEARCH_KW = ("censys", "shodan", "shadowserver", "rapid7", "stretchoid", "internet-measurement",
               "binaryedge", "onyphe", "leakix", "criminalip", "netsystemsresearch", "recyber",
               "internettl", "alphastrike", "ipip", "cyber.casa", "scan", "research", "probe")

# ----------------------------------------------------------------------------- I/O

def open_text(path):
    if path.endswith(".zst"):
        import zstandard
        fh = open(path, "rb")
        return io.TextIOWrapper(zstandard.ZstdDecompressor().stream_reader(fh), encoding="utf-8",
                                errors="replace")
    return open(path, encoding="utf-8", errors="replace")


def batch_hour(label, w0):
    """'2026-09-24.07' -> hour index relative to window start datetime w0."""
    d, h = label.split(".")
    dt = datetime.datetime.strptime(d, "%Y-%m-%d") + datetime.timedelta(hours=int(h))
    return int((dt - w0).total_seconds() // 3600)


def parse_samples(s):
    out = []
    if not s:
        return out
    for x in s.split(";"):
        f = x.split(",")
        if len(f) < 13:
            continue
        try:
            out.append((int(f[0]), int(f[1]), int(f[2]), int(f[3]), f[4], int(f[5]), int(f[6]),
                        int(f[7]), f[8], f[9], f[10], f[11], f[12]))
        except ValueError:
            continue
    return out  # t,dport,proto,dst,sport,ttl,ipid,df,win,optsig,seqrel,plen,phead


class Acc:
    __slots__ = ("pm", "nrec", "ports", "nports_max", "pk", "by", "ud", "ud24", "udmax", "t0", "t1",
                 "zm", "ms", "mi", "asn", "cc", "org", "rdns", "sp", "sn", "sb", "hbits", "hpk",
                 "sam", "nsam")

    def __init__(self):
        self.pm = self.nrec = self.nports_max = self.pk = self.by = self.ud = self.ud24 = 0
        self.udmax = self.zm = self.ms = self.mi = self.sp = self.sn = self.sb = 0
        self.ports = collections.Counter()
        self.t0, self.t1 = 1 << 62, 0
        self.asn, self.cc, self.org, self.rdns = "0", "", "", ""
        self.hbits = 0
        self.hpk = {}
        self.sam = []
        self.nsam = 0


SAMPLE_CAP = 160


def iter_rows(paths):
    for p in paths:
        with open_text(p) as fh:
            for line in fh:
                yield line.rstrip("\n").split("\t")


def accumulate(paths, w0, shard=None, nshards=1, hours=None, max_lines=None):
    return accumulate_rows(iter_rows(paths), w0, shard, nshards, hours, max_lines)


def accumulate_rows(rows, w0, shard=None, nshards=1, hours=None, max_lines=None, acc=None):
    acc = {} if acc is None else acc
    nrows = 0
    if True:
        if True:
            for f in rows:
                if len(f) < 25:
                    continue
                src = f[1]
                if nshards > 1 and (zlib.crc32(src.encode()) % nshards) != shard:
                    continue
                h = batch_hour(f[0], w0)
                if hours and not (hours[0] <= h < hours[1]):
                    continue
                a = acc.get(src)
                if a is None:
                    a = acc[src] = Acc()
                    a.asn, a.cc, a.org, a.rdns = f[17], f[18], f[19], f[20]
                nrows += 1
                a.pm |= int(f[2]); a.nrec += int(f[3])
                np_ = int(f[4])
                if np_ > a.nports_max:
                    a.nports_max = np_
                for kv in f[5].split(","):
                    if kv:
                        pp, cc = kv.split(":")
                        a.ports[int(pp)] += int(cc)
                pk = int(f[6]); a.pk += pk; a.by += int(f[7])
                a.ud += int(f[8]); a.ud24 += int(f[9]); a.udmax = max(a.udmax, int(f[10]))
                t0, t1 = int(f[11]), int(f[12])
                if t0 and t0 < a.t0:
                    a.t0 = t0
                if t1 > a.t1:
                    a.t1 = t1
                a.zm += int(f[14]); a.ms += int(f[15]); a.mi += int(f[16])
                if not a.rdns and f[20]:
                    a.rdns = f[20]
                a.sp += int(f[21]); a.sn += int(f[22]); a.sb += int(f[23])
                if h >= 0:
                    a.hbits |= (1 << h)
                    a.hpk[h] = a.hpk.get(h, 0) + pk
                for s in parse_samples(f[24]):
                    a.nsam += 1
                    if len(a.sam) < SAMPLE_CAP:
                        a.sam.append(s)
                    else:
                        j = random.randrange(a.nsam)
                        if j < SAMPLE_CAP:
                            a.sam[j] = s
                if max_lines and nrows >= max_lines:
                    return acc, nrows
    return acc, nrows

# ----------------------------------------------------------------------------- statistics


def runs_test(x):
    """Wald-Wolfowitz runs test on a series (dichotomized at the median). Returns p-value."""
    x = np.asarray(x, dtype=float)
    med = np.median(x)
    b = x[x != med] > med
    n1 = int(b.sum()); n2 = int(len(b) - n1)
    if n1 < 2 or n2 < 2:
        return 1.0
    r = 1 + int(np.count_nonzero(b[1:] != b[:-1]))
    mu = 2.0 * n1 * n2 / (n1 + n2) + 1
    var = 2.0 * n1 * n2 * (2 * n1 * n2 - n1 - n2) / ((n1 + n2) ** 2 * (n1 + n2 - 1))
    if var <= 0:
        return 1.0
    z = (r - mu) / math.sqrt(var)
    return float(2 * stats.norm.sf(abs(z)))


def mann_kendall(x):
    """Mann-Kendall trend test (tie-corrected). Returns (z, p)."""
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n < 4:
        return 0.0, 1.0
    d = np.sign(x[None, :] - x[:, None])
    s = float(np.triu(d, 1).sum())
    _, c = np.unique(x, return_counts=True)
    var = (n * (n - 1) * (2 * n + 5) - float((c * (c - 1) * (2 * c + 5)).sum())) / 18.0
    if var <= 0:
        return 0.0, 1.0
    z = (s - np.sign(s)) / math.sqrt(var) if s != 0 else 0.0
    return float(z), float(2 * stats.norm.sf(abs(z)))


def chi2_uniform(off, k):
    h, _ = np.histogram(off, bins=k, range=(0, DSIZE))
    return float(stats.chisquare(h).pvalue)


def rel_uncertainty(vals):
    n = len(vals)
    if n < 2:
        return float("nan")
    _, c = np.unique(vals, return_counts=True)
    p = c / n
    return float(-(p * np.log(p)).sum() / math.log(n))


def init_ttl(t):
    return 32 if t <= 32 else 64 if t <= 64 else 128 if t <= 128 else 255


def mode(xs, default=""):
    if not xs:
        return default
    return collections.Counter(xs).most_common(1)[0][0]


def ipid_class(ids):
    if not ids:
        return "na"
    if all(i == 54321 for i in ids):
        return "ip54321"
    if all(i == 0 for i in ids):
        return "ip0"
    if len(ids) < 2:
        return "ip1"
    if len(set(ids)) == 1:
        return "ipconst"
    d = [(b - a) % 65536 for a, b in zip(ids, ids[1:])]
    if sum(1 for v in d if 1 <= v <= 2048) >= 0.8 * len(d):
        return "ipinc"
    return "iprand"


def sport_policy(sp):
    sp = [int(x) for x in sp if x != ""]
    if len(sp) < 2:
        return "sp1"
    if len(set(sp)) == 1:
        return "spfix"
    if max(sp) - min(sp) < 64:
        return "spnarrow"
    d = [b - a for a, b in zip(sp, sp[1:])]
    if sum(1 for v in d if 1 <= v <= 64) >= 0.8 * len(d):
        return "spinc"
    return "sprand"


def win_bucket(w):
    return w if w in ("", "0") else str(w)

# ----------------------------------------------------------------------------- profiles

PROF_COLS = ["src", "pm", "nrec", "nports", "portsig", "top1", "pk", "ud", "ud24", "nact", "span",
             "hfirst", "hlast", "hbits", "duty", "pph", "fano", "p_ww", "lam_lo", "lam_hi", "nsam",
             "z_mk", "p_mk", "p_chi", "frac_seq", "dir_seq", "strategy", "ru", "rho", "disp", "cov",
             "ttl", "hops", "ipid", "win", "opts", "seqrel", "sportp", "df", "phead", "fp", "fpc", "zmf",
             "msf", "mif", "asn", "cc", "org", "rdns", "research", "dmode", "dmode_k"]


def proto_name(pm):
    return {1: "tcp", 2: "udp", 4: "icmp"}.get(pm, "mix%d" % pm)


def portsig_of(a):
    tops = [p for p, _ in a.ports.most_common(3)]
    if a.nports_max <= 3 and len(a.ports) <= 3:
        return "p" + "+".join(str(p) for p in sorted(tops)), tops[0] if tops else 0
    return "multi", tops[0] if tops else 0


def profile(src, a, d0, H):
    hs = sorted(a.hpk)
    nact = len(hs)
    if nact:
        hf, hl = hs[0], hs[-1]
        span = hl - hf + 1
        series = np.array([a.hpk.get(h, 0) for h in range(hf, hl + 1)], dtype=float)
    else:
        hf = hl = -1; span = 0; series = np.zeros(0)
    duty = nact / span if span else 0.0
    pph = a.pk / max(nact, 1)
    if len(series) >= 3 and series.mean() > 0:
        fano = float(series.var() / series.mean())
    else:
        fano = float("nan")
    p_ww = runs_test(series) if len(series) >= 6 else 1.0
    T = max(span, 1)
    lam_lo = stats.chi2.ppf(0.025, 2 * a.pk) / 2 / T if a.pk > 0 else 0.0
    lam_hi = stats.chi2.ppf(0.975, 2 * a.pk + 2) / 2 / T

    sam = sorted(a.sam, key=lambda s: s[0])
    offs = [s[3] - d0 for s in sam if 0 <= s[3] - d0 < DSIZE]
    nsam = len(offs)
    z_mk, p_mk = mann_kendall(offs) if nsam >= 8 else (0.0, 1.0)
    p_chi = chi2_uniform(offs, min(16, max(2, nsam // 5))) if nsam >= 10 else float("nan")
    st = a.sp + a.sn + a.sb
    frac_seq = (a.sp + a.sn) / st if st else float("nan")
    dir_seq = (a.sp - a.sn) / (a.sp + a.sn) if (a.sp + a.sn) else 0.0
    if st >= 3 and frac_seq >= 0.7:
        strategy = "seq_fwd" if dir_seq > 0 else "seq_rev"
    elif nsam >= 8 and p_mk < 0.005:
        strategy = "seq_fwd" if z_mk > 0 else "seq_rev"
    elif nsam >= 10:
        strategy = "uperm" if p_chi >= 0.05 else "perm"
    else:
        strategy = "undet"
    ru = rel_uncertainty([o >> 8 for o in offs]) if nsam >= 4 else float("nan")
    rho = a.ud24 / a.ud if a.ud else float("nan")
    if not math.isnan(ru):
        disp = "dispersed" if ru >= 0.8 else "targeted"
    else:
        disp = "dispersed" if (rho == rho and rho >= 0.5) else "targeted"
    cov = min(1.0, a.ud / DSIZE)

    ttls = [s[5] for s in sam]
    ttl0 = mode([init_ttl(t) for t in ttls], 0)
    hops = int(ttl0 - float(np.median(ttls))) if ttls else -1
    ipid = ipid_class([s[6] for s in sam])
    win = mode([s[8] for s in sam if s[2] == 6], "")
    opts = mode([s[9] for s in sam if s[2] == 6], "")
    seqrel = "D" if sum(1 for s in sam if s[10] == "D") > 0.5 * max(1, len(sam)) else \
             ("Z" if sum(1 for s in sam if s[10] == "Z") > 0.5 * max(1, len(sam)) else "-")
    sportp = sport_policy([s[4] for s in sam])
    df = mode([s[7] for s in sam], 0)
    phead = mode([s[12] for s in sam if s[2] == 17], "")   # 8-byte payload head (hex)
    pn = proto_name(a.pm)
    if pn == "udp":
        fp = "udp|t%d|%s|%s|df%d|ph%s" % (ttl0, ipid, sportp, df, phead[:4])
    else:
        fp = "%s|t%d|%s|w%s|o%s|s%s|%s|df%d" % (pn, ttl0, ipid, win_bucket(win), opts or "-",
                                                seqrel, sportp, df)
    # Core fingerprint (blocking key): only fields measurable from a single packet and stable
    # per tool, so per-packet randomization (e.g. Mirai's random window) is a class, not a value.
    ids = [s[6] for s in sam]
    ipc = "54321" if ids and all(i == 54321 for i in ids) else \
          ("0" if ids and all(i == 0 for i in ids) else "x")
    wins = [s[8] for s in sam if s[2] == 6]
    winc = wins[0] if (wins and len(set(wins)) == 1 and wins[0] in KNOWN_WIN) else "x"
    if pn == "udp":
        fpc = "udp|t%d|df%d|i%s|ph%s" % (ttl0, df, ipc, phead[:4])
    else:
        fpc = "%s|t%d|df%d|o%s|s%s|i%s|w%s" % (pn, ttl0, df, opts or "-", seqrel, ipc, winc)
    portsig, top1 = portsig_of(a)
    nr = max(a.nrec, 1)
    low = (a.rdns + " " + a.org).lower()
    research = int(any(k in low for k in RESEARCH_KW[:-3]) or ("scan" in a.rdns.lower()))
    # modal sampled dark destination: residue converges on one address, scanning does not
    dmode, dmode_k = collections.Counter(offs).most_common(1)[0] if offs else (-1, 0)
    return [src, a.pm, a.nrec, a.nports_max, portsig, top1, a.pk, a.ud, a.ud24, nact, span, hf, hl,
            "%x" % a.hbits, round(duty, 4), round(pph, 3), fano, p_ww, lam_lo, lam_hi, nsam,
            round(z_mk, 3), p_mk, p_chi, frac_seq, round(dir_seq, 3), strategy, ru, rho, disp,
            cov, ttl0, hops, ipid, win, opts, seqrel, sportp, df, phead, fp, fpc, a.zm / nr,
            a.ms / nr, a.mi / nr, a.asn, a.cc, a.org, a.rdns, research, dmode, dmode_k]


def infer_d0(paths, n=20000):
    c = collections.Counter()
    for p in paths[:2]:
        with open_text(p) as fh:
            for i, line in enumerate(fh):
                f = line.rstrip("\n").split("\t")
                if len(f) >= 25:
                    for s in parse_samples(f[24]):
                        c[s[3] >> DBITS] += 1
                if i > n:
                    break
    top = c.most_common(1)[0][0]
    return top << DBITS


def cmd_features(args):
    paths = sorted(glob.glob(os.path.join(args.red, args.glob + ".tsv.zst")) +
                   glob.glob(os.path.join(args.red, args.glob + ".tsv")))
    if args.max_files:
        paths = paths[:args.max_files]
    w0 = datetime.datetime.strptime(args.w0, "%Y-%m-%d")
    d0 = args.d0 if args.d0 else infer_d0(paths)
    hours = tuple(int(x) for x in args.hours.split(":")) if args.hours else None
    random.seed(args.shard)
    acc, nrows = accumulate(paths, w0, args.shard, args.nshards, hours)
    H = 24 * 7
    n_all = len(acc)
    n_cand = 0
    with open(args.out, "w", encoding="utf-8") as out:
        out.write("\t".join(PROF_COLS) + "\n")
        for src, a in acc.items():
            if a.ud < args.min_ud and len(a.sam) < 2:
                continue
            n_cand += 1
            row = profile(src, a, d0, H)
            out.write("\t".join(("%.6g" % v) if isinstance(v, float) else str(v) for v in row) + "\n")
    meta = {"rows": nrows, "sources": n_all, "candidates": n_cand, "d0": d0, "files": len(paths)}
    with open(args.out + ".meta.json", "w") as fh:
        json.dump(meta, fh)
    print(json.dumps(meta))

# ----------------------------------------------------------------------------- inference


def load_profiles(paths):
    import pandas as pd
    frames = [pd.read_csv(p, sep="\t", dtype={"asn": str, "cc": str, "org": str, "rdns": str,
                                               "win": str, "opts": str, "phead": str,
                                               "hbits": str, "src": str},
                          keep_default_na=False, na_values=["nan"]) for p in paths]
    return pd.concat(frames, ignore_index=True)


# Clustering features deliberately exclude absolute time (first/last hour): synchrony is what the
# coordination test measures, so clustering on it would make the test circular.
NUM_FEATS = ["l_pph", "l_ud", "duty", "rho_c", "ru_c", "frac_seq_c", "dir_seq", "l_nports",
             "st_seq_fwd", "st_seq_rev", "st_uperm", "st_perm", "ipinc_n", "spstable_n"]
FEATURE_GROUPS = {"temporal": ["duty"], "rate": ["l_pph"], "coverage": ["l_ud"],
                  "dispersion": ["rho_c", "ru_c"],
                  "strategy": ["frac_seq_c", "dir_seq", "st_seq_fwd", "st_seq_rev", "st_uperm",
                               "st_perm"],
                  "targets": ["l_nports"], "policy": ["ipinc_n", "spstable_n"]}


def numeric_matrix(df, H, drop=()):
    X = {}
    X["l_pph"] = np.log10(df.pph.clip(lower=1e-3))
    X["l_ud"] = np.log10(df.ud.clip(lower=1))
    X["duty"] = df.duty
    X["rho_c"] = df.rho.fillna(0.5)
    X["ru_c"] = df.ru.fillna(0.5)
    X["frac_seq_c"] = df.frac_seq.fillna(0.0)
    X["dir_seq"] = df.dir_seq
    X["l_nports"] = np.log10(df.nports.clip(lower=1))
    undet = (df.strategy == "undet").to_numpy()
    for s in ("seq_fwd", "seq_rev", "uperm", "perm"):
        X["st_" + s] = np.where(undet, 0.25, (df.strategy == s).astype(float))  # undet: neutral
    X["ipinc_n"] = np.where(df.ipid == "ipinc", 1.0,
                            np.where(df.ipid.isin(["ip1", "na"]), 0.5, 0.0))
    X["spstable_n"] = np.where(df.sportp.isin(["spfix", "spnarrow"]), 1.0,
                               np.where(df.sportp == "sp1", 0.5, 0.0))
    groups = FEATURE_GROUPS
    cols = [c for c in NUM_FEATS if not any(c in groups.get(g, []) for g in drop)]
    M = np.column_stack([np.asarray(X[c], dtype=float) for c in cols]) if cols else \
        np.zeros((len(df), 1))
    M = np.nan_to_num(M, nan=0.0, posinf=0.0, neginf=0.0)
    sd = M.std(axis=0); sd[sd == 0] = 1
    return (M - M.mean(axis=0)) / sd


def hdbscan_labels(M, min_size, rng, cap=12000):
    """Split a (port, tool-fingerprint) block into behavioral modes. A homogeneous block stays
    one campaign (allow_single_cluster); points HDBSCAN calls noise are reassigned to the
    nearest mode when within 1.5x that mode's 95th-percentile radius."""
    from sklearn.cluster import HDBSCAN
    n = len(M)
    if n < 2 * min_size:
        return np.zeros(n, dtype=np.int64)
    mcs = max(min_size, int(n * MCS_FRAC))
    fit_idx = np.arange(n) if n <= cap else rng.choice(n, cap, replace=False)
    sub = HDBSCAN(min_cluster_size=mcs, min_samples=5, allow_single_cluster=True, cluster_selection_epsilon=EPS,
                  copy=True).fit_predict(M[fit_idx])
    lab = np.full(n, -1, dtype=np.int64)
    lab[fit_idx] = sub
    ks = [k for k in sorted(set(sub)) if k >= 0]
    if not ks:
        return np.zeros(n, dtype=np.int64)
    Mf = M[fit_idx]
    cent = np.array([Mf[sub == k].mean(axis=0) for k in ks])
    rad = np.array([1.5 * max(1e-6, np.percentile(np.linalg.norm(Mf[sub == k] - c, axis=1), 95))
                    for k, c in zip(ks, cent)])
    todo = np.where(lab < 0)[0]
    for chunk in np.array_split(todo, max(1, len(todo) // 20000)):
        if len(chunk) == 0:
            continue
        d = np.linalg.norm(M[chunk][:, None, :] - cent[None, :, :], axis=2)
        j = d.argmin(axis=1)
        ok = d[np.arange(len(chunk)), j] <= rad[j]
        lab[chunk[ok]] = np.array(ks)[j[ok]]
    return lab


def pair_index(n, rng, max_pairs=600):
    tot = n * (n - 1) // 2
    if tot <= max_pairs:
        return [(i, j) for i in range(n) for j in range(i + 1, n)]
    pi = rng.integers(0, n, size=(max_pairs, 2))
    return [(int(i), int(j)) for i, j in pi if i != j]


def coact(bits, pairs):
    """Mean pairwise Jaccard similarity of active-hour sets."""
    if not pairs:
        return 0.0
    s = 0.0
    for i, j in pairs:
        a, b = bits[i], bits[j]
        u = (a | b).bit_count()
        s += ((a & b).bit_count() / u) if u else 0.0
    return s / len(pairs)


def nact_bin(x):
    return 0 if x <= 1 else 1 if x <= 3 else 2 if x <= 7 else 3 if x <= 15 else 4 if x <= 31 \
        else 5 if x <= 63 else 6


M64 = (1 << 64) - 1


def pack_bits(xs):
    """Python-int activity bitmaps (<= 192 hours) -> (n, 3) uint64 words."""
    return np.array([[x & M64, (x >> 64) & M64, (x >> 128) & M64] for x in xs],
                    dtype=np.uint64).reshape(-1, 3)


def coact_vec(W, I, J):
    """Mean pairwise Jaccard co-activity of each surrogate group. W: (B, n, 3) packed bitmaps,
    I, J: pair index arrays into the n members. Returns an array of B values."""
    a, b = W[:, I, :], W[:, J, :]
    inter = np.bitwise_count(a & b).sum(-1).astype(np.float64)
    union = np.bitwise_count(a | b).sum(-1).astype(np.float64)
    jac = np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)
    return jac.mean(-1)


def local_pairs(n, rng):
    """Sampled member pairs, re-indexed onto the members that occur in some pair."""
    pairs = pair_index(n, rng)
    need = sorted({i for p in pairs for i in p})
    pos = {k: m for m, k in enumerate(need)}
    I = np.array([pos[i] for i, _ in pairs], dtype=np.int64)
    J = np.array([pos[j] for _, j in pairs], dtype=np.int64)
    return np.array(need, dtype=np.int64), I, J


def zscore_p(t_obs, null):
    sd = null.std()
    if sd < 1e-9:
        return 1.0, 0.0
    z = (t_obs - null.mean()) / sd
    return float(stats.norm.sf(z)), float(z)


# p-values of the synchrony tests. "mc" (default): exact Monte Carlo p-values
# (1 + #{T_b >= T}) / (B + 1), valid by construction when the observed campaign is exchangeable
# with its surrogates under the null; up to three stages: B surrogates, MC_B2 fresh surrogates when
# p <= MC_SCREEN, and MC_B3 fresh ones when then p <= MC_SCREEN2 (fresh draws keep each stage's
# p-value valid; the last stage lifts the 1/(MC_B2+1) floor that would stall BH). "normal": the
# Gaussian tail of the surrogate distribution (optimistic for skewed nulls; kept for comparison).
PVAL = os.environ.get("PVAL", "mc")
MC_B2 = int(os.environ.get("MC_B2", "1000"))
MC_SCREEN = float(os.environ.get("MC_SCREEN", "0.1"))
MC_B3 = int(os.environ.get("MC_B3", "10000"))
MC_SCREEN2 = float(os.environ.get("MC_SCREEN2", "0.01"))
MC_CHUNK = 2000                                        # surrogates per draw call (memory bound)


def mc_p(t_obs, null):
    return float((1 + np.sum(null >= t_obs - 1e-12)) / (len(null) + 1))


def _draw_chunked(draw, n):
    return np.concatenate([draw(min(MC_CHUNK, n - k)) for k in range(0, n, MC_CHUNK)])


def two_stage_p(t_obs, draw, B):
    """draw(n) -> n surrogate statistics. Returns (p, z) with z the surrogate z-score (effect size)."""
    null = draw(B)
    if PVAL != "mc":
        return zscore_p(t_obs, null)
    p, z = mc_p(t_obs, null), zscore_p(t_obs, null)[1]
    for screen, nb in ((MC_SCREEN, MC_B2), (MC_SCREEN2, MC_B3)):
        if p <= screen and nb > len(null):
            null = _draw_chunked(draw, nb)
            p, z = mc_p(t_obs, null), zscore_p(t_obs, null)[1]
    return p, z


# A group whose members converge on the same dark address (most members' modal sampled destination
# is one address) is P2P or misconfiguration residue, not scanning: it is reported, never tested.
CONV_TAU = float(os.environ.get("CONV_TAU", "0.5"))


def convergence(members, dmode, min_n=5):
    if dmode is None:
        return 0.0, False
    d = dmode[np.asarray(members)]
    d = d[d >= 0]
    if len(d) < min_n:
        return 0.0, False
    _, cnt = np.unique(d, return_counts=True)
    share = float(cnt.max() / len(d))
    return share, share >= CONV_TAU


def rot(b, s, H, mask):
    return ((b << s) | (b >> (H - s))) & mask if s else b


def coordination_test(members, bits_all, H, rng, B=200, unit=None, local=None):
    """Synchrony test against circular-shift surrogates (rotation null).

    Null: each member keeps its own activity pattern (duty cycle, burst structure) but is
    rotated by an independent random offset, destroying cross-bot alignment only. Over multi-day
    windows the offsets are whole days, so each source keeps its time-of-day rhythm. Always-on
    bots carry no alignment information and are (correctly) never significant."""
    if unit is None:
        unit = 24 if H >= 48 else 1
    steps = max(2, H // unit)
    mask = (1 << H) - 1
    members = np.asarray(members)
    need, I, J = local if local is not None else local_pairs(len(members), rng)
    if len(I) == 0:
        return 0.0, 1.0, 0.0
    R = np.stack([pack_bits([rot(bits_all[i] & mask, (s * unit) % H, H, mask)
                             for s in range(steps)]) for i in members[need]])  # (n, steps, 3)
    t_obs = float(coact_vec(R[None, :, 0, :], I, J)[0])

    def draw(nb):
        S = rng.integers(0, steps, size=(nb, len(need)))
        return coact_vec(R[np.arange(len(need))[None, :], S], I, J)
    p, z = two_stage_p(t_obs, draw, B)
    return t_obs, p, z


class Pops:
    """Port (and port x country) populations stratified by activity volume, for the population
    null."""

    def __init__(self, df, bits_all, key="portsig"):
        import pandas as pd
        self.pbin = np.array([nact_bin(x) for x in df.nact.to_numpy()], dtype=np.int64)
        self.key = key
        self.ports = df[key].astype(str).to_numpy()
        g = pd.DataFrame({"p": self.ports, "b": self.pbin}).groupby(["p", "b"]).indices
        self.port_bin = {(p, int(b)): np.asarray(v) for (p, b), v in g.items()}
        self.port_all = {p: np.asarray(v) for p, v in
                         pd.Series(self.ports).groupby(self.ports).indices.items()}
        self.bin_all = {int(b): np.where(self.pbin == b)[0] for b in np.unique(self.pbin)}
        self.cc = (df.cc.astype(str).to_numpy() if "cc" in df.columns
                   else np.full(len(df), "", dtype=object))
        g = pd.DataFrame({"p": self.ports, "c": self.cc, "b": self.pbin}).groupby(["p", "c", "b"]).indices
        self.port_cc_bin = {(p, c, int(b)): np.asarray(v) for (p, c, b), v in g.items()}
        g = pd.DataFrame({"p": self.ports, "c": self.cc}).groupby(["p", "c"]).indices
        self.port_cc_all = {(p, c): np.asarray(v) for (p, c), v in g.items()}
        self.packed = pack_bits(bits_all)
        self.is_mem = np.zeros(len(df), dtype=bool)


PEER_RATIO = float(os.environ.get("PEER_RATIO", "1"))   # testable: port has >= this x non-member peers
PEER_KEY = os.environ.get("PEER_KEY", "top1")   # peer population: primary port (or "portsig")
PEER_CC = os.environ.get("PEER_CC", "1") == "1"  # match peers on country too (shared time zone)


def peer_pool(P, port, b, min_pool=40):
    """Peers for activity bin b among non-members probing the same port: the exact bin, else
    the adjacent bins, else any bin of that port. Never other ports, which drift differently.
    Requires P.is_mem to mark the campaign's members."""
    cands = [P.port_bin.get((port, b + d)) for d in (0, -1, 1)]
    for pool in cands:
        if pool is not None and len(pool) - int(P.is_mem[pool].sum()) >= min_pool:
            return pool
    parts = [x for x in cands if x is not None]
    if parts:
        pool = np.concatenate(parts)
        if len(pool) - int(P.is_mem[pool].sum()) >= min_pool:
            return pool
    return P.port_all[port]


def peer_pool_cc(P, port, c, b, min_pool=40):
    """Peers of the same port AND country for activity bin b (exact bin, else adjacent bins, else
    any bin of that country on the port); if the country has too few non-member peers on the
    port, the port-only pool of peer_pool. Requires P.is_mem to mark the campaign's members."""
    cands = [P.port_cc_bin.get((port, c, b + d)) for d in (0, -1, 1)]
    for pool in cands + [P.port_cc_all.get((port, c))]:
        if pool is not None and len(pool) - int(P.is_mem[pool].sum()) >= min_pool:
            return pool
    return peer_pool(P, port, b, min_pool)


def campaign_port(P, members):
    vals, cnt = np.unique(P.ports[np.asarray(members)], return_counts=True)
    return vals[cnt.argmax()]


def testable(P, members, port=None):
    """True if the campaign's port offers at least PEER_RATIO x as many non-member peers."""
    port = campaign_port(P, members) if port is None else port
    pool = P.port_all.get(port)
    if pool is None:
        return False
    P.is_mem[members] = True
    n_peers = len(pool) - int(P.is_mem[pool].sum())
    P.is_mem[members] = False
    return n_peers >= PEER_RATIO * len(members)


def population_test(members, port, P, rng, B=200, local=None, min_pool=40):
    """Synchrony test against activity-matched peers (population null).

    Null: the campaign is no more aligned than other sources probing the same port, matched
    member by member on activity volume (active-hour bin) and, with PEER_CC, on country (shared
    time zone and national events), falling back to port peers for countries with too few.
    Peers share the port-wide modulation of activity across days, so that modulation cannot
    masquerade as coordination. Campaign members are never drawn as their own peers."""
    members = np.asarray(members)
    need, I, J = local if local is not None else local_pairs(len(members), rng)
    if len(I) == 0:
        return 0.0, 1.0, 0.0
    mem_need = members[need]
    t_obs = float(coact_vec(P.packed[mem_need][None], I, J)[0])
    bins = P.pbin[mem_need]
    strata = ([(P.cc[i], int(b)) for i, b in zip(mem_need, bins)] if PEER_CC
              else [("", int(b)) for b in bins])
    keys = sorted(set(strata))
    lab = np.array([keys.index(s) for s in strata])
    pools = {k: (peer_pool_cc_marked(P, members, port, c, b, min_pool) if PEER_CC
                 else peer_pool_marked(P, members, port, b, min_pool))
             for k, (c, b) in enumerate(keys)}

    def draw(nb):
        P.is_mem[members] = True
        try:
            draws = np.empty((nb, len(need)), dtype=np.int64)
            for b, pool in pools.items():
                cols = np.where(lab == b)[0]
                d = pool[rng.integers(0, len(pool), size=(nb, len(cols)))]
                for _ in range(8):                    # redraw any campaign member
                    bad = P.is_mem[d]
                    if not bad.any():
                        break
                    d[bad] = pool[rng.integers(0, len(pool), size=int(bad.sum()))]
                draws[:, cols] = d
        finally:
            P.is_mem[members] = False
        return coact_vec(P.packed[draws], I, J)
    p, z = two_stage_p(t_obs, draw, B)
    return t_obs, p, z


def peer_pool_marked(P, members, port, b, min_pool):
    P.is_mem[members] = True
    try:
        return peer_pool(P, port, b, min_pool)
    finally:
        P.is_mem[members] = False


def peer_pool_cc_marked(P, members, port, c, b, min_pool):
    P.is_mem[members] = True
    try:
        return peer_pool_cc(P, port, c, b, min_pool)
    finally:
        P.is_mem[members] = False


def sync_test(members, port, bits_all, H, P, rng, B=200):
    """Orchestration test: a campaign must be more aligned than its own members' shifted
    patterns (rotation null) AND than activity-matched peers on the same port (population
    null). The intersection-union p-value max(p_rot, p_pop) is valid if either null holds.
    Campaigns whose port offers too few peers are untestable (p = 1): their orchestration is
    undetermined rather than judged against other ports. `port` is ignored; the campaign's
    port is taken from the peer key of P (port signature or primary port)."""
    members = np.asarray(members)
    port = campaign_port(P, members)
    local = local_pairs(len(members), rng)
    t, p_rot, z_rot = coordination_test(members, bits_all, H, rng, B=B, local=local)
    ok = testable(P, members, port)
    if ok:
        _, p_pop, z_pop = population_test(members, port, P, rng, B=B, local=local)
    else:
        p_pop, z_pop = 1.0, 0.0
    return {"coact": t, "p_rot": p_rot, "z_rot": z_rot, "p_pop": p_pop, "z_pop": z_pop,
            "p": max(p_rot, p_pop), "z": min(z_rot, z_pop), "testable": bool(ok)}


def _sub_bits(b, hours):
    out = 0
    for k, h in enumerate(hours):
        out |= ((b >> h) & 1) << k
    return out


def sync_decompose(members, bits_all, H, rng, min_size, B=100, cap=4000):
    """Split-sample synchrony decomposition of one fingerprint block.

    Candidate synchronized sub-groups are found on EVEN hours only (HDBSCAN, Jaccard metric on
    activity bits); each candidate is then tested on the disjoint ODD hours with the
    circular-shift null. Selection and testing use disjoint data, so the test stays calibrated.
    Returns a list of (member_index_array, z, p) for candidates (p not yet FDR-corrected)."""
    from sklearn.cluster import HDBSCAN
    mem = np.asarray(members)
    if len(mem) < 2 * min_size or H < 8:
        return []
    he = list(range(0, H, 2))
    ho = list(range(1, H, 2))
    Ev = np.array([[(bits_all[i] >> h) & 1 for h in he] for i in mem], dtype=bool)
    keep = Ev.sum(1) >= 1
    cand_idx = np.where(keep)[0]
    if len(cand_idx) < 2 * min_size:
        return []
    fit = cand_idx if len(cand_idx) <= cap else rng.choice(cand_idx, cap, replace=False)
    lab = HDBSCAN(min_cluster_size=min_size, min_samples=min(5, min_size), metric="jaccard",
                  copy=True).fit_predict(Ev[fit])
    out = []
    Ho = len(ho)
    mask = (1 << Ho) - 1
    for k in sorted(set(lab)):
        if k < 0:
            continue
        grp = fit[lab == k]
        if len(fit) < len(cand_idx):           # extend to unsampled members by centroid cosine
            cen = Ev[grp].mean(0)
            rest = np.setdiff1d(cand_idx, fit)
            if len(rest):
                num = Ev[rest] @ cen
                den = np.sqrt(Ev[rest].sum(1)) * np.sqrt((cen ** 2).sum()) + 1e-12
                sim = num / den
                own = (Ev[grp] @ cen) / (np.sqrt(Ev[grp].sum(1)) * np.sqrt((cen ** 2).sum()) + 1e-12)
                grp = np.concatenate([grp, rest[sim >= np.percentile(own, 10)]])
        if len(grp) < min_size:
            continue
        ob = [_sub_bits(bits_all[mem[i]], ho) for i in grp]
        if sum(1 for b in ob if b) < min_size:
            continue
        pairs = pair_index(len(ob), rng)
        t_obs = coact(ob, pairs)
        null = np.array([coact([rot(x, int(s), Ho, mask) for x, s in
                                zip(ob, rng.integers(0, Ho, len(ob)))], pairs) for _ in range(B)])
        sd = null.std()
        z = (t_obs - null.mean()) / sd if sd > 1e-9 else 0.0
        out.append((mem[grp], float(z), float(stats.norm.sf(z)) if sd > 1e-9 else 1.0))
    return out


def onset_bursts(members, hfirst, H, min_size, w=3, censor=2):
    """Onset-burst scan statistic for one fingerprint block.

    A commanded campaign recruits many bots that start within a short window. Count new sources
    (first-active hour) per window of w hours and compare with the block's robust baseline onset
    rate (median per-hour onsets): p = P[Poisson(lambda*w) >= k]. Onsets in the first `censor`
    hours are left-censored (may predate the window) and excluded. Returns non-overlapping
    candidate bursts as (member_index_array, z, p), most significant first."""
    mem = np.asarray(members)
    on = hfirst[mem]
    ok = on >= censor
    if ok.sum() < min_size or H - censor < w + 1:
        return []
    counts = np.bincount(on[ok], minlength=H)[:H]
    base = max(float(np.median(counts[censor:])), 0.25)
    lam = base * w
    cands = []
    for h0 in range(censor, H - w + 1):
        k = int(counts[h0:h0 + w].sum())
        if k >= min_size:
            cands.append((float(stats.poisson.sf(k - 1, lam)), h0, k))
    out, used = [], np.zeros(H, dtype=bool)
    for p, h0, k in sorted(cands):
        if used[h0:h0 + w].any():
            continue
        used[h0:h0 + w] = True
        sel = mem[ok & (on >= h0) & (on < h0 + w)]
        out.append((sel, float((k - lam) / math.sqrt(lam)), p))
    return out


def pd_factorize(series):
    import pandas as pd
    return pd.factorize(series)[0]


def pd_numeric(s):
    import pandas as pd
    return pd.to_numeric(s, errors="coerce").fillna(-1).to_numpy().astype(np.int64)


def bh(pvals, q=0.05):
    p = np.asarray(pvals)
    n = len(p)
    if n == 0:
        return np.zeros(0, dtype=bool)
    o = np.argsort(p)
    thr = q * np.arange(1, n + 1) / n
    passed = p[o] <= thr
    k = np.max(np.where(passed)[0]) + 1 if passed.any() else 0
    out = np.zeros(n, dtype=bool)
    out[o[:k]] = True
    return out


# Campaigns are tested independently, each with its own seed, so the result does not depend on
# how many worker processes run the tests (TEST_PROCS; forked workers share the profiles).
TEST_PROCS = int(os.environ.get("TEST_PROCS", "24"))
_TP = {}


def _test_one(k):
    c = _TP["camps"][k]
    rng = np.random.default_rng((_TP["seed"], int(c["id"])))
    return k, sync_test(c["members"], None, _TP["bits"], _TP["H"], _TP["P"], rng, B=_TP["B"])


def run_tests(camps, idx, bits_all, H, P, B, seed):
    import multiprocessing as mp
    _TP.update(camps=camps, bits=bits_all, H=H, P=P, B=B, seed=seed)
    try:
        if (TEST_PROCS > 1 and len(idx) > 50 and "fork" in mp.get_all_start_methods()
                and not mp.current_process().daemon):
            with mp.get_context("fork").Pool(TEST_PROCS) as pool:
                return dict(pool.map(_test_one, idx, chunksize=8))
        return dict(_test_one(k) for k in idx)
    finally:
        _TP.clear()


def enhanced_infer(df, H, min_size=10, drop=(), use_fp=True, use_hdb=True, use_test=True, seed=7,
                   B=200, use_sync=False):
    """Two-level inference. Level 1: machinery campaigns (fingerprint blocks + HDBSCAN-eps on
    behavior, no timing). Level 2: split-sample synchrony decomposition extracts orchestrated
    sub-campaigns hidden inside populations that share their tooling. Returns per-source label
    (-1 = none) and the list of campaign dicts."""
    rng = np.random.default_rng(seed)
    n = len(df)
    labels = np.full(n, -1, dtype=np.int64)
    use_fpc = use_fp and "fingerprint" not in drop
    if "targets" in drop:
        key = df.fpc.astype(str) if use_fpc else df.pm.astype(str)
    else:
        key = df.portsig.astype(str) + ("||" + df.fpc.astype(str) if use_fpc else "")
    M = numeric_matrix(df, H, drop)
    bits_all = [int(b, 16) if b else 0 for b in df.hbits]
    camps = []
    cid = 0
    for k, idx in key.groupby(key).indices.items():
        if len(idx) < min_size:
            continue
        sub = hdbscan_labels(M[idx], min_size, rng) if use_hdb else np.zeros(len(idx), np.int64)
        for c in sorted(set(sub)):
            if c < 0:
                continue
            mem = idx[sub == c]
            if len(mem) < min_size:
                continue
            labels[mem] = cid
            camps.append({"id": cid, "key": k, "members": mem})
            cid += 1
    sync_camps = []
    burst_flag = {}                       # machinery campaign id -> (z, p) of a dominant burst
    if use_sync and use_test:
        hfirst = df.hfirst.to_numpy().astype(int)
        cands = []
        for k, idx in key.groupby(key).indices.items():
            if len(idx) >= min_size:
                for mem, z, p in onset_bursts(idx, hfirst, H, min_size):
                    cands.append((k, mem, z, p))
        if cands:
            sig = bh([c[3] for c in cands])
            taken = np.zeros(n, dtype=bool)
            camp_size = {c["id"]: len(c["members"]) for c in camps}
            for (k, mem, z, p), s in sorted(zip(cands, sig), key=lambda t: t[0][3]):
                if not s:
                    continue
                mem = mem[~taken[mem]]
                if len(mem) < min_size:
                    continue
                host = collections.Counter(labels[mem])
                hid, hc = host.most_common(1)[0]
                if hid >= 0 and hid in burst_flag:
                    continue                  # later wave of an already-flagged campaign
                if hid >= 0 and hc >= 0.5 * camp_size[hid]:
                    burst_flag[hid] = (z, p)  # burst IS the campaign: flag, do not split it
                    continue
                taken[mem] = True             # burst hidden among look-alikes: extract it
                sync_camps.append({"key": k, "members": mem, "z": z, "p": p,
                                   "orchestrated": True, "sync": True})
            if taken.any():
                labels[taken] = -1
                kept = []
                for c in camps:
                    m = c["members"][~taken[c["members"]]]
                    if len(m) >= min_size:
                        c["members"] = m
                        kept.append(c)
                    else:
                        labels[m] = -1
                camps = kept
    dmode = None
    if "dmode" in df.columns:
        dmode = pd_numeric(df["dmode"])
    for c in camps:
        c["conv"], c["residue"] = convergence(c["members"], dmode)
    if use_test and camps:
        P = Pops(df, bits_all, key=PEER_KEY)
        test_idx = [k for k, c in enumerate(camps) if not c["residue"]]   # residue is never tested
        res = run_tests(camps, test_idx, bits_all, H, P, B, seed)
        pv, pv_rot = [], []
        for k in test_idx:
            camps[k].update(res[k])
            pv.append(res[k]["p"]); pv_rot.append(res[k]["p_rot"])
        sig, sig_rot = bh(pv), bh(pv_rot)
        dec = {k: (bool(s), bool(sr)) for k, s, sr in zip(test_idx, sig, sig_rot)}
        for k, c in enumerate(camps):
            if c["residue"]:
                c.update({"coact": 0.0, "p_rot": 1.0, "z_rot": 0.0, "p_pop": 1.0, "z_pop": 0.0,
                          "p": 1.0, "z": 0.0, "testable": False})
            s, sr = dec.get(k, (False, False))
            c["orchestrated"] = s or (c["id"] in burst_flag)
            c["orchestrated_rot"] = sr                # rotation null alone (diagnostic)
            c["burst"] = burst_flag.get(c["id"])
            c["sync"] = False
    for c in sync_camps:
        c["id"] = cid
        c["coact"] = None
        labels[c["members"]] = cid
        cid += 1
        camps.append(c)
    return labels, camps


def signature(df, mem):
    sub = df.iloc[mem]

    def top(col, k=1):
        vc = sub[col].astype(str).value_counts(normalize=True)
        return [(i, round(float(v), 3)) for i, v in vc.head(k).items()]
    return {
        "bots": int(len(mem)), "portsig": top("portsig")[0][0], "proto": top("pm")[0][0],
        "fp": top("fp")[0], "strategy": top("strategy", 3), "disp": top("disp")[0],
        "ipid": top("ipid")[0], "ttl": top("ttl")[0], "sportp": top("sportp")[0],
        "phead": top("phead")[0][0],
        "pph_med": float(sub.pph.median()), "ud_sum": int(sub.ud.sum()),
        "cov_union_ub": float(min(1.0, sub.ud.sum() / DSIZE)), "duty_med": float(sub.duty.median()),
        "hfirst_med": float(sub.hfirst.median()), "hlast_med": float(sub.hlast.median()),
        "n_asn": int(sub.asn.nunique()), "n_cc": int(sub.cc.nunique()),
        "top_cc": top("cc", 3), "top_org": top("org", 3),
        "zmap": float((sub.zmf > 0.5).mean()), "masscan": float((sub.msf > 0.5).mean()),
        "mirai": float((sub.mif > 0.5).mean()), "research": float(sub.research.mean()),
    }


def cmd_infer(args):
    df = load_profiles(sorted(glob.glob(args.profiles)))
    H = args.H
    if args.min_ud:
        df = df[df.ud >= args.min_ud].reset_index(drop=True)
    print("profiles:", len(df), flush=True)
    labels, camps = enhanced_infer(df, H, min_size=args.min_size, B=args.B)
    for c in camps:
        c["sig"] = signature(df, c["members"])
        c["n"] = int(len(c["members"]))
        c["members"] = None
    camps.sort(key=lambda c: -c["n"])
    out = {"n_profiles": int(len(df)), "n_campaigns": len(camps),
           "n_orchestrated": int(sum(c.get("orchestrated", False) for c in camps)),
           "covered_sources": int((labels >= 0).sum()), "campaigns": camps}
    with open(args.out, "w") as fh:
        json.dump(out, fh, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    np.save(args.out.replace(".json", "_labels.npy"), labels)
    print(json.dumps({k: v for k, v in out.items() if k != "campaigns"}))


def negative_control(df, H, camps, rng, B=100, reps=1, keep_p=False):
    """Empirical false-positive rate on REAL data. For every testable campaign, a pseudo-campaign
    of the same size is drawn at random from its port population EXCLUDING its own members, so it
    mixes unrelated sources of the same service, and goes through the identical test + BH.
    keep_p=True also returns every pseudo-campaign's p-values (two-null and rotation alone) and
    members (row indices)."""
    bits_all = [int(b, 16) if b else 0 for b in df.hbits]
    P = Pops(df, bits_all, key=PEER_KEY)
    dmode = pd_numeric(df["dmode"]) if "dmode" in df.columns else None
    pv, pv_rot, mems = [], [], []
    n_res = 0
    for _ in range(reps):
        for c in camps:
            if c.get("residue"):
                continue                          # residue is never tested
            mem = np.asarray(c["members"])
            port = campaign_port(P, mem)
            if not testable(P, mem, port):
                continue
            pool = np.setdiff1d(P.port_all[port], mem, assume_unique=True)
            pseudo = rng.choice(pool, len(mem), replace=False)
            if convergence(pseudo, dmode)[1]:
                n_res += 1
                continue                          # a residue pseudo-group would not be tested
            r = sync_test(pseudo, None, bits_all, H, P, rng, B=B)
            pv.append(r["p"]); pv_rot.append(r["p_rot"]); mems.append(pseudo)
    sig, sig_rot = bh(pv), bh(pv_rot)
    nan = float("nan")
    res = {"pseudo_campaigns": len(pv), "pseudo_residue_skipped": n_res, "flagged": int(sig.sum()),
           "fpr": float(sig.mean()) if len(pv) else nan,
           "raw_p_lt_0.05": float(np.mean(np.array(pv) < 0.05)) if pv else nan,
           "flagged_rot": int(sig_rot.sum()),
           "fpr_rot": float(sig_rot.mean()) if len(pv) else nan,
           "raw_rot_p_lt_0.05": float(np.mean(np.array(pv_rot) < 0.05)) if pv else nan}
    if keep_p:
        res["p"], res["p_rot"] = [float(x) for x in pv], [float(x) for x in pv_rot]
        res["members"] = mems
    return res


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    f = sp.add_parser("features")
    f.add_argument("--red", required=True)
    f.add_argument("--out", required=True)
    f.add_argument("--w0", default="2026-09-24")
    f.add_argument("--d0", type=int, default=0)
    f.add_argument("--shard", type=int, default=0)
    f.add_argument("--nshards", type=int, default=1)
    f.add_argument("--hours", default="")
    f.add_argument("--min-ud", type=int, default=2)
    f.add_argument("--max-files", type=int, default=0)
    f.add_argument("--glob", default="*", help="reduced-file stem pattern, e.g. 2026-09-25.*")
    i = sp.add_parser("infer")
    i.add_argument("--profiles", required=True)
    i.add_argument("--out", required=True)
    i.add_argument("--H", type=int, default=168)
    i.add_argument("--min-size", type=int, default=10)
    i.add_argument("--min-ud", type=int, default=0)
    i.add_argument("--B", type=int, default=200)
    a = ap.parse_args()
    {"features": cmd_features, "infer": cmd_infer}[a.cmd](a)


if __name__ == "__main__":
    main()
