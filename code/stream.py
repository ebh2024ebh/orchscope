#!/usr/bin/env python3
"""Real-time engine (OrchScope): consume hourly reduced batches as a stream.

Deployment model: a weekly landmark window. The engine is warm-started with the window's earlier
days, then consumes one hourly batch at a time. Per-source state is updated incrementally;
profiles are recomputed only for sources touched in the new hour; campaign inference and the
synchrony test run after every batch. The test sees the trailing whole days of the window, so
day shifts keep each source's time-of-day rhythm and never rotate activity into hours that have
not been observed yet. Records per-stage latency and campaign-identity stability (member-set
Jaccard between consecutive updates)."""
import argparse, datetime, glob, json, os, time

import pandas as pd

import engine as E


def match(prev, cur, thr=0.5):
    """Fraction of current campaigns that continue a previous campaign (member Jaccard >= thr)."""
    if not prev or not cur:
        return float("nan")
    inv = {}
    for k, s in prev.items():
        for x in s:
            inv[x] = k
    cont = 0
    for s in cur.values():
        cnt = {}
        for x in s:
            k = inv.get(x)
            if k is not None:
                cnt[k] = cnt.get(k, 0) + 1
        if cnt:
            k, c = max(cnt.items(), key=lambda kv: kv[1])
            if c / len(s | prev[k]) >= thr:
                cont += 1
    return cont / len(cur)


def infer(cache, n_hours, B, seed):
    """Inference over the cached profiles; the synchrony test sees the trailing whole days."""
    df = pd.DataFrame(list(cache.values()), columns=E.PROF_COLS)
    days = max(1, n_hours // 24)
    off = n_hours - 24 * days
    if off:
        df["hbits"] = ["%x" % (int(b, 16) >> off) if b else "0" for b in df.hbits]
    labels, camps = E.enhanced_infer(df, 24 * days, min_size=10, B=B, seed=seed)
    return df, labels, camps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--red", default="red")
    ap.add_argument("--w0", default="2026-09-24", help="window start (landmark)")
    ap.add_argument("--day", default="2026-09-30", help="day replayed hour by hour")
    ap.add_argument("--out", default="stream_0930.json")
    ap.add_argument("--d0", type=int, default=591396864)
    ap.add_argument("--B", type=int, default=200)
    ap.add_argument("--hours", type=int, default=24)
    a = ap.parse_args()
    w0 = datetime.datetime.strptime(a.w0, "%Y-%m-%d")
    day = datetime.datetime.strptime(a.day, "%Y-%m-%d")
    pre = []
    d = w0
    while d < day:
        pre += sorted(glob.glob(os.path.join(a.red, d.strftime("%Y-%m-%d") + ".*.tsv.zst")))
        d += datetime.timedelta(days=1)
    files = sorted(glob.glob(os.path.join(a.red, a.day + ".*.tsv.zst")))[:a.hours]
    log = []
    # warm start: the window's earlier days (not part of the per-hour cost)
    t0 = time.time()
    acc, _ = E.accumulate_rows(E.iter_rows(pre), w0)
    t_ing = time.time() - t0
    t1 = time.time()
    cache = {s: E.profile(s, x, a.d0, 168) for s, x in acc.items()
             if x.ud >= 2 or len(x.sam) >= 2}
    t_prof = time.time() - t1
    h0 = E.batch_hour(os.path.basename(files[0]).split(".tsv")[0], w0)
    t2 = time.time()
    df, labels, camps = infer(cache, h0, a.B, seed=h0 - 1)
    t_inf = time.time() - t2
    prev = {c["id"]: set(df.src.iloc[c["members"]]) for c in camps}
    rec = {"hour": h0 - 1, "warm": True, "files": len(pre), "sources": len(acc),
           "profiles": int(len(df)), "campaigns": len(camps),
           "orchestrated": int(sum(c.get("orchestrated", False) for c in camps)),
           "residue": int(sum(c.get("residue", False) for c in camps)),
           "t_ingest": t_ing, "t_profile": t_prof, "t_infer": t_inf}
    log.append(rec)
    print(json.dumps(rec), flush=True)
    for path in files:
        label = os.path.basename(path).split(".tsv")[0]
        h = E.batch_hour(label, w0)
        t0 = time.time()
        rows = list(E.iter_rows([path]))
        t_read = time.time() - t0
        touched = {r[1] for r in rows if len(r) >= 25}
        t1 = time.time()
        E.accumulate_rows(iter(rows), w0, acc=acc)
        t_ingest = time.time() - t1
        t2 = time.time()
        for s in touched:
            x = acc.get(s)
            if x is not None and (x.ud >= 2 or len(x.sam) >= 2):
                cache[s] = E.profile(s, x, a.d0, 168)
        t_prof = time.time() - t2
        t3 = time.time()
        df, labels, camps = infer(cache, h + 1, a.B, seed=h)
        t_infer = time.time() - t3
        cur = {c["id"]: set(df.src.iloc[c["members"]]) for c in camps}
        rec = {"hour": h, "file": label, "rows": len(rows), "touched": len(touched),
               "sources": len(acc), "profiles": int(len(df)), "campaigns": len(camps),
               "orchestrated": int(sum(c.get("orchestrated", False) for c in camps)),
           "residue": int(sum(c.get("residue", False) for c in camps)),
               "t_read": t_read, "t_ingest": t_ingest, "t_profile": t_prof, "t_infer": t_infer,
               "t_total": time.time() - t0, "continuity": match(prev, cur)}
        log.append(rec)
        prev = cur
        print(json.dumps(rec), flush=True)
        with open(a.out, "w") as fh:
            json.dump(log, fh)


if __name__ == "__main__":
    main()
