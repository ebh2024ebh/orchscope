#!/usr/bin/env python3
"""Aggregate timeline of the 2011 sipscan (Paper 3, auxiliary evaluation): per UTC hour from Jan 31
to Feb 14, the distinct bots and packets at the telescope /8, and the mean distinct bots per /13
slice (an ORION-sized view). Aggregates only.     sipscan/sipscan.npz -> sipscan_timeline.json"""
import datetime, json, os, sys

import numpy as np

DAY0 = datetime.datetime(2011, 1, 31, tzinfo=datetime.timezone.utc)


def main(npz=os.path.join("sipscan", "sipscan.npz"), out="sipscan_timeline.json"):
    Z = np.load(npz)
    t, src, dst = Z["t"], Z["src"].astype(np.int64), Z["dst24"]
    h = ((t - DAY0.timestamp()) // 3600).astype(np.int64)
    nh = int(h.max()) + 1
    bots = np.bincount(np.unique(h << 32 | src) >> 32, minlength=nh)
    pk = np.bincount(h, minlength=nh)
    sl = (dst >> 19).astype(np.int64)
    k = np.unique((h << 40) | (sl << 32) | src) >> 32           # distinct (hour, slice, bot)
    per = np.bincount(k >> 8, minlength=nh) / 32.0               # mean over the 32 slices
    res = {"day0": DAY0.isoformat(), "hours": nh, "bots": bots.tolist(), "packets": pk.tolist(),
           "bots_per_slice_mean": [round(float(x), 1) for x in per],
           "total_packets": int(len(t)), "total_bots": int(len(np.unique(src))),
           "first": float(t.min()), "last": float(t.max()), "countries": int(len(Z["cc_names"]))}
    json.dump(res, open(out, "w"))
    print(json.dumps({k: v for k, v in res.items() if not isinstance(v, list)}))


if __name__ == "__main__":
    main(*sys.argv[1:])
