#!/usr/bin/env python3
"""Day-to-day campaign lineage (OrchScope): link campaigns inferred independently on consecutive
days by member overlap; report track lengths and identity stability."""
import collections, glob, json, sys

import pandas as pd


def main(out, pattern="camp_day_*_assign.tsv", thr=0.2):
    days = sorted(glob.glob(pattern))
    members = []
    for p in days:
        df = pd.read_csv(p, sep="\t", dtype={"src": str})
        df = df[df.camp >= 0]
        members.append({c: set(g.src) for c, g in df.groupby("camp")})
    links = []
    for d in range(len(members) - 1):
        a, b = members[d], members[d + 1]
        inv = collections.defaultdict(set)
        for c, s in b.items():
            for x in s:
                inv[x].add(c)
        for c, s in a.items():
            cand = collections.Counter(c2 for x in s for c2 in inv.get(x, ()))
            for c2, k in cand.most_common(3):
                j = k / len(s | b[c2])
                if j >= thr:
                    links.append((d, c, c2, j))
    # chain links into tracks (greedy best successor)
    nxt = {}
    for d, c, c2, j in sorted(links, key=lambda t: -t[3]):
        if (d, c) not in nxt and (d + 1, c2) not in set(nxt.values()):
            nxt[(d, c)] = (d + 1, c2)
    has_pred = set(nxt.values())
    tracks = []
    for d, m in enumerate(members):
        for c in m:
            if (d, c) in has_pred:
                continue
            t, cur = [(d, c)], (d, c)
            while cur in nxt:
                cur = nxt[cur]
                t.append(cur)
            tracks.append(t)
    lens = collections.Counter(len(t) for t in tracks)
    persist = sum(1 for t in tracks if len(t) >= 2)
    res = {"days": len(days), "campaigns_per_day": [len(m) for m in members],
           "links": len(links), "tracks": len(tracks), "track_len_hist": dict(sorted(lens.items())),
           "multi_day_tracks": persist,
           "multi_day_fraction": persist / len(tracks) if tracks else 0.0,
           "median_link_jaccard": float(pd.Series([l[3] for l in links]).median()) if links else 0}
    with open(out, "w") as fh:
        json.dump(res, fh)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "lineage.json")
