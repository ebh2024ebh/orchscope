#!/usr/bin/env python3
"""DarkVec baseline (Gioacchini et al., CoNEXT'21), re-implemented on telescope flow records.

Following the published design:
  * services: the top-10 ports (by sender-hour records) are one service each; all other ports
    form a single 'other' service;
  * sentences: for every service and every 1-hour window, the time-ordered sequence of sender
    addresses; a flow record contributes min(5, packets) occurrences spread over its
    first..last timestamps (records carry no per-packet times);
  * senders with fewer than 10 packets in the window are discarded;
  * Word2Vec skip-gram with negative sampling: V=50 dimensions, context c=25 (dynamic window),
    5 negatives, 20 epochs, frequent-word subsampling 1e-3, learning rate 0.025 -> 1e-4;
  * clusters: Louvain communities on the k'=3 nearest-neighbour graph with cosine weights.

Input : TSV rows "hour  src  topports(p:c,...)  t0  t1  packets"
Output: TSV "src  label" (label -1: filtered out or in a community smaller than --min-size).
Runs with any Python that has numpy, numba and networkx."""
import argparse, collections, time

import networkx as nx
import numba as nb
import numpy as np


@nb.njit(cache=True)
def _train(flat, ptr, w_in, w_out, table, keep, window, negative, epochs, alpha0, seed):
    np.random.seed(seed)
    dim = w_in.shape[1]
    total = flat.shape[0] * epochs + 1.0
    done = 0.0
    grad = np.zeros(dim, dtype=np.float32)
    buf = np.empty(flat.shape[0], dtype=np.int64)
    for _ in range(epochs):
        for s in range(ptr.shape[0] - 1):
            n = 0
            for i in range(ptr[s], ptr[s + 1]):
                w = flat[i]
                if keep[w] >= 1.0 or np.random.random() < keep[w]:
                    buf[n] = w
                    n += 1
            for i in range(n):
                done += 1.0
                alpha = alpha0 * max(1e-4, 1.0 - done / total)
                center = buf[i]
                b = np.random.randint(1, window + 1)
                for j in range(max(0, i - b), min(n, i + b + 1)):
                    if j == i:
                        continue
                    ctx = buf[j]
                    for k in range(dim):
                        grad[k] = 0.0
                    for d in range(negative + 1):
                        if d == 0:
                            target = center
                            label = 1.0
                        else:
                            target = table[np.random.randint(0, table.shape[0])]
                            if target == center:
                                continue
                            label = 0.0
                        f = 0.0
                        for k in range(dim):
                            f += w_in[ctx, k] * w_out[target, k]
                        if f > 6.0:
                            g = (label - 1.0) * alpha
                        elif f < -6.0:
                            g = label * alpha
                        else:
                            g = (label - 1.0 / (1.0 + np.exp(-f))) * alpha
                        for k in range(dim):
                            grad[k] += g * w_out[target, k]
                            w_out[target, k] += g * w_in[ctx, k]
                    for k in range(dim):
                        w_in[ctx, k] += grad[k]


@nb.njit(parallel=True, cache=True)
def _train_par(flat, ptr, w_in, w_out, table, keep, window, negative, epochs, alpha0):
    """Hogwild variant (sentences in parallel) for corpora of a full telescope day."""
    dim = w_in.shape[1]
    n_tok = flat.shape[0]
    total = n_tok * epochs + 1.0
    n_sent = ptr.shape[0] - 1
    for ep in range(epochs):
        for s in nb.prange(n_sent):
            a, b_ = ptr[s], ptr[s + 1]
            buf = np.empty(b_ - a, dtype=np.int64)
            grad = np.zeros(dim, dtype=np.float32)
            n = 0
            for i in range(a, b_):
                w = flat[i]
                if keep[w] >= 1.0 or np.random.random() < keep[w]:
                    buf[n] = w
                    n += 1
            for i in range(n):
                prog = (ep * n_tok + a + i * (b_ - a) / max(n, 1)) / total
                alpha = alpha0 * max(1e-4, 1.0 - prog)
                center = buf[i]
                bw = np.random.randint(1, window + 1)
                for j in range(max(0, i - bw), min(n, i + bw + 1)):
                    if j == i:
                        continue
                    ctx = buf[j]
                    for k in range(dim):
                        grad[k] = 0.0
                    for d in range(negative + 1):
                        if d == 0:
                            target = center
                            label = 1.0
                        else:
                            target = table[np.random.randint(0, table.shape[0])]
                            if target == center:
                                continue
                            label = 0.0
                        f = 0.0
                        for k in range(dim):
                            f += w_in[ctx, k] * w_out[target, k]
                        if f > 6.0:
                            g = (label - 1.0) * alpha
                        elif f < -6.0:
                            g = label * alpha
                        else:
                            g = (label - 1.0 / (1.0 + np.exp(-f))) * alpha
                        for k in range(dim):
                            grad[k] += g * w_out[target, k]
                            w_out[target, k] += g * w_in[ctx, k]
                    for k in range(dim):
                        w_in[ctx, k] += grad[k]


def read_rows(path):
    rows = []
    with open(path) as fh:
        for line in fh:
            f = line.rstrip("\n").split("\t")
            if len(f) < 6:
                continue
            ports = {}
            for kv in f[2].split(","):
                if kv:
                    p, c = kv.split(":")
                    ports[int(p)] = int(c)
            rows.append((int(f[0]), f[1], ports, int(f[3] or 0), int(f[4] or 0), int(f[5])))
    return rows


def corpus(rows, n_services=10, min_packets=10, cap=5, max_sent=0):
    pk = collections.Counter()
    for _, src, _, _, _, p in rows:
        pk[src] += p
    keep_src = {s for s, p in pk.items() if p >= min_packets}
    port_pop = collections.Counter()
    for _, src, ports, _, _, _ in rows:
        if src in keep_src:
            port_pop.update(ports.keys())
    services = {p: i for i, (p, _) in enumerate(port_pop.most_common(n_services))}
    other = len(services)
    vocab = {s: i for i, s in enumerate(sorted(keep_src))}
    events = collections.defaultdict(list)        # (service, hour) -> [(t, word)]
    for h, src, ports, t0, t1, p in rows:
        w = vocab.get(src)
        if w is None:
            continue
        tot = sum(ports.values()) or 1
        for port, c in ports.items():
            n = int(min(cap, max(1, round(p * c / tot))))
            lo, hi = (t0, t1) if t1 >= t0 > 0 else (h * 3600, h * 3600 + 3599)
            for i in range(n):
                events[(services.get(port, other), h)].append((lo + (hi - lo) * (i + 0.5) / n, w))
    flat, ptr = [], [0]
    for key in sorted(events):
        seq = [w for _, w in sorted(events[key])]
        step = max_sent if max_sent > 0 else len(seq) or 1
        for a in range(0, len(seq), step):          # long sentences: consecutive chunks
            flat.extend(seq[a:a + step])
            ptr.append(len(flat))
    return vocab, np.array(flat, dtype=np.int64), np.array(ptr, dtype=np.int64)


def embed(flat, ptr, n_words, dim=50, window=25, negative=5, epochs=20, sample=1e-3, seed=7,
          threads=1):
    rng = np.random.default_rng(seed)
    w_in = ((rng.random((n_words, dim)) - 0.5) / dim).astype(np.float32)
    w_out = np.zeros((n_words, dim), dtype=np.float32)
    cnt = np.bincount(flat, minlength=n_words).astype(np.float64)
    freq = cnt / max(cnt.sum(), 1.0)
    keep = np.where(freq > 0, (np.sqrt(freq / sample) + 1) * sample / np.maximum(freq, 1e-12), 1.0)
    p = cnt ** 0.75
    p /= p.sum()
    table = rng.choice(n_words, size=min(10_000_000, max(1_000_000, 50 * n_words)), p=p)
    if threads > 1:
        nb.set_num_threads(threads)
        _train_par(flat, ptr, w_in, w_out, table.astype(np.int64), keep.astype(np.float64),
                   window, negative, epochs, 0.025)
    else:
        _train(flat, ptr, w_in, w_out, table.astype(np.int64), keep.astype(np.float64), window,
               negative, epochs, 0.025, seed)
    return w_in


def knn_graph(emb, k=3, chunk=1024):
    x = emb / np.maximum(np.linalg.norm(emb, axis=1, keepdims=True), 1e-12)
    g = nx.Graph()
    g.add_nodes_from(range(len(x)))
    for a in range(0, len(x), chunk):
        s = x[a:a + chunk] @ x.T
        for r in range(s.shape[0]):
            s[r, a + r] = -np.inf
        idx = np.argpartition(-s, k, axis=1)[:, :k]
        for r in range(s.shape[0]):
            for j in idx[r]:
                w = float(s[r, j])
                if w > 0:
                    u, v = a + r, int(j)
                    if not g.has_edge(u, v) or g[u][v]["weight"] < w:
                        g.add_edge(u, v, weight=w)
    return g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("rows")
    ap.add_argument("out")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--min-size", type=int, default=10)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--max-sent", type=int, default=0, help="split longer sentences (0: never)")
    a = ap.parse_args()
    t0 = time.time()
    rows = read_rows(a.rows)
    vocab, flat, ptr = corpus(rows, max_sent=a.max_sent)
    t1 = time.time()
    emb = embed(flat, ptr, len(vocab), epochs=a.epochs, seed=a.seed, threads=a.threads)
    t2 = time.time()
    g = knn_graph(emb)
    comms = nx.community.louvain_communities(g, weight="weight", resolution=1.0, seed=a.seed)
    t3 = time.time()
    inv = {i: s for s, i in vocab.items()}
    lab = {}
    cid = 0
    for c in sorted(comms, key=len, reverse=True):
        if len(c) < a.min_size:
            continue
        for i in c:
            lab[inv[i]] = cid
        cid += 1
    with open(a.out, "w") as fh:
        for s in vocab:
            fh.write("%s\t%d\n" % (s, lab.get(s, -1)))
    print("rows %d words %d tokens %d sentences %d | clusters %d | corpus %.0fs embed %.0fs "
          "cluster %.0fs" % (len(rows), len(vocab), len(flat), len(ptr) - 1, cid, t1 - t0,
                             t2 - t1, t3 - t2), flush=True)


if __name__ == "__main__":
    main()
