#!/usr/bin/env python3
"""Per-batch reducer for ORION hourly flow records (runs on the data host; stdlib only).

Input  (stdin): ORION per-flow JSON lines for one hourly file.
Output (stdout): one TSV row per *probing* source in the batch, carrying per-source
aggregates plus up to KEEP header-parsed samples (evenly spaced in time) for the
source profiles. Backscatter/misconfiguration classes are dropped here.

Columns: batch src pmask nrec nports topports pk bytes ud ud24 udmax t0 t1 dur
         zm ms mi asn cc org rdns stride_pos stride_neg stride_big samples
Sample : t,dport,proto,dst,sport,ttl,ipid,df,win,optsig,seqrel,plen,phead
"""
import sys, json, base64, random, datetime

PROBE = {11: 1, 16: 2, 0: 4}   # ORION Traffic code -> bit: TCP SYN, UDP, ICMP echo request
RES = 32                       # per-source sample reservoir
KEEP = 8                       # samples emitted per source
random.seed(13)


def ts(s):
    if not s:
        return 0
    if s[-1] == "Z":
        s = s[:-1] + "+00:00"
    try:
        return int(datetime.datetime.fromisoformat(s).timestamp())
    except ValueError:
        i = s.find(".")
        if i > 0:
            j = i + 1
            while j < len(s) and s[j].isdigit():
                j += 1
            try:
                return int(datetime.datetime.fromisoformat(s[:i] + s[j:]).timestamp())
            except ValueError:
                return 0
        return 0


def fast_dst(b64):
    # Ethernet(14)+IPv4: dst address at packet offset 30, a multiple of 3 -> base64 char 40.
    try:
        return int.from_bytes(base64.b64decode(b64[40:48])[:4], "big")
    except Exception:
        return -1


def opts_sig(o):
    out, i, n = [], 0, len(o)
    while i < n:
        k = o[i]
        if k == 0:
            out.append("E")
            break
        if k == 1:
            out.append("N")
            i += 1
            continue
        if i + 1 >= n:
            break
        ln = o[i + 1]
        if ln < 2 or i + ln > n:
            break
        if k == 2 and ln == 4:
            out.append("M%d" % ((o[i + 2] << 8) | o[i + 3]))
        elif k == 3 and ln == 3:
            out.append("W%d" % o[i + 2])
        elif k == 4:
            out.append("S")
        elif k == 8:
            out.append("T")
        else:
            out.append("K%d" % k)
        i += ln
    return ".".join(out)


def parse_sample(b64):
    try:
        p = base64.b64decode(b64)
    except Exception:
        return None
    if len(p) < 34:
        return None
    et, off = (p[12] << 8) | p[13], 14
    if et == 0x8100 and len(p) >= 18:
        et, off = (p[16] << 8) | p[17], 18
    if et != 0x0800 or len(p) < off + 20:
        return None
    v = p[off]
    if v >> 4 != 4:
        return None
    ihl = (v & 15) * 4
    if ihl < 20 or len(p) < off + ihl:
        return None
    ttl, proto = p[off + 8], p[off + 9]
    ipid = (p[off + 4] << 8) | p[off + 5]
    df = (p[off + 6] >> 6) & 1
    dst = int.from_bytes(p[off + 16:off + 20], "big")
    l4 = off + ihl
    sport = win = plen = optsig = seqrel = phead = ""
    if proto == 6 and len(p) >= l4 + 20:
        sport = (p[l4] << 8) | p[l4 + 1]
        seq = int.from_bytes(p[l4 + 4:l4 + 8], "big")
        doff = (p[l4 + 12] >> 4) * 4
        win = (p[l4 + 14] << 8) | p[l4 + 15]
        seqrel = "D" if seq == dst else ("Z" if seq == 0 else "")
        if doff > 20 and len(p) >= l4 + doff:
            optsig = opts_sig(p[l4 + 20:l4 + doff])
    elif proto == 17 and len(p) >= l4 + 8:
        sport = (p[l4] << 8) | p[l4 + 1]
        pl = p[l4 + 8:]
        plen, phead = len(pl), pl[:8].hex()
    elif proto == 1 and len(p) >= l4 + 1:
        win = p[l4]  # ICMP type
    return proto, dst, sport, ttl, ipid, df, win, optsig, seqrel, plen, phead


def clean(s, n):
    return (s or "").replace("\t", " ").replace("\n", " ").replace(";", ",")[:n]


class Src:
    __slots__ = ("pm", "nrec", "ports", "pk", "by", "ud", "ud24", "udmax", "t0", "t1", "dur",
                 "zm", "ms", "mi", "asn", "cc", "org", "rdns", "sp", "sn", "sb", "res", "seen")

    def __init__(self, r):
        self.pm = self.nrec = 0
        self.ports = {}
        self.pk = self.by = self.ud = self.ud24 = self.udmax = 0
        self.t0, self.t1, self.dur = 1 << 62, 0, 0
        self.zm = self.ms = self.mi = 0
        self.asn = r.get("ASN") or 0
        self.cc = clean(r.get("Country"), 4)
        self.org = clean(r.get("Org"), 48)
        rd = r.get("RDNS") or []
        self.rdns = clean(rd[0] if rd else "", 64)
        self.sp = self.sn = self.sb = 0
        self.res, self.seen = [], 0


def main():
    batch = sys.argv[1] if len(sys.argv) > 1 else "-"
    for row in reduce_lines(sys.stdin, batch):
        sys.stdout.write(row)


def reduce_lines(lines, batch):
    """Reduce an iterable of ORION JSON lines for one batch; yields TSV rows (with newline)."""
    src = {}
    loads = json.loads
    for line in lines:
        try:
            r = loads(line)
        except Exception:
            continue
        bit = PROBE.get(r.get("Traffic"))
        if bit is None:
            continue
        ip = r.get("SourceIP")
        st = src.get(ip)
        if st is None:
            st = src[ip] = Src(r)
        st.pm |= bit
        st.nrec += 1
        port = r.get("Port") or 0
        st.ports[port] = st.ports.get(port, 0) + 1
        st.pk += r.get("Packets") or 0
        st.by += r.get("Bytes") or 0
        u = r.get("UniqueDests") or 0
        st.ud += u
        st.ud24 += r.get("UniqueDest24s") or 0
        if u > st.udmax:
            st.udmax = u
        a, b = ts(r.get("First")), ts(r.get("Last"))
        if a and a < st.t0:
            st.t0 = a
        if b > st.t1:
            st.t1 = b
        if a and b >= a:
            st.dur += b - a
        if r.get("Zmap"):
            st.zm += 1
        if r.get("Masscan"):
            st.ms += 1
        if r.get("Mirai"):
            st.mi += 1
        sm = r.get("Samples") or []
        if len(sm) >= 2:
            ds = [fast_dst(x) for x in sm]
            for k in range(1, len(ds)):
                if ds[k] < 0 or ds[k - 1] < 0:
                    continue
                d = ds[k] - ds[k - 1]
                if 1 <= d <= 1024:
                    st.sp += 1
                elif -1024 <= d <= -1:
                    st.sn += 1
                else:
                    st.sb += 1
        for j, x in enumerate(sm):
            st.seen += 1
            item = (a + j * 1e-3, port, x)
            if len(st.res) < RES:
                st.res.append(item)
            else:
                m = random.randrange(st.seen)
                if m < RES:
                    st.res[m] = item
    for ip, st in src.items():
        res = sorted(st.res, key=lambda t: t[0])
        if len(res) > KEEP:
            step = len(res) / KEEP
            res = [res[int(i * step)] for i in range(KEEP)]
        sams = []
        for t, port, x in res:
            ps = parse_sample(x)
            if ps is None:
                continue
            proto, dst, sport, ttl, ipid, df, win, optsig, seqrel, plen, phead = ps
            sams.append("%d,%d,%d,%d,%s,%d,%d,%d,%s,%s,%s,%s,%s" % (
                int(t), port, proto, dst, sport, ttl, ipid, df, win, optsig, seqrel, plen, phead))
        top = sorted(st.ports.items(), key=lambda kv: -kv[1])[:5]
        yield ("\t".join((
            batch, ip, str(st.pm), str(st.nrec), str(len(st.ports)),
            ",".join("%d:%d" % kv for kv in top), str(st.pk), str(st.by), str(st.ud),
            str(st.ud24), str(st.udmax), str(st.t0 if st.t0 < (1 << 62) else 0), str(st.t1),
            str(st.dur), str(st.zm), str(st.ms), str(st.mi), str(st.asn), st.cc, st.org,
            st.rdns, str(st.sp), str(st.sn), str(st.sb), ";".join(sams))) + "\n")


if __name__ == "__main__":
    main()
