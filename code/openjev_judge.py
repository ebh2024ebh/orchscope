#!/usr/bin/env python3
"""openjev decision layer (OrchScope): typed, calibrated judgments over inferred campaigns.

Each campaign signature is serialized to text (fingerprint + behavior, never ORION tool flags or
simulator labels), and openjev answers typed questions with calibrated probabilities via its
/v1/systemone shim. A multimodal question reads the scanning strategy from a rendered
destination-vs-time plot.
"""
import base64, io, json, time, urllib.request

URL = "http://127.0.0.1:3000/v1/systemone"

TOOLS = ["Mirai-family IoT malware", "ZMap", "Masscan", "Nmap",
         "Operating-system TCP stack (connect scan or script)", "Other or custom scanner"]
SERVICES = ["Telnet on IoT devices", "SSH", "Web (HTTP/HTTPS)", "Windows SMB/RDP", "VoIP (SIP)",
            "Databases", "Many services (port sweep)", "Other"]
CLASSES = ["IoT botnet propagation", "Internet-wide research or measurement scanning",
           "Vulnerability or exploit sweep", "Service discovery for later targeting",
           "Misconfiguration or backscatter, not scanning"]
STRATS = ["Sequential, increasing addresses", "Sequential, decreasing addresses",
          "Uniform random permutation", "Non-uniform permutation concentrated in a few address blocks"]
STRAT_KEYS = ["seq_fwd", "seq_rev", "uperm", "perm"]


def choice(instr, opts):
    return {"type": "choice", "instructions": instr, "criteria": {o: None for o in opts}}


QUESTIONS = {
    "tool": choice("Which scanning software most likely generated this traffic?", TOOLS),
    "service": choice("Which service are these sources most likely searching for?", SERVICES),
    "cls": choice("What best describes this group of sources?", CLASSES),
    "coordinated": {"type": "noul", "instructions":
                    "Do these sources behave as one coordinated campaign rather than unrelated "
                    "scanners?"},
    "severity": {"type": "score", "instructions":
                 "How urgently should a network defender investigate this activity?",
                 "criteria": ["can wait", "this week", "today", "right now"]},
}
STRAT_Q = {"strategy": choice(
    "The image plots the destination address (vertical axis, within the monitored address block) "
    "against time (horizontal axis) for probes sent by a group of scanning sources. Which "
    "scanning strategy does it show?", STRATS)}

IPID = {"ip54321": "always 54321", "ip0": "always zero", "ipconst": "constant", "ipinc":
        "incrementing", "iprand": "random", "ip1": "unknown (one sample)", "na": "unknown"}
SPORT = {"spfix": "fixed", "spnarrow": "confined to a narrow fixed range", "spinc": "incrementing",
         "sprand": "random", "sp1": "unknown (one sample)"}
STRAT_TXT = {"seq_fwd": "sequential, increasing addresses", "seq_rev":
             "sequential, decreasing addresses", "uperm": "uniform random permutation", "perm":
             "non-uniform permutation", "undet": "undetermined (too few samples)"}


def opts_words(o):
    if not o or o in ("-", "nan"):
        return "none"
    w = {"N": "NOP", "S": "SACK-permitted", "T": "timestamps", "E": "end-of-list"}
    out = []
    for t in o.split("."):
        if t.startswith("M"):
            out.append("MSS=" + t[1:])
        elif t.startswith("W"):
            out.append("window-scale=" + t[1:])
        else:
            out.append(w.get(t, t))
    return ", ".join(out)


PORT_NAMES = {21: "FTP", 22: "SSH", 23: "Telnet", 25: "SMTP", 53: "DNS", 80: "HTTP", 81: "HTTP-alt",
              110: "POP3", 123: "NTP", 135: "MS-RPC", 137: "NetBIOS", 139: "NetBIOS-SSN",
              143: "IMAP", 161: "SNMP", 389: "LDAP", 443: "HTTPS", 445: "SMB", 500: "IKE",
              554: "RTSP", 1433: "MSSQL", 1723: "PPTP", 1900: "SSDP", 2323: "Telnet-alt",
              3306: "MySQL", 3389: "RDP", 5060: "SIP", 5555: "Android ADB", 5900: "VNC",
              6379: "Redis", 7547: "TR-069", 8080: "HTTP-alt", 8443: "HTTPS-alt",
              8888: "HTTP-alt", 9200: "Elasticsearch", 11211: "Memcached", 27017: "MongoDB",
              37215: "Huawei HG532", 52869: "UPnP"}


def ports_text(portsig):
    if portsig in ("multi", "pmulti"):
        return "many ports (port sweep)"
    out = []
    for p in portsig.lstrip("p").split("+"):
        try:
            n = int(p)
        except ValueError:
            continue
        out.append("%d (%s)" % (n, PORT_NAMES[n]) if n in PORT_NAMES else str(n))
    return ", ".join(out) or portsig


def ascii_head(hexs):
    try:
        b = bytes.fromhex(hexs)
    except ValueError:
        return ""
    return "".join(chr(c) if 32 <= c < 127 else "." for c in b)


PLAYBOOK = ("Analyst playbook of public scanner fingerprints: "
            "Mirai-family malware sends TCP SYN whose sequence number equals the destination IP "
            "address, typically to Telnet (23/2323). ZMap fixes the IP ID at 54321 and uses TCP "
            "window 65535. Masscan uses TCP window 1024 with no TCP options. Nmap SYN scans use "
            "small windows (1024-4096) with an MSS option. Operating-system TCP stacks (connect "
            "scans, scripts) carry full option sets (MSS, SACK, timestamps, window scale), the DF "
            "bit, and initial TTL 64 (Linux) or 128 (Windows). Anything else, including "
            "application-payload probes, is an other or custom scanner.")


def describe(sig, rich=True, kb=False):
    """Text state for one campaign: fingerprint + behavior only (no tool labels)."""
    fp = sig["fp"][0] if isinstance(sig["fp"], (list, tuple)) else sig["fp"]
    f = fp.split("|")
    proto = f[0].upper()
    ttl = f[1][1:] if len(f) > 1 else "?"
    strat = ", ".join("%s (%.0f%%)" % (STRAT_TXT.get(k, k), 100 * v) for k, v in sig["strategy"])
    lines = ["A group of %d source IP addresses observed by a passive network telescope (unused "
             "address space) during the measurement window." % sig["bots"],
             "Transport: %s. Destination port(s): %s." % (
                 "TCP SYN" if proto == "TCP" else proto,
                 ports_text(sig["portsig"]) if rich else sig["portsig"].lstrip("p").replace(
                     "multi", "many ports"))]
    if proto == "TCP" and len(f) >= 8:
        win = f[3][1:]
        lines.append("Shared packet-header fingerprint: initial IP TTL %s; IP ID %s; TCP window "
                     "%s; TCP options %s; TCP sequence number %s; source port %s; DF bit %s." % (
                         ttl, IPID.get(f[2], f[2]), "random" if win in ("", "x") else win,
                         opts_words(f[4][1:]), {"D": "equal to the destination IP address",
                                                "Z": "zero"}.get(f[5][1:], "random-looking"),
                         SPORT.get(f[6], f[6]), f[7][2:]))
    elif proto == "UDP" and len(f) >= 6:
        ph = sig.get("phead") or f[5][2:]
        pay = ("payload begins with bytes %s (hex), ASCII \"%s\"" % (ph, ascii_head(ph))) if rich             else "payload begins with bytes %s (hex)" % f[5][2:]
        lines.append("Shared packet-header fingerprint: initial IP TTL %s; IP ID %s; source port "
                     "%s; DF bit %s; %s." % (ttl, IPID.get(f[2], f[2]), SPORT.get(f[3], f[3]),
                                             f[4][2:], pay))
    lines.append("Target-selection strategy (statistical tests on sampled destinations): %s." %
                 strat)
    lines.append("Destinations are %s; median rate at the telescope %.1f packets per hour per "
                 "source; %d distinct dark destinations in total." % (
                     sig["disp"][0], sig["pph_med"], sig["ud_sum"]))
    lines.append("Sources span %d countries and %d autonomous systems." % (sig["n_cc"],
                                                                           sig["n_asn"]))
    if kb:
        lines.append(PLAYBOOK)
    return "\n".join(lines)


def ask(state, questions, timeout=180, retries=2):
    body = json.dumps({"model": "openjev", "state": state, "questions": questions}).encode()
    for k in range(retries + 1):
        try:
            req = urllib.request.Request(URL, data=body, headers={"Content-Type": "application/json"})
            t0 = time.time()
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.loads(r.read()), time.time() - t0
        except Exception as e:  # transient shim/vLLM hiccup
            if k == retries:
                return {"error": str(e)}, 0.0
            time.sleep(2)


def probs(ans, name):
    """Per-option probabilities for question `name` from a /v1/systemone answer.
    choice -> {option: p}; score -> {criterion: p}; noul -> {"yes": p, "no": 1-p}."""
    a = ans.get("answers", {}).get(name, {}) if isinstance(ans, dict) else {}
    if a.get("type") == "noul" and "noul" in a:
        return {"yes": float(a["noul"]), "no": 1.0 - float(a["noul"])}
    p = a.get("probabilities")
    if isinstance(p, dict):
        lg = a.get("legend")
        return {lg.get(k, k): v for k, v in p.items()} if isinstance(lg, dict) else p
    return {}


def strategy_png(points, size=(4.0, 3.0)):
    """Render (t, offset) points as a bare scatter; return a PNG data URL."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=size, dpi=96)
    if points:
        t, o = zip(*points)
        ax.scatter(t, o, s=4, c="black")
    ax.set_xlabel("time"); ax.set_ylabel("destination address")
    ax.set_xticks([]); ax.set_yticks([])
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
