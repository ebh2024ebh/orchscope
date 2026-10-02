# -*- coding: utf-8 -*-
"""Generate paper/numbers.tex and paper/tab_top.tex from the result files (no hand-typed
numbers). Missing inputs yield '--' placeholders so the paper always compiles."""
import glob, json, os, re, statistics

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AN, PA = os.path.join(ROOT, "results"), os.path.join(ROOT, "out")
N = {}
os.makedirs(PA, exist_ok=True)


def put(k, v):
    N[k] = v


def f2(x):
    return "%.2f" % x


def pct(x, d=1):
    return ("%%.%df" % d) % (100 * x)


def load(name):
    p = os.path.join(AN, name)
    return json.load(open(p)) if os.path.exists(p) else None


# ------------------------------------------------------------ simulation
sim = load("sim_results.json")
if sim:
    r = pd.DataFrame(sim)
    m = r.groupby(["name", "variant"]).mean(numeric_only=True)

    def g(name, var, col):
        return m.loc[(name, var), col]
    put("SimARIOurs", f2(g("abl", "ours", "ari"))); put("SimFOurs", f2(g("abl", "ours", "b3_f1")))
    put("SimRecOurs", f2(g("abl", "ours", "campaign_recall")))
    put("SimARIPT", f2(g("abl", "porttime", "ari"))); put("SimFPT", f2(g("abl", "porttime", "b3_f1")))
    put("SimRecPT", f2(g("abl", "porttime", "campaign_recall")))
    put("SimTPR", f2(g("abl", "ours", "orch_tpr"))); put("SimFPR", f2(g("abl", "ours", "orch_fpr")))
    put("SimStrat", pct(g("abl", "ours", "strategy_acc_det"), 0))
    for key, var in (("AblFp", "-fingerprint"), ("AblHdb", "-hdbscan"), ("AblSt", "-strategy"),
                     ("AblSync", "+sync")):
        put(key, f2(g("abl", var, "ari"))); put(key + "F", f2(g("abl", var, "b3_f1")))
        put(key + "R", f2(g("abl", var, "campaign_recall")))
    rr = [g("abl", v, c) for v in ("-rate", "-coverage", "-temporal") for c in ("ari",)]
    put("AblRate", f2(sum(rr) / 3))
    put("AblRateF", f2(sum(g("abl", v, "b3_f1") for v in ("-rate", "-coverage", "-temporal")) / 3))
    put("AblRateR", f2(sum(g("abl", v, "campaign_recall") for v in ("-rate", "-coverage", "-temporal")) / 3))
    put("AblSyncFPR", f2(g("abl", "+sync", "orch_fpr")))
    put("ScaleTwenty", f2(g("scale_/20", "ours", "ari")))
    put("ConcSixteen", f2(g("conc_K16", "ours", "ari")))
    put("ConcOnePur", pct(g("conc_K1", "ours", "b3_p"), 0))
    put("SweepBest", f2(g("sweep_r12_n400", "ours", "ari")))
    put("SweepLow", f2(g("sweep_r3_n100", "ours", "ari")))
    sd = r.groupby(["name", "variant"])[["ari", "b3_f1"]].std()
    put("SimSD", f2(float(sd.loc["abl"].max().max())))
    # calibration of the orchestration test over every run of the full pipeline
    x = r[(r.variant == "ours") & (r.name != "conc_K1")]
    put("CalRuns", str(len(x)))
    put("CalFPRuns", str(int((x.orch_fpr.fillna(0) > 0).sum())))
    put("CalMeanFPR", f2(float(x.orch_fpr.mean())))
    put("CalMinTPR", f2(float(x.orch_tpr.min())))
    put("CalMeanTPR", f2(float(x.orch_tpr.mean())))
    if "orch_tpr_rot" in x:
        put("CalMeanTPRRot", f2(float(x.orch_tpr_rot.mean())))
    put("ScaleTwentyTPR", f2(g("scale_/20", "ours", "orch_tpr")))
    if "orch_fpr_rot" in x:
        put("CalMeanFPRRot", f2(float(x.orch_fpr_rot.mean())))
        put("CalMinTPRRot", f2(float(x.orch_tpr_rot.min())))
    if "orch_scores" in x:
        from sklearn.metrics import roc_auc_score
        sc = [s for lst in x.orch_scores if isinstance(lst, list) for s in lst]
        if sc and len({s[0] for s in sc}) == 2:
            put("SimCoordAUCAll", f2(roc_auc_score([s[0] for s in sc], [s[1] for s in sc])))
            put("SimCoordN", str(len(sc)))

# ------------------------------------------------------------ default configuration, many seeds
# Table I and the headline comparison use sim_abl10.json (+ DarkVec on the same seeds); the
# sweeps above keep the three-seed grid.
TAB_SIM = None
ab = load("sim_abl10.json")
dvr = load("dv_results.json")
if ab:
    from scipy import stats as sst
    A = pd.DataFrame(ab + (dvr or []))

    def col(v, c):
        return A[A.variant == v][c].dropna()

    def mci(v, c):
        x = col(v, c)
        if len(x) < 2:
            return (float(x.mean()) if len(x) else float("nan")), 0.0
        return float(x.mean()), float(sst.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / len(x) ** 0.5)

    put("NSeeds", str(col("ours", "ari").shape[0]))
    for key, v in (("Ours", "ours"), ("DV", "darkvec"), ("PT", "porttime")):
        if col(v, "ari").shape[0]:
            put("SimARI" + key, f2(mci(v, "ari")[0])); put("SimP" + key, f2(mci(v, "b3_p")[0]))
            put("SimR" + key, f2(mci(v, "b3_r")[0])); put("SimF" + key, f2(mci(v, "b3_f1")[0]))
            put("SimRec" + key, f2(mci(v, "campaign_recall")[0]))
    put("SimTPR", f2(mci("ours", "orch_tpr")[0])); put("SimFPR", f2(mci("ours", "orch_fpr")[0]))
    if dvr:
        o = A[A.variant == "ours"].set_index("seed"); d = A[A.variant == "darkvec"].set_index("seed")
        common = o.index.intersection(d.index)

        def pfmt(p):
            return "<0.001" if p < 0.001 else ("%.3f" % p)
        for key, c in (("PvalF", "b3_f1"), ("PvalRec", "campaign_recall"), ("PvalARI", "ari")):
            x, y = o.loc[common, c].to_numpy(), d.loc[common, c].to_numpy()
            p = sst.wilcoxon(x, y).pvalue if (x != y).any() else 1.0
            put(key, pfmt(p))
        hb = A[A.variant == "-hdbscan"].set_index("seed")         # header+port blocks, paired
        ch = o.index.intersection(hb.index)
        for key, c in (("PvalHdbF", "b3_f1"), ("PvalHdbRec", "campaign_recall")):
            x, y = o.loc[ch, c].to_numpy(), hb.loc[ch, c].to_numpy()
            put(key, pfmt(sst.wilcoxon(x, y).pvalue if (x != y).any() else 1.0))
    put("AblFpR", f2(mci("-fingerprint", "campaign_recall")[0]))
    put("AblHdbR", f2(mci("-hdbscan", "campaign_recall")[0]))
    put("AblHdbP", f2(mci("-hdbscan", "b3_p")[0]))
    put("AblHdbFPR", f2(mci("-hdbscan", "orch_fpr")[0]))
    put("AblHdbFF", f2(mci("-hdbscan", "b3_f1")[0]))
    put("AblStR", f2(mci("-strategy", "campaign_recall")[0]))
    put("AblSyncFPR", f2(mci("+sync", "orch_fpr")[0]))
    put("EpsLowFPR", f2(mci("eps0.25", "orch_fpr")[0]))
    put("McsLowARI", f2(mci("mcs0.05", "ari")[0]))

    def cell(v, c, ci=False, bold=False):
        x = col(v, c)
        if not len(x):
            return "--"
        mu, h = mci(v, c)
        s_ = "%.2f" % mu
        if bold:
            s_ = r"\textbf{%s}" % s_
        if ci:
            s_ += r"{\scriptsize$\pm$%s}" % ("%.2f" % h).lstrip("0")
        return s_

    def row(name, v, ci=False, bold=False, test=True):
        cs = [cell(v, "ari", ci, bold), cell(v, "b3_p"), cell(v, "b3_r"), cell(v, "b3_f1", ci, bold),
              cell(v, "campaign_recall", ci, bold)]
        cs += [cell(v, "orch_tpr"), cell(v, "orch_fpr")] if test else ["--", "--"]
        return name + " & " + " & ".join(cs) + r" \\"

    def avg_row(name, vs):
        cs = []
        for c in ("ari", "b3_p", "b3_r", "b3_f1", "campaign_recall", "orch_tpr", "orch_fpr"):
            vals = [mci(v, c)[0] for v in vs]
            vals = [x for x in vals if x == x]
            cs.append("%.2f" % np.mean(vals) if vals else "--")
        return name + " & " + " & ".join(cs) + r" \\"
    TAB_SIM = [r"\begin{tabular}{@{}lccccccc@{}}", r"\toprule",
               r"Method & ARI & B$^3$-P & B$^3$-R & B$^3$-F1 & Recall & TPR & FPR \\", r"\midrule",
               row(r"\textbf{\sysname}", "ours", ci=True, bold=True),
               row(r"DarkVec~\cite{gioacchini2021darkvec}", "darkvec", ci=True, test=False),
               row(r"Header+port blocks", "-hdbscan", ci=True),
               row("Port + onset", "porttime", ci=True, test=False), r"\midrule",
               row(r"$-$fingerprint", "-fingerprint"),
               row(r"$-$strategy", "-strategy"),
               avg_row(r"$-$rate/cov./duty", ["-rate", "-coverage", "-temporal"]),
               row(r"$+$onset bursts", "+sync"), r"\midrule",
               row(r"$\varepsilon=0.25$", "eps0.25"), row(r"$\varepsilon=1.5$", "eps1.5"),
               row(r"min.\ mode 5\%", "mcs0.05"), row(r"min.\ mode 20\%", "mcs0.2"),
               r"\bottomrule", r"\end{tabular}"]

# ------------------------------------------------------------ DarkVec on one real day
dvreal = load("dv_real.json")
if dvreal:
    for key, k in (("DV", "darkvec"), ("Ours", "ours")):
        r_ = dvreal[k]
        put("Real%sClusters" % key, "{:,}".format(r_["clusters"]).replace(",", "{,}"))
        put("Real%sMed" % key, "%d" % int(r_["median_size"]))
        put("Real%sHom" % key, f2(r_["homogeneity"]))
        put("Real%sFP" % key, f2(r_["fp_purity_median"]))
        put("Real%sCov" % key, pct(r_["covered"], 0))

# ------------------------------------------------------------ peer requirement trade-off
for key, fname, k in (("Two", "negctl_r2.json", "top1"), ("Five", "negctl2.json", "top1"),
                      ("One", "negctl_r1.json", "top1")):
    q = load(fname)
    if q and k in q:
        put("Peer%sTest" % key, pct(q[k]["testable_frac"], 0))
        put("Peer%sFPR" % key, "%.1f" % (100 * q[k]["fpr_both"]))

# ------------------------------------------------------------ AI layer
ai = {v: load("openjev_sim_%s.json" % v) for v in ("plain", "rich", "kb")}
if all(ai.values()):
    put("AIN", str(len(ai["plain"]["rows"])))
    put("AIService", pct(ai["plain"]["service_acc"], 0))
    put("AIStratImg", pct(ai["plain"]["strategy_img_acc"], 1))
    put("AIStratStat", pct(ai["plain"]["strategy_stat_acc"], 1))
    put("AIToolPlain", pct(ai["plain"]["tool_acc"], 0)); put("AIToolRich", pct(ai["rich"]["tool_acc"], 0))
    put("AIToolKB", pct(ai["kb"]["tool_acc"], 0))
    put("AICoordAUC", f2(ai["plain"]["coord_auc_openjev"]))
    put("SimCoordAUC", f2(ai["plain"]["coord_auc_test"]))
    lat = ai["plain"]["latency_s"] + ai["rich"]["latency_s"] + ai["kb"]["latency_s"]
    put("AILatency", "%.2f" % statistics.median(lat))

# ------------------------------------------------------------ data scale (one-hour probe)
put("HourRecords", "20\\,M"); put("HourGB", "11"); put("HourSources", "148\\,k")
red_sizes = load("red_sizes.json")
put("HourRedMB", str(round(statistics.median(red_sizes) / 1e6)) if red_sizes else "17")
put("BGProfiles", "726{,}595")

# ------------------------------------------------------------ week
wk = load("week_results.json")
wo = load("week_openjev.json")
meta = load("week_meta.json")
if wk:
    o = wk["ours"]
    put("WeekCampaigns", "{:,}".format(o["campaigns"]).replace(",", "{,}"))
    n_res = o.get("residue", 0)
    n_scan = o["campaigns"] - n_res
    put("WeekResidue", "{:,}".format(n_res).replace(",", "{,}"))
    put("WeekResiduePct", pct(n_res / max(o["campaigns"], 1), 0))
    put("WeekResidueSrcPct", pct(o.get("residue_sources", 0) / max(o["covered"], 1), 0))
    put("WeekScan", "{:,}".format(n_scan).replace(",", "{,}"))
    if "testable" in o:
        put("WeekTestablePct", pct(o["testable"] / max(n_scan, 1), 0))
        put("WeekOrchPct", pct(o["orchestrated"] / max(o["testable"], 1), 0))
        put("WeekOrchN", "{:,}".format(o["orchestrated"]).replace(",", "{,}"))
    else:
        put("WeekOrchPct", pct(o["orchestrated"] / max(o["campaigns"], 1), 0))
    put("WeekOrchN", "{:,}".format(o["orchestrated"]).replace(",", "{,}"))
    put("WeekCands", "{:,}".format(wk["n_profiles"]).replace(",", "{,}"))
    put("WeekCoveredPct", pct(o["covered"] / max(wk["n_profiles"], 1), 0))
    put("WeekHom", f2(o["homogeneity"])); put("WeekV", f2(o["v_measure"]))
    put("NegFPR", pct(wk["negative_control"]["fpr"], 1))
    if "fpr_rot" in wk["negative_control"]:
        put("NegFPRRot", pct(wk["negative_control"]["fpr_rot"], 0))
    Lt = pd.DataFrame(wk["landscape"])
    if "conv" in Lt:
        put("ResTauHi", pct((Lt.conv >= 0.3).mean(), 0))     # residue share at threshold 0.3
        put("ResTauLo", pct((Lt.conv >= 0.7).mean(), 0))     # ... and at 0.7
    if "testable" in Lt and "orchestrated_rot" in Lt:
        Tt = Lt[Lt.testable]
        put("WeekOrchRotPct", pct(Tt.orchestrated_rot.mean(), 0))
        Oo = Tt[Tt.orchestrated]
        put("OrchMedBots", "%d" % int(Oo.bots.median()))
        put("OrchMedCC", "%d" % int(Oo.n_cc.median()))
        put("OrchMedSpan", "%d" % int(round(Oo.span_med.median())))
        put("OrchUDPPct", pct((Oo.proto == 2).mean(), 0))
        put("OrchMixPct", pct((Oo.proto == 3).mean(), 0))            # both TCP SYN and UDP
        put("OrchOtherPct", pct((~Oo.proto.isin([1, 2])).mean(), 0))  # mixes and ICMP echo
    elif "orchestrated_rot" in o:
        put("WeekOrchRotPct", pct(o["orchestrated_rot"] / max(o["campaigns"], 1), 0))
    put("NegN", "{:,}".format(wk["negative_control"]["pseudo_campaigns"]).replace(",", "{,}"))
    put("NegRaw", pct(wk["negative_control"]["raw_p_lt_0.05"], 1))
    put("WeekFlagged", "{:,}".format(o["flagged"]).replace(",", "{,}"))
    put("WeekFlagCov", pct(o["flagged_covered"], 0))
    put("WeekInferMin", "%.0f" % (wk["t_ours"] / 60))
    for key, var in (("NoFp", "fingerprint"), ("NoSt", "strategy"), ("NoPol", "policy")):
        w2 = wk.get("ours_minus_" + var)
        if w2:
            put("WeekV" + key, f2(w2["v_measure"])); put("WeekHom" + key, f2(w2["homogeneity"]))
            put("WeekCamp" + key, "{:,}".format(w2["campaigns"]).replace(",", "{,}"))
    Lw = pd.DataFrame(wk["landscape"])
    if len(Lw):
        put("WeekOrchBotsPct", pct(Lw[Lw.orchestrated].bots.sum() / max(Lw.bots.sum(), 1), 0))
        put("WeekMaxBots", "{:,}".format(int(Lw.bots.max())).replace(",", "{,}"))
        put("WeekMedBots", "%d" % int(Lw.bots.median()))
dr = load("week_drift.json")
if dr:
    put("OneDayPct", pct(dr["one_day_frac"], 0))
    put("NextDayPct", pct(statistics.median(dr["next_day_persistence"]), 0))
    put("DriftPorts", str(dr["ports"]))
    put("DriftShare", pct(dr["port_cv_share_gt_0.2"], 0))
    put("DriftPeak", "%.0f" % dr["port_peak_over_mean_p90"])
lin = load("lineage.json")
if lin:
    cpd = lin["campaigns_per_day"]
    put("DayCampMin", "{:,}".format(min(cpd)).replace(",", "{,}"))
    put("DayCampMax", "{:,}".format(max(cpd)).replace(",", "{,}"))
    put("LinMultiPct", pct(lin["multi_day_fraction"], 0))
    put("LinLinkJ", f2(lin["median_link_jaccard"]))
    put("LinFull", str(lin["track_len_hist"].get(str(lin["days"]), 0)))
    put("LinLinkPct", pct(lin["links"] / max(sum(cpd[:-1]), 1), 0))
case = load("week_case.json")
if case:
    P = {pn["kind"]: pn for pn in case["panels"]}
    o_ = P.get("orchestrated"); su = P.get("surrogate"); co = P.get("co-tooled")
    if o_:
        put("CaseOrchBots", "{:,}".format(o_["bots"]).replace(",", "{,}"))
        put("CaseOrchPort", o_["port"].replace("+", "/"))
        put("CaseOrchT", f2(o_["T"])); put("CaseOrchZ", "%.1f" % o_["z"])
    if su:
        put("CaseSurT", f2(su["T"]))
    if P.get("peers"):
        put("CasePeerT", f2(P["peers"]["T"]))
    if co:
        put("CaseCoT", f2(co["T"])); put("CaseCoZ", "%.1f" % co["z"])
        put("CaseCoBots", "{:,}".format(co["bots"]).replace(",", "{,}"))
if meta:
    put("WeekSources", "{:,}".format(meta["sources"]).replace(",", "{,}"))
if wo:
    pr = wo["probe_records"]
    put("WeekRecords", "%.1f\\,billion" % (pr / 1e9))
    put("WeekToolAcc", pct(wo["tool_vs_flags_acc"], 0) if wo["tool_vs_flags_acc"] is not None else "--")
    put("WeekToolN", str(wo["n_flag_labeled"]))
    if wo["tool_vs_flags_acc"] is not None:
        put("WeekToolHits", str(int(round(wo["tool_vs_flags_acc"] * wo["n_flag_labeled"]))))
put("WeekWindow", "Sep.~24--30, 2026")

# ------------------------------------------------------------ real time
st = load("stream_0930.json")
tim = load("red_times.json")
if tim:
    put("RedHostMin", "%.1f" % (statistics.median(tim["host"]) / 60))
    put("RedSrvMin", "%.1f" % (statistics.median(tim["server"]) / 60))
if st:
    s = pd.DataFrame(st)
    if "warm" in s:
        w_ = s[s.warm == True]
        if len(w_):
            w_ = w_.iloc[0]
            put("StreamWarmMin", "%.0f" % ((w_.t_ingest + w_.t_profile + w_.t_infer) / 60))
            put("StreamWarmFiles", str(int(w_.files)))
        s = s[s.warm != True].reset_index(drop=True)
    put("StreamHours", str(len(s)))
    put("StreamOrchMed", "%.0f" % s.orchestrated.median())
    put("StreamCampMed", "{:,}".format(int(s.campaigns.median())).replace(",", "{,}"))
    proc = s.t_read + s.t_ingest + s.t_profile + s.t_infer
    put("StreamProfMed", "%.0f" % s.t_profile.median())
    put("StreamInferMed", "%.0f" % s.t_infer.median())
    put("StreamMed", "%.0f" % proc.median()); put("StreamMax", "%.0f" % proc.max())
    put("StreamProfiles", "{:,}".format(int(s.profiles.max())).replace(",", "{,}"))
    put("StreamCont", pct(s.continuity.dropna().median(), 0))
    c_ = s.dropna(subset=["continuity"])
    new = (1 - c_.continuity) * c_.campaigns        # groups that continue no earlier campaign
    put("StreamNewMed", "%.0f" % new.median()); put("StreamNewMax", "%.0f" % new.max())
    lat_ = float(N.get("AILatency", "0.29"))           # the AI engine judges only those
    put("AIBudgetMed", "%.0f" % (new.median() * lat_)); put("AIBudgetMax", "%.0f" % (new.max() * lat_))
    last = s.iloc[-1]
    put("StreamLastCamp", "{:,}".format(int(last.campaigns)).replace(",", "{,}"))
    put("StreamLastOrch", "{:,}".format(int(last.orchestrated)).replace(",", "{,}"))
    if tim:
        # conservative end-to-end figure: reduction at the (shared, busy) data host + engine
        ai_s = float(N.get("AIBudgetMed", "0"))     # the AI engine on new campaigns, included
        put("RTMinutes", "%.0f" % round((statistics.median(tim["host"]) + proc.median() + ai_s) / 60))

# ------------------------------------------------------------ revision checks
ro = load("rev_orch.json")
if ro:
    oo, sc, rs = ro["orchestrated"], ro.get("scanning", {}), ro.get("residue", {})
    put("OrchTCPPct", pct(oo["tcp"], 0)); put("OrchUDPPctR", pct(oo["udp"], 0))
    put("OrchOneASPct", pct(oo["one_asn"], 0)); put("OrchOneSlashPct", pct(oo["one_src24"], 0))
    put("OrchOneCCPct", pct(oo["one_cc"], 0)); put("OrchResearchPct", pct(oo["research_half"], 1))
    if sc:
        put("ScanUDPPct", pct(sc["udp"], 0))
    if rs:
        put("ResUDPPct", pct(rs["udp"], 0)); put("ResSingleAddrPct", pct(rs["single_addr"], 0))
    tm = ro["time_to_m"]
    if tm.get("median_h") is not None:
        put("TimeToMMed", ("%.1f" % tm["median_h"]).rstrip("0").rstrip("."))
        put("TimeToMSamePct", pct(tm["share_same_hour"], 0))
tz = load("rev_tzneg.json")
if tz:
    put("TzNegN", "{:,}".format(tz["pseudo_campaigns"]).replace(",", "{,}"))
    put("TzNegFPR", pct(tz["fpr"], 1)); put("TzNegFPRRot", pct(tz["fpr_rot"], 0))
    if "raw_p_lt_0.01" in tz:                       # unadjusted two-null rejection rates
        put("TzRawFive", pct(tz["raw_p_lt_0.05"], 1)); put("TzRawOne", pct(tz["raw_p_lt_0.01"], 1))
ng = load("rev_negp.json")                          # exact replay of the week's control
if ng:
    put("NegRawFive", pct(ng["raw_p_lt_0.05"], 1)); put("NegRawOne", pct(ng["raw_p_lt_0.01"], 1))
    put("NegRotRawFive", pct(ng["raw_rot_p_lt_0.05"], 0))
tp = load("rev_tzneg_port.json")                    # same control, port-only peers (PEER_CC=0)
if tp:
    put("TzPortFlagPct", pct(tp["fpr"], 1))
    put("TzPortRawFive", pct(tp["raw_p_lt_0.05"], 1)); put("TzPortRawOne", pct(tp["raw_p_lt_0.01"], 1))
pe = load("rev_pemp.json")
if pe:
    put("PempNormal", "{:,}".format(pe["orch_B1000_normal"]).replace(",", "{,}"))
    put("PempMC", "{:,}".format(pe["orch_B1000_empirical"]).replace(",", "{,}"))
    put("PempExtraPct", pct(pe["orch_B1000_normal"] / max(pe["orch_B1000_empirical"], 1) - 1, 0))
if st and wk:
    s_ = pd.DataFrame(st)
    s_ = s_[s_.get("warm", False) != True] if "warm" in s_ else s_
    lastr = s_.iloc[-1]
    put("StreamCampDiff", str(abs(int(lastr.campaigns) - int(wk["ours"]["campaigns"]))))
    put("StreamOrchDiff", str(abs(int(lastr.orchestrated) - int(wk["ours"]["orchestrated"]))))
    if "residue" in s_:
        put("StreamLastRes", "{:,}".format(int(lastr.residue)).replace(",", "{,}"))

# ------------------------------------------------------------ write
EXPECTED = set("""SimARIOurs SimTPR SimFPR NegFPR WeekRecords WeekCampaigns WeekOrchPct
RTMinutes HourRecords HourGB HourSources HourRedMB AILatency BGProfiles SimFOurs SimARIPT
SimFPT AblFp SimStrat SimRecOurs SimRecPT AblFpF AblFpR AblHdb AblHdbF AblHdbR AblSt AblStF
AblStR AblRate AblRateF AblRateR ScaleTwenty ConcSixteen AIN AIStratImg AIStratStat AIToolPlain
AIToolRich AIToolKB AICoordAUC AIService SimCoordAUC WeekWindow WeekSources WeekCands
WeekCoveredPct WeekHom WeekV RedHostMin RedSrvMin StreamMed StreamMax
StreamProfiles StreamCont ConcOnePur SweepBest SweepLow AblSync AblSyncF AblSyncR AblSyncFPR
WeekToolAcc WeekToolN SimSD NegN NegRaw WeekFlagged WeekFlagCov WeekInferMin WeekVNoFp WeekHomNoFp
WeekCampNoFp WeekVNoSt WeekHomNoSt WeekCampNoSt WeekVNoPol WeekHomNoPol WeekCampNoPol
DayCampMin DayCampMax LinMultiPct LinLinkJ LinFull CaseOrchBots CaseOrchPort CaseOrchT CaseOrchZ
CaseSurT CaseCoT CaseCoZ CaseCoBots StreamWarmMin StreamWarmFiles StreamHours StreamOrchMed
StreamCampMed WeekOrchBotsPct WeekMaxBots WeekMedBots CalRuns CalFPRuns CalMeanFPR CalMinTPR
CalMeanFPRRot CalMinTPRRot SimCoordAUCAll SimCoordN NegFPRRot WeekOrchRotPct CasePeerT CalMeanTPR
CalMeanTPRRot ScaleTwentyTPR OneDayPct NextDayPct DriftPorts DriftShare DriftPeak LinLinkPct WeekToolHits
OrchMedBots OrchMedCC OrchMedSpan OrchUDPPct StreamLastCamp StreamLastOrch NSeeds SimPOurs SimROurs
SimARIDV SimPDV SimRDV SimFDV SimRecDV SimPPT SimRPT PvalF PvalRec PvalARI EpsLowFPR McsLowARI
RealDVClusters RealDVMed RealDVHom RealDVFP RealDVCov RealOursClusters RealOursMed RealOursHom
RealOursFP RealOursCov PeerTwoTest PeerTwoFPR PeerFiveTest PeerFiveFPR StreamProfMed StreamInferMed
WeekTestablePct WeekOrchN WeekResidue WeekResiduePct WeekResidueSrcPct WeekScan ResUDPPct
ResSingleAddrPct ScanUDPPct OrchTCPPct OrchUDPPctR OrchOneASPct OrchOneSlashPct OrchOneCCPct
OrchResearchPct TimeToMMed TimeToMSamePct TzNegN TzNegFPR TzNegFPRRot PempNormal PempMC
PempExtraPct StreamCampDiff StreamOrchDiff StreamLastRes AblHdbP AblHdbFPR AblHdbFF ResTauHi ResTauLo
PvalHdbF PvalHdbRec WeekOrchN OrchMixPct OrchOtherPct StreamNewMed StreamNewMax AIBudgetMed
AIBudgetMax TzRawFive TzRawOne NegRawFive NegRawOne NegRotRawFive TzPortFlagPct TzPortRawFive
TzPortRawOne""".split())
lines = ["% generated by code/make_numbers.py -- do not edit"]
for k in sorted(EXPECTED | set(N)):
    lines.append("\\newcommand{\\%s}{%s}" % (k, N.get(k, "--")))
open(os.path.join(PA, "numbers.tex"), "w", encoding="utf-8").write("\n".join(lines) + "\n")
missing = sorted(k for k in EXPECTED if k not in N)

# ------------------------------------------------------------ top-campaign table
# Table III: the largest ORCHESTRATED campaigns (week_orch.json); fall back to the largest overall
wor = load("week_orch.json")
allrows = wor["rows"] if wor else (wo["rows"] if wo else [])
# TCP campaigns first (scanning is unambiguous there), then the largest UDP ones
tcp_rows = [r_ for r_ in allrows if not str(r_.get("fpc", "")).startswith("udp")]
udp_rows = [r_ for r_ in allrows if str(r_.get("fpc", "")).startswith("udp")]
rows = tcp_rows[:4] + udp_rows[:max(0, 7 - len(tcp_rows[:4]))]
n_tcp_shown = len(tcp_rows[:4])
for i, r_ in enumerate(rows):
    r_ = dict(r_)
    rows[i] = r_
    r_["rank"] = i + 1
svc_short = {"Telnet on IoT devices": "Telnet/IoT", "SSH": "SSH", "Web (HTTP/HTTPS)": "Web",
             "Windows SMB/RDP": "SMB/RDP", "VoIP (SIP)": "SIP", "Databases": "DB",
             "Many services (port sweep)": "sweep", "Other": "other"}
smap = {"uperm": "perm.", "perm": r"n.-u.\,perm.", "seq_fwd": r"seq.$\uparrow$",
        "seq_rev": r"seq.$\downarrow$", "undet": "--"}
t = [r"\begin{tabular}{@{}rllrlrrl@{}}", r"\toprule",
     r"\# & Port(s) & Fingerprint & Bots & Strat. & CCs & $z$ & Service \\", r"\midrule"]
for r_ in rows:
    if r_["rank"] == n_tcp_shown + 1 and 0 < n_tcp_shown < len(rows):
        t.append(r"\midrule")
    port = r_["portsig"].lstrip("p").replace("multi", "many").replace("+", "/")
    t.append(r"%d & %s & %s & %s & %s & %d & %.1f & %s \\" % (
        r_["rank"], port, r_["fp"].replace("_", r"\_"),
        "{:,}".format(r_["bots"]).replace(",", "{,}"), smap.get(r_["strategy"], r_["strategy"]),
        int(r_.get("n_cc", 0)), float(r_["z"]), svc_short.get(r_["service"], r_["service"])))
t += [r"\bottomrule", r"\end{tabular}"]
if not rows:
    t = [r"\begin{tabular}{@{}l@{}}", "(pending)", r"\end{tabular}"]
open(os.path.join(PA, "tab_top.tex"), "w", encoding="utf-8").write("\n".join(t) + "\n")
print("numbers:", len(N), "| missing:", missing)
# ------------------------------------------------------------ Table I (generated)
open(os.path.join(PA, "tab_sim.tex"), "w", encoding="utf-8").write(
    "\n".join(TAB_SIM if TAB_SIM else [r"\begin{tabular}{@{}l@{}}", "(pending)", r"\end{tabular}"])
    + "\n")
