#!/usr/bin/env bash
# End-to-end OrchScope pipeline on one week of ORION data (Sep. 24-30, 2026 in the paper).
# Run from a work directory containing raw/<YYYY-MM-DD.HH>.json.zst; CODE points to code/.
# Steps that query openjev (5, 10) need its local server at http://127.0.0.1:3000.
set -euo pipefail
CODE=${CODE:-$(cd "$(dirname "$0")/../code" && pwd)}
PY=${PY:-python3}
W0=2026-09-24           # window start (landmark)
DAY=2026-09-30          # day replayed hour by hour
mkdir -p red prof

# 1. reduce every hourly file (this runs at the data host; standard library only)
for f in raw/*.json.zst; do
  l=$(basename "$f" .json.zst)
  [ -s "red/$l.tsv.zst" ] && continue
  zstd -dc "$f" | $PY "$CODE/camp_reduce.py" "$l" | zstd -q -3 -c > "red/$l.tsv.zst.tmp"
  mv "red/$l.tsv.zst.tmp" "red/$l.tsv.zst"
done

# 2. per-source week profiles (24 shards in parallel)
for k in $(seq 0 23); do
  $PY -W ignore "$CODE/engine.py" features --red red --out "prof/week_$k.tsv" --w0 $W0 \
      --shard "$k" --nshards 24 &
done
wait

# 3. week analysis: campaigns, residue, orchestration test, negative control
$PY -W ignore "$CODE/analyze_week.py" --profiles 'prof/week_*.tsv' --out week_results.json --H 168 --B 200

# 4. peer-ratio negative controls and revision checks
PEER_RATIO=1 $PY -W ignore "$CODE/negctl.py" negctl_r1.json top1
PEER_RATIO=2 $PY -W ignore "$CODE/negctl.py" negctl_r2.json top1
PEER_RATIO=5 $PY -W ignore "$CODE/negctl.py" negctl2.json top1
$PY -W ignore "$CODE/revision_checks.py" orch      # -> rev_orch.json
$PY -W ignore "$CODE/revision_checks.py" tzneg     # -> rev_tzneg.json (same-country control)
$PY -W ignore "$CODE/revision_checks.py" pemp      # -> rev_pemp.json (Monte Carlo vs Gaussian)
$PY -W ignore "$CODE/revision_checks.py" negp      # -> rev_negp.json (control replay, all p-values)
PEER_CC=0 $PY -W ignore "$CODE/revision_checks.py" tzneg --out rev_tzneg_port.json   # port-only peers
$PY -W ignore "$CODE/revision_checks.py" fallback  # -> rev_fallback.json (port-only fallback share)
$PY -W ignore "$CODE/revision_checks.py" stage3    # -> rev_stage3.json (what the 10^4 stage adds)

# 5. case study, largest orchestrated campaigns, openjev labels, drift, demo data
$PY -W ignore "$CODE/case_study.py"                # -> week_case.json
$PY -W ignore "$CODE/week_orch.py"                 # -> week_orch.json (openjev)
$PY -W ignore "$CODE/week_extras.py"               # -> week_activity.json, week_openjev.json
$PY -W ignore "$CODE/drift.py"                     # -> week_drift.json
$PY -W ignore "$CODE/export_demo.py" demo_data.js  # anonymized offline replay

# 6. per-day inference and day-to-day lineage
for d in 0 1 2 3 4 5 6; do
  day=$(date -u -d "$W0 +$d day" +%F)
  for k in 0 1 2 3; do
    $PY -W ignore "$CODE/engine.py" features --red red --glob "$day.*" --out "prof/day${d}_$k.tsv" \
        --w0 "$day" --shard "$k" --nshards 4 &
  done
done
wait
for d in 0 1 2 3 4 5 6; do
  $PY -W ignore "$CODE/analyze_week.py" --profiles "prof/day${d}_*.tsv" --out "camp_day_${d}.json" \
      --H 24 --B 150 &
done
wait
$PY "$CODE/lineage.py" lineage.json

# 7. hourly streaming replay of $DAY over the weekly landmark window
$PY -W ignore "$CODE/stream.py" --red red --w0 $W0 --day $DAY --out stream_0930.json --B 200

# 8. simulation study on real background profiles of the first day
for k in $(seq 0 7); do
  $PY "$CODE/engine.py" features --red red --glob "$W0.*" --out "prof/bg24_$k.tsv" --w0 $W0 \
      --shard "$k" --nshards 8 &
done
wait
awk 'FNR==1 && NR!=1 {next} {print}' prof/bg24_*.tsv > prof_bg24.tsv
export TEST_PROCS=1   # simulate.py parallelizes over runs; each run tests its campaigns serially
$PY -W ignore "$CODE/simulate.py" --bg prof_bg24.tsv --out sim_results.json --procs 16 --seeds 3
$PY -W ignore "$CODE/simulate.py" --bg prof_bg24.tsv --out sim_abl10.json --procs 16 --seeds 10 --only-abl --hp
unset TEST_PROCS

# 9. DarkVec baseline: simulated seeds and one real day
$PY -W ignore "$CODE/dv_sim.py" prep --seeds 10
for s in $(seq 1 10); do   # single-threaded per seed (deterministic), seeds in parallel
  $PY "$CODE/darkvec.py" "dv/rows_$s.tsv" "dv/labels_$s.tsv" --seed "$s" &
done
wait
$PY -W ignore "$CODE/dv_sim.py" eval --seeds 10   # -> dv_results.json
$PY -W ignore "$CODE/dv_real.py" prep
$PY "$CODE/darkvec.py" dv_real/rows.tsv dv_real/labels.tsv --seed 7 --threads 40 --max-sent 20000
$PY -W ignore "$CODE/dv_real.py" eval --labels labels.tsv --out dv_real.json

# 10. openjev on ground-truth campaigns
$PY -W ignore "$CODE/openjev_sim.py" openjev_sim_plain.json plain
$PY -W ignore "$CODE/openjev_sim.py" openjev_sim_rich.json
$PY -W ignore "$CODE/openjev_sim.py" openjev_sim_kb.json kb

echo "done: copy the *.json results into results/ and run code/make_numbers.py and code/make_figs3.py"
