# OrchScope

Code and aggregate results for **"OrchScope: A Calibrated, Near-Real-Time System to Infer
Orchestrated Scanning Campaigns at a Network Telescope"** (submitted to IEEE ICC 2027).

OrchScope reads the hourly flow records of the Merit ORION network telescope (a /13) and outputs
well-defined scanning campaigns with calibrated orchestration decisions. It has four engines:

| Engine | What it does | Code |
|---|---|---|
| Profiling | per-source randomness, trend, and dispersion statistics plus a packet-header fingerprint, from flow records with at most three packet samples | `code/engine.py` (`profile`) |
| Inference | fingerprint-blocked HDBSCAN-eps campaigns; groups whose members converge on one dark address are set aside as residue | `code/engine.py` (`enhanced_infer`) |
| Orchestration | two-null synchrony test (whole-day rotation null and activity-matched peer null), exact Monte Carlo p-values, intersection-union decision, Benjamini-Hochberg | `code/engine.py` (`sync_test`) |
| AI decision | typed questions to the open-weights openjev decision model | `code/openjev_judge.py` |

A data-host reducer (`code/camp_reduce.py`, standard library only) turns each hourly ORION file
into per-source rows, and `code/stream.py` runs the engines after every hourly batch.

## Repository layout

```
code/      pipeline, evaluation, and generators of the paper's numbers and figures
results/   aggregate result files (per run / per campaign; no per-source data, no IP addresses)
demo/      offline replay of the tool (open demo/index.html in a browser; no server needed)
scripts/   run_all.sh: the end-to-end pipeline on ORION data
```

## Reproduce every number, table, and figure of the paper

```bash
pip install -r requirements.txt
python code/make_numbers.py   # -> out/numbers.tex, out/tab_sim.tex, out/tab_top.tex
python code/make_figs3.py     # -> out/fig_arch.pdf, fig_sens.pdf, fig_case.pdf, fig_rt.pdf
```

Both scripts read only `results/`. `out/numbers.tex` defines every number quoted in the paper.

## Re-run the full pipeline (requires ORION data)

The raw input is Merit ORION flow data, available to researchers under a data-use agreement with
Merit Network; it is not redistributed here. `scripts/run_all.sh` documents every step, run from
a work directory with:

```
raw/<label>.json.zst   ORION hourly files, label = YYYY-MM-DD.HH (UTC)
red/                   reduced hourly files (step 1)
prof/                  per-source profiles (steps 2 and 6)
```

Steps: (1) reduce each hourly file; (2) build week profiles; (3) week analysis and negative
control; (4) peer-ratio negative controls, residue/latency characterization, same-country
control, and the Monte Carlo vs. Gaussian p-value comparison; (5) case study, largest campaigns,
openjev labels, drift; (6) per-day inference and lineage; (7) hourly streaming replay;
(8) simulation study; (9) DarkVec baseline on simulated and real traffic; (10) openjev on
ground-truth campaigns. Copy the resulting JSON files into `results/` and regenerate the outputs.

Settings (environment variables read by `code/engine.py`):

| Variable | Default | Meaning |
|---|---|---|
| `PVAL` | `mc` | exact Monte Carlo p-values; `normal` reproduces the Gaussian-tail comparison |
| `MC_B2` | `1000` | fresh surrogates drawn when the stage-1 p-value is at most `MC_SCREEN` |
| `MC_SCREEN` | `0.1` | stage-1 screening level |
| `CONV_TAU` | `0.5` | residue threshold: share of members sharing a modal sampled destination |
| `PEER_RATIO` | `1` | a campaign is testable if its port has this many times as many non-member sources |
| `PEER_KEY` | `top1` | peer population: primary port |

The AI decision engine and `week_orch.py`, `week_extras.py`, and `openjev_sim.py` query openjev
([model card](https://huggingface.co/openjev/openjev), CC BY-NC 4.0) through a local server
exposing `/v1/systemone` at `http://127.0.0.1:3000` (see the model card for serving).

## Data and privacy

No file in this repository contains IP addresses or other per-source records. `results/` holds
per-run and per-campaign aggregates with network operator names removed; `demo/data.js` holds
campaign summaries only (counts, country tallies, header-field classes, hourly activity, and
destination offsets inside the monitored block). Traffic was analyzed passively under a data-use
agreement with Merit Network.

## License

The code is released under the MIT License (see `LICENSE`). openjev is governed by its own
license (CC BY-NC 4.0).
