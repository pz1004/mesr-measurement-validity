# Reproducing every artifact

Run in the order below. Each stage writes JSON to `results/` and is independent of the
later ones, so a failure never silently corrupts a downstream number. Wall times are the
measured values from the run that produced the shipped artifacts, on 20 cores + one
GTX 1660 SUPER.

## Tier 0 — no dataset needed (~15 s total)

```bash
pip install -r requirements.txt
python -m pytest -q tests/                          #  88 passed
python -m dataset_assessment.audit                  # 9/9 claims verified from source
python -m dataset_assessment.analyze                # rank_analysis.json + results/figures/
python -m dataset_assessment.make_numbers_digest > NUMBERS.md
```

`audit` needs the `EDformer/`, `EDmamba/` and `cuke-emlb/` symlinks (source inspection only,
nothing is executed). `analyze` needs `results/benchmark_*.json`, which ship with this
repository.

`analyze` also writes **Figure 1** (`results/figures/fig_retention_curves.pdf`). It is drawn
at the size it is printed at -- 516 x 63 pt, the full text width of a two-column IEEEtran
`figure*` -- and included with `width=\linewidth`, so nothing downscales the type. The float
it sits in is capped at 101.2 pt, which is that graphic plus a three-line caption; the cap is
asserted in `tests/test_figures.py::test_figure_fits_the_float_budget`, because growing the
figure silently costs a page at $265. Bands are `cluster_bootstrap_delta_ci` at 2,000 draws
on the same seed as the reported intervals, and the strip under each panel is `n(r)`,
asserted equal to `common_support.coverage`.

## Tier 1 — metric analysis (needs DND21 + DVSCLEAN, ~3 min)

```bash
python -m dataset_assessment.measure_metric_deps    # results/metric_dependencies.json
python -m dataset_assessment.profile_datasets       # results/noise_regimes.json  (211 s)
```

`profile_datasets` reads all five corpora at a 300,000-event cap.

## Tier 2 — the benchmark grid (needs the classical denoisers built)

Build first — **read `BUILD_CUKE_EMLB.md` before running this**, especially the
pinned-commit warning. Without the build, `available_methods()` returns `['raw']` only and
the grid is meaningless.

```bash
python -m dataset_assessment.run_benchmark --dataset dnd21    --max-events 1000000 \
    --methods raw random_null dwf evflow knoise red ts ynoise label_oracle      #  106 s
python -m dataset_assessment.run_benchmark --dataset dvsclean --max-events 1000000 \
    --methods raw random_null dwf evflow knoise red ts ynoise label_oracle      #  122 s
python -m dataset_assessment.run_benchmark --dataset emlb     --max-events 1000000 \
    --scene-stride 4 --methods raw random_null dwf evflow knoise red ts ynoise  # 2098 s
python -m dataset_assessment.run_benchmark --dataset pure_ba  --max-events 1000000 \
    --methods raw random_null dwf evflow knoise red ts ynoise                   # 3420 s
python -m dataset_assessment.run_benchmark --dataset dvsd22 --max-events 1000000 \
    --methods raw random_null dwf evflow knoise red ts ynoise                   #  878 s
python -m dataset_assessment.analyze                                            # re-aggregate
```

Every run writes its JSON incrementally, one recording at a time, so a killed job keeps its
partial output and can be inspected.

### Matched-retention label quality

Whether a filter that out-scores the label oracle on MESR actually selects events better.
Needs per-event labels, so it runs on the two labelled corpora only.

```bash
python -m dataset_assessment.label_quality --dataset dnd21     # results/label_quality_dnd21.json     ~90 s
python -m dataset_assessment.label_quality --dataset dvsclean  # results/label_quality_dvsclean.json  ~110 s
```

### Properties of ESR

Section IV-A's mechanism, checked against the released `esr.esr` rather than re-derived. Needs
no corpus: the counterexample is analytic and the stationarity contrast is synthetic, so this
is the one run a reader can reproduce without downloading anything. About 40 s.

```bash
python -m dataset_assessment.esr_properties   # results/esr_properties.json
```

It reports three things: a slice that majorises another and scores lower (so ESR is not
monotone in concentration), ESR's invariance to a bijection of the pixel grid, and the
stationary-versus-drifting 2x2 showing that the null's sign needs a nonstationary scene
component and not merely a high hot-pixel share.

### Common support for AUC_r

Whether the AUC_r ranking is a property of the methods or of which retentions each recording
could be measured at. Reads the five `results/benchmark_*.json`; a few seconds.

```bash
python -m dataset_assessment.common_support   # results/common_support.json
```

The `own_range` column reproduces the published Table III exactly; `common_grid` and
`common_cohort` are the two repairs. It also reports `n(r)` per corpus, which is what makes a
shrinking sample at low retention visible.

### Paired contrasts on the E-MLB ranking

What the E-MLB ranking table (`tab:emlb`) can carry. Every classical row is scored on the
same 384 recordings, so the comparison is paired; this differences within each recording
and bootstraps the 96 scene clusters. Reads `results/benchmark_emlb.json`; a few seconds.

```bash
python -m dataset_assessment.paired_contrasts --dataset emlb   # results/paired_contrasts.json
```

It replaced reasoning from whether two marginal intervals overlap, which is invalid in both
directions the table used it. 14 of the 15 pairs separate; only DWF over TS does not.

### Event-cap sensitivity

The measurement that replaced an unmeasured cap-invariance assertion. Three caps over all
384 E-MLB recordings, then the comparison. The two computed lanes run in parallel and take
about 2.5 hours wall, bounded by the 2,000,000-event lane.

The 1,000,000-event lane is not recomputed: `results/benchmark_emlb.json` **is** that run —
384 recordings, the same eight methods, the same 20-point retention grid, `max_events`
1000000 — so copying it in anchors the sweep to the table the paper prints rather than to a
claimed reproduction of it.

```bash
mkdir -p results/cap_sensitivity
cp results/benchmark_emlb.json results/cap_sensitivity/emlb_cap_1000000.json
for CAP in 500000 2000000; do
  python -m dataset_assessment.run_benchmark --dataset emlb --max-events $CAP \
    --methods raw random_null dwf evflow knoise red ts ynoise \
    --out results/cap_sensitivity/emlb_cap_${CAP}.json &
done
wait
python -m dataset_assessment.cap_sensitivity results/cap_sensitivity/emlb_cap_*.json \
    --out results/cap_sensitivity.json
python -m dataset_assessment.cap_sensitivity \
    results/cap_sensitivity/emlb_cap_{1000000,2000000}.json \
    --out results/cap_sensitivity/pair_1M_vs_2M.json   # the pair with an identical grid
```

**`--scene-stride` is omitted deliberately: the sweep needs all 384 recordings.** An earlier
48-recording run at `--scene-stride 8` is kept under `results/cap_sensitivity/subset48/`, and
it does not support the paper's numbers. Two of its conclusions are false at full scale: it
found the evaluable range moving on every cell, where 294 of 2688 hold still, and it found
every corpus mean steady between 1M and 2M, where RED's passes the unit of scale on both
cohorts (0.0395 over the cells the cap binds, 0.0317 over all 384 recordings). Scene stride selects
which recordings are scored and changes nothing about how one is scored — all 7,296 evaluable
curve points shared with the full run match it exactly at the 500,000-event cap — so the
subset was unrepresentative, not wrong.

**`--scene-stride 4` on E-MLB is deliberate.** It keeps 96 recordings balanced at 48 daylight
/ 48 night and 24 per ND level. Capping by `--max-recordings 96` instead would truncate inside
D-END and drop N-END entirely, because iteration runs part → scene → ND.

## Tier 3 — downstream validity (needs DVS Gesture + a GPU, ~2 h)

```bash
python -m dataset_assessment.downstream_gesture     # results/downstream_gesture.json
```

5 retentions x **3 seeds** (`20260726/7/8`) per method, with identical architecture, epochs,
batch size and optimiser in every one. `available_methods()` returns the eight the paper
reports; if the sibling checkout `../zero-parameter-event-denoising` is present it also
returns `native3d` and `native3d_gated`, giving 150 trainings instead of 120. Those rows are
kept in the artifact but excluded from every headline statistic — see `PAPER_METHODS` and the
`extended_scope` block. Pass `--methods raw random_null dwf evflow knoise red ts ynoise` to
run the paper's set only.

To recompute the statistics after changing how a summary is scoped or defined, without
retraining anything:

```bash
python -m dataset_assessment.downstream_gesture --from-artifact
```

**Self-check:** at `r = 1.0` all eight methods must return the *same* accuracy and the same
MESR to the last digit, because there every method is the unfiltered stream by construction.
If they differ, determinism has broken and nothing else in that file should be trusted.

**Read `coverage` in the output before quoting any accuracy.** A gesture that emits fewer
than `MAX_EVENTS` events cannot be evaluated at the lowest retention and is excluded, and
that exclusion is not class-neutral — low-activity gestures are lost first. The current
settings (whole gesture eligible, 80,000 events, `r >= 0.4`) keep 285/288 test gestures with
at least 92% of every class. An earlier version used the first 4 s and 160,000 events and
kept 205/288, with only 21% of class 1; the correlations were not qualitatively different,
but the task being measured was. If you change `MAX_EVENTS` or `RETENTIONS`, re-read
`coverage` — the two move together.

## Verifying a reported claim

Every numeric claim maps to an artifact through the digest, which is generated, not written:

```bash
python -m dataset_assessment.make_numbers_digest | less
```

If a reported number is not in that output, it has no artifact behind it.

## Equivalence checks worth re-running

**The vendored AEDAT-4 reader.** `dataset_assessment/aedat4.py` is copied verbatim from the
parent project so this release does not depend on that project's model code. Confirm it is
byte-identical wherever both are available:

```python
import hashlib
from pathlib import Path
from dataset_assessment.aedat4 import read_aedat4 as vendored
from native3d_ssm.phase5_emlb_mesr import read_aedat4 as original   # parent project

for p in sorted(Path("E-MLB/D-END").glob("*/*-ND00-1.aedat4"))[:2]:
    for cap in (None, 60_000):
        a, b = vendored(p, cap), original(p, cap)
        assert hashlib.sha1(a.tobytes()).hexdigest() == hashlib.sha1(b.tobytes()).hexdigest()
```

Verified here on 12 (file, cap) combinations across E-MLB and Pure_BA: all identical.

**The metric core against the official implementation.** `dataset_assessment/esr.py` was
checked against `cuke-emlb/python/src/utils/metric.py::EventStructuralRatio._calc_esr`
(maxdiff 0.00e+00). The invariance identity that follows from it is asserted in
`tests/test_esr.py::test_esr_is_invariant_to_the_declared_sensor_size`.

## Environment used

| | |
|---|---|
| OS | Ubuntu 22.04 (jammy) under WSL2, kernel 6.6.87.2, 20 cores, 31 GB RAM |
| Python | 3.13.9 (Anaconda) |
| numpy / scipy / pandas / scikit-learn / h5py | 2.3.5 / 1.16.3 / 2.3.3 / 1.7.2 / 3.15.1 |
| torch | 2.10.0+cu128, CUDA on GTX 1660 SUPER |
| dv-processing | 2.0.3 |
| dv_toolkit | 0.2.0, built from `yam-toolkit` @ `0fe3b3832187c07e09069d1df4006f4682268ab5` |
| compiler | **gcc-13 / g++-13 (13.4.0)** — dv-processing 2.0.3 rejects gcc < 13 |
| CMake | 3.22.1 |

Randomness is confined to three places, all seeded: `random_null`'s ranking
(`RANDOM_NULL_SEED = 20260726`), the downstream classifier (`SEEDS = 20260726/7/8`, with
`cudnn.deterministic = True`), and the bootstrap intervals in `analyze.py`
(`bootstrap_delta_ci(seed=20260726)`, and Figure 1's bands through the same seeded path at
2,000 draws instead of 10,000). Everything else is deterministic given the inputs.

**Two cohorts, and they are not interchangeable.** `per_method` in the comparison averages
over the cells the cap can bind — the recordings longer than the smallest cap, times six
filters — which is the right denominator for "how large is the cap effect where the cap acts".
`means_over_all_recordings` averages over all 384 recordings, which is what the article's
E-MLB table prints; its 1,000,000-event column reproduces that table to the printed digit, and
`tests/test_cap_sensitivity.py::test_the_1m_column_reproduces_the_published_emlb_table` pins
it. Quoting the first as though it were the second overstates the movement, by 0.0395 against
0.0317 on RED.
