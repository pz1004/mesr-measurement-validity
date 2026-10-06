# Reproducing every artifact

Run in the order below. Each stage writes JSON to `results/` and is independent of the
later ones, so a failure never silently corrupts a downstream number. Wall times are the
measured values from the run that produced the shipped artifacts, on 20 cores + one
GTX 1660 SUPER.

## Tier 0 — no dataset needed (~35 s total)

```bash
pip install -r requirements.txt
python -m dataset_assessment.analyze                # rank_analysis.json + results/figures/
python -m dataset_assessment.figures_protocol       # the two protocol figures
python -m pytest -q tests/                          # 364 passed with every checkout in place
python -m dataset_assessment.audit                  # 10/10 claims verified from source
python -m dataset_assessment.make_numbers_digest > NUMBERS.md
```

`analyze` needs `results/benchmark_*.json`, which ship with this repository; the figure tests
read what it and `figures_protocol` draw, so run those first. Without the corpora, 12 tests
skip and none fails. `audit` needs the `EDformer/`, `EDmamba/` and `cuke-emlb/` symlinks
(source inspection only, nothing is executed) and writes `results/reproducibility_audit.json`.

`analyze` also writes **Figure 1** (`results/figures/fig_retention_curves.pdf`). It is drawn
at the size it is printed at -- 516 x 63 pt, the full text width of a two-column IEEEtran
`figure*` -- and included with `width=\linewidth`, so nothing downscales the type. The float
it sits in is capped at 101.2 pt, which is that graphic plus a three-line caption; the cap is
asserted in `tests/test_figures.py::test_figure_fits_the_float_budget`, because growing the
figure silently costs a page at $265. Bands are `cluster_bootstrap_delta_ci` at 2,000 draws
on the same seed as the reported intervals, and the strip under each panel is `n(r)`,
asserted equal to `common_support.coverage`.

## Tier 1 — metric analysis (needs the five corpora, ~4 min)

```bash
python -m dataset_assessment.measure_metric_deps    # results/metric_dependencies.json
python -m dataset_assessment.profile_datasets       # results/noise_regimes.json  (211 s)
python -m dataset_assessment.nd_levels              # results/nd_levels.json  (<1 s)
```

`profile_datasets` reads all five corpora at a 300,000-event cap; `nd_levels` reads its output
and tests what E-MLB's neutral-density levels do to busiest-pixel share (App. J).

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
    --methods raw random_null dwf evflow knoise red ts ynoise                   # 4880 s
python -m dataset_assessment.run_benchmark --dataset pure_ba  --max-events 1000000 \
    --methods raw random_null dwf evflow knoise red ts ynoise                   # 3420 s
python -m dataset_assessment.run_benchmark --dataset dvsd22 --max-events 1000000 \
    --methods raw random_null dwf evflow knoise red ts ynoise                   #  878 s
python -m dataset_assessment.analyze                                            # re-aggregate
```

Every run writes its JSON incrementally, one recording at a time, so a killed job keeps its
partial output and can be inspected.

### Scene intervals for the nulls

Whether each null's gain exceeds scene-resampling variation at every retention, with the
Bonferroni-corrected count (Sec. V-B). Reads the five `results/benchmark_*.json`; a few seconds.

```bash
python -m dataset_assessment.null_intervals   # results/null_intervals.json
```

### The busiest 0.1% of pixels removed first

Specificity and the blank-sample response fail only on DVSD22 and Pure_BA. These runs repeat
both with the busiest 0.1% of occupied pixels removed from the input first; the comparison reads
each pair of runs on the recordings evaluable under both (Sec. V-B, Sec. V-C).

```bash
python -m dataset_assessment.run_benchmark --dataset dvsd22  --hot-pixel-removal \
    --methods raw random_null --out results/benchmark_dvsd22_hotpixel.json          #   20 s
python -m dataset_assessment.run_benchmark --dataset pure_ba --hot-pixel-removal \
    --methods raw random_null --out results/benchmark_pure_ba_hotpixel.json         #   36 s
python -m dataset_assessment.run_benchmark --dataset pure_ba --hot-pixel-removal \
    --methods raw random_null dwf evflow knoise red ts ynoise \
    --out results/benchmark_pure_ba_hotpixel_all.json                               # 2266 s
python -m dataset_assessment.hotpixel_contrast   # results/hotpixel_contrast.json
```

### The filters' own outputs

The label-quality test and the E-MLB table read each filter's native output, not a quota near
it. `native_oracle` scores every filter against a label oracle matched to it per block on the
two labelled corpora, as released and with the busiest pixels removed; `native_emlb` applies
protocol item 1 to E-MLB; `paired_contrasts --source native` differences its rows within
recordings.

```bash
python -m dataset_assessment.native_oracle --dataset dnd21                          #   69 s
python -m dataset_assessment.native_oracle --dataset dvsclean                       #   79 s
python -m dataset_assessment.native_oracle --dataset dnd21    --hot-pixel-removal   #   70 s
python -m dataset_assessment.native_oracle --dataset dvsclean --hot-pixel-removal   #  612 s
python -m dataset_assessment.native_emlb --workers 16                               #   37 s
python -m dataset_assessment.paired_contrasts --dataset emlb --source native        # paired_contrasts_native.json
```

### Slice size on the filters' own outputs

Whether the slice-size convention reorders filters or shifts every score together, on DND21
(Sec. V-E).

```bash
python -m dataset_assessment.slice_ranking   # results/slice_ranking_dnd21.json  70 s
```

### EDformer

The two `edformer` commands run in EDformer's own environment (Python 3.10, torch 2.2.1+cu118,
pytorch3d 0.7.5; `DATA.md`). The first caches its per-recording scores on the capped E-MLB
inputs, which the rest of the harness reads; the last follows EDformer's released protocol,
uncapped, so its per-cell MESR is comparable with the published table.

```bash
python -m dataset_assessment.edformer --cap 1000000      # results/edformer_emlb_manifest.json  6639 s
python -m dataset_assessment.run_benchmark --dataset emlb --max-events 1000000 \
    --methods raw edformer_native --out results/benchmark_emlb_edformer.json      #  429 s
python -m dataset_assessment.native_emlb --methods edformer_native --merge
python -m dataset_assessment.edformer --cap 0 --out results/edformer_emlb_uncapped.json   # 22478 s
```

### MESR repeatability across E-MLB's repetitions

E-MLB records every (scene, ND) cell three times under nominally identical conditions; every
other run here reads the first, as the published tables do. This scores all three and reports
the within-cell spread — a Type A repeatability for the statistic (App. H).

```bash
python -m dataset_assessment.emlb_repeatability --workers 16   # results/emlb_repeatability.json  ~81 s
```

Needs no denoiser build: every reading is the unfiltered stream.

### How much of the null's gain is its draw

`random_null` is seeded, so every number the paper quotes for it reads one realisation per
recording, and the scene bootstrap behind those numbers resamples recordings rather than the
selector. This re-scores the null under ten draws on the two corpora carrying the specificity
and blank-sample verdicts, and reports the across-draw spread of the quantities the main text
states (Sec. V-B, Sec. V-C).

```bash
for S in 20260726 20260727 20260728 20260729 20260730 \
         20260731 20260732 20260733 20260734 20260735; do
  for D in dvsd22 pure_ba; do
    python -m dataset_assessment.run_benchmark --dataset "$D" --methods random_null \
           --null-seed "$S" --out "results/null_seeds/${D}_s${S}.json"
  done
done
python -m dataset_assessment.null_seed_sensitivity   # results/null_seed_sensitivity.json  <1 s
```

The grid runs take about 10 min in total; the aggregator is instant. The filters do not depend
on the seed, so the blank-sample lead reads their side from `benchmark_pure_ba.json` and varies
only the control. Seed 20260726 is the released default and reproduces the published
`+1.3052`, which is the check that the aggregator matches the paper's definition.

### The null's gain on a matched span

MESR scores whole slices, so at a low retention the subsampler's scored events come from a
shorter leading span of the input than the unfiltered stream's. This scores both on one leading
frame that leaves neither an incomplete slice, under the same ten draws on all five corpora
(App. M).

```bash
python -m dataset_assessment.null_matched_span --workers 16   # results/null_matched_span.json  ~24 min
```

### Matched-retention label quality

Whether a filter that out-scores the label oracle on MESR actually selects events better.
Needs per-event labels, so it runs on the two labelled corpora only.

```bash
python -m dataset_assessment.label_quality --dataset dnd21     # results/label_quality_dnd21.json     ~108 s
python -m dataset_assessment.label_quality --dataset dvsclean  # results/label_quality_dvsclean.json  ~125 s
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

The `own_range` column reproduces the published values exactly (`tab:support`); `common_grid` and
`common_cohort` are the two repairs. It also reports `n(r)` per corpus, which is what makes a
shrinking sample at low retention visible.

### Paired contrasts on the E-MLB ranking

What the E-MLB ranking table (`tab:emlb`) can carry. Every classical row is scored on the
same recordings wherever both reach their own operating point, so the comparison is paired;
this differences within each recording and bootstraps the scene clusters. Reads
`results/benchmark_emlb.json`; a few seconds.

```bash
python -m dataset_assessment.paired_contrasts --dataset emlb   # results/paired_contrasts.json
```

Differencing within recordings avoids reading the overlap of two marginal intervals, which is
invalid in both directions. 12 of the 21 pairs separate; the nine that do not all lie among the
five middle methods (YNoise, EDformer, DWF, EvFlow, TS), of which only YNoise and DWF separate.
Both this and Table IV apply
`protocol.native_point_is_eligible`, which drops the 445 of 2304 E-MLB cells whose native
retention is below the recording's measurable floor -- cells the sweep can only score at a
retention the filter never chose.

### Is the swept curve at native the released filter?

No, and `native_mask` measures the gap. `esr.retain_mask` keeps a fixed count per
10,240-event block while a binary filter's accepted count varies by block, so the two
retained sets differ by `sum_b |a_b - k_b(r)|`. Held at each filter's exact native rate, the
median cell differs on about 25% of its retained events and not one of 168 cells recovers the
filter exactly.

```bash
python -m dataset_assessment.native_mask --dataset emlb --recordings 24   # ~3 min
python -m dataset_assessment.native_mask --dataset dnd21 --recordings 4
```

### How far off is the unused ESR variant?

Not by orders of magnitude: the `/K` in `EventStructuralRatioV2` cancels most of its `1000`,
and what remains is a size-3 median filter that erases isolated pixels. Needs the corpora;
about a minute.

```bash
python -m dataset_assessment.esr_variant     # results/esr_variant.json
```

It reports median ratios to the official value of 3.27x on E-MLB, 2.13x on DND21, 1.03x on
DVSCLEAN, 0.003x on Pure_BA and 0.0001x on the sparse DVSD22 slices, three of which collapse to
exactly zero.

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

**`--scene-stride` is omitted deliberately: the sweep needs all 384 recordings.** A
48-recording sweep at `--scene-stride 8`, summarised in `results/cap_sensitivity_subset48.json`
(its per-cap runs are not shipped), does not support the paper's numbers. Two of its
conclusions are false at full scale: it finds the evaluable range moving on every cell, where
294 of 2688 hold still, and every corpus mean steady between 1M and 2M, where RED's passes the
unit of scale on both cohorts (0.0395 over the cells the cap binds, 0.0317 over all 384
recordings). Scene stride selects which recordings are scored and changes nothing about how one
is scored, so the subset was unrepresentative, not wrong.

**For a quick E-MLB check, use `--scene-stride 4`, not `--max-recordings`.** It keeps 96
recordings balanced at 48 daylight / 48 night and 24 per ND level; capping by
`--max-recordings 96` would truncate inside D-END and drop N-END entirely, because iteration
runs part → scene → ND. The shipped `results/benchmark_emlb.json` scores all 384.

## Tier 3 — downstream validity (needs DVS Gesture + a GPU, ~10 h in all)

```bash
python -m dataset_assessment.downstream_gesture     # results/downstream_gesture.json  2339 s
python -m dataset_assessment.downstream_gesture --arch cnn3d \
    --out results/downstream_gesture_cnn3d.json                                  # 28335 s
python -m dataset_assessment.downstream_gesture --arch mlp \
    --out results/downstream_gesture_mlp.json                                    #  1863 s
python -m dataset_assessment.downstream_paired      # results/downstream_paired_seeds.json  <1 s
```

The default architecture is the 2D CNN. Each trains 5 retentions x **3 seeds**
(`20260726/7/8`) per method, with identical architecture, epochs,
batch size and optimiser in every one. `available_methods()` returns the eight the paper
reports; if the sibling checkout `../zero-parameter-event-denoising` is present it also
returns `native3d` and `native3d_gated`, giving 150 trainings instead of 120. Those rows are
kept in the artifact but excluded from every reported statistic — see `PAPER_METHODS` and the
`extended_scope` block. Pass `--methods raw random_null dwf evflow knoise red ts ynoise` to
run the paper's set only.

### The released classifier, frozen

`downstream_gesture_frozen` scores the same conditions with the task's released classifier, the
SEW 7B-Net in `dataset_assessment/sew7b/`, held frozen, so no seed or training schedule enters.
Its evaluator runs in an environment with torch and SpikingJelly 0.0.0.0.15, named by
`SNN_PYTHON`, and reads the released checkpoint from `sew7b-checkpoint/` (`DATA.md`). The
module stops unless the 288 unfiltered test clips reproduce the published 0.979167 top-1 (282 of
288) and the 0.977321 macro-F1 that cell has; `--smoke` runs that gate alone and writes
`results/downstream_gesture_frozen.smoke.json`.

```bash
export SNN_PYTHON=/path/to/snn-env/bin/python
python -m dataset_assessment.downstream_gesture_frozen --smoke   # the gate alone, ~30 s
python -m dataset_assessment.downstream_gesture_frozen           # results/downstream_gesture_frozen.json  1620 s
cp results/downstream_gesture_frozen_frames/evaluation.json \
   results/downstream_gesture_frozen_evaluation.json
```

The evaluator writes its per-clip predictions and the paired comparisons, bootstrapped over the
six held-out subjects (10,000 draws), beside the frames; the last line ships them. Run the gate
before the full run, not after: both write `evaluation.json` to the same frames directory, and
the copy must take the full run's.

The evaluator always runs in full precision (`--no-amp`). In half precision, on a GTX 1660 SUPER,
rounding near the firing threshold flips spikes and changes four unfiltered predictions, three
of them to wrong ones (0.968750); the released training script used half precision on other
hardware.

### How much of the downstream verdict rests on RED

RED holds the highest MESR at every partial retention of the frozen probe and alone produces
the ranking inversion. This recomputes the Sec. V-F correlations with and without it, from the
shipped artifacts rather than from the tables that print them.

```bash
python -m dataset_assessment.red_sensitivity    # results/red_sensitivity.json  <1 s
```

Reads `downstream_gesture.json` and `downstream_gesture_frozen.json`; reproduces the published
`+0.135` (trained 2D CNN, 40 conditions) and `+0.036` (frozen, within retention) exactly.

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
at least 92% of every class. If you change `MAX_EVENTS` or `RETENTIONS`, re-read `coverage` —
the two move together.

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
| classifier environment (`SNN_PYTHON`) | Python 3.12.12, torch 2.10.0+cu128, SpikingJelly 0.0.0.0.15, numpy 1.26.4, scipy 1.17.1, scikit-learn 1.8.0 |
| EDformer environment | Python 3.10.12, torch 2.2.1+cu118, pytorch3d 0.7.5 |
| dv-processing | 2.0.3 |
| dv_toolkit | 0.2.0, built from `yam-toolkit` @ `0fe3b3832187c07e09069d1df4006f4682268ab5` |
| compiler | **gcc-13 / g++-13 (13.4.0)** — dv-processing 2.0.3 rejects gcc < 13 |
| CMake | 3.22.1 |

Every random draw is seeded. `random_null`'s ranking uses `RANDOM_NULL_SEED = 20260726`
(`--null-seed` overrides it; `results/null_seeds/` and `null_matched_span` use
20260726–20260735); the random control on E-MLB's native outputs uses `CONTROL_SEED = 20260917`
(`native_emlb`); the trained classifiers use `SEEDS = 20260726/7/8` with
`cudnn.deterministic = True`; EDformer keeps its released seed (230086), and the frozen
classifier's evaluator seeds its run and its subject bootstrap from 1024. Bootstrap intervals
draw from generators seeded with 20260726 (`analyze.py`, `measure_metric_deps.py`; 10,000
draws, and 2,000 for Figure 1's bands), and the synthetic streams in `esr_properties` and
`measure_metric_deps` use seed 0. Everything else is deterministic given the inputs.

**Two cohorts, and they are not interchangeable.** `per_method` in the comparison averages
over the cells the cap can bind — the recordings longer than the smallest cap, times six
filters — which is the right denominator for "how large is the cap effect where the cap acts".
`means_over_all_recordings` averages over all 384 recordings, which is what the article's
E-MLB table prints; its 1,000,000-event column reproduces that table to the printed digit, and
`tests/test_cap_sensitivity.py::test_the_1m_column_reproduces_the_published_emlb_table` pins
it. Quoting the first as though it were the second overstates the movement, by 0.0395 against
0.0317 on RED.
