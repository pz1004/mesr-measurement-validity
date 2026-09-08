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
(`bootstrap_delta_ci(seed=20260726)`). Everything else is deterministic given the inputs.
