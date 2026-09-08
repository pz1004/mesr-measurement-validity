# Datasets and third-party baselines

Nothing in this section is vendored here. The corpora total roughly 250 GB and are
redistributed by their original authors under their own terms; the baselines are separate
repositories with their own licences. In this working tree each is a **symlink to a sibling
directory**, so the harness resolves them without copying:

```
E-MLB -> ../E-MLB              DVSGesture -> ../DVSGesture      cuke-emlb -> ../cuke-emlb
DVSCLEAN -> ../DVSCLEAN        DVSD22 -> ../DVSD22              EDformer -> ../EDformer
ECCV2024_datasets -> ../ECCV2024_datasets                       EDmamba -> ../EDmamba
```

Replace each symlink with a real checkout (or re-point it) to run elsewhere. Every reader
takes an explicit `root=` argument, so nothing is hard-coded beyond these defaults.

## Corpora

| Name | Resolution | Labels | Real/synthetic | Used for | Expected path |
|---|---|---|---|---|---|
| **DND21** | 346×260 | per-event | synthetic injection, 1–10 Hz/px | retention & slice studies, oracle test | `ECCV2024_datasets/AUC_test/{1,3,5,7,10}hz/*.txt` |
| **ED24** | 346×260 | per-event | synthetic, 41 levels × 100 scenes | reader only; no experiment in this paper | `ECCV2024_datasets/ED24/<Scene>/<Scene>_<level>.csv` |
| **DVSCLEAN** | 1280×720 | per-event | simulated, 50 %/100 % | second labelled corpus; the null result | `DVSCLEAN/simulated_data/<seq>_{50,100}.hdf5` |
| **E-MLB** | 346×260 | none | **real**, 4 ND levels × 2 lighting | the main benchmark, 96 recordings | `E-MLB/{D-END,N-END}/<Scene>/<Scene>-ND{00,04,16,64}-<rep>.aedat4` |
| **Pure_BA** | 346×260 | none (all noise) | **real**, signal-free | the specificity control | `ECCV2024_datasets/Pure_BA_noise/<level>-<stamp>.aedat4` |
| **DVSD22** | 346×260 | none | **real**, falling water drops | second real corpus; retention-floor result | `DVSD22/Recordings/<session>/*.aedat` |
| **DVS Gesture** | 128×128 | class only | **real** | downstream-validity probe | `DVSGesture/DvsGesture/*.aedat` + `*_labels.csv` |

### Provenance and citations

- **E-MLB** — Ding, Chen, Wang, Kang, Song, Cheng, Cao, *E-MLB: Multilevel Benchmark for
  Event-Based Camera Denoising*, IEEE TMM 26:65–76, 2024 (`ding2024emlb`).
- **DND21** — distributed with Guo & Delbruck, *Low Cost and Latency Event Camera Background
  Activity Denoising*, IEEE TPAMI, 2022 (`guo2023lowcost`).
- **DVSCLEAN, ED24, Pure_BA** — distributed with prior denoising work; cited in the manuscript.
- **DVS Gesture** — Amir et al., *A Low Power, Fully Event-Based Gesture Recognition System*,
  CVPR 2017 (`amir2017gesture`).
- **DVSD22** — Micev, Steiner, Aydin, Rieckermann & Delbruck, *Measuring diameters and
  velocities of artificial raindrops with a neuromorphic event camera*, Atmospheric
  Measurement Techniques 17(1):335–357, 2024, [doi:10.5194/amt-17-335-2024](https://doi.org/10.5194/amt-17-335-2024)
  (`micev2024raindrops`). Released as **“DVSD22 — Dynamic Vision Sensor Disdrometer 2022”**
  at [sensors.ini.ch/datasets](https://sensors.ini.ch/datasets); project page and download at
  [sites.google.com/view/dvs-disdrometer](https://sites.google.com/view/dvs-disdrometer/home),
  which asks that the paper be cited.

  **Do not read the name as a member of the DND denoising family.** This is a drop-metrology
  release with **no per-event noise labels**, and its source paper applies **no denoising or
  noise filtering of any kind** to the recordings. That is exactly why we use it — a second
  real 346×260 corpus whose acquisition choices were made with no denoising benchmark in view.
  The directory was named `DND22/` until 2026-09-07; older working notes use that path and
  the placeholder corpus name `DROPLETS22`.

### A note on DVSD22 and downstream validity

We initially planned the downstream probe on DVSD22 because it ships physical ground
truth. That does not work, and its release paper says why: droplets were *"manually
selected"* and measured by hand with jAER's `Speedometer` plugin (Micev et al. 2024). Two
automatic estimators were built and both fail on the **unfiltered** stream — fall velocity by two-row time-of-flight
(−35 % to +6791 % error) and drop rate by periodogram (within 10 % on 3 of 11 recordings).
A downstream probe whose estimator fails at `r = 1.0` measures the estimator, not the
denoiser. The probe therefore runs on DVS Gesture.

## Third-party baselines

| Repository | Why it is needed | Notes |
|---|---|---|
| **cuke-emlb** | Supplies the six classical denoisers (DWF, EvFlow, KNoise, RED, TS, YNoise) and the official ESR implementation the metric core was checked against | Must be **built**; see `BUILD_CUKE_EMLB.md`. Read the pinned-commit warning there — the pinned `dv-toolkit` submodule does not work. |
| **EDformer** | `dataset_assessment/audit.py` greps its released source; its verified per-cell E-MLB numbers are the published-protocol reference row | Not re-run here — it needs its own pinned environment. Numbers come from `reproduction/results/emlb_edformer_mesr.json`. |
| **EDmamba** | `audit.py` greps its released source for the four Table-2 reproducibility claims | Not run at all; only inspected statically. |

`EDnCNN` and `MLPF` are absent from the benchmark because the cuke-emlb release ships no
weights for them — `modules/net/` contains only `.gitkeep`. That is itself one of the nine
audited claims.

## Reference values this project relies on

`reproduction/results/emlb_edformer_mesr.json` (vendored, 1.3 MB) holds EDformer's per-cell
E-MLB MESR from an independent reproduction that matched the published table to **+0.0010**
(mean 1.0069 against 1.0059), at its released `sigmoid ≥ 0.005` threshold with mean retention
0.081–0.483. It is the only external numeric input to the analysis, and it is compared on
Δ-over-Raw rather than absolute MESR because the two runs use different event caps.
