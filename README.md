# Discarding Events Is Not Denoising

**A Measurement-Validity Study of the Event Structural Ratio**
Sooyoung Jang (Hanbat National University) and Kyuseung Han (ETRI)

Event-camera denoisers are compared by MESR, the mean Event Structural Ratio, a no-reference
score introduced with the E-MLB benchmark. This repository holds the code and the output of
every run behind a study that tests, on six datasets, four validation criteria and whether MESR preserves event-label dominance, and a reporting protocol for papers that use it.

It is the artifact cited in the paper's Data availability statement:
<https://github.com/pz1004/mesr-measurement-validity>

## Findings

On the events MESR scores from their own outputs, filters out-score a label oracle that keeps
as many events from each input block, signal first, in 76 of 81 (method, recording) cells, while those events hold less signal and more noise, so they are worse on precision, recall, F1 and false-positive rate alike. A seeded random subsampler, which selects nothing, raises MESR at all 19
retentions on two of the three real corpora as released and lowers it on the third. Against the
released DVS Gesture classifier, the quota-adapted filter MESR ranks first is the least accurate at three of four retentions on the classified stream and at two on the events MESR scores.

| Requirement | Test | What MESR returns |
|---|---|---|
| Event-label dominance (direct test) | Each filter's own scored output against a label oracle that keeps signal first, matched to it per block | **Not preserved.** Strict reversals (higher score, less signal and more noise, so worse on precision, recall, F1 and false-positive rate alike) in 52/55 DND21 and 24/26 DVSCLEAN cells, 76 in all; 75 with the busiest 0.1% of occupied pixels removed first |
| Specificity (null response) | A random subsampler and a block-prefix control on a common retention axis | **Fails as released.** Random removal raises MESR at 19/19 retentions on DVSD22 and Pure_BA, by up to +0.7839, under each of ten draws and with both streams scored over the same input span; it lowers MESR on E-MLB and DND21. With the busiest pixels removed: 5/19 and 0/19 |
| Blank-sample response | Quota-adapted filters against the random subsampler at equal count, on a nominally blank capture | **Fails as released.** The best of six filters gains +1.35 against the subsampler's +0.05; with the busiest pixels removed, the lead falls from +1.1291 to +0.0111 on the 25 recordings evaluable both ways |
| Ruggedness: operating point | Retention swept on fixed synthetic event scores (ROC AUC 0.93 before the block quota) | **A single value does not identify selectivity.** Retention moves MESR by 0.3029 under fixed event scores; the published tables report one value per method, at the operating point each chooses |
| Ruggedness: slice convention | Slice size swept from 10⁴ to 10⁵ events on a fixed output | **Comparisons need one stated convention.** On DND21 it moves MESR by 0.5221, over four times the widest gap between two filters' own outputs (0.1207), both taken on each recording's first 10⁶ events, and reorders filters on 9 of 10 recordings; both published tables cite the benchmark that sets it |
| Criterion validity | DVS Gesture: the task's released classifier, frozen, and three architectures trained under three seeds | **Counterexample.** With the released classifier (0.9792, its published accuracy), quota-adapted RED, first by MESR at every retention below 1, is the least accurate at three of four on the classified stream and at two on the events MESR scores, trails the block-prefix control at r = 0.6 by 0.1181 and 0.1111 respectively on each of the six test subjects, and stays least accurate there under each of ten random tie-breaks of the quota. For the trained classifiers the association depends on the architecture (MLP ρ = +0.882; CNNs +0.362 and +0.275, over the 33 distinct conditions) |
| Published known-input check | Noise added to a fixed stream, re-run on DND21's injected series | **Reproduces.** MESR falls as injected noise rises |
| Declared grid size | Cancellation proof, and one stream re-scored at four declared resolutions | **Invariant.** The pixel count K = W·H cancels; spread 2.1×10⁻¹⁴ |

*Quota-adapted* filters keep the same count per 10,240-event input block as the controls, so
their retained set is not the filter's own. A filter's own output is its *native* output.

On E-MLB, EDmamba's released test script does not compute MESR and its loader reads one of the
four ND levels, so the released code does not regenerate the published EDformer–EDmamba
difference of 0.0092.

Every number above is in the paper, and `python -m dataset_assessment.make_numbers_digest`
maps each to the output that produces it. The manuscript itself is not part of this release.

## Why removal alone can move the score

ESR scores a slice of N = 30,000 events, with M = 20,000 and n_p events at pixel p of K:

```
ntss = Σ_p n_p (n_p − 1) / (N (N − 1))
ℓn   = K − Σ_all p (1 − M/N)^n_p
     = Σ_occupied p [1 − (1 − M/N)^n_p]      # an empty pixel contributes exactly 1: K cancels
ESR  = √(ntss · ℓn)
```

MESR is the mean of ESR over a recording's complete slices. A slice holds a fixed count of
events, so a filter that keeps half of them is scored over a window about twice as long.
Whether that moves the score, and in which direction, depends on the stream. Under stationary
Poisson rates it moves nothing, and concentrating a slice's counts raises ntss but lowers ℓn. On
DVSD22 and Pure_BA random removal raises MESR, and removing the busiest 0.1% of occupied pixels
first removes almost all of that gain; on E-MLB and DND21 it lowers MESR. Shuffling each
recording's pixel and polarity marks over its own timestamps keeps every pixel's count and
removes the effect in both directions, so it needs the order of events on scales longer than a
slice. The paper states its account of why, a stationary hot component set against a drifting
one, as a hypothesis.

## Quick start

```bash
pip install -r requirements.txt
python -m dataset_assessment.analyze            # rank_analysis.json + Figure 1
python -m dataset_assessment.figures_protocol   # the protocol figures and the oracle scatter
python -m pytest -q tests/                      # expect: no failures
python -m dataset_assessment.audit              # expect: 10/10 claims verified from source
```

None of these needs a corpus. Without the corpora, 430 tests pass and 14 skip: 8 read the
corpora, 3 check the manuscript, 1 needs EDformer's score cache, 1 the cuke-emlb build and 1
the audited checkouts. `audit` reads the EDformer,
EDmamba and cuke-emlb sources at pinned commits, so it needs those checkouts. To regenerate
everything from raw event data, see [`REPRODUCE.md`](REPRODUCE.md); to obtain the corpora, see
[`DATA.md`](DATA.md).

## What is here

| Path | Contents |
|---|---|
| `dataset_assessment/` | The harness: the metric core, checked against E-MLB's reference implementation; readers for every corpus used; the denoiser adapter, which attributes each filter's kept events to its input exactly, even where the input repeats an event verbatim; the grid runner; and one module per analysis |
| `dataset_assessment/sew7b/` | The released DVS Gesture classifier's network, its evaluator and the frame integrator, used by `downstream_gesture_frozen` (the checkpoint is third-party; see `DATA.md`) |
| `tests/` | 444 tests, including the ESR closed-form hand case, the K-cancellation identity and the denoiser adapter's packet-wise attribution of kept events |
| `results/` | 51 JSON artifacts, with `null_seeds/` (the subsampler under ten draws) and `cap_sensitivity/`; every reported number traces to one (`benchmark_emlb_native3d.json` belongs to a sibling project and backs none) |
| `reproduction/results/` | Our uncapped run of EDformer's released evaluation script, which verifies its published E-MLB table |
| `BUILD_CUKE_EMLB.md` | How the six classical denoisers were built, with verbatim errors and the pinned-commit warning |
| `REPRODUCE.md`, `DATA.md` | The command, wall time and environment for every artifact; where each corpus, baseline and checkpoint comes from |

Datasets, third-party baselines and the classifier checkpoint are **not redistributed**; in the
authors' working tree they are symlinks to sibling checkouts (`DATA.md`).

## Reporting protocol

A paper that reports MESR should print four items. They make a MESR number interpretable; they
do not make MESR valid.

1. **What the method is.** A fixed binary filter has one operating point: report its score on
   its native output, the retention it achieves, whether it is evaluable, the matched nulls,
   and the scoring convention (N, M, event cap, hot-pixel rule, incomplete-slice handling, and the nulls' block length and tie-break). A
   method with genuine configurations gets a curve over those: MESR at a stated grid of
   retentions, summarised by (r\*, MESR\*) and by AUC_r over a stated span. r\* is the argmax
   of the test metric, so it bounds what a method could reach; rank on AUC_r over a common
   range, or on the value at the method's own operating point. Either order is an order of MESR
   responses, not of denoising quality, until MESR is validated for that use.
2. **Δ-over-Raw beside every absolute number.** It removes the recording's own baseline, though
   not the event cap's effect.
3. **The (ntss, ℓn, #occupied) decomposition**, so a score difference can be traced to its
   numerical contributors, though not to the sensor or denoiser that produced them. No
   grid-size normalisation applies, since K cancels.
4. **The nulls, matched to the comparison.** Against a native output, the random subsampler
   and, where labels exist, the label oracle take the scored output's own count from every
   input block, so achieved retention varies by recording. On a retention sweep, the
   block-prefix control, the random subsampler and the oracle share the methods' prespecified
   grid and a common evaluable domain, read out as the count of retentions at which a null
   gains. A method that does not beat the nulls has not been shown to denoise; one that beats
   them has been shown to select within blocks, not to denoise. Never choose a recording's operating point by
   maximising the score reported.

## Limits

- The study evaluates the released implementations at their default parameters, as a reader
  reproducing a published table would; a filter that underperforms here may be misconfigured
  in the reference benchmark rather than weak.
- Binary filters have no retention parameter, so their curves are quota-adapted constructions;
  only their native outputs are the filters' own.
- The downstream probe is one task. The three trained architectures reach 0.4503, 0.5146 and
  0.4082 unfiltered, far below what the task allows, and three seeds bound their variability
  without removing it; the frozen released classifier has neither limitation.
- One filter, RED, dominates several results. The RED-excluded reading is reported wherever it
  changes a verdict; without RED, the frozen classifier's mean within-retention ρ is +0.45 and
  no retention inverts.
- Four corpus statistics leave open whether injected noise resembles real background activity:
  none clears the separation bar set in advance. Withholding E-MLB makes busiest-pixel share
  separate the groups perfectly, but a grouping that ignores noise origin separates about as
  sharply.
- No replacement metric is proposed. The paper states the requirements one would have to meet.

## How to cite

The paper is under review at *IEEE Transactions on Instrumentation and Measurement*; this
block will be updated with volume and pages on acceptance.

```bibtex
@unpublished{jang2026mesr,
  title  = {Discarding Events Is Not Denoising: A Measurement-Validity Study of the
            Event Structural Ratio},
  author = {Jang, Sooyoung and Han, Kyuseung},
  year   = {2026},
  note   = {Under review, IEEE Transactions on Instrumentation and Measurement},
  url    = {https://github.com/pz1004/mesr-measurement-validity}
}
```

If you use the harness rather than the findings, cite the repository directly at the release
tag you ran, so the numbers you quote are the ones that version produces.

## License

Code and documents in this repository: MIT (see `LICENSE`). Datasets, third-party baselines and
the classifier checkpoint keep their original authors' terms; none are redistributed here.
