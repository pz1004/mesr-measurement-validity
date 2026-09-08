# The zero-parameter method under this paper's E-MLB protocol

Date: 2026-07-29
Artifact: `results/benchmark_emlb_native3d.json`
Command:

```bash
python -m dataset_assessment.run_benchmark --dataset emlb \
    --methods raw random_null native3d native3d_gated \
    --max-events 1000000 --scene-stride 4 \
    --out results/benchmark_emlb_native3d.json
```

Same 96 recordings, same cap and stride as `results/benchmark_emlb.json`.

## Validation

`raw` and `random_null` reproduce the existing artifact **exactly** — max
`|Δ MESR*| = 0.00e+00` across all 96 recordings — so this run is a faithful extension of
the published protocol, not a re-derivation under different settings.

The `native3d` score is **byte-identical** to the sibling project's own scorer: verified by
calling `native3d_ssm.phase10_emlb_blind.score_stream_both` directly on the same events and
comparing arrays (`np.array_equal` → `True`).

## Result

| Method | MESR* | r* | Δ-over-Raw | AUC_r | native r |
|---|---:|---:|---:|---:|---:|
| `raw` | 0.8161 | 0.82 | +0.0071 | 0.7656 | 1.000 |
| `random_null` | 0.8158 | 0.80 | +0.0069 | 0.7734 | — |
| `native3d_gated` | 0.8460 | 0.77 | +0.0370 | 0.7941 | 0.896 |
| **`native3d`** | **1.1088** | **0.14** | **+0.2999** | **0.9346** | — |

Unfiltered raw MESR mean = 0.8089.

## Readings

**1. The method beats the nulls decisively.** On `AUC_r`, the oracle-free summary this
paper's protocol says rankings should use, `native3d` scores **0.9346** against
`random_null`'s 0.7734 and `raw`'s 0.7656 — **+0.1612** over the null. On Δ-over-Raw at r*
it is **+0.2999 against the null's +0.0069, a factor of 43**. The E-MLB result is not a
discarding artifact; the score genuinely discriminates.

**2. `native3d` is the first method here with a real curve.** All six classical filters are
binary, so their `AUC_r` is dominated by the trivial endpoints. `native3d` is a continuous
score, so its `AUC_r` is a genuine summary over the retention grid. This is the protocol
working as intended: it separates methods that can be swept from methods that cannot.

**3. The method's own operating point is far from the MESR-optimal one — and that is a
point in its favour.** `native3d_gated` applies the method's frozen `gated_retention` rule,
which keeps **89.6%** of events and gains only +0.0370. The MESR argmax sits at
**r\* = 0.14**, discarding 86%. The sibling's published E-MLB row uses a frozen
**r = 0.28**, pre-registered from dev MESR before the E-MLB run and never fitted on E-MLB.

So the method leaves most of the available MESR gain on the table. Whatever it reports on
E-MLB is **not** obtained by exploiting the metric's bias toward discarding events — the
configuration that would exploit it (r ≈ 0.14) is not the one it uses. Under §3.2 of this
paper that is the distinction between a method and a metric-oracle, and the method lands on
the right side of it.

**4. The gap between `native3d` and `native3d_gated` is the honest caveat.** The score
ranks well (AUC_r 0.9346); the self-selected operating point captures little of that
(0.7941). The sibling project already tracks this as "retention drift" (its Phase 9). It
should be stated, not smoothed over: the ranking quality and the retention rule are
separate claims, and only the first is strong here.

## What this does not show

- E-MLB is the corpus where MESR behaves **best**: the null gains only +0.0069 here, against
  +0.8199 on DVSD22. Beating the null on E-MLB is therefore a weaker result than beating
  it on DVSD22 would be, and does not license a general claim about the method.
- Absolute MESR here is **not** comparable with the sibling's Table 2 (mean 1.004). That
  table uses uncapped streams and its own slice convention; this run uses a 10⁶-event cap
  and a 4-scene stride. Compare Δ-over-Raw and AUC_r across protocols, never absolute MESR.
- No labelled-corpus claim is made. This is the unlabelled E-MLB path only.
