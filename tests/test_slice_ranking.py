"""The slice-size ranking check must compare like with like.

`slice_ranking` asks whether the (N, M) convention reorders filters or only shifts them. Three
properties keep that question honest:

* **Every size ranks the same set.** A filter too short for a 100,000-event slice leaves the
  cohort at every size, so a changed ranking is never a changed membership.
* **An unchanged ordering reads as agreement.** A common shift at every size gives tau = 1
  and no discordant pair.
* **The per-size score is MESR itself**, with NaN below one complete slice.
"""

import numpy as np

from dataset_assessment.esr import mesr
from dataset_assessment.slice_ranking import paired_flips, rank_agreement, scores_by_slice

SLICES = (10, 20, 30)


def test_a_common_shift_is_full_agreement():
    table = {m: {s: base + s / 100 for s in SLICES}
             for m, base in (("a", 1.0), ("b", 2.0), ("c", 3.0))}
    out = rank_agreement(table, slices=SLICES, reference=30)
    assert out["cohort"] == ["a", "b", "c"]
    for v in out["by_slice"].values():
        assert v["tau"] == 1.0 and v["discordant_pairs"] == 0 and not v["top_changes"]


def test_a_crossing_is_counted():
    table = {"a": {10: 2.0, 20: 1.0, 30: 1.0}, "b": {10: 1.0, 20: 2.0, 30: 2.0},
             "c": {10: 3.0, 20: 3.0, 30: 3.0}}
    out = rank_agreement(table, slices=SLICES, reference=30)
    assert out["by_slice"][10]["discordant_pairs"] == 1
    assert out["by_slice"][20]["discordant_pairs"] == 0


def test_a_short_output_leaves_the_cohort_at_every_size():
    table = {"a": {10: 1.0, 20: 2.0, 30: 3.0}, "b": {10: 2.0, 20: 3.0, 30: 4.0},
             "short": {10: 9.0, 20: float("nan"), 30: float("nan")}}
    assert rank_agreement(table, slices=SLICES, reference=30)["cohort"] == ["a", "b"]


def test_the_score_is_mesr_and_nan_below_one_slice():
    rng = np.random.default_rng(3)
    x, y = rng.integers(0, 20, 25), rng.integers(0, 10, 25)
    out = scores_by_slice(x, y, 20, 10, slices=SLICES)
    assert out[10] == mesr(x, y, 20, 10, slice_size=10)
    assert out[20] == mesr(x, y, 20, 10, slice_size=20)
    assert np.isnan(out[30])


def test_pairs_keep_one_cohort_across_the_sweep():
    per = {"r1": {"a": {10: 1.0, 20: 1.0, 30: 1.0}, "b": {10: 2.0, 20: 0.0, 30: 0.5}},
           "r2": {"a": {10: 1.0, 20: 1.0, 30: 1.0},
                  "b": {10: 5.0, 20: float("nan"), 30: 9.0}}}
    (pair,) = paired_flips(per, slices=SLICES, reference=30)
    assert pair["n_recordings"] == 1                # r2 drops out: b is short at size 20
    assert pair["flips"] == [10]                    # a-b is +0.5 at 30, -1.0 at 10
