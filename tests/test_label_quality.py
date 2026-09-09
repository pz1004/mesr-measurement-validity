"""The oracle comparison must be matched on retention, and its classes must be exhaustive.

These tests encode why the oracle result changed. The published number compared each side at
its own `mesr_star` -- different retentions, so possibly different retained counts, and
selected on the test metric the reporting protocol forbids ranking on. Matched retention
fixes both, and the fix only means anything if the two sides really do retain the same
number of events; `measure` asserts that at runtime and `_classify` depends on it.

A regression here would silently restore an unmatched comparison, which is the one defect
that would let a filter "beat" the oracle by being measured somewhere else on the curve.
"""

import numpy as np
import pytest

from dataset_assessment.esr import retain_mask
from dataset_assessment.label_quality import _classify, _quality, summarise


def _cell(method_tp, oracle_tp, kept=100, mesr_win=True, r=0.5, native=0.5,
          recording="rec", method="dwf"):
    """One matched cell. TP + FP = kept on both sides, which is what matching guarantees."""

    return {
        "recording": recording, "method": method, "r": r, "kept": kept,
        "n_signal": 200, "n_noise": 200,
        "mesr": 1.2 if mesr_win else 0.8, "oracle_mesr": 1.0, "mesr_win": mesr_win,
        "tp": method_tp, "fp": kept - method_tp,
        "oracle_tp": oracle_tp, "oracle_fp": kept - oracle_tp,
        "signal_retention": method_tp / 200, "noise_retention": (kept - method_tp) / 200,
        "oracle_signal_retention": oracle_tp / 200,
        "oracle_noise_retention": (kept - oracle_tp) / 200,
        "label_class": _classify({"tp": method_tp}, {"tp": oracle_tp}),
        "native_retention": native,
    }


def test_retain_mask_keeps_the_same_count_for_every_ranking():
    """The whole matched comparison rests on this: same r, same retained count.

    `retain_mask` fixes the count per block from the block length alone, so the count cannot
    depend on the scores. If it ever did, TP + FP would differ between method and oracle and
    the three-class split below would be meaningless.
    """

    rng = np.random.default_rng(0)
    for n in (100, 8192, 20_000):
        binary = (rng.random(n) < 0.3).astype(np.float32)      # a keep/drop filter
        labels = (rng.random(n) < 0.4).astype(np.float32)      # the label oracle
        continuous = rng.random(n).astype(np.float32)          # a scoring method
        for r in (0.05, 0.25, 0.5, 0.95, 1.0):
            counts = {retain_mask(s, r).sum() for s in (binary, labels, continuous)}
            assert len(counts) == 1, f"n={n} r={r} retained {counts}"


def test_oracle_maximises_retained_signal_at_matched_count():
    """`impossible` must never occur: no ranking retains more signal than the labels do."""

    rng = np.random.default_rng(1)
    n = 20_000
    labels = (rng.random(n) < 0.4).astype(np.float32)          # 1 = noise
    other = rng.random(n).astype(np.float32)
    for r in (0.1, 0.3, 0.6, 0.9):
        ok, mk = retain_mask(labels, r), retain_mask(other, r)
        oq = _quality(labels.astype(np.int64), ok)
        mq = _quality(labels.astype(np.int64), mk)
        assert mq["kept"] == oq["kept"]
        assert mq["tp"] <= oq["tp"]
        assert _classify(mq, oq) in {"strict", "tie"}


def test_classify_is_exhaustive_and_ordered():
    assert _classify({"tp": 10}, {"tp": 20}) == "strict"
    assert _classify({"tp": 20}, {"tp": 20}) == "tie"
    assert _classify({"tp": 30}, {"tp": 20}) == "impossible"


def test_strict_reversal_means_worse_on_both_axes():
    """At a matched count, retaining less signal *necessarily* means retaining more noise.

    This is why the paper can claim a strict reversal from the signal count alone.
    """

    c = _cell(method_tp=40, oracle_tp=70, kept=100)
    assert c["label_class"] == "strict"
    assert c["signal_retention"] < c["oracle_signal_retention"]
    assert c["noise_retention"] > c["oracle_noise_retention"]


def test_summarise_counts_only_mesr_wins_as_reversals():
    """A cell with worse label quality but *lower* MESR is not a reversal -- it is expected."""

    payload = {"dataset": "d", "cells": [
        _cell(40, 70, mesr_win=True),
        _cell(40, 70, mesr_win=False, recording="rec2"),
    ]}
    s = summarise(payload)
    assert s["all_matched"]["cells"] == 2
    assert s["all_matched"]["mesr_wins"] == 1
    assert s["all_matched"]["strict_reversals"] == 1


def test_at_native_takes_one_cell_per_pair_nearest_the_operating_point():
    payload = {"dataset": "d", "cells": [
        _cell(40, 70, r=0.30, native=0.52),
        _cell(40, 70, r=0.50, native=0.52),      # nearest
        _cell(40, 70, r=0.90, native=0.52),
    ]}
    s = summarise(payload)
    assert s["at_native"]["cells"] == 1
    assert s["at_native"]["eligible_pairs"] == 1
    assert s["at_native"]["total_pairs"] == 1


def test_native_below_the_grid_floor_is_excluded_not_snapped():
    """A method that natively keeps 0.1% has no matched cell; snapping it to r=0.05 would
    compare its operating point against something 50x away. 30 of DVSCLEAN's 60 pairs are
    excluded for this reason or the relative one below; that exclusion is an eligibility rule, not a null
    result."""

    payload = {"dataset": "d", "cells": [_cell(40, 70, r=0.05, native=0.001)]}
    s = summarise(payload)
    assert s["at_native"]["eligible_pairs"] == 0
    assert s["at_native"]["total_pairs"] == 1


def test_relative_tolerance_excludes_a_grid_point_far_from_a_small_native():
    """native=0.01 vs r=0.05 is under one grid step but five times the retained events."""

    payload = {"dataset": "d", "cells": [_cell(40, 70, r=0.05, native=0.01)]}
    assert summarise(payload)["at_native"]["eligible_pairs"] == 0
    # the same absolute gap is fine when the operating point is large
    payload = {"dataset": "d", "cells": [_cell(40, 70, r=0.55, native=0.51)]}
    assert summarise(payload)["at_native"]["eligible_pairs"] == 1


def test_per_cell_any_is_reported_but_is_not_the_headline():
    """Counting a pair that wins at any one of 20 retentions is 20 looks, and inflates."""

    payload = {"dataset": "d", "cells": [
        _cell(40, 70, r=0.30, mesr_win=False),
        _cell(40, 70, r=0.50, mesr_win=True),
    ]}
    s = summarise(payload)
    assert s["per_cell_any"]["pairs_with_a_win"] == 1
    assert s["all_matched"]["mesr_wins"] == 1
    assert s["all_matched"]["cells"] == 2


@pytest.mark.parametrize("dataset,pairs,wins", [("dnd21", 55, 52), ("dvsclean", 30, 19)])
def test_published_counts_match_the_artifact(dataset, pairs, wins):
    """The numbers the paper quotes come from the artifact, not from prose."""

    import json
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "results" / f"label_quality_{dataset}.json"
    if not path.is_file():
        pytest.skip(f"{path.name} not generated")
    s = json.loads(path.read_text())["summary"]["at_native"]
    assert s["eligible_pairs"] == pairs
    assert s["mesr_wins"] == wins
    assert s["strict_reversals"] == wins, "every win must be a strict reversal"
    assert s["ties"] == 0 and s["impossible"] == 0
