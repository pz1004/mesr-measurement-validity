"""The random tie-break changes the order inside each group and nothing else."""

from __future__ import annotations

import json

import numpy as np
import pytest

from dataset_assessment.downstream_tiebreak import (OUT, RETENTION, condition, summarise,
                                                    tiebroken)
from dataset_assessment.esr import BLOCK, retain_mask


def _accepted(n_blocks: int = 3, accepted_per_block=(6_000, 8_000, 100), seed: int = 0):
    rng = np.random.default_rng(seed)
    blocks = []
    for count in accepted_per_block[:n_blocks]:
        block = np.zeros(BLOCK, dtype=bool)
        block[rng.choice(BLOCK, count, replace=False)] = True
        blocks.append(block)
    return np.concatenate(blocks)


def test_quotas_and_accepted_first_are_kept():
    accepted = _accepted()
    arrival = retain_mask(np.where(accepted, 0.0, 1.0), RETENTION)
    random_order = retain_mask(tiebroken(accepted, seed=1, clip_index=0), RETENTION)
    for start in range(0, len(accepted), BLOCK):
        a, r = arrival[start:start + BLOCK], random_order[start:start + BLOCK]
        acc = accepted[start:start + BLOCK]
        assert a.sum() == r.sum()                              # the same quota
        assert (a & acc).sum() == (r & acc).sum()              # as many accepted kept
        if acc.sum() <= a.sum():
            assert (r & acc).sum() == acc.sum()                # every accepted event kept
    assert not np.array_equal(arrival, random_order)           # but different events


def test_the_order_is_seeded_per_clip():
    accepted = _accepted()
    a = retain_mask(tiebroken(accepted, 1, 0), RETENTION)
    assert np.array_equal(a, retain_mask(tiebroken(accepted, 1, 0), RETENTION))
    assert not np.array_equal(a, retain_mask(tiebroken(accepted, 2, 0), RETENTION))
    assert not np.array_equal(a, retain_mask(tiebroken(accepted, 1, 1), RETENTION))


def test_conditions_pair_with_the_block_prefix_control():
    """The released evaluator pairs `prefix__x` with `prefix__unfiltered`."""

    red, control = condition(20261006, "red"), condition(20261006, "raw")
    assert control == "r060t20261006__unfiltered"
    assert red.split("__", 1)[0] + "__unfiltered" == control


def test_summary_counts_ties_for_least_accurate():
    rows = [{"seed": 1, "method": "red", "accuracy": 0.5, "mesr": 2.0},
            {"seed": 1, "method": "dwf", "accuracy": 0.5, "mesr": 1.0},
            {"seed": 2, "method": "red", "accuracy": 0.4, "mesr": 2.0},
            {"seed": 2, "method": "dwf", "accuracy": 0.6, "mesr": 1.0}]
    summary = summarise(rows)
    assert summary["red_least_accurate_in_seeds"] == 2
    assert summary["red_least_accurate_alone_in_seeds"] == 1
    assert summary["red_first_by_mesr_in_seeds"] == 2
    assert summary["red_first_by_mesr_alone_in_seeds"] == 2
    rows[1]["mesr"] = 2.0                                   # a tie for first by MESR
    tied = summarise(rows)
    assert (tied["red_first_by_mesr_in_seeds"], tied["red_first_by_mesr_alone_in_seeds"]) == (2, 1)


def test_the_artifact_reports_every_seed_and_method():
    if not OUT.exists():
        pytest.skip("downstream_gesture_tiebreak.json not present")
    report = json.loads(OUT.read_text())
    assert report["gate"]["reproduced"] and report["complete"]
    seeds = len(report["protocol"]["seeds"])
    assert len(report["rows"]) == 8 * seeds
    assert len(report["comparisons"]) == 7 * seeds          # each filter and the null vs control
