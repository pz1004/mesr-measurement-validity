"""The classifier is handed exactly the events MESR scores, and nothing else changes."""

from __future__ import annotations

import json

import numpy as np
import pytest

from dataset_assessment.downstream_gesture_frozen import frames_for
from dataset_assessment.downstream_matched_events import (OUT, RETENTIONS, _write_frames, cohort,
                                                          condition, frozen_accuracy, summarise)
from dataset_assessment.esr import SLICE, mesr, retain_mask
from dataset_assessment.native_oracle import scored_prefix
from dataset_assessment.sew7b.frames import events_to_number_frames


def _clip(n: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    clip = np.zeros((n, 4), dtype=np.int64)
    clip[:, 0] = np.sort(rng.integers(0, 10 * n, n))
    clip[:, 1] = rng.integers(0, 128, n)
    clip[:, 2] = rng.integers(0, 128, n)
    clip[:, 3] = rng.integers(0, 2, n)
    return clip


def test_the_scored_prefix_is_the_retained_sets_first_slices_and_scores_the_same():
    clip = _clip(200_000)
    scores = np.random.default_rng(1).random(len(clip))
    for retention in RETENTIONS:
        keep = retain_mask(scores, retention)
        prefix = scored_prefix(keep)
        assert prefix.sum() == SLICE * (keep.sum() // SLICE)
        kept = np.flatnonzero(keep)
        assert np.array_equal(np.flatnonzero(prefix), kept[:prefix.sum()])
        # MESR never reads past the last complete slice, so the two score identically.
        assert (mesr(clip[prefix][:, 1], clip[prefix][:, 2], 128, 128)
                == mesr(clip[keep][:, 1], clip[keep][:, 2], 128, 128))


def test_a_clip_without_a_complete_slice_leaves_the_cohort():
    short, long = _clip(40_000, 2), _clip(120_000, 3)
    prefixes = [scored_prefix(retain_mask(np.zeros(len(c)), 0.6)) for c in (short, long)]
    assert cohort(prefixes) == [1]           # 24,000 retained events hold no slice


def test_frame_rows_follow_the_cohort_and_skip_the_clips_outside_it(tmp_path):
    clips = [_clip(n, seed) for n, seed in ((40_000, 4), (60_000, 5), (90_000, 6))]
    masks = [scored_prefix(retain_mask(np.zeros(len(c)), 0.6)) for c in clips]
    members = cohort(masks)
    assert members == [1, 2]
    path = tmp_path / "frames.npy"
    _write_frames(path, clips, masks, members)
    stack = np.load(path)
    assert len(stack) == len(members)
    for row, index in enumerate(members):
        assert np.array_equal(stack[row],
                              frames_for(clips[index], masks[index], events_to_number_frames))


def test_conditions_pair_with_the_block_prefix_control():
    """The released evaluator pairs `prefix__x` with `prefix__unfiltered`."""

    assert condition("raw", 0.6) == "m060__unfiltered"
    assert condition("red", 0.4) == "m040__red"
    assert condition("red", 0.6, scored=False) == "r060__red"     # the frozen run's own name
    for retention in RETENTIONS:
        red = condition("red", retention)
        assert red.split("__", 1)[0] + "__unfiltered" == condition("raw", retention)


def test_the_full_stream_accuracy_reads_only_the_cohort():
    evaluation = {"conditions": {"r040__red": {"prediction": [0, 1, 2, 3],
                                               "target": [0, 1, 0, 0]}}}
    assert frozen_accuracy(evaluation, "r040__red", [0, 1, 2, 3]) == 0.5
    assert frozen_accuracy(evaluation, "r040__red", [0, 1]) == 1.0


def test_the_summary_keeps_ties_and_reads_the_control_contrast():
    rows = [{"retention": 0.6, "method": m, "accuracy": a, "mesr": s, "clips": 288}
            for m, a, s in (("raw", 0.78, 1.43), ("red", 0.66, 1.49), ("dwf", 0.66, 1.42),
                            ("knoise", 0.85, 1.44))]
    comparisons = {"m060__red_vs_m060__unfiltered": {
        "accuracy_delta": -0.12, "accuracy_delta_95ci": [-0.17, -0.07],
        "by_subject": {"a": {"accuracy_delta": -0.2}, "b": {"accuracy_delta": 0.0},
                       "c": {"accuracy_delta": -0.1}}}}
    summary = summarise(rows, comparisons)
    cell = summary["per_retention"]["0.6"]
    assert cell["least_accurate"] == ["dwf", "red"]
    assert cell["most_accurate"] == ["knoise"]
    assert cell["first_by_mesr"] == ["red"]
    assert (cell["red_minus_control"]["subjects_red_lower"],
            cell["red_minus_control"]["subjects_tied"]) == (2, 1)
    assert summary["red_least_accurate_at"] == ["0.6"]
    assert summary["red_least_accurate_alone_at"] == []        # tied with DWF


def test_the_artifact_reproduces_the_frozen_runs_retained_sets():
    if not OUT.exists():
        pytest.skip("downstream_matched_events.json not present")
    report = json.loads(OUT.read_text())
    assert report["complete"] and report["gate"]["reproduced"]
    assert report["reproduction"]["identical_predictions"]
    assert len(report["rows"]) == 8 * len(RETENTIONS)
    assert len(report["comparisons"]) == 7 * len(RETENTIONS)   # each procedure vs the control
    frozen = json.loads((OUT.parent / "downstream_gesture_frozen.json").read_text())
    stored = {(row["method"], row["retention"]): row for row in frozen["conditions"]}
    for row in report["rows"]:
        reference = stored[(row["method"], row["retention"])]
        assert abs(row["mesr"] - reference["mesr"]) <= 1e-12
        assert row["clips"] == reference["evaluable_test_samples"]
