"""The matched-span comparison must change the scored span and nothing else.

`null_matched_span` scores the random subsampler and the unfiltered stream on one leading
frame of the input. That is only a like-for-like comparison if the frame leaves neither side
an incomplete slice, if the block quota rebuilt on the frame keeps exactly r of it, and if the
subsampler's priorities are the benchmark's own. Each is pinned here, together with two
anchors: a frame equal to the whole stream reproduces the benchmark's Delta, and a stream with
no temporal structure gives a matched Delta near zero.
"""

import json
from fractions import Fraction

import numpy as np
import pytest

from dataset_assessment import denoisors, null_matched_span
from dataset_assessment.esr import BLOCK, SLICE, mesr, retain_mask
from dataset_assessment.null_matched_span import (CORPORA, GRID, OUT, SEEDS, across_draws,
                                                  analyse, compare, frame_length, null_scores,
                                                  scored_span, summarise)
from dataset_assessment.readers import Recording


@pytest.mark.parametrize("n_events", [1_000_000, 777_777, 181_346])
@pytest.mark.parametrize("r", GRID)
def test_the_frame_holds_whole_slices_on_both_sides(n_events, r):
    frame = frame_length(n_events, r)
    assert 0 <= frame <= n_events
    assert frame % SLICE == 0
    kept = Fraction(r).limit_denominator(100) * frame
    assert kept.denominator == 1 and int(kept) % SLICE == 0


@pytest.mark.parametrize("r", GRID)
def test_the_rebuilt_quota_keeps_exactly_r_of_the_frame(r):
    frame = frame_length(1_000_000, r)
    scores = np.random.default_rng(7).random(frame, dtype=np.float32)
    assert int(retain_mask(scores, r, BLOCK).sum()) == round(r * frame)


def test_the_scored_span_reads_the_last_scored_event():
    keep = np.zeros(100_000, dtype=bool)
    keep[::2] = True                     # 50,000 kept, the first 30,000 scored
    assert scored_span(keep) == pytest.approx(60_000 / 100_000, abs=1e-4)


def test_priorities_are_the_benchmarks_own():
    rec = Recording(np.zeros((1_234, 4), dtype=np.int64), None, 4, 4, "toy", False)
    mine = null_scores(rec, denoisors.RANDOM_NULL_SEED)
    theirs = denoisors.score_events(denoisors.RANDOM_NULL, rec)
    assert np.array_equal(mine, theirs)
    assert not np.array_equal(null_scores(rec, denoisors.RANDOM_NULL_SEED + 1), theirs)


def _stream(n_events, drifting, seed=3, width=64, height=48):
    """Twenty hot pixels emitting a fifth of the events, plus a scene that is fixed or drifts."""

    rng = np.random.default_rng(seed)
    hot = rng.choice(width * height, 20, replace=False)
    pixel = np.empty(n_events, dtype=np.int64)
    is_hot = rng.random(n_events) < 0.2
    pixel[is_hot] = rng.choice(hot, int(is_hot.sum()))
    scene = rng.integers(0, 200, int((~is_hot).sum()))
    if drifting:
        scene = (scene + np.linspace(0, width * height, scene.size).astype(np.int64))
    pixel[~is_hot] = scene % (width * height)
    return pixel % width, pixel // width, width, height


def test_a_frame_equal_to_the_stream_reproduces_the_benchmark_delta():
    x, y, width, height = _stream(600_000, drifting=True)
    scores = np.random.default_rng(11).random(len(x), dtype=np.float32)
    row = next(p for p in compare(x, y, width, height, scores, [0.05]))
    assert row["frame"] == len(x)
    keep = retain_mask(scores, 0.05, BLOCK)
    expected = mesr(x[keep], y[keep], width, height) - mesr(x, y, width, height)
    assert row["delta_matched"] == pytest.approx(expected)
    assert row["delta_original"] == pytest.approx(expected)


def test_a_stream_without_temporal_structure_gives_a_matched_delta_near_zero():
    x, y, width, height = _stream(600_000, drifting=False)
    scores = np.random.default_rng(13).random(len(x), dtype=np.float32)
    for row in compare(x, y, width, height, scores, [0.05, 0.2, 0.5]):
        assert abs(row["delta_matched"]) < 0.01


def test_the_artifact_reproduces_the_benchmark_means():
    if not OUT.exists():
        pytest.skip("null_matched_span.json not present")
    corpora = json.loads(OUT.read_text())["corpora"]
    published = {("dvsd22", 0.05): 0.7786, ("pure_ba", 0.2): 0.0684}
    for (dataset, r), value in published.items():
        if dataset not in corpora:
            continue
        default = corpora[dataset]["by_seed"][0]
        assert default["seed"] == denoisors.RANDOM_NULL_SEED
        row = next(p for p in default["by_retention"] if abs(p["r"] - r) < 1e-9)
        assert row["mean_original_all"] == pytest.approx(value, abs=5e-5)


def test_the_draw_summary_counts_each_sign_per_retention():
    def entry(matched, original):
        rows = [{"r": r, "n": 0, "n_original": 0} for r in GRID]
        rows[0] = {"r": GRID[0], "n": 5, "mean_matched": matched, "mean_original": original}
        return {"by_retention": rows}

    (row,) = across_draws([entry(+0.2, +0.1), entry(-0.1, +0.1), entry(-0.3, -0.2)])
    assert row["r"] == GRID[0] and row["seeds"] == 3
    assert (row["seeds_positive"], row["seeds_negative"], row["seeds_same_sign"]) == (1, 2, 2)
    assert (row["matched_mean_min"], row["matched_mean_max"]) == (-0.3, 0.2)


def _recording(n_events, seed, name):
    x, y, width, height = _stream(n_events, drifting=True, seed=seed)
    events = np.column_stack([np.arange(len(x)), x, y, np.ones_like(x)]).astype(np.int64)
    return Recording(events, None, width, height, name, True)


def test_scoring_recordings_in_parallel_changes_no_value(monkeypatch):
    recordings = [_recording(300_000, 3, "a"), _recording(20_000, 5, "short"),
                  _recording(180_000, 4, "b")]
    monkeypatch.setattr(null_matched_span, "_recordings", lambda dataset: iter(recordings))
    block = analyse("toy", SEEDS[:2], workers=2)
    scored = [rec for rec in recordings if len(rec.events) >= SLICE]
    assert block["n_recordings"] == len(scored) == 2
    for entry in block["by_seed"]:
        direct = {rec.name: compare(rec.x, rec.y, rec.width, rec.height,
                                    null_scores(rec, entry["seed"])) for rec in scored}
        np.testing.assert_equal(entry["by_retention"], summarise(direct))
        if entry["seed"] == SEEDS[0]:
            np.testing.assert_equal(block["recordings"],
                                    [{"recording": name, "points": points}
                                     for name, points in direct.items()])


def test_the_artifact_scores_every_corpus_under_every_draw():
    if not OUT.exists():
        pytest.skip("null_matched_span.json not present")
    corpora = json.loads(OUT.read_text())["corpora"]
    assert sorted(corpora) == sorted(CORPORA)
    for block in corpora.values():
        assert block["seeds"] == list(SEEDS)
        assert all(a["seeds"] == len(SEEDS) for a in block["across_seeds"])
