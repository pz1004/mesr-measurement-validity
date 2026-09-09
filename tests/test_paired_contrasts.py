"""Table IV reasoned about its ranking from whether two marginal intervals overlap.

That is invalid in both directions the table used it, and the repair is available: every
classical row is scored on the same 384 E-MLB recordings, so the differences are paired.
These tests encode what changed.

The measured consequence is not cosmetic. Under the overlap rule the paper wrote off the
lower half of the table as "not a ranking"; paired, 14 of 15 pairs resolve and only DWF
against TS does not. A regression that restored marginal-interval reasoning would restore a
claim the data contradicts.
"""

import json

import numpy as np
import pytest

from dataset_assessment.paired_contrasts import (compare, native_deltas_by_recording,
                                                 paired_difference)


def _deltas(spec):
    """recording -> method -> Delta, from {method: [per-recording values]}."""

    n = len(next(iter(spec.values())))
    return {f"rec-{i}": {m: v[i] for m, v in spec.items()} for i in range(n)}


def test_a_paired_difference_can_resolve_what_marginals_cannot():
    """The whole reason for the change: shared recording-level variance cancels.

    Both methods swing over a wide range across recordings, so their marginal intervals are
    wide and overlap completely. The difference is a constant, and the paired interval sees
    it.
    """

    base = np.linspace(0.0, 10.0, 60)
    d = _deltas({"a": list(base + 0.5), "b": list(base)})
    r = paired_difference(d, "a", "b")
    assert r["mean_difference"] == pytest.approx(0.5)
    assert r["excludes_zero"]
    assert r["share_a_above_b"] == pytest.approx(1.0)
    # The marginals, by contrast, are nowhere near separated.
    assert np.std(base) > 2.0


def test_a_genuine_tie_is_not_resolved():
    rng = np.random.default_rng(0)
    noise = rng.normal(0, 1, 60)
    d = _deltas({"a": list(noise), "b": list(rng.normal(0, 1, 60))})
    assert not paired_difference(d, "a", "b")["excludes_zero"]


def test_only_recordings_scoring_both_methods_are_paired():
    d = _deltas({"a": [1.0, 2.0, 3.0], "b": [0.0, 1.0, 2.0]})
    del d["rec-2"]["b"]
    r = paired_difference(d, "a", "b")
    assert r["n_recordings"] == 2
    assert r["mean_difference"] == pytest.approx(1.0)


def test_sign_consistency_is_reported_separately_from_the_mean():
    """A mean driven by one recording is not the same claim as a consistent ordering."""

    d = _deltas({"a": [0.0, 0.0, 0.0, 90.0], "b": [1.0, 1.0, 1.0, 0.0]})
    r = paired_difference(d, "a", "b")
    assert r["mean_difference"] > 0
    assert r["share_a_above_b"] == pytest.approx(0.25)
    assert r["median_difference"] < 0


def test_published_contrasts_match_the_artifact():
    """The numbers Table IV now quotes come from `results/paired_contrasts.json`."""

    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "results" / "paired_contrasts.json"
    if not path.is_file():
        pytest.skip("paired_contrasts.json not generated")
    r = json.loads(path.read_text())
    assert r["ranked_by_mean"] == ["red", "ynoise", "dwf", "ts", "knoise", "evflow"]
    assert r["n_recordings"] == 384 and r["n_scenes"] == 96
    assert r["pairs_resolved"] == 14 and r["pairs_total"] == 15

    by = {(p["a"], p["b"]): p for p in r["pairs"]}
    # The overlap rule called this pair unresolvable and it is; that half of the old claim
    # survives, and the difference is 0.0092 -- the published gap, to four decimals.
    assert not by[("dwf", "ts")]["excludes_zero"]
    assert by[("dwf", "ts")]["mean_difference"] == pytest.approx(0.0092, abs=5e-5)
    # The overlap rule called this pair unresolvable and it is not. This is the claim the
    # old reasoning got wrong.
    assert by[("knoise", "evflow")]["excludes_zero"]
    assert by[("knoise", "evflow")]["lo"] > 0
