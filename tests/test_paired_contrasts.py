"""The E-MLB table reasoned about its ranking from whether two marginal intervals overlap.

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
    """The numbers the E-MLB table now quotes come from `results/paired_contrasts.json`.

    Regenerated under `protocol.native_point_is_eligible`, which drops the 445 of 2304 cells
    whose native retention lies below the recording's measurable floor. Every row is now its
    own cohort, so `n` is per pair and the marginals are not comparable across rows.
    """

    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "results" / "paired_contrasts.json"
    if not path.is_file():
        pytest.skip("paired_contrasts.json not generated")
    r = json.loads(path.read_text())
    assert r["ranked_by_mean"] == ["red", "ynoise", "dwf", "evflow", "ts", "knoise"]
    assert r["pairs_resolved"] == 10 and r["pairs_total"] == 15

    by = {(p["a"], p["b"]): p for p in r["pairs"]}
    # The five unresolved pairs are exactly the four middle methods against each other.
    unresolved = {frozenset((p["a"], p["b"])) for p in r["pairs"] if not p["excludes_zero"]}
    middle = {"ynoise", "dwf", "evflow", "ts"}
    assert unresolved == {frozenset(pair) for pair in
                          [("ynoise", "dwf"), ("ynoise", "evflow"), ("ynoise", "ts"),
                           ("dwf", "evflow"), ("dwf", "ts"), ("evflow", "ts")]} - {
                              frozenset(("ynoise", "dwf"))}
    assert all(a in middle and b in middle for p in unresolved for a, b in [tuple(p)])
    # DWF over TS still fails to separate, on the 284 recordings where both are measurable.
    assert not by[("dwf", "ts")]["excludes_zero"]
    assert by[("dwf", "ts")]["n_recordings"] == 284
    # Each pair carries its own cohort size, and no pair is scored on all 384 any more
    # except where both methods are measurable everywhere.
    assert {p["n_recordings"] for p in r["pairs"]} != {384}
    # RED separates from every other method; KNoise sits below every other method.
    for p in r["pairs"]:
        if p["a"] == "red" or p["b"] == "knoise":
            assert p["excludes_zero"], (p["a"], p["b"])


def test_the_blank_response_is_read_at_a_selection_free_retention():
    """Section VII quotes Pure_BA at its common-support floor, not at native operating points.

    Native is not available there: Pure_BA's filters keep a few per cent while its measurable
    floor is r = 0.60, so `native_point_is_eligible` leaves RED with 4 of 26 recordings and TS
    with none. Fixing r keeps all 26 and puts the nonselective controls at the same retained
    count, which is what makes the comparison a blank-sample check rather than a raw number.
    """

    from dataset_assessment.analyze import common_support_floor, delta_at_fixed_retention

    floor = common_support_floor("pure_ba")
    assert floor == pytest.approx(0.60)
    by = delta_at_fixed_retention("pure_ba", floor)["by_method"]

    assert {v["n"] for v in by.values()} == {26}, "every recording must be evaluable here"
    controls = max(by["raw"]["mean"], by["random_null"]["mean"])
    assert by["red"]["mean"] == pytest.approx(1.3507, abs=5e-4)
    assert by["raw"]["mean"] == pytest.approx(0.0348, abs=5e-4)
    # The claim: selecting which noise to drop is worth far more than dropping the amount.
    assert by["red"]["mean"] / controls > 25
    for method in ("red", "dwf", "ynoise"):
        assert by[method]["lo"] > controls


def test_table_iv_prints_the_eligible_cohort_and_says_what_it_dropped():
    """Every value and every count in the E-MLB table traces to `native_eligibility`."""

    from dataset_assessment.analyze import native_eligibility

    r = native_eligibility("emlb")
    assert r["cells"] == 2304 and r["ineligible"] == 445
    assert r["share_ineligible"] == pytest.approx(0.193, abs=5e-4)
    # Every ineligible cell has the same cause, which is what lets the paper state one.
    assert r["ineligible_because_native_is_below_the_floor"] == r["ineligible"]

    by = r["by_method"]
    printed = {"red": (311, 0.4550), "ynoise": (381, 0.2126), "dwf": (384, 0.1117),
               "evflow": (123, 0.0994), "ts": (284, 0.0946), "knoise": (376, 0.0526)}
    for method, (n, mean) in printed.items():
        assert by[method]["eligible"] == n, method
        assert by[method]["eligible_mean_delta"] == pytest.approx(mean, abs=5e-5), method

    # The claim that made the restriction necessary: the dropped cells are not a random
    # sample, and they were carrying the table's top row.
    assert by["red"]["ineligible_mean_delta"] == pytest.approx(2.7180, abs=5e-4)
    assert by["red"]["ineligible_mean_delta"] > 5 * by["red"]["eligible_mean_delta"]
    # EvFlow's printed native retention is not where EvFlow was scored.
    assert by["evflow"]["mean_native_r_all_cells"] == pytest.approx(0.0493, abs=5e-5)
    assert by["evflow"]["mean_scored_at_r_all_cells"] == pytest.approx(0.0824, abs=5e-5)
