"""Per-retention null intervals must read the same deltas the headline count reads.

`null_intervals` adds an interval at every retention to the sign count of
`analyze.null_gain_at_fixed_retention`. If the two read different deltas, the interval
counts would describe a different quantity from the one the paper counts, so the means and
the positive-mean count are checked against it. The toy cases pin the counting rule.
"""

import json

import pytest

from dataset_assessment.analyze import RESULTS, null_gain_at_fixed_retention
from dataset_assessment.null_intervals import deltas_by_retention, profile


def test_a_clear_rise_counts_and_a_straddle_does_not():
    by_r = {0.5: ([0.9, 1.0, 1.1, 1.0], ["a", "b", "c", "d"]),
            0.8: ([-1.0, 1.0, -1.0, 1.0], ["a", "b", "c", "d"]),
            1.0: ([0.0, 0.0, 0.0, 0.0], ["a", "b", "c", "d"])}
    out = profile(by_r)
    assert out["n_retentions_below_one"] == 2          # r = 1 is the identity, never counted
    assert out["n_above_zero"] == 1 and out["n_below_zero"] == 0
    assert out["not_above_zero_at"] == [0.8]
    assert out["bonferroni_alpha"] == pytest.approx(0.025)


@pytest.mark.parametrize("dataset", ["pure_ba", "dvsd22"])
def test_means_match_the_headline_count(dataset):
    path = RESULTS / f"benchmark_{dataset}.json"
    if not path.exists():
        pytest.skip("benchmark results not present")
    payload = json.loads(path.read_text())
    headline = null_gain_at_fixed_retention(dataset)
    for null in ("raw", "random_null"):
        mine = profile(deltas_by_retention(payload, null))
        theirs = {p["r"]: p["mean"] for p in headline[null]["profile"]}
        assert mine["n_positive_mean"] == headline[null]["n_retentions_with_positive_mean"]
        for point in mine["profile"]:
            assert point["mean"] == pytest.approx(theirs[point["r"]])


@pytest.mark.parametrize("dataset", ["dnd21", "dvsclean", "emlb", "dvsd22", "pure_ba"])
def test_scene_means_recombine_into_the_stored_fixed_retention_mean(dataset):
    """The digest's per-scene means must read the deltas the stored headline mean reads.

    They split Table III's grid-point mean by scene; weighted by recordings they must give it
    back, over the same recordings and the same number of scenes the interval resamples.
    """

    from dataset_assessment.make_numbers_digest import _scene_means

    path = RESULTS / f"benchmark_{dataset}.json"
    if not path.exists():
        pytest.skip("benchmark results not present")
    records = json.loads(path.read_text())["records"]
    stored = json.loads((RESULTS / "rank_analysis.json").read_text())[dataset]
    for null in ("raw", "random_null"):
        best = stored["null_gain_fixed_retention"][null]["best_fixed"]
        means = _scene_means(records, null, best["r"])
        assert len(means) == best["cluster_ci"]["n_clusters"]
        assert sum(n for _, n in means.values()) == best["n"]
        pooled = sum(mean * n for mean, n in means.values()) / best["n"]
        assert pooled == pytest.approx(best["mean"], abs=1e-12)


def test_dnd21_scene_interval_spans_its_two_base_sequences():
    """DND21's scene interval resamples two base sequences, so its ends are their two means.

    With two scenes a bootstrap draw repeats one scene with probability 1/4, above the 2.5%
    tail, so the percentile ends are those means exactly. The supplement reports them for
    that reason: the interval describes the two sequences, not a population of scenes.
    """

    from dataset_assessment.make_numbers_digest import _scene_means

    path = RESULTS / "benchmark_dnd21.json"
    if not path.exists():
        pytest.skip("benchmark results not present")
    best = json.loads((RESULTS / "rank_analysis.json").read_text())["dnd21"][
        "null_gain_fixed_retention"]["random_null"]["best_fixed"]
    means = _scene_means(json.loads(path.read_text())["records"], "random_null", best["r"])
    assert sorted(means) == ["DND21/driving", "DND21/hotel-bar"]
    lo, hi = sorted(mean for mean, _ in means.values())
    assert best["cluster_ci"]["lo"] == pytest.approx(lo, abs=1e-12)
    assert best["cluster_ci"]["hi"] == pytest.approx(hi, abs=1e-12)
