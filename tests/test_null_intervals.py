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
