"""AUC_r was described as a mean and aggregated across incomparable domains.

Two defects, both encoded here:

* **The description.** `protocol.summarise_curve` computes a span-normalised trapezoid.
  A trapezoid weights interior grid points twice as heavily as the endpoints and divides by
  the span rather than the count, so it is not the mean the paper called it.
* **The aggregation.** The measurable floor is set by how many complete 30,000-event slices a
  recording yields, so recordings do not share an evaluable range. Averaging per-recording
  AUCs then averages integrals over different domains. Within a recording every method does
  share the range -- `retain_mask` keeps the same count for every ranking -- so the raggedness
  is between recordings only, and that is what the repair has to address.

The measured outcome is that the AUC_r ranking survives both repairs on four of five corpora
and weakens slightly on the fifth. A regression that silently restored own-range aggregation
would remove the evidence for that.
"""

import json
from pathlib import Path

import numpy as np
import pytest

from dataset_assessment.common_support import (auc, coverage, curves, recompute)
from dataset_assessment.esr import BLOCK, retain_mask
from dataset_assessment.protocol import summarise_curve

DATASETS = ("dnd21", "dvsclean", "emlb", "dvsd22", "pure_ba")


def _curve(pairs, evaluable_from=0.0):
    return [{"r": r, "mesr": m, "evaluable": r >= evaluable_from - 1e-9} for r, m in pairs]


def test_auc_is_a_normalised_trapezoid_not_a_mean_over_grid_points():
    """The two disagree whenever the curve is not symmetric about its midpoint."""

    pairs = [(0.2, 0.0), (0.6, 3.0), (1.0, 0.0)]
    got = auc(pairs)
    mean_over_points = float(np.mean([m for _, m in pairs]))
    assert got == pytest.approx(1.5)                 # trapezoid / span
    assert mean_over_points == pytest.approx(1.0)    # what "mean" would give
    assert got != pytest.approx(mean_over_points)


def test_auc_matches_the_published_summariser_on_a_full_curve():
    """The repair must reproduce what the tables printed before it narrows anything."""

    pairs = [(round(0.05 * i, 2), 1.0 + 0.1 * i) for i in range(1, 21)]
    assert auc(pairs) == pytest.approx(summarise_curve(_curve(pairs))["auc_over_r"])


def test_restricting_the_support_changes_the_value():
    """Which is the whole point: an own-range AUC is not comparable to another range's."""

    pairs = [(0.1, 10.0), (0.5, 1.0), (1.0, 1.0)]
    assert auc(pairs) > auc(pairs, 0.5, 1.0)


@pytest.mark.parametrize("dataset", DATASETS)
def test_every_method_on_a_recording_shares_its_evaluable_range(dataset):
    """Evaluability is a property of the recording, so the repair is a cohort/grid question
    and not a per-method one. If this ever fails, the common-support argument changes."""

    path = Path(__file__).resolve().parents[1] / "results" / f"benchmark_{dataset}.json"
    if not path.is_file():
        pytest.skip(f"benchmark_{dataset}.json not generated")
    for recording, cell in curves(dataset).items():
        ranges = {(pts[0][0], pts[-1][0]) for pts in cell.values()}
        assert len(ranges) == 1, f"{dataset}/{recording} methods disagree: {ranges}"


def test_binary_filters_are_swept_by_thinning_then_reintroducing():
    """The construction the paper must state, verified against `retain_mask` itself.

    A binary decision becomes a two-level score (0.0 accepted, 1.0 rejected). Below the
    native retention the sweep keeps the temporally-first *accepted* events of each block and
    drops the rest of the accepted set; above it, it keeps every accepted event and refills
    with the temporally-first *rejected* ones. So it is a filter-plus-selection procedure and
    not a parameter sweep of the filter.
    """

    n, native = 10_000, 0.40
    rng = np.random.default_rng(0)
    accepted = np.zeros(n, dtype=bool)
    accepted[rng.choice(n, int(n * native), replace=False)] = True
    score = np.where(accepted, 0.0, 1.0).astype(np.float32)

    below = retain_mask(score, 0.10)
    assert (~accepted[below]).sum() == 0, "below native, no rejected event may enter"
    for start in range(0, n, BLOCK):
        block_keep = below[start:start + BLOCK]
        block_acc = accepted[start:start + BLOCK]
        assert np.array_equal(np.flatnonzero(block_keep),
                              np.flatnonzero(block_acc)[:block_keep.sum()]), \
            "below native the kept set must be the temporally-first accepted events"

    at = retain_mask(score, native)
    assert np.array_equal(at, accepted), "at native the sweep must be the filter itself"

    above = retain_mask(score, 0.70)
    assert accepted[~above].sum() == 0, "above native, no accepted event may be dropped"
    assert (~accepted[above]).sum() > 0, "above native, rejected events must be reintroduced"


def test_coverage_reports_a_shrinking_sample():
    """`n(r)` has to be visible, or a summary computed on fewer recordings at low r reads as
    a property of the metric."""

    path = Path(__file__).resolve().parents[1] / "results" / "benchmark_pure_ba.json"
    if not path.is_file():
        pytest.skip("benchmark_pure_ba.json not generated")
    cov = coverage("pure_ba")
    n = cov["recordings_at_r"]
    assert n["0.05"] < n["1.00"], "Pure_BA must lose recordings at the low end"
    assert n["1.00"] == cov["n_recordings"]


def test_published_common_support_matches_the_artifact():
    """Own-range reproduces Table III exactly; the repairs are quoted against it."""

    path = Path(__file__).resolve().parents[1] / "results" / "common_support.json"
    if not path.is_file():
        pytest.skip("common_support.json not generated")
    r = json.loads(path.read_text())

    published = {"dnd21": 1.000, "dvsclean": 0.786, "emlb": 1.000,
                 "dvsd22": 0.571, "pure_ba": 0.964}
    for dataset, rho in published.items():
        assert r[dataset]["vs_native"]["own_range"]["spearman"] == pytest.approx(rho,
                                                                                abs=5e-4)
    # Three corpora have ragged support; the conclusion survives on four of five.
    ragged = [d for d in published
              if r[d]["evaluable_range_per_recording"]["ragged"]]
    assert sorted(ragged) == ["dvsclean", "emlb", "pure_ba"]
    for dataset in published:
        for variant in ("common_grid", "common_cohort"):
            assert r[dataset]["vs_native"][variant]["differ"] == \
                r[dataset]["vs_native"]["own_range"]["differ"], \
                f"{dataset}/{variant} changed whether the rankings differ"
    # Pure_BA is the one that moves, and it moves against the paper's convenience.
    assert r["pure_ba"]["vs_native"]["common_grid"]["spearman"] == pytest.approx(0.929,
                                                                                abs=5e-4)
