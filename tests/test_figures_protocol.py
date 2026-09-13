"""The two protocol figures carry printed numbers, so their data contracts are pinned.

Figure~1's caption prints the null's peak and Figure~3's panel titles print Spearman
$\\rho$. Both are drawn from released artifacts rather than typed, and both restate values the
body prints, so a figure that silently disagreed with the prose would be invisible in a build
that reports every counter as zero. Each is asserted against the artifact here.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pytest

from dataset_assessment.analyze import RESULTS
from dataset_assessment.figures_protocol import (RANK_DATASETS, RANK_INCHES, RANK_KEYS,
                                                 RANK_RHO, VIEW_DATASET, VIEW_INCHES,
                                                 VIEW_METHODS, _scalar_at_native,
                                                 delta_over_raw_curve)

FIGURES = Path(__file__).resolve().parents[1] / "results/figures"
VIEW_PDF = FIGURES / "fig_protocol_view.pdf"
RANK_PDF = FIGURES / "fig_rank_instability.pdf"
REGENERATE = "run `python -m dataset_assessment.figures_protocol` first"


def media_box(path: Path) -> tuple:
    boxes = re.findall(rb"/MediaBox\s*\[([^\]]*)\]", path.read_bytes())
    assert boxes, f"no /MediaBox in {path.name}"
    return tuple(float(v) for v in boxes[0].split()[2:4])


def test_protocol_view_fits_one_column():
    """Drawn at print size and included at `width=\\linewidth` in a single IEEEtran column,
    which is 252 pt. Drawing it wider would downscale the type below legibility."""

    assert VIEW_PDF.exists(), REGENERATE
    width, height = media_box(VIEW_PDF)
    assert width == pytest.approx(VIEW_INCHES[0] * 72, abs=0.5)
    assert height == pytest.approx(VIEW_INCHES[1] * 72, abs=0.5)
    assert width <= 252.0, "wider than an IEEEtran column"


def test_rank_instability_fits_the_two_column_float():
    assert RANK_PDF.exists(), REGENERATE
    width, height = media_box(RANK_PDF)
    assert width == pytest.approx(RANK_INCHES[0] * 72, abs=0.5)
    assert height == pytest.approx(RANK_INCHES[1] * 72, abs=0.5)
    assert width == pytest.approx(516.0, abs=1.0)


def test_the_null_curve_reproduces_the_peak_the_caption_prints():
    """$+0.7786$ on DVSD22 is printed in Table~I, in the abstract's $85\\times$ and in this
    figure's caption. All three must come from one computation."""

    null = delta_over_raw_curve(VIEW_DATASET, "random_null")
    below_one = {r: v for r, v in null.items() if r < 1.0}
    assert max(below_one.values()) == pytest.approx(0.7786, abs=5e-5)


def test_the_null_gains_at_every_evaluable_retention_on_dvsd22():
    """The figure's right panel sits entirely above zero, which is the $19/19$ readout."""

    null = delta_over_raw_curve(VIEW_DATASET, "random_null")
    below_one = {r: v for r, v in null.items() if r < 1.0}
    assert len(below_one) == 19
    assert sum(1 for v in below_one.values() if v > 0) == 19


def test_delta_is_taken_against_the_unfiltered_input_not_the_thinned_control():
    """The paper uses one reference throughout: the method at $r$ minus `raw` at $r = 1$. A
    thinned-control reference would make the `raw` row identically zero, which it is not."""

    raw = delta_over_raw_curve(VIEW_DATASET, "raw")
    assert raw[1.0] == pytest.approx(0.0, abs=1e-12)
    assert any(abs(v) > 1e-6 for r, v in raw.items() if r < 1.0)


@pytest.mark.parametrize("dataset", RANK_DATASETS)
def test_plotted_ranks_are_the_released_rankings(dataset):
    """The bump chart reads `rank_analysis.json` directly; nothing is re-derived in the plot."""

    report = json.loads((RESULTS / "rank_analysis.json").read_text())
    orders = []
    for key in RANK_KEYS:
        order = report[dataset][key]
        assert len(order) == len(set(order)), f"{dataset}/{key} ranks a method twice"
        orders.append(order)
    for key, order in zip(RANK_KEYS[1:], orders[1:]):
        assert set(order) == set(orders[0]), f"{dataset}/{key} covers different methods"


def test_the_figure_shows_all_three_summaries_the_subsection_contrasts():
    """Dropping the metric-oracle column would show E-MLB as the corpus with nothing wrong
    with it, the reverse of what the subsection concludes about E-MLB."""

    report = json.loads((RESULTS / "rank_analysis.json").read_text())
    assert RANK_KEYS == ("rank_native", "rank_auc_over_r", "rank_protocol")
    assert report["emlb"]["rank_auc_over_r"] == report["emlb"]["rank_native"]
    assert report["emlb"]["rank_protocol"] != report["emlb"]["rank_native"]


def test_four_corpora_disagree_somewhere_and_three_disagree_on_auc():
    """The caption bolds a corpus when any summary reorders it, and prints "three of five"
    for the AUC comparison alone. Both counts are read off the artifact, not typed."""

    report = json.loads((RESULTS / "rank_analysis.json").read_text())
    bolded = [d for d in RANK_DATASETS
              if any(report[d][k] != report[d][RANK_KEYS[0]] for k in RANK_KEYS[1:])]
    assert sorted(bolded) == ["dvsclean", "dvsd22", "emlb", "pure_ba"]
    on_auc = [d for d in RANK_DATASETS
              if report[d][RANK_KEYS[1]] != report[d][RANK_KEYS[0]]]
    assert sorted(on_auc) == ["dvsclean", "dvsd22", "pure_ba"]
    for dataset in RANK_DATASETS:
        for rho_key, rank_key in zip(RANK_RHO[1:], RANK_KEYS[1:]):
            agrees = report[dataset][rank_key] == report[dataset][RANK_KEYS[0]]
            assert (report[dataset][rho_key] == pytest.approx(1.0)) is agrees


def test_the_null_clears_the_filter_band_at_one_retention_only():
    """Figure 1's caption says the null gains everywhere but clears every filter only at
    r = 0.05. An earlier caption said it left the band outright, which is false at 18 of the
    19 retentions and is the selected-point claim protocol item 1 forbids."""

    scalars = _scalar_at_native(VIEW_DATASET)
    high = max(scalars[m] for m in VIEW_METHODS)
    null = delta_over_raw_curve(VIEW_DATASET, "random_null")
    below_one = {r: v for r, v in null.items() if r < 1.0}
    assert [r for r, v in below_one.items() if v > high] == [0.05]
    assert len(below_one) == 19
    assert sum(1 for v in below_one.values() if v > 0) == 19

def test_the_view_panel_covers_every_filter_the_benchmark_scores():
    """The shaded band is the span of all six filters; leaving one out would understate it."""

    report = json.loads((RESULTS / "rank_analysis.json").read_text())
    scalars = report[VIEW_DATASET]["delta_over_raw_at_native"]
    assert set(VIEW_METHODS) | {"raw"} == set(scalars)
    assert all(np.isfinite(scalars[m]) for m in VIEW_METHODS)
