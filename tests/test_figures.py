"""Figure 1 has a page budget and a data contract; both are asserted here.

The figure carries three things the prose depends on -- the mean curve, an interval, and
`n(r)` -- and it has to carry them inside a float whose height is fixed by the ten-page
limit. Neither constraint is visible from reading `analyze.figures`, so both are pinned.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pytest

from dataset_assessment.analyze import (CURVE_BAND_SERIES, CURVE_FIG_INCHES, CURVE_ORDER,
                                        CURVE_STYLE, CURVE_STYLE_SCALE, RESULTS,
                                        _curve_style, curves_by_recording, recordings_at_r)
from dataset_assessment.common_support import coverage

DATASETS = ("dnd21", "dvsclean", "emlb", "dvsd22", "pure_ba")
FIGURE = Path(__file__).resolve().parents[1] / "results/figures/fig_retention_curves.pdf"


def available() -> list:
    return [d for d in DATASETS if (RESULTS / f"benchmark_{d}.json").exists()]


@pytest.mark.parametrize("dataset", DATASETS)
def test_n_of_r_agrees_with_the_common_support_module(dataset):
    """The strip under each panel must be the same `n(r)` the AUC repair reports.

    Two modules count evaluable recordings from the same files by different routes. If they
    ever disagree, the figure is annotating a cohort the text does not describe.
    """

    if dataset not in available():
        pytest.skip(f"{dataset} benchmark not present")
    mine = recordings_at_r(curves_by_recording(dataset))
    theirs = coverage(dataset)["recordings_at_r"]
    assert {f"{r:.2f}": n for r, n in mine.items()} == theirs


@pytest.mark.parametrize("dataset", DATASETS)
def test_curves_keep_only_evaluable_points_and_keep_the_recording(dataset):
    if dataset not in available():
        pytest.skip(f"{dataset} benchmark not present")
    payload = json.loads((RESULTS / f"benchmark_{dataset}.json").read_text())
    by_method = curves_by_recording(dataset)
    known = {rec["recording"] for rec in payload["records"]}
    for cells in by_method.values():
        assert set(cells) <= known
        for points in cells.values():
            assert points and all(np.isfinite(v) for v in points.values())
    # Every point kept must be flagged evaluable in the artifact.
    kept = {(rec, m, round(r, 6))
            for m, cells in by_method.items() for rec, pts in cells.items() for r in pts}
    evaluable = {(rec["recording"], m, round(float(p["r"]), 6))
                 for rec in payload["records"] for m, row in rec["methods"].items()
                 if "curve" in row for p in row["curve"]
                 if p["evaluable"] and np.isfinite(p["mesr"])}
    assert kept == evaluable


def test_n_of_r_is_a_property_of_the_recording_not_the_method():
    """`retain_mask` keeps the same count for every ranking, so every method in a recording
    reaches the same grid points. Pooling methods in `recordings_at_r` is only sound if that
    holds, and it is the assumption the whole common-support argument rests on."""

    for dataset in available():
        by_method = curves_by_recording(dataset)
        per_recording = {}
        for method, cells in by_method.items():
            for recording, points in cells.items():
                grid = frozenset(round(r, 6) for r in points)
                assert per_recording.setdefault(recording, grid) == grid, (
                    f"{dataset}/{recording}: {method} has a different evaluable grid")


def test_figure_fits_the_float_budget():
    """The float is 101.2 pt: this graphic plus a three-line caption plus IEEEtran's caption
    skip. Growing the graphic silently pushes the article to eleven pages at $265 a page, so
    the size is asserted rather than trusted."""

    assert FIGURE.exists(), "run `python -m dataset_assessment.analyze` first"
    boxes = re.findall(rb"/MediaBox\s*\[([^\]]*)\]", FIGURE.read_bytes())
    assert boxes, "no /MediaBox in the figure PDF"
    width, height = [float(v) for v in boxes[0].split()[2:4]]
    assert width == pytest.approx(CURVE_FIG_INCHES[0] * 72, abs=0.5)
    assert height == pytest.approx(CURVE_FIG_INCHES[1] * 72, abs=0.5)
    # Included at `width=\linewidth` in a two-column IEEEtran figure*, which is 516 pt.
    assert width == pytest.approx(516.0, abs=1.0)
    assert height <= 64.0


def test_curve_style_draws_no_markers_and_scales_line_weight():
    """Markers merged into a solid band on a panel 1.4 in wide and hid the dash pattern the
    two nulls are told apart by, which is the one distinction the figure cannot lose."""

    style = _curve_style("raw", CURVE_STYLE_SCALE)
    assert "marker" not in style and "markersize" not in style
    assert style["linewidth"] == pytest.approx(2.2 * CURVE_STYLE_SCALE)
    assert _curve_style("not_a_method")["color"] == "#aaaaaa"


def test_the_two_nulls_are_separable_where_they_coincide():
    """On E-MLB and Pure_BA `raw` and `random_null` nearly overlap, and with one dash pattern
    whichever was drawn second erased the other."""

    raw, null = CURVE_STYLE["raw"], CURVE_STYLE["random_null"]
    assert raw["dashes"] != null["dashes"]
    assert raw["color"] != null["color"]
    assert raw["zorder"] > null["zorder"], "the coarser dash must be on top"


def test_banded_series_are_the_ones_the_figure_argues_about():
    assert set(CURVE_BAND_SERIES) <= set(CURVE_ORDER)
    assert "random_null" in CURVE_BAND_SERIES and "raw" in CURVE_BAND_SERIES
