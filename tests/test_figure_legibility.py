"""Every figure prints its text at 6 pt or more, this paper's floor.

The figures are drawn at print size and included at `width=\\linewidth`, so the size set in
code is the size on paper. Two failure modes are pinned: type set too small in code, and a
canvas wider than the column it is printed in, which LaTeX shrinks along with its type.
"""

from __future__ import annotations

import re
from typing import Callable, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.figure  # noqa: E402
import matplotlib.text  # noqa: E402
import pytest  # noqa: E402

from dataset_assessment import analyze, figures_protocol  # noqa: E402

MIN_PT = 6.0
REGIME = analyze.FIGURES / "fig_regime_separation.pdf"


@pytest.fixture
def printed_sizes(monkeypatch) -> List[float]:
    """Font sizes of every visible, non-empty text artist in each figure saved."""

    sizes: List[float] = []
    original = matplotlib.figure.Figure.savefig

    def recording_savefig(self, *args, **kwargs):
        # Save first: tick labels are filled in only when the figure is drawn, so reading
        # them before the draw would skip every numeric tick label.
        result = original(self, *args, **kwargs)
        sizes.extend(text.get_fontsize() for text in self.findobj(matplotlib.text.Text)
                     if text.get_visible() and text.get_text().strip())
        return result

    monkeypatch.setattr(matplotlib.figure.Figure, "savefig", recording_savefig)
    return sizes


@pytest.mark.parametrize("draw", [figures_protocol.protocol_view,
                                  figures_protocol.rank_instability,
                                  figures_protocol.native_oracle_view])
def test_protocol_figures_print_no_text_below_six_points(
        draw: Callable, printed_sizes: List[float], tmp_path) -> None:
    draw(tmp_path / "figure.pdf")
    assert printed_sizes and min(printed_sizes) >= MIN_PT


def test_curve_and_regime_figures_print_no_text_below_six_points(
        printed_sizes: List[float], tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(analyze, "FIGURES", tmp_path)
    report = {d: {"rows": [True]} for d in analyze.DATASETS
              if (analyze.RESULTS / f"benchmark_{d}.json").exists()}
    analyze.figures(report)
    assert printed_sizes and min(printed_sizes) >= MIN_PT


def test_regime_figure_is_drawn_at_column_width() -> None:
    """Drawn at print size so the floor above holds on paper. It was 432 pt wide and printed
    at 0.62 of a column, which shrank its 10 pt labels to about 3.6 pt."""

    assert REGIME.exists(), "run `python -m dataset_assessment.analyze` first"
    boxes = re.findall(rb"/MediaBox\s*\[([^\]]*)\]", REGIME.read_bytes())
    assert boxes, "no /MediaBox in the figure PDF"
    width = float(boxes[0].split()[2])
    assert width == pytest.approx(analyze.REGIME_FIG_INCHES[0] * 72, abs=0.5)
    assert width <= 252.0, "wider than an IEEEtran column"
