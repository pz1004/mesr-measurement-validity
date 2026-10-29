"""Every revised figure prints base text at 8 pt or more and within the page.

The figures are drawn at print size and included at `width=\\linewidth`, so the size set in
code is the size on paper. Two failure modes are pinned: type set too small in code, and a
canvas wider than the column it is printed in, which LaTeX shrinks along with its type.
"""

from __future__ import annotations

import re
from pathlib import Path
import subprocess
from typing import Callable, List
import xml.etree.ElementTree as ET

import matplotlib
matplotlib.use("Agg")
import matplotlib.figure  # noqa: E402
import matplotlib.text  # noqa: E402
import pytest  # noqa: E402

from dataset_assessment import analyze, figures_protocol  # noqa: E402
from dataset_assessment.figure_measurement_chain import measurement_chain  # noqa: E402

MIN_PT = 8.0
REGIME = analyze.FIGURES / "fig_regime_separation.pdf"


def pdf_text_bounds_errors(path: Path) -> List[str]:
    """Inspect exported words, including ticks that extend outside the canvas."""
    xml = subprocess.run(["pdftotext", "-bbox", str(path), "-"],
                         check=True, capture_output=True, text=True).stdout
    errors = []
    for page in ET.fromstring(xml).findall(".//{*}page"):
        width, height = float(page.attrib["width"]), float(page.attrib["height"])
        for word in page.findall(".//{*}word"):
            a = word.attrib
            if (float(a["xMin"]) < -.5 or float(a["yMin"]) < -.5
                    or float(a["xMax"]) > width + .5
                    or float(a["yMax"]) > height + .5):
                errors.append(f"{path.name}: clipped text {word.text!r}")
    return errors


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
        if args and Path(args[0]).suffix.lower() == ".pdf":
            assert pdf_text_bounds_errors(Path(args[0])) == []
        return result

    monkeypatch.setattr(matplotlib.figure.Figure, "savefig", recording_savefig)
    return sizes


@pytest.mark.parametrize("draw", [figures_protocol.protocol_view,
                                  figures_protocol.rank_instability,
                                  figures_protocol.native_oracle_view,
                                  measurement_chain])
def test_protocol_figures_print_no_text_below_eight_points(
        draw: Callable, printed_sizes: List[float], tmp_path) -> None:
    draw(tmp_path / "figure.pdf")
    assert printed_sizes and min(printed_sizes) >= MIN_PT


def test_curve_and_regime_figures_print_no_text_below_eight_points(
        printed_sizes: List[float], tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(analyze, "FIGURES", tmp_path)
    report = {d: {"rows": [True]} for d in analyze.DATASETS
              if (analyze.RESULTS / f"benchmark_{d}.json").exists()}
    analyze.figures(report)
    assert printed_sizes and min(printed_sizes) >= MIN_PT


def test_regime_figure_is_drawn_at_column_width() -> None:
    """Drawn at print size so the floor above holds on paper. It was 432 pt wide and printed
    at 0.62 of a column, which shrank its 10 pt labels to about 3.6 pt."""

    assert REGIME.exists(), "render the curve and regime figures first"
    boxes = re.findall(rb"/MediaBox\s*\[([^\]]*)\]", REGIME.read_bytes())
    assert boxes, "no /MediaBox in the figure PDF"
    width = float(boxes[0].split()[2])
    assert width == pytest.approx(analyze.REGIME_FIG_INCHES[0] * 72, abs=0.5)
    assert width <= 252.0, "wider than an IEEEtran column"
