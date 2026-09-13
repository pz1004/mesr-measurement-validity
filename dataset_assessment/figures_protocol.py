"""The two figures that show the protocol working, rather than the metric failing.

Both are deliberately *not* in `analyze.figures`. That function is reachable only by running
`python -m dataset_assessment.analyze`, which re-runs the whole ranking aggregation and
rewrites `results/rank_analysis.json` as a side effect; redrawing a figure must not be able to
rewrite an artifact the paper cites. Everything here reads the released JSON and writes only
PDFs.

    python -m dataset_assessment.figures_protocol

`fig_protocol_view` answers "what does the protocol buy": the left panel is the scalar a
published table prints, the right panel is what the same corpus looks like once the null is on
the retention axis beside it. The readout drawn on it is the count of retentions at which the
null gains, not the peak -- the peak is a selected point, and selecting one is the error
protocol items 1 and 4 exist to stop.

`fig_rank_instability` answers "which method leads": the ranking at each method's own operating
point against the rankings on AUC_r and on the metric-oracle r*, per corpus. Four of five
corpora disagree somewhere, which is \\S\\ref{sec:ranks}'s claim and had no exhibit.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np

from .analyze import CURVE_CORPUS_NAME, FIGURES, RESULTS, _curve_style

logger = logging.getLogger(__name__)

#: The corpus the left/right contrast is drawn on. DVSD22 is real and unfiltered, is where the
#: null gains most, and is one of the two corpora no published MESR table includes -- so the
#: left panel is exactly the view the literature reports and the right is what it omits.
VIEW_DATASET = "dvsd22"
#: Filters only. `raw` is the reference Delta-over-Raw is taken against and is identically zero.
VIEW_METHODS = ("dwf", "evflow", "knoise", "red", "ts", "ynoise")
#: Drawn at print size and included with `width=\\linewidth`, so no type is downscaled.
VIEW_INCHES = (3.45, 1.55)
RANK_INCHES = (7.17, 1.62)
#: The three summaries \\S\\ref{sec:ranks} contrasts: each method's own operating point, the
#: oracle-free AUC_r the protocol recommends, and the metric-oracle r*. All three are needed:
#: on E-MLB the first two agree and only r* inverts the order, which is that subsection's
#: whole point, so a two-column figure would show E-MLB as the corpus with no problem.
RANK_KEYS = ("rank_native", "rank_auc_over_r", "rank_protocol")
RANK_LABELS = ("native", "$\\mathrm{AUC}_r$", "$r^\\star$")
#: Spearman against the native ranking, printed under the column it belongs to.
RANK_RHO = (None, "spearman_auc_vs_native", "spearman_protocol_vs_native")
RANK_DATASETS = ("dnd21", "dvsclean", "emlb", "dvsd22", "pure_ba")


def delta_over_raw_curve(dataset: str, method: str) -> Dict[float, float]:
    """Mean Delta-over-Raw at each evaluable retention.

    The reference is the unfiltered input, `raw_mesr`, taken once per recording on the full
    stream -- not the thinned control at the same `r`. That is the one reference the whole
    paper uses, and using it here is what makes this figure and Table~II the same quantity.
    """

    payload = json.loads((RESULTS / f"benchmark_{dataset}.json").read_text())
    pooled: Dict[float, List[float]] = {}
    for record in payload["records"]:
        row = record["methods"].get(method)
        if row is None or "curve" not in row:
            continue
        base = float(record["raw_mesr"])
        for point in row["curve"]:
            if point["evaluable"] and np.isfinite(point["mesr"]):
                pooled.setdefault(round(float(point["r"]), 4), []).append(
                    float(point["mesr"]) - base
                )
    return {r: float(np.mean(v)) for r, v in sorted(pooled.items())}


def _scalar_at_native(dataset: str) -> Dict[str, float]:
    """Delta-over-Raw at each method's own operating point, as Table~III reports it."""

    report = json.loads((RESULTS / "rank_analysis.json").read_text())
    return report[dataset]["delta_over_raw_at_native"]


def protocol_view(path: Path) -> Path:
    """Left: the scalar a table prints. Right: the same corpus with the null on the axis."""

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    matplotlib.rcParams.update({"pdf.fonttype": 42, "font.size": 6.0})
    try:
        scalars = _scalar_at_native(VIEW_DATASET)
        null = delta_over_raw_curve(VIEW_DATASET, "random_null")
        figure, (left, right) = plt.subplots(
            1, 2, figsize=VIEW_INCHES, sharey=True,
            gridspec_kw={"width_ratios": [1.0, 1.75], "wspace": 0.08},
        )

        for index, method in enumerate(VIEW_METHODS):
            style = _curve_style(method)
            left.plot([index], [scalars[method]], marker="o", markersize=3.2,
                      color=style["color"], linestyle="none")
        left.set_xticks(range(len(VIEW_METHODS)))
        left.set_xticklabels([m.upper() for m in VIEW_METHODS], rotation=90)
        left.set_ylabel(r"$\Delta$-over-Raw")
        left.set_title("as published:\none scalar per method", fontsize=6.0, pad=3)
        left.margins(x=0.12)

        grid = sorted(r for r in null if r < 1.0)
        # The span every filter occupies, carried across both panels. The point of the figure
        # is where the right-hand curve sits relative to it, so it has to be visible on both.
        low, high = min(scalars[m] for m in VIEW_METHODS), max(scalars[m] for m in VIEW_METHODS)
        right.plot(grid, [null[r] for r in grid], **_curve_style("random_null"))
        right.axhline(0.0, color="#000000", linewidth=0.6, dashes=(3.0, 1.6))
        right.invert_xaxis()
        right.set_xlabel("retention $r$")
        right.set_title("under the protocol:\nthe null on the same axis", fontsize=6.0, pad=3)

        for axis in (left, right):
            axis.axhspan(low, high, color="#999999", alpha=0.18, linewidth=0)
            axis.tick_params(labelsize=5.4, length=2, pad=1.5)
            for side in ("top", "right"):
                axis.spines[side].set_visible(False)

        # The readout is the count, not the peak. The null clears the band at one retention
        # out of nineteen, so a caption that led with the peak would be making the
        # metric-oracle claim protocol items 1 and 4 tell everyone else not to make.
        gaining = sum(1 for r in grid if null[r] > 0)
        right.annotate(f"$\\Delta > 0$ at ${gaining}/{len(grid)}$ retentions",
                       xy=(0.03, 0.88), xycoords="axes fraction",
                       fontsize=5.4, ha="left", va="top", color="#d62728")
        exits = [r for r in grid if null[r] > high]
        if exits:
            clear = max(exits)
            right.annotate(f"clears every filter\nonly at $r={clear:g}$",
                           # the axis is inverted, so the smallest r sits at the right edge
                           xy=(clear, null[clear]), xytext=(-5, -11),
                           textcoords="offset points", fontsize=5.0, ha="right",
                           va="top", color="#555555",
                           arrowprops={"arrowstyle": "-", "linewidth": 0.4,
                                       "color": "#555555", "shrinkA": 0.5, "shrinkB": 1.5})
        right.annotate("shaded: the six filters' span", xy=(0.03, 0.22),
                       xycoords="axes fraction", fontsize=5.4, ha="left", color="#555555")

        figure.subplots_adjust(left=0.135, right=0.995, top=0.80, bottom=0.30)
        path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(path)
        plt.close(figure)
    finally:
        matplotlib.rcdefaults()
    logger.info("wrote %s", path)
    return path


def rank_instability(path: Path, datasets: Sequence[str] = RANK_DATASETS) -> Path:
    """The three summaries of one curve, ranked side by side, one panel per corpus."""

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    matplotlib.rcParams.update({"pdf.fonttype": 42, "font.size": 6.0})
    try:
        report = json.loads((RESULTS / "rank_analysis.json").read_text())
        figure, axes = plt.subplots(1, len(datasets), figsize=RANK_INCHES)

        last = len(RANK_KEYS) - 1
        for axis, dataset in zip(np.atleast_1d(axes), datasets):
            orders = [report[dataset][key] for key in RANK_KEYS]
            moved = any(order != orders[0] for order in orders[1:])
            for method in orders[0]:
                positions = [order.index(method) + 1 for order in orders]
                style = _curve_style(method)
                shifted = len(set(positions)) > 1
                axis.plot(range(len(orders)), positions, color=style["color"],
                          linewidth=1.4 if shifted else 0.6,
                          alpha=1.0 if shifted else 0.35,
                          marker="o", markersize=2.4, zorder=3 if shifted else 2)
            # Only the outer columns carry names; labelling the middle one would put text on
            # both sides of it and collide with the lines arriving from either direction.
            for side in (0, last):
                for rank, method in enumerate(orders[side], start=1):
                    axis.annotate(method, xy=(side, rank), fontsize=4.6,
                                  ha="left" if side else "right",
                                  va="center", xytext=(6 if side else -6, 0),
                                  textcoords="offset points",
                                  color=_curve_style(method)["color"])
            # Both correlations go in the title: three tick labels plus a rho apiece do not
            # fit across a fifth of a two-column float without overprinting each other.
            rhos = " / ".join("%.3f" % report[dataset][key] for key in RANK_RHO if key)
            axis.set_title(f"{CURVE_CORPUS_NAME[dataset]}\n" + r"$\rho$ " + rhos,
                           fontsize=5.6, pad=2, linespacing=1.5,
                           fontweight="bold" if moved else "normal")
            axis.set_xticks(range(len(orders)))
            axis.set_xticklabels(RANK_LABELS, fontsize=5.0)
            axis.set_xlim(-0.75, last + 0.75)
            axis.invert_yaxis()
            axis.set_yticks([])
            axis.tick_params(length=0, pad=1.5)
            for side in ("top", "right", "left"):
                axis.spines[side].set_visible(False)

        np.atleast_1d(axes)[0].set_ylabel("rank", fontsize=5.6)
        figure.subplots_adjust(left=0.030, right=0.952, top=0.76, bottom=0.13, wspace=0.34)
        path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(path)
        plt.close(figure)
    finally:
        matplotlib.rcdefaults()
    logger.info("wrote %s", path)
    return path


def main() -> List[Path]:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    # matplotlib subsets its fonts through fontTools, which narrates every glyph at INFO.
    for noisy in ("matplotlib", "fontTools"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    written = [
        protocol_view(FIGURES / "fig_protocol_view.pdf"),
        rank_instability(FIGURES / "fig_rank_instability.pdf"),
    ]
    for path in written:
        print(path)
    return written


if __name__ == "__main__":
    main()
