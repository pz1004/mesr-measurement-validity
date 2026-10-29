"""The figures that show the protocol working, and the native-output reversal they rest on.

All three read released JSON and write only figure PDFs and PNG previews. The companion
retention and regime figures can likewise be rendered by calling `analyze.figures` directly.
Avoid the full `python -m dataset_assessment.analyze` entry point for a visual revision:
it also re-runs ranking aggregation and rewrites `results/rank_analysis.json`.

    python -m dataset_assessment.figures_protocol

`fig_protocol_view` compares six computed quota-adapted single-point summaries with random
subsampling across retention on the same corpus. Its fixed shaded range spans those six
summaries; it is not an uncertainty band. The caption distinguishes the 19 positive means
from the one retained fraction at which the random mean exceeds all six summaries.

`fig_rank_instability` answers "which method leads": the ranking at each method's own operating
point against the rankings on AUC_r and on the metric-oracle r*, per corpus. Four of five
corpora disagree somewhere, which is \\S\\ref{sec:ranks}'s claim and had no exhibit.

`fig_native_oracle` shows the size of \\S\\ref{sec:oracle}'s reversals, not only their count:
each filter's own scored output against its matched label oracle, the retained-purity
difference against the MESR difference, as released and with the busiest pixels removed.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np

from .analyze import CURVE_CORPUS_NAME, FIGURES, RESULTS, _curve_style, display_name

logger = logging.getLogger(__name__)

#: The corpus for the two-panel comparison. The single-point summaries are computed from
#: quota-adapted responses near native retention; they are not published DVSD22 measurements.
VIEW_DATASET = "dvsd22"
#: Filters only. Delta subtracts the unfiltered input, not the prefix control at matching r.
VIEW_METHODS = ("dwf", "evflow", "knoise", "red", "ts", "ynoise")
#: Drawn at print size and included with `width=\\linewidth`, so no type is downscaled.
VIEW_INCHES = (7.17, 2.30)
#: Three corpus panels above two, with a sixth cell for definitions and the shared key.
RANK_INCHES = (7.17, 3.75)
#: The three summaries \\S\\ref{sec:ranks} contrasts: each method's own operating point, the
#: oracle-free AUC_r the protocol recommends, and the metric-oracle r*. All three are needed:
#: on E-MLB the first two agree and only r* inverts the order, which is that subsection's
#: whole point, so a two-column figure would show E-MLB as the corpus with no problem.
RANK_KEYS = ("rank_native", "rank_auc_over_r", "rank_protocol")
RANK_LABELS = ("Nearest\ngrid", "Area over\nretention", "Maximum\nMESR")
#: Spearman against the nearest-grid ranking (the quota-adapted point nearest each native
#: retention, not the native output), printed above the column it belongs to.
RANK_RHO = (None, "spearman_auc_vs_native", "spearman_protocol_vs_native")
RANK_DATASETS = ("dnd21", "dvsclean", "emlb", "dvsd22", "pure_ba")
ORACLE_INCHES = (3.45, 2.55)
#: The two labelled corpora: marker and colour per corpus; fill tells the two inputs apart.
ORACLE_CORPORA = (("dnd21", "o", "#1f4e79", "DND21"), ("dvsclean", "^", "#b8541f", "DVSCLEAN"))

FIGURE_STYLE = {
    "font.family": "STIXGeneral", "font.size": 8.0,
    "mathtext.fontset": "stix", "mathtext.default": "it",
    "axes.labelsize": 8.5, "axes.titlesize": 9.0, "axes.linewidth": 0.6,
    "xtick.labelsize": 8.0, "ytick.labelsize": 8.0,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "xtick.major.pad": 2.0, "ytick.major.pad": 2.0,
    "pdf.fonttype": 42, "ps.fonttype": 42,
}


def _axes_in_points(figure, x, y, width, height):
    """Position axes in typographic points at the final publication dimensions."""
    fw, fh = figure.get_size_inches() * 72
    axis = figure.add_axes([x/fw, y/fh, width/fw, height/fh])
    axis.spines[["top", "right"]].set_visible(False)
    return axis


def _figure_text(figure, x, y, label, **kwargs):
    fw, fh = figure.get_size_inches() * 72
    return figure.text(x/fw, y/fh, label, fontsize=kwargs.pop("fontsize", 8),
                       ha=kwargs.pop("ha", "left"), va=kwargs.pop("va", "center"),
                       **kwargs)


def _save_figure(figure, path: Path) -> Path:
    """Write the vector figure and a matching review preview, never an evidence artifact."""
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    with plt.rc_context(FIGURE_STYLE):
        figure.savefig(path, facecolor="white")
        figure.savefig(path.with_suffix(".png"), dpi=240, facecolor="white")
    plt.close(figure)
    logger.info("wrote %s", path)
    return path


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
    """Delta-over-Raw at the grid point nearest each method's native retention.

    These are quota-adapted values, the nearest-grid reading of Fig. 4 (unrestricted cohort),
    not native-output scores of the kind Table IV reports for E-MLB; the caption says so.
    """

    report = json.loads((RESULTS / "rank_analysis.json").read_text())
    return report[dataset]["delta_over_raw_at_native"]


def build_protocol_view():
    """Build the fixed single-point range and random-control response without writing."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    with plt.rc_context(FIGURE_STYLE):
        scalars = _scalar_at_native(VIEW_DATASET)
        null = delta_over_raw_curve(VIEW_DATASET, "random_null")
        figure = plt.figure(figsize=VIEW_INCHES)
        left = _axes_in_points(figure, 43, 52, 194, 82)
        right = _axes_in_points(figure, 292, 52, 216, 82)
        _figure_text(figure, 7, 155, "(a) Single-point summaries", fontsize=9, weight="bold")
        _figure_text(figure, 274, 155, "(b) Random subsampling across retention",
                     fontsize=9, weight="bold")

        for index, method in enumerate(VIEW_METHODS):
            point, = left.plot([index], [scalars[method]], marker="o", markersize=4.0,
                               color=_curve_style(method)["color"], linestyle="none", zorder=4)
            point.set_gid(f"protocol:{method}")
        left.set_xticks(range(len(VIEW_METHODS)))
        left.set_xticklabels([display_name(m) for m in VIEW_METHODS])
        left.set_xlim(-0.55, len(VIEW_METHODS)-0.45)
        left.set_ylabel("MESR change from\nunfiltered input", labelpad=5)

        grid = sorted(r for r in null if r < 1.0)
        low, high = min(scalars[m] for m in VIEW_METHODS), max(scalars[m] for m in VIEW_METHODS)
        null_style = {**_curve_style("random_null"), "linewidth": 1.5}
        curve, = right.plot(grid, [null[r] for r in grid], **null_style)
        curve.set_gid("protocol:random_null")
        right.set(xlim=(min(grid)-0.025, max(grid)+0.025),
                  xticks=[0.05, 0.25, 0.50, 0.75, 0.95], xlabel="Retained fraction $r$")
        right.set_xticklabels(["0.05", "0.25", "0.50", "0.75", "0.95"])
        top = max(high, max(null[r] for r in grid)) + 0.06
        for name, axis in (("summaries", left), ("retention", right)):
            band = axis.axhspan(low, high, color="#D8DDE1", alpha=0.55, linewidth=0, zorder=0)
            band.set_gid(f"protocol:fixed-range:{name}")
            zero = axis.axhline(0, color="#555555", lw=0.7, ls=(0, (3, 2)), zorder=2)
            zero.set_gid(f"protocol:zero:{name}")
            axis.set(ylim=(-0.04, top), yticks=[0.0, 0.25, 0.50, 0.75])
            axis.grid(axis="y", color="#E4E7EA", lw=0.45, zorder=1)
            axis.tick_params(labelsize=8)
        right.tick_params(labelleft=False)

        band_key = Patch(facecolor="#D8DDE1", edgecolor="none", alpha=0.55,
                         label="Fixed range of six filter summaries")
        figure.legend([curve, band_key], ["Random subsampling", band_key.get_label()],
                      loc="center", bbox_to_anchor=(0.5, 17/(VIEW_INCHES[1]*72)),
                      ncol=2, fontsize=8, frameon=False, handlelength=2.1,
                      handletextpad=0.55, columnspacing=1.7, borderpad=0)
        _figure_text(figure, VIEW_INCHES[0]*36, 5,
                     "Positive = higher MESR than unfiltered input.", ha="center", color="#555555")
        figure.canvas.draw()
    return figure


def protocol_view(path: Path) -> Path:
    """Render the two-column single-point and random-subsampling comparison."""
    return _save_figure(build_protocol_view(), path)


def build_rank_instability(datasets: Sequence[str] = RANK_DATASETS):
    """Build all three recorded rank permutations, with a separate explanatory key."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with plt.rc_context(FIGURE_STYLE):
        report = json.loads((RESULTS / "rank_analysis.json").read_text())
        rows = max(2, (len(datasets)+3)//3)
        figure = plt.figure(figsize=(RANK_INCHES[0], RANK_INCHES[1]*rows/2))
        height = figure.get_size_inches()[1]*72
        last = len(RANK_KEYS)-1
        for panel, dataset in enumerate(datasets):
            row, column = divmod(panel, 3)
            x = 18+172*column
            top = height-33-128*row
            axis = _axes_in_points(figure, x, top-76, 150, 76)
            axis.set_gid(f"rank-panel:{dataset}")
            orders = [report[dataset][key] for key in RANK_KEYS]
            for method in orders[0]:
                positions = [order.index(method)+1 for order in orders]
                shifted = len(set(positions)) > 1
                line, = axis.plot(range(len(orders)), positions,
                                  color=_curve_style(method)["color"],
                                  linewidth=1.25 if shifted else 0.8,
                                  alpha=1.0 if shifted else 0.65,
                                  marker="o", markersize=3.0,
                                  zorder=3 if shifted else 2)
                line.set_gid(f"rank:{dataset}:{method}")
            for side in (0, last):
                for rank, method in enumerate(orders[side], start=1):
                    axis.annotate(display_name(method), xy=(side, rank), fontsize=8,
                                  ha="left" if side else "right", va="center",
                                  xytext=(4 if side else -4, 0), textcoords="offset points",
                                  color=_curve_style(method)["color"])
            axis.set(xlim=(-0.85, last+0.85), ylim=(7.35, 0.65),
                     xticks=range(len(RANK_KEYS)))
            axis.set_xticklabels(RANK_LABELS, fontsize=8, linespacing=1.15)
            axis.set_yticks(range(1, len(orders[0])+1) if column == 0 else [])
            axis.tick_params(axis="both", length=0, pad=3, labelsize=8)
            axis.spines[["top", "right", "left", "bottom"]].set_visible(False)
            _figure_text(figure, x-5, top+24,
                         f"({chr(97+panel)}) {CURVE_CORPUS_NAME[dataset]}",
                         fontsize=9, weight="bold")
            for side, rho_key in enumerate(RANK_RHO[1:], start=1):
                label = axis.annotate(r"$\rho=%.3f$" % report[dataset][rho_key],
                                      xy=(side, 1), xycoords=("data", "axes fraction"),
                                      xytext=(0, 8), textcoords="offset points",
                                      ha="center", va="center", fontsize=8)
                label.set_gid(f"rank-rho:{dataset}:{rho_key}")

        key_row, key_column = divmod(len(datasets), 3)
        x = 18+172*key_column
        top = height-33-128*key_row
        _figure_text(figure, x-5, top+24, "Reading the rankings", fontsize=9, weight="bold")
        notes = (
            (8, "Nearest grid: point nearest"),
            (-3, "native retention."),
            (-18, r"Area: mean normalized area over $r$."),
            (-33, "Maximum: mean post-hoc peak."),
            (-48, "ρ: versus nearest-grid ranking."),
            (-63, "Rank 1 = highest MESR."),
            (-78, "prefix: earliest events per block."),
            (-93, "Thicker paths: changing rank."),
        )
        for offset, note in notes:
            _figure_text(figure, x-5, top+offset, note)
        figure.canvas.draw()
    return figure


def rank_instability(path: Path, datasets: Sequence[str] = RANK_DATASETS) -> Path:
    """Render the three summaries per corpus in a readable three-plus-two arrangement."""
    return _save_figure(build_rank_instability(datasets), path)


def native_oracle_points(dataset: str, hot_pixel_removal: bool = False) -> List[Dict]:
    """Per evaluable cell: retained purity and MESR, the filter's minus its matched oracle's.

    Purity is the signal share of the `m` events MESR scores, the same count on both sides, so
    the difference is `(TP_filter - TP_oracle) / m`, never positive by the oracle's dominance.
    """

    suffix = "_hotpixel" if hot_pixel_removal else ""
    payload = json.loads((RESULTS / f"native_oracle_{dataset}{suffix}.json").read_text())
    return [{"recording": cell["recording"], "method": cell["method"],
             "purity": (cell["tp_scored"] - cell["oracle_tp_scored"]) / cell["scored"],
             "mesr": cell["mesr"] - cell["oracle_mesr"],
             "reversal": bool(cell["label_class"] == "strict" and cell["mesr_win"])}
            for cell in payload["cells"]]


def build_native_oracle_view():
    """Build the unchanged cell coordinates with separate dataset and preprocessing keys."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    with plt.rc_context(FIGURE_STYLE):
        figure = plt.figure(figsize=ORACLE_INCHES)
        axis = _axes_in_points(figure, 37, 60, 203, 105)
        counts = {}
        lowest, highest, bottom = 0.0, 0.0, 0.0
        for removal in (False, True):
            reversals = cells = 0
            for dataset, marker, color, _label in ORACLE_CORPORA:
                points = native_oracle_points(dataset, removal)
                x = [p["purity"] for p in points]
                y = [p["mesr"] for p in points]
                dots = axis.scatter(x, y, marker=marker, s=18 if removal else 12,
                                    linewidths=0.75 if removal else 0.45,
                                    facecolors="none" if removal else color, edgecolors=color,
                                    alpha=0.95 if removal else 0.65,
                                    zorder=4 if removal else 3)
                dots.set_gid(f"oracle:{dataset}:{'removed' if removal else 'released'}")
                reversals += sum(p["reversal"] for p in points)
                cells += len(points)
                lowest, highest = min(lowest, min(x)), max(highest, max(y))
                bottom = min(bottom, min(y))
            counts[removal] = (reversals, cells)

        left, top = lowest*1.05, highest+0.025
        shade = axis.fill_between([left, 0.0], 0.0, top, color="#E7EBEE", alpha=0.65,
                                  linewidth=0, zorder=0)
        shade.set_gid("oracle:reversal-quadrant")
        for name, line in (("y", axis.axhline(0, color="#555555", lw=0.7, ls=(0, (3, 2)), zorder=1)),
                           ("x", axis.axvline(0, color="#555555", lw=0.7, ls=(0, (3, 2)), zorder=1))):
            line.set_gid(f"oracle:zero:{name}")
        axis.set(xlim=(left, -lowest*0.04), ylim=(bottom-0.018, top),
                 xticks=[-0.25, -0.15, -0.05, 0.0], yticks=[0, 0.1, 0.2, 0.3],
                 xlabel="Retained-purity difference", ylabel="MESR difference")
        axis.set_xticklabels(["−0.25", "−0.15", "−0.05", "0.00"])
        axis.tick_params(labelsize=8)
        _figure_text(figure, 138.5, 174, "Filter − matched oracle", ha="center", fontsize=9,
                     weight="bold")

        corpus_handles = [Line2D([], [], marker=marker, linestyle="none", markersize=4,
                                 markerfacecolor=color, markeredgecolor=color, label=label)
                          for _dataset, marker, color, label in ORACLE_CORPORA]
        figure.legend(handles=corpus_handles, loc="center", ncol=2, frameon=False,
                      bbox_to_anchor=(0.5, 23/(ORACLE_INCHES[1]*72)), fontsize=8,
                      handletextpad=0.5, columnspacing=1.7, borderpad=0)
        state_handles = [Line2D([], [], marker="o", linestyle="none", markersize=3.5,
                                markerfacecolor="#666666", markeredgecolor="#666666",
                                label="As released"),
                         Line2D([], [], marker="o", linestyle="none", markersize=4.2,
                                markerfacecolor="none", markeredgecolor="#666666",
                                label="Busiest pixels removed")]
        figure.legend(handles=state_handles, loc="center", ncol=2, frameon=False,
                      bbox_to_anchor=(0.5, 10/(ORACLE_INCHES[1]*72)), fontsize=8,
                      handletextpad=0.5, columnspacing=1.2, borderpad=0)
        figure.canvas.draw()
    return figure


def native_oracle_view(path: Path) -> Path:
    """Render filter-minus-oracle cell differences without placing prose over observations."""
    return _save_figure(build_native_oracle_view(), path)


def main() -> List[Path]:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    # matplotlib subsets its fonts through fontTools, which narrates every glyph at INFO.
    for noisy in ("matplotlib", "fontTools"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    written = [
        protocol_view(FIGURES / "fig_protocol_view.pdf"),
        rank_instability(FIGURES / "fig_rank_instability.pdf"),
        native_oracle_view(FIGURES / "fig_native_oracle.pdf"),
    ]
    for path in written:
        print(path)
    return written


if __name__ == "__main__":
    main()
