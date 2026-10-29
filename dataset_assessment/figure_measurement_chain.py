"""Render the measurement-chain diagram without reading or rewriting evidence.

    python -m dataset_assessment.figure_measurement_chain

The main lane is the scoring procedure. The adjacent labels identify validation
checks; they are not processing stages or classifier inputs.
"""

from __future__ import annotations

from pathlib import Path

from .analyze import FIGURES
from .figures_protocol import FIGURE_STYLE, _save_figure


CHAIN_INCHES = (3.45, 3.72)


def build_measurement_chain():
    """Build a publication-size workflow and a separate lane of validation checks."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch, Rectangle

    with plt.rc_context(FIGURE_STYLE):
        figure = plt.figure(figsize=CHAIN_INCHES)
        width, height = figure.get_size_inches() * 72
        axis = figure.add_axes([0, 0, 1, 1], xlim=(0, width), ylim=(0, height))
        axis.set_axis_off()
        text_color, note_color = "#20262B", "#465058"
        box_x, box_width = 78, 164
        centre = box_x + box_width / 2

        def text(x, y, label, *, size=8, bold=False, color=text_color, gid=None):
            artist = axis.text(x, y, label, ha="center", va="center", fontsize=size,
                               weight="bold" if bold else "normal", color=color,
                               linespacing=1.15)
            artist.set_gid(gid)
            return artist

        text(33, 255, "Validation\nchecks", size=9, bold=True)
        text(centre, 259, "Measurement chain", size=9, bold=True)
        # (identifier, bottom, top, title, secondary lines, adjacent check)
        stages = (
            ("input", 215, 242, "Released recording",
             ((223, "Event cap; busiest-pixel rule", 8),),
             "Blank input;\nadded noise"),
            ("selection", 170, 207, "Event selection",
             ((188, r"Native output or quota at $r$", 8),
              (177, "Quota blocks: 10,240 input events", 8)),
             "Controls;\nlabel oracle;\nretention sweep"),
            ("slices", 132, 162, "Complete event slices",
             ((142, r"$N=30{,}000$; drop incomplete tail", 8),),
             "Joint $(N,M)$\nsweep"),
            ("esr", 74, 124, "Event Structural Ratio (ESR)",
             ((99, r"$\mathrm{ESR}=\sqrt{\mathrm{ntss}\,\ell_n}$", 12),
              (82, r"$M=\lfloor 2N/3\rfloor=20{,}000$", 8)),
             "Declared-grid\ncheck"),
            ("mesr", 39, 66, "Recording mean (MESR)",
             ((47, "Mean ESR over complete slices", 8),), None),
            ("report", 4, 31, "Reporting",
             ((12, "MESR value or method ranking", 8),),
             "Task accuracy\ncomparison"),
        )
        for index, (name, bottom, top, title, notes, check) in enumerate(stages):
            box = Rectangle((box_x, bottom), box_width, top-bottom, linewidth=.6,
                            edgecolor="#7D858B", facecolor="#F5F7F8", zorder=1)
            box.set_gid(f"chain:box:{name}")
            axis.add_patch(box)
            text(centre, top-8, title, size=9, bold=True, gid=f"chain:title:{name}")
            for line, (y, note, size) in enumerate(notes):
                text(centre, y, note, size=size, color=note_color,
                     gid=f"chain:note:{name}:{line}")
            if check:
                middle = (bottom+top)/2
                text(33, middle, check, gid=f"chain:check:{name}")
                leader, = axis.plot([68, box_x], [middle, middle], color="#92999F",
                                    linewidth=.55, linestyle=(0, (2, 2)), zorder=0)
                leader.set_gid(f"chain:leader:{name}")
            if index < len(stages)-1:
                next_top = stages[index+1][2]
                arrow = FancyArrowPatch((centre, bottom), (centre, next_top),
                                        arrowstyle="-|>", mutation_scale=7,
                                        linewidth=.7, color="#465058",
                                        shrinkA=.5, shrinkB=.5, zorder=3)
                arrow.set_gid(f"chain:flow:{name}")
                axis.add_patch(arrow)
        figure.canvas.draw()
    return figure


def measurement_chain(path: Path) -> Path:
    """Write matching vector PDF and PNG previews at the manuscript's column width."""
    return _save_figure(build_measurement_chain(), path)


if __name__ == "__main__":
    print(measurement_chain(FIGURES / "fig_measurement_chain.pdf"))
