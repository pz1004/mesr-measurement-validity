"""The repeatability summary App. I quotes: per-level medians and the repetition draws.

Both are computed from the stored per-repetition readings, so they need no corpus.
"""

from __future__ import annotations

import numpy as np

from dataset_assessment.emlb_repeatability import (lighting_of, nd_level_of,
                                                   repetition_draws, summarise)


def _records(cell: str, readings) -> list:
    return [{"recording": f"{cell}-{i + 1}", "cell": cell, "events": 1, "mesr": float(v)}
            for i, v in enumerate(readings)]


def test_levels_and_lighting_are_read_from_the_cell_name():
    assert nd_level_of("E-MLB/N-END/Car_Night-ND00") == "ND00"
    assert lighting_of("E-MLB/N-END/Car_Night-ND00") == "N-END"
    records = (_records("E-MLB/D-END/A-ND00", [1.0, 1.2, 1.1])
               + _records("E-MLB/N-END/B-ND00", [1.0, 1.0, 1.4])
               + _records("E-MLB/D-END/A-ND64", [2.0, 2.1, 2.0]))
    summary = summarise(records)
    by_level, by_light = summary["median_spread_by_nd_level"], summary["median_spread_by_lighting"]
    assert sorted(by_level) == ["ND00", "ND64"] and sorted(by_light) == ["D-END", "N-END"]
    assert np.isclose(by_level["ND00"], np.median([0.2, 0.4]))
    assert np.isclose(by_level["ND64"], 0.1)
    assert np.isclose(by_light["D-END"], np.median([0.2, 0.1]))
    assert np.isclose(by_light["N-END"], 0.4)


def test_repetition_draws_vanish_without_spread_and_are_seeded():
    flat = [{"mesr": [1.0, 1.0, 1.0]}, {"mesr": [2.0, 2.0, 2.0]}]
    still = repetition_draws(flat, draws=500)
    assert still["sd"] == 0.0 and still["central_95_width"] == 0.0
    assert still["first_repetition_mean"] == 1.5

    varied = [{"mesr": [0.0, 1.0, 2.0]}, {"mesr": [1.0, 1.0, 1.0]}, {"mesr": [3.0, 1.0]}]
    first, second = repetition_draws(varied, draws=500), repetition_draws(varied, draws=500)
    assert first == second                      # seeded: the quoted figures regenerate
    assert first["cells"] == 2                  # the incomplete cell is left out
    assert 0.0 < first["sd"] < 1.0 and first["central_95_width"] > first["sd"]
