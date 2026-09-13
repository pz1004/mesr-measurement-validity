"""The two numbers `tab:emlb` and the abstract print that no artifact stored.

Both were correct when checked by hand and neither could be found in `NUMBERS.md`, which
is the same defect the paper charges others with. These tests pin the derivations so the
digest cannot drift from the manuscript silently.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from dataset_assessment.esr import SLICE

RESULTS = Path(__file__).resolve().parents[1] / "results"
CORPORA = ("dnd21", "dvsclean", "emlb", "dvsd22", "pure_ba")


def _load(name: str) -> dict:
    return json.loads((RESULTS / name).read_text())


def _evaluable(records: list) -> list:
    return [r for r in records
            if isinstance(r.get("raw_mesr"), float) and np.isfinite(r["raw_mesr"])]


def test_the_abstract_cell_count_is_derivable_from_the_five_released_corpora():
    total = sum(len([m for r in _evaluable(_load(f"benchmark_{d}.json")["records"])
                     for m in r["methods"].values() if "curve" in m])
                for d in CORPORA)
    assert total == 3556


def test_the_pure_ba_shortfall_is_short_recordings_and_not_a_dropped_corpus():
    """41 records, 26 counted. The gap is the evaluability rule, not a selection."""

    records = _load("benchmark_pure_ba.json")["records"]
    assert len(records) == 41
    assert len(_evaluable(records)) == 26
    too_short = [r for r in records if r["events"] < SLICE]
    assert len(too_short) == 15
    # The two rules agree: every dropped recording is dropped because it is short.
    assert {r["recording"] for r in too_short} == (
        {r["recording"] for r in records} - {r["recording"] for r in _evaluable(records)})


def test_the_sibling_project_file_is_not_part_of_the_five_corpora():
    """`benchmark_emlb_native3d.json` would add 384 cells. The paper's count excludes it."""

    extra = _load("benchmark_emlb_native3d.json")["records"]
    assert sorted({k for r in extra for k in r["methods"]}) == [
        "native3d", "native3d_gated", "random_null", "raw"]
    assert len([m for r in extra for m in r["methods"].values() if "curve" in m]) == 384


def test_tab_emlb_prints_the_scene_clustered_interval_not_the_recording_one():
    """The caption promises scene clusters. Two intervals exist in the release for the
    identical row, and they differ; this pins which file the table reads."""

    marginals = _load("paired_contrasts.json")["marginals"]
    recording_level = _load("rank_analysis.json")["emlb"][
        "delta_over_raw_at_native_eligible"]

    printed = {"red": (0.350, 0.579), "ynoise": (0.154, 0.278), "dwf": (0.091, 0.135),
               "evflow": (0.041, 0.168), "ts": (0.058, 0.134), "knoise": (0.028, 0.079)}
    for method, (lo, hi) in printed.items():
        clustered = marginals[method]
        assert round(clustered["lo"], 3) == lo
        assert round(clustered["hi"], 3) == hi
        # Same point estimate on both sides, so only the interval distinguishes them.
        assert clustered["mean"] == recording_level[method]["mean"]
        assert clustered["n_recordings"] == recording_level[method]["n"]
        # And the clustered one is the wider: fewer independent units.
        assert clustered["lo"] < recording_level[method]["lo"]
        assert clustered["hi"] > recording_level[method]["hi"]
