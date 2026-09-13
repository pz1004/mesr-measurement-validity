"""The one published calibration of ESR, and the limits of what it shows.

E-MLB validate ESR by adding known noise to a fixed stream and reporting that the score
falls. DND21 injects uniform noise into two real recordings at five rates, so the unfiltered
row of `benchmark_dnd21.json` is a controlled-addition series of the same kind and the
supplement quotes it as a reproduction.

These tests guard that quotation. They do **not** assert any mechanism: the paper offers none,
because the obvious candidate is ruled out by `app:properties` -- ESR is not monotone in
concentration, so "uniform noise dilutes concentration" does not explain the direction. A
test asserting a mechanism here would be asserting something the paper cannot support.
"""

import json
from pathlib import Path

import pytest

ARTIFACT = Path(__file__).resolve().parents[1] / "results" / "benchmark_dnd21.json"

#: What the supplement's table prints, at 1 and 10 Hz/px.
PUBLISHED_ENDPOINTS = {"hotel-bar": (1.1754, 0.6580), "driving": (0.7577, 0.6164)}


def series_by_base_scene():
    if not ARTIFACT.is_file():
        pytest.skip("benchmark_dnd21.json not generated")
    out = {}
    for rec in json.loads(ARTIFACT.read_text())["records"]:
        rate, _, scene = rec["recording"].split("/")[-1].partition("_")
        out.setdefault(scene, []).append((int(rate.rstrip("hz")), rec["raw_mesr"]))
    return {scene: sorted(points) for scene, points in out.items()}


def test_the_injected_series_is_two_base_scenes_at_five_rates():
    """The unit of the reproduction. Ten recordings, but two independent captures."""

    series = series_by_base_scene()
    assert sorted(series) == ["driving", "hotel-bar"]
    for points in series.values():
        assert [r for r, _ in points] == [1, 3, 5, 7, 10]


def test_unfiltered_mesr_falls_monotonically_with_the_injected_rate():
    """The reproduction itself: strictly decreasing on both base scenes."""

    for scene, points in series_by_base_scene().items():
        values = [v for _, v in points]
        assert all(b < a for a, b in zip(values, values[1:])), f"{scene}: {values}"


def test_the_published_endpoints_match_the_supplement_table():
    for scene, (first, last) in PUBLISHED_ENDPOINTS.items():
        values = [v for _, v in series_by_base_scene()[scene]]
        assert values[0] == pytest.approx(first, abs=5e-5)
        assert values[-1] == pytest.approx(last, abs=5e-5)
