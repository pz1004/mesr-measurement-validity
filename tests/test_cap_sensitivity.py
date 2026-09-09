"""Delta-over-Raw was asserted to be cap-invariant. These tests encode how that was tested.

The assertion licensed placing capped classical rows beside an uncapped EDformer row. It was
never measured, and it is false. Two things move when the cap changes and they must not be
conflated:

* the value of Delta-over-Raw at a *fixed* retention, and
* which retentions are evaluable at all -- fewer events means fewer complete slices, so the
  measurable floor rises and a native operating point below it gets snapped somewhere else.

`fixed_retention_effect` isolates the first; `evaluable_range_changes` reports whether the
second is even held constant. A regression that silently merged them would let the at-native
spread be read as a pure cap effect, which it is not.
"""

import json

import numpy as np
import pytest

from dataset_assessment.cap_sensitivity import (UNIT_OF_SCALE, compare,
                                                evaluable_range_changes,
                                                fixed_retention_effect, native_deltas)


def _curve(value, retentions=(0.05, 0.5, 0.95, 1.0), evaluable_from=0.0):
    return [{"r": r, "mesr": value, "kept_events": 10 ** 6,
             "evaluable": r >= evaluable_from} for r in retentions]


def _record(name, methods, events=10 ** 6):
    return {"recording": name, "events": events, "methods": methods}


def _payload(cap, raw_mesr, filt_mesr, evaluable_from=0.0, events=10 ** 6):
    methods = {
        "raw": {"method": "raw", "native_retention": 1.0, "is_null": True,
                "evaluable_range": [evaluable_from, 1.0], "curve": _curve(raw_mesr)},
        "dwf": {"method": "dwf", "native_retention": 0.5, "is_null": False,
                "evaluable_range": [evaluable_from, 1.0],
                "curve": _curve(filt_mesr, evaluable_from=evaluable_from)},
    }
    return {"dataset": "emlb", "max_events": cap,
            "records": [_record("rec-1", methods, events)]}


def test_native_deltas_is_filter_minus_raw():
    d = native_deltas(_payload(10 ** 6, raw_mesr=0.8, filt_mesr=1.0))
    assert d["rec-1"]["dwf"] == pytest.approx(0.2)


def test_raw_has_zero_delta_against_itself():
    """A self-check on the whole comparison: `raw` must move by exactly zero at every cap."""

    result = compare_payloads(_payload(5 * 10 ** 5, 0.8, 1.0), _payload(10 ** 6, 0.9, 1.2))
    assert result["per_method"]["raw"]["mean_spread_across_caps"] == pytest.approx(0.0)


def compare_payloads(*payloads, tmp=None):
    """`compare` reads paths, so round-trip these through files."""

    import tempfile
    from pathlib import Path

    tmp = tmp or Path(tempfile.mkdtemp())
    paths = []
    for i, p in enumerate(payloads):
        q = tmp / f"run{i}.json"
        q.write_text(json.dumps(p))
        paths.append(q)
    return compare(paths)


def test_a_genuinely_invariant_quantity_reports_zero_spread():
    """If Delta really did not move with the cap, the summary must say so."""

    a = _payload(5 * 10 ** 5, raw_mesr=0.8, filt_mesr=1.0)
    b = _payload(10 ** 6, raw_mesr=0.5, filt_mesr=0.7)      # both shift, difference constant
    result = compare_payloads(a, b)
    assert result["per_method"]["dwf"]["max_spread"] == pytest.approx(0.0)
    assert result["spread"]["share_above_unit_of_scale"] == pytest.approx(0.0)


def test_a_moving_delta_is_reported_against_the_unit_of_scale():
    a = _payload(5 * 10 ** 5, raw_mesr=0.8, filt_mesr=1.0)   # delta 0.2
    b = _payload(10 ** 6, raw_mesr=0.8, filt_mesr=1.3)       # delta 0.5
    result = compare_payloads(a, b)
    assert result["per_method"]["dwf"]["max_spread"] == pytest.approx(0.3)
    assert result["spread"]["max_in_units_of_scale"] == pytest.approx(0.3 / UNIT_OF_SCALE)
    assert result["spread"]["share_above_unit_of_scale"] == pytest.approx(1.0)


def test_untruncated_recordings_are_excluded_from_the_summary():
    """A recording shorter than the smallest cap cannot show cap sensitivity."""

    short = 1000
    a = _payload(5 * 10 ** 5, 0.8, 1.0, events=short)
    b = _payload(10 ** 6, 0.8, 1.3, events=short)
    result = compare_payloads(a, b)
    assert result["cells_compared"] == 2          # raw and dwf
    assert result["cells_bound_by_cap"] == 0      # neither is truncated


def test_evaluable_range_change_is_reported_separately():
    """Zero identical ranges means the at-native spread is not a pure cap effect."""

    a = _payload(5 * 10 ** 5, 0.8, 1.0, evaluable_from=0.10)
    b = _payload(10 ** 6, 0.8, 1.0, evaluable_from=0.05)
    er = evaluable_range_changes([a, b])
    assert er["cells"] == 2 and er["identical_range"] == 0

    same = evaluable_range_changes([a, _payload(10 ** 6, 0.8, 1.0, evaluable_from=0.10)])
    assert same["identical_range"] == same["cells"]


def test_fixed_retention_effect_holds_r_constant():
    """At a fixed r the floor cannot contaminate the difference."""

    a = _payload(5 * 10 ** 5, raw_mesr=0.8, filt_mesr=1.0)   # delta 0.2 at every r
    b = _payload(10 ** 6, raw_mesr=0.8, filt_mesr=1.0)
    got = fixed_retention_effect([], [a, b])
    for v in got.values():
        assert v["max"] == pytest.approx(0.0)

    c = _payload(10 ** 6, raw_mesr=0.8, filt_mesr=1.5)       # delta 0.7
    got = fixed_retention_effect([], [a, c])
    for v in got.values():
        assert v["max"] == pytest.approx(0.5)


def test_measured_result_refutes_the_invariance_claim():
    """The artifact must keep saying what the paper now says: it is not cap-invariant."""

    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "results" / "cap_sensitivity.json"
    if not path.is_file():
        pytest.skip("cap-sensitivity run not generated")
    r = json.loads(path.read_text())
    er = r["evaluable_range"]
    assert er["identical_range"] / er["cells"] < 0.25, (
        "the at-native reading assumes the floor moves on the large majority of cells; if "
        "it ever settles, the extremes must be re-attributed to the value rather than the "
        "grid")
    assert r["spread"]["share_above_unit_of_scale"] > 0.5
    assert r["at_fixed_retention"]["r=0.50"]["share_above_unit_of_scale"] > 0.1, (
        "even with r held fixed the cap must still move Delta for a real share of cells")


def test_the_clean_pair_moves_a_corpus_mean_past_the_unit_of_scale():
    """1M vs 2M is the pair with an identical grid, so it isolates the cap alone.

    The paper reports that one method's *corpus mean* moves past the published gap there.
    That claim was false on a 48-recording subset and true on all 384, so it is pinned to
    the artifact rather than left to a reader's memory of an earlier run.
    """

    from pathlib import Path

    path = (Path(__file__).resolve().parents[1] / "results" / "cap_sensitivity"
            / "pair_1M_vs_2M.json")
    if not path.is_file():
        pytest.skip("cap-sensitivity run not generated")
    p = json.loads(path.read_text())
    er = p["evaluable_range"]
    assert er["identical_range"] == er["cells"], (
        "this pair is only interpretable as the cap alone while the grid is identical")

    # The appendix quotes the cohort the E-MLB table averages over -- all 384 recordings --
    # not the subset of cells the cap can bind. The two differ: on the bindable cells RED
    # moves 0.0395, on the printed cohort 0.0317. Quoting the first as if it were the second
    # is the error this pin exists to catch.
    over_all = p["means_over_all_recordings"]
    assert over_all["recordings"] == 384
    moves = sorted(((v["spread_across_caps"], m)
                    for m, v in over_all["by_method"].items() if m != "raw"), reverse=True)
    assert moves[0][0] > UNIT_OF_SCALE, (
        "the appendix states one printed mean moves past the published gap between these caps")
    assert moves[1][0] < UNIT_OF_SCALE, (
        "the appendix attributes that movement to a single method")


def test_the_1m_column_reproduces_the_published_emlb_table():
    """`means_over_all_recordings` is only quotable beside Table V if it *is* Table V at 1M.

    These six values are transcribed from the article's E-MLB table. If the pipeline drifts,
    the cap columns stop being comparable with the printed ones and the appendix's framing
    silently breaks.
    """

    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "results" / "cap_sensitivity.json"
    if not path.is_file():
        pytest.skip("cap-sensitivity run not generated")
    printed = {"red": 0.8852, "ynoise": 0.2150, "dwf": 0.1117, "ts": 0.1025,
               "knoise": 0.0498, "evflow": 0.0231}
    by_method = json.loads(path.read_text())["means_over_all_recordings"]["by_method"]
    for method, expected in printed.items():
        got = by_method[method]["mean_by_cap"]["1000000"]
        assert got == pytest.approx(expected, abs=5e-5), method
