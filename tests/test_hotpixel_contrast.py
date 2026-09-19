"""The hot-pixel contrast must compare one cohort before and after, and read what the paper reads.

`hotpixel_contrast` puts the paper's specificity and blank-sample numbers beside their
counterparts with the busiest 0.1% of pixels removed. Three properties keep that honest:

* **One cohort.** A recording evaluable on one side only never enters a before/after mean,
  so a change is a change in the stream, not in who is averaged.
* **The as-released side is the published one.** The paper's hand-traced numbers must come
  back from the module: the blank's paired RED-null +1.3052 and the specificity counts.
* **The null is the same draw in both hot-pixel runs.** Pure_BA's filters were rerun with the
  nulls; with the seed and preprocessing fixed, the nulls must match the earlier run exactly.
"""

import json

import pytest

from dataset_assessment.analyze import RESULTS
from dataset_assessment.hotpixel_contrast import blank, deltas, paired, specificity


def _payload(rows, hot=False):
    """rows: recording -> method -> {r: mesr}, with raw_mesr = value at r = 1."""

    records = []
    for rec, methods in rows.items():
        records.append({"recording": rec, "methods": {
            m: {"raw_mesr": curve[1.0],
                "curve": [{"r": r, "mesr": v, "evaluable": v is not None}
                          for r, v in curve.items() if v is not None]}
            for m, curve in methods.items()}})
    return {"retentions": [0.5, 1.0], "records": records, "hot_pixel_removal": hot}


def test_one_cohort_before_and_after():
    before = {"a": 1.0, "b": 3.0}
    after = {"a": 0.0, "c": 9.0}                  # b and c are one-sided and never enter
    assert paired(before, after) == {"n": 1, "before": 1.0, "after": 0.0, "n_drop": 1}


def test_specificity_counts_the_null_on_the_shared_cohort():
    null = lambda v: {"random_null": {0.5: v, 1.0: 0.0}}
    before = _payload({"a": null(0.2), "b": null(0.4)})
    after = _payload({"a": null(-0.1), "b": null(None)}, hot=True)
    out = specificity(before, after, 0.5)
    assert out["n_positive_before"] == 1 and out["n_positive_after"] == 0
    assert out["at_r"]["n"] == 1 and out["at_r"]["before"] == pytest.approx(0.2)


def test_blank_margin_is_filter_minus_null_per_recording():
    rows = {"a": {"random_null": {0.5: 0.1, 1.0: 0.0}, "raw": {0.5: 0.1, 1.0: 0.0},
                  **{m: {0.5: 0.1, 1.0: 0.0} for m in ("dwf", "evflow", "knoise", "ts", "ynoise")},
                  "red": {0.5: 1.1, 1.0: 0.0}}}
    out = blank(_payload(rows), _payload(rows, hot=True), retentions=(0.5,))["0.50"]
    assert out["best"] == {"before": "red", "after": "red"}
    assert out["before"]["red"]["minus_null"] == pytest.approx(1.0)


def _load(name):
    path = RESULTS / name
    if not path.exists():
        pytest.skip(f"{name} not present")
    return json.loads(path.read_text())


def test_as_released_side_reproduces_the_published_blank():
    payload = _load("benchmark_pure_ba.json")
    red, null = deltas(payload, "red", 0.60), deltas(payload, "random_null", 0.60)
    assert len(red) == 26
    assert sum(red[k] - null[k] for k in red) / 26 == pytest.approx(1.3052, abs=5e-5)


@pytest.mark.parametrize("dataset,hot,expect", [
    ("dvsd22", "benchmark_dvsd22_hotpixel.json", (19, 5)),
    ("pure_ba", "benchmark_pure_ba_hotpixel.json", (19, 0)),
])
def test_specificity_counts_match_the_paper(dataset, hot, expect):
    out = specificity(_load(f"benchmark_{dataset}.json"), _load(hot), 0.05)
    assert (out["n_positive_before"], out["n_positive_after"]) == expect


def test_rerun_nulls_are_the_earlier_draw():
    old, new = _load("benchmark_pure_ba_hotpixel.json"), _load("benchmark_pure_ba_hotpixel_all.json")
    if not any("random_null" in r["methods"] for r in new["records"]):
        pytest.skip("rerun has no random_null yet")
    new_by = {r["recording"]: r["methods"] for r in new["records"]}
    for rec in old["records"]:
        for m in ("raw", "random_null"):
            assert rec["methods"][m] == new_by[rec["recording"]][m]
