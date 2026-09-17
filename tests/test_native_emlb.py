"""E-MLB scored on the filters' own outputs, beside a nonselective control matched to them.

Table IV used to read each filter at the grid point nearest its native retention, through a
per-block quota that recovers the filter's retained *count* but never its retained *set*.
`native_emlb` removes the quota. Three properties make its contrasts mean what they say:

* **The control is matched block by block.** The random control draws exactly as many events
  from each input block as the filter's scored prefix does, so filter and control differ in
  *which* events they keep and in nothing else.
* **Too short is unevaluable, not refilled.** A native output under one complete slice has no
  MESR; the cell is reported as unevaluable rather than scored at a retention the filter
  never chose.
* **Keeping everything is the reference.** A filter that keeps the whole stream scores the
  unfiltered input exactly, so its Delta-over-Raw is zero.
"""

import numpy as np

from dataset_assessment.native_emlb import matched_random_mask, measure_recording
from dataset_assessment.native_oracle import block_counts, scored_prefix
from dataset_assessment.readers import Recording


def _recording(n=120_000, width=64, height=48, seed=20260917):
    rng = np.random.default_rng(seed)
    events = np.stack([np.arange(n), rng.integers(0, width, n), rng.integers(0, height, n),
                       np.ones(n, dtype=np.int64)], axis=1)
    return Recording(events=events, labels=None, width=width, height=height,
                     name="E-MLB/D-END/Synthetic-ND00-1", synthetic=True)


def test_the_control_draws_the_prefix_count_from_every_block():
    rng = np.random.default_rng(1)
    accepted = rng.random(50_000) < 0.4
    prefix = scored_prefix(accepted, slice_size=1_000)
    counts = block_counts(prefix, block=1_024)
    control = matched_random_mask(counts, len(accepted), block=1_024,
                                  rng=np.random.default_rng(2))
    np.testing.assert_array_equal(block_counts(control, block=1_024), counts)
    assert control.sum() == prefix.sum()


def test_the_control_is_label_blind_and_seeded():
    counts = np.array([3, 0, 7])
    a = matched_random_mask(counts, 30, block=10, rng=np.random.default_rng(5))
    b = matched_random_mask(counts, 30, block=10, rng=np.random.default_rng(5))
    np.testing.assert_array_equal(a, b)


def test_a_native_output_under_one_slice_is_unevaluable():
    rec = _recording()
    keep_few = lambda method, r: np.where(np.arange(len(r.events)) < 20_000, 0.0, 1.0)
    result = measure_recording(rec, methods=("sparse",), scorer=keep_few)
    assert result["cells"] == []
    assert result["unevaluable"][0]["method"] == "sparse"
    assert result["unevaluable"][0]["native_kept"] == 20_000


def test_keeping_everything_scores_the_input_exactly():
    rec = _recording()
    keep_all = lambda method, r: np.zeros(len(r.events))
    result = measure_recording(rec, methods=("all",), scorer=keep_all)
    cell = result["cells"][0]
    assert cell["native_retention"] == 1.0
    assert cell["delta_over_raw"] == 0.0
    # The matched control of a keep-all mask is the same prefix, so it scores identically.
    assert cell["null_delta_over_raw"] == 0.0
    assert cell["filter_minus_null"] == 0.0
