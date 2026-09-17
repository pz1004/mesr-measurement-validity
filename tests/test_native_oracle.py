"""The native-output comparison must be matched without a quota, and the oracle must bound.

`label_quality` matches filter and oracle by giving both the same per-block quota, which is
why `native_mask` could show that neither side is the released filter. `native_oracle` drops
the quota: it takes the filter's own mask, keeps the part MESR reads, and builds the oracle to
match that prefix block by block.

Two invariants carry the whole comparison, and a regression in either would silently restore
an unmatched or an unbounded contrast:

* **Equal scored counts.** Both sides hold exactly `m = SLICE * (|F| // SLICE)` events, so
  neither has a tail of its own to truncate. This is why the quota is built from the scored
  subset and not from the full native mask.
* **Oracle dominance.** `TP_oracle = sum_b min(a_b, S_b) >= TP_prefix`, so the `impossible`
  class cannot occur. A cell landing there means the mask or the oracle is wrong, not that a
  filter out-selected the oracle.
"""

import numpy as np

from dataset_assessment.label_quality import _classify, _quality
from dataset_assessment.native_oracle import (available_signal, block_counts, oracle_mask,
                                              scored_prefix)


def _mask(pattern):
    """Boolean mask from a 0/1 sequence."""

    return np.array(pattern, dtype=bool)


def test_the_scored_prefix_is_a_complete_slice_multiple():
    accepted = _mask([1] * 25 + [0] * 75)
    prefix = scored_prefix(accepted, slice_size=10)
    assert prefix.sum() == 20                       # 25 accepted, 2 complete slices of 10
    assert not (prefix & ~accepted).any()           # a subset of what the filter accepted


def test_the_prefix_takes_the_earliest_accepted_events():
    """MESR scores consecutive complete slices from the start, so the prefix must too."""

    accepted = _mask([0, 1, 1, 0, 1, 1, 0, 1])
    prefix = scored_prefix(accepted, slice_size=2)
    assert list(np.flatnonzero(prefix)) == [1, 2, 4, 5]   # the first 4 of 5 accepted


def test_a_filter_under_one_slice_is_unevaluable_rather_than_padded():
    accepted = _mask([1] * 9 + [0] * 91)
    assert scored_prefix(accepted, slice_size=10).sum() == 0


def test_the_oracle_keeps_exactly_the_quota_in_every_block():
    labels = np.array([0, 1, 0, 1] * 4)                   # 16 events, blocks of 4
    counts = np.array([1, 2, 3, 4])
    keep = oracle_mask(labels, counts, block=4)
    per_block = [int(keep[s:s + 4].sum()) for s in range(0, 16, 4)]
    assert per_block == [1, 2, 3, 4]
    assert keep.sum() == counts.sum()


def test_the_oracle_prefers_signal_and_breaks_ties_by_arrival():
    """Signal is label 0, so a stable argsort puts it first and keeps arrival order within."""

    labels = np.array([1, 0, 1, 0])                       # noise, signal, noise, signal
    keep = oracle_mask(labels, np.array([2]), block=4)
    assert list(np.flatnonzero(keep)) == [1, 3]           # both signal events, in order

    keep_three = oracle_mask(labels, np.array([3]), block=4)
    # Signal exhausted at 2, so the third is the earliest noise event.
    assert list(np.flatnonzero(keep_three)) == [0, 1, 3]


def test_the_oracle_retains_the_sum_of_min_quota_signal_bound():
    """`TP_oracle = sum_b min(a_b, S_b)` -- the identity the runtime assertion checks."""

    rng = np.random.default_rng(20260915)
    labels = rng.integers(0, 2, size=64)
    counts = np.array([3, 8, 1, 6])                       # blocks of 16
    keep = oracle_mask(labels, counts, block=16)
    available = available_signal(labels, block=16)
    assert int((labels[keep] == 0).sum()) == int(np.minimum(counts, available).sum())


def test_the_oracle_is_never_out_selected_by_the_filter():
    """The `impossible` class cannot occur, on any mask the filter could produce."""

    rng = np.random.default_rng(20260916)
    labels = rng.integers(0, 2, size=200)
    for _ in range(40):
        accepted = _mask(rng.integers(0, 2, size=200))
        prefix = scored_prefix(accepted, slice_size=10)
        if not prefix.any():
            continue
        keep = oracle_mask(labels, block_counts(prefix, block=20), block=20)
        mq, oq = _quality(labels, prefix, slice_size=10), _quality(labels, keep, slice_size=10)
        assert mq["kept"] == oq["kept"]                    # matched without any quota
        assert _classify(mq, oq) in {"strict", "tie"}


def test_block_counts_sum_to_the_mask():
    mask = _mask([1, 0, 1, 1, 0, 0, 1, 0, 1])
    counts = block_counts(mask, block=4)
    assert list(counts) == [3, 1, 1]
    assert counts.sum() == mask.sum()


def test_hot_pixel_removal_is_applied_to_the_input_after_the_cap():
    """The preprocessed run removes pixels from the *input*, so labels must follow the events.

    Deleting pixels from an existing filter output would evaluate a different pipeline; the
    filters have to see the trimmed stream and the oracle has to be rebuilt from it.
    """

    from dataset_assessment.native_oracle import prepare
    from dataset_assessment.readers import Recording

    rng = np.random.default_rng(20260917)
    n = 4000
    xs = rng.integers(0, 16, size=n)
    ys = rng.integers(0, 16, size=n)
    xs[::4], ys[::4] = 5, 7                                # one pixel carries a quarter
    events = np.stack([np.arange(n), xs, ys, np.ones(n, dtype=np.int64)], axis=1)
    labels = (rng.random(n) < 0.3).astype(np.int64)
    rec = Recording(events=events, labels=labels, width=16, height=16, name="synthetic",
                    synthetic=True)

    capped, none = prepare(rec, max_events=3000, hot_pixel_removal=False)
    assert none is None and len(capped.events) == 3000

    trimmed, summary = prepare(rec, max_events=3000, hot_pixel_removal=True)
    assert summary["events_before"] == 3000                 # the cap came first
    assert summary["events_removed"] > 0
    assert len(trimmed.events) == len(trimmed.labels) == summary["events_after"]
    hot = (capped.events[:, 1] == 5) & (capped.events[:, 2] == 7)
    assert not ((trimmed.events[:, 1] == 5) & (trimmed.events[:, 2] == 7)).any()
    np.testing.assert_array_equal(trimmed.labels, capped.labels[~hot])
