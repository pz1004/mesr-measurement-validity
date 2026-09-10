"""Appendix D claimed the swept curve at r = native is "exactly the filter". It is not.

`esr.retain_mask` keeps the same count in every 10,240-event block. A binary filter accepts a
count that varies from block to block -- that variation *is* the filter responding to scene
content. So the two retained sets agree only when every block happens to match, and the gap
is exactly `sum_b |a_b - k_b|`.

These tests pin the identity, pin the one case where recovery is exact, and pin the direction
that is easy to get backwards: below the global native rate the sweep still *adds* rejected
events in blocks where the filter was stingier than the quota.
"""

import numpy as np

from dataset_assessment.esr import BLOCK, retain_mask
from dataset_assessment.native_mask import (block_accept_counts, block_quota, compare_masks)


def _two_level(accepts_per_block):
    """A binary filter's score vector: 0 accepted, 1 rejected, accepted events first."""

    blocks = []
    for accepted in accepts_per_block:
        block = np.ones(BLOCK)
        block[:accepted] = 0.0
        blocks.append(block)
    return np.concatenate(blocks)


def test_block_quota_reproduces_retain_mask():
    scores = _two_level([4000, 6000])
    for r in (0.05, 0.3, 0.5, 0.95, 1.0):
        kept = retain_mask(scores, r)
        counts = [int(kept[s:s + BLOCK].sum()) for s in range(0, len(scores), BLOCK)]
        assert counts == list(block_quota(len(scores), r))


def test_the_symmetric_difference_equals_the_block_identity():
    """`|F_native XOR F_adapted| = sum_b |a_b - k_b|`, the reviewer's identity."""

    scores = _two_level([1000, 9000, 5000, 200])
    native = float((scores == 0.0).mean())
    for r in (native, 0.1, 0.25, 0.5, 0.9):
        result = compare_masks(scores, r)
        a_b, k_b = block_accept_counts(scores), block_quota(len(scores), r)
        assert result["symmetric_difference"] == int(np.abs(a_b - k_b).sum())
        assert result["symmetric_difference"] == result["predicted_by_block_identity"]


def test_a_uniform_filter_is_the_one_case_that_recovers_exactly():
    """Recovery needs equality in every block, which only a block-uniform filter gives."""

    uniform = _two_level([5120, 5120])                  # exactly half of each block
    assert compare_masks(uniform, 0.5)["symmetric_difference"] == 0

    lumpy = _two_level([10_240, 0])                     # same global rate, different blocks
    assert float((lumpy == 0.0).mean()) == 0.5
    assert compare_masks(lumpy, 0.5)["symmetric_difference"] == 10_240


def test_a_sub_native_retention_still_reintroduces_rejected_events():
    """The direction that is easy to state backwards, and Appendix D used to."""

    scores = _two_level([10_000, 100])                  # native rate ~0.493
    native = float((scores == 0.0).mean())
    result = compare_masks(scores, round(native, 2))
    assert result["blocks_reintroducing_rejects"] >= 1
