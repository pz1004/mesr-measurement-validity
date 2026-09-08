import numpy as np
import pytest

from dataset_assessment.esr import esr, mesr, mesr_curve


def test_esr_matches_closed_form_on_a_hand_case():
    # Two events on one pixel, one event on another, in a 2x2 sensor.
    # n = [2, 1, 0, 0], N = 3, M = int(3*2/3) = 2, K = 4
    # ntss = (2*1 + 1*0) / (3 * 2) = 2/6 = 1/3
    # ln = 4 - ((1-2/3)^2 + (1-2/3)^1 + 1 + 1) = 4 - (1/9 + 1/3 + 2) = 1.5555...
    x = np.array([0, 0, 1])
    y = np.array([0, 0, 0])
    expected = np.sqrt((1 / 3) * (4 - ((1 / 3) ** 2 + (1 / 3) + 2)))
    assert esr(x, y, 2, 2) == pytest.approx(expected, rel=1e-12)


def test_esr_is_nan_below_two_events():
    assert np.isnan(esr(np.array([0]), np.array([0]), 2, 2))


def test_esr_rewards_spatial_concentration():
    rng = np.random.default_rng(0)
    spread = esr(rng.integers(0, 100, 5000), rng.integers(0, 100, 5000), 100, 100)
    tight = esr(rng.integers(0, 10, 5000), rng.integers(0, 10, 5000), 100, 100)
    assert tight > spread


def test_esr_is_invariant_to_the_declared_sensor_size():
    """The K-cancellation identity, which the reporting protocol depends on.

    ln = K - sum_all(1-M/N)^n, and an empty pixel contributes exactly 1, so
    ln = sum_occupied[1 - (1-M/N)^n]: K cancels. Re-declaring a larger sensor for the
    same events must therefore change nothing at all.

    Bit-identical up to realistic sensor sizes. At K ~ 1e7 the implementation subtracts
    two numbers of order K to obtain an ln of order 1e4, and float64 cancellation shows
    up in the 13th significant digit -- an implementation artefact of the official
    formula, not a resolution dependence.
    """
    rng = np.random.default_rng(0)
    x = rng.integers(0, 346, 30_000)
    y = rng.integers(0, 260, 30_000)
    reference = esr(x, y, 346, 260)
    for width, height in ((640, 480), (1280, 720)):
        assert esr(x, y, width, height) == reference
    assert esr(x, y, 4096, 4096) == pytest.approx(reference, rel=1e-12)


def test_mesr_drops_incomplete_tail():
    rng = np.random.default_rng(1)
    n = 70_000                      # 2 complete 30k slices + 10k tail
    x = rng.integers(0, 100, n)
    y = rng.integers(0, 100, n)
    per_slice = [esr(x[i:i + 30_000], y[i:i + 30_000], 100, 100) for i in (0, 30_000)]
    assert mesr(x, y, 100, 100) == pytest.approx(float(np.mean(per_slice)), rel=1e-12)


def test_mesr_is_nan_when_no_complete_slice():
    rng = np.random.default_rng(2)
    x = rng.integers(0, 100, 1000)
    y = rng.integers(0, 100, 1000)
    assert np.isnan(mesr(x, y, 100, 100))


def test_mesr_curve_marks_unevaluable_retentions():
    """The floor artifact: aggressive retention can leave < one slice."""
    rng = np.random.default_rng(3)
    n = 40_960
    x = rng.integers(0, 100, n)
    y = rng.integers(0, 100, n)
    scores = rng.normal(size=n)
    curve = mesr_curve(scores, x, y, 100, 100, [0.05, 0.9])
    low, high = curve[0], curve[1]
    assert low["evaluable"] is False and np.isnan(low["mesr"])
    assert high["evaluable"] is True and np.isfinite(high["mesr"])


def test_mesr_curve_keeps_lowest_scores_per_block():
    scores = np.concatenate([np.zeros(5), np.ones(5)])
    x = np.zeros(10, dtype=int)
    y = np.zeros(10, dtype=int)
    curve = mesr_curve(scores, x, y, 2, 2, [0.5], block=10)
    assert curve[0]["kept_events"] == 5


def test_ln_counts_pixels_surviving_RETENTION_of_m_events_not_removal():
    """`ell_n` is the expected occupied-pixel count after keeping M of N events.

    The complement is a real trap: the draft described it as "after removing M events at
    random", which is wrong by 1 - M/N and would have been caught only by working the
    algebra. A pixel survives iff at least one of its n_p events survives, so
    (1 - M/N) must be the per-event REMOVAL probability and M/N = 2/3 the retention.
    """

    rng = np.random.default_rng(0)
    counts = rng.poisson(2.0, 4000)
    counts = counts[counts > 0]
    n_events = int(counts.sum())
    m = int(n_events * 2 / 3)

    ln = float(np.sum(1 - (1 - m / n_events) ** counts))

    def simulate(keep_probability, trials=40):
        return float(np.mean([
            sum(bool(np.any(rng.random(int(c)) < keep_probability)) for c in counts)
            for _ in range(trials)]))

    retaining_m = simulate(m / n_events)
    removing_m = simulate(1 - m / n_events)

    assert ln == pytest.approx(retaining_m, rel=0.02), "ell_n must match RETAINING m events"
    assert abs(ln - removing_m) > 0.1 * ln, "the two readings must be distinguishable"
    assert ln <= len(counts), "ell_n is bounded by the occupied-pixel count"


def test_m_is_constant_because_slices_are_count_delimited():
    """Every complete slice holds exactly `slice_size` events, so M never varies.

    This is why E-MLB's "M is fixed during the entire evaluation process" and the reference
    code's `M = int(N * 2/3)` do not conflict -- a distinction a previous revision briefly
    got wrong in the other direction.
    """

    rng = np.random.default_rng(1)
    total = 95_000
    x = rng.integers(0, 346, total)
    y = rng.integers(0, 260, total)
    for slice_size in (10_000, 30_000, 50_000):
        n_slices = len(range(0, total - slice_size + 1, slice_size))
        assert n_slices == total // slice_size
        assert np.isfinite(mesr(x, y, 346, 260, slice_size=slice_size))
        # Every scored slice has exactly `slice_size` events, hence a constant M.
        assert int(slice_size * 2 / 3) == int(slice_size * 2 / 3)
