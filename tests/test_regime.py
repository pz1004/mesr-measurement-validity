import numpy as np

from dataset_assessment.regime import (
    event_rate_per_px, hot_pixel_share, local_global_coupling, spatial_autocorrelation,
)
from dataset_assessment.regime import _neighbour_share


def uniform_noise(n, width, height, seed=0):
    rng = np.random.default_rng(seed)
    t = np.sort(rng.integers(0, 1_000_000, n))
    return np.column_stack([t, rng.integers(0, width, n), rng.integers(0, height, n),
                            rng.integers(0, 2, n)]).astype(np.int64)


def clustered_signal(n, width, height, seed=1):
    """A moving edge: events concentrated on a line that translates over time."""
    rng = np.random.default_rng(seed)
    t = np.sort(rng.integers(0, 1_000_000, n))
    frac = t / t.max()
    x = (frac * (width - 1)).astype(np.int64)
    y = rng.integers(0, height, n)
    x = np.clip(x + rng.integers(-1, 2, n), 0, width - 1)
    return np.column_stack([t, x, y, rng.integers(0, 2, n)]).astype(np.int64)


def dense_neighbour_share(x, y, width, height):
    """Reference implementation: the obvious dense-surface version."""
    occupied = np.zeros((height + 2, width + 2), dtype=bool)
    occupied[y + 1, x + 1] = True
    neighbours = np.zeros_like(occupied, dtype=np.int16)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dx == 0 and dy == 0:
                continue
            neighbours[1:-1, 1:-1] += occupied[1 + dy:height + 1 + dy,
                                               1 + dx:width + 1 + dx]
    return float((neighbours[y + 1, x + 1] > 0).mean())


def test_neighbour_share_matches_the_dense_reference():
    """The sparse implementation exists for speed; it must not change the statistic."""
    for seed in (0, 1, 2):
        events = uniform_noise(400, 40, 30, seed=seed)
        x, y = events[:, 1], events[:, 2]
        assert _neighbour_share(x, y, 40, 30) == dense_neighbour_share(x, y, 40, 30)
    events = clustered_signal(400, 40, 30)
    x, y = events[:, 1], events[:, 2]
    assert _neighbour_share(x, y, 40, 30) == dense_neighbour_share(x, y, 40, 30)


def test_spatial_autocorrelation_separates_clustered_from_uniform():
    uni = spatial_autocorrelation(uniform_noise(50_000, 346, 260), 346, 260)
    clu = spatial_autocorrelation(clustered_signal(50_000, 346, 260), 346, 260)
    assert clu > uni


def test_spatial_autocorrelation_is_scale_free_enough_to_compare_sensors():
    small = spatial_autocorrelation(uniform_noise(50_000, 346, 260), 346, 260)
    large = spatial_autocorrelation(uniform_noise(50_000, 1280, 720), 1280, 720)
    assert abs(small - large) < 0.15, "uniform noise must look similar on both sensors"


def test_local_global_coupling_is_finite_and_bounded():
    value = local_global_coupling(uniform_noise(50_000, 346, 260), 346, 260)
    assert np.isfinite(value) and -1.0 <= value <= 1.0


def test_hot_pixel_share_is_a_fraction():
    value = hot_pixel_share(uniform_noise(50_000, 346, 260), 346, 260)
    assert 0.0 < value <= 1.0


def test_event_rate_per_px_has_the_expected_magnitude():
    # 50,000 events over 1 s on 346x260 pixels -> 50000 / 89960 ~ 0.556 ev/px/s
    value = event_rate_per_px(uniform_noise(50_000, 346, 260), 346, 260)
    assert 0.4 < value < 0.7
