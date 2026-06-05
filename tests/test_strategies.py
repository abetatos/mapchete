"""Strategies are pure window generators: testable without any disk I/O.

These tests build a :class:`TilingState` directly on a tiny in-memory array and
drive the strategy, asserting on the windows it yields and on the density map it
maintains -- no GeoTIFF, no temp files.
"""
import numpy as np
import pytest

from mapchete.core import TilingState
from mapchete.strategies import (
    RandomStrategy,
    MaxCoverage,
    InfoCoverage,
    PoissonDisk,
    SlidingWindow,
    get_strategy,
)


class FakeContext:
    """Minimal stand-in for RasterContext (no file, no rasterio dataset)."""

    def __init__(self, array, nodata=0):
        self.array = array
        self.height, self.width = array.shape
        self.nodata = nodata
        self.identifier = "fake"


def make_state(array, size, avg_density=1.0, no_data_percentage=0.3):
    ctx = FakeContext(array)
    return TilingState.create(ctx, size, no_data_percentage, avg_density)


def test_random_windows_stay_in_bounds_and_converge():
    array = np.ones((40, 40), dtype=np.uint8)
    state = make_state(array, size=10, avg_density=1.0)

    windows = list(RandomStrategy().windows(state))

    assert windows
    for w in windows:
        assert 0 <= w.row_off <= 30 and 0 <= w.col_off <= 30
        assert w.width == w.height == 10
    # Stops once the average visit count reaches the target density.
    assert state.counter_array.mean() >= 1.0


def test_maxcoverage_avoids_nodata_heavy_regions():
    # Left half valid, right half nodata (0).
    array = np.zeros((40, 40), dtype=np.uint8)
    array[:, :20] = 200
    state = make_state(array, size=10, avg_density=1.0, no_data_percentage=0.1)

    windows = list(MaxCoverage().windows(state))

    assert windows
    for w in windows:
        # Every accepted tile must stay within the nodata budget (0.1).
        tile = array[w.row_off:w.row_off + w.height, w.col_off:w.col_off + w.width]
        assert np.mean(tile == 0) <= 0.1


def test_sliding_window_overlap_is_deterministic():
    array = np.ones((50, 50), dtype=np.uint8)
    state = make_state(array, size=20)  # avg_density irrelevant (single pass)

    windows = list(SlidingWindow(overlap=0.5).windows(state))  # stride = 10
    rows = sorted({w.row_off for w in windows})
    cols = sorted({w.col_off for w in windows})

    assert rows == [0, 10, 20, 30]      # last clamped to height - size
    assert cols == [0, 10, 20, 30]
    assert len(windows) == 16


def test_sliding_window_rejects_bad_overlap():
    with pytest.raises(ValueError):
        SlidingWindow(overlap=1.0)


def test_poisson_disk_respects_minimum_spacing():
    array = np.ones((80, 80), dtype=np.uint8)
    state = make_state(array, size=10)

    min_dist = 18
    windows = list(PoissonDisk(min_dist=min_dist).windows(state))
    centres = [(w.row_off, w.col_off) for w in windows]

    assert len(centres) > 1
    for a in range(len(centres)):
        for b in range(a + 1, len(centres)):
            d = np.hypot(centres[a][0] - centres[b][0], centres[a][1] - centres[b][1])
            assert d >= min_dist - 2  # small tolerance for integer rounding
    for w in windows:
        assert 0 <= w.row_off <= 70 and 0 <= w.col_off <= 70


def test_poisson_disk_default_covers_most_of_the_raster():
    # Guards against a too-large default spacing leaving the footprint uncovered.
    array = np.full((120, 120), 200, dtype=np.uint8)  # all valid (nodata = 0)
    state = make_state(array, size=20)  # default min_dist = 10

    list(PoissonDisk().windows(state))

    coverage = (state.counter_array >= 1).mean()
    assert coverage > 0.85


def test_infocoverage_prefers_textured_regions():
    # Left half flat, right half high-frequency stripes (strong gradient).
    array = np.full((40, 40), 50, dtype=np.uint8)
    array[:, 20:] = np.tile([100, 200], (40, 10))  # alternating columns
    state = make_state(array, size=10, avg_density=0.5, no_data_percentage=0.3)

    windows = list(InfoCoverage().windows(state))

    assert windows
    # Most early tiles should fall on the textured (right) half.
    right = sum(1 for w in windows if w.col_off + w.width // 2 >= 20)
    assert right / len(windows) > 0.6


def test_get_strategy_roundtrip():
    for name in ("maxchete", "infochete", "poischete", "slidechete"):
        assert get_strategy(name).name == name
    with pytest.raises(ValueError):
        get_strategy("does-not-exist")
