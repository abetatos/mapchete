"""Sampling strategies: the only thing that differs between algorithms.

A strategy is a **pure window generator**. Given a :class:`TilingState` it yields
:class:`rasterio.windows.Window` objects and updates ``state.counter_array`` (its
steering signal). It never reads or writes the disk and never touches
``distribution_array`` -- that is the engine's job. This makes every algorithm a
small, side-effect-free object that can be unit-tested on an in-memory state.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from math import sqrt, sin, cos, pi
from typing import Iterator, Optional, Type
import random

import numpy as np
from rasterio.windows import Window

from .core import RasterContext, TilingState


def _ensure_fits(context: RasterContext, size: int) -> None:
    if size > context.height or size > context.width:
        raise ValueError(
            f"Tile size {size} exceeds the raster extent "
            f"({context.height}x{context.width}); choose a smaller size."
        )


def _integral_image(a: np.ndarray) -> np.ndarray:
    """Summed-area table padded with a zero row/column for O(1) window sums."""
    ii = np.zeros((a.shape[0] + 1, a.shape[1] + 1), dtype=np.float64)
    ii[1:, 1:] = np.cumsum(np.cumsum(a, axis=0), axis=1)
    return ii


def _window_sums(integral: np.ndarray, size: int) -> np.ndarray:
    """Sum over every ``size``x``size`` window, for all top-left positions at once.

    Returns an array of shape ``(H - size + 1, W - size + 1)`` where element
    ``[i, j]`` is the sum of the window whose top-left corner is ``(i, j)``.
    """
    return (
        integral[size:, size:] - integral[:-size, size:]
        - integral[size:, :-size] + integral[:-size, :-size]
    )


class SamplingStrategy(ABC):
    """Decides which windows to cut. Subclasses implement :meth:`windows`."""

    #: Short identifier used to name output files and in FARMchete dispatch.
    name: str = "base"

    @abstractmethod
    def windows(self, state: TilingState) -> Iterator[Window]:
        """Yield windows until the target average density is reached."""
        raise NotImplementedError


class RandomStrategy(SamplingStrategy):
    """Uniformly random windows (the *randchete* baseline)."""

    name = "randchete"

    def windows(self, state: TilingState) -> Iterator[Window]:
        ctx, size = state.context, state.size
        _ensure_fits(ctx, size)

        while state.counter_array.mean() < state.avg_density:
            i = random.randint(0, ctx.height - size)
            j = random.randint(0, ctx.width - size)
            state.counter_array[i:i + size, j:j + size] += 1
            yield Window(col_off=j, row_off=i, width=size, height=size)


class MaxCoverage(SamplingStrategy):
    """Probabilistic strategy that steers tiles towards low-density zones.

    Holding a live density map in memory, it places each tile where coverage is
    currently lowest, augmenting the covered area while skipping windows whose
    nodata fraction exceeds the budget (their nodata pixels are still counted so
    the same dead zone is not revisited).
    """

    name = "maxchete"

    def windows(self, state: TilingState) -> Iterator[Window]:
        ctx, size = state.context, state.size
        _ensure_fits(ctx, size)

        while True:
            self._guard_progress(state)
            window = self._pick(state)
            if window is None:
                continue
            yield window
            if state.counter_array.mean() >= state.avg_density:
                return

    @staticmethod
    def _guard_progress(state: TilingState) -> None:
        """Abort if no tile can ever be produced (empty data / size too big)."""
        if state.distribution_array.max() == 0:
            if state.counter_array.min() > 0 or state.counter_array.max() > 100:
                raise ValueError(
                    "Looks like your data is empty or the selected tile size is "
                    "bigger than the achievable output."
                )

    def _pick(self, state: TilingState):
        ctx, size = state.context, state.size
        counter = state.counter_array
        height, width = ctx.height, ctx.width
        output_length = size ** 2

        stride = int(max(size / 10, random.random() * (size - 1)))
        x_iter = int(np.floor((height - size) / stride))
        y_iter = int(np.floor((width - size) / stride))
        if not x_iter or not y_iter:
            stride = max(int(random.random() * (min(height, width) - size - 1)), 1)
            x_iter = int(np.floor((height - size) / stride))
            y_iter = int(np.floor((width - size) / stride))
            if not x_iter or not y_iter:
                counter += 1
                return None

        # Sliding window sum of the density map; the j>0 branch reuses the
        # previous column's sum (~20% faster than recomputing each window).
        tmp_aux = np.zeros((x_iter, y_iter))
        for i in range(x_iter):
            i_init, i_end = i * stride, i * stride + size
            for j in range(y_iter):
                j_init, j_end = j * stride, j * stride + size
                if j == 0:
                    tmp_aux[i, j] = counter[i_init:i_end, j_init:j_end].sum()
                else:
                    delta = (
                        - counter[i_init:i_end, (j - 1) * stride:j_init].sum()
                        + counter[i_init:i_end, (j - 1) * stride + size:j_end].sum()
                    )
                    tmp_aux[i, j] = tmp_aux[i, j - 1] + delta

        minimums = np.argwhere(tmp_aux == tmp_aux.min())
        min_i, min_j = minimums[random.randrange(minimums.shape[0])] * stride

        output_array = ctx.array[min_i:size + min_i, min_j:size + min_j]
        no_data_count = np.count_nonzero(output_array == ctx.nodata)

        if no_data_count / output_length > state.no_data_percentage:
            mask = output_array == ctx.nodata
            counter[min_i:size + min_i, min_j:size + min_j][mask] += 1
            return None

        counter[min_i:size + min_i, min_j:size + min_j] += 1
        return Window(col_off=min_j, row_off=min_i, width=size, height=size)


class InfoCoverage(SamplingStrategy):
    """Content-aware sampling: more tiles where the image is more informative.

    It scores every candidate window by ``information / (1 + current coverage)``,
    where *information* is the local gradient energy (edges and texture). High
    texture regions are sampled first and densely; flat valid regions are still
    covered, but later and sparsely. This concentrates the dataset on the pixels a
    model actually learns from. Windows over the nodata budget are never chosen.
    """

    name = "infochete"

    def windows(self, state: TilingState) -> Iterator[Window]:
        ctx, size = state.context, state.size
        _ensure_fits(ctx, size)

        array = ctx.array.astype(np.float64)
        valid = ctx.array != ctx.nodata

        # Information map = gradient magnitude (edges/texture), zero on nodata.
        gy, gx = np.gradient(array)
        info = np.hypot(gy, gx)
        info[~valid] = 0.0

        info_sums = _window_sums(_integral_image(info), size)
        nodata_sums = _window_sums(_integral_image((~valid).astype(np.float64)), size)
        allowed = nodata_sums <= state.no_data_percentage * size * size
        if not allowed.any():
            return

        base = info_sums + 1.0  # +1 so flat valid windows still get covered
        counter = state.counter_array
        valid_pixels = int(valid.sum())
        # Safety cap so a pathological input can never loop forever.
        cap = int(valid_pixels * state.avg_density / (size * size)) * 10 + 100

        for _ in range(cap):
            if valid_pixels and counter[valid].mean() >= state.avg_density:
                return
            cov_sums = _window_sums(_integral_image(counter.astype(np.float64)), size)
            score = base / (1.0 + cov_sums)
            score[~allowed] = -1.0
            best = score.max()
            if best <= 0:
                return
            choices = np.argwhere(score == best)
            r0, c0 = choices[random.randrange(len(choices))]
            counter[r0:r0 + size, c0:c0 + size] += 1
            yield Window(col_off=int(c0), row_off=int(r0), width=size, height=size)


class PoissonDisk(SamplingStrategy):
    """Blue-noise tile placement with a guaranteed minimum spacing (Bridson).

    Tile centres are drawn so no two are closer than ``min_dist`` pixels, giving
    an even, cluster-free layout with controlled overlap -- ideal for a
    representative, low-redundancy dataset. Tiles centred on nodata are skipped.
    ``avg_density`` does not apply: this is a single, finite pass.

    ``min_dist`` defaults to half the tile size, which keeps the whole footprint
    well covered (a larger spacing leaves gaps, since the worst-case hole between
    blue-noise samples is ~2x ``min_dist``).
    """

    name = "poischete"

    def __init__(self, min_dist: Optional[float] = None, k: int = 30) -> None:
        self.min_dist = min_dist
        self.k = k

    def windows(self, state: TilingState) -> Iterator[Window]:
        ctx, size = state.context, state.size
        _ensure_fits(ctx, size)

        h_span = ctx.height - size  # inclusive max row offset
        w_span = ctx.width - size
        radius = self.min_dist if self.min_dist is not None else max(1.0, size * 0.5)

        for r0, c0 in self._bridson(h_span, w_span, radius):
            if ctx.array[r0 + size // 2, c0 + size // 2] == ctx.nodata:
                continue  # bias towards valid data
            state.counter_array[r0:r0 + size, c0:c0 + size] += 1
            yield Window(col_off=c0, row_off=r0, width=size, height=size)

    def _bridson(self, h_span: int, w_span: int, r: float):
        """Bridson's Poisson-disk sampling over [0, h_span] x [0, w_span]."""
        cell = r / sqrt(2)
        gh = int(h_span / cell) + 1
        gw = int(w_span / cell) + 1
        grid = -np.ones((gh, gw), dtype=int)  # sample index per grid cell, -1 = empty
        samples, active = [], []

        def add(point):
            samples.append(point)
            gi, gj = int(point[0] / cell), int(point[1] / cell)
            grid[gi, gj] = len(samples) - 1
            active.append(len(samples) - 1)

        add((random.uniform(0, h_span), random.uniform(0, w_span)))
        while active:
            idx = active[random.randrange(len(active))]
            base = samples[idx]
            for _ in range(self.k):
                ang, rad = random.uniform(0, 2 * pi), random.uniform(r, 2 * r)
                p = (base[0] + rad * sin(ang), base[1] + rad * cos(ang))
                if not (0 <= p[0] <= h_span and 0 <= p[1] <= w_span):
                    continue
                gi, gj = int(p[0] / cell), int(p[1] / cell)
                ok = True
                for di in range(-2, 3):
                    for dj in range(-2, 3):
                        ni, nj = gi + di, gj + dj
                        if 0 <= ni < gh and 0 <= nj < gw and grid[ni, nj] != -1:
                            q = samples[grid[ni, nj]]
                            if (p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2 < r * r:
                                ok = False
                                break
                    if not ok:
                        break
                if ok:
                    add(p)
                    break
            else:
                active.remove(idx)

        for p in samples:
            yield (min(max(int(round(p[0])), 0), h_span),
                   min(max(int(round(p[1])), 0), w_span))


class SlidingWindow(SamplingStrategy):
    """Deterministic sliding window with a user-defined overlap.

    Classic, reproducible tiling: a regular grid where consecutive tiles overlap
    by ``overlap`` (0 = no overlap, 0.5 = half a tile, ...). The last row/column
    is clamped to the edge so the whole raster is covered. ``avg_density`` does
    not apply: this is a single, finite pass.
    """

    name = "slidechete"

    def __init__(self, overlap: float = 0.5) -> None:
        if not 0 <= overlap < 1:
            raise ValueError("overlap must be in [0, 1).")
        self.overlap = overlap

    def windows(self, state: TilingState) -> Iterator[Window]:
        ctx, size = state.context, state.size
        _ensure_fits(ctx, size)

        stride = max(1, int(round(size * (1 - self.overlap))))
        rows = list(range(0, ctx.height - size + 1, stride))
        cols = list(range(0, ctx.width - size + 1, stride))
        if rows[-1] != ctx.height - size:
            rows.append(ctx.height - size)
        if cols[-1] != ctx.width - size:
            cols.append(ctx.width - size)

        for r0 in rows:
            for c0 in cols:
                state.counter_array[r0:r0 + size, c0:c0 + size] += 1
                yield Window(col_off=c0, row_off=r0, width=size, height=size)


STRATEGIES: dict[str, Type[SamplingStrategy]] = {
    RandomStrategy.name: RandomStrategy,
    MaxCoverage.name: MaxCoverage,
    InfoCoverage.name: InfoCoverage,
    PoissonDisk.name: PoissonDisk,
    SlidingWindow.name: SlidingWindow,
}


def get_strategy(name: str) -> SamplingStrategy:
    """Instantiate a strategy by its short name (e.g. ``maxchete`` / ``infochete`` / ``poischete``)."""
    try:
        return STRATEGIES[name]()
    except KeyError:
        raise ValueError(
            f"Unknown strategy {name!r}; choose one of {sorted(STRATEGIES)}."
        ) from None
