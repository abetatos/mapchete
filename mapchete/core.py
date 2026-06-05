"""Core engine shared by every cropping algorithm.

The three algorithms shipped by mapchete (random, sequential and max-coverage)
differ **only** in how they choose the next window to cut. Everything else --
loading the raster, trimming the nodata border, validating the nodata budget,
writing tiles, bookkeeping the spatial distribution and deciding when to stop --
is identical and lives here.

A :class:`~mapchete.strategies.SamplingStrategy` is a pure window generator: it
yields :class:`rasterio.windows.Window` objects and never touches the disk. The
:class:`Tiler` drives that generator, materialises each window and persists the
valid tiles. This keeps the sampling logic small, side-effect free and trivially
unit-testable without any I/O.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Tuple, Union
import logging
import os
import pickle as pkl
import shutil

import numpy as np
import rasterio as rio
import rasterio.plot as rplt
from rasterio.io import MemoryFile
from rasterio.windows import Window, get_data_window, transform
from skimage.transform import resize
import matplotlib.pyplot as plt
from tqdm.auto import tqdm


class RasterContext:
    """Loads a raster, trims its nodata border and exposes the working array.

    Args:
        filepath: Path to any raster readable by rasterio, usually a GeoTIFF.
    """

    def __init__(self, filepath: str) -> None:
        self.identifier = os.path.splitext(os.path.basename(filepath))[0]

        with rio.open(filepath) as src:
            profile = src.profile.copy()
            data_window = get_data_window(src.read(masked=True))
            data_transform = transform(data_window, src.transform)
            profile.update(
                transform=data_transform,
                height=data_window.height,
                width=data_window.width,
            )
            data = src.read(window=data_window)

        # Keep the trimmed dataset alive in memory for fast windowed reads.
        memfile = MemoryFile()
        self.src = memfile.open(**profile)
        self.src.write(data)
        memfile.close()

        self.array = data[0]  # first band drives the nodata/density logic
        self.width = self.src.width
        self.height = self.src.height
        self.nodata = self.src.nodata


@dataclass
class TilingState:
    """Mutable bookkeeping shared between the engine and the strategy.

    ``counter_array`` tracks how many times each pixel has been *visited* (the
    signal strategies use to steer towards low-density zones). ``distribution_array``
    tracks how many *saved* tiles cover each pixel (used by the 3D plot). Both are
    wide integers to avoid the uint8 overflow that would corrupt the sums.
    """

    context: RasterContext
    size: int
    no_data_percentage: float
    avg_density: float
    counter_array: np.ndarray
    distribution_array: np.ndarray

    @classmethod
    def create(cls, context: RasterContext, size: int,
               no_data_percentage: float, avg_density: float) -> "TilingState":
        shape = context.array.shape
        return cls(
            context=context,
            size=size,
            no_data_percentage=no_data_percentage,
            avg_density=avg_density,
            counter_array=np.zeros(shape, dtype=np.int32),
            distribution_array=np.zeros(shape, dtype=np.int32),
        )


class Tiler:
    """Cuts a raster into tiles using a pluggable sampling strategy.

    Args:
        filepath: Path to the input raster.
        strategy: The sampling strategy that decides which windows to cut.
        log_level: Logger level. Defaults to ``logging.INFO``.
    """

    def __init__(self, filepath: str,
                 strategy: "SamplingStrategy",
                 log_level: Union[str, int] = logging.INFO) -> None:
        self.logger = logging.getLogger("mapchete")
        self.logger.setLevel(log_level)

        self.context = RasterContext(filepath)
        self.strategy = strategy
        self.state: Union[TilingState, None] = None

    @classmethod
    def from_name(cls, filepath: str, strategy: str,
                  log_level: Union[str, int] = logging.INFO) -> "Tiler":
        """Build a Tiler from a strategy name (e.g. ``maxchete``/``infochete``/``poischete``)."""
        from .strategies import get_strategy  # late import to avoid a cycle
        return cls(filepath, get_strategy(strategy), log_level=log_level)

    # ------------------------------------------------------------------ #
    # Convenience accessors so the public API stays flat and notebook-friendly.
    # ------------------------------------------------------------------ #
    @property
    def array(self) -> np.ndarray:
        return self.context.array

    @property
    def nodata(self):
        return self.context.nodata

    @property
    def width(self) -> int:
        return self.context.width

    @property
    def height(self) -> int:
        return self.context.height

    @property
    def identifier(self) -> str:
        return self.context.identifier

    @property
    def distribution_array(self) -> np.ndarray:
        if self.state is None:
            raise RuntimeError("Run get_rasters() before reading the distribution.")
        return self.state.distribution_array

    # ------------------------------------------------------------------ #
    # Tiling
    # ------------------------------------------------------------------ #
    def get_rasters(self, avg_density: float = 4, size: int = 512,
                    no_data_percentage: float = 0.2,
                    output_path: str = "raster_clip",
                    clear_output_path: bool = False) -> None:
        """Generate tiles on disk by consuming the sampling strategy.

        Args:
            avg_density: Average per-pixel visit count at which to stop.
            size: Square side, in pixels, of every output tile.
            no_data_percentage: Tiles with a larger nodata fraction are skipped.
            output_path: Folder where the tiles are written.
            clear_output_path: If True, wipe ``output_path`` before writing.
        """
        self.clear_opath(output_path, clear_output_path)

        self.state = TilingState.create(self.context, size, no_data_percentage, avg_density)
        valid, skipped = 0, 0
        pbar = tqdm()

        for window in self.strategy.windows(self.state):
            new_array, profile, is_valid = self._read_window(window)
            if is_valid:
                self._save_tile(window, new_array, profile, output_path, valid)
                self._mark_distribution(window)
                valid += 1
            else:
                skipped += 1
            pbar.set_description(f"Mean density {self.state.counter_array.mean():.2f}")
            pbar.update(1)
        pbar.close()

        self._log_summary(valid, skipped)

    def _read_window(self, window: Window) -> Tuple[np.ndarray, dict, bool]:
        """Read a window from the source and check it against the nodata budget."""
        profile = self.context.src.profile
        profile.update({
            "height": self.state.size,
            "width": self.state.size,
            "transform": self.context.src.window_transform(window),
        })

        new_array = self.context.src.read(window=window)
        nodata_ratio = np.count_nonzero(new_array == self.nodata) / new_array.size
        is_valid = bool(new_array.any() and nodata_ratio <= self.state.no_data_percentage)

        return new_array, profile, is_valid

    def _save_tile(self, window: Window, new_array: np.ndarray, profile: dict,
                   output_path: str, index: int) -> None:
        """Persist a tile as a GeoTIFF plus a pickle of its source window."""
        stem = f"croped_{self.identifier}_{self.strategy.name}_{index}"
        try:
            with open(os.path.join(output_path, f"{stem}.pickle"), "wb") as f:
                pkl.dump(window, f)
            with rio.open(os.path.join(output_path, f"{stem}.tif"), "w", **profile) as dst:
                dst.write(new_array)
        except Exception as exc:  # pragma: no cover - I/O failure path
            self.logger.error("Writing error: %s", exc)

    def _mark_distribution(self, window: Window) -> None:
        r0, c0 = int(window.row_off), int(window.col_off)
        size = self.state.size
        self.state.distribution_array[r0:r0 + size, c0:c0 + size] += 1

    def _log_summary(self, valid: int, skipped: int) -> None:
        total = valid + skipped
        pct = round(valid / total * 100, 1) if total else 0.0
        self.logger.info(
            "Process finished: %d tiles written (%.1f%% of attempts kept).",
            valid, pct,
        )

    @staticmethod
    def clear_opath(output_path: str, clear_output_path: bool) -> None:
        """Create ``output_path``, optionally wiping it first."""
        if clear_output_path:
            shutil.rmtree(output_path, ignore_errors=True)
        os.makedirs(output_path, exist_ok=True)

    # ------------------------------------------------------------------ #
    # Plotting helpers
    # ------------------------------------------------------------------ #
    def plot_bands(self):
        """Show the (nodata-trimmed) input raster."""
        rplt.show(self.array, cmap="Greys")

    def plot_hist(self, delete_nodata: bool = True):
        """Show the histogram of the input raster.

        Args:
            delete_nodata: If True, replace nodata with 0 before plotting.
        """
        array = np.where(self.array != self.nodata, self.array, 0) if delete_nodata else self.array
        rplt.show_hist(array)

    def get_3Ddistribution(self, permute: Tuple[int, int] = (1, 0)):
        """Plot the 3D spatial distribution of the generated tiles.

        Args:
            permute: Axis permutation to reorient the surface. Defaults to (1, 0).

        Returns:
            Tuple of the matplotlib ``fig`` and ``ax``.
        """
        final = np.transpose(self.distribution_array, permute)
        resized = resize(final, (300, 300))

        xx, yy = [], []
        for i, row in enumerate(resized):
            xx.append(list(range(len(row))))
            yy.append([i] * len(row))

        fig = plt.figure(figsize=(13, 7))
        ax = plt.axes(projection="3d")
        surf = ax.plot_surface(xx, yy, resized, rstride=1, cstride=1,
                               cmap="coolwarm", edgecolor="none")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("Frequency")
        ax.set_title("Surface plot")
        fig.colorbar(surf, shrink=0.5, aspect=5)
        ax.view_init(60, 35)

        return fig, ax
