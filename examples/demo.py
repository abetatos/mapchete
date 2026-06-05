"""Visual demo: prove that maxchete spreads tiles more evenly than random.

Run it with::

    uv run python examples/demo.py

It synthesizes an **irregular polygon footprint surrounded by nodata**, runs the
random and the max-coverage strategies on it, and writes a set of PNGs to
``examples/output/`` so the result can be inspected by eye:

* ``00_input.png``            - the synthetic input raster.
* ``10_distribution_*.png``   - per-pixel tile-coverage heatmap of each strategy,
                                titled with the coverage standard deviation over
                                the valid footprint (lower = more even).
* ``20_surface_*.png``        - the 3D coverage surface of each strategy.
* ``30_tiles_maxchete.png``   - a montage of a few tiles actually written to disk.

The headline number is the std: max-coverage should report a clearly lower value
than random, which is exactly the library's purpose.
"""
from __future__ import annotations

import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless: write files, never open a window
import matplotlib.pyplot as plt
import numpy as np
import rasterio as rio
from rasterio.transform import from_origin
from skimage.draw import polygon as draw_polygon
from skimage.filters import gaussian

from mapchete import Tiler

NODATA = 0
SIZE = 512          # input raster side, in pixels
TILE = 64           # output tile side
AVG_DENSITY = 3     # stop once each pixel is covered ~3 times on average

# An irregular, concave polygon (row, col) that loosely resembles a real AOI.
POLYGON = np.array([
    (70, 210), (130, 90), (250, 70), (300, 170), (250, 250), (370, 300),
    (450, 250), (410, 390), (300, 450), (190, 410), (150, 330), (95, 380),
    (75, 280),
])


def _terrain_heightfield(size: int, seed: int = 7) -> np.ndarray:
    """A synthetic mountainous height field with ridged striations."""
    rng = np.random.default_rng(seed)
    yy, xx = (np.mgrid[0:size, 0:size].astype(float) / size)  # 0..1

    z = np.zeros((size, size))
    # Large-scale massif so the terrain has an overall relief.
    z += 1.3 * np.exp(-(((xx - 0.5) ** 2 + (yy - 0.55) ** 2) / 0.18))
    z += 0.7 * np.exp(-(((xx - 0.3) ** 2 + (yy - 0.35) ** 2) / 0.05))
    # Ridged striations at several orientations/frequencies (sharp crests).
    for theta_deg, freq, amp in [(20, 8, 0.55), (75, 13, 0.40),
                                 (115, 21, 0.26), (50, 33, 0.14)]:
        th = np.radians(theta_deg)
        proj = xx * np.cos(th) + yy * np.sin(th)
        ridge = 1.0 - np.abs(np.sin(2 * np.pi * freq * proj + rng.uniform(0, 2 * np.pi)))
        z += amp * ridge ** 2
    z += 0.05 * rng.standard_normal((size, size))
    return gaussian(z, sigma=1.0)


def _hillshade(z: np.ndarray, azimuth: float = 315, altitude: float = 45) -> np.ndarray:
    """Standard hillshade of a height field, returned in 0..1."""
    az = np.radians(360.0 - azimuth + 90.0)
    alt = np.radians(altitude)
    dy, dx = np.gradient(z)
    slope = np.arctan(8.0 * np.hypot(dx, dy))  # vertical exaggeration
    aspect = np.arctan2(dy, -dx)
    shaded = np.sin(alt) * np.cos(slope) + np.cos(alt) * np.sin(slope) * np.cos(az - aspect)
    return np.clip(shaded, 0.0, 1.0)


def irregular_polygon_raster(path: str, size: int = SIZE) -> str:
    """Write a single-band GeoTIFF: a mountainous hillshade clipped to an
    irregular polygon, surrounded by nodata.

    ``POLYGON`` is defined on a 512px canvas and scaled to ``size``, so the same
    shape can drive both the visual demo and the (smaller, faster) test fixture.
    """
    poly = POLYGON * (size / 512.0)
    rr, cc = draw_polygon(poly[:, 0], poly[:, 1], shape=(size, size))
    mask = np.zeros((size, size), dtype=bool)
    mask[rr, cc] = True

    # Mountainous hillshade with ridges/striations, scaled to 1..255.
    shade = _hillshade(_terrain_heightfield(size))
    values = (1 + shade * 254).astype(np.uint8)
    array = np.where(mask, values, NODATA).astype(np.uint8)

    profile = {
        "driver": "GTiff", "dtype": "uint8", "count": 1,
        "height": size, "width": size,
        "crs": "EPSG:32630", "transform": from_origin(500000, 4600000, 10, 10),
        "nodata": NODATA,
    }
    with rio.open(path, "w", **profile) as dst:
        dst.write(array, 1)
    return path


def tile_metrics(tiler: Tiler, n_tiles: int) -> dict:
    """Quality metrics of a run, all computed over the valid footprint.

    * ``std``          - std of the per-pixel coverage (lower = more even).
    * ``coverage``     - fraction of valid pixels covered by at least one tile.
    * ``mean_density`` - average number of tiles covering a valid pixel.
    * ``max_overlap``  - highest number of tiles stacked on a single pixel.
    * ``n_tiles``      - number of tiles actually written to disk.
    """
    valid = tiler.array != tiler.nodata
    cov = tiler.distribution_array[valid]
    return {
        "std": float(cov.std()),
        "coverage": float((cov >= 1).mean()),
        "mean_density": float(cov.mean()),
        "max_overlap": int(tiler.distribution_array.max()),
        "n_tiles": n_tiles,
    }


def _save_input(tiler: Tiler, outdir: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 6))
    shown = np.where(tiler.array == tiler.nodata, np.nan, tiler.array)
    ax.imshow(shown, cmap="Greys_r")
    ax.set_title("Input: irregular polygon over nodata")
    ax.axis("off")
    fig.savefig(outdir / "00_input.png", dpi=120, bbox_inches="tight")
    plt.close(fig)


def _save_distribution(tiler: Tiler, name: str, metrics: dict, outdir: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 6))
    dist = tiler.distribution_array.astype(float)
    dist[tiler.array == tiler.nodata] = np.nan
    im = ax.imshow(dist, cmap="coolwarm")
    ax.set_title(f"{name}: tile coverage\n"
                 f"std={metrics['std']:.2f}  coverage={metrics['coverage']*100:.0f}%"
                 f"  tiles={metrics['n_tiles']}")
    ax.axis("off")
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.savefig(outdir / f"10_distribution_{name}.png", dpi=120, bbox_inches="tight")
    plt.close(fig)


def _save_surface(tiler: Tiler, name: str, outdir: Path) -> None:
    fig, _ = tiler.get_3Ddistribution()
    fig.suptitle(f"{name}: 3D coverage surface")
    fig.savefig(outdir / f"20_surface_{name}.png", dpi=120, bbox_inches="tight")
    plt.close(fig)


def _save_tile_montage(tiles_dir: Path, name: str, outdir: Path, n: int = 9) -> None:
    tifs = sorted(tiles_dir.glob("*.tif"))[:n]
    if not tifs:
        return
    cols = 3
    rows = int(np.ceil(len(tifs) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows))
    for ax, tif in zip(np.atleast_1d(axes).ravel(), tifs):
        with rio.open(tif) as src:
            ax.imshow(src.read(1), cmap="Greys_r")
        ax.set_title(tif.stem.replace("croped_", ""), fontsize=7)
        ax.axis("off")
    for ax in np.atleast_1d(axes).ravel()[len(tifs):]:
        ax.axis("off")
    fig.suptitle(f"{name}: sample tiles written to disk")
    fig.tight_layout()
    fig.savefig(outdir / f"30_tiles_{name}.png", dpi=120, bbox_inches="tight")
    plt.close(fig)


STRATEGIES = ["randchete", "maxchete", "infochete", "poischete", "slidechete"]


def _save_comparison(distributions: dict, valid, outdir: Path) -> None:
    """One figure with every strategy's coverage heatmap side by side."""
    n = len(distributions)
    cols = 3
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows))
    axes = np.atleast_1d(axes).ravel()
    for ax, name in zip(axes, distributions):
        dist, metrics = distributions[name]
        shown = dist.astype(float)
        shown[~valid] = np.nan
        im = ax.imshow(shown, cmap="coolwarm")
        ax.set_title(f"{name}\nstd={metrics['std']:.2f}  "
                     f"coverage={metrics['coverage']*100:.0f}%")
        ax.axis("off")
        fig.colorbar(im, ax=ax, shrink=0.7)
    for ax in axes[n:]:
        ax.axis("off")
    fig.suptitle("Tile coverage per strategy (lower std = more even, higher coverage = better)",
                 fontsize=14)
    fig.tight_layout()
    fig.savefig(outdir / "11_comparison.png", dpi=110, bbox_inches="tight")
    plt.close(fig)


def main(outdir: str = "examples/output") -> dict:
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)

    raster = str(out / "HS.tif")
    irregular_polygon_raster(raster)

    results, distributions = {}, {}
    valid = None
    for name in STRATEGIES:
        tiles_dir = Path(out / f"tiles_{name}")
        tiler = Tiler.from_name(raster, name)
        tiler.get_rasters(avg_density=AVG_DENSITY, size=TILE, no_data_percentage=0.3,
                          output_path=str(tiles_dir), clear_output_path=True)

        metrics = tile_metrics(tiler, n_tiles=len(list(tiles_dir.glob("*.tif"))))
        results[name] = metrics
        distributions[name] = (tiler.distribution_array.copy(), metrics)
        if valid is None:
            valid = tiler.array != tiler.nodata
            _save_input(tiler, out)
        _save_distribution(tiler, name, metrics, out)
        # 3D surface + a tile montage for the two "smart" strategies.
        if name in ("maxchete", "infochete"):
            _save_surface(tiler, name, out)
            _save_tile_montage(tiles_dir, name, out)

    _save_comparison(distributions, valid, out)

    print(f"\n{'strategy':11s} {'tiles':>6s} {'coverage':>9s} {'mean_dens':>10s} "
          f"{'max_ovl':>8s} {'std':>6s}")
    for name, m in sorted(results.items(), key=lambda kv: kv[1]["std"]):
        print(f"{name:11s} {m['n_tiles']:6d} {m['coverage']*100:8.1f}% "
              f"{m['mean_density']:10.2f} {m['max_overlap']:8d} {m['std']:6.2f}")
    print(f"\nImages written to {out}/")
    return results


if __name__ == "__main__":
    main()
