<p align="center">
  <img width="300" alt="mapchete logo" src="https://user-images.githubusercontent.com/76526314/219464092-ee4e075c-c8c7-4d39-8017-cb0ede17248f.png">
</p>

<h3 align="center">Cut your geospatial data into smaller pieces</h3>

---

**mapchete** crops large geospatial rasters (GeoTIFFs, hillshades, ...) into small tiles for deep learning. It spreads the tiles to cover the data evenly — avoiding both the redundant hotspots of random cropping and tiles that are mostly `nodata`.

## Installation

```bash
git clone https://github.com/abetatos/mapchete.git
cd mapchete
pip install .
```

Requires Python ≥ 3.9. `rasterio` is only lower-bounded (`>=1.3`) for flexibility.

## Quickstart

```python
from mapchete import Tiler

tiler = Tiler.from_name("HS.tif", "maxchete")
tiler.plot_bands()                       # inspect the input

tiler.get_rasters(
    avg_density=4,          # avg. number of tiles covering each input pixel
    size=512,               # output tile side, in pixels
    no_data_percentage=0.3, # discard tiles with more nodata than this
    output_path="raster_clip",
    clear_output_path=True,
)

fig, ax = tiler.get_3Ddistribution()     # study the tile distribution
```

Each tile is written as a GeoTIFF, plus a pickle of its source `rasterio` window.

### Discarding empty tiles (`no_data_percentage`)

Geospatial rasters rarely fill their bounding box — the valid data has an
irregular footprint surrounded by `nodata`. Tiles cut near that edge would be
partly (or almost entirely) empty, which is useless for training.

To prevent this, **before saving every tile mapchete counts the fraction of
`nodata` pixels inside it and drops the tile if that fraction exceeds
`no_data_percentage`**:

```
nodata_fraction = (#nodata pixels in tile) / (tile area)
keep the tile only if  nodata_fraction <= no_data_percentage
```

- `no_data_percentage=0.3` → keep tiles that are **at most 30% empty** (a sensible default).
- `no_data_percentage=0.1` → stricter: only tiles that are almost fully valid.
- `no_data_percentage=0` → keep **only completely full** tiles.

So lowering the value removes the half-empty edge tiles; raising it keeps more of
the boundary at the cost of emptier tiles.

## Sampling strategies

The strategy is the only thing that changes between algorithms; everything else (loading, nodata trimming, validation, saving) is shared.

| Strategy | Approach |
| --- | --- |
| **maxchete** | Probabilistic max-coverage: steers tiles towards low-density zones for an even spread. |
| **infochete** | Content-aware: more tiles where the **image gradient** is high (edges, ridges, texture), fewer in flat areas. |
| **poischete** | Blue-noise (Poisson-disk): a minimum spacing between tiles for an even, cluster-free layout. |
| **slidechete** | Deterministic sliding window with a user-defined `overlap` — classic, reproducible tiling. |
| **randchete** | Uniformly random windows (baseline). |

Pick one by name, or build it explicitly to pass parameters:

```python
from mapchete import Tiler, SlidingWindow, PoissonDisk

Tiler.from_name("HS.tif", "infochete")
Tiler("HS.tif", SlidingWindow(overlap=0.5))
Tiler("HS.tif", PoissonDisk(min_dist=400))
```

> **How `infochete` measures information.** It uses the **magnitude of the input's gradient** (`np.hypot(*np.gradient(image))`), not the raw pixel values. Flat regions — however bright or dark — score ≈ 0, while edges, ridges and rough terrain score high. Each candidate tile is ranked by `gradient_energy / (1 + current_coverage)`, so tiles pile up where there is the most to learn while still spreading out.

## See it in action

The repo ships a runnable demo that synthesizes an **eroded volcano hillshade clipped to an irregular polygon** (valid data surrounded by `nodata` — exactly what mapchete targets) and runs every strategy on it:

```bash
uv run python examples/demo.py     # writes the images below to examples/output/
```

**Input** — a volcano (high at the centre, fading to the sides) with eroded gullies on its flanks, over a `nodata` background:

<p align="center">
  <img width="320" alt="synthetic mountainous input" src="docs/img/input.png">
</p>

**Tile coverage per strategy** — each heatmap counts how many tiles cover each pixel:

<p align="center">
  <img width="820" alt="coverage comparison" src="docs/img/comparison.png">
</p>

Metrics over the valid footprint for one run (lower **std** = more even, higher **coverage** = fewer gaps):

Maxima of each column are in **bold**.

| Strategy | Tiles | Coverage | Std (evenness) |
| --- | --- | --- | --- |
| poischete | 37 | 94% | 0.93 |
| maxchete | **70** | **99%** | 1.00 |
| slidechete | 63 | 97% | 1.20 |
| infochete | 66 | **99%** | 1.37 |
| randchete | 60 | 90% | **1.90** |

Random cropping (`randchete`) piles tiles into arbitrary hotspots (high std) while still leaving gaps. `maxchete`, `poischete` and `slidechete` instead spread tiles evenly (low std, high coverage). `infochete` is the odd one out *on purpose*: its higher std comes from deliberately concentrating tiles on the high-relief flanks and crater of the volcano — see the red hotspot over the textured centre in its panel above.

### 3D coverage surface

`get_3Ddistribution()` makes the difference obvious. Random sampling builds sharp spikes (some areas over-sampled, others bare), while `maxchete` produces a smooth, even plateau:

<p align="center">
  <img width="390" alt="randchete 3D coverage" src="docs/img/surface_randchete.png">
  <img width="390" alt="maxchete 3D coverage" src="docs/img/surface_maxchete.png">
</p>

### Sample tiles

The generated tiles. Edge tiles are kept only when their `nodata` fraction stays within `no_data_percentage`:

<p align="center">
  <img width="540" alt="sample generated tiles" src="docs/img/tiles.png">
</p>

## Merging tiles back

`merge_tiffs` stitches the generated tiles back into a single mosaic — handy to visually check the coverage:

```python
from mapchete import merge_tiffs
merge_tiffs(folder="raster_clip")
```

## Development

The project uses [uv](https://docs.astral.sh/uv/) for environment management:

```bash
uv sync --extra test
uv run pytest
```
