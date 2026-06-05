<p align="center">
  <img width="300" alt="mapchete_final" src="https://user-images.githubusercontent.com/76526314/219464092-ee4e075c-c8c7-4d39-8017-cb0ede17248f.png">
</p>

<h3 align="center">
    <p>Cut your geospatial data into smaller pieces</p>
</h3>

# MAPchete

Welcome to my Github project! This repository was created to assist with the preparation of geospatial data for deep learning purposes. Specifically, the project focuses on efficiently cropping large datasets into smaller tiles, with the goal of generating a dataset with minimum overlap between tiles (Which could result in the loss of representativeness) and avoiding nodata tiles. This process is essential for achieving optimal model performance, and can be applied to various other applications within the geospatial imagery field. Thank you for checking out my project, and feel free to explore the code and contribute to its development!

# What does MAPchete have to offer?

It generates patches based on a probabilistic approach that tries to augment the covered area distributing images more efficiently while avoiding images with a nodata percentage avobe a given threshold. It is perfect for deep learning purposes as it will maximize the outcome of your model! 

If we generate the dataset randomly we can see that there are zones that have great number of tiles, while with this approach you can obtain a more well distributed dataset. 

With an input of shape: 
<p align="center">
  <img width="300" alt="mapchete_final" src="https://user-images.githubusercontent.com/76526314/219682129-756f265c-6f4c-4c20-bc2e-bc5e438f4721.png">
</p>

How is the spatial distribution of tiles? (Not normalized scales)

Iterations | RANDchete (random) | MAXchete (maximize)
--- | --- | ---
100 iter | ![](https://user-images.githubusercontent.com/76526314/219666167-64e7f0a8-df76-4422-8665-a6f908b0a98b.png) | ![](https://user-images.githubusercontent.com/76526314/219665645-7eefad2e-bc33-43cb-99fa-5374f6c84ea4.png)
1000 iter | ![image](https://user-images.githubusercontent.com/76526314/219706410-985e57b5-5698-49e6-afdb-856fe01c073b.png) | ![image](https://user-images.githubusercontent.com/76526314/219707072-d8134441-64ba-41a3-a23a-74466f6c5bda.png)
std (1000 iter) | &plusmn; 9.1  |  &plusmn; 3.9

Compared to the random sample, the data is now more evenly distributed and the standard deviation has been reduced by over 2 times which shows library's effectiveness.

# See it in action

The library ships a runnable demo that synthesizes a **mountainous hillshade clipped to an irregular polygon** — valid data surrounded by `nodata`, exactly the kind of geospatial raster mapchete targets — and runs every strategy on it:

```bash
uv run python examples/demo.py   # writes the images below to examples/output/
```

**Input** — a hillshade with ridges and striations over a `nodata` background:

<p align="center">
  <img width="340" alt="synthetic mountainous input" src="docs/img/input.png">
</p>

**Tile coverage per strategy** — each heatmap counts how many tiles cover each pixel; a lower standard deviation over the footprint means a more even spread:

<p align="center">
  <img width="780" alt="coverage comparison" src="docs/img/comparison.png">
</p>

On this run the metrics over the valid footprint were (lower **std** = more even, higher **coverage** = fewer gaps):

| Strategy | Tiles | Coverage | Std (evenness) |
| --- | --- | --- | --- |
| infochete | 65 | 100% | **0.73** |
| maxchete | 67 | 99% | 1.02 |
| poischete | 35 | 85% | 1.07 |
| slidechete | 63 | 97% | 1.20 |
| randchete | 59 | 87% | 1.89 |

The random baseline (`randchete`) clusters tiles into hotspots (high std) while still leaving gaps. The smart strategies spread the tiles evenly across the whole footprint — or, in the case of `infochete`, deliberately concentrate them on the most textured terrain while still covering everything.

# How does it work?

There are several ways of creating your dataset, each with a different sampling strategy:

| Strategy | Approach |
| --- | --- |
| **randchete** | Random windows (baseline). |
| **maxchete** | Probabilistic max-coverage: steers tiles towards low-density zones for an even spread. |
| **infochete** | Content-aware: samples more tiles where the image is informative (high texture/edges), fewer in flat areas. |
| **poischete** | Blue-noise (Poisson-disk): tiles with a guaranteed minimum spacing — an even, cluster-free layout with controlled overlap. |
| **slidechete** | Deterministic sliding window with a user-defined `overlap` — reproducible classic tiling. |

Just instantiate the class and machete the data!

```python 
from mapchete import Tiler

maxchete = Tiler.from_name(input_file, "maxchete")
maxchete.plot_bands()

# Strategies with parameters can be built explicitly:
from mapchete import SlidingWindow, PoissonDisk
slide = Tiler(input_file, SlidingWindow(overlap=0.5))
pois  = Tiler(input_file, PoissonDisk(min_dist=400))
```

<p align="center">
  <img width="400" alt="mapchete_final" src="https://user-images.githubusercontent.com/76526314/219875276-3a05f852-d68b-4f41-a684-f48147edbda5.png">
</p>

### Run to get the tiles: 

```python
maxchete.get_rasters(avg_density=4, size=512 , no_data_percentage=0.3, output_path="raster_clip", clear_output_path=True)
# avg_density In how many output images a given pixel of the input image will be in average.
```

Each saved tile is a GeoTIFF (plus a pickle of its source `Window`). Edge tiles are kept only when their `nodata` fraction stays within `no_data_percentage`:

<p align="center">
  <img width="540" alt="sample generated tiles" src="docs/img/tiles.png">
</p>

### Study the output
```python 
fig, ax = maxchete.get_3Ddistribution()
```

<p align="center">
  <img width="500" alt="mapchete_final" src="https://user-images.githubusercontent.com/76526314/219876116-ce051ecf-021d-4996-bc1b-e68274f624b1.png">
</p>


There is another useful function called merge_tiffs which can merge generated images to se check the distribution fo tiles. If you use lower sampling, this becomes a useful tool, but if you opt for higher sampling, the algorithm should be capable of generating the complete extent of the original image.

``` python
from mapchete import merge_tiffs
merge_tiffs(folder="raster_clip")
```
<p align="center">
  <img width="400" alt="mapchete_final" src="https://user-images.githubusercontent.com/76526314/219876203-2e36d9b6-9edf-4982-b9ba-c3d8c559c962.png">
</p>

# Installation

There is no PyPI release yet, so install from source. The project ships a
standard `pyproject.toml`, so any modern installer works:

```bash
git clone https://github.com/abetatos/mapchete.git
cd mapchete
pip install .
```

### Development setup

The repository uses [uv](https://docs.astral.sh/uv/) for environment
management. To set up an isolated environment with the test dependencies and run
the suite:

```bash
uv sync --extra test
uv run pytest
```

`mapchete` requires Python >= 3.9. It was developed against `rasterio` 1.5, but
the dependency is only lower-bounded (`>=1.3`) to give the user more
flexibility.
