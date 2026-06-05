"""End-to-end tests for the three cropping algorithms."""
import glob
import os

import pytest
import rasterio as rio

from mapchete import FARMchete


SIZE = 64


@pytest.mark.parametrize(
    "crop_type",
    ["randchete", "maxchete", "infochete", "poischete", "slidechete"],
)
def test_get_rasters_produces_valid_tiles(crop_type, raster_path, tmp_path):
    output_path = str(tmp_path / "out")

    chete = FARMchete(raster_path).get(crop_type)
    chete.get_rasters(
        avg_density=1,
        size=SIZE,
        no_data_percentage=0.3,
        output_path=output_path,
        clear_output_path=True,
    )

    tiffs = glob.glob(os.path.join(output_path, "*.tif"))
    assert tiffs, f"{crop_type} produced no tiles"

    # Every tile must have the requested size and respect the nodata budget.
    for tif in tiffs:
        with rio.open(tif) as src:
            assert src.width == SIZE
            assert src.height == SIZE
            band = src.read(1)
            nodata_ratio = (band == chete.nodata).sum() / band.size
            assert nodata_ratio <= 0.3 + 1e-9


@pytest.mark.parametrize(
    "crop_type",
    ["randchete", "maxchete", "infochete", "poischete", "slidechete"],
)
def test_window_pickle_is_written_per_tile(crop_type, raster_path, tmp_path):
    output_path = str(tmp_path / "out")

    chete = FARMchete(raster_path).get(crop_type)
    chete.get_rasters(
        avg_density=1, size=SIZE, no_data_percentage=0.3,
        output_path=output_path, clear_output_path=True,
    )

    n_tif = len(glob.glob(os.path.join(output_path, "*.tif")))
    n_pickle = len(glob.glob(os.path.join(output_path, "*.pickle")))
    assert n_tif == n_pickle


def test_farmchete_rejects_unknown_type(raster_path):
    with pytest.raises(ValueError):
        FARMchete(raster_path).get("nonsense")
