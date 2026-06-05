"""Tests for the merge_tiffs helper."""
import os

import pytest

from mapchete import FARMchete, merge_tiffs


def test_merge_tiffs_creates_mosaic(raster_path, tmp_path):
    output_path = str(tmp_path / "tiles")
    chete = FARMchete(raster_path).get("slidechete")
    chete.get_rasters(avg_density=1, size=64, no_data_percentage=0.3,
                      output_path=output_path, clear_output_path=True)

    merge_tiffs(folder=output_path)

    mosaic = os.path.join(output_path, "merged", "mosaic_output.tif")
    assert os.path.exists(mosaic)


def test_merge_tiffs_raises_on_empty_folder(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(FileNotFoundError):
        merge_tiffs(folder=str(empty))
