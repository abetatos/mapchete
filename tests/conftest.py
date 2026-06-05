"""Shared fixtures for the mapchete test suite.

There is no real input data shipped with the repository, so we synthesize the
same kind of raster mapchete is built for: an **irregular polygon footprint of
valid data surrounded by nodata**, mimicking a hillshade AOI as described in the
README. The polygon generator is the one used by ``examples/demo.py`` so tests
and the visual demo exercise the exact same data.
"""
import importlib.util
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: never pop up a window during tests

import pytest

# Load the demo module by path (examples/ is not a package) and reuse its
# synthetic-data generator as the single source of truth.
_DEMO_PATH = Path(__file__).resolve().parents[1] / "examples" / "demo.py"
_spec = importlib.util.spec_from_file_location("mapchete_demo", _DEMO_PATH)
_demo = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_demo)


@pytest.fixture
def raster_path(tmp_path):
    """Write a 256px synthetic single-band GeoTIFF (irregular polygon) and return its path."""
    return _demo.irregular_polygon_raster(str(tmp_path / "HS.tif"), size=256)
