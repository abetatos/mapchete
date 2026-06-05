"""Smoke-test the visual demo so it never silently rots.

The demo lives in ``examples/demo.py`` (not a package); we load it by path and
run it into a temp dir, asserting it produces the expected images and that
max-coverage really is more even than random.
"""
import importlib.util
from pathlib import Path

import pytest

DEMO = Path(__file__).resolve().parents[1] / "examples" / "demo.py"


@pytest.fixture
def demo_module():
    spec = importlib.util.spec_from_file_location("mapchete_demo", DEMO)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_demo_generates_images_and_proves_purpose(demo_module, tmp_path):
    out = tmp_path / "demo"
    results = demo_module.main(outdir=str(out))

    # maxchete spreads tiles more evenly than random (lower std).
    assert results["maxchete"]["std"] < results["randchete"]["std"]
    # The coverage-oriented strategies must actually cover the footprint.
    assert results["infochete"]["coverage"] > 0.8
    assert results["poischete"]["coverage"] > 0.8

    for png in ("00_input.png",
                "11_comparison.png",
                "10_distribution_maxchete.png",
                "10_distribution_infochete.png",
                "20_surface_randchete.png",
                "20_surface_maxchete.png",
                "20_surface_infochete.png",
                "30_tiles_maxchete.png",
                "30_tiles_infochete.png"):
        assert (out / png).exists(), f"missing {png}"
