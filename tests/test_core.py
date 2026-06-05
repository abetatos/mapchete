"""Tests for the Tiler engine and the public API."""
import pytest

from mapchete import FARMchete, Tiler, MaxCoverage


def test_tiler_direct_api(raster_path):
    tiler = Tiler(raster_path, MaxCoverage())
    assert tiler.identifier == "HS"
    assert tiler.strategy.name == "maxchete"


def test_from_name_matches_farmchete(raster_path):
    assert isinstance(Tiler.from_name(raster_path, "maxchete"), Tiler)
    assert isinstance(FARMchete(raster_path).get("slidechete"), Tiler)


def test_loads_raster_metadata(raster_path):
    tiler = Tiler.from_name(raster_path, "maxchete")
    # The loader trims the nodata border via get_data_window, so the working
    # array is the data footprint, never larger than the source (256x256).
    assert 0 < tiler.width <= 256
    assert 0 < tiler.height <= 256
    assert tiler.nodata == 0
    assert tiler.identifier == "HS"
    assert tiler.array.shape == (tiler.height, tiler.width)


def test_plot_helpers_do_not_raise(raster_path):
    tiler = Tiler.from_name(raster_path, "maxchete")
    tiler.plot_bands()
    tiler.plot_hist()
    tiler.plot_hist(delete_nodata=False)


def test_distribution_requires_a_run(raster_path):
    tiler = Tiler.from_name(raster_path, "maxchete")
    with pytest.raises(RuntimeError):
        _ = tiler.distribution_array


def test_get_3d_distribution_returns_fig_ax(raster_path, tmp_path):
    tiler = Tiler.from_name(raster_path, "maxchete")
    tiler.get_rasters(avg_density=1, size=64, no_data_percentage=0.3,
                      output_path=str(tmp_path / "out"), clear_output_path=True)
    fig, ax = tiler.get_3Ddistribution()
    assert fig is not None and ax is not None


def test_clear_opath_creates_and_clears(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    (out / "stale.txt").write_text("old")

    Tiler.clear_opath(str(out), clear_output_path=True)
    assert out.exists()
    assert not (out / "stale.txt").exists()


@pytest.mark.parametrize(
    "crop_type",
    ["randchete", "maxchete", "infochete", "poischete", "slidechete"],
)
def test_raises_when_size_too_big(raster_path, tmp_path, crop_type):
    tiler = Tiler.from_name(raster_path, crop_type)
    # A tile bigger than the input leaves no room to place a window.
    with pytest.raises(ValueError):
        tiler.get_rasters(avg_density=1, size=512, no_data_percentage=0.3,
                          output_path=str(tmp_path / "out"), clear_output_path=True)


def test_unknown_strategy_name_raises(raster_path):
    with pytest.raises(ValueError):
        Tiler.from_name(raster_path, "nonsense")
