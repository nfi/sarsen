from typing import Any

import numpy as np
import pytest
import xarray as xr

from sarsen import scene


def test_open_dem_raster_close(monkeypatch: pytest.MonkeyPatch) -> None:
    closed: list[bool] = []

    def open_dataarray(*args: Any, **kwargs: Any) -> xr.DataArray:
        # north-up, so that the raster is flipped as well as renamed and squeezed
        opened = xr.DataArray(
            np.zeros((1, 2, 2)),
            dims=("band", "y", "x"),
            coords={"y": [1.0, 0.0], "x": [0.0, 1.0]},
        )
        opened.set_close(lambda: closed.append(True))
        return opened

    monkeypatch.setattr(xr, "open_dataarray", open_dataarray)

    with scene.open_dem_raster("unused") as dem_raster:
        assert dem_raster.dims == ("y", "x")
        assert not closed

    assert closed == [True]


def test_convert_to_dem_3d(dem_raster: xr.DataArray) -> None:
    res = scene.convert_to_dem_3d(dem_raster)

    assert res.dims == ("axis", "y", "x")
    assert res.name == "dem_3d"
    assert res.sel(x=12.5, y=42, method="nearest")[2] == 17.0


@pytest.mark.xfail()
def test_transform_dem_3d(dem_raster: xr.DataArray) -> None:
    dem_3d = scene.convert_to_dem_3d(dem_raster)

    # from height over the geoid to height over the ellipsoid
    res = scene.transform_dem_3d(dem_3d, dem_3d.rio.crs, "EPSG:4979")

    assert res.dims == ("axis", "y", "x")
    # this assert fails if proj-data is not properly installed on the system
    assert abs(res.sel(x=12.5, y=42, method="nearest")[2] - 65.613) < 0.001

    expected = [4634523.742, 1027449.178, 4245647.74]

    # from geographic to geocentric (ECEF)
    res = scene.transform_dem_3d(dem_3d, dem_3d.rio.crs, "EPSG:4978")

    assert res.dims == ("axis", "y", "x")
    assert np.allclose(
        res.sel(x=12.5, y=42, method="nearest"), expected, rtol=0, atol=0.001
    )


def test_compute_dem_oriented_area(dem_raster: xr.DataArray) -> None:
    dem_3d = scene.convert_to_dem_3d(dem_raster)

    res = scene.compute_dem_oriented_area(dem_3d)

    assert set(res.dims) == {"axis", "y", "x"}
    assert res.name == "dem_oriented_area"
