# cf2ogc — CF-NetCDF to rasters an OGC service can serve
# Copyright (C) 2026 Norwegian Meteorological Institute (MET Norway)
# Author: Massimo Di Stefano <massimods@met.no>
# SPDX-License-Identifier: GPL-3.0-or-later

"""cf2ogc.vector: u, v and speed as the three bands of one raster."""

from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

xr = pytest.importorskip("xarray")
pytest.importorskip("osgeo")
pytest.importorskip("pyproj")

from osgeo import gdal  # noqa: E402

from cf2ogc.grid import SourceError  # noqa: E402
from cf2ogc.vector import (  # noqa: E402
    VectorField,
    components_from_direction,
    convert,
    direction_convention,
    speed_standard_name,
)

gdal.UseExceptions()

# NorKyst v3's grid: north polar stereographic, central meridian 70°E.
STERE = "+proj=stere +lat_0=90 +lat_ts=60 +lon_0=70 +x_0=3369600 +y_0=1844800 +R=6371000 +units=m +no_defs"


def _bands(path: str | Path) -> list[np.ndarray]:
    raster = gdal.Open(str(path))
    try:
        return [raster.GetRasterBand(i).ReadAsArray() for i in range(1, raster.RasterCount + 1)]
    finally:
        raster = None


def _uv_lonlat(tmp_path: Path, *, u_attrs: dict | None = None, v_attrs: dict | None = None) -> Path:
    """(time, depth, lat, lon) currents on a regular lon/lat grid; u = 1 + t + k/10, v = -2."""

    times = np.array(["2026-11-15T01:00:00", "2026-11-15T02:00:00"], dtype="datetime64[ns]")
    depths = np.array([0.0, 10.0])
    rows, cols = 4, 3
    u = np.empty((2, 2, rows, cols), dtype="float32")
    for t in range(2):
        for k in range(2):
            u[t, k] = 1.0 + t + k / 10.0
    v = np.full((2, 2, rows, cols), -2.0, dtype="float32")
    u[1, 1, 0, 0] = np.nan
    ds = xr.Dataset(
        {
            "u_eastward": (("time", "depth", "lat", "lon"), u, u_attrs if u_attrs is not None else {
                "standard_name": "eastward_sea_water_velocity", "units": "m s-1",
                "long_name": "eastward current", "_FillValue": -32767.0}),
            "v_northward": (("time", "depth", "lat", "lon"), v, v_attrs if v_attrs is not None else {
                "standard_name": "northward_sea_water_velocity", "units": "m s-1",
                "long_name": "northward current", "_FillValue": -32767.0}),
        },
        coords={
            "time": ("time", times, {"axis": "T", "standard_name": "time"}),
            "depth": ("depth", depths, {"standard_name": "depth", "units": "m", "positive": "down", "axis": "Z"}),
            "lat": ("lat", np.linspace(66.0, 63.0, rows), {"standard_name": "latitude", "units": "degrees_north"}),
            "lon": ("lon", np.linspace(5.0, 7.0, cols), {"standard_name": "longitude", "units": "degrees_east"}),
        },
    )
    path = tmp_path / "uv.nc"
    ds.to_netcdf(path)
    ds.close()
    return path


def _waves(tmp_path: Path, *, dir_attrs: dict, values: list[float], magnitudes: list[float]) -> Path:
    """(time, lat, lon) direction + magnitude, one value per column."""

    times = np.array(["2026-11-15T00:00:00"], dtype="datetime64[ns]")
    direction = np.array([[values, values]], dtype="float32")
    magnitude = np.array([[magnitudes, magnitudes]], dtype="float32")
    ds = xr.Dataset(
        {
            "thq": (("time", "lat", "lon"), direction, dir_attrs),
            "hs": (("time", "lat", "lon"), magnitude, {
                "standard_name": "sea_surface_wave_significant_height", "units": "m",
                "long_name": "significant wave height"}),
        },
        coords={
            "time": ("time", times, {"axis": "T", "standard_name": "time"}),
            "lat": ("lat", np.array([61.0, 60.0]), {"standard_name": "latitude", "units": "degrees_north"}),
            "lon": ("lon", np.linspace(4.0, 4.0 + len(values) - 1, len(values)),
                    {"standard_name": "longitude", "units": "degrees_east"}),
        },
    )
    path = tmp_path / "waves.nc"
    ds.to_netcdf(path)
    ds.close()
    return path


def _stere_source(tmp_path: Path, *, u: float, v: float) -> tuple[Path, np.ndarray, np.ndarray]:
    """A uniform (u, v) field on a NorKyst-like polar stereographic grid, with 2-D lon/lat."""

    from pyproj import Transformer

    rows, cols = 6, 8
    x = 3369600.0 - 1_400_000.0 + np.arange(cols) * 200_000.0     # spans lon ~ 10°E .. 70°E+
    y = 1844800.0 - 2_800_000.0 + np.arange(rows) * 100_000.0
    xx, yy = np.meshgrid(x, y)
    to_lonlat = Transformer.from_crs(STERE, "EPSG:4326", always_xy=True)
    lon, lat = to_lonlat.transform(xx, yy)
    shape = (1, rows, cols)
    ds = xr.Dataset(
        {
            "u_eastward": (("time", "Y", "X"), np.full(shape, u, dtype="float32"), {
                "standard_name": "eastward_sea_water_velocity", "units": "m s-1",
                "grid_mapping": "projection_stere"}),
            "v_northward": (("time", "Y", "X"), np.full(shape, v, dtype="float32"), {
                "standard_name": "northward_sea_water_velocity", "units": "m s-1",
                "grid_mapping": "projection_stere"}),
            "projection_stere": ((), 0, {
                "grid_mapping_name": "polar_stereographic",
                "straight_vertical_longitude_from_pole": 70.0,
                "latitude_of_projection_origin": 90.0,
                "standard_parallel": 60.0,
                "false_easting": 3369600.0,
                "false_northing": 1844800.0,
                "proj4": STERE,
            }),
        },
        coords={
            "time": ("time", np.array(["2026-11-15T01:00:00"], dtype="datetime64[ns]"),
                     {"axis": "T", "standard_name": "time"}),
            "X": ("X", x, {"standard_name": "projection_x_coordinate", "units": "m", "axis": "X"}),
            "Y": ("Y", y, {"standard_name": "projection_y_coordinate", "units": "m", "axis": "Y"}),
            "lon": (("Y", "X"), lon, {"standard_name": "longitude", "units": "degrees_east"}),
            "lat": (("Y", "X"), lat, {"standard_name": "latitude", "units": "degrees_north"}),
        },
    )
    path = tmp_path / "stere.nc"
    ds.to_netcdf(path)
    ds.close()
    return path, lon, lat


# ---------------------------------------------------------------------------
# u, v and speed bands, with the time and vertical axes of cf2ogc.grid
# ---------------------------------------------------------------------------

def test_components_are_written_as_u_v_and_speed_at_the_selected_time_and_level(tmp_path):
    path = _uv_lonlat(tmp_path)
    out = tmp_path / "uv.tif"

    described = convert(path, out, u="u_eastward", v="v_northward", time="2026-11-15T02:00:00Z", level=0.0)

    u, v, speed = _bands(out)
    np.testing.assert_allclose(u, 2.0)
    np.testing.assert_allclose(v, -2.0)
    np.testing.assert_allclose(speed, math.hypot(2.0, -2.0), rtol=1e-6)
    assert [b["role"] for b in described["bands"]] == ["u", "v", "speed"]
    assert [b["variable"] for b in described["bands"]] == ["u_eastward", "v_northward", "speed"]
    assert described["bands"][2]["standard_name"] == "sea_water_speed"
    assert described["time"]["value"] == "2026-11-15T02:00:00Z"
    assert described["level"]["value"] == 0.0 and described["level"]["positive"] == "down"
    # A lon/lat grid has east along +x and north along +y: nothing to turn.
    assert described["vector"]["rotated"] is False
    assert described["vector"]["components"] == "grid"


def test_a_missing_component_blanks_all_three_bands(tmp_path):
    path = _uv_lonlat(tmp_path)
    out = tmp_path / "uv.tif"

    convert(path, out, u="u_eastward", v="v_northward", time_index=1, level_index=1)

    u, v, speed = _bands(out)
    raster = gdal.Open(str(out))
    fill = raster.GetRasterBand(3).GetNoDataValue()
    raster = None
    assert fill == -32767.0
    assert u[0, 0] == v[0, 0] == speed[0, 0] == fill
    np.testing.assert_allclose(u[1:, :], 2.1, rtol=1e-6)


def test_the_band_roles_and_inputs_are_recorded_in_the_raster(tmp_path):
    path = _uv_lonlat(tmp_path)
    out = tmp_path / "uv.tif"
    convert(path, out, u="u_eastward", v="v_northward", time_index=0, level_index=0, driver="GTiff")

    raster = gdal.Open(str(out))
    assert [raster.GetRasterBand(i).GetDescription() for i in (1, 2, 3)] == ["u", "v", "speed"]
    meta = raster.GetMetadata()
    assert meta["CF2OGC_BANDS"] == "u,v,speed"
    assert meta["CF2OGC_COMPONENTS"] == "grid"
    assert meta["CF2OGC_U_VARIABLE"] == "u_eastward"


def test_an_unselected_axis_is_refused_as_in_grid(tmp_path):
    path = _uv_lonlat(tmp_path)
    with pytest.raises(SourceError, match="select one"):
        convert(path, tmp_path / "x.tif", u="u_eastward", v="v_northward", time_index=0)


def test_components_without_a_frame_need_one(tmp_path):
    bare = {"units": "m s-1", "_FillValue": -32767.0}
    path = _uv_lonlat(tmp_path, u_attrs=dict(bare), v_attrs=dict(bare))
    with pytest.raises(SourceError, match="frame="):
        convert(path, tmp_path / "x.tif", u="u_eastward", v="v_northward", time_index=0, level_index=0)
    described = convert(path, tmp_path / "x.tif", u="u_eastward", v="v_northward",
                        time_index=0, level_index=0, frame="earth")
    assert described["vector"]["input_frame"] == "earth"


def test_components_of_different_shapes_are_refused(tmp_path):
    path = _uv_lonlat(tmp_path)
    ds = xr.open_dataset(path)
    ds["v2"] = ds["v_northward"].isel(depth=0)
    with pytest.raises(SourceError):
        VectorField(ds, "u_eastward", "v2")
    ds.close()


def test_speed_standard_names():
    assert speed_standard_name("eastward_sea_water_velocity") == "sea_water_speed"
    assert speed_standard_name("eastward_wind") == "wind_speed"
    assert speed_standard_name("northward_sea_ice_velocity") == "sea_ice_speed"
    assert speed_standard_name("air_temperature") == ""


# ---------------------------------------------------------------------------
# Direction and magnitude, with the convention read from CF
# ---------------------------------------------------------------------------

def test_from_and_to_conventions_are_read_from_cf_attributes():
    class V:
        def __init__(self, **attrs):
            self.attrs = attrs

    assert direction_convention(V(standard_name="sea_surface_wave_from_direction")) == "from"
    assert direction_convention(V(standard_name="wind_from_direction")) == "from"
    assert direction_convention(V(standard_name="sea_water_velocity_to_direction")) == "to"
    assert direction_convention(V(long_name="mean wave direction, coming from")) == "from"
    assert direction_convention(V(long_name="current direction (going to)")) == "to"
    assert direction_convention(V(long_name="mean wave direction")) is None


def test_hand_computed_components_from_a_bearing_and_a_magnitude():
    # (bearing, magnitude, convention) -> (u, v), worked by hand:
    #   from 0° (a northerly), 2 m: flows south   -> ( 0, -2)
    #   from 90° (an easterly), 1 m: flows west   -> (-1,  0)
    #   to 225° (south-west), √2 m                 -> (-1, -1)
    u, v = components_from_direction([0.0, 90.0], [2.0, 1.0], "from")
    np.testing.assert_allclose(u, [0.0, -1.0], atol=1e-6)
    np.testing.assert_allclose(v, [-2.0, 0.0], atol=1e-6)
    u, v = components_from_direction([225.0], [math.sqrt(2.0)], "to")
    np.testing.assert_allclose(u, [-1.0], atol=1e-6)
    np.testing.assert_allclose(v, [-1.0], atol=1e-6)


def test_a_from_direction_field_is_converted_and_its_convention_recorded(tmp_path):
    path = _waves(
        tmp_path,
        dir_attrs={"standard_name": "sea_surface_wave_from_direction", "units": "degree"},
        values=[0.0, 90.0, 225.0],
        magnitudes=[2.0, 1.0, math.sqrt(2.0)],
    )
    out = tmp_path / "waves.tif"

    described = convert(path, out, direction="thq", magnitude="hs", time_index=0)

    u, v, speed = _bands(out)
    np.testing.assert_allclose(u[0], [0.0, -1.0, 1.0], atol=1e-6)
    np.testing.assert_allclose(v[0], [-2.0, 0.0, 1.0], atol=1e-6)
    np.testing.assert_allclose(speed[0], [2.0, 1.0, math.sqrt(2.0)], rtol=1e-6)
    assert described["vector"]["source"] == "direction_magnitude"
    assert described["vector"]["direction_convention"] == "from"
    assert described["vector"]["direction_variable"] == "thq"
    assert described["bands"][0]["units"] == "m"
    meta = gdal.Open(str(out)).GetMetadata()
    assert meta["CF2OGC_DIRECTION_CONVENTION"] == "from"


def test_a_direction_without_a_convention_needs_one_and_radians_are_read(tmp_path):
    path = _waves(tmp_path, dir_attrs={"units": "rad"}, values=[math.pi / 2] * 2, magnitudes=[3.0] * 2)
    with pytest.raises(SourceError, match="convention="):
        convert(path, tmp_path / "x.tif", direction="thq", magnitude="hs", time_index=0)

    described = convert(path, tmp_path / "x.tif", direction="thq", magnitude="hs", time_index=0, convention="to")
    u, v, _ = _bands(tmp_path / "x.tif")
    np.testing.assert_allclose(u, 3.0, atol=1e-5)
    np.testing.assert_allclose(v, 0.0, atol=1e-5)
    assert described["vector"]["direction_convention"] == "to"


def test_a_direction_in_unknown_units_is_refused(tmp_path):
    path = _waves(tmp_path, dir_attrs={"units": "K", "standard_name": "wind_from_direction"},
                  values=[1.0, 1.0], magnitudes=[1.0, 1.0])
    with pytest.raises(SourceError, match="degrees or radians"):
        convert(path, tmp_path / "x.tif", direction="thq", magnitude="hs", time_index=0)


# ---------------------------------------------------------------------------
# Components turned into the output grid's axes
# ---------------------------------------------------------------------------

def test_on_a_polar_stereographic_grid_north_is_turned_towards_the_pole(tmp_path):
    path, lon, lat = _stere_source(tmp_path, u=0.0, v=1.0)
    out = tmp_path / "stere.tif"

    described = convert(path, out, u="u_eastward", v="v_northward", time_index=0)

    gu, gv, speed = _bands(out)
    assert described["vector"]["rotated"] is True
    # The source grid is written as it is (y rows run south to north here, so
    # pixel row r holds Y[r]); north at a point points to the pole: on a north
    # polar stereographic grid with central meridian 70°E that is
    # (-sin(lon-70°), cos(lon-70°)) in grid x/y.
    raster = gdal.Open(str(out))
    gt = raster.GetGeoTransform()
    raster = None
    rows = gu.shape[0]
    flip = gt[5] < 0      # a north-up output stores Y descending
    for r in range(rows):
        for c in range(gu.shape[1]):
            source_r = rows - 1 - r if flip else r
            delta = math.radians(lon[source_r, c] - 70.0)
            np.testing.assert_allclose(gu[r, c], -math.sin(delta), atol=2e-4)
            np.testing.assert_allclose(gv[r, c], math.cos(delta), atol=2e-4)
    np.testing.assert_allclose(speed, 1.0, rtol=1e-6)
    # At 10°E the grid is 60° off: north is (sin 60°, cos 60°) on the grid.
    probe = np.argmin(np.abs(lon - 10.0))
    pr, pc = np.unravel_index(probe, lon.shape)
    if abs(lon[pr, pc] - 10.0) < 1.0:
        row = rows - 1 - pr if flip else pr
        delta = math.radians(lon[pr, pc] - 70.0)
        assert gu[row, pc] == pytest.approx(-math.sin(delta), abs=2e-4)


def test_earth_components_are_left_eastward_and_northward_when_asked(tmp_path):
    path, _, _ = _stere_source(tmp_path, u=0.0, v=1.0)
    out = tmp_path / "earth.tif"

    described = convert(path, out, u="u_eastward", v="v_northward", time_index=0, components="earth")

    gu, gv, _ = _bands(out)
    np.testing.assert_allclose(gu, 0.0, atol=1e-7)
    np.testing.assert_allclose(gv, 1.0, atol=1e-7)
    assert described["vector"]["components"] == "earth"
    assert described["vector"]["rotated"] is False


def test_warped_to_web_mercator_east_and_north_are_the_grid_axes(tmp_path):
    path, _, _ = _stere_source(tmp_path, u=0.6, v=0.8)
    out = tmp_path / "merc.tif"

    described = convert(path, out, u="u_eastward", v="v_northward", time_index=0, target_epsg=3857)

    gu, gv, speed = _bands(out)
    valid = np.isfinite(gu) & (gu != described["bands"][0]["nodata"])
    assert valid.sum() > 0
    np.testing.assert_allclose(gu[valid], 0.6, atol=1e-6)
    np.testing.assert_allclose(gv[valid], 0.8, atol=1e-6)
    np.testing.assert_allclose(speed[valid], 1.0, rtol=1e-6)
    assert described["reprojected"] is True
    assert described["vector"]["rotated"] is False
    assert described["epsg"] == 3857


def test_grid_relative_components_are_not_warped(tmp_path):
    path, _, _ = _stere_source(tmp_path, u=1.0, v=0.0)
    ds = xr.open_dataset(path)
    field = VectorField(ds, "u_eastward", "v_northward", frame="grid")
    chosen = field.selection(time_index=0)
    with pytest.raises(SourceError, match="own grid"):
        field.write(field.read(chosen), tmp_path / "x.tif", target_epsg=3857)
    ds.close()


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------

def test_the_command_line_writes_a_vector_raster(tmp_path):
    path = _uv_lonlat(tmp_path)
    out = tmp_path / "cli.tif"
    result = subprocess.run(
        [sys.executable, "-m", "cf2ogc", "vector", str(path), "-o", str(out),
         "--u", "u_eastward", "--v", "v_northward", "--time-index", "0", "--level", "10"],
        capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, result.stderr
    described = json.loads(result.stdout)
    assert described["level"]["value"] == 10.0
    np.testing.assert_allclose(_bands(out)[0], 1.1, rtol=1e-6)
