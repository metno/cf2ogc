# cf2ogc — CF-NetCDF to rasters an OGC service can serve
# Copyright (C) 2026 Norwegian Meteorological Institute (MET Norway)
# Author: Massimo Di Stefano <massimods@met.no>
# SPDX-License-Identifier: GPL-3.0-or-later

"""cf2ogc.grid: one variable, one time step, one vertical level to a raster."""

from __future__ import annotations

import json
import subprocess
from collections import Counter
import sys
from pathlib import Path

import numpy as np
import pytest

xr = pytest.importorskip("xarray")
pytest.importorskip("osgeo")

from osgeo import gdal, osr  # noqa: E402

from cf2ogc import grid  # noqa: E402
from cf2ogc.grid import GridField, ReadError, SourceError, convert, find_vertical_axis, open_source  # noqa: E402

gdal.UseExceptions()


def _four_d_source(tmp_path: Path, *, depth_attrs: dict | None = None, name: str = "depth") -> Path:
    """(time, depth, y, x) on a regular lon/lat grid; value = 100*t + 10*k + a ramp."""

    times = np.array(["2024-01-01T00:00:00", "2024-01-01T06:00:00"], dtype="datetime64[ns]")
    depths = np.array([0.0, 10.0, 50.0])
    rows, cols = 5, 4
    ramp = np.arange(rows * cols, dtype="float32").reshape(rows, cols) / 100.0
    data = np.empty((len(times), len(depths), rows, cols), dtype="float32")
    for t in range(len(times)):
        for k in range(len(depths)):
            data[t, k] = 100.0 * t + 10.0 * k + ramp
    data[1, 2, 0, 0] = np.nan
    attrs = {"standard_name": "depth", "units": "m", "positive": "down", "axis": "Z"}
    if depth_attrs is not None:
        attrs = depth_attrs
    ds = xr.Dataset(
        {"temp": (("time", name, "lat", "lon"), data, {
            "standard_name": "sea_water_temperature", "units": "degC", "_FillValue": -999.0})},
        coords={
            "time": ("time", times, {"axis": "T", "standard_name": "time"}),
            name: (name, depths, attrs),
            "lat": ("lat", np.linspace(70.0, 66.0, rows), {"standard_name": "latitude", "units": "degrees_north"}),
            "lon": ("lon", np.linspace(5.0, 8.0, cols), {"standard_name": "longitude", "units": "degrees_east"}),
        },
    )
    path = tmp_path / "four_d.nc"
    ds.to_netcdf(path)
    ds.close()
    return path


def _read(path: str | Path) -> np.ndarray:
    raster = gdal.Open(str(path))
    try:
        return raster.GetRasterBand(1).ReadAsArray()
    finally:
        raster = None


# ---------------------------------------------------------------------------
# The vertical axis is found by CF, and selected by value or index
# ---------------------------------------------------------------------------

def test_a_four_d_field_writes_the_selected_time_and_level(tmp_path):
    path = _four_d_source(tmp_path)
    out = tmp_path / "t1_d10.tif"

    described = convert(path, "temp", out, time="2024-01-01T06:00:00Z", level=10.0)

    expected = 100.0 + 10.0 + np.arange(20, dtype="float32").reshape(5, 4) / 100.0
    np.testing.assert_array_equal(_read(out), expected)
    assert described["time"] == {"dim": "time", "index": 1, "value": "2024-01-01T06:00:00Z"}
    assert described["level"] == {
        "dim": "depth", "index": 1, "value": 10.0, "units": "m", "positive": "down",
        "standard_name": "depth", "parametric": False,
    }
    band = described["bands"][0]
    assert (band["variable"], band["units"], band["nodata"]) == ("temp", "degC", -999.0)
    assert described["epsg"] == 4326 and described["reprojected"] is False


def test_the_same_slice_by_index(tmp_path):
    path = _four_d_source(tmp_path)
    by_value = convert(path, "temp", tmp_path / "a.tif", time="2024-01-01T06:00:00Z", level=50.0)
    by_index = convert(path, "temp", tmp_path / "b.tif", time_index=1, level_index=2)

    np.testing.assert_array_equal(_read(by_value["path"]), _read(by_index["path"]))
    # The NaN became the declared fill, and the band says so.
    assert _read(by_index["path"])[0, 0] == -999.0
    assert gdal.Open(by_index["path"]).GetRasterBand(1).GetNoDataValue() == -999.0


def test_an_unselected_axis_is_never_guessed(tmp_path):
    path = _four_d_source(tmp_path)
    with pytest.raises(SourceError, match="3 level values"):
        convert(path, "temp", tmp_path / "x.tif", time_index=0)
    with pytest.raises(SourceError, match="2 time values"):
        convert(path, "temp", tmp_path / "x.tif", level_index=0)


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"time_index": 0, "level": 20.0}, "Level 20 is not on the depth axis"),
        ({"time_index": 0, "level_index": 3}, "level index 3 is outside 0..2"),
        ({"time": "2024-02-01", "level_index": 0}, "Time 2024-02-01T00:00:00Z is not on the axis"),
        ({"time_index": 0, "time": "2024-01-01", "level_index": 0}, "by value or by index, not both"),
    ],
)
def test_a_selection_off_the_axis_is_refused(tmp_path, kwargs, message):
    path = _four_d_source(tmp_path)
    with pytest.raises(SourceError, match=message):
        convert(path, "temp", tmp_path / "x.tif", **kwargs)


@pytest.mark.parametrize(
    "attrs, source, positive",
    [
        ({"axis": "Z", "units": "m"}, "axis", ""),
        ({"positive": "up", "units": "m"}, "positive", "up"),
        ({"standard_name": "height", "units": "m"}, "standard_name", "up"),
        ({"units": "hPa"}, "units", "down"),
        ({}, "name", ""),
    ],
)
def test_the_vertical_axis_is_recognised_by_cf_attributes(tmp_path, attrs, source, positive):
    ds = open_source(str(_four_d_source(tmp_path, depth_attrs=attrs, name="lev")))
    try:
        axis = find_vertical_axis(ds, "temp", "time")
    finally:
        ds.close()
    assert axis is not None
    assert (axis.dim, axis.source, axis.positive, axis.values) == ("lev", source, positive, [0.0, 10.0, 50.0])


def test_a_parametric_coordinate_is_reported_as_model_levels(tmp_path):
    attrs = {"standard_name": "ocean_s_coordinate_g2", "positive": "up", "units": "1"}
    ds = open_source(str(_four_d_source(tmp_path, depth_attrs=attrs, name="s_rho")))
    try:
        axis = find_vertical_axis(ds, "temp", "time")
    finally:
        ds.close()
    assert axis.parametric is True and axis.dim == "s_rho"


def test_a_dimension_that_is_neither_time_nor_vertical_is_refused(tmp_path):
    ds = xr.Dataset(
        {"v": (("member", "lat", "lon"), np.zeros((2, 3, 3), dtype="float32"))},
        coords={
            "member": ("member", [0, 1], {"standard_name": "realization"}),
            "lat": ("lat", [60.0, 59.0, 58.0], {"standard_name": "latitude", "units": "degrees_north"}),
            "lon": ("lon", [5.0, 6.0, 7.0], {"standard_name": "longitude", "units": "degrees_east"}),
        },
    )
    with pytest.raises(SourceError, match="neither time nor vertical"):
        GridField(ds, "v")


def test_a_two_d_field_needs_no_selection(tmp_path, source_path):
    described = convert(source_path, "land_mask", tmp_path / "mask.tif")
    assert described["time"] is None and described["level"] is None
    assert (_read(described["path"]) == 1.0).any()


# ---------------------------------------------------------------------------
# Output: a COG by default, the writer's options honoured
# ---------------------------------------------------------------------------

def test_the_default_output_is_a_cloud_optimized_geotiff(tmp_path, projected_source):
    out = tmp_path / "age.tif"
    described = convert(projected_source, "sea_ice_age", out, time_index=0)

    raster = gdal.Open(str(out))
    assert raster.GetMetadataItem("LAYOUT", "IMAGE_STRUCTURE") == "COG"
    assert raster.GetMetadataItem("COMPRESSION", "IMAGE_STRUCTURE") == "DEFLATE"
    assert described["driver"] == "COG"
    assert described["epsg"] == 6931
    assert described["bounds"] == pytest.approx([-100000.0, -100000.0, 100000.0, 100000.0])
    assert described["geotransform"] == pytest.approx(list(raster.GetGeoTransform()))
    assert "6931" in raster.GetProjection()


def test_a_gtiff_and_a_cog_hold_the_same_pixels(tmp_path, source_path):
    cog = convert(source_path, "snow_probability", tmp_path / "c.tif", time_index=1)
    tif = convert(
        source_path, "snow_probability", tmp_path / "g.tif", time_index=1,
        driver="GTiff", creation_options=["TILED=YES", "COMPRESS=DEFLATE", "BIGTIFF=IF_SAFER"],
    )
    np.testing.assert_array_equal(_read(cog["path"]), _read(tif["path"]))
    assert cog["geotransform"] == tif["geotransform"]


def test_a_packed_variable_is_written_in_physical_units(tmp_path, projected_source):
    described = convert(projected_source, "conc_1yi", tmp_path / "conc.tif", time_index=0)
    assert np.allclose(_read(described["path"]), 0.5)
    assert described["bands"][0]["valid_range"] == pytest.approx([0.0, 1.1])


def test_a_nodata_override_replaces_the_declared_fill(tmp_path, projected_source):
    described = convert(projected_source, "sea_ice_age", tmp_path / "o.tif", time_index=0, nodata=-1.0)
    assert described["bands"][0]["nodata"] == -1.0
    assert gdal.Open(described["path"]).GetRasterBand(1).GetNoDataValue() == -1.0


def test_an_explicit_target_epsg_reprojects(tmp_path, projected_source):
    described = convert(projected_source, "sea_ice_age", tmp_path / "m.tif", time_index=0, target_epsg=3857)
    assert described["reprojected"] is True and described["epsg"] == 3857
    assert "3857" in gdal.Open(described["path"]).GetProjection()


def test_every_slice_of_a_field_shares_one_target_grid(tmp_path, source_path):
    ds = open_source(str(source_path))
    try:
        field = GridField(ds, "snow_probability")
        transforms = {
            tuple(field.convert(tmp_path / f"s{i}.tif", time_index=i)["geotransform"]) for i in range(3)
        }
    finally:
        ds.close()
    # All timesteps must land on one grid, or a time series would jitter.
    assert len(transforms) == 1


# ---------------------------------------------------------------------------
# Curvilinear grids: 2-D lon/lat warped through GDAL geolocation arrays
# ---------------------------------------------------------------------------

def _rotated_curvilinear(tmp_path: Path) -> Path:
    """A grid rotated 30 degrees against the meridians; each cell holds its own index."""

    rows, cols = 40, 30
    j, i = np.meshgrid(np.arange(cols, dtype="float64"), np.arange(rows, dtype="float64"))
    angle = np.deg2rad(30.0)
    east = (j * np.cos(angle) + i * np.sin(angle)) * 0.05
    north = (j * np.sin(angle) - i * np.cos(angle)) * 0.05
    lon = 10.0 + east / np.cos(np.deg2rad(60.0))
    lat = 60.0 + north
    values = (i * cols + j).astype("float32")
    ds = xr.Dataset(
        {"cell": (("y", "x"), values, {"units": "1", "_FillValue": -1.0})},
        coords={
            "lon": (("y", "x"), lon, {"standard_name": "longitude", "units": "degrees_east"}),
            "lat": (("y", "x"), lat, {"standard_name": "latitude", "units": "degrees_north"}),
        },
    )
    path = tmp_path / "rotated.nc"
    ds.to_netcdf(path)
    ds.close()
    return path


def _offsets(path: Path, described: dict) -> tuple[float, Counter]:
    """For each interior source cell: which source cell the output holds at its lon/lat."""

    out = gdal.Open(described["path"])
    pixels = out.GetRasterBand(1).ReadAsArray()
    gt = out.GetGeoTransform()
    src = xr.open_dataset(path)
    try:
        lon, lat, cell = src["lon"].values, src["lat"].values, src["cell"].values
    finally:
        src.close()
    rows, cols = cell.shape
    seen: Counter = Counter()
    for i in range(2, rows - 2):
        for j in range(2, cols - 2):
            col = int((lon[i, j] - gt[0]) / gt[1])
            row = int((lat[i, j] - gt[3]) / gt[5])
            gi, gj = divmod(int(pixels[row, col]), cols)
            seen[(gi - i, gj - j)] += 1
    total = sum(seen.values())
    return seen[(0, 0)] / total, seen


def test_a_curvilinear_grid_lands_where_its_lon_lat_say(tmp_path):
    path = _rotated_curvilinear(tmp_path)
    described = convert(path, "cell", tmp_path / "rotated.tif", target_epsg=4326)
    assert described["georeferencing"] == "geolocation_arrays"
    assert described["reprojected"] is True

    # An interior cell, looked up at its own lon/lat, holds itself or a direct
    # neighbour (nearest neighbour on a rotated grid, and the half-cell shift
    # of the next test); never anything further than two cells away.
    _, seen = _offsets(path, described)
    total = sum(seen.values())
    within_one = sum(n for (di, dj), n in seen.items() if abs(di) <= 1 and abs(dj) <= 1)
    assert within_one / total > 0.99, seen
    assert all(abs(di) <= 2 and abs(dj) <= 2 for di, dj in seen), seen


def test_the_default_geolocation_convention_reads_lon_lat_as_cell_corners(tmp_path):
    """GDAL's default places a curvilinear grid half a cell off; it is kept, and measured here.

    The neighbour found at a cell's own lon/lat is biased to the previous row
    or column; with PIXEL_CENTER it is the cell itself, and the misses are
    spread evenly around it.
    """

    path = _rotated_curvilinear(tmp_path)
    default_exact, default_seen = _offsets(path, convert(path, "cell", tmp_path / "d.tif", target_epsg=4326))
    centre_exact, centre_seen = _offsets(
        path, convert(path, "cell", tmp_path / "c.tif", target_epsg=4326, geoloc_convention="PIXEL_CENTER")
    )

    assert default_seen[(-1, 0)] + default_seen[(0, -1)] > 2 * default_seen[(0, 0)]
    assert centre_exact > 0.45 > default_exact
    assert abs(centre_seen[(1, 0)] - centre_seen[(-1, 0)]) < 0.1 * sum(centre_seen.values())


def test_the_default_crs_of_a_curvilinear_grid_is_web_mercator(tmp_path, source_path):
    described = convert(source_path, "snow_probability", tmp_path / "w.tif", time_index=0)
    assert described["epsg"] == 3857
    srs = osr.SpatialReference(wkt=gdal.Open(described["path"]).GetProjection())
    assert srs.GetAuthorityCode(None) == "3857"


def test_geolocation_rasters_are_shared_and_scratch_is_removed(tmp_path, source_path):
    shared = tmp_path / "geoloc"
    scratch = tmp_path / "scratch"
    ds = open_source(str(source_path))
    try:
        field = GridField(ds, "snow_probability")
        for i in range(2):
            field.write(field.read({"time": i}), tmp_path / f"o{i}.tif", scratch_dir=scratch, geoloc_dir=shared)
    finally:
        ds.close()
    assert sorted(p.name for p in shared.iterdir()) == ["_geoloc_lat.tif", "_geoloc_lon.tif"]
    assert list(scratch.iterdir()) == []


# ---------------------------------------------------------------------------
# Reading
# ---------------------------------------------------------------------------

class _Flaky:
    def __init__(self, failures: int):
        self.failures = failures
        self.calls = 0

    def isel(self, selection):
        self.calls += 1
        if self.calls <= self.failures:
            raise OSError("connection reset")
        return xr.DataArray(np.ones((2, 2)))


def test_a_read_is_retried_after_a_transient_failure(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda _s: None)
    flaky = _Flaky(failures=2)
    values = grid.read_slice(flaky, {"time": 0, "depth": 1})
    assert flaky.calls == 3 and values.dtype == np.float32


def test_a_read_that_keeps_failing_names_the_slice(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda _s: None)
    with pytest.raises(ReadError, match="time=4, depth=1 after 3 attempts"):
        grid.read_slice(_Flaky(failures=9), {"time": 4, "depth": 1})
    with pytest.raises(ReadError, match="timestep 4 after 3 attempts"):
        grid.read_timestep(_Flaky(failures=9), "time", 4)


def test_an_unreadable_source_is_a_read_error(tmp_path):
    (tmp_path / "junk.nc").write_text("not netcdf")
    with pytest.raises(ReadError, match="Unable to open NetCDF source"):
        convert(tmp_path / "junk.nc", "v", tmp_path / "x.tif")


def test_an_unknown_variable_is_a_source_error(tmp_path, source_path):
    with pytest.raises(SourceError, match="Unknown variable: nope"):
        convert(source_path, "nope", tmp_path / "x.tif")


# ---------------------------------------------------------------------------
# The command line
# ---------------------------------------------------------------------------

def test_the_command_line_writes_a_slice_and_prints_its_description(tmp_path):
    path = _four_d_source(tmp_path)
    out = tmp_path / "cli.tif"
    run = subprocess.run(
        [sys.executable, "-m", "cf2ogc", "grid", str(path), "temp", "-o", str(out),
         "--time-index", "0", "--level", "50"],
        capture_output=True, text=True, check=True,
    )
    described = json.loads(run.stdout)
    assert described["level"]["value"] == 50.0 and described["time"]["index"] == 0
    np.testing.assert_array_equal(
        _read(out), 20.0 + np.arange(20, dtype="float32").reshape(5, 4) / 100.0
    )


def test_the_command_line_reports_a_bad_selection(tmp_path):
    path = _four_d_source(tmp_path)
    run = subprocess.run(
        [sys.executable, "-m", "cf2ogc", "grid", str(path), "temp", "-o", str(tmp_path / "x.tif"),
         "--time-index", "0"],
        capture_output=True, text=True,
    )
    assert run.returncode == 2
    assert "3 level values" in run.stderr
