# cf2ogc — CF-NetCDF to rasters an OGC service can serve
# Copyright (C) 2026 Norwegian Meteorological Institute (MET Norway)
# Author: Massimo Di Stefano <massimods@met.no>
# SPDX-License-Identifier: GPL-3.0-or-later

"""Georeferencing, CF decoding and timestamps of gridded CF fields."""

from __future__ import annotations

import datetime as dt
import re

import numpy as np
import pytest

xr = pytest.importorskip("xarray")
pytest.importorskip("osgeo")

from cf2ogc.grid import (  # noqa: E402
    SourceError,
    _axis_role,
    _iso,
    _lonlat_axes_1d,
    _same_crs_numerically,
    open_source,
    parse_value_classes,
    resolve_georeferencing,
    valid_range_for,
)

from conftest import make_projected_source  # noqa: E402


# ---------------------------------------------------------------------------
# Value classes
# ---------------------------------------------------------------------------

def test_value_classes_parsed_from_long_name(source_path):
    ds = xr.open_dataset(source_path)
    try:
        classes = parse_value_classes(ds["classed_product"])
    finally:
        ds.close()

    assert [c[2] for c in classes] == ["ocean", "snow free", "snow", "clouded", "no data"]
    # Each class brackets its integer code, so [pixel] expressions select exactly one.
    assert classes[0][0] == -1.5 and classes[0][1] == -0.5
    # Labels drive physically sensible colours rather than an arbitrary ramp.
    assert classes[2][3] == [255, 255, 255]  # snow -> white
    assert classes[0][3] == [15, 45, 95]     # ocean -> deep blue


def test_value_classes_parsed_from_cf_flag_attributes():
    var = xr.DataArray(
        np.zeros((2, 2), dtype="int32"),
        attrs={"flag_values": [0, 1], "flag_meanings": "water land"},
    )
    classes = parse_value_classes(var)
    assert [c[2] for c in classes] == ["water", "land"]


def test_continuous_variable_has_no_value_classes(source_path):
    ds = xr.open_dataset(source_path)
    try:
        assert parse_value_classes(ds["snow_probability"]) == []
    finally:
        ds.close()


# ---------------------------------------------------------------------------
# Tiered georeferencing: a source that carries a real CRS must not be resampled.
# ---------------------------------------------------------------------------

def test_grid_mapping_source_resolves_to_a_crs_not_geolocation_arrays(projected_source):
    ds = open_source(str(projected_source))
    try:
        grid = resolve_georeferencing(ds, "sea_ice_age")
    finally:
        ds.close()

    assert grid.kind == "crs"
    assert grid.source == "grid_mapping:Lambert_Azimuthal_Equal_Area"
    # The file names its own CRS, so the authority is adopted rather than lost.
    assert grid.epsg == 6931
    # Cell centres become cell edges: the first centre is -87500 at 25 km.
    assert grid.geotransform == pytest.approx((-100000.0, 25000.0, 0.0, 100000.0, 0.0, -25000.0))


def test_missing_grid_mapping_variable_still_falls_back_to_geolocation_arrays(source_path):
    """The SnowPilot non-conformance must keep taking the geolocation path."""

    ds = open_source(str(source_path))
    try:
        grid = resolve_georeferencing(ds, "snow_probability")
    finally:
        ds.close()

    assert grid.kind == "geoloc"
    assert (grid.lon_var, grid.lat_var) == ("lon", "lat")


def test_kilometre_axes_are_converted_to_metres(tmp_path):
    path = make_projected_source(tmp_path, axis_units="km", step=25.0, filename="km.nc")
    ds = open_source(str(path))
    try:
        grid = resolve_georeferencing(ds, "sea_ice_age")
    finally:
        ds.close()

    assert grid.kind == "crs"
    assert grid.geotransform == pytest.approx((-100000.0, 25000.0, 0.0, 100000.0, 0.0, -25000.0))


def test_irregular_axes_are_rejected_for_the_crs_path(tmp_path):
    """A geotransform can only describe an evenly spaced grid."""

    path = make_projected_source(tmp_path, irregular=True, filename="irregular.nc")
    ds = open_source(str(path))
    try:
        # No lat/lon arrays either, so there is nothing left to fall back to.
        with pytest.raises(SourceError) as excinfo:
            resolve_georeferencing(ds, "sea_ice_age")
    finally:
        ds.close()
    assert "lon/lat" in str(excinfo.value)


def test_one_dimensional_lonlat_axes_resolve_to_epsg_4326(tmp_path):
    times = np.array(["2021-01-01T00:00:00"], dtype="datetime64[ns]")
    lon = np.linspace(-10.0, -3.0, 8)
    lat = np.linspace(65.0, 58.0, 8)
    ds = xr.Dataset(
        {"sst": (("time", "lat", "lon"), np.zeros((1, 8, 8), dtype="float32"), {"units": "K"})},
        coords={
            "time": ("time", times, {"axis": "T", "standard_name": "time"}),
            "lon": ("lon", lon, {"standard_name": "longitude", "units": "degrees_east"}),
            "lat": ("lat", lat, {"standard_name": "latitude", "units": "degrees_north"}),
        },
    )
    path = tmp_path / "regular.nc"
    ds.to_netcdf(path)
    ds.close()

    opened = open_source(str(path))
    try:
        grid = resolve_georeferencing(opened, "sst")
    finally:
        opened.close()

    assert grid.kind == "crs"
    assert grid.epsg == 4326
    assert grid.source == "lonlat_axes"


def test_bounds_variables_are_not_mistaken_for_coordinates(tmp_path):
    """lon_bnds carries degrees_east and is 2D, so a naive match would take it."""

    times = np.array(["2021-01-01T00:00:00"], dtype="datetime64[ns]")
    rows, cols = 6, 5
    lon2d, lat2d = np.meshgrid(np.linspace(5.0, 9.0, cols), np.linspace(70.0, 66.0, rows))
    ds = xr.Dataset(
        {
            "field": (("time", "yc", "xc"), np.zeros((1, rows, cols), dtype="float32"), {}),
            "lon_bnds": (("xc", "nv"), np.zeros((cols, 2)), {"units": "degrees_east"}),
            "lat_bnds": (("yc", "nv"), np.zeros((rows, 2)), {"units": "degrees_north"}),
        },
        coords={
            "time": ("time", times, {"axis": "T", "standard_name": "time"}),
            "lon": (("yc", "xc"), lon2d, {"standard_name": "longitude", "units": "degrees_east"}),
            "lat": (("yc", "xc"), lat2d, {"standard_name": "latitude", "units": "degrees_north"}),
        },
    )
    path = tmp_path / "bnds.nc"
    ds.to_netcdf(path)
    ds.close()

    opened = open_source(str(path))
    try:
        grid = resolve_georeferencing(opened, "field")
    finally:
        opened.close()

    assert grid.kind == "geoloc"
    assert (grid.lon_var, grid.lat_var) == ("lon", "lat")


# ---------------------------------------------------------------------------
# Decoding: flag conventions and packed valid ranges
# ---------------------------------------------------------------------------

def test_flag_meanings_without_flag_values_uses_the_valid_range(projected_source):
    ds = open_source(str(projected_source))
    try:
        classes = parse_value_classes(ds["status_flag"])
    finally:
        ds.close()

    assert [c[2] for c in classes] == ["nominal", "land", "invalid"]
    assert [(c[0], c[1]) for c in classes] == [(0.5, 1.5), (1.5, 2.5), (2.5, 3.5)]


def test_flag_meanings_are_not_guessed_when_the_range_disagrees(tmp_path):
    """A wrong legend is worse than none, so an inexact range invents nothing."""

    var = xr.DataArray(
        np.zeros((2, 2), dtype="int8"),
        attrs={"flag_meanings": "nominal land invalid", "valid_min": 1, "valid_max": 9},
    )
    assert parse_value_classes(var) == []


def test_valid_range_is_unpacked_before_use(projected_source):
    ds = open_source(str(projected_source))
    try:
        # Packed 0..1100 with scale_factor 0.001 is 0..1.1 in decoded units.
        assert valid_range_for(ds["conc_1yi"]) == pytest.approx((0.0, 1.1))
        # Already unpacked, so it must be left alone.
        assert valid_range_for(ds["sea_ice_age"]) == pytest.approx((0.0, 20.0))
    finally:
        ds.close()


# ---------------------------------------------------------------------------
# A CRS described only by CF parameters (SO-SIMBA)
# ---------------------------------------------------------------------------

def _cf_only_source(tmp_path, lat0=-90.0, filename="cf_only.nc"):
    """A source whose CRS variable carries CF parameters and nothing else.

    No proj4_string, no crs_wkt, no geospatial_bounds_crs - so the CRS has to be
    rebuilt from numbers, which is what defeats AutoIdentifyEPSG.
    """
    times = np.array(["2019-09-28T12:00:00"], dtype="datetime64[ns]")
    n, step = 8, 25000.0
    x = np.arange(n, dtype="float64") * step - (n / 2.0 - 0.5) * step
    y = (n / 2.0 - 0.5) * step - np.arange(n, dtype="float64") * step
    ds = xr.Dataset(
        {
            "snow_depth": (("time", "yc", "xc"), np.ones((1, n, n), dtype="float32"), {
                "standard_name": "surface_snow_thickness", "grid_mapping": "crs"}),
        },
        coords={
            "time": ("time", times, {"axis": "T", "standard_name": "time"}),
            "xc": ("xc", x, {"standard_name": "projection_x_coordinate", "units": "m", "axis": "X"}),
            "yc": ("yc", y, {"standard_name": "projection_y_coordinate", "units": "m", "axis": "Y"}),
            "crs": ((), np.int8(0), {
                "grid_mapping_name": "lambert_azimuthal_equal_area",
                "latitude_of_projection_origin": lat0,
                "longitude_of_projection_origin": 0.0,
                "false_easting": 0.0, "false_northing": 0.0,
                "semi_major_axis": 6378137.0, "inverse_flattening": 298.257223563}),
        },
        attrs={"title": "CF-parameters-only source"},
    )
    path = tmp_path / filename
    ds.to_netcdf(path)
    ds.close()
    return path


def test_cf_parameter_only_crs_is_identified(tmp_path):
    """EASE-Grid 2.0 spelled by numbers still resolves to its EPSG code.

    AutoIdentifyEPSG cannot match a CRS whose ellipsoid is given by parameters
    rather than a named datum, and a WKT lookup fails too; going through PROJ.4
    lets pyproj identify it.
    """

    for lat0, expected in ((-90.0, 6932), (90.0, 6931)):
        path = _cf_only_source(tmp_path, lat0=lat0, filename=f"cf_{expected}.nc")
        ds = open_source(str(path))
        try:
            grid = resolve_georeferencing(ds, "snow_depth")
        finally:
            ds.close()
        assert grid.kind == "crs"
        assert grid.epsg == expected


def test_a_wrong_epsg_guess_is_rejected():
    """to_epsg matches by confidence, so its answer is verified geometrically."""

    from osgeo import osr

    north = osr.SpatialReference()
    north.ImportFromEPSG(6931)
    assert _same_crs_numerically(north, 6931) is True
    # The southern twin places the same projected coordinates elsewhere.
    assert _same_crs_numerically(north, 6932) is False


# ---------------------------------------------------------------------------
# Timestamps
# ---------------------------------------------------------------------------

def test_a_non_standard_calendar_yields_well_formed_stamps():
    """cftime str() is space-separated; sniffing it produced a malformed stamp.

    "1950-01-16 12:00:00" gained a "T00:00:00" suffix and became
    "1950-01-16 12:00:00T00:00:00Z", which no TIME dimension accepts.
    """

    cftime = pytest.importorskip("cftime")

    stamp = _iso(cftime.DatetimeNoLeap(1950, 1, 16, 12, 0, 0))
    assert stamp == "1950-01-16T12:00:00Z"
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z", stamp)


def test_iso_handles_the_other_timestamp_flavours():
    assert _iso(np.datetime64("2020-03-01T12:00:00.000000000")) == "2020-03-01T12:00:00Z"
    assert _iso(np.datetime64("2020-02-23")) == "2020-02-23T00:00:00Z"
    assert _iso(dt.datetime(2016, 5, 13, 12, 0, 0)) == "2016-05-13T12:00:00Z"


# ---------------------------------------------------------------------------
# Coordinate roles: standard_name is authoritative, the variable name is not
# ---------------------------------------------------------------------------

def _swapped_names_source(tmp_path):
    """A source whose axes are named lat/lon the wrong way round.

    The NORCE NDVI product does exactly this: the array named "lat" holds
    longitudes and declares standard_name="longitude", and vice versa. Both are
    internally consistent - only the names mislead.
    """

    times = np.array(["2000-05-10T00:00:00"], dtype="datetime64[ns]")
    lons = np.linspace(9.1, 33.8, 12)     # named "lat"
    lats = np.linspace(80.8, 76.0, 6)     # named "lon"
    ds = xr.Dataset(
        {"NDVI": (("time", "lon", "lat"), np.ones((1, 6, 12), dtype="float32"),
                  {"standard_name": "normalized_difference_vegetation_index", "units": "1"})},
        coords={
            "time": ("time", times, {"axis": "T", "standard_name": "time"}),
            "lat": ("lat", lons, {"standard_name": "longitude", "units": "degrees_east"}),
            "lon": ("lon", lats, {"standard_name": "latitude", "units": "degrees_north"}),
        },
    )
    path = tmp_path / "swapped.nc"
    ds.to_netcdf(path)
    ds.close()
    return path


def test_standard_name_beats_a_misleading_variable_name(tmp_path):
    ds = open_source(str(_swapped_names_source(tmp_path)))
    try:
        # The array named "lat" is the longitude axis, because it says so.
        assert _lonlat_axes_1d(ds, "NDVI") == ("lat", "lon")
        grid = resolve_georeferencing(ds, "NDVI")
    finally:
        ds.close()

    assert grid.kind == "crs"
    assert grid.epsg == 4326
    assert grid.geotransform[0] == pytest.approx(9.1, abs=1.5)      # x origin is a longitude
    assert grid.geotransform[3] == pytest.approx(80.8, abs=1.5)     # y origin is a latitude
    assert grid.geotransform[5] < 0                                  # north-up


def test_one_array_can_never_fill_both_roles():
    """Ranked signals: a name must not re-match what standard_name settled."""

    lat_named_lon = xr.DataArray(
        np.zeros(3), attrs={"standard_name": "longitude", "units": "degrees_east"}
    )
    assert _axis_role("lat", lat_named_lon) == "lon"

    # Units decide when standard_name is absent.
    by_units = xr.DataArray(np.zeros(3), attrs={"units": "degrees_north"})
    assert _axis_role("x", by_units) == "lat"

    # The name decides only when neither is present.
    bare = xr.DataArray(np.zeros(3), attrs={})
    assert _axis_role("lon", bare) == "lon"
    assert _axis_role("something", bare) == ""

    # A coordinate declaring something else entirely is not guessed past.
    projected = xr.DataArray(np.zeros(3), attrs={"standard_name": "projection_x_coordinate"})
    assert _axis_role("lon", projected) == ""
