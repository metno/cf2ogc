# cf2ogc — CF-NetCDF to rasters an OGC service can serve
# Copyright (C) 2026 Norwegian Meteorological Institute (MET Norway)
# Author: Massimo Di Stefano <massimods@met.no>
# SPDX-License-Identifier: GPL-3.0-or-later

"""A regular projected grid recovered from 2-D lat/lon and its grid_mapping."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

xr = pytest.importorskip("xarray")
pytest.importorskip("osgeo")
pytest.importorskip("pyproj")

from osgeo import gdal  # noqa: E402
from pyproj import Transformer  # noqa: E402

from cf2ogc.grid import (  # noqa: E402
    _PROJECTED_LONLAT_TOLERANCE,
    _srs_from_nonstandard_mapping,
    convert,
    open_source,
    resolve_georeferencing,
)

gdal.UseExceptions()

STEP = 500.0
X0, Y0 = 398077.5, 8461801.5  # the first cell centre, south-west, as the snow product has it
ROWS, COLS = 9, 7

# The grid_mapping container exactly as the NORCE MODIS snow cover product
# for Svalbard writes it: not a CF grid_mapping_name, numbers as strings.
SNOW_MAPPING = {
    "grid_mapping_name": "UTM Zone 33N",
    "utm_zone_number": "33N",
    "semi_major_axis": "6378137",
    "inverse_flattening": "298.257223563",
    "_CoordinateTransformType": "Projection",
    "_CoordinateAxisTypes": "GeoX GeoY",
}


def _lonlat(epsg: int = 32633, *, x0: float = X0, y0: float = Y0, rotate: float = 0.0, nudge: float = 0.0):
    """float32 lat/lon of a regular grid whose rows run north, as the snow product's do.

    ``rotate`` turns the grid (degrees) in the projected plane; ``nudge`` moves
    one interior cell centre east by that many metres.
    """

    rows, cols = np.meshgrid(np.arange(ROWS, dtype="float64"), np.arange(COLS, dtype="float64"), indexing="ij")
    angle = np.radians(rotate)
    x = x0 + STEP * (cols * np.cos(angle) - rows * np.sin(angle))
    y = y0 + STEP * (cols * np.sin(angle) + rows * np.cos(angle))
    x[ROWS // 2, COLS // 2] += nudge
    lon, lat = Transformer.from_crs(epsg, 4326, always_xy=True).transform(x, y)
    return lon.astype("float32"), lat.astype("float32")


def _source(
    tmp_path: Path,
    *,
    mapping: dict | None = None,
    mapping_name: str = "UTMZ33N",
    filename: str = "snow.nc",
    **lonlat_options,
) -> Path:
    """A source shaped like the snow product: 2-D lat/lon, no x/y, a non-CF grid_mapping."""

    lon, lat = _lonlat(**lonlat_options)
    times = np.array([48.0, 49.0], dtype="float32")
    snow = (np.arange(2 * ROWS * COLS, dtype="float32") % 101).reshape(2, ROWS, COLS)
    snow[0, 0, 0] = np.nan
    ds = xr.Dataset(
        {
            "snowfr": (("time", "az_size", "ra_size"), snow, {
                "long_name": "Snow cover fraction",
                "standard_name": "surface_snow_area_fraction",
                "units": "%",
                "grid_mapping": mapping_name,
            }),
            "lat": (("az_size", "ra_size"), lat, {"standard_name": "latitude", "units": "degrees_north"}),
            "lon": (("az_size", "ra_size"), lon, {"standard_name": "longitude", "units": "degrees_east"}),
            mapping_name: ((), np.float32(0), dict(SNOW_MAPPING if mapping is None else mapping)),
        },
        coords={"time": ("time", times, {"standard_name": "time", "units": "days since 2023-1-1 12:00:00"})},
        attrs={"title": "Synthetic snow cover fraction"},
    )
    path = tmp_path / filename
    ds.to_netcdf(path)
    ds.close()
    return path


def _resolve(path: Path, variable: str = "snowfr"):
    ds = open_source(str(path))
    try:
        return resolve_georeferencing(ds, variable)
    finally:
        ds.close()


def test_a_non_cf_utm_mapping_with_regular_lonlat_resolves_to_its_projected_grid(tmp_path):
    grid = _resolve(_source(tmp_path))

    assert grid.kind == "crs"
    assert grid.source == "projected_lonlat:UTMZ33N"
    assert grid.epsg == 32633
    # Cell centres become cell edges; rows run north, so the y step is positive.
    assert grid.geotransform == pytest.approx(
        (X0 - STEP / 2, STEP, 0.0, Y0 - STEP / 2, 0.0, STEP), abs=0.5
    )
    # float32 lat/lon round to under a metre here.
    assert grid.max_residual < 1.0
    assert (grid.lon_var, grid.lat_var) == ("lon", "lat")


def test_the_recovered_grid_is_written_without_a_warp(tmp_path):
    """Through the public entry point: the source's own pixels on its own grid."""

    out = tmp_path / "snow.tif"
    described = convert(_source(tmp_path), "snowfr", out, time_index=1)

    assert described["reprojected"] is False
    assert described["georeferencing"] == "projected_lonlat:UTMZ33N"
    assert described["epsg"] == 32633
    assert (described["width"], described["height"]) == (COLS, ROWS)
    raster = gdal.Open(str(out))
    try:
        written = raster.GetRasterBand(1).ReadAsArray()
        assert raster.GetGeoTransform() == pytest.approx(tuple(described["geotransform"]))
    finally:
        raster = None
    expected = (np.arange(2 * ROWS * COLS, dtype="float32") % 101).reshape(2, ROWS, COLS)[1]
    np.testing.assert_array_equal(written, expected)


def test_a_cf_mapping_with_regular_lonlat_and_no_axes_takes_the_same_path(tmp_path):
    cf_utm = {
        "grid_mapping_name": "transverse_mercator",
        "longitude_of_central_meridian": 15.0,
        "latitude_of_projection_origin": 0.0,
        "scale_factor_at_central_meridian": 0.9996,
        "false_easting": 500000.0,
        "false_northing": 0.0,
        "semi_major_axis": 6378137.0,
        "inverse_flattening": 298.257223563,
    }
    grid = _resolve(_source(tmp_path, mapping=cf_utm, mapping_name="crs"))

    assert grid.kind == "crs"
    assert grid.source == "projected_lonlat:crs"
    assert grid.epsg == 32633
    assert grid.geotransform[1] == pytest.approx(STEP, abs=0.01)


@pytest.mark.parametrize(
    "lonlat_options",
    [
        pytest.param({"rotate": 10.0}, id="rotated"),
        pytest.param({"epsg": 32631, "x0": 600000.0}, id="regular-in-another-zone"),
        pytest.param({"nudge": 2.0 * _PROJECTED_LONLAT_TOLERANCE * STEP}, id="one-cell-off-twice-the-tolerance"),
    ],
)
def test_lonlat_that_are_not_a_regular_grid_in_the_mapping_keep_the_geolocation_path(tmp_path, lonlat_options):
    grid = _resolve(_source(tmp_path, **lonlat_options))

    assert grid.kind == "geoloc"
    assert grid.source == "geolocation_arrays"
    assert grid.max_residual is None


def test_a_cell_off_by_less_than_the_tolerance_is_still_regular(tmp_path):
    grid = _resolve(_source(tmp_path, nudge=0.5 * _PROJECTED_LONLAT_TOLERANCE * STEP))

    assert grid.kind == "crs"
    assert grid.max_residual <= _PROJECTED_LONLAT_TOLERANCE * STEP


def test_an_unrecognised_mapping_keeps_the_geolocation_path(tmp_path):
    grid = _resolve(_source(tmp_path, mapping={"grid_mapping_name": "something else"}, mapping_name="crs"))

    assert grid.kind == "geoloc"


class _Container:
    def __init__(self, **attrs):
        self.attrs = attrs


def _epsg(srs) -> int | None:
    srs.AutoIdentifyEPSG()
    code = srs.GetAuthorityCode(None)
    return int(code) if code else None


@pytest.mark.parametrize(
    "attrs, name, latitudes, expected",
    [
        pytest.param(SNOW_MAPPING, "UTMZ33N", None, 32633, id="snow-product"),
        pytest.param({"utm_zone_number": "33N"}, "crs", None, 32633, id="zone-attribute"),
        pytest.param({"grid_mapping_name": "UTM Zone 33N"}, "crs", None, 32633, id="grid-mapping-name"),
        pytest.param({}, "UTMZ33N", None, 32633, id="container-name"),
        pytest.param({}, "utm_33s", None, 32733, id="southern-letter"),
        pytest.param({"utm_zone": "33"}, "crs", [78.0, 79.0], 32633, id="hemisphere-from-latitudes-north"),
        pytest.param({"utm_zone": "19"}, "crs", [-40.0, -41.0], 32719, id="hemisphere-from-latitudes-south"),
    ],
)
def test_utm_zones_written_outside_cf_are_recognised(attrs, name, latitudes, expected):
    srs = _srs_from_nonstandard_mapping(_Container(**attrs), name, latitudes)

    assert srs is not None
    assert _epsg(srs) == expected


@pytest.mark.parametrize(
    "attrs, name",
    [
        pytest.param({"utm_zone": "33"}, "crs", id="no-hemisphere-and-no-latitudes"),
        pytest.param({"utm_zone_number": "61N"}, "crs", id="no-such-zone"),
        pytest.param({"grid_mapping_name": "polar_stereographic"}, "crs", id="not-utm"),
    ],
)
def test_what_is_not_a_utm_zone_is_not_guessed(attrs, name):
    assert _srs_from_nonstandard_mapping(_Container(**attrs), name) is None


def test_a_utm_zone_on_another_ellipsoid_keeps_that_ellipsoid():
    srs = _srs_from_nonstandard_mapping(
        _Container(utm_zone_number="33N", semi_major_axis="6378388", inverse_flattening="297"), "crs"
    )

    assert srs is not None
    assert srs.GetSemiMajor() == pytest.approx(6378388.0)
    assert srs.GetInvFlattening() == pytest.approx(297.0)
    assert "+zone=33" in srs.ExportToProj4()
