# cf2ogc — CF-NetCDF to rasters an OGC service can serve
# Copyright (C) 2026 Norwegian Meteorological Institute (MET Norway)
# Author: Massimo Di Stefano <massimods@met.no>
# SPDX-License-Identifier: GPL-3.0-or-later

"""Synthetic CF sources shared by the grid tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

xr = pytest.importorskip("xarray")


def make_curvilinear_source(tmp_path: Path, filename: str = "agg.nc") -> Path:
    """A small curvilinear source shaped like the SIOS SnowPilot product.

    2D lat/lon geolocation arrays, a time axis, and a grid_mapping attribute
    naming a variable that is never written.
    """

    times = np.array(
        ["2020-03-01T12:00:00", "2020-03-02T12:00:00", "2020-03-04T12:00:00"],
        dtype="datetime64[ns]",
    )
    rows, cols = 12, 10
    lon2d, lat2d = np.meshgrid(
        np.linspace(10.0, 20.0, cols),
        np.linspace(80.0, 76.0, rows),
    )

    continuous = np.random.default_rng(0).random((len(times), rows, cols)).astype("float32")
    classed = np.tile(
        np.array([-1, 0, 1, 3], dtype="float32").repeat(rows * cols // 4)[: rows * cols],
        (len(times), 1),
    ).reshape(len(times), rows, cols)

    ds = xr.Dataset(
        {
            "snow_probability": (("time", "yc", "xc"), continuous, {
                "long_name": "Aggregated snow probability",
                "units": "1",
                "_FillValue": -990.0,
                # References a grid_mapping variable that is never written -
                # exactly the non-conformance seen in the real product.
                "grid_mapping": "lambert_conformal_conic",
            }),
            "classed_product": (("time", "yc", "xc"), classed, {
                "long_name": "-1: ocean, 0: snow free, 1: snow, 3: clouded, 4: no data",
                "_FillValue": -99.0,
            }),
            "land_mask": (("yc", "xc"), np.ones((rows, cols), dtype="float32")),
        },
        coords={
            "time": ("time", times, {"axis": "T", "standard_name": "time"}),
            "lon": (("yc", "xc"), lon2d, {"standard_name": "longitude", "units": "degrees_east"}),
            "lat": (("yc", "xc"), lat2d, {"standard_name": "latitude", "units": "degrees_north"}),
        },
        attrs={"title": "Test snow product", "summary": "Synthetic fixture"},
    )
    path = tmp_path / filename
    ds.to_netcdf(path)
    ds.close()
    return path


def make_projected_source(
    tmp_path: Path,
    *,
    axis_units: str = "m",
    step: float = 25000.0,
    filename: str = "projected.nc",
    irregular: bool = False,
    with_crs_var: bool = True,
) -> Path:
    """A source shaped like the NERSC Arctic sea ice age record.

    A written grid_mapping variable, regular 1D projection axes, and no lat/lon
    arrays at all.
    """

    times = np.array(["2013-05-17T12:00:00", "2013-05-18T12:00:00"], dtype="datetime64[ns]")
    n = 8
    x = np.arange(n, dtype="float64") * step - (n / 2.0 - 0.5) * step
    y = (n / 2.0 - 0.5) * step - np.arange(n, dtype="float64") * step
    if irregular:
        y = y.copy()
        y[3] += step / 3.0

    age = np.random.default_rng(1).random((len(times), n, n)).astype("float32") * 5.0
    flags = np.tile(np.array([1, 2, 3, 1], dtype="int8").repeat(n * n // 4)[: n * n], (len(times), 1))
    flags = flags.reshape(len(times), n, n)
    # Stored as int16 with scale_factor 0.001, so valid_max is in packed units
    # while the decoded values are the fractions below.
    packed = np.full((len(times), n, n), 0.5, dtype="float32")

    variables = {
        "sea_ice_age": (("time", "y", "x"), age, {
            "long_name": "Weighted Average of Sea Ice Age",
            "standard_name": "age_of_sea_ice",
            "units": "years",
            "valid_min": 0.0,
            "valid_max": 20.0,
            "grid_mapping": "Lambert_Azimuthal_Equal_Area",
        }),
        "status_flag": (("time", "y", "x"), flags, {
            "long_name": "status flag array for sea ice age",
            # flag_meanings without flag_values - not strictly conformant, but
            # the range accounts for every meaning.
            "flag_meanings": "nominal land invalid",
            "valid_min": 1,
            "valid_max": 3,
            "grid_mapping": "Lambert_Azimuthal_Equal_Area",
        }),
        "conc_1yi": (("time", "y", "x"), packed, {
            "long_name": "Concentration of First Year Sea Ice",
            "units": "1",
            "valid_min": 0,
            "valid_max": 1100,
            "grid_mapping": "Lambert_Azimuthal_Equal_Area",
        }),
    }
    coords = {
        "time": ("time", times, {"axis": "T", "standard_name": "time"}),
        "x": ("x", x, {"standard_name": "projection_x_coordinate", "units": axis_units, "axis": "X"}),
        "y": ("y", y, {"standard_name": "projection_y_coordinate", "units": axis_units, "axis": "Y"}),
    }
    if with_crs_var:
        coords["Lambert_Azimuthal_Equal_Area"] = ((), np.int8(0), {
            "grid_mapping_name": "lambert_azimuthal_equal_area",
            "longitude_of_projection_origin": 0.0,
            "latitude_of_projection_origin": 90.0,
            "false_easting": 0.0,
            "false_northing": 0.0,
            "semi_major_axis": 6378137.0,
            "inverse_flattening": 298.257223563,
            "proj4_string": "+proj=laea +lat_0=90 +lon_0=0 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs",
        })

    ds = xr.Dataset(variables, coords=coords, attrs={
        "title": "Test sea ice age",
        "geospatial_bounds_crs": "EPSG:6931",
    })
    ds["conc_1yi"].encoding.update(
        {"scale_factor": 0.001, "add_offset": 0.0, "dtype": "int16", "_FillValue": -32768}
    )
    path = tmp_path / filename
    ds.to_netcdf(path)
    ds.close()
    return path


@pytest.fixture
def source_path(tmp_path) -> Path:
    return make_curvilinear_source(tmp_path)


@pytest.fixture
def projected_source(tmp_path) -> Path:
    return make_projected_source(tmp_path)
