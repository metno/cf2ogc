# cf2ogc — CF-NetCDF to rasters an OGC service can serve
# Copyright (C) 2026 Norwegian Meteorological Institute (MET Norway)
# Author: Massimo Di Stefano <massimods@met.no>
# SPDX-License-Identifier: GPL-3.0-or-later

import os

import numpy as np
import xarray as xr

from cf2ogc.io import open_netcdf_dataset


def test_open_netcdf_dataset_chunks_and_disables_hdf5_locking(tmp_path, monkeypatch):
    path = tmp_path / "p.nc"
    xr.Dataset({"X": ("point", np.arange(10, dtype=np.float64))}).to_netcdf(path)
    monkeypatch.delenv("HDF5_USE_FILE_LOCKING", raising=False)

    ds = open_netcdf_dataset(str(path), chunks={"point": 4})
    try:
        assert ds.chunks["point"] == (4, 4, 2)
        assert os.environ["HDF5_USE_FILE_LOCKING"] == "FALSE"
    finally:
        ds.close()


def test_open_netcdf_dataset_keeps_an_explicit_locking_setting(tmp_path, monkeypatch):
    path = tmp_path / "p.nc"
    xr.Dataset({"X": ("point", np.arange(3, dtype=np.float64))}).to_netcdf(path)
    monkeypatch.setenv("HDF5_USE_FILE_LOCKING", "TRUE")
    open_netcdf_dataset(str(path)).close()
    assert os.environ["HDF5_USE_FILE_LOCKING"] == "TRUE"


def _write_drift_like(path):
    # A drift product: per-cell start and end times as data variables beside
    # the time coordinate, all in CF time units.
    seconds = np.array([[0.0, 3600.0], [7200.0, 10800.0]])
    xr.Dataset(
        {
            "t0": (("time", "x"), seconds, {"units": "seconds since 2020-01-01 00:00:00"}),
            "t1": (("time", "x"), seconds + 60.0, {"units": "seconds since 2020-01-01 00:00:00"}),
            "dt": (("time", "x"), np.ones((2, 2)), {"units": "days"}),
        },
        coords={
            "time": ("time", np.array([0.0, 1.0]), {"units": "days since 2020-01-01", "calendar": "standard"}),
            "x": ("x", np.array([0.0, 1.0])),
        },
    ).to_netcdf(path)


def test_open_netcdf_dataset_leaves_time_unit_data_variables_undecoded(tmp_path):
    path = tmp_path / "drift.nc"
    _write_drift_like(path)
    with open_netcdf_dataset(str(path)) as ds:
        assert ds["time"].dtype.kind == "M"
        assert ds["time"].values[1] == np.datetime64("2020-01-02")
        for name in ("t0", "t1", "dt"):
            assert ds[name].dtype.kind == "f", name
        assert ds["t0"].values[1, 1] == 10800.0
        assert ds["t0"].attrs["units"] == "seconds since 2020-01-01 00:00:00"


def test_grid_open_source_leaves_time_unit_data_variables_undecoded(tmp_path):
    from cf2ogc.grid import open_source

    path = tmp_path / "drift.nc"
    _write_drift_like(path)
    ds = open_source(str(path))
    try:
        assert ds["time"].dtype.kind == "M"
        assert ds["t1"].dtype.kind == "f"
        assert ds["t1"].values[0, 0] == 60.0
    finally:
        ds.close()
