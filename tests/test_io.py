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
