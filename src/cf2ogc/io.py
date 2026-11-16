# cf2ogc — CF-NetCDF to rasters an OGC service can serve
# Copyright (C) 2026 Norwegian Meteorological Institute (MET Norway)
# Author: Massimo Di Stefano <massimods@met.no>
# SPDX-License-Identifier: GPL-3.0-or-later

"""NetCDF access shared by the converters."""

from __future__ import annotations

import os
from typing import Any


def decode_coordinate_times(ds: Any) -> Any:
    """Decode CF times on the coordinates of a dataset opened with ``decode_times=False``.

    Data variables keep their numbers: a variable in time units that is the
    field itself (a drift product's start and end times, a duration) has to
    stay numeric to be written as a raster band, while the time axis still
    needs dates to be matched against requests.
    """

    import xarray as xr

    coords = ds.coords.to_dataset()
    if not coords.variables:
        return ds
    decoded = xr.decode_cf(
        coords,
        mask_and_scale=False,
        decode_times=True,
        decode_timedelta=True,
        decode_coords=False,
    )
    return ds.assign_coords({name: decoded[name] for name in decoded.coords})


def open_netcdf_dataset(path: str, *, chunks: dict[str, int] | None = None, decode_coords: str = "all") -> Any:
    """Open a NetCDF dataset with conservative settings for Lustre/HDF5-backed files.

    CF times are decoded on the coordinates only (``decode_coordinate_times``).

    HDF5 file locking is switched off unless the environment already sets it:
    on Lustre and NFS, locking fails or stalls on read-only mounts.
    """

    import xarray as xr

    os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

    open_kwargs: dict[str, Any] = {
        "engine": "netcdf4",
        "cache": False,
        "decode_coords": decode_coords,
        "decode_times": False,
        "decode_timedelta": False,
    }
    if chunks is not None:
        open_kwargs["chunks"] = chunks

    return decode_coordinate_times(xr.open_dataset(path, **open_kwargs))
