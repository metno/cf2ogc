# cf2ogc — CF-NetCDF to rasters an OGC service can serve
# Copyright (C) 2026 Norwegian Meteorological Institute (MET Norway)
# Author: Massimo Di Stefano <massimods@met.no>
# SPDX-License-Identifier: GPL-3.0-or-later

"""NetCDF access shared by the converters."""

from __future__ import annotations

import os
from typing import Any


def open_netcdf_dataset(path: str, *, chunks: dict[str, int] | None = None, decode_coords: str = "all") -> Any:
    """Open a NetCDF dataset with conservative settings for Lustre/HDF5-backed files.

    HDF5 file locking is switched off unless the environment already sets it:
    on Lustre and NFS, locking fails or stalls on read-only mounts.
    """

    import xarray as xr

    os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

    open_kwargs: dict[str, Any] = {
        "engine": "netcdf4",
        "cache": False,
        "decode_coords": decode_coords,
    }
    if chunks is not None:
        open_kwargs["chunks"] = chunks

    return xr.open_dataset(path, **open_kwargs)
