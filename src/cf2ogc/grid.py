# cf2ogc — CF-NetCDF to rasters an OGC service can serve
# Copyright (C) 2026 Norwegian Meteorological Institute (MET Norway)
# Author: Massimo Di Stefano <massimods@met.no>
# SPDX-License-Identifier: GPL-3.0-or-later

"""Gridded CF fields to rasters: one variable, one time step, one vertical level.

A CF variable shaped (y, x), (time, y, x), (vertical, y, x) or
(time, vertical, y, x) is written one 2-D slice at a time: ``convert`` (or
``GridField.convert`` for many slices of one field) selects a time step and a
vertical level, by value or by index, writes a Cloud Optimized GeoTIFF and
returns a description of it (bounds, CRS, the band with its variable, units,
nodata and value classes, the time and the level). Nothing is averaged or
collapsed: an axis with more than one value must be selected explicitly.

Axes are found by their CF attributes (``axis``, ``standard_name``,
``positive``, ``units``) before their names. Georeferencing is resolved in
three tiers, best first, by ``resolve_georeferencing``:

1. A CF ``grid_mapping`` container variable that actually exists, plus regular
   1D projection axes - the grid is then described by a six-number geotransform
   and no resampling is needed at all.
2. Regular 1D lat/lon axes - the same, in EPSG:4326.
3. 2D lat/lon coordinate arrays (a curvilinear grid), used as a GDAL
   geolocation array and warped per pixel.

Tier 3 exists because real products routinely reference a grid_mapping variable
they never actually write - the SIOS SnowPilot SLSTR aggregation is one, and its
``xc``/``yc`` axes are in km despite claiming metres, with ``yc`` negated
relative to true projected y. For such a source the lat/lon arrays are the
authoritative georeferencing. For a well-formed source such as the NERSC Arctic
sea ice age record, tier 1 avoids the warp entirely and preserves the published
grid exactly.

Values are read through xarray's ``mask_and_scale``, so packed variables come
out in physical units; ``valid_range_for`` unpacks a declared valid range to
match, and ``unscaled_nodata`` does the same for a GDAL band's nodata.

GDAL's Python bindings (``osgeo``) and pyproj are needed by this module; see
the ``grid`` extra.
"""
from __future__ import annotations

import os
import re
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Shared archives are commonly NFS-mounted with "nolock", and HDF5 takes a POSIX
# lock when opening a file - which fails there with an opaque
# "NetCDF: HDF error" rather than a permissions message. Locking only guards
# concurrent writers and every source we open is read-only, so disable it.
# Set before netCDF4/h5py load the HDF5 library; an explicit environment value
# still wins.
os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")


class GridError(Exception):
    """Base class of every error ``cf2ogc.grid`` raises on purpose."""


class SourceError(GridError, ValueError):
    """The source cannot be converted as asked.

    No time axis, no usable georeferencing, an unknown variable, a time or
    level that is not on the axis, an axis that was not selected.
    """


class ReadError(GridError, OSError):
    """Reading the source failed: it could not be opened, or a read kept failing."""


class ConversionError(GridError, RuntimeError):
    """GDAL could not produce the output grid."""


def open_source(path: str) -> Any:
    """Open a NetCDF file or OPeNDAP URL the way the converters read it.

    ``decode_coords="all"`` keeps grid_mapping and bounds variables reachable,
    and ``mask_and_scale=True`` hands back physical values with fill as NaN.
    """

    import xarray as xr

    try:
        return xr.open_dataset(path, decode_coords="all", mask_and_scale=True)
    except Exception as exc:
        raise ReadError(f"Unable to open NetCDF source: {exc}") from exc


_LON_NAMES = ("lon", "longitude", "nav_lon", "XLONG")
_LAT_NAMES = ("lat", "latitude", "nav_lat", "XLAT")

# Used when a variable declares no _FillValue of its own.
_DEFAULT_FILL = -9999.0

# A source georeferenced only by 2D lat/lon arrays has no grid to preserve, so
# one has to be chosen. Web Mercator stays the default for those, as before.
_GEOLOC_FALLBACK_EPSG = 3857

# Colours for categorical classes, matched on the class label. Keeps
# auto-generated legends physically sensible instead of an arbitrary ramp.
_LABEL_COLORS: tuple[tuple[tuple[str, ...], tuple[int, int, int]], ...] = (
    (("ocean", "sea", "water"), (15, 45, 95)),
    (("snow free", "snow-free", "bare", "land"), (140, 105, 60)),
    (("snow", "ice"), (255, 255, 255)),
    (("cloud", "clouded"), (150, 150, 155)),
    (("no data", "nodata", "missing", "fill", "invalid", "unknown"), (60, 60, 60)),
)
_FALLBACK_PALETTE = (
    (31, 119, 180), (255, 127, 14), (44, 160, 44), (214, 39, 40),
    (148, 103, 189), (140, 86, 75), (227, 119, 194), (127, 127, 127),
)


def _find_time_dim(ds: Any) -> str:
    """Return the name of the unlimited/time dimension."""

    for name, coord in ds.coords.items():
        if str(coord.attrs.get("axis", "")).upper() == "T":
            return str(name)
        if str(coord.attrs.get("standard_name", "")).lower() == "time":
            return str(name)
    for candidate in ("time", "t"):
        if candidate in ds.dims:
            return candidate
    raise SourceError("Source has no recognisable time axis")


@dataclass
class GridRef:
    """How one source's pixels are placed on the Earth.

    ``kind`` is ``"crs"`` when the grid is described by a geotransform plus a
    spatial reference - the cheap, lossless case - and ``"geoloc"`` when it can
    only be described by 2D lat/lon arrays and must be warped per pixel.
    """

    kind: str
    srs_wkt: str = ""
    epsg: int | None = None
    geotransform: tuple[float, float, float, float, float, float] | None = None
    shape: tuple[int, int] = (0, 0)
    x_var: str = ""
    y_var: str = ""
    lon_var: str = ""
    lat_var: str = ""
    source: str = ""

    @property
    def is_geoloc(self) -> bool:
        return self.kind == "geoloc"


# Axis units we know how to convert to metres. A source that declares anything
# else is left to a lower tier rather than guessed at.
_AXIS_UNIT_SCALE: dict[str, float] = {
    "m": 1.0, "meter": 1.0, "meters": 1.0, "metre": 1.0, "metres": 1.0,
    "km": 1000.0, "kilometer": 1000.0, "kilometers": 1000.0,
    "kilometre": 1000.0, "kilometres": 1000.0,
}

_X_AXIS_NAMES = ("x", "xc", "xdim", "x_coord", "projection_x_coordinate")
_Y_AXIS_NAMES = ("y", "yc", "ydim", "y_coord", "projection_y_coordinate")


def _is_bounds_like(ds: Any, name: str) -> bool:
    """True when a variable is a cell-bounds array rather than a coordinate.

    Bounds arrays carry the same ``units`` as the coordinate they bound and are
    2D, so a naive units match would happily georeference from ``lon_bnds``.
    """

    lowered = name.lower()
    if lowered.endswith(("_bnds", "_bounds", "_vertices", "_corners")):
        return True
    for _, var in list(ds.coords.items()) + list(ds.data_vars.items()):
        if str(var.attrs.get("bounds", "") or "") == name:
            return True
    return False


def _regular_step(values: Any) -> float | None:
    """Return the constant step of a 1D axis, or None when it is irregular."""

    import numpy as np

    array = np.asarray(values, dtype="float64")
    if array.ndim != 1 or array.size < 2:
        return None
    diffs = np.diff(array)
    step = float(diffs[0])
    if step == 0.0:
        return None
    # A geotransform can only describe an evenly spaced grid. Allow a little
    # slack for float32 axes written by the producer.
    if not np.allclose(diffs, step, rtol=1e-5, atol=abs(step) * 1e-4):
        return None
    return step


def _spatial_dims(ds: Any, variable: str) -> tuple[str, str] | None:
    """Return the (y_dim, x_dim) of a variable's trailing two dimensions."""

    if not variable or variable not in ds.variables:
        return None
    dims = [str(d) for d in ds[variable].dims]
    if len(dims) < 2:
        return None
    return dims[-2], dims[-1]


def _geotransform_from_axes(x_values: Any, y_values: Any, scale: float) -> tuple | None:
    """Build a GDAL geotransform from two regular 1D axes of cell centres."""

    xstep = _regular_step(x_values)
    ystep = _regular_step(y_values)
    if xstep is None or ystep is None:
        return None
    xres = xstep * scale
    yres = ystep * scale
    x0 = float(x_values[0]) * scale - xres / 2.0
    y0 = float(y_values[0]) * scale - yres / 2.0
    return (x0, xres, 0.0, y0, 0.0, yres)


def _attach_authority(srs: Any, declared: str = "") -> int | None:
    """Give an SRS an EPSG authority node where we can prove one applies.

    A CRS rebuilt from proj4 or from CF parameters carries no authority code, so
    ``AutoIdentifyEPSG`` often fails on it and the layer would end up advertising
    only EPSG:4326. When the file names its own CRS (ACDD
    ``geospatial_bounds_crs``) we verify that claim and adopt it.
    """

    from osgeo import osr

    match = re.search(r"EPSG:\s*(\d+)", declared or "", re.IGNORECASE)
    if match:
        candidate = osr.SpatialReference()
        try:
            candidate.ImportFromEPSG(int(match.group(1)))
            if candidate.IsSame(srs):
                srs.ImportFromEPSG(int(match.group(1)))
                return int(match.group(1))
        except Exception:
            pass
    probe = srs.Clone()
    try:
        if probe.AutoIdentifyEPSG() == 0:
            code = probe.GetAuthorityCode(None)
            if code:
                srs.ImportFromEPSG(int(code))
                return int(code)
    except Exception:
        pass

    # A CRS rebuilt from CF parameters describes its ellipsoid by numbers rather
    # than naming a datum, which defeats both AutoIdentifyEPSG and a WKT match -
    # the SO-SIMBA products are EASE-Grid 2.0 South spelled exactly this way.
    # Going through PROJ.4 drops the unnamed baggage and lets pyproj match it.
    try:
        from pyproj import CRS

        candidate = CRS.from_proj4(srs.ExportToProj4()).to_epsg()
    except Exception:
        candidate = None
    if candidate and _same_crs_numerically(srs, int(candidate)):
        srs.ImportFromEPSG(int(candidate))
        return int(candidate)
    return None


def _same_crs_numerically(srs: Any, epsg: int) -> bool:
    """True when an EPSG definition places points exactly where ``srs`` does.

    ``to_epsg`` matches by confidence, so its answer is verified rather than
    trusted: adopting a wrong code would make the layer advertise a CRS that
    does not match its pixels, and clients would silently misplace the data.
    """

    from osgeo import osr

    try:
        target = osr.SpatialReference()
        target.ImportFromEPSG(int(epsg))
        wgs84 = osr.SpatialReference()
        wgs84.ImportFromEPSG(4326)
        for ref in (srs, target, wgs84):
            ref.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
        from_srs = osr.CoordinateTransformation(srs, wgs84)
        from_target = osr.CoordinateTransformation(target, wgs84)
    except Exception:
        return False

    for x, y in ((0.0, 0.0), (1_000_000.0, 0.0), (0.0, 1_000_000.0), (-2_000_000.0, 1_500_000.0)):
        try:
            a_lon, a_lat, *_ = from_srs.TransformPoint(x, y)
            b_lon, b_lat, *_ = from_target.TransformPoint(x, y)
        except Exception:
            return False
        # ~1e-7 degrees is well under a centimetre.
        if abs(a_lon - b_lon) > 1e-6 or abs(a_lat - b_lat) > 1e-6:
            return False
    return True


def _srs_from_crs_variable(crs_var: Any) -> Any | None:
    """Build a spatial reference from a CF grid_mapping container variable.

    Prefers what the producer wrote verbatim (WKT, then PROJ.4) over a CRS
    rebuilt from the CF parameters, because the verbatim forms usually carry the
    datum and the reconstruction does not.
    """

    from osgeo import osr

    attrs = {k: v for k, v in crs_var.attrs.items()}
    for key in ("crs_wkt", "spatial_ref", "esri_pe_string"):
        text = str(attrs.get(key, "") or "").strip()
        if text:
            srs = osr.SpatialReference()
            if srs.ImportFromWkt(text) == 0 and srs.GetName():
                return srs
    proj4 = str(attrs.get("proj4_string", "") or attrs.get("proj4", "") or "").strip()
    if proj4:
        srs = osr.SpatialReference()
        if srs.ImportFromProj4(proj4) == 0 and srs.GetName():
            return srs
    if attrs.get("grid_mapping_name"):
        try:
            from pyproj import CRS

            srs = osr.SpatialReference()
            if srs.ImportFromWkt(CRS.from_cf(attrs).to_wkt()) == 0 and srs.GetName():
                return srs
        except Exception:
            return None
    return None


def _grid_mapping_name(ds: Any, variable: str) -> str:
    """Return the grid_mapping container referenced by a variable, if any."""

    names: list[str] = []
    candidates = [variable] if variable and variable in ds.variables else list(ds.data_vars)
    for name in candidates:
        var = ds[name]
        raw = var.attrs.get("grid_mapping") or (getattr(var, "encoding", {}) or {}).get("grid_mapping")
        if raw:
            # CF also allows "crsvar: xcoord ycoord" for multiple mappings.
            names.append(str(raw).split(":")[0].strip())
    for name in names:
        if name in ds.variables or name in ds.coords:
            return name
    return ""


def _projection_axes(ds: Any, variable: str) -> tuple[str, str, float] | None:
    """Find regular 1D projection axes for a variable, with their metre scale."""

    dims = _spatial_dims(ds, variable)
    x_name = y_name = ""
    for name, var in list(ds.coords.items()) + list(ds.data_vars.items()):
        if var.ndim != 1:
            continue
        text = str(name)
        std = str(var.attrs.get("standard_name", "")).lower()
        axis = str(var.attrs.get("axis", "")).upper()
        if not x_name and (std == "projection_x_coordinate" or axis == "X" or text.lower() in _X_AXIS_NAMES):
            x_name = text
        if not y_name and (std == "projection_y_coordinate" or axis == "Y" or text.lower() in _Y_AXIS_NAMES):
            y_name = text
    if not x_name or not y_name:
        return None
    if dims and (str(ds[y_name].dims[0]) != dims[0] or str(ds[x_name].dims[0]) != dims[1]):
        return None
    units = str(ds[x_name].attrs.get("units", "") or "").lower().strip()
    y_units = str(ds[y_name].attrs.get("units", "") or "").lower().strip()
    if units != y_units:
        return None
    # An axis with no units at all is assumed to already be in the CRS's own
    # linear unit; anything we cannot convert is left to a lower tier.
    scale = 1.0 if not units else _AXIS_UNIT_SCALE.get(units, 0.0)
    if not scale:
        return None
    return x_name, y_name, scale


def _axis_role(name: str, var: Any) -> str:
    """Say whether a coordinate is longitude or latitude, or neither.

    The signals are ranked and exclusive, so one array can never fill both roles.
    That is not hypothetical: the NORCE NDVI product names its longitude axis
    "lat" and its latitude axis "lon" while declaring the correct standard_names,
    so a matcher that also consults the name assigned both roles to the same
    array and then rejected the pair for disagreeing with the data's dimensions.

    A declared standard_name is taken as the truth. Units are consulted only when
    it is absent, and the name only when both are.
    """

    standard_name = str(var.attrs.get("standard_name", "")).strip().lower()
    if standard_name:
        if standard_name == "longitude":
            return "lon"
        if standard_name == "latitude":
            return "lat"
        # It declares itself as something else entirely; do not guess past that.
        return ""

    units = str(var.attrs.get("units", "")).strip().lower()
    if units in {"degrees_east", "degree_east"}:
        return "lon"
    if units in {"degrees_north", "degree_north"}:
        return "lat"
    if units:
        return ""

    if name in _LON_NAMES:
        return "lon"
    if name in _LAT_NAMES:
        return "lat"
    return ""


def _lonlat_axes_1d(ds: Any, variable: str) -> tuple[str, str] | None:
    """Find regular 1D lon/lat axes for a variable."""

    dims = _spatial_dims(ds, variable)
    lon_name = lat_name = ""
    for name, var in list(ds.coords.items()) + list(ds.data_vars.items()):
        if var.ndim != 1 or _is_bounds_like(ds, str(name)):
            continue
        text = str(name)
        role = _axis_role(text, var)
        if role == "lon" and not lon_name:
            lon_name = text
        elif role == "lat" and not lat_name:
            lat_name = text
    if not lon_name or not lat_name or lon_name == lat_name:
        return None
    if dims and (str(ds[lat_name].dims[0]) != dims[0] or str(ds[lon_name].dims[0]) != dims[1]):
        return None
    return lon_name, lat_name


def _find_lonlat(ds: Any, variable: str = "") -> tuple[str, str]:
    """Return (lon_name, lat_name) for the 2D geolocation arrays."""

    dims = _spatial_dims(ds, variable)
    lon_name = lat_name = ""
    for name, var in list(ds.coords.items()) + list(ds.data_vars.items()):
        text = str(name)
        if var.ndim != 2 or _is_bounds_like(ds, text):
            continue
        if dims and tuple(str(d) for d in var.dims) != dims:
            continue
        role = _axis_role(text, var)
        if role == "lon" and not lon_name:
            lon_name = text
        elif role == "lat" and not lat_name:
            lat_name = text
    if not lon_name or not lat_name or lon_name == lat_name:
        raise SourceError("Source has no lon/lat coordinate arrays to georeference from")
    return lon_name, lat_name


def resolve_georeferencing(ds: Any, variable: str = "") -> GridRef:
    """Decide how to place a variable's pixels on the Earth.

    Tries a real CRS first, then 1D lat/lon axes, then 2D geolocation arrays,
    so a well-formed source is published without resampling and a malformed one
    still works.
    """

    from osgeo import osr

    dims = _spatial_dims(ds, variable)
    shape = (0, 0)
    if dims:
        shape = (int(ds.sizes[dims[0]]), int(ds.sizes[dims[1]]))

    # Tier 1: a grid_mapping variable that exists, plus regular projection axes.
    crs_name = _grid_mapping_name(ds, variable)
    axes = _projection_axes(ds, variable)
    if crs_name and axes:
        srs = _srs_from_crs_variable(ds[crs_name])
        if srs is not None:
            x_name, y_name, scale = axes
            gt = _geotransform_from_axes(ds[x_name].values, ds[y_name].values, scale)
            if gt is not None:
                epsg = _attach_authority(srs, str(ds.attrs.get("geospatial_bounds_crs", "") or ""))
                return GridRef(
                    kind="crs",
                    srs_wkt=srs.ExportToWkt(),
                    epsg=epsg,
                    geotransform=gt,
                    shape=shape,
                    x_var=x_name,
                    y_var=y_name,
                    source=f"grid_mapping:{crs_name}",
                )

    # Tier 2: regular 1D lon/lat axes describe a plate carree grid directly.
    lonlat = _lonlat_axes_1d(ds, variable)
    if lonlat:
        lon_name, lat_name = lonlat
        gt = _geotransform_from_axes(ds[lon_name].values, ds[lat_name].values, 1.0)
        if gt is not None:
            srs = osr.SpatialReference()
            srs.ImportFromEPSG(4326)
            return GridRef(
                kind="crs",
                srs_wkt=srs.ExportToWkt(),
                epsg=4326,
                geotransform=gt,
                shape=shape,
                x_var=lon_name,
                y_var=lat_name,
                source="lonlat_axes",
            )

    # Tier 3: 2D geolocation arrays, warped per pixel.
    lon_name, lat_name = _find_lonlat(ds, variable)
    if ds[lon_name].ndim != 2 or ds[lat_name].ndim != 2:
        raise SourceError(
            "Source has no usable georeferencing: no CF grid_mapping with regular "
            "projection axes, no regular 1D lat/lon axes, and no 2D lat/lon arrays"
        )
    return GridRef(
        kind="geoloc",
        shape=shape,
        lon_var=lon_name,
        lat_var=lat_name,
        source="geolocation_arrays",
    )


def _iso(value: Any) -> str:
    """Format a timestamp as an ISO 8601 instant MapServer accepts.

    Deliberately not by sniffing ``str()``: a non-standard calendar decodes to a
    cftime object whose str() is space-separated, so appending "T00:00:00" to it
    produced "1950-01-16 12:00:00T00:00:00Z" - a malformed stamp that goes
    straight into the tileindex and the advertised time extent, breaking the TIME
    dimension without any error. cftime and datetime both expose isoformat();
    numpy datetime64 does not and falls through to str(), which is already
    ISO-shaped.
    """

    text = ""
    isoformat = getattr(value, "isoformat", None)
    if callable(isoformat):
        try:
            text = str(isoformat())
        except Exception:
            text = ""
    if not text:
        text = str(value)
    text = text.strip().replace(" ", "T")
    if "T" not in text:
        text = f"{text}T00:00:00"
    return f"{text.split('.')[0].rstrip('Z')}Z"


def _packing_for(var: Any) -> tuple[float, float]:
    """Return the (scale_factor, add_offset) applied when the data was unpacked."""

    encoded = getattr(var, "encoding", {}) or {}
    scale = encoded.get("scale_factor", var.attrs.get("scale_factor"))
    offset = encoded.get("add_offset", var.attrs.get("add_offset"))
    try:
        scale = float(scale) if scale is not None else 1.0
    except Exception:
        scale = 1.0
    try:
        offset = float(offset) if offset is not None else 0.0
    except Exception:
        offset = 0.0
    return scale, offset


def valid_range_for(var: Any) -> tuple[float, float] | None:
    """Return a variable's valid range in the same units as the decoded data.

    CF requires ``valid_min``/``valid_max``/``valid_range`` to be given in the
    *packed* units of the stored data, but we read through
    ``mask_and_scale=True`` and therefore see unpacked values - so the range has
    to be unpacked too before it can be compared against a pixel. The NERSC sea
    ice age record has both conventions in one file: ``conc_1yi`` is int16 with
    ``scale_factor = 0.001`` and ``valid_max = 1100`` (meaning 1.1), while
    ``sea_ice_age`` is float32 with ``valid_max = 20.0`` already unpacked.
    """

    low = var.attrs.get("valid_min")
    high = var.attrs.get("valid_max")
    if low is None or high is None:
        pair = var.attrs.get("valid_range")
        if pair is None:
            return None
        try:
            low, high = list(pair)[:2]
        except Exception:
            return None
    scale, offset = _packing_for(var)
    try:
        return float(low) * scale + offset, float(high) * scale + offset
    except Exception:
        return None


def parse_value_classes(var: Any) -> list[list[Any]]:
    """Derive a discrete legend for a categorical variable.

    Supports the CF ``flag_values``/``flag_meanings`` convention, the same
    convention with ``flag_values`` omitted but a ``valid_min``/``valid_max``
    range that accounts for every meaning, and the ``long_name``
    "-1: ocean, 0: snow free, 1: snow" style used by the SIOS products. Returns
    [low, high, label, [r, g, b]] entries, or [] when the variable is continuous.
    """

    pairs: list[tuple[float, str]] = []

    flag_values = var.attrs.get("flag_values")
    flag_meanings = str(var.attrs.get("flag_meanings", "") or "").split()
    if flag_values is not None and flag_meanings:
        try:
            values = [float(v) for v in list(flag_values)]
            pairs = list(zip(values, flag_meanings))
        except Exception:
            pairs = []

    if not pairs and flag_meanings:
        # flag_meanings without flag_values is not strictly conformant, but it
        # is common - the NERSC sea ice age status_flag declares
        # "nominal land invalid" with valid_min 1 and valid_max 3. Only adopt
        # the implied codes when the range accounts for every meaning exactly,
        # so a mislabelled legend is never invented.
        bounds = valid_range_for(var)
        if bounds is not None:
            low, high = bounds
            if float(low).is_integer() and float(high).is_integer():
                codes = list(range(int(low), int(high) + 1))
                if len(codes) == len(flag_meanings):
                    pairs = [(float(c), m) for c, m in zip(codes, flag_meanings)]

    if not pairs:
        long_name = str(var.attrs.get("long_name", "") or "")
        found = re.findall(r"(-?\d+)\s*:\s*([^,;]+)", long_name)
        if len(found) >= 2:
            pairs = [(float(v), label.strip()) for v, label in found]

    classes: list[list[Any]] = []
    for idx, (value, label) in enumerate(pairs):
        clean = label.replace("_", " ").strip()
        color = _FALLBACK_PALETTE[idx % len(_FALLBACK_PALETTE)]
        lowered = clean.lower()
        for keywords, rgb in _LABEL_COLORS:
            if any(k in lowered for k in keywords):
                color = rgb
                break
        classes.append([value - 0.5, value + 0.5, clean, list(color)])
    return classes


def _bbox_wgs84(ds: Any, grid: GridRef) -> list[float]:
    """Return [minx, miny, maxx, maxy] in WGS84 for a resolved grid.

    Prefers the source's own ACDD ``geospatial_*`` bounds, which are authoritative
    and, unlike a corner transform, stay correct when the grid encloses a pole.
    """

    import numpy as np

    declared = []
    for key in ("geospatial_lon_min", "geospatial_lat_min", "geospatial_lon_max", "geospatial_lat_max"):
        try:
            declared.append(float(ds.attrs[key]))
        except Exception:
            declared = []
            break
    if len(declared) == 4 and declared[0] < declared[2] and declared[1] < declared[3]:
        return declared

    if grid.is_geoloc:
        lon = ds[grid.lon_var]
        lat = ds[grid.lat_var]
        return [float(lon.min()), float(lat.min()), float(lon.max()), float(lat.max())]

    from osgeo import osr

    height, width = grid.shape
    if not height or not width or grid.geotransform is None:
        raise ConversionError("Cannot derive bounds for the resolved grid")

    source = osr.SpatialReference()
    source.ImportFromWkt(grid.srs_wkt)
    target = osr.SpatialReference()
    target.ImportFromEPSG(4326)
    for srs in (source, target):
        try:
            srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
        except Exception:
            pass
    transform = osr.CoordinateTransformation(source, target)

    # Sample the whole grid rather than just the corners: for a polar grid the
    # extreme latitude sits in the interior, not on the boundary.
    gt = grid.geotransform
    stride = max(1, int(max(height, width) // 128))
    rows = np.append(np.arange(0, height, stride), height)
    cols = np.append(np.arange(0, width, stride), width)
    lons: list[float] = []
    lats: list[float] = []
    for row in rows:
        for col in cols:
            x = gt[0] + gt[1] * float(col)
            y = gt[3] + gt[5] * float(row)
            try:
                lon, lat, *_ = transform.TransformPoint(x, y)
            except Exception:
                continue
            if np.isfinite(lon) and np.isfinite(lat) and abs(lat) <= 90.0:
                lons.append(lon)
                lats.append(lat)
    if not lons:
        raise ConversionError("Unable to project the grid into WGS84")
    minx, maxx = min(lons), max(lons)
    if maxx - minx > 350.0:
        # The grid wraps the antimeridian; the only honest bbox is the full span.
        minx, maxx = -180.0, 180.0
    return [minx, min(lats), maxx, max(lats)]


def read_slice(var: Any, selection: dict[str, int], attempts: int = 3, *, label: str = "") -> Any:
    """Read one 2-D slice of a variable, retrying transient OPeNDAP failures.

    ``selection`` maps each non-spatial dimension (time, vertical) to an index.
    A full ingest is one request per timestep against a remote THREDDS server,
    and a bulk registration multiplies that by the number of products. Servers
    drop connections and rate-limit; without a retry a single hiccup discards
    every timestep already transferred.
    """

    import time as _time

    import numpy as np

    last: Exception | None = None
    for attempt in range(attempts):
        try:
            return np.asarray(var.isel(selection).values, dtype="float32")
        except Exception as exc:  # noqa: BLE001 - any transport error is retryable
            last = exc
            if attempt == attempts - 1:
                break
            _time.sleep(2.0 * (2**attempt))
    what = label or ", ".join(f"{dim}={index}" for dim, index in selection.items()) or "the field"
    raise ReadError(f"Failed to read {what} after {attempts} attempts: {last}")


def _read_timestep(var: Any, time_dim: str, index: int, attempts: int = 3) -> Any:
    """Read one timestep of a (time, y, x) variable; see ``read_slice``."""

    return read_slice(var, {time_dim: index}, attempts, label=f"timestep {index}")


def _fill_value_for(var: Any) -> float:
    """Return a nodata value for a variable, preferring its declared _FillValue."""

    for key in ("_FillValue", "missing_value"):
        value = var.attrs.get(key)
        if value is not None:
            try:
                return float(value)
            except Exception:
                continue
    encoded = getattr(var, "encoding", {}) or {}
    for key in ("_FillValue", "missing_value"):
        value = encoded.get(key)
        if value is not None:
            try:
                return float(value)
            except Exception:
                continue
    return _DEFAULT_FILL



def _write_geolocation_rasters(ds: Any, lon_name: str, lat_name: str, out_dir: Path) -> tuple[Path, Path]:
    """Write the 2D lon/lat arrays as GeoTIFFs for use as a GDAL geolocation array.

    These are scratch: the VRT that references them is warped and deleted per
    timestep, and the resulting tiles carry baked pixels, so nothing reads them
    once materialisation ends. Write them into the temporary directory, which is
    removed with the rest of the scratch - keeping them beside the tiles left
    four byte-identical copies of the same two rasters (one pair per variable)
    permanently on disk, which was over half of what the pipeline stored.
    """

    import numpy as np
    from osgeo import gdal

    lon_path = out_dir / "_geoloc_lon.tif"
    lat_path = out_dir / "_geoloc_lat.tif"
    if lon_path.exists() and lat_path.exists():
        return lon_path, lat_path

    driver = gdal.GetDriverByName("GTiff")
    for path, values in ((lon_path, ds[lon_name].values), (lat_path, ds[lat_name].values)):
        array = np.asarray(values, dtype="float64")
        height, width = array.shape
        dst = driver.Create(str(path), width, height, 1, gdal.GDT_Float64)
        dst.GetRasterBand(1).WriteArray(array)
        dst = None
    return lon_path, lat_path


def _geolocation_metadata(lon_path: Path, lat_path: Path) -> dict[str, str]:
    from osgeo import osr

    return {
        "X_DATASET": str(lon_path),
        "X_BAND": "1",
        "Y_DATASET": str(lat_path),
        "Y_BAND": "1",
        "PIXEL_OFFSET": "0",
        "LINE_OFFSET": "0",
        "PIXEL_STEP": "1",
        "LINE_STEP": "1",
        "SRS": osr.SRS_WKT_WGS84_LAT_LONG,
    }


def _probe_target_grid(
    source_path: Path,
    target_srs_wkt: str,
    resolution: float | None,
    geoloc: bool = True,
) -> dict[str, Any]:
    """Derive one target grid (resolution + bounds) that every timestep will share."""

    from osgeo import gdal

    probe = gdal.Warp(
        "",
        str(source_path),
        format="VRT",
        dstSRS=target_srs_wkt,
        geoloc=geoloc,
        resampleAlg="near",
    )
    if probe is None:
        raise ConversionError("Unable to derive a target grid from geolocation arrays")
    gt = probe.GetGeoTransform()
    minx, maxy = gt[0], gt[3]
    maxx = minx + gt[1] * probe.RasterXSize
    miny = maxy + gt[5] * probe.RasterYSize
    xres = float(resolution) if resolution else abs(gt[1])
    yres = float(resolution) if resolution else abs(gt[5])
    probe = None
    return {"xres": xres, "yres": yres, "bounds": (minx, miny, maxx, maxy)}


def _unscaled_nodata(band: Any) -> float | None:
    """Return a band's nodata in the units the data has after unscaling.

    GDAL applies scale_factor to the pixels but leaves the nodata value in packed
    units, so a packed fill of -32768 with scale 0.001 reads as -32.768 while the
    band still claims -32768. Left uncorrected, every fill pixel renders as data.
    """

    nodata = band.GetNoDataValue()
    if nodata is None:
        return None
    scale = band.GetScale()
    offset = band.GetOffset()
    if scale in (None, 1.0) and offset in (None, 0.0):
        return float(nodata)
    return float(nodata) * float(scale or 1.0) + float(offset or 0.0)



# ---------------------------------------------------------------------------
# The vertical axis
# ---------------------------------------------------------------------------

# Vertical coordinates CF names directly.
_VERTICAL_STANDARD_NAMES = frozenset({
    "depth",
    "height",
    "altitude",
    "height_above_mean_sea_level",
    "height_above_reference_ellipsoid",
    "height_above_geopotential_datum",
    "air_pressure",
    "model_level_number",
    "sea_water_pressure",
})
# Units that make a coordinate a pressure level; CF says such an axis is
# vertical and increases downwards.
_PRESSURE_UNITS = frozenset({
    "pa", "hpa", "kpa", "mpa", "mbar", "millibar", "millibars", "bar", "dbar", "decibar", "atm",
})
# The last resort, when a dimension carries no CF signal at all.
_VERTICAL_NAMES = frozenset({
    "depth", "height", "altitude", "level", "levels", "lev", "z", "plev", "pressure",
    "isobaric", "deptht", "depthu", "depthv", "depthw", "s_rho", "s_w", "sigma", "hybrid",
})


def _is_parametric(standard_name: str) -> bool:
    """True for the CF dimensionless vertical coordinates (sigma, s, hybrid).

    Their values are model levels; turning them into metres or pascals needs
    the formula terms, which this module does not evaluate.
    """

    return bool(re.fullmatch(r"(atmosphere|ocean)_\w+_coordinate(_g\d)?", standard_name))


@dataclass
class VerticalAxis:
    """The vertical dimension of a variable and what its values mean.

    ``values`` is empty when the dimension has no coordinate variable; the
    level can then only be selected by index. ``positive`` is "up", "down" or
    "" when nothing says. ``parametric`` marks a dimensionless coordinate
    (sigma, s-levels, hybrid): its values are model levels, not metres.
    """

    dim: str
    values: list[float] = field(default_factory=list)
    units: str = ""
    positive: str = ""
    standard_name: str = ""
    parametric: bool = False
    source: str = ""

    def __len__(self) -> int:
        return len(self.values)


def _vertical_signal(name: str, coord: Any | None) -> str:
    """Say why a dimension is vertical ("axis", "positive", ...), or "" when it is not."""

    if coord is not None:
        attrs = coord.attrs
        if str(attrs.get("axis", "")).strip().upper() == "Z":
            return "axis"
        if str(attrs.get("positive", "")).strip().lower() in {"up", "down"}:
            return "positive"
        standard_name = str(attrs.get("standard_name", "")).strip().lower()
        if standard_name in _VERTICAL_STANDARD_NAMES or _is_parametric(standard_name):
            return "standard_name"
        if str(attrs.get("units", "")).strip().lower() in _PRESSURE_UNITS:
            return "units"
        if standard_name:
            # It declares itself as something else entirely; do not guess past that.
            return ""
    if name.lower() in _VERTICAL_NAMES:
        return "name"
    return ""


def _coordinate_for(ds: Any, dim: str) -> Any | None:
    """The 1-D coordinate variable along ``dim``, by name first, then by dimension."""

    if dim in ds.variables and tuple(ds[dim].dims) == (dim,):
        return ds[dim]
    for _, coord in ds.coords.items():
        if tuple(str(d) for d in coord.dims) == (dim,):
            return coord
    return None


def find_vertical_axis(ds: Any, variable: str, time_dim: str = "") -> VerticalAxis | None:
    """Return the vertical axis of ``variable``, or None when it has none.

    Every dimension of the variable other than ``time_dim`` and the trailing
    (y, x) pair must be vertical: a variable with any other leading dimension
    (an ensemble member, a band) is refused rather than sliced arbitrarily.
    """

    import numpy as np

    if variable not in ds.variables:
        raise SourceError(f"Unknown variable: {variable}")
    dims = [str(d) for d in ds[variable].dims]
    extra = [d for d in dims[:-2] if d != time_dim]
    if not extra:
        return None
    if len(extra) > 1:
        raise SourceError(
            f"Variable {variable} has more than one dimension besides time, y and x: {', '.join(extra)}"
        )
    dim = extra[0]
    coord = _coordinate_for(ds, dim)
    source = _vertical_signal(dim, coord)
    if not source:
        raise SourceError(f"Variable {variable} has a dimension {dim} that is neither time nor vertical")

    attrs = coord.attrs if coord is not None else {}
    units = str(attrs.get("units", "") or "").strip()
    standard_name = str(attrs.get("standard_name", "") or "").strip().lower()
    positive = str(attrs.get("positive", "") or "").strip().lower()
    if positive not in {"up", "down"}:
        positive = ""
        if units.lower() in _PRESSURE_UNITS or standard_name in {"air_pressure", "sea_water_pressure", "depth"}:
            positive = "down"
        elif standard_name.startswith(("height", "altitude")):
            positive = "up"
    values: list[float] = []
    if coord is not None:
        try:
            values = [float(v) for v in np.asarray(coord.values, dtype="float64")]
        except (TypeError, ValueError):
            values = []
    return VerticalAxis(
        dim=dim,
        values=values,
        units=units,
        positive=positive,
        standard_name=standard_name,
        parametric=_is_parametric(standard_name),
        source=source,
    )


# ---------------------------------------------------------------------------
# One field, many slices
# ---------------------------------------------------------------------------

# Output format defaults. A COG is what an OGC server and an object store both
# read well; GTiff is there for callers that need a plain tiled GeoTIFF.
_CREATION_OPTIONS: dict[str, list[str]] = {
    "COG": ["COMPRESS=DEFLATE", "BIGTIFF=IF_SAFER"],
    "GTiff": ["TILED=YES", "COMPRESS=DEFLATE", "BIGTIFF=IF_SAFER"],
}


def _slug(text: str) -> str:
    return re.sub(r"[^0-9A-Za-z]+", "_", text).strip("_") or "slice"


class GridField:
    """One variable of an open dataset, ready to be written slice by slice.

    The time axis, the vertical axis and the georeferencing are resolved once,
    and the target grid of a warp is probed once and then forced on every
    slice: the source grid is constant across time and levels, so letting GDAL
    re-derive it per slice only risks outputs landing on slightly different
    pixel sizes.

    ``nodata`` replaces the variable's declared fill as the output's nodata.
    """

    def __init__(self, ds: Any, variable: str, *, nodata: float | None = None) -> None:
        if variable not in ds.data_vars:
            raise SourceError(f"Unknown variable: {variable}")
        self.ds = ds
        self.variable = variable
        self.var = ds[variable]
        try:
            time_dim = _find_time_dim(ds)
        except SourceError:
            time_dim = ""
        self.time_dim = time_dim if time_dim in self.var.dims else ""
        self.vertical = find_vertical_axis(ds, variable, self.time_dim)
        expected = 2 + bool(self.time_dim) + (self.vertical is not None)
        if self.var.ndim != expected:
            raise SourceError(
                f"Variable {variable} has dimensions {tuple(str(d) for d in self.var.dims)}; "
                "expected (time?, vertical?, y, x)"
            )
        self.grid = resolve_georeferencing(ds, variable)
        self.fill = float(nodata) if nodata is not None else _fill_value_for(self.var)
        self.value_classes = parse_value_classes(self.var)
        self.valid_range = valid_range_for(self.var)
        self._probes: dict[tuple[str, float | None, str | None], dict[str, Any]] = {}
        self._times: list[str] | None = None

    # -- axes ---------------------------------------------------------------

    @property
    def times(self) -> list[str]:
        """The time axis as ISO 8601 stamps (empty when the field has no time)."""

        if self._times is None:
            self._times = [_iso(v) for v in self.ds[self.time_dim].values] if self.time_dim else []
        return self._times

    def _time_count(self) -> int:
        return int(self.ds.sizes[self.time_dim]) if self.time_dim else 0

    def _level_count(self) -> int:
        return int(self.ds.sizes[self.vertical.dim]) if self.vertical is not None else 0

    def selection(
        self,
        *,
        time: Any = None,
        time_index: int | None = None,
        level: float | None = None,
        level_index: int | None = None,
    ) -> dict[str, int]:
        """Turn a time and a level, each by value or by index, into dimension indices.

        An axis of one value is selected without being asked. An axis of more
        is never guessed: leaving it unselected is an error.
        """

        chosen: dict[str, int] = {}
        if self.time_dim:
            chosen[self.time_dim] = self._pick(
                "time", self._time_count(), time, time_index, self._time_position
            )
        elif time is not None or time_index is not None:
            raise SourceError(f"Variable {self.variable} has no time axis")
        if self.vertical is not None:
            chosen[self.vertical.dim] = self._pick(
                "level", self._level_count(), level, level_index, self._level_position
            )
        elif level is not None or level_index is not None:
            raise SourceError(f"Variable {self.variable} has no vertical axis")
        return chosen

    def _pick(self, what: str, count: int, value: Any, index: int | None, by_value: Any) -> int:
        if value is not None and index is not None:
            raise SourceError(f"Give the {what} by value or by index, not both")
        if index is not None:
            if not 0 <= int(index) < count:
                raise SourceError(f"{what} index {index} is outside 0..{count - 1}")
            return int(index)
        if value is not None:
            return by_value(value)
        if count == 1:
            return 0
        raise SourceError(f"Variable {self.variable} has {count} {what} values; select one")

    def _time_position(self, value: Any) -> int:
        stamp = _iso(value)
        try:
            return self.times.index(stamp)
        except ValueError:
            raise SourceError(f"Time {stamp} is not on the axis of {self.variable}") from None

    def _level_position(self, value: float) -> int:
        assert self.vertical is not None
        if not self.vertical.values:
            raise SourceError(
                f"The vertical dimension {self.vertical.dim} has no coordinate values; select it by index"
            )
        target = float(value)
        for position, candidate in enumerate(self.vertical.values):
            if abs(candidate - target) <= 1e-6 * max(1.0, abs(target)):
                return position
        raise SourceError(
            f"Level {target:g} is not on the {self.vertical.dim} axis "
            f"({', '.join(f'{v:g}' for v in self.vertical.values)})"
        )

    # -- reading and writing ------------------------------------------------

    def read(self, selection: dict[str, int], attempts: int = 3) -> Any:
        """Read the 2-D slice ``selection`` names, in physical units, as float32."""

        if self.time_dim and set(selection) == {self.time_dim}:
            return _read_timestep(self.var, self.time_dim, selection[self.time_dim], attempts)
        return read_slice(self.var, selection, attempts)

    def target_srs(self, target_epsg: int | None = None) -> tuple[Any, Any]:
        """Return (source SRS, target SRS) as GDAL spatial references.

        ``target_epsg`` of None keeps the source's own grid; only a source with
        no grid of its own (2D geolocation arrays) is written in Web Mercator,
        since one must be chosen.
        """

        from osgeo import osr

        source_srs = osr.SpatialReference()
        if self.grid.is_geoloc:
            source_srs.ImportFromEPSG(4326)
        else:
            source_srs.ImportFromWkt(self.grid.srs_wkt)

        target_srs = osr.SpatialReference()
        if target_epsg:
            target_srs.ImportFromEPSG(int(target_epsg))
        elif self.grid.is_geoloc:
            target_srs.ImportFromEPSG(_GEOLOC_FALLBACK_EPSG)
        else:
            target_srs.ImportFromWkt(self.grid.srs_wkt)
        return source_srs, target_srs

    def needs_warp(self, target_epsg: int | None = None, resolution: float | None = None) -> bool:
        """True unless the source grid can be written as it is."""

        source_srs, target_srs = self.target_srs(target_epsg)
        return self.grid.is_geoloc or resolution is not None or not source_srs.IsSame(target_srs)

    def write(
        self,
        values: Any,
        out_path: str | Path,
        *,
        target_epsg: int | None = None,
        resolution: float | None = None,
        scratch_dir: str | Path | None = None,
        geoloc_dir: str | Path | None = None,
        driver: str = "COG",
        creation_options: list[str] | None = None,
        selection: dict[str, int] | None = None,
        geoloc_convention: str | None = None,
    ) -> dict[str, Any]:
        """Write one 2-D slice ``values`` (from ``read``) and describe the output.

        Non-finite values become the field's nodata. A source that already has
        a grid in the target CRS is written straight out with its geotransform;
        anything else is warped (nearest neighbour) onto the field's one probed
        target grid. ``scratch_dir`` holds the intermediate files, and
        ``geoloc_dir`` the lon/lat rasters of a geolocation warp, shared by
        every variable of one source; both default to a temporary directory
        removed afterwards.

        ``geoloc_convention`` is GDAL's GEOREFERENCING_CONVENTION for the 2-D
        lon/lat of a curvilinear grid. None leaves GDAL's default,
        ``TOP_LEFT_CORNER``, which reads each lon/lat as the corner of its
        cell; CF coordinates are cell centres, so that places the output half
        a cell towards increasing row and column. ``PIXEL_CENTER`` reads them
        as CF means them. The default is kept because outputs already served
        were made with it.
        """

        import numpy as np
        from osgeo import gdal

        gdal.UseExceptions()
        if driver not in _CREATION_OPTIONS:
            raise SourceError(f"Unsupported output driver {driver!r}; use one of {', '.join(_CREATION_OPTIONS)}")
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        creation = list(creation_options) if creation_options is not None else list(_CREATION_OPTIONS[driver])

        grid_ref = self.grid
        fill = self.fill
        source_srs, target_srs = self.target_srs(target_epsg)
        target_wkt = target_srs.ExportToWkt()
        needs_warp = grid_ref.is_geoloc or resolution is not None or not source_srs.IsSame(target_srs)

        filled = np.where(np.isfinite(values), values, fill)
        height, width = filled.shape

        own_scratch = scratch_dir is None
        scratch = Path(tempfile.mkdtemp(prefix="cf2ogc_")) if own_scratch else Path(scratch_dir)
        scratch.mkdir(parents=True, exist_ok=True)
        try:
            if not needs_warp:
                if driver == "COG":
                    # The COG driver only copies; build the raster in memory first.
                    dst = gdal.GetDriverByName("MEM").Create("", width, height, 1, gdal.GDT_Float32)
                else:
                    dst = gdal.GetDriverByName(driver).Create(
                        str(out_path), width, height, 1, gdal.GDT_Float32, options=creation
                    )
                dst.SetGeoTransform(grid_ref.geotransform)
                dst.SetProjection(grid_ref.srs_wkt)
                band = dst.GetRasterBand(1)
                band.WriteArray(filled)
                band.SetNoDataValue(float(fill))
                if driver == "COG":
                    gdal.GetDriverByName("COG").CreateCopy(str(out_path), dst, options=creation)
                dst = None
                out_gt = tuple(grid_ref.geotransform)
                out_size = (width, height)
            else:
                stem = _slug(out_path.stem)
                raw_path = scratch / f"raw_{stem}.tif"
                raw = gdal.GetDriverByName("GTiff").Create(str(raw_path), width, height, 1, gdal.GDT_Float32)
                band = raw.GetRasterBand(1)
                band.WriteArray(filled)
                band.SetNoDataValue(float(fill))
                if not grid_ref.is_geoloc:
                    raw.SetGeoTransform(grid_ref.geotransform)
                    raw.SetProjection(grid_ref.srs_wkt)
                raw = None

                warp_source = raw_path
                vrt_path: Path | None = None
                if grid_ref.is_geoloc:
                    # The lat/lon arrays describe the grid, not the variable, so
                    # every variable of a source reuses one pair.
                    shared = Path(geoloc_dir) if geoloc_dir is not None else scratch
                    shared.mkdir(parents=True, exist_ok=True)
                    lon_path, lat_path = _write_geolocation_rasters(
                        self.ds, grid_ref.lon_var, grid_ref.lat_var, shared
                    )
                    vrt_path = scratch / f"geo_{stem}.vrt"
                    vrt = gdal.Translate(str(vrt_path), gdal.Open(str(raw_path)), format="VRT")
                    geoloc_meta = _geolocation_metadata(lon_path, lat_path)
                    if geoloc_convention:
                        geoloc_meta["GEOREFERENCING_CONVENTION"] = str(geoloc_convention)
                    vrt.SetMetadata(geoloc_meta, "GEOLOCATION")
                    vrt = None
                    warp_source = vrt_path

                key = (target_wkt, resolution, geoloc_convention)
                probe = self._probes.get(key)
                if probe is None:
                    probe = _probe_target_grid(warp_source, target_wkt, resolution, geoloc=grid_ref.is_geoloc)
                    self._probes[key] = probe

                warped = gdal.Warp(
                    str(out_path),
                    str(warp_source),
                    format=driver,
                    dstSRS=target_wkt,
                    geoloc=grid_ref.is_geoloc,
                    resampleAlg="near",
                    srcNodata=float(fill),
                    dstNodata=float(fill),
                    multithread=True,
                    xRes=probe["xres"],
                    yRes=probe["yres"],
                    outputBounds=probe["bounds"],
                    creationOptions=creation,
                )
                out_size = (warped.RasterXSize, warped.RasterYSize)
                out_gt = tuple(warped.GetGeoTransform())
                warped = None

                raw_path.unlink(missing_ok=True)
                if vrt_path is not None:
                    vrt_path.unlink(missing_ok=True)
        finally:
            if own_scratch:
                shutil.rmtree(scratch, ignore_errors=True)

        return self.describe(
            out_path,
            driver=driver,
            geotransform=out_gt,
            size=out_size,
            srs_wkt=target_wkt,
            epsg=int(target_epsg) if target_epsg else (_GEOLOC_FALLBACK_EPSG if grid_ref.is_geoloc else grid_ref.epsg),
            reprojected=needs_warp,
            selection=selection or {},
        )

    def convert(
        self,
        out_path: str | Path,
        *,
        time: Any = None,
        time_index: int | None = None,
        level: float | None = None,
        level_index: int | None = None,
        **write_options: Any,
    ) -> dict[str, Any]:
        """Select one slice, read it and write it; ``write`` documents the options."""

        chosen = self.selection(time=time, time_index=time_index, level=level, level_index=level_index)
        return self.write(self.read(chosen), out_path, selection=chosen, **write_options)

    def describe(
        self,
        out_path: str | Path,
        *,
        driver: str,
        geotransform: tuple[float, ...],
        size: tuple[int, int],
        srs_wkt: str,
        epsg: int | None,
        reprojected: bool,
        selection: dict[str, int],
    ) -> dict[str, Any]:
        """The description of one written slice, so a caller need not reopen it."""

        width, height = size
        gt = geotransform
        xs = (gt[0], gt[0] + gt[1] * width)
        ys = (gt[3], gt[3] + gt[5] * height)
        attrs = self.var.attrs
        time_info = None
        if self.time_dim:
            index = selection.get(self.time_dim)
            time_info = {
                "dim": self.time_dim,
                "index": index,
                "value": self.times[index] if index is not None else None,
            }
        level_info = None
        if self.vertical is not None:
            index = selection.get(self.vertical.dim)
            values = self.vertical.values
            level_info = {
                "dim": self.vertical.dim,
                "index": index,
                "value": values[index] if index is not None and values else None,
                "units": self.vertical.units,
                "positive": self.vertical.positive,
                "standard_name": self.vertical.standard_name,
                "parametric": self.vertical.parametric,
            }
        return {
            "path": str(out_path),
            "driver": driver,
            "width": int(width),
            "height": int(height),
            "crs_wkt": srs_wkt,
            "epsg": epsg,
            "geotransform": [float(v) for v in gt],
            "bounds": [min(xs), min(ys), max(xs), max(ys)],
            "resolution": abs(float(gt[1])),
            "reprojected": bool(reprojected),
            "georeferencing": self.grid.source,
            "bands": [
                {
                    "band": 1,
                    "variable": self.variable,
                    "long_name": str(attrs.get("long_name", "") or ""),
                    "standard_name": str(attrs.get("standard_name", "") or ""),
                    "units": str(attrs.get("units", "") or ""),
                    "dtype": "Float32",
                    "nodata": float(self.fill),
                    "valid_range": list(self.valid_range) if self.valid_range else [],
                    "value_classes": self.value_classes,
                }
            ],
            "time": time_info,
            "level": level_info,
        }


def convert(
    source: Any,
    variable: str,
    out_path: str | Path,
    *,
    time: Any = None,
    time_index: int | None = None,
    level: float | None = None,
    level_index: int | None = None,
    nodata: float | None = None,
    target_epsg: int | None = None,
    resolution: float | None = None,
    driver: str = "COG",
    creation_options: list[str] | None = None,
    geoloc_convention: str | None = None,
) -> dict[str, Any]:
    """Write one variable, one time step and one vertical level of a CF field.

    ``source`` is a path, an OPeNDAP URL or an open xarray Dataset. The time is
    given as a value (an ISO 8601 stamp or a datetime) or an index, and so is
    the level (a coordinate value in the axis' units, or an index). Returns the
    description of the output (``GridField.describe``).
    """

    opened = not hasattr(source, "data_vars")
    ds = open_source(str(source)) if opened else source
    try:
        grid_field = GridField(ds, variable, nodata=nodata)
        return grid_field.convert(
            out_path,
            time=time,
            time_index=time_index,
            level=level,
            level_index=level_index,
            target_epsg=target_epsg,
            resolution=resolution,
            driver=driver,
            creation_options=creation_options,
            geoloc_convention=geoloc_convention,
        )
    finally:
        if opened:
            ds.close()


# Public names for the helpers the service and other callers use.
find_time_dim = _find_time_dim
iso_time = _iso
packing_for = _packing_for
fill_value_for = _fill_value_for
unscaled_nodata = _unscaled_nodata
bbox_wgs84 = _bbox_wgs84
read_timestep = _read_timestep

__all__ = [
    "ConversionError",
    "GridError",
    "GridField",
    "GridRef",
    "ReadError",
    "SourceError",
    "VerticalAxis",
    "bbox_wgs84",
    "convert",
    "fill_value_for",
    "find_time_dim",
    "find_vertical_axis",
    "iso_time",
    "open_source",
    "packing_for",
    "parse_value_classes",
    "read_slice",
    "read_timestep",
    "resolve_georeferencing",
    "unscaled_nodata",
    "valid_range_for",
]
