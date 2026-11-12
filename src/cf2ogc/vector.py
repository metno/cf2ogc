# cf2ogc — CF-NetCDF to rasters an OGC service can serve
# Copyright (C) 2026 Norwegian Meteorological Institute (MET Norway)
# Author: Massimo Di Stefano <massimods@met.no>
# SPDX-License-Identifier: GPL-3.0-or-later

"""Vector fields to rasters: u, v and speed as the three bands of one raster.

A vector field is given either as two components (``u`` east or along the
grid's x axis, ``v`` north or along y) or as a direction and a magnitude
(waves, wind given as "from" a bearing). Either way one time step and one
vertical level is written as a 3-band raster:

* band 1: the u component,
* band 2: the v component,
* band 3: the speed, ``sqrt(u² + v²)``.

The time and vertical axes, the georeferencing and the target grid are those of
``cf2ogc.grid``: ``VectorField`` wraps one ``GridField`` per input variable and
selects, reads and warps exactly as it does, so a vector field shares the tile
layout of the scalar fields of the same product.

**Direction and magnitude.** A direction is a bearing in degrees (or radians)
clockwise from north. CF says whether it is the direction the flow comes
*from* (``wind_from_direction``, ``sea_surface_wave_from_direction``) or goes
*to* (``sea_water_velocity_to_direction``); the convention is read from the
variable's ``standard_name``, then its ``long_name``, and recorded in the
output. With a "to" bearing θ and magnitude m, ``u = m·sin θ`` and
``v = m·cos θ``; a "from" bearing points the other way.

**Which axes the components are in.** CF's ``eastward_*``/``northward_*``
components are geographic, but a raster is drawn along its own grid, and on a
projected grid east is not "right" except along one meridian (on a polar
stereographic grid the two differ by the longitude's distance from the central
meridian, tens of degrees over Norway). With ``components="grid"`` (the
default) each pixel's vector is turned into the output grid's axes: the
direction an arrow drawn on that grid must point, its length kept. With
``components="earth"`` the bands stay eastward and northward. On a conformal
grid whose north is up everywhere, such as EPSG:3857, the two are the same.
The choice is recorded in the output's description and in its metadata
(``CF2OGC_COMPONENTS``).

GDAL's Python bindings (``osgeo``) and pyproj are needed, as for
``cf2ogc.grid``; see the ``grid`` extra.
"""
from __future__ import annotations

import math
import re
import shutil
import tempfile
from pathlib import Path
from typing import Any

from .grid import (
    _CREATION_OPTIONS,
    GridField,
    SourceError,
    _geolocation_metadata,
    _probe_target_grid,
    _slug,
    _write_geolocation_rasters,
    open_source,
)

# How the components of the output are oriented (see the module docstring).
COMPONENTS = ("grid", "earth")

# A direction's convention, as CF names it.
DIRECTION_CONVENTIONS = ("from", "to")

# The bands of every output, in order.
BAND_ROLES = ("u", "v", "speed")

_DEGREE_UNITS = frozenset({
    "degree", "degrees", "deg", "degree_true", "degrees_true", "degree_north", "degrees_north",
    "degrees_clockwise_from_north", "degree_clockwise_from_north", "°",
})
_RADIAN_UNITS = frozenset({"rad", "radian", "radians"})

# Metres of the short ground step used to find east and north on the output grid.
_PROBE_METRES = 10.0
_EARTH_RADIUS = 6_371_008.8


def direction_convention(var: Any) -> str | None:
    """Return "from" or "to" for a direction variable, from its CF attributes.

    The ``standard_name`` decides when it names a direction
    (``*_from_direction``, ``*_to_direction``); otherwise the ``long_name`` is
    read for "from" or "to"/"towards". None when neither says.
    """

    standard_name = str(var.attrs.get("standard_name", "") or "").strip().lower()
    if standard_name.endswith("from_direction") or "_from_direction" in standard_name:
        return "from"
    if standard_name.endswith("to_direction") or "_to_direction" in standard_name:
        return "to"
    words = re.findall(r"[a-z]+", str(var.attrs.get("long_name", "") or "").lower())
    if "from" in words:
        return "from"
    if "towards" in words or "toward" in words or "to" in words or "going" in words:
        return "to"
    return None


def _direction_scale(var: Any) -> float:
    """Return the factor turning the direction's values into radians."""

    units = str(var.attrs.get("units", "") or "").strip().lower()
    if not units or units in _DEGREE_UNITS:
        return math.pi / 180.0
    if units in _RADIAN_UNITS:
        return 1.0
    raise SourceError(f"Direction variable has units {units!r}; expected degrees or radians")


def components_from_direction(direction: Any, magnitude: Any, convention: str, *, scale: float = math.pi / 180.0) -> tuple[Any, Any]:
    """Return (u, v), eastward and northward, of a bearing and a magnitude.

    ``direction`` is clockwise from north in the unit ``scale`` turns into
    radians (degrees by default); ``convention`` is "to" (the way the flow
    goes) or "from" (the way it comes from, pointing the opposite way).
    """

    import numpy as np

    if convention not in DIRECTION_CONVENTIONS:
        raise SourceError(f"Direction convention {convention!r}; use one of {', '.join(DIRECTION_CONVENTIONS)}")
    theta = np.asarray(direction, dtype="float64") * float(scale)
    m = np.asarray(magnitude, dtype="float64")
    sign = -1.0 if convention == "from" else 1.0
    return (sign * m * np.sin(theta)).astype("float32"), (sign * m * np.cos(theta)).astype("float32")


def _is_fill(values: Any, fill: float) -> Any:
    """Where ``values`` hold the nodata ``fill`` (which may itself be NaN)."""

    import numpy as np

    return np.isnan(values) if math.isnan(fill) else (values == fill) | np.isnan(values)


def _frame_of(var: Any) -> str | None:
    """"earth" for eastward/northward components, "grid" for x/y ones, else None."""

    name = str(var.attrs.get("standard_name", "") or "").strip().lower()
    if name.startswith(("eastward_", "northward_")):
        return "earth"
    if name.startswith(("x_", "y_", "grid_eastward_", "grid_northward_")):
        return "grid"
    return None


def speed_standard_name(u_standard_name: str) -> str:
    """The CF standard name of the speed of a component, where CF has one."""

    name = str(u_standard_name or "").strip()
    for prefix in ("eastward_", "northward_"):
        if name.startswith(prefix):
            base = name[len(prefix):]
            if base == "wind":
                return "wind_speed"
            if base.endswith("_velocity"):
                return base[: -len("_velocity")] + "_speed"
            return ""
    return ""


class VectorField:
    """A vector field of an open dataset, written slice by slice as u, v and speed.

    Give ``u`` and ``v`` (component variables) or ``direction`` and
    ``magnitude``. The variables must share their dimensions and grid. The
    frame of u/v components is read from their standard names (eastward and
    northward are "earth", x and y "grid"); ``frame`` overrides it. A
    direction's convention is read from its attributes; ``convention``
    overrides it. ``nodata`` replaces the declared fill as the output's nodata
    on all three bands.
    """

    def __init__(
        self,
        ds: Any,
        u: str | None = None,
        v: str | None = None,
        *,
        direction: str | None = None,
        magnitude: str | None = None,
        convention: str | None = None,
        frame: str | None = None,
        nodata: float | None = None,
    ) -> None:
        by_components = u is not None or v is not None
        by_direction = direction is not None or magnitude is not None
        if by_components == by_direction:
            raise SourceError("Give either u and v, or direction and magnitude")
        self.ds = ds
        self.source = "components" if by_components else "direction_magnitude"
        self.convention: str | None = None
        self._direction_scale = 1.0
        if by_components:
            if not (u and v):
                raise SourceError("Both u and v are needed")
            first, second = GridField(ds, u, nodata=nodata), GridField(ds, v, nodata=nodata)
            self.frame = frame or _frame_of(first.var) or _frame_of(second.var)
            if self.frame is None:
                raise SourceError(
                    f"Cannot tell whether {u}/{v} are eastward/northward or grid x/y components "
                    "from their standard names; pass frame='earth' or frame='grid'"
                )
        else:
            if not (direction and magnitude):
                raise SourceError("Both direction and magnitude are needed")
            first, second = GridField(ds, direction, nodata=nodata), GridField(ds, magnitude, nodata=nodata)
            self.convention = convention or direction_convention(first.var)
            if self.convention is None:
                raise SourceError(
                    f"The attributes of {direction} do not say whether it is the direction the flow comes "
                    "from or goes to; pass convention='from' or convention='to'"
                )
            if self.convention not in DIRECTION_CONVENTIONS:
                raise SourceError(f"Direction convention {self.convention!r}; use one of from, to")
            self._direction_scale = _direction_scale(first.var)
            # A bearing is clockwise from (true) north: geographic.
            self.frame = "earth"
        if self.frame not in ("earth", "grid"):
            raise SourceError(f"Component frame {self.frame!r}; use 'earth' or 'grid'")
        if tuple(first.var.dims) != tuple(second.var.dims) or first.var.shape != second.var.shape:
            raise SourceError(
                f"{first.variable} {tuple(first.var.dims)} and {second.variable} {tuple(second.var.dims)} "
                "do not share their dimensions"
            )
        self.first, self.second = first, second
        # The output's one nodata, on all three bands.
        self.fill = first.fill
        self._rotations: dict[tuple, tuple[Any, Any, Any, Any] | None] = {}
        self._probes: dict[tuple, dict[str, Any]] = {}

    # -- what GridField already knows ---------------------------------------

    @property
    def grid(self) -> Any:
        return self.first.grid

    @property
    def time_dim(self) -> str:
        return self.first.time_dim

    @property
    def vertical(self) -> Any:
        return self.first.vertical

    @property
    def times(self) -> list[str]:
        return self.first.times

    @property
    def variables(self) -> list[str]:
        return [self.first.variable, self.second.variable]

    def selection(self, **kwargs: Any) -> dict[str, int]:
        """Select a time and a level, as ``GridField.selection`` does."""

        return self.first.selection(**kwargs)

    def target_srs(self, target_epsg: int | None = None) -> tuple[Any, Any]:
        return self.first.target_srs(target_epsg)

    def needs_warp(self, target_epsg: int | None = None, resolution: float | None = None) -> bool:
        return self.first.needs_warp(target_epsg, resolution)

    # -- reading ---------------------------------------------------------------

    def read(self, selection: dict[str, int], attempts: int = 3) -> tuple[Any, Any]:
        """Read one slice as (u, v), in physical units, float32, NaN where either is missing.

        A direction and magnitude are converted to eastward and northward
        components here.
        """

        import numpy as np

        a = np.asarray(self.first.read(selection, attempts), dtype="float32")
        b = np.asarray(self.second.read(selection, attempts), dtype="float32")
        if self.source == "direction_magnitude":
            a, b = components_from_direction(a, b, str(self.convention), scale=self._direction_scale)
        missing = ~(np.isfinite(a) & np.isfinite(b))
        a = np.where(missing, np.nan, a).astype("float32")
        b = np.where(missing, np.nan, b).astype("float32")
        return a, b

    # -- turning components onto the output grid -----------------------------

    def _rotation(self, target_wkt: str, geotransform: tuple, size: tuple[int, int]) -> tuple[Any, Any, Any, Any] | None:
        """The output grid's displacement per metre east and per metre north, at each pixel.

        Returns (ex, ey, nx, ny) arrays, or None when east is +x and north is
        +y everywhere already (nothing to turn).
        """

        key = (target_wkt, tuple(geotransform), tuple(size))
        if key in self._rotations:
            return self._rotations[key]

        import numpy as np
        from pyproj import CRS, Transformer

        width, height = size
        gt = geotransform
        cols = np.arange(width, dtype="float64") + 0.5
        rows = np.arange(height, dtype="float64") + 0.5
        cc, rr = np.meshgrid(cols, rows)
        x = gt[0] + cc * gt[1] + rr * gt[2]
        y = gt[3] + cc * gt[4] + rr * gt[5]

        crs = CRS.from_wkt(target_wkt)
        to_lonlat = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
        from_lonlat = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
        lon, lat = to_lonlat.transform(x, y)
        coslat = np.maximum(np.cos(np.radians(lat)), 1e-6)
        dlon = np.degrees(_PROBE_METRES / (_EARTH_RADIUS * coslat))
        dlat = np.degrees(_PROBE_METRES / _EARTH_RADIUS)
        xe, ye = from_lonlat.transform(lon + dlon, lat)
        xn, yn = from_lonlat.transform(lon, np.minimum(lat + dlat, 90.0))
        # Grid displacement per metre, in grid "up" = +y (rows run the other way
        # when the geotransform's y step is negative, which is handled by
        # working in CRS coordinates, not pixels).
        ex, ey = (xe - x) / _PROBE_METRES, (ye - y) / _PROBE_METRES
        nx, ny = (xn - x) / _PROBE_METRES, (yn - y) / _PROBE_METRES
        rotation: tuple[Any, Any, Any, Any] | None
        # Already aligned: east along +x, north along +y (any scale, any pixel).
        aligned = (
            np.nanmax(np.abs(ey)) <= 1e-9 * np.nanmax(np.abs(ex))
            and np.nanmax(np.abs(nx)) <= 1e-9 * np.nanmax(np.abs(ny))
            and np.nanmin(ex) > 0
            and np.nanmin(ny) > 0
        )
        rotation = None if aligned else (
            ex.astype("float32"), ey.astype("float32"), nx.astype("float32"), ny.astype("float32")
        )
        self._rotations[key] = rotation
        return rotation

    @staticmethod
    def _to_grid(u: Any, v: Any, rotation: tuple[Any, Any, Any, Any]) -> tuple[Any, Any]:
        """Turn eastward/northward (u, v) into grid (x, y) components, keeping each length."""

        import numpy as np

        ex, ey, nx, ny = rotation
        gx = u * ex + v * nx
        gy = u * ey + v * ny
        length = np.hypot(u, v)
        scale = np.hypot(gx, gy)
        with np.errstate(invalid="ignore", divide="ignore"):
            factor = np.where(scale > 0, length / scale, 0.0)
        return (gx * factor).astype("float32"), (gy * factor).astype("float32")

    # -- writing ---------------------------------------------------------------

    def write(
        self,
        values: tuple[Any, Any],
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
        components: str = "grid",
    ) -> dict[str, Any]:
        """Write one slice ``values`` = (u, v) from ``read`` as u, v, speed and describe it.

        The grid options are ``GridField.write``'s: the output lands on the
        same grid a scalar field of the product would. ``components`` is
        "grid" (turned into the output grid's axes) or "earth" (left
        eastward/northward); see the module docstring. Grid-relative input
        components can only be written onto their own grid.
        """

        import numpy as np
        from osgeo import gdal

        gdal.UseExceptions()
        if components not in COMPONENTS:
            raise SourceError(f"components={components!r}; use one of {', '.join(COMPONENTS)}")
        if driver not in _CREATION_OPTIONS:
            raise SourceError(f"Unsupported output driver {driver!r}; use one of {', '.join(_CREATION_OPTIONS)}")
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        creation = list(creation_options) if creation_options is not None else list(_CREATION_OPTIONS[driver])

        grid_ref = self.grid
        fill = float(self.fill)
        source_srs, target_srs = self.target_srs(target_epsg)
        target_wkt = target_srs.ExportToWkt()
        needs_warp = grid_ref.is_geoloc or resolution is not None or not source_srs.IsSame(target_srs)
        if self.frame == "grid" and needs_warp:
            raise SourceError(
                "Grid-relative components can only be written on their own grid; "
                "leave target_epsg and resolution unset"
            )
        if self.frame == "grid" and components == "earth":
            raise SourceError("Grid-relative components cannot be written as eastward/northward")

        u, v = (np.asarray(a, dtype="float32") for a in values)
        speed = np.hypot(u, v).astype("float32")
        height, width = u.shape
        stack = [np.where(np.isfinite(a), a, fill).astype("float32") for a in (u, v, speed)]

        own_scratch = scratch_dir is None
        scratch = Path(tempfile.mkdtemp(prefix="cf2ogc_")) if own_scratch else Path(scratch_dir)
        scratch.mkdir(parents=True, exist_ok=True)
        try:
            mem = gdal.GetDriverByName("MEM")
            if not needs_warp:
                grid_out = mem.Create("", width, height, 3, gdal.GDT_Float32)
                grid_out.SetGeoTransform(grid_ref.geotransform)
                grid_out.SetProjection(grid_ref.srs_wkt)
                for index, array in enumerate(stack, start=1):
                    band = grid_out.GetRasterBand(index)
                    band.WriteArray(array)
                    band.SetNoDataValue(fill)
            else:
                stem = _slug(out_path.stem)
                raw_path = scratch / f"raw_{stem}.tif"
                raw = gdal.GetDriverByName("GTiff").Create(str(raw_path), width, height, 3, gdal.GDT_Float32)
                for index, array in enumerate(stack, start=1):
                    band = raw.GetRasterBand(index)
                    band.WriteArray(array)
                    band.SetNoDataValue(fill)
                if not grid_ref.is_geoloc:
                    raw.SetGeoTransform(grid_ref.geotransform)
                    raw.SetProjection(grid_ref.srs_wkt)
                raw = None

                warp_source = raw_path
                vrt_path: Path | None = None
                if grid_ref.is_geoloc:
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
                # Nearest neighbour keeps every output vector one the source
                # holds; the speed band is warped with the components, so it
                # stays their length.
                grid_out = gdal.Warp(
                    "",
                    str(warp_source),
                    format="MEM",
                    dstSRS=target_wkt,
                    geoloc=grid_ref.is_geoloc,
                    resampleAlg="near",
                    srcNodata=fill,
                    dstNodata=fill,
                    multithread=True,
                    xRes=probe["xres"],
                    yRes=probe["yres"],
                    outputBounds=probe["bounds"],
                )
                raw_path.unlink(missing_ok=True)
                if vrt_path is not None:
                    vrt_path.unlink(missing_ok=True)

            out_gt = tuple(grid_out.GetGeoTransform())
            out_size = (grid_out.RasterXSize, grid_out.RasterYSize)
            rotated = False
            if components == "grid" and self.frame == "earth":
                rotation = self._rotation(target_wkt, out_gt, out_size)
                if rotation is not None:
                    gu = grid_out.GetRasterBand(1).ReadAsArray().astype("float32")
                    gv = grid_out.GetRasterBand(2).ReadAsArray().astype("float32")
                    valid = ~(_is_fill(gu, fill) | _is_fill(gv, fill))
                    ru, rv = self._to_grid(gu, gv, rotation)
                    grid_out.GetRasterBand(1).WriteArray(np.where(valid, ru, fill).astype("float32"))
                    grid_out.GetRasterBand(2).WriteArray(np.where(valid, rv, fill).astype("float32"))
                    rotated = True

            for index, role in enumerate(BAND_ROLES, start=1):
                band = grid_out.GetRasterBand(index)
                band.SetDescription(role)
                band.SetNoDataValue(fill)
            for key, value in self._metadata(components).items():
                grid_out.SetMetadataItem(key, value)
            final = gdal.GetDriverByName(driver).CreateCopy(str(out_path), grid_out, options=creation)
            final = None
            grid_out = None
        finally:
            if own_scratch:
                shutil.rmtree(scratch, ignore_errors=True)

        return self.describe(
            out_path,
            driver=driver,
            geotransform=out_gt,
            size=out_size,
            srs_wkt=target_wkt,
            epsg=int(target_epsg) if target_epsg else (3857 if grid_ref.is_geoloc else grid_ref.epsg),
            reprojected=needs_warp,
            selection=selection or {},
            components=components,
            rotated=rotated,
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

    # -- describing --------------------------------------------------------

    def vector_info(self, components: str = "grid") -> dict[str, Any]:
        """How the bands were made: the inputs, the convention, the component axes."""

        info: dict[str, Any] = {
            "source": self.source,
            "bands": {"u": 1, "v": 2, "speed": 3},
            "input_frame": self.frame,
            "components": "grid" if self.frame == "grid" else components,
        }
        if self.source == "components":
            info.update({"u_variable": self.first.variable, "v_variable": self.second.variable})
        else:
            info.update({
                "direction_variable": self.first.variable,
                "magnitude_variable": self.second.variable,
                "direction_convention": self.convention,
                "direction_reference": "clockwise from north",
            })
        return info

    def _metadata(self, components: str) -> dict[str, str]:
        info = self.vector_info(components)
        meta = {
            "CF2OGC_VECTOR_SOURCE": info["source"],
            "CF2OGC_COMPONENTS": info["components"],
            "CF2OGC_BANDS": "u,v,speed",
        }
        if self.source == "components":
            meta["CF2OGC_U_VARIABLE"] = info["u_variable"]
            meta["CF2OGC_V_VARIABLE"] = info["v_variable"]
        else:
            meta["CF2OGC_DIRECTION_VARIABLE"] = info["direction_variable"]
            meta["CF2OGC_MAGNITUDE_VARIABLE"] = info["magnitude_variable"]
            meta["CF2OGC_DIRECTION_CONVENTION"] = str(info["direction_convention"])
        return meta

    def _band_descriptions(self) -> list[dict[str, Any]]:
        if self.source == "components":
            u_attrs, v_attrs = self.first.var.attrs, self.second.var.attrs
            units = str(u_attrs.get("units", "") or "")
            u_name, v_name = self.first.variable, self.second.variable
            u_std = str(u_attrs.get("standard_name", "") or "")
            v_std = str(v_attrs.get("standard_name", "") or "")
            u_long = str(u_attrs.get("long_name", "") or "")
            v_long = str(v_attrs.get("long_name", "") or "")
            speed_std = speed_standard_name(u_std)
            base = u_long.replace("eastward", "").replace("Eastward", "").strip(" ,") if u_long else ""
            speed_long = f"{base} speed".strip() if base else "speed"
        else:
            m_attrs = self.second.var.attrs
            units = str(m_attrs.get("units", "") or "")
            u_name, v_name = f"{self.second.variable}_u", f"{self.second.variable}_v"
            u_std = v_std = speed_std = ""
            magnitude_long = str(m_attrs.get("long_name", "") or self.second.variable)
            u_long = f"eastward component of {magnitude_long}"
            v_long = f"northward component of {magnitude_long}"
            speed_long = magnitude_long
        return [
            {"band": 1, "role": "u", "variable": u_name, "long_name": u_long, "standard_name": u_std, "units": units},
            {"band": 2, "role": "v", "variable": v_name, "long_name": v_long, "standard_name": v_std, "units": units},
            {"band": 3, "role": "speed", "variable": "speed", "long_name": speed_long,
             "standard_name": speed_std, "units": units},
        ]

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
        components: str = "grid",
        rotated: bool = False,
    ) -> dict[str, Any]:
        """The description of one written slice: ``GridField.describe``'s, with three bands."""

        described = self.first.describe(
            out_path,
            driver=driver,
            geotransform=geotransform,
            size=size,
            srs_wkt=srs_wkt,
            epsg=epsg,
            reprojected=reprojected,
            selection=selection,
        )
        bands = []
        for entry in self._band_descriptions():
            bands.append({
                **entry,
                "dtype": "Float32",
                "nodata": float(self.fill),
                "valid_range": [],
                "value_classes": [],
            })
        described["bands"] = bands
        described["vector"] = {**self.vector_info(components), "rotated": bool(rotated)}
        return described


def convert(
    source: Any,
    out_path: str | Path,
    *,
    u: str | None = None,
    v: str | None = None,
    direction: str | None = None,
    magnitude: str | None = None,
    convention: str | None = None,
    frame: str | None = None,
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
    components: str = "grid",
) -> dict[str, Any]:
    """Write one time step and vertical level of a vector field as u, v, speed.

    ``source`` is a path, an OPeNDAP URL or an open xarray Dataset. Give ``u``
    and ``v``, or ``direction`` and ``magnitude``. Returns the description of
    the output (``VectorField.describe``).
    """

    opened = not hasattr(source, "data_vars")
    ds = open_source(str(source)) if opened else source
    try:
        field = VectorField(
            ds, u, v, direction=direction, magnitude=magnitude, convention=convention, frame=frame, nodata=nodata
        )
        return field.convert(
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
            components=components,
        )
    finally:
        if opened:
            ds.close()


__all__ = [
    "BAND_ROLES",
    "COMPONENTS",
    "DIRECTION_CONVENTIONS",
    "VectorField",
    "components_from_direction",
    "convert",
    "direction_convention",
    "speed_standard_name",
]
