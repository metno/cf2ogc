# cf2ogc

CF-compliant NetCDF in, rasters an OGC service (WMS, WCS) can serve out.

Two converters:

- `cf2ogc.points` takes a point cloud stored along a single `point` dimension
  and grids it onto a regular raster, written as a tiled, LZW-compressed
  Float32 GeoTIFF with internal overviews. Airborne hyperspectral (HySpex)
  point clouds are the first use, but the code has nothing HySpex-specific in
  it: any NetCDF with `X`, `Y` and value variables along `point` works.
- `cf2ogc.grid` takes a gridded CF variable, (y, x) with an optional time and
  an optional vertical dimension, and writes one time step at one vertical
  level as a Cloud Optimized GeoTIFF, with a description of what it wrote.
- `cf2ogc.vector` takes a vector field (u and v components, or a direction and
  a magnitude) on the same kind of grid and writes one time step at one
  vertical level as a 3-band raster: u, v and speed.

Licence: GPL-3.0-or-later (see `LICENSE`).
© 2026 Norwegian Meteorological Institute (MET Norway).

## Install

```sh
pip install "cf2ogc @ git+https://github.com/metno/cf2ogc@v0.3.0"
# with the numba JIT accumulators used by the fast implementation:
pip install "cf2ogc[numba] @ git+https://github.com/metno/cf2ogc@v0.3.0"
# with what cf2ogc.grid needs besides GDAL's Python bindings:
pip install "cf2ogc[grid] @ git+https://github.com/metno/cf2ogc@v0.3.0"
```

Dependencies: numpy, xarray, dask, netCDF4, rasterio, PyYAML; numba is optional.
`cf2ogc.grid` also needs GDAL's own Python bindings (`osgeo`), which come with
the system's GDAL (`apt install python3-gdal`, conda's `gdal`) rather than from
a wheel, and pyproj (the `grid` extra).
Python 3.10 or newer.

## Gridded fields: `cf2ogc.grid`

```python
from cf2ogc.grid import convert

described = convert(
    "ocean_model.nc",            # a path, an OPeNDAP URL or an open xarray Dataset
    "temp",
    "temp_20240101T06_10m.tif",
    time="2024-01-01T06:00:00Z", # or time_index=1
    level=10.0,                  # or level_index=1, in the axis' own units
)
```

The input is one CF variable whose last two dimensions are the grid (y, x),
optionally preceded by a time dimension and a vertical one. Any other leading
dimension (an ensemble member, a spectral band) is refused. Nothing is
averaged or picked for you: an axis with more than one value must be selected,
by value or by index; an axis of one value is taken as it is.

**Axes are found by CF attributes first.** Time: `axis = "T"` or
`standard_name = "time"`, then the names `time` and `t`. Vertical: `axis = "Z"`,
`positive` (`up`/`down`), a vertical `standard_name` (`depth`, `height`,
`altitude`, `air_pressure`, `model_level_number`, the dimensionless
`ocean_*_coordinate` / `atmosphere_*_coordinate`), pressure units, and only
then a name such as `depth`, `lev` or `plev`. A dimensionless coordinate
(sigma, s-levels, hybrid) is reported with `parametric: true`: its values are
model levels, not metres.

**Georeferencing, best first:** a CF `grid_mapping` variable that exists plus
regular 1-D projection axes (written as they are, no resampling; axes in km
are converted); regular 1-D lon/lat axes (EPSG:4326, no resampling); 2-D
lon/lat arrays, a curvilinear grid, warped through GDAL geolocation arrays
(nearest neighbour) to Web Mercator unless `target_epsg` says otherwise. The
CRS is read from `crs_wkt`, `spatial_ref`, `proj4_string`, then the CF
parameters, and given its EPSG code when one can be proven.

`geoloc_convention` sets GDAL's `GEOREFERENCING_CONVENTION` for a curvilinear
grid. Left unset, GDAL reads each lon/lat as the corner of its cell
(`TOP_LEFT_CORNER`), which puts the output half a cell towards increasing row
and column, since CF coordinates are cell centres; `PIXEL_CENTER` reads them
as CF means them.

**Values** are read in physical units (packing undone), non-finite values
become the nodata (the variable's `_FillValue` or `missing_value`, else
-9999, or `nodata=`), and the output is Float32. A read that fails is retried
twice with a growing pause (2 s, 4 s), since a remote OPeNDAP server drops
connections.

**The description** returned (and printed by the command line):

```text
path, driver, width, height, crs_wkt, epsg, geotransform,
bounds [minx, miny, maxx, maxy] in the output CRS, resolution, reprojected,
georeferencing (how the grid was placed),
bands: [{band, variable, long_name, standard_name, units, dtype, nodata,
         valid_range (unpacked), value_classes}],
time:  {dim, index, value (ISO 8601)} or null,
level: {dim, index, value, units, positive, standard_name, parametric} or null
```

`value_classes` is a legend for a categorical variable, from `flag_values` /
`flag_meanings`, from `flag_meanings` with a `valid_min`/`valid_max` that
accounts for every meaning, or from a `long_name` such as
`"-1: ocean, 0: snow free, 1: snow"`: `[low, high, label, [r, g, b]]` per class.

**Many slices of one field.** `GridField(ds, variable)` resolves the axes and
the grid once; `field.convert(path, time_index=i, level_index=k)` writes each
slice, and every slice of a warp lands on one target grid. `field.read(...)`
and `field.write(values, path, ...)` split the two steps, for a caller that
reads the values elsewhere. `driver="GTiff"` with `creation_options` writes a
plain GeoTIFF instead of a COG.

Errors: `cf2ogc.grid.SourceError` (the source cannot be converted as asked),
`ReadError` (it could not be opened or read), `ConversionError` (GDAL could not
produce the grid), all `GridError`.

## Vector fields: `cf2ogc.vector`

```python
from cf2ogc.vector import convert

described = convert(
    "https://thredds.met.no/thredds/dodsC/fou-hi/norkystv3_800m_m00_be",
    "current_0m.tif",
    u="u_eastward", v="v_northward",
    time="2026-11-15T12:00:00Z", level=0.0,
    target_epsg=3857,
)
```

The output has three Float32 bands, always in this order: **1 = u**, **2 = v**,
**3 = speed** (`sqrt(u² + v²)`), one nodata on all three (a pixel where either
component is missing is missing in every band). Time, level, grid and warp work
as in `cf2ogc.grid`; `VectorField(ds, u=..., v=...)` is the many-slices form.

**Direction and magnitude.** `convert(src, out, direction="thq", magnitude="hs", ...)`
converts a bearing (degrees or radians, clockwise from north) and a magnitude
to eastward/northward components. Whether the bearing is where the flow comes
*from* or goes *to* is read from CF: the `standard_name` (`*_from_direction`,
`*_to_direction`), else the `long_name`; `convention="from"|"to"` overrides it,
and a variable that says neither is refused rather than guessed. With a "to"
bearing θ and magnitude m: `u = m·sin θ`, `v = m·cos θ`; "from" points the other
way. The convention is recorded in the description (`vector.direction_convention`)
and in the raster (`CF2OGC_DIRECTION_CONVENTION`).

**Which axes u and v are in.** `eastward_*`/`northward_*` components are
geographic, but an arrow is drawn along the raster's grid, and on a projected
grid "east" is not "right" (on a polar stereographic grid they differ by the
distance from the central meridian). With `components="grid"` (the default)
each vector is turned into the output grid's axes, its length kept; with
`components="earth"` the bands stay eastward/northward. On EPSG:3857 (conformal,
north up everywhere) the two are the same, which is why a service that draws
arrows with MapServer's `UVRASTER` (it reads the bands in its own pixel axes)
should write vector rasters there. Grid-relative inputs (`x_*`/`y_*` standard
names, or `frame="grid"`) are written only onto their own grid. The choice is
recorded in the description (`vector.components`, `vector.rotated`) and in the
raster's metadata (`CF2OGC_COMPONENTS`, band descriptions `u`, `v`, `speed`).

## The input

- a 1-D dimension `point`;
- coordinate variables `X` and `Y` along `point`, in the CRS of the data;
- one or more value variables along `point` (for example `Z`, or `red`,
  `green`, `blue`); a variable with an extra `band` dimension (a spectrum) is
  read at one wavelength with `band_selection`;
- the CRS in a CF `crs` variable (`crs_wkt`, or `epsg_code`). Without it the
  GeoTIFF has no CRS, and a warning is printed.

## The configuration

```yaml
input_dataset: /path/to/cloud.nc     # used by the command line when -i is not given
output_file: cloud_Z.tif             # used by the command line when -o is not given

grid:
  resolution: 10.0                   # pixel size, in the units of X and Y
  bounds_strategy: rounded           # "rounded": align the top-left corner to the resolution

processing:
  implementation: classic            # classic | fast
  chunk_size: 500000                 # points read per chunk
  extent_strategy: compute           # auto | metadata | compute
  extent_engine: dask                # dask | numba | serial
  extent_workers: 8
  force_chunking: true               # chunk an unchunked file along 'point'
  # fast implementation only:
  parallel_mode: numba               # threads | numba | none
  parallel_workers: 8
  use_numba: true
  tile_size: 256                     # numba mode: tiles that group points to avoid write collisions
  auto_tile_size: false

output_bands:                        # one entry per band of the output
  - variable: Z
    method: mean                     # mean | min | max | count
    nodata_val: 0                    # optional, default 0; band 1's value is the file's nodata
  # - variable: intensity
  #   band_selection: 550            # wavelength, nearest match
  #   method: mean
```

A pixel that no point falls into gets the band's `nodata_val`. Nothing is
interpolated.

`classic` accumulates with `numpy.add.at`; `fast` uses `numpy.bincount`, or
numba when `use_numba` is true or `parallel_mode` is `numba`. Both write the
same grid, up to the order in which floating-point sums are added.

Required keys: `grid.resolution`, `grid.bounds_strategy`,
`processing.chunk_size` and `output_bands`.

## From Python

```python
from cf2ogc.points import grid_points

config = {
    "grid": {"resolution": 10.0, "bounds_strategy": "rounded"},
    "processing": {"implementation": "classic", "chunk_size": 500000},
    "output_bands": [{"variable": "Z", "method": "mean"}],
}
grid_points(config, "cloud.nc", "cloud_Z.tif")
```

The library functions use the configuration exactly as given and add no
defaults: a missing `implementation` means `classic`, and every other missing
key takes the default written in the code where it is read. The functions are
also available one by one: `process_point_cloud` (classic) and
`process_point_cloud_fast`.

**Cancelling.** Pass `cancel=`, a callable that takes no argument and returns
true to stop. It is called before each chunk (by `fast`, in the numba-parallel
mode only); a true result raises `cf2ogc.points.Cancelled`, a `RuntimeError`.
An exception raised inside the callable counts as "keep going".

```python
grid_points(config, src, dst, cancel=lambda: job_was_cancelled(job_id))
```

`cf2ogc.io.open_netcdf_dataset(path, chunks=..., decode_coords="all")` opens a
file with the netCDF4 engine, no file cache, and HDF5 file locking off unless
the environment sets `HDF5_USE_FILE_LOCKING` (locking fails on read-only Lustre
and NFS mounts).

## From the command line

```sh
cf2ogc points config.yaml -i cloud.nc -o cloud_Z.tif
cf2ogc points config.yaml --implementation classic
cf2ogc grid ocean_model.nc temp -o temp.tif --time 2024-01-01T06:00:00Z --level 10
cf2ogc grid snow.nc snow_probability -o snow.tif --time-index 0 --target-epsg 3035
```

```sh
cf2ogc vector ocean_model.nc -o current.tif --u u_eastward --v v_northward --time-index 0 --level 0
cf2ogc vector waves.nc -o waves.tif --direction thq --magnitude hs --time-index 0
```

`cf2ogc grid` and `cf2ogc vector` print the description as JSON; a selection
that is not on an axis exits with status 2.

The command line fills in its own defaults for keys the file leaves out:
`implementation: fast`, `chunk_size: 1000000`, `tile_size: 80`,
`parallel_workers: 8`, `extent_workers: 16`, `auto_tile_size: false`,
`use_numba: true`, `parallel_mode: numba` (`cf2ogc.points.load_config`).

## Benchmark

```sh
python scripts/benchmark_points.py -i cloud.nc -c config.yaml -o bench/ --use-numba
```

runs both implementations on one file and writes their times to
`bench/benchmark_report.json`.

## Development

```sh
pip install -e ".[test,numba,grid]"
pytest -q
```

The grid tests skip themselves when `osgeo` is missing; CI runs them in a job
with the system's GDAL (`.github/workflows/ci.yml`).
```
