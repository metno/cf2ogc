# cf2ogc

CF-compliant NetCDF in, rasters an OGC service (WMS, WCS) can serve out.

Version 0.1.0 has one converter, `cf2ogc.points`. It takes a point cloud stored
along a single `point` dimension and grids it onto a regular raster, written as
a tiled, LZW-compressed Float32 GeoTIFF with internal overviews. Airborne
hyperspectral (HySpex) point clouds are the first use, but the code has nothing
HySpex-specific in it: any NetCDF with `X`, `Y` and value variables along
`point` works.

Licence: GPL-3.0-or-later (see `LICENSE`).
© 2026 Norwegian Meteorological Institute (MET Norway).

## Install

```sh
pip install "cf2ogc @ git+https://github.com/metno/cf2ogc@v0.1.0"
# with the numba JIT accumulators used by the fast implementation:
pip install "cf2ogc[numba] @ git+https://github.com/metno/cf2ogc@v0.1.0"
```

Dependencies: numpy, xarray, dask, netCDF4, rasterio, PyYAML; numba is optional.
Python 3.10 or newer.

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
```

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
pip install -e ".[test,numba]"
pytest -q
```
