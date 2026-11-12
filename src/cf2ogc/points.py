# cf2ogc — CF-NetCDF to rasters an OGC service can serve
# Copyright (C) 2026 Norwegian Meteorological Institute (MET Norway)
# Author: Massimo Di Stefano <massimods@met.no>
# SPDX-License-Identifier: GPL-3.0-or-later

"""Grid a CF point cloud (a 1-D ``point`` dimension with X, Y and value
variables) onto a regular raster and write it as a tiled GeoTIFF with
internal overviews.

Two implementations write the same raster: ``process_point_cloud`` (classic,
``np.add.at``) and ``process_point_cloud_fast`` (bincount, optional numba).
``grid_points`` picks one from ``config["processing"]["implementation"]``.

The functions take the configuration they are given and add no defaults of
their own; ``load_config`` (used by the command line) fills in the defaults of
the command line.
"""

from typing import Callable, Optional

import yaml
import os
import sys
import numpy as np
import xarray as xr
import rasterio
import time
from rasterio.transform import from_origin

from .io import open_netcdf_dataset
from rasterio.crs import CRS
from rasterio.enums import Resampling
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing as mp
from dask import compute as dask_compute

try:
    import numba as _numba
    from numba import njit, prange
    _NUMBA_AVAILABLE = True
except Exception:
    _NUMBA_AVAILABLE = False
    njit = None  # type: ignore[assignment]
    prange = None  # type: ignore[assignment]
    _numba = None  # type: ignore[assignment]

CancelCheck = Callable[[], bool]


class Cancelled(RuntimeError):
    """Raised when the caller's ``cancel`` callable returns true between chunks."""

    def __init__(self, message: str = "task revoked"):
        super().__init__(message)


def load_config(config_path):
    with open(config_path, 'r') as file:
        cfg = yaml.safe_load(file) or {}

    # Ensure processing defaults are present so callers can rely on keys
    proc = cfg.setdefault('processing', {})

    # User-requested defaults:
    # - use Numba-enabled fast implementation
    proc.setdefault('implementation', 'fast')
    proc.setdefault('chunk_size', 1000000)
    proc.setdefault('tile_size', 80)
    proc.setdefault('parallel_workers', 8)
    proc.setdefault('extent_workers', 16)
    proc.setdefault('auto_tile_size', False)
    proc.setdefault('use_numba', True)
    proc.setdefault('parallel_mode', 'numba')

    return cfg

def get_crs_from_netcdf(ds):
    """
    Attempts to extract the CRS WKT from the NetCDF CF-compliant 'crs' variable.
    """
    try:
        if 'crs' in ds.variables:
            wkt = ds['crs'].attrs.get('crs_wkt', None)
            if wkt:
                return CRS.from_wkt(wkt)
            
            epsg = ds['crs'].attrs.get('epsg_code', None)
            if epsg:
                return CRS.from_epsg(epsg)
    except Exception as e:
        print(f"Warning: Could not extract CRS from NetCDF. Error: {e}")
    
    print("Warning: No CRS found. GeoTIFF will not be projected.")
    return None


def _overview_factors(width: int, height: int) -> list[int]:
    """Return power-of-two overview factors for reasonably large rasters."""

    min_dim = min(int(width), int(height))
    factors: list[int] = []
    factor = 2
    while min_dim // factor >= 256:
        factors.append(factor)
        factor *= 2
    return factors


def _build_output_overviews(dst, width: int, height: int) -> None:
    """Build internal GeoTIFF overviews to speed up low-zoom rendering."""

    factors = _overview_factors(width, height)
    if not factors:
        return
    dst.build_overviews(factors, Resampling.average)
    dst.update_tags(ns='rio_overview', resampling='average')


def _extent_from_metadata(ds):
    """
    Try to read spatial extent from NetCDF metadata to avoid full scans.
    Returns (min_x, max_x, min_y, max_y, source) or None.
    """
    # CF-convention global attributes (commonly lon/lat)
    for x_key, y_key in (
        ("geospatial_lon_min", "geospatial_lat_min"),
        ("geospatial_x_min", "geospatial_y_min"),
    ):
        if x_key in ds.attrs and y_key in ds.attrs:
            try:
                min_x = float(ds.attrs[x_key])
                min_y = float(ds.attrs[y_key])
                max_x = float(ds.attrs.get(x_key.replace("_min", "_max")))
                max_y = float(ds.attrs.get(y_key.replace("_min", "_max")))
                return min_x, max_x, min_y, max_y, x_key
            except Exception:
                pass

    # Variable attributes on X/Y
    for var_name in ("X", "Y"):
        if var_name not in ds.variables:
            return None

    x_var = ds["X"]
    y_var = ds["Y"]
    for attr in ("actual_range", "valid_range", "min_max"):
        if attr in x_var.attrs and attr in y_var.attrs:
            try:
                x_vals = x_var.attrs[attr]
                y_vals = y_var.attrs[attr]
                if len(x_vals) >= 2 and len(y_vals) >= 2:
                    min_x, max_x = float(x_vals[0]), float(x_vals[1])
                    min_y, max_y = float(y_vals[0]), float(y_vals[1])
                    return min_x, max_x, min_y, max_y, attr
            except Exception:
                pass
    return None


def _compute_extent(ds, config=None):
    if config is None:
        config = {}
    workers = int(config.get("processing", {}).get("extent_workers", 4) or 4)
    extent_engine = str(config.get("processing", {}).get("extent_engine", "dask")).strip().lower()
    if extent_engine not in {"dask", "numba", "serial"}:
        extent_engine = "dask"
    print(f"Extent compute: using {workers} worker(s)")
    if extent_engine == "numba" and _NUMBA_AVAILABLE:
        try:
            if _numba is not None:
                _numba.set_num_threads(workers)
            x_vals = ds["X"].values
            y_vals = ds["Y"].values
            if x_vals.nbytes + y_vals.nbytes > 512 * 1024 * 1024:
                print("Warning: extent_engine=numba will load full X/Y arrays into memory (large dataset).")
            print("Extent compute: numba parallel reduction")
            min_x, max_x = _minmax_numba(x_vals.ravel())
            min_y, max_y = _minmax_numba(y_vals.ravel())
            print("Extent compute: numba reduction complete")
            return float(min_x), float(max_x), float(min_y), float(max_y)
        except Exception as exc:
            print(f"Extent compute: numba reduction failed ({exc}); falling back to dask")
    try:
        x = ds["X"]
        y = ds["Y"]
        # Prefer Dask parallel compute when available.
        if hasattr(x.data, "compute"):
            print("Extent compute: dask parallel reduction")
            scheduler = "processes"
            if mp.current_process().daemon:
                scheduler = "threads"
                print("Extent compute: running inside daemon process; switching to threads scheduler")
            min_x, max_x, min_y, max_y = dask_compute(
                x.min(),
                x.max(),
                y.min(),
                y.max(),
                scheduler=scheduler,
                num_workers=workers,
            )
            print("Extent compute: dask reduction complete")
            return float(min_x), float(max_x), float(min_y), float(max_y)
    except Exception as exc:
        if "daemonic processes are not allowed to have children" in str(exc):
            try:
                print("Extent compute: retrying with threads scheduler")
                min_x, max_x, min_y, max_y = dask_compute(
                    x.min(),
                    x.max(),
                    y.min(),
                    y.max(),
                    scheduler="threads",
                    num_workers=workers,
                )
                print("Extent compute: dask threads reduction complete")
                return float(min_x), float(max_x), float(min_y), float(max_y)
            except Exception as exc2:
                print(f"Extent compute: dask threads reduction failed ({exc2}); falling back to serial compute")
        else:
            print(f"Extent compute: dask reduction failed ({exc}); falling back to serial compute")
        pass
    min_x = ds["X"].min().values
    max_x = ds["X"].max().values
    min_y = ds["Y"].min().values
    max_y = ds["Y"].max().values
    return float(min_x), float(max_x), float(min_y), float(max_y)


def _get_spatial_extent(ds, config):
    strategy = str(config.get("processing", {}).get("extent_strategy", "auto")).strip().lower()
    if strategy not in {"auto", "metadata", "compute"}:
        strategy = "auto"

    if strategy in {"auto", "metadata"}:
        meta = _extent_from_metadata(ds)
        if meta is not None:
            min_x, max_x, min_y, max_y, source = meta
            # Avoid using lon/lat metadata when the dataset is clearly projected.
            if strategy == "auto" and source.startswith("geospatial_lon"):
                crs = get_crs_from_netcdf(ds)
                if crs is not None and crs.to_epsg() not in (None, 4326):
                    return _compute_extent(ds)
            return min_x, max_x, min_y, max_y
        if strategy == "metadata":
            print("Warning: extent_strategy=metadata but no metadata extent found; falling back to compute.")

    return _compute_extent(ds, config)

def _cancelled(cancel: Optional[CancelCheck]) -> bool:
    # A failing check must not kill a long run: the old in-process check
    # treated any error as "not revoked", and so does this one.
    if cancel is None:
        return False
    try:
        return bool(cancel())
    except Exception:
        return False


def process_point_cloud(config, input_path, output_path, *, cancel: Optional[CancelCheck] = None):
    """Classic implementation. ``cancel``, when given, is called before each
    chunk; a true result stops the run with :class:`Cancelled`."""
    print(f"Opening {input_path}...")
    
    # Lazy load the dataset with chunking
    ds = open_netcdf_dataset(input_path, chunks={'point': config['processing']['chunk_size']})
    force_chunking = bool(config.get("processing", {}).get("force_chunking", True))
    if not getattr(ds, "chunks", None):
        if force_chunking:
            print("Warning: dataset is not chunked; applying chunking on 'point'.")
            ds = ds.chunk({"point": config["processing"]["chunk_size"]})
        else:
            print("Warning: dataset is not chunked; proceeding without chunking.")
    else:
        print(f"Dataset chunks: {ds.chunks}")
    
    # 1. Determine Grid Bounds
    print("Calculating spatial extents...")
    t0 = time.perf_counter()
    min_x, max_x, min_y, max_y = _get_spatial_extent(ds, config)
    print(f"Extent compute: {time.perf_counter() - t0:.2f}s")

    res = config['grid']['resolution']
    
    # Align bounds to resolution if requested
    if config['grid']['bounds_strategy'] == 'rounded':
        min_x = np.floor(min_x / res) * res
        max_y = np.ceil(max_y / res) * res

    # Calculate grid dimensions
    width = int(np.ceil((max_x - min_x) / res))
    height = int(np.ceil((max_y - min_y) / res))
    
    print(f"Grid Size: {width} x {height} pixels")

    # Define Affine Transform (Top-Left convention)
    transform = from_origin(min_x, max_y, res, res)

    # 2. Initialize Accumulators
    bands_config = config['output_bands']
    num_bands = len(bands_config)
    
    # Accumulators
    # Float64 used during accumulation for precision, converted to Float32 at write time
    grid_sums = np.zeros((num_bands, height, width), dtype=np.float64)
    # Use int64 to match np.bincount output without casting issues.
    grid_counts = np.zeros((num_bands, height, width), dtype=np.int64)
    
    # Initialize min/max grids
    grid_mins = np.full((num_bands, height, width), np.inf, dtype=np.float64)
    grid_maxs = np.full((num_bands, height, width), -np.inf, dtype=np.float64)

    # 3. Process Chunks
    total_points = ds.sizes['point']
    chunk_size = config['processing']['chunk_size']
    
    print(f"Rasterizing {total_points} points in chunks...")

    for i in range(0, total_points, chunk_size):
        if _cancelled(cancel):
            print("[process_point_cloud] cancelled by the caller; aborting")
            raise Cancelled()
        subset = ds.isel(point=slice(i, i + chunk_size))
        
        # Load spatial coordinates
        xs = subset['X'].values
        ys = subset['Y'].values
        
        # Calculate pixel indices
        cols = ((xs - min_x) / res).astype(np.int32)
        rows = ((max_y - ys) / res).astype(np.int32)
        
        # Boundary filter
        valid_mask = (cols >= 0) & (cols < width) & (rows >= 0) & (rows < height)
        
        if not np.any(valid_mask):
            continue

        rows = rows[valid_mask]
        cols = cols[valid_mask]
        flat_indices = rows * width + cols
        
        for band_idx, band_cfg in enumerate(bands_config):
            var_name = band_cfg['variable']
            method = band_cfg['method']
            
            if var_name not in subset:
                print(f"Skipping missing variable: {var_name}")
                continue
            
            # --- UNIVERSAL DATA HANDLING ---
            # Get the variable (lazy load)
            data_var = subset[var_name]

            # 1. Handle Hyperspectral Band Selection
            if 'band_selection' in band_cfg:
                target = band_cfg['band_selection']
                try:
                    # 'method=nearest' ensures we find the closest wavelength if exact match fails
                    data_var = data_var.sel(band=target, method='nearest')
                except Exception as e:
                    print(f"Error selecting band {target} for {var_name}: {e}")
                    continue
            
            # 2. Dimensionality Check
            # By this point, the data MUST be 1D (only dependent on 'point').
            # If it still has 2 dimensions (e.g. point, band), the config is missing 'band_selection'.
            if len(data_var.dims) > 1:
                print(f"Error: Variable '{var_name}' is multi-dimensional {data_var.dims}. "
                      f"Please add 'band_selection' to config. Skipping.")
                continue

            # 3. Load values for this chunk
            values = data_var.values[valid_mask]
            
            # 4. Aggregate
            if method == 'mean':
                np.add.at(grid_sums[band_idx].ravel(), flat_indices, values)
                np.add.at(grid_counts[band_idx].ravel(), flat_indices, 1)
            elif method == 'count':
                np.add.at(grid_sums[band_idx].ravel(), flat_indices, 1)
            elif method == 'max':
                np.maximum.at(grid_maxs[band_idx].ravel(), flat_indices, values)
                np.add.at(grid_counts[band_idx].ravel(), flat_indices, 1) 
            elif method == 'min':
                np.minimum.at(grid_mins[band_idx].ravel(), flat_indices, values)
                np.add.at(grid_counts[band_idx].ravel(), flat_indices, 1) 
        
        sys.stdout.write(f"\rProcessed {min(i + chunk_size, total_points)} / {total_points}")
        sys.stdout.flush()
    
    print("\nFinalizing grid...")

    # 4. Finalize Aggregation
    final_raster = np.zeros((num_bands, height, width), dtype=np.float32)
    
    for band_idx, band_cfg in enumerate(bands_config):
        method = band_cfg['method']
        nodata = band_cfg.get('nodata_val', 0)
        
        counts = grid_counts[band_idx]
        has_data_mask = counts > 0
        
        if method == 'mean':
            final_raster[band_idx][has_data_mask] = (
                grid_sums[band_idx][has_data_mask] / counts[has_data_mask]
            )
        elif method == 'count':
            final_raster[band_idx] = grid_sums[band_idx]
        elif method == 'max':
            final_raster[band_idx][has_data_mask] = grid_maxs[band_idx][has_data_mask]
        elif method == 'min':
            final_raster[band_idx][has_data_mask] = grid_mins[band_idx][has_data_mask]

        # Apply transparency (No Interpolation)
        final_raster[band_idx][~has_data_mask] = nodata

    # 5. Write GeoTIFF
    crs = get_crs_from_netcdf(ds)
    global_nodata = bands_config[0].get('nodata_val', 0)

    print(f"Writing to {output_path}...")
    
    meta = {
        'driver': 'GTiff',
        'height': height,
        'width': width,
        'count': num_bands,
        'dtype': 'float32',
        'crs': crs,
        'transform': transform,
        'compress': 'lzw',
        'tiled': True,
        'blockxsize': 256,
        'blockysize': 256,
        'nodata': global_nodata
    }

    with rasterio.open(output_path, 'w', **meta) as dst:
        for i in range(num_bands):
            dst.write(final_raster[i], i + 1)
            
            # Metadata Description
            var_name = bands_config[i]['variable']
            desc = f"{var_name}"
            
            if 'band_selection' in bands_config[i]:
                desc += f" ({bands_config[i]['band_selection']}nm)"
                
            dst.set_band_description(i + 1, desc)
        _build_output_overviews(dst, width, height)

    print(f"Done. Saved to {output_path}")


if _NUMBA_AVAILABLE:
    @njit(cache=True)
    def _accumulate_numba(
        rows,
        cols,
        values,
        width,
        grid_sums,
        grid_counts,
        grid_mins,
        grid_maxs,
        method_code,
    ):
        for i in range(rows.shape[0]):
            idx = rows[i] * width + cols[i]
            v = values[i]
            if method_code == 0:  # mean
                grid_sums[idx] += v
                grid_counts[idx] += 1
            elif method_code == 1:  # count
                grid_sums[idx] += 1.0
                grid_counts[idx] += 1
            elif method_code == 2:  # max
                if v > grid_maxs[idx]:
                    grid_maxs[idx] = v
                grid_counts[idx] += 1
            elif method_code == 3:  # min
                if v < grid_mins[idx]:
                    grid_mins[idx] = v
                grid_counts[idx] += 1

    @njit(cache=True, parallel=True)
    def _accumulate_tiles_numba_parallel(
        tile_starts,
        tile_counts,
        rows,
        cols,
        values,
        width,
        grid_sums,
        grid_counts,
        grid_mins,
        grid_maxs,
        method_code,
    ):
        for t in prange(tile_starts.shape[0]):
            start = tile_starts[t]
            count = tile_counts[t]
            end = start + count
            for i in range(start, end):
                idx = rows[i] * width + cols[i]
                v = values[i]
                if method_code == 0:  # mean
                    grid_sums[idx] += v
                    grid_counts[idx] += 1
                elif method_code == 1:  # count
                    grid_sums[idx] += 1.0
                    grid_counts[idx] += 1
                elif method_code == 2:  # max
                    if v > grid_maxs[idx]:
                        grid_maxs[idx] = v
                    grid_counts[idx] += 1
                elif method_code == 3:  # min
                    if v < grid_mins[idx]:
                        grid_mins[idx] = v
                    grid_counts[idx] += 1

    @njit(cache=True, parallel=True)
    def _minmax_numba(values):
        n = values.size
        nthreads = _numba.get_num_threads()
        mins = np.full(nthreads, np.inf)
        maxs = np.full(nthreads, -np.inf)
        for i in prange(n):
            v = values[i]
            if np.isnan(v):
                continue
            tid = _numba.get_thread_id()
            if v < mins[tid]:
                mins[tid] = v
            if v > maxs[tid]:
                maxs[tid] = v
        minv = np.inf
        maxv = -np.inf
        for i in range(nthreads):
            if mins[i] < minv:
                minv = mins[i]
            if maxs[i] > maxv:
                maxv = maxs[i]
        return minv, maxv


def process_point_cloud_fast(config, input_path, output_path, *, cancel: Optional[CancelCheck] = None):
    """
    Faster point-cloud rasterization using vectorized bincount reductions, with optional Numba JIT.
    Enable with config.processing.implementation = "fast".
    Optionally set config.processing.use_numba = true to use the JIT accumulator when available.
    ``cancel`` is checked before each chunk in the numba-parallel mode only, as
    before the move.
    """
    print(f"Opening {input_path} (fast)...")

    ds = open_netcdf_dataset(input_path, chunks={'point': config['processing']['chunk_size']})
    force_chunking = bool(config.get("processing", {}).get("force_chunking", True))
    if not getattr(ds, "chunks", None):
        if force_chunking:
            print("Warning: dataset is not chunked; applying chunking on 'point'.")
            ds = ds.chunk({"point": config["processing"]["chunk_size"]})
        else:
            print("Warning: dataset is not chunked; proceeding without chunking.")
    else:
        print(f"Dataset chunks: {ds.chunks}")

    print("Calculating spatial extents...")
    t0 = time.perf_counter()
    min_x, max_x, min_y, max_y = _get_spatial_extent(ds, config)
    print(f"Extent compute: {time.perf_counter() - t0:.2f}s")

    res = config['grid']['resolution']

    if config['grid']['bounds_strategy'] == 'rounded':
        min_x = np.floor(min_x / res) * res
        max_y = np.ceil(max_y / res) * res

    width = int(np.ceil((max_x - min_x) / res))
    height = int(np.ceil((max_y - min_y) / res))

    print(f"Grid Size: {width} x {height} pixels")

    transform = from_origin(min_x, max_y, res, res)

    bands_config = config['output_bands']
    num_bands = len(bands_config)

    grid_sums = np.zeros((num_bands, height, width), dtype=np.float64)
    # Use int64 to avoid casting issues with np.add.at and bincount outputs.
    grid_counts = np.zeros((num_bands, height, width), dtype=np.int64)
    grid_mins = np.full((num_bands, height, width), np.inf, dtype=np.float64)
    grid_maxs = np.full((num_bands, height, width), -np.inf, dtype=np.float64)

    total_points = ds.sizes['point']
    chunk_size = config['processing']['chunk_size']

    use_numba = bool(config.get('processing', {}).get('use_numba', False)) and _NUMBA_AVAILABLE

    parallel_workers = int(config.get('processing', {}).get('parallel_workers', 0) or 0)
    if parallel_workers <= 0:
        parallel_workers = max(1, (os.cpu_count() or 1))
    parallel_mode = str(config.get('processing', {}).get('parallel_mode', 'threads')).strip().lower()
    if parallel_mode not in {"threads", "numba", "none"}:
        parallel_mode = "threads"

    def _process_chunk(i0: int, i1: int):
        subset = ds.isel(point=slice(i0, i1))
        xs = subset['X'].values
        ys = subset['Y'].values
        cols = ((xs - min_x) / res).astype(np.int32)
        rows = ((max_y - ys) / res).astype(np.int32)
        valid_mask = (cols >= 0) & (cols < width) & (rows >= 0) & (rows < height)
        if not np.any(valid_mask):
            return None
        rows = rows[valid_mask]
        cols = cols[valid_mask]
        flat_indices = rows * width + cols
        count_bc = np.bincount(flat_indices)
        count_idx = np.nonzero(count_bc)[0]
        band_updates = []
        for band_idx, band_cfg in enumerate(bands_config):
            var_name = band_cfg['variable']
            method = band_cfg['method']
            if var_name not in subset:
                band_updates.append(None)
                continue
            data_var = subset[var_name]
            if 'band_selection' in band_cfg:
                target = band_cfg['band_selection']
                try:
                    data_var = data_var.sel(band=target, method='nearest')
                except Exception:
                    band_updates.append(None)
                    continue
            if len(data_var.dims) > 1:
                band_updates.append(None)
                continue
            values = data_var.values[valid_mask]
            if use_numba:
                method_code = {"mean": 0, "count": 1, "max": 2, "min": 3}.get(method)
                if method_code is None:
                    band_updates.append(None)
                    continue
                band_updates.append(
                    ("numba", method_code, rows, cols, values.astype(np.float64))
                )
            else:
                if method == 'mean':
                    sum_bc = np.bincount(flat_indices, weights=values)
                    sum_idx = np.nonzero(sum_bc)[0]
                    band_updates.append(("mean", sum_idx, sum_bc, count_idx, count_bc))
                elif method == 'count':
                    band_updates.append(("count", count_idx, count_bc))
                elif method == 'max':
                    band_updates.append(("max", flat_indices, values, count_idx, count_bc))
                elif method == 'min':
                    band_updates.append(("min", flat_indices, values, count_idx, count_bc))
                else:
                    band_updates.append(None)
        return band_updates

    print(f"Rasterizing {total_points} points in chunks...")
    raster_start = time.perf_counter()

    if parallel_mode == "numba":
        if not _NUMBA_AVAILABLE:
            print("Warning: parallel_mode=numba requested but numba is unavailable; falling back to threads.")
            parallel_mode = "threads"
        else:
            use_numba = True

    if use_numba:
        if _numba is not None:
            try:
                ver = _numba.__version__
            except Exception:
                ver = "unknown"
            print(f"Numba JIT enabled for accumulation (numba {ver}).")
        else:
            print("Numba JIT enabled for accumulation.")
        if parallel_mode == "numba" and _numba is not None:
            try:
                _numba.set_num_threads(parallel_workers)
                print(f"Numba threads set to {parallel_workers}")
            except Exception:
                pass

    if parallel_mode == "numba" and parallel_workers > 1:
        print(f"Using numba-parallel tiles with {parallel_workers} workers.")
        for i in range(0, total_points, chunk_size):
            if _cancelled(cancel):
                print("[process_point_cloud_fast] cancelled by the caller; aborting")
                raise Cancelled()
            subset = ds.isel(point=slice(i, i + chunk_size))
            xs = subset['X'].values
            ys = subset['Y'].values
            cols = ((xs - min_x) / res).astype(np.int32)
            rows = ((max_y - ys) / res).astype(np.int32)
            valid_mask = (cols >= 0) & (cols < width) & (rows >= 0) & (rows < height)
            if not np.any(valid_mask):
                continue
            rows = rows[valid_mask]
            cols = cols[valid_mask]

            # Tile grouping to avoid write collisions in parallel.
            tile_size = int(config.get('processing', {}).get('tile_size', 512) or 512)
            auto_tile = bool(config.get('processing', {}).get('auto_tile_size', False))
            if auto_tile:
                candidates = [128, 256, 512, 1024]
                target = max(2, parallel_workers) * 2
                chosen = tile_size
                for cand in candidates:
                    tiles_x_c = max(1, int(np.ceil(width / cand)))
                    tiles_y_c = max(1, int(np.ceil(height / cand)))
                    if tiles_x_c * tiles_y_c >= target:
                        chosen = cand
                        break
                if chosen != tile_size:
                    tile_size = chosen
                    print(f"Auto tile size enabled: using tile_size={tile_size} to target >= {target} tiles")
            tiles_x = max(1, int(np.ceil(width / tile_size)))
            tile_ids = (rows // tile_size) * tiles_x + (cols // tile_size)
            order = np.argsort(tile_ids)
            tile_ids_sorted = tile_ids[order]
            rows_sorted = rows[order]
            cols_sorted = cols[order]

            _, tile_starts, tile_counts = np.unique(
                tile_ids_sorted, return_index=True, return_counts=True
            )
            if i == 0:
                tile_count = len(tile_starts)
                print(
                    f"Numba tiles: tile_size={tile_size} tiles_x={tiles_x} tiles={tile_count}"
                )
                if tile_count < parallel_workers:
                    print(
                        "Warning: tile count is lower than worker count; reduce tile_size, enable auto_tile_size, "
                        "or reduce parallel_workers for better parallelism."
                    )

            for band_idx, band_cfg in enumerate(bands_config):
                var_name = band_cfg['variable']
                method = band_cfg['method']
                if var_name not in subset:
                    continue
                data_var = subset[var_name]
                if 'band_selection' in band_cfg:
                    target = band_cfg['band_selection']
                    try:
                        data_var = data_var.sel(band=target, method='nearest')
                    except Exception:
                        continue
                if len(data_var.dims) > 1:
                    continue
                values = data_var.values[valid_mask]
                values_sorted = values[order].astype(np.float64)
                method_code = {"mean": 0, "count": 1, "max": 2, "min": 3}.get(method)
                if method_code is None:
                    continue
                if i == 0 and band_idx == 0:
                    # Warm up JIT on first pass to avoid skewing timings.
                    _accumulate_tiles_numba_parallel(
                        tile_starts[:1],
                        tile_counts[:1],
                        rows_sorted[:1],
                        cols_sorted[:1],
                        values_sorted[:1],
                        width,
                        grid_sums[band_idx].ravel(),
                        grid_counts[band_idx].ravel(),
                        grid_mins[band_idx].ravel(),
                        grid_maxs[band_idx].ravel(),
                        method_code,
                    )
                _accumulate_tiles_numba_parallel(
                    tile_starts,
                    tile_counts,
                    rows_sorted,
                    cols_sorted,
                    values_sorted,
                    width,
                    grid_sums[band_idx].ravel(),
                    grid_counts[band_idx].ravel(),
                    grid_mins[band_idx].ravel(),
                    grid_maxs[band_idx].ravel(),
                    method_code,
                )

            sys.stdout.write(f"\rProcessed {min(i + chunk_size, total_points)} / {total_points}")
            sys.stdout.flush()

    elif parallel_workers > 1 and parallel_mode == "threads":
        print(f"Using parallel workers: {parallel_workers}")
        futures = []
        with ThreadPoolExecutor(max_workers=parallel_workers) as executor:
            for i in range(0, total_points, chunk_size):
                futures.append(executor.submit(_process_chunk, i, min(i + chunk_size, total_points)))
            completed = 0
            for future in as_completed(futures):
                updates = future.result()
                completed += 1
                if updates is None:
                    continue
                for band_idx, upd in enumerate(updates):
                    if upd is None:
                        continue
                    if upd[0] == "numba":
                        _, method_code, rows, cols, values = upd
                        _accumulate_numba(
                            rows,
                            cols,
                            values,
                            width,
                            grid_sums[band_idx].ravel(),
                            grid_counts[band_idx].ravel(),
                            grid_mins[band_idx].ravel(),
                            grid_maxs[band_idx].ravel(),
                            method_code,
                        )
                    elif upd[0] == "mean":
                        _, sum_idx, sum_bc, count_idx, count_bc = upd
                        grid_sums[band_idx].ravel()[sum_idx] += sum_bc[sum_idx]
                        grid_counts[band_idx].ravel()[count_idx] += count_bc[count_idx]
                    elif upd[0] == "count":
                        _, count_idx, count_bc = upd
                        grid_sums[band_idx].ravel()[count_idx] += count_bc[count_idx]
                        grid_counts[band_idx].ravel()[count_idx] += count_bc[count_idx]
                    elif upd[0] == "max":
                        _, flat_indices, values, count_idx, count_bc = upd
                        np.maximum.at(grid_maxs[band_idx].ravel(), flat_indices, values)
                        grid_counts[band_idx].ravel()[count_idx] += count_bc[count_idx]
                    elif upd[0] == "min":
                        _, flat_indices, values, count_idx, count_bc = upd
                        np.minimum.at(grid_mins[band_idx].ravel(), flat_indices, values)
                        grid_counts[band_idx].ravel()[count_idx] += count_bc[count_idx]
            sys.stdout.write(f"\rProcessed {min(completed * chunk_size, total_points)} / {total_points}")
            sys.stdout.flush()
    else:
        for i in range(0, total_points, chunk_size):
            updates = _process_chunk(i, i + chunk_size)
            if updates is None:
                continue
            for band_idx, upd in enumerate(updates):
                if upd is None:
                    continue
                if upd[0] == "numba":
                    _, method_code, rows, cols, values = upd
                    _accumulate_numba(
                        rows,
                        cols,
                        values,
                        width,
                        grid_sums[band_idx].ravel(),
                        grid_counts[band_idx].ravel(),
                        grid_mins[band_idx].ravel(),
                        grid_maxs[band_idx].ravel(),
                        method_code,
                    )
                elif upd[0] == "mean":
                    _, sum_idx, sum_bc, count_idx, count_bc = upd
                    grid_sums[band_idx].ravel()[sum_idx] += sum_bc[sum_idx]
                    grid_counts[band_idx].ravel()[count_idx] += count_bc[count_idx]
                elif upd[0] == "count":
                    _, count_idx, count_bc = upd
                    grid_sums[band_idx].ravel()[count_idx] += count_bc[count_idx]
                    grid_counts[band_idx].ravel()[count_idx] += count_bc[count_idx]
                elif upd[0] == "max":
                    _, flat_indices, values, count_idx, count_bc = upd
                    np.maximum.at(grid_maxs[band_idx].ravel(), flat_indices, values)
                    grid_counts[band_idx].ravel()[count_idx] += count_bc[count_idx]
                elif upd[0] == "min":
                    _, flat_indices, values, count_idx, count_bc = upd
                    np.minimum.at(grid_mins[band_idx].ravel(), flat_indices, values)
                    grid_counts[band_idx].ravel()[count_idx] += count_bc[count_idx]
            sys.stdout.write(f"\rProcessed {min(i + chunk_size, total_points)} / {total_points}")
            sys.stdout.flush()

    print(f"\nRasterization: {time.perf_counter() - raster_start:.2f}s")

    print("\nFinalizing grid...")

    final_raster = np.zeros((num_bands, height, width), dtype=np.float32)
    for band_idx, band_cfg in enumerate(bands_config):
        method = band_cfg['method']
        nodata = band_cfg.get('nodata_val', 0)

        counts = grid_counts[band_idx]
        has_data_mask = counts > 0

        if method == 'mean':
            final_raster[band_idx][has_data_mask] = (
                grid_sums[band_idx][has_data_mask] / counts[has_data_mask]
            )
        elif method == 'count':
            final_raster[band_idx] = grid_sums[band_idx]
        elif method == 'max':
            final_raster[band_idx][has_data_mask] = grid_maxs[band_idx][has_data_mask]
        elif method == 'min':
            final_raster[band_idx][has_data_mask] = grid_mins[band_idx][has_data_mask]

        final_raster[band_idx][~has_data_mask] = nodata

    crs = get_crs_from_netcdf(ds)
    global_nodata = bands_config[0].get('nodata_val', 0)

    print(f"Writing to {output_path}...")

    meta = {
        'driver': 'GTiff',
        'height': height,
        'width': width,
        'count': num_bands,
        'dtype': 'float32',
        'crs': crs,
        'transform': transform,
        'compress': 'lzw',
        'tiled': True,
        'blockxsize': 256,
        'blockysize': 256,
        'nodata': global_nodata
    }

    with rasterio.open(output_path, 'w', **meta) as dst:
        for i in range(num_bands):
            dst.write(final_raster[i], i + 1)
            var_name = bands_config[i]['variable']
            desc = f"{var_name}"
            if 'band_selection' in bands_config[i]:
                desc += f" ({bands_config[i]['band_selection']}nm)"
            dst.set_band_description(i + 1, desc)
        _build_output_overviews(dst, width, height)

    print(f"Done. Saved to {output_path}")


IMPLEMENTATIONS = {
    "classic": process_point_cloud,
    "fast": process_point_cloud_fast,
}


def grid_points(config, input_path, output_path, *, cancel: Optional[CancelCheck] = None):
    """Run the implementation named by ``config["processing"]["implementation"]``.

    ``"fast"`` runs :func:`process_point_cloud_fast`; anything else, or no key,
    runs :func:`process_point_cloud`, the rule the service has always applied.
    """
    implementation = str(config.get("processing", {}).get("implementation", "classic")).strip().lower()
    if implementation not in IMPLEMENTATIONS:
        implementation = "classic"
    IMPLEMENTATIONS[implementation](config, input_path, output_path, cancel=cancel)
    return implementation
