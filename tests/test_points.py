# cf2ogc — CF-NetCDF to rasters an OGC service can serve
# Copyright (C) 2026 Norwegian Meteorological Institute (MET Norway)
# Author: Massimo Di Stefano <massimods@met.no>
# SPDX-License-Identifier: GPL-3.0-or-later

import copy
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import rasterio
import xarray as xr
import yaml

from cf2ogc.points import (
    Cancelled,
    _NUMBA_AVAILABLE,
    _build_output_overviews,
    _compute_extent,
    _get_spatial_extent,
    _overview_factors,
    grid_points,
    load_config,
    process_point_cloud,
    process_point_cloud_fast,
)


def _make_dataset():
    x = xr.DataArray(np.array([0.0, 2.0, 4.0], dtype=np.float32), dims=["point"])
    y = xr.DataArray(np.array([10.0, 5.0, -1.0], dtype=np.float32), dims=["point"])
    return xr.Dataset({"X": x, "Y": y})


def test_compute_extent_serial():
    ds = _make_dataset()
    cfg = {"processing": {"extent_engine": "serial"}}
    min_x, max_x, min_y, max_y = _compute_extent(ds, cfg)
    assert min_x == 0.0
    assert max_x == 4.0
    assert min_y == -1.0
    assert max_y == 10.0


def test_get_spatial_extent_auto_compute():
    ds = _make_dataset()
    cfg = {"processing": {"extent_strategy": "compute", "extent_engine": "serial"}}
    min_x, max_x, min_y, max_y = _get_spatial_extent(ds, cfg)
    assert (min_x, max_x, min_y, max_y) == (0.0, 4.0, -1.0, 10.0)


def _write_simple_netcdf(path: Path) -> None:
    ds = xr.Dataset(
        {
            "X": ("point", np.array([0.1, 0.9, 1.1, 1.9], dtype=np.float32)),
            "Y": ("point", np.array([0.1, 0.9, 1.1, 1.9], dtype=np.float32)),
            "red": ("point", np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)),
            "green": ("point", np.array([5.0, 6.0, 7.0, 8.0], dtype=np.float32)),
            "blue": ("point", np.array([9.0, 10.0, 11.0, 12.0], dtype=np.float32)),
        }
    )
    ds.to_netcdf(path)


def _read_tif(path: Path):
    with rasterio.open(path) as src:
        return src.read()


def test_fast_matches_classic(tmp_path):
    nc_path = tmp_path / "input.nc"
    _write_simple_netcdf(nc_path)
    classic_out = tmp_path / "classic.tif"
    fast_out = tmp_path / "fast.tif"

    config = {
        "grid": {"resolution": 1.0, "bounds_strategy": "rounded"},
        "processing": {"chunk_size": 2, "implementation": "classic", "extent_engine": "serial"},
        "output_bands": [
            {"variable": "red", "method": "mean", "nodata_val": -9999},
            {"variable": "green", "method": "mean", "nodata_val": -9999},
            {"variable": "blue", "method": "mean", "nodata_val": -9999},
        ],
    }

    process_point_cloud(config, str(nc_path), str(classic_out))
    config["processing"]["implementation"] = "fast"
    config["processing"]["parallel_mode"] = "none"
    process_point_cloud_fast(config, str(nc_path), str(fast_out))

    classic = _read_tif(classic_out)
    fast = _read_tif(fast_out)
    assert np.allclose(classic, fast, equal_nan=True)


def test_numba_parallel_path_smoke(tmp_path):
    if not _NUMBA_AVAILABLE:
        return
    nc_path = tmp_path / "input.nc"
    _write_simple_netcdf(nc_path)
    fast_out = tmp_path / "fast_numba.tif"

    config = {
        "grid": {"resolution": 1.0, "bounds_strategy": "rounded"},
        "processing": {
            "chunk_size": 2,
            "implementation": "fast",
            "parallel_mode": "numba",
            "parallel_workers": 2,
            "extent_engine": "serial",
        },
        "output_bands": [
            {"variable": "red", "method": "mean", "nodata_val": -9999},
        ],
    }

    process_point_cloud_fast(config, str(nc_path), str(fast_out))
    out = _read_tif(fast_out)
    assert out.shape[0] == 1


def test_overview_factors_only_for_large_rasters():
    assert _overview_factors(255, 255) == []
    assert _overview_factors(512, 512) == [2]
    assert _overview_factors(1024, 1024) == [2, 4]


def test_build_output_overviews_adds_internal_pyramids(tmp_path):
    tif_path = tmp_path / "overview.tif"
    data = np.ones((1024, 1024), dtype=np.float32)

    with rasterio.open(
        tif_path,
        "w",
        driver="GTiff",
        height=1024,
        width=1024,
        count=1,
        dtype="float32",
        crs="EPSG:4326",
        transform=rasterio.transform.from_origin(0, 1024, 1, 1),
        tiled=True,
        blockxsize=256,
        blockysize=256,
    ) as dst:
        dst.write(data, 1)
        _build_output_overviews(dst, 1024, 1024)

    with rasterio.open(tif_path) as src:
        assert src.overviews(1) == [2, 4]


# --- added with the move into the library -----------------------------------

_BASE_CONFIG = {
    "grid": {"resolution": 1.0, "bounds_strategy": "rounded"},
    "processing": {"chunk_size": 2, "extent_engine": "serial"},
    "output_bands": [{"variable": "red", "method": "mean", "nodata_val": -9999}],
}


def _config(**processing):
    cfg = copy.deepcopy(_BASE_CONFIG)
    cfg["processing"].update(processing)
    return cfg


def test_cancel_stops_classic_before_any_output(tmp_path):
    nc_path = tmp_path / "input.nc"
    _write_simple_netcdf(nc_path)
    out = tmp_path / "out.tif"
    calls = []

    def cancel():
        calls.append(1)
        return len(calls) >= 2  # let the first chunk run, stop at the second

    with pytest.raises(Cancelled, match="task revoked"):
        process_point_cloud(_config(), str(nc_path), str(out), cancel=cancel)
    assert len(calls) == 2
    assert not out.exists()


def test_cancelled_is_a_runtime_error():
    # Callers that caught RuntimeError("task revoked") before the move still do.
    assert issubclass(Cancelled, RuntimeError)


def test_cancel_stops_fast_numba_parallel(tmp_path):
    if not _NUMBA_AVAILABLE:
        pytest.skip("numba not installed")
    nc_path = tmp_path / "input.nc"
    _write_simple_netcdf(nc_path)
    with pytest.raises(Cancelled):
        process_point_cloud_fast(
            _config(parallel_mode="numba", parallel_workers=2),
            str(nc_path),
            str(tmp_path / "out.tif"),
            cancel=lambda: True,
        )


def test_a_failing_cancel_check_does_not_stop_the_run(tmp_path):
    nc_path = tmp_path / "input.nc"
    _write_simple_netcdf(nc_path)
    out = tmp_path / "out.tif"

    def broken():
        raise ConnectionError("broker down")

    process_point_cloud(_config(), str(nc_path), str(out), cancel=broken)
    assert out.exists()


def test_grid_points_defaults_to_classic(tmp_path, monkeypatch):
    import cf2ogc.points as points

    seen = []
    monkeypatch.setitem(points.IMPLEMENTATIONS, "classic", lambda *a, **k: seen.append("classic"))
    monkeypatch.setitem(points.IMPLEMENTATIONS, "fast", lambda *a, **k: seen.append("fast"))
    assert grid_points(_config(), "in.nc", "out.tif") == "classic"
    assert grid_points(_config(implementation="fast"), "in.nc", "out.tif") == "fast"
    assert grid_points(_config(implementation=" FAST "), "in.nc", "out.tif") == "fast"
    assert grid_points(_config(implementation="other"), "in.nc", "out.tif") == "classic"
    assert seen == ["classic", "fast", "fast", "classic"]


def test_grid_points_adds_no_defaults_to_the_config(tmp_path):
    nc_path = tmp_path / "input.nc"
    _write_simple_netcdf(nc_path)
    cfg = _config()
    before = yaml.safe_dump(cfg)
    grid_points(cfg, str(nc_path), str(tmp_path / "out.tif"))
    assert yaml.safe_dump(cfg) == before


def test_load_config_fills_the_command_line_defaults(tmp_path):
    path = tmp_path / "c.yaml"
    path.write_text(yaml.safe_dump({"grid": {"resolution": 1.0}}))
    proc = load_config(str(path))["processing"]
    assert proc == {
        "implementation": "fast",
        "chunk_size": 1000000,
        "tile_size": 80,
        "parallel_workers": 8,
        "extent_workers": 16,
        "auto_tile_size": False,
        "use_numba": True,
        "parallel_mode": "numba",
    }


def test_load_config_keeps_given_values(tmp_path):
    path = tmp_path / "c.yaml"
    path.write_text(yaml.safe_dump({"processing": {"implementation": "classic", "chunk_size": 500000}}))
    proc = load_config(str(path))["processing"]
    assert proc["implementation"] == "classic"
    assert proc["chunk_size"] == 500000


def _run_cli(*args, cwd):
    return subprocess.run(
        [sys.executable, "-m", "cf2ogc", *args], cwd=cwd, capture_output=True, text=True
    )


def test_cli_points_writes_the_same_raster_as_the_library(tmp_path):
    nc_path = tmp_path / "input.nc"
    _write_simple_netcdf(nc_path)
    cfg = _config(implementation="classic")
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))

    result = _run_cli("points", str(cfg_path), "-i", str(nc_path), "-o", str(tmp_path / "cli.tif"), cwd=tmp_path)
    assert result.returncode == 0, result.stderr

    process_point_cloud(_config(), str(nc_path), str(tmp_path / "lib.tif"))
    assert np.array_equal(_read_tif(tmp_path / "cli.tif"), _read_tif(tmp_path / "lib.tif"))


def test_cli_points_output_defaults_to_input_name(tmp_path):
    nc_path = tmp_path / "input.nc"
    _write_simple_netcdf(nc_path)
    cfg = _config(implementation="fast", parallel_mode="none")
    cfg["input_dataset"] = str(nc_path)
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))

    result = _run_cli("points", str(cfg_path), cwd=tmp_path)
    assert result.returncode == 0, result.stderr
    assert (tmp_path / "input.tif").exists()


def test_cli_points_errors(tmp_path):
    missing = _run_cli("points", str(tmp_path / "nope.yaml"), cwd=tmp_path)
    assert missing.returncode == 1
    assert "Config file not found" in missing.stderr

    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(yaml.safe_dump(_config()))
    no_input = _run_cli("points", str(cfg_path), cwd=tmp_path)
    assert no_input.returncode == 2
    assert "no input" in no_input.stderr
