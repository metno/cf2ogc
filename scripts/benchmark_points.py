# cf2ogc — CF-NetCDF to rasters an OGC service can serve
# Copyright (C) 2026 Norwegian Meteorological Institute (MET Norway)
# Author: Massimo Di Stefano <massimods@met.no>
# SPDX-License-Identifier: GPL-3.0-or-later

import argparse
import json
import time
from pathlib import Path

from cf2ogc.points import load_config, process_point_cloud, process_point_cloud_fast


def _run(label: str, fn, config: dict, input_path: str, output_path: Path) -> float:
    start = time.perf_counter()
    fn(config, input_path, str(output_path))
    elapsed = time.perf_counter() - start
    print(f"{label}: {elapsed:.2f}s -> {output_path}")
    return elapsed


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark NetCDF -> GeoTIFF processing implementations.")
    parser.add_argument("-i", "--input", required=True, help="Input NetCDF/OPeNDAP URL")
    parser.add_argument("-c", "--config", required=True, help="Config YAML used for conversion")
    parser.add_argument("-o", "--output-dir", required=True, help="Directory to write output GeoTIFFs")
    parser.add_argument("--skip-classic", action="store_true", help="Skip the classic implementation")
    parser.add_argument("--skip-fast", action="store_true", help="Skip the fast implementation")
    parser.add_argument("--use-numba", action="store_true", help="Enable numba JIT in fast implementation")
    parser.add_argument("--chunk-size", type=int, help="Override processing.chunk_size")
    parser.add_argument("--parallel-mode", choices=["threads", "numba", "none"], help="Override processing.parallel_mode")
    parser.add_argument("--parallel-workers", type=int, help="Override processing.parallel_workers")
    parser.add_argument("--tile-size", type=int, help="Override processing.tile_size (numba mode)")
    parser.add_argument("--extent-strategy", choices=["auto", "metadata", "compute"], help="Override processing.extent_strategy")
    parser.add_argument("--extent-engine", choices=["dask", "numba", "serial"], help="Override processing.extent_engine")
    parser.add_argument("--extent-workers", type=int, help="Override processing.extent_workers")
    parser.add_argument("--force-chunking", action="store_true", help="Force chunking on 'point'")
    parser.add_argument("--no-force-chunking", action="store_true", help="Disable force chunking")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = load_config(args.config)
    config = dict(config or {})
    config.setdefault("processing", {})
    if not isinstance(config["processing"], dict):
        config["processing"] = {}
    processing = config["processing"]
    processing["use_numba"] = bool(args.use_numba)
    if args.chunk_size:
        processing["chunk_size"] = int(args.chunk_size)
    if args.parallel_mode:
        processing["parallel_mode"] = args.parallel_mode
    if args.parallel_workers:
        processing["parallel_workers"] = int(args.parallel_workers)
    if args.tile_size:
        processing["tile_size"] = int(args.tile_size)
    if args.extent_strategy:
        processing["extent_strategy"] = args.extent_strategy
    if args.extent_engine:
        processing["extent_engine"] = args.extent_engine
    if args.extent_workers:
        processing["extent_workers"] = int(args.extent_workers)
    if args.force_chunking:
        processing["force_chunking"] = True
    if args.no_force_chunking:
        processing["force_chunking"] = False

    stats: dict[str, float] = {}
    input_path = args.input

    if not args.skip_classic:
        processing["implementation"] = "classic"
        classic_out = output_dir / "classic.tif"
        stats["classic"] = _run("classic", process_point_cloud, config, input_path, classic_out)

    if not args.skip_fast:
        processing["implementation"] = "fast"
        fast_out = output_dir / "fast.tif"
        stats["fast"] = _run("fast", process_point_cloud_fast, config, input_path, fast_out)

    report = output_dir / "benchmark_report.json"
    report.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(f"Report written to {report}")


if __name__ == "__main__":
    main()
