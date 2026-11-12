# cf2ogc — CF-NetCDF to rasters an OGC service can serve
# Copyright (C) 2026 Norwegian Meteorological Institute (MET Norway)
# Author: Massimo Di Stefano <massimods@met.no>
# SPDX-License-Identifier: GPL-3.0-or-later

"""The ``cf2ogc`` command line."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from . import __version__


def _points(args: argparse.Namespace) -> int:
    from .points import grid_points, load_config

    if not os.path.exists(args.config):
        print(f"Error: Config file not found: {args.config}", file=sys.stderr)
        return 1

    # The command line's defaults (implementation "fast", numba, ...) come from
    # load_config; the library functions add none of their own.
    config = load_config(args.config)
    if args.implementation:
        config["processing"]["implementation"] = args.implementation

    input_dataset = args.input or config.get("input_dataset")
    if not input_dataset:
        print("Error: no input: pass -i/--input or set input_dataset in the config", file=sys.stderr)
        return 2

    # Output: the -o argument, then the config's output_file, then the input
    # name with .tif.
    if args.output:
        output_file = args.output
    elif config.get("output_file"):
        output_file = config["output_file"]
    else:
        output_file = str(Path(input_dataset).with_suffix(".tif"))

    grid_points(config, input_dataset, output_file)
    return 0


def _grid(args: argparse.Namespace) -> int:
    import json

    from .grid import GridError, convert

    try:
        described = convert(
            args.source,
            args.variable,
            args.output,
            time=args.time,
            time_index=args.time_index,
            level=args.level,
            level_index=args.level_index,
            nodata=args.nodata,
            target_epsg=args.target_epsg,
            resolution=args.resolution,
            driver=args.driver,
        )
    except GridError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(described, indent=2))
    return 0


def _vector(args: argparse.Namespace) -> int:
    import json

    from .grid import GridError
    from .vector import convert

    try:
        described = convert(
            args.source,
            args.output,
            u=args.u,
            v=args.v,
            direction=args.direction,
            magnitude=args.magnitude,
            convention=args.convention,
            frame=args.frame,
            time=args.time,
            time_index=args.time_index,
            level=args.level,
            level_index=args.level_index,
            nodata=args.nodata,
            target_epsg=args.target_epsg,
            resolution=args.resolution,
            driver=args.driver,
            components=args.components,
        )
    except GridError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(described, indent=2))
    return 0


def _slice_options(parser: argparse.ArgumentParser) -> None:
    when = parser.add_mutually_exclusive_group()
    when.add_argument("--time", help="The time step, as an ISO 8601 stamp on the axis")
    when.add_argument("--time-index", type=int, help="The time step, as an index")
    where = parser.add_mutually_exclusive_group()
    where.add_argument("--level", type=float, help="The vertical level, as a coordinate value")
    where.add_argument("--level-index", type=int, help="The vertical level, as an index")
    parser.add_argument("--nodata", type=float, help="Replace the variable's declared fill as the output's nodata")
    parser.add_argument("--target-epsg", type=int, help="Reproject to this EPSG code (default: keep the source grid)")
    parser.add_argument("--resolution", type=float, help="Output pixel size, in the target CRS's units")
    parser.add_argument("--driver", choices=["COG", "GTiff"], default="COG", help="Output format (default: COG)")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="cf2ogc", description="CF-NetCDF to OGC-ready rasters")
    parser.add_argument("--version", action="version", version=f"cf2ogc {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("points", help="Grid a CF point cloud (1-D 'point' dimension) into a GeoTIFF")
    p.add_argument("config", help="Path to the YAML configuration")
    p.add_argument("-i", "--input", help="Input NetCDF path or OPeNDAP URL (default: the config's input_dataset)")
    p.add_argument("-o", "--output", help="Output GeoTIFF (default: the config's output_file, else the input name with .tif)")
    p.add_argument(
        "--implementation",
        choices=["classic", "fast"],
        help="Override processing.implementation (the command line's default is fast)",
    )
    p.set_defaults(func=_points)

    g = sub.add_parser(
        "grid",
        help="Write one time step and vertical level of a gridded CF variable as a COG; print its description",
    )
    g.add_argument("source", help="Input NetCDF path or OPeNDAP URL")
    g.add_argument("variable", help="The variable to write")
    g.add_argument("-o", "--output", required=True, help="Output raster")
    _slice_options(g)
    g.set_defaults(func=_grid)

    w = sub.add_parser(
        "vector",
        help="Write one time step and vertical level of a vector field as u, v and speed bands; print its description",
    )
    w.add_argument("source", help="Input NetCDF path or OPeNDAP URL")
    w.add_argument("-o", "--output", required=True, help="Output raster")
    w.add_argument("--u", help="The u (eastward or grid x) component variable")
    w.add_argument("--v", help="The v (northward or grid y) component variable")
    w.add_argument("--direction", help="A direction variable (clockwise from north), instead of u and v")
    w.add_argument("--magnitude", help="The magnitude variable that goes with --direction")
    w.add_argument("--convention", choices=["from", "to"], help="The direction's convention, when its attributes do not say")
    w.add_argument("--frame", choices=["earth", "grid"], help="The u/v components' axes, when their standard names do not say")
    w.add_argument(
        "--components",
        choices=["grid", "earth"],
        default="grid",
        help="Turn the components into the output grid's axes (grid, the default) or keep them eastward/northward",
    )
    _slice_options(w)
    w.set_defaults(func=_vector)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args) or 0)
