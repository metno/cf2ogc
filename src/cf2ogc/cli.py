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
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args) or 0)
