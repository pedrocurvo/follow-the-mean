#!/usr/bin/env python3
"""Shared CLI parser helpers for RGM scripts."""

from __future__ import annotations

import argparse
from pathlib import Path

import experiment_config

# ---------------------------------------------------------------------------
# Shared Parsers
# ---------------------------------------------------------------------------


def config_args(description: str, script: str) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", type=Path, required=True, help="YAML experiment config")
    args = parser.parse_args()
    return experiment_config.namespace_from_config(args.config, script=script)


def argument_parser(description: str) -> argparse.ArgumentParser:
    return argparse.ArgumentParser(description=description)
