#!/usr/bin/env python3
"""Generate a deterministic mechanism-only SERE similarity fixture."""

import argparse
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("--layers", type=int, default=40)
    parser.add_argument("--experts", type=int, default=256)
    parser.add_argument("--force", action="store_true", help="replace an existing output")
    args = parser.parse_args()

    if args.layers <= 0 or args.experts <= 0:
        parser.error("layers and experts must be positive")
    if not args.output.parent.is_dir():
        parser.error(f"output directory does not exist: {args.output.parent}")
    if args.output.exists() and not args.force:
        parser.error(f"output already exists: {args.output}; pass --force to replace it")

    similarity = np.full(
        (args.layers, args.experts, args.experts), 0.95, dtype=np.float32
    )
    diagonal = np.arange(args.experts)
    similarity[:, diagonal, diagonal] = 1.0
    np.savez_compressed(args.output, similarity_matrices=similarity)

    print(
        f"wrote mechanism-only fixture: {args.output} "
        f"shape={similarity.shape} offdiag=0.95 diagonal=1.0"
    )


if __name__ == "__main__":
    main()
