#!/usr/bin/env python3
"""Convert SERE expert similarity matrices to a llama.cpp sidecar."""

import argparse
import os
import re
import struct
import sys
import tempfile
from collections.abc import Mapping
from numbers import Integral
from pathlib import Path

try:
    import numpy as np
except ImportError:
    np = None


MAGIC = b"MOESERE1"
VERSION = 1
HEADER = struct.Struct("<8sIIII")
UINT32_MAX = (1 << 32) - 1
METRIC_IDS = {
    "unspecified": 0,
    "frobenius": 1,
    "cosine": 2,
    "cka": 3,
}


class ExportError(RuntimeError):
    pass


def parse_metric(value):
    normalized = value.lower()
    if normalized in METRIC_IDS:
        return METRIC_IDS[normalized]
    try:
        metric = int(value, 0)
    except ValueError as exc:
        choices = ", ".join(METRIC_IDS)
        raise argparse.ArgumentTypeError(
            f"metric must be one of {choices}, or a uint32 ID"
        ) from exc
    if not 0 <= metric <= UINT32_MAX:
        raise argparse.ArgumentTypeError("numeric metric ID must fit in uint32")
    return metric


def positive_uint32(value):
    try:
        number = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a positive integer") from exc
    if not 1 <= number <= UINT32_MAX:
        raise argparse.ArgumentTypeError("must be between 1 and 4294967295")
    return number


def parse_layer_key(key):
    if isinstance(key, bool):
        raise ExportError(f"invalid boolean layer key: {key!r}")
    if isinstance(key, Integral):
        layer = int(key)
    elif isinstance(key, str):
        match = re.fullmatch(r"(?:layer[_-]?)?([0-9]+)", key)
        if match is None:
            raise ExportError(
                f"invalid layer key {key!r}; use 0, '0', or 'layer_0'"
            )
        layer = int(match.group(1))
    else:
        raise ExportError(
            f"invalid layer key type {type(key).__name__}; expected integer or string"
        )
    if not 0 <= layer <= UINT32_MAX:
        raise ExportError(f"layer key is outside uint32 range: {layer}")
    return layer


def normalize_layer_mapping(raw_layers):
    if not isinstance(raw_layers, Mapping):
        raise ExportError("input must contain a mapping from layer keys to matrices")
    if not raw_layers:
        raise ExportError("input contains no similarity matrices")

    layers = {}
    original_keys = {}
    for key, matrix in raw_layers.items():
        layer = parse_layer_key(key)
        if layer in layers:
            raise ExportError(
                f"duplicate layer {layer} after normalizing keys "
                f"{original_keys[layer]!r} and {key!r}"
            )
        layers[layer] = matrix
        original_keys[layer] = key

    indices = sorted(layers)
    expected = list(range(len(indices)))
    if indices != expected:
        raise ExportError(
            "layer keys must be contiguous and start at 0; "
            f"found {indices}"
        )
    return [layers[layer] for layer in indices]


def load_pt(path):
    try:
        import torch
    except ImportError as exc:
        raise ExportError(
            "PyTorch is required for .pt input; use .npz input to export without it"
        ) from exc

    try:
        raw = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError as exc:
        raise ExportError(
            "this PyTorch version cannot safely load the .pt file with "
            "weights_only=True; upgrade PyTorch or convert the file to .npz"
        ) from exc
    except Exception as exc:
        raise ExportError(f"failed to load PyTorch input: {exc}") from exc

    matrices = normalize_layer_mapping(raw)
    converted = []
    for layer, matrix in enumerate(matrices):
        if not isinstance(matrix, torch.Tensor):
            raise ExportError(
                f"layer {layer} in .pt input is {type(matrix).__name__}, not a tensor"
            )
        try:
            converted.append(
                matrix.detach().to(device="cpu", dtype=torch.float32).numpy()
            )
        except Exception as exc:
            raise ExportError(f"failed to convert layer {layer} tensor: {exc}") from exc
    return converted


def load_npz(path):
    try:
        with np.load(path, allow_pickle=False) as archive:
            keys = list(archive.files)
            if not keys:
                raise ExportError(".npz input contains no arrays")

            if len(keys) == 1 and keys[0] in {
                "arr_0",
                "similarity",
                "similarity_matrices",
            }:
                dense = archive[keys[0]]
                if dense.ndim != 3:
                    raise ExportError(
                        f"dense .npz array {keys[0]!r} must have shape "
                        "[layers, experts, experts]"
                    )
                return [np.array(dense[layer], copy=True) for layer in range(dense.shape[0])]

            raw_layers = {key: np.array(archive[key], copy=True) for key in keys}
    except ExportError:
        raise
    except (OSError, ValueError) as exc:
        raise ExportError(f"failed to load NumPy input: {exc}") from exc

    return normalize_layer_mapping(raw_layers)


def load_input(path):
    suffix = path.suffix.lower()
    if suffix in {".pt", ".pth"}:
        return load_pt(path)
    if suffix == ".npz":
        return load_npz(path)
    raise ExportError("input extension must be .pt, .pth, or .npz")


def validate_matrices(matrices, expected_layers=None, expected_experts=None):
    if not matrices:
        raise ExportError("input contains no similarity matrices")
    if len(matrices) > UINT32_MAX:
        raise ExportError("layer count does not fit in uint32")
    if expected_layers is not None and len(matrices) != expected_layers:
        raise ExportError(
            f"expected {expected_layers} layers, found {len(matrices)}"
        )

    validated = []
    n_experts = None
    for layer, matrix in enumerate(matrices):
        array = np.asarray(matrix)
        if array.dtype == np.dtype("O") or not np.issubdtype(array.dtype, np.number):
            raise ExportError(f"layer {layer} matrix must have a numeric dtype")
        if np.issubdtype(array.dtype, np.complexfloating):
            raise ExportError(f"layer {layer} matrix must not contain complex values")
        if np.issubdtype(array.dtype, np.bool_):
            raise ExportError(f"layer {layer} matrix must not contain boolean values")
        if array.ndim != 2 or array.shape[0] != array.shape[1]:
            raise ExportError(
                f"layer {layer} must be a square 2D matrix, found shape {array.shape}"
            )
        if array.shape[0] == 0:
            raise ExportError(f"layer {layer} matrix must not be empty")
        if array.shape[0] > UINT32_MAX:
            raise ExportError("expert count does not fit in uint32")
        if n_experts is None:
            n_experts = array.shape[0]
        elif array.shape != (n_experts, n_experts):
            raise ExportError(
                f"layer {layer} has shape {array.shape}; expected "
                f"({n_experts}, {n_experts})"
            )
        if not np.all(np.isfinite(array)):
            raise ExportError(f"layer {layer} contains NaN or infinity")

        with np.errstate(over="ignore", invalid="ignore"):
            array_f32 = np.asarray(array, dtype="<f4", order="C")
        if not np.all(np.isfinite(array_f32)):
            raise ExportError(
                f"layer {layer} contains values outside the finite float32 range"
            )
        validated.append(array_f32)

    if expected_experts is not None and n_experts != expected_experts:
        raise ExportError(
            f"expected {expected_experts} experts, found {n_experts}"
        )
    return np.stack(validated, axis=0), n_experts


def write_sidecar(path, matrices, n_experts, metric, force):
    if path.exists() and not force:
        raise ExportError(f"output already exists: {path}; pass --force to replace it")
    if not path.parent.is_dir():
        raise ExportError(f"output directory does not exist: {path.parent}")

    payload = np.asarray(matrices, dtype="<f4", order="C")
    header = HEADER.pack(MAGIC, VERSION, matrices.shape[0], n_experts, metric)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", prefix=f".{path.name}.", dir=path.parent, delete=False
        ) as output:
            temporary = Path(output.name)
            output.write(header)
            output.write(payload.tobytes(order="C"))
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
    except OSError as exc:
        if temporary is not None:
            try:
                temporary.unlink()
            except OSError:
                pass
        raise ExportError(f"failed to write output: {exc}") from exc


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Convert SERE similarity_matrices.pt or NumPy matrices to a "
            "MOESERE1 little-endian float32 sidecar."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Accepted .npz layouts:\n"
            "  similarity_matrices: one [layers, experts, experts] array\n"
            "  layer_0, layer_1, ...: one [experts, experts] array per key\n"
            "The keys 0, 1, ... and a single dense arr_0 are also accepted."
        ),
    )
    parser.add_argument("input", type=Path, help="input .pt, .pth, or .npz file")
    parser.add_argument("output", type=Path, help="output MOESERE1 sidecar path")
    parser.add_argument(
        "--metric",
        required=True,
        type=parse_metric,
        metavar="NAME_OR_ID",
        help="similarity metric: unspecified, frobenius, cosine, cka, or uint32 ID",
    )
    parser.add_argument(
        "--n-layers",
        type=positive_uint32,
        help="require this number of layers in the input",
    )
    parser.add_argument(
        "--n-experts",
        type=positive_uint32,
        help="require this matrix width and height",
    )
    parser.add_argument("--force", action="store_true", help="replace an existing output")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        if np is None:
            raise ExportError("NumPy is required to export similarity matrices")
        if not args.input.is_file():
            raise ExportError(f"input file does not exist: {args.input}")
        if args.input.resolve() == args.output.resolve():
            raise ExportError("input and output paths must differ")

        matrices = load_input(args.input)
        matrices, n_experts = validate_matrices(
            matrices,
            expected_layers=args.n_layers,
            expected_experts=args.n_experts,
        )
        write_sidecar(args.output, matrices, n_experts, args.metric, args.force)
    except ExportError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(
        f"wrote {args.output}: layers={matrices.shape[0]}, "
        f"experts={n_experts}, metric={args.metric}, bytes={args.output.stat().st_size}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
