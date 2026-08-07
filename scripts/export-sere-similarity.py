#!/usr/bin/env python3
"""Convert SERE expert similarity matrices to a llama.cpp sidecar."""

import argparse
import hashlib
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


MAGIC = b"MOESERE2"
VERSION = 2
BINDING_SCHEMA = 1
MANIFEST_DOMAIN = b"llama.cpp-sere-model-manifest-v1"
HEADER = struct.Struct("<8s10I32s")
UINT32_MAX = (1 << 32) - 1
UINT64_MAX = (1 << 64) - 1
METRIC_IDS = {
    "unspecified": 0,
    "frobenius": 1,
    "cosine": 2,
    "cka": 3,
}


class ExportError(RuntimeError):
    pass


def append_u32(output, value):
    if not 0 <= value <= UINT32_MAX:
        raise ExportError(f"manifest uint32 value is out of range: {value}")
    output.extend(struct.pack("<I", value))


def append_u64(output, value):
    if not 0 <= value <= UINT64_MAX:
        raise ExportError(f"manifest uint64 value is out of range: {value}")
    output.extend(struct.pack("<Q", value))


def append_string(output, value):
    encoded = value.encode("utf-8")
    append_u32(output, len(encoded))
    output.extend(encoded)


def gguf_value(reader, key):
    field = reader.fields.get(key)
    if field is None:
        raise ExportError(f"model is missing required GGUF metadata: {key}")
    try:
        return field.contents()
    except Exception as exc:
        raise ExportError(f"failed to read GGUF metadata {key}: {exc}") from exc


def load_model_binding(model_path, expected_layers, expected_experts):
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root / "gguf-py"))
    try:
        from gguf.gguf_reader import GGUFReader
    except ImportError as exc:
        raise ExportError("the repository gguf-py package is required for --model") from exc

    try:
        reader = GGUFReader(model_path)
    except Exception as exc:
        raise ExportError(f"failed to read model GGUF metadata: {exc}") from exc

    architecture = str(gguf_value(reader, "general.architecture"))
    model_name = str(gguf_value(reader, "general.name"))
    general_file_type = int(gguf_value(reader, "general.file_type"))
    quantization_version = int(gguf_value(reader, "general.quantization_version"))
    manifest_version = int(gguf_value(reader, "moe_offload.version"))
    layout = str(gguf_value(reader, "moe_offload.layout"))
    n_layers = int(gguf_value(reader, "moe_offload.n_moe_layers"))
    n_experts = int(gguf_value(reader, "moe_offload.n_experts_per_layer"))
    expert_blob_size_max = int(gguf_value(reader, "moe_offload.expert_blob_size_max"))
    layer_ids = [int(value) for value in gguf_value(reader, "moe_offload.layer_ids")]
    records = [int(value) for value in gguf_value(reader, "moe_offload.expert_blob.table")]

    top_k_key = f"{architecture}.expert_used_count"
    if top_k_key in reader.fields:
        model_top_k = int(gguf_value(reader, top_k_key))
    else:
        candidates = [key for key in reader.fields if key.endswith(".expert_used_count")]
        if len(candidates) != 1:
            raise ExportError(
                f"model is missing unambiguous router top-k metadata {top_k_key}"
            )
        model_top_k = int(gguf_value(reader, candidates[0]))

    for label, value in {
        "layers": n_layers,
        "experts": n_experts,
        "model top-k": model_top_k,
        "manifest version": manifest_version,
    }.items():
        if not 1 <= value <= UINT32_MAX:
            raise ExportError(f"model {label} must be a positive uint32, found {value}")
    if not architecture or not model_name or not layout:
        raise ExportError("model architecture, name, and MoE layout must be non-empty")
    if not 0 <= general_file_type <= UINT32_MAX or not 0 <= quantization_version <= UINT32_MAX:
        raise ExportError("model quantization metadata does not fit in uint32")
    if expected_layers != n_layers or expected_experts != n_experts:
        raise ExportError(
            f"matrix dimensions {expected_layers}x{expected_experts} do not match "
            f"model {n_layers}x{n_experts}"
        )
    if len(layer_ids) != n_layers:
        raise ExportError("model MoE layer ID table does not match its layer count")

    record_count = n_layers * n_experts * 3
    if len(records) != record_count * 2:
        raise ExportError(
            f"model expert blob table has {len(records)} values; expected {record_count * 2}"
        )

    tensors_by_offset = {}
    for tensor in reader.tensors:
        relative_offset = int(tensor.data_offset) - int(reader.data_offset)
        if relative_offset in tensors_by_offset:
            raise ExportError(f"multiple GGUF tensors share relative offset {relative_offset}")
        tensors_by_offset[relative_offset] = tensor

    tensor_layouts = []
    for logical in range(n_layers):
        for kind in range(3):
            first_index = ((logical * n_experts) * 3 + kind) * 2
            first_offset = records[first_index]
            first_size = records[first_index + 1]
            tensor = tensors_by_offset.get(first_offset)
            if tensor is None:
                raise ExportError(
                    f"no GGUF tensor begins at expert blob offset {first_offset} "
                    f"for logical layer {logical}, kind {kind}"
                )
            tensor_size = int(tensor.n_bytes)
            if first_size <= 0 or tensor_size != first_size * n_experts:
                raise ExportError(
                    f"expert tensor {tensor.name} size does not match its blob table"
                )
            for expert in range(n_experts):
                index = ((logical * n_experts + expert) * 3 + kind) * 2
                if records[index] != first_offset + expert * first_size or records[index + 1] != first_size:
                    raise ExportError(
                        "model expert blob table is not a contiguous fused-tensor layout"
                    )
            shape = [int(value) for value in tensor.shape]
            if len(shape) > 4:
                raise ExportError(f"expert tensor {tensor.name} has more than four dimensions")
            shape.extend([1] * (4 - len(shape)))
            tensor_layouts.append(
                {
                    "name": tensor.name,
                    "type": int(tensor.tensor_type),
                    "rel_offset": first_offset,
                    "size": tensor_size,
                    "shape": shape,
                }
            )

    canonical = bytearray(MANIFEST_DOMAIN)
    append_u32(canonical, BINDING_SCHEMA)
    append_string(canonical, architecture)
    append_string(canonical, model_name)
    append_string(canonical, layout)
    append_u32(canonical, manifest_version)
    append_u32(canonical, n_layers)
    append_u32(canonical, n_experts)
    append_u32(canonical, model_top_k)
    append_u32(canonical, general_file_type)
    append_u32(canonical, quantization_version)
    append_u64(canonical, expert_blob_size_max)
    append_u64(canonical, model_path.stat().st_size)
    append_u32(canonical, len(layer_ids))
    for layer in layer_ids:
        append_u32(canonical, layer)
    append_u32(canonical, len(tensor_layouts))
    for tensor in tensor_layouts:
        append_string(canonical, tensor["name"])
        append_u32(canonical, tensor["type"])
        append_u64(canonical, tensor["rel_offset"])
        append_u64(canonical, tensor["size"])
        for value in tensor["shape"]:
            append_u64(canonical, value)
    append_u64(canonical, record_count)
    for index in range(record_count):
        append_u64(canonical, records[2 * index])
        append_u64(canonical, records[2 * index + 1])

    return {
        "n_layers": n_layers,
        "n_experts": n_experts,
        "model_top_k": model_top_k,
        "manifest_version": manifest_version,
        "general_file_type": general_file_type,
        "quantization_version": quantization_version,
        "manifest_sha256": hashlib.sha256(canonical).digest(),
    }


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


def write_sidecar(path, matrices, n_experts, metric, binding, force):
    if path.exists() and not force:
        raise ExportError(f"output already exists: {path}; pass --force to replace it")
    if not path.parent.is_dir():
        raise ExportError(f"output directory does not exist: {path.parent}")

    payload = np.asarray(matrices, dtype="<f4", order="C")
    header = HEADER.pack(
        MAGIC,
        VERSION,
        HEADER.size,
        BINDING_SCHEMA,
        matrices.shape[0],
        n_experts,
        metric,
        binding["model_top_k"],
        binding["manifest_version"],
        binding["general_file_type"],
        binding["quantization_version"],
        binding["manifest_sha256"],
    )
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
            "model-bound MOESERE2 little-endian float32 sidecar."
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
    parser.add_argument("output", type=Path, help="output MOESERE2 sidecar path")
    parser.add_argument(
        "--model",
        required=True,
        type=Path,
        help=(
            "repacked GGUF whose architecture, router top-k, quantization, and "
            "expert layout will be bound to the sidecar"
        ),
    )
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
        if not args.model.is_file():
            raise ExportError(f"model file does not exist: {args.model}")
        if args.input.resolve() == args.output.resolve():
            raise ExportError("input and output paths must differ")

        matrices = load_input(args.input)
        matrices, n_experts = validate_matrices(
            matrices,
            expected_layers=args.n_layers,
            expected_experts=args.n_experts,
        )
        binding = load_model_binding(args.model, matrices.shape[0], n_experts)
        write_sidecar(
            args.output, matrices, n_experts, args.metric, binding, args.force
        )
    except ExportError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(
        f"wrote {args.output}: layers={matrices.shape[0]}, "
        f"experts={n_experts}, model_top_k={binding['model_top_k']}, "
        f"metric={args.metric}, manifest_sha256={binding['manifest_sha256'].hex()}, "
        f"bytes={args.output.stat().st_size}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
