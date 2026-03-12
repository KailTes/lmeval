"""
Offline INT8 per-channel weight quantization → compressed-tensors format.

Input:  FP16/BF16 model directory (safetensors)
Output: INT8 quantized model directory (compressed-tensors, compatible with vllm/omni-infer)

Usage:
  python3 quantize_safetensors_int8.py --model /path/to/fp16-model --output /path/to/output

No GPU required. No model code required. Pure tensor math on safetensors.

Blacklist mode: all 2D Linear weights are quantized EXCEPT those matching skip patterns.
Default skip: embed (embedding), kv_b_proj (MLA compressed KV), lm_head, shared_head.head (output heads).

quantization_config is always written to config.json with:
  - ignore list: all Linear layers NOT quantized
  - global_compression_ratio: computed from actual size reduction
  - quantize: w8a8_dynamic

Quantization:
  - Weights: INT8, symmetric, per-output-channel (RTN)
  - Scale dtype: bfloat16 (required by NPU npu_grouped_matmul)
  - Activations: dynamic per-token at inference time (handled by runtime)
"""

import argparse
import json
import shutil
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

# Blacklist: tensors matching these patterns are SKIPPED (kept in original dtype).
SKIP_PATTERNS = [
    "embed",              # Embedding layer — keep high precision for token quality
    "kv_b_proj",          # MLA compressed KV projection (special cache handling)
    "lm_head",            # Output head — keep high precision
    "shared_head.head",   # Shared output head — keep high precision
]


def should_quantize(name: str, tensor: torch.Tensor, skip_patterns: list[str] = SKIP_PATTERNS) -> bool:
    if not name.endswith(".weight"):
        return False
    if tensor.ndim != 2:
        return False
    for pattern in skip_patterns:
        if pattern in name:
            return False
    return True


def quantize_per_channel(weight: torch.Tensor, scale_dtype: torch.dtype = torch.bfloat16):
    """RTN per-output-channel symmetric INT8 quantization.

    weight shape: [out_features, in_features]
    returns: (int8_weight, scale)
      int8_weight: [out_features, in_features] torch.int8
      scale:       [out_features, 1]           scale_dtype
    """
    # per-output-channel: amax along input dim (dim=1)
    scale = weight.abs().amax(dim=1, keepdim=True).float() / 127.0
    scale = scale.clamp(min=1e-10)
    int8_weight = (weight.float() / scale).round().clamp(-128, 127).to(torch.int8)
    scale = scale.to(scale_dtype)
    return int8_weight, scale


def build_quantization_config(ignore_list, global_compression_ratio=None):
    return {
        "quant_method": "compressed-tensors",
        "quantize": "w8a8_dynamic",
        "format": "int-quantized",
        "quantization_status": "compressed",
        "config_groups": {
            "group_0": {
                "format": "int-quantized",
                "targets": ["Linear"],
                "weights": {
                    "type": "int",
                    "num_bits": 8,
                    "symmetric": True,
                    "strategy": "channel",
                    "dynamic": False,
                    "observer": "minmax",
                    "actorder": None,
                    "group_size": None,
                    "block_structure": None,
                    "observer_kwargs": {},
                },
                "input_activations": {
                    "type": "int",
                    "num_bits": 8,
                    "symmetric": True,
                    "strategy": "token",
                    "dynamic": True,
                    "actorder": None,
                    "group_size": None,
                    "block_structure": None,
                    "observer": None,
                    "observer_kwargs": {},
                },
                "output_activations": None,
            }
        },
        "ignore": ignore_list,
        "sparsity_config": {},
        "transform_config": {},
        "global_compression_ratio": global_compression_ratio,
        "kv_cache_scheme": None,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Offline INT8 per-channel quantization (compressed-tensors format)"
    )
    parser.add_argument("--model", required=True, help="FP16/BF16 model directory")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument(
        "--skip-patterns",
        nargs="*",
        default=None,
        help="Blacklist patterns for weight names to skip (default: embed, kv_b_proj, lm_head, shared_head.head)",
    )
    parser.add_argument(
        "--scale-dtype",
        choices=["float16", "bfloat16"],
        default="bfloat16",
        help="Dtype for weight_scale tensors (default: bfloat16)",
    )
    args = parser.parse_args()

    skip = args.skip_patterns if args.skip_patterns is not None else SKIP_PATTERNS
    scale_dtype = torch.float16 if args.scale_dtype == "float16" else torch.bfloat16

    model_dir = Path(args.model)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    st_files = sorted(model_dir.glob("*.safetensors"))
    if not st_files:
        raise FileNotFoundError(f"No safetensors files in {model_dir}")

    print(f"Model: {model_dir}")
    print(f"Output: {output_dir}")
    print(f"Skip patterns (blacklist): {skip}")
    print(f"Scale dtype: {scale_dtype}")
    print(f"Found {len(st_files)} safetensors file(s)")

    total_quantized = 0
    total_skipped = 0
    new_index_map = {}
    ignored_linear_names = []
    original_bytes = 0
    quantized_bytes = 0

    for st_file in st_files:
        print(f"\n--- {st_file.name} ---")
        tensors = load_file(str(st_file))
        output_tensors = {}

        for name in sorted(tensors.keys()):
            tensor = tensors[name]
            tensor_bytes = tensor.nelement() * tensor.element_size()
            original_bytes += tensor_bytes
            if should_quantize(name, tensor, skip):
                int8_w, scale = quantize_per_channel(tensor, scale_dtype)
                output_tensors[name] = int8_w
                scale_name = name.replace(".weight", ".weight_scale")
                output_tensors[scale_name] = scale
                quantized_bytes += int8_w.nelement() * int8_w.element_size()
                quantized_bytes += scale.nelement() * scale.element_size()
                total_quantized += 1
                print(f"  [Q] {name}: {list(tensor.shape)} {tensor.dtype} → int8 + scale{list(scale.shape)}")
            else:
                output_tensors[name] = tensor
                quantized_bytes += tensor_bytes
                total_skipped += 1
                # Collect Linear weight names that were NOT quantized for the ignore list
                if name.endswith(".weight") and tensor.ndim == 2:
                    ignored_linear_names.append(name.removesuffix(".weight"))

        out_path = output_dir / st_file.name
        save_file(output_tensors, str(out_path))
        print(f"  Saved → {out_path}")

        for tname in output_tensors:
            new_index_map[tname] = st_file.name

    # Copy non-safetensors files
    for f in model_dir.iterdir():
        if f.suffix == ".safetensors" or f.name == "model.safetensors.index.json":
            continue
        dst = output_dir / f.name
        if f.is_file() and not dst.exists():
            shutil.copy2(f, dst)

    # Regenerate index.json if sharded
    if len(st_files) > 1:
        index_path = model_dir / "model.safetensors.index.json"
        metadata = {}
        if index_path.exists():
            with open(index_path) as f:
                metadata = json.load(f).get("metadata", {})
        with open(output_dir / "model.safetensors.index.json", "w") as f:
            json.dump({"metadata": metadata, "weight_map": new_index_map}, f, indent=2)

    # Compute global compression ratio
    global_compression_ratio = round(original_bytes / quantized_bytes, 4) if quantized_bytes > 0 else None

    # Always write quantization_config to config.json
    config_path = output_dir / "config.json"
    if config_path.exists():
        with open(config_path) as f:
            config = json.load(f)
        config["quantization_config"] = build_quantization_config(
            ignored_linear_names, global_compression_ratio
        )
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        print(f"\nUpdated config.json with quantization_config:")
        print(f"  ignore list: {len(ignored_linear_names)} layers")
        print(f"  global_compression_ratio: {global_compression_ratio}")
        print(f"  quantize: w8a8_dynamic")

    print(f"\n{'='*50}")
    print(f"Quantized: {total_quantized} layers, Skipped: {total_skipped} tensors")
    print(f"Output: {output_dir}")


if __name__ == "__main__":
    main()
