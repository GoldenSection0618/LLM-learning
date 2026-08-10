"""检查固定版本 MiniMind Dense 模型的 forward 与 KV Cache。"""

from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import torch


MINIMIND_REVISION = "89d674b8a517010f5561b6d8ab2dcbb58e2fb91b"


def load_official_model_module(source_dir: Path) -> ModuleType:
    """从本地固定版本源码加载 MiniMind 模型模块。"""
    source_dir = source_dir.resolve()
    model_path = source_dir / "model" / "model_minimind.py"
    if not model_path.is_file():
        raise FileNotFoundError(f"找不到 MiniMind 模型文件：{model_path}")

    completed = subprocess.run(
        ["git", "-C", str(source_dir), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    actual_revision = completed.stdout.strip()
    if actual_revision != MINIMIND_REVISION:
        raise ValueError(
            "MiniMind revision 不匹配："
            f"期望 {MINIMIND_REVISION}，实际 {actual_revision}"
        )

    status = subprocess.run(
        [
            "git",
            "-C",
            str(source_dir),
            "status",
            "--porcelain",
            "--untracked-files=no",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    if status.stdout.strip():
        raise ValueError("MiniMind 固定源码包含未提交修改，请恢复后再运行检查")

    spec = importlib.util.spec_from_file_location(
        "llm_learning_official_minimind",
        model_path,
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"无法加载 MiniMind 模型模块：{model_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def inspect_attention_variant(
    module: ModuleType,
    num_key_value_heads: int,
    seed: int,
    device: str | torch.device = "cpu",
) -> dict[str, Any]:
    """运行一种 attention 配置并返回 shape 与 cache 记录。"""
    device = torch.device(device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("请求使用 CUDA，但当前环境没有可用的 CUDA 设备")
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    config = module.MiniMindConfig(
        hidden_size=256,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=num_key_value_heads,
        vocab_size=6400,
        max_position_embeddings=128,
        flash_attn=True,
        use_moe=False,
        dropout=0.0,
    )
    model = module.MiniMindForCausalLM(config).to(device).eval()
    input_ids = torch.randint(0, config.vocab_size, (2, 8), device=device)

    observed_shapes: dict[str, list[int]] = {}
    hooks = []

    def capture_shape(name: str, num_heads: int | None = None):
        def hook(_module, _inputs, output):
            if num_heads is None:
                observed_shapes[name] = list(output.shape)
            else:
                observed_shapes[name] = list(
                    output.unflatten(-1, (num_heads, config.head_dim)).shape
                )

        return hook

    hooks.append(
        model.model.embed_tokens.register_forward_hook(capture_shape("embedding"))
    )
    first_attention = model.model.layers[0].self_attn
    hooks.extend(
        [
            first_attention.q_proj.register_forward_hook(
                capture_shape("query", config.num_attention_heads)
            ),
            first_attention.k_proj.register_forward_hook(
                capture_shape("key", config.num_key_value_heads)
            ),
            first_attention.v_proj.register_forward_hook(
                capture_shape("value", config.num_key_value_heads)
            ),
        ]
    )
    for index, layer in enumerate(model.model.layers):
        hooks.append(
            layer.register_forward_hook(
                lambda _module, _inputs, output, index=index: observed_shapes.__setitem__(
                    f"layer_{index}", list(output[0].shape)
                )
            )
        )

    with torch.no_grad():
        no_cache = model(input_ids, use_cache=False)
    for hook in hooks:
        hook.remove()

    with torch.no_grad():
        prefix = model(input_ids[:, :-1], use_cache=True)
        incremental = model(
            input_ids[:, -1:],
            past_key_values=prefix.past_key_values,
            use_cache=True,
        )
        full = model(input_ids, use_cache=True)

    return {
        "attention_type": (
            "MHA"
            if config.num_key_value_heads == config.num_attention_heads
            else "GQA"
        ),
        "config": {
            "hidden_size": config.hidden_size,
            "num_hidden_layers": config.num_hidden_layers,
            "num_attention_heads": config.num_attention_heads,
            "num_key_value_heads": config.num_key_value_heads,
            "head_dim": config.head_dim,
            "vocab_size": config.vocab_size,
        },
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "device": str(next(model.parameters()).device),
        "device_name": (
            torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU"
        ),
        "dtype": str(next(model.parameters()).dtype),
        "input_shape": list(input_ids.shape),
        "embedding_shape": observed_shapes["embedding"],
        "query_shape_before_transpose": observed_shapes["query"],
        "key_shape_before_repeat": observed_shapes["key"],
        "value_shape_before_repeat": observed_shapes["value"],
        "layer_output_shapes": {
            f"layer_{index}": observed_shapes[f"layer_{index}"]
            for index in range(config.num_hidden_layers)
        },
        "logits_shape": list(no_cache.logits.shape),
        "cache_disabled_entries_are_none": [
            item is None for item in no_cache.past_key_values
        ],
        "prefix_key_shape": list(prefix.past_key_values[0][0].shape),
        "prefix_value_shape": list(prefix.past_key_values[0][1].shape),
        "incremental_key_shape": list(incremental.past_key_values[0][0].shape),
        "incremental_value_shape": list(incremental.past_key_values[0][1].shape),
        "cache_vs_full_last_logit_max_abs_diff": float(
            (incremental.logits[:, -1] - full.logits[:, -1]).abs().max()
        ),
    }


def inspect_minimind(
    source_dir: Path,
    seed: int = 2026,
    device: str | torch.device = "auto",
) -> dict[str, Any]:
    """检查 GQA 与 MHA 两种缩小配置。"""
    if str(device) == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)
    module = load_official_model_module(source_dir)
    return {
        "source_revision": MINIMIND_REVISION,
        "torch_version": torch.__version__,
        "seed": seed,
        "gqa": inspect_attention_variant(module, 2, seed, device),
        "mha": inspect_attention_variant(module, 4, seed, device),
    }


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=Path("third_party/minimind"),
        help="MiniMind 官方源码目录",
    )
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--device",
        default="auto",
        help="运行设备，例如 auto、cuda、cuda:0 或 cpu",
    )
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    """运行检查并输出 JSON。"""
    args = parse_args()
    result = inspect_minimind(args.source_dir, args.seed, args.device)
    rendered = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
