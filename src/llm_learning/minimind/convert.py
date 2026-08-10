"""调用 MiniMind 官方转换脚本导出 Transformers checkpoint。"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Protocol, cast

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .config import PretrainConfig, load_training_config
from .inspect import load_official_model_module
from .train import build_model_and_tokenizer


class OfficialConverter(Protocol):
    """官方动态转换模块中本项目实际使用的接口。"""

    lm_config: Any

    def convert_torch2transformers(
        self,
        torch_path: str,
        transformers_path: str,
        dtype: torch.dtype,
    ) -> Any: ...


@contextmanager
def working_directory(path: Path):
    """临时切换工作目录，兼容官方脚本中的相对路径。"""
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def load_official_converter(source_dir: Path) -> OfficialConverter:
    """从固定源码加载官方 convert_model.py。"""
    source_dir = source_dir.resolve()
    script_path = source_dir / "scripts" / "convert_model.py"
    if not script_path.is_file():
        raise FileNotFoundError(f"找不到官方转换脚本：{script_path}")
    source_text = script_path.read_text(encoding="utf-8")
    if "def convert_torch2transformers" not in source_text:
        raise ValueError("固定源码中缺少 convert_torch2transformers")

    source_string = str(source_dir)
    added_to_path = source_string not in sys.path
    if added_to_path:
        sys.path.insert(0, source_string)
    try:
        spec = importlib.util.spec_from_file_location(
            "llm_learning_official_convert_model",
            script_path,
        )
        if spec is None or spec.loader is None:
            raise ImportError(f"无法加载官方转换脚本：{script_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return cast(OfficialConverter, module)
    finally:
        if added_to_path:
            sys.path.remove(source_string)


def convert_checkpoint(
    config: PretrainConfig,
    weights_path: Path,
    output_dir: Path,
) -> Path:
    """用官方 Qwen3 兼容转换函数导出一个原生模型权重。"""
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"转换输出目录不是空目录：{output_dir}")
    official_model = load_official_model_module(config.source_dir)
    tokenizer = AutoTokenizer.from_pretrained(config.source_dir / "model")
    converter = load_official_converter(config.source_dir)
    converter.lm_config = official_model.MiniMindConfig(
        **config.model_kwargs(len(tokenizer))
    )
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    resolved_weights_path = weights_path.resolve()
    resolved_output_dir = output_dir.resolve()
    with working_directory(config.source_dir.resolve() / "scripts"):
        converter.convert_torch2transformers(
            str(resolved_weights_path),
            str(resolved_output_dir),
            dtype=torch.float16,
        )
    return output_dir


@torch.inference_mode()
def verify_conversion(
    config: PretrainConfig,
    weights_path: Path,
    converted_dir: Path,
) -> dict[str, float | bool]:
    """比较原生模型与转换模型在固定输入上的 logits。"""
    native, tokenizer = build_model_and_tokenizer(config, torch.device("cpu"))
    state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)
    native.load_state_dict(state_dict, strict=True)
    native.eval()
    converted = AutoModelForCausalLM.from_pretrained(converted_dir).float().eval()
    converted_tokenizer = AutoTokenizer.from_pretrained(converted_dir)
    tokenizer_matches = tokenizer.get_vocab() == converted_tokenizer.get_vocab()
    input_ids = torch.tensor([[1, 41, 128, 2]], dtype=torch.long)
    native_logits = native(input_ids).logits.float()
    converted_logits = converted(input_ids).logits.float()
    maximum_difference = (native_logits - converted_logits).abs().max().item()
    return {
        "tokenizer_matches": tokenizer_matches,
        "maximum_logit_absolute_difference": maximum_difference,
        "logits_close_at_1e-3": bool(
            torch.allclose(native_logits, converted_logits, atol=1e-3, rtol=1e-3)
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("docs/stages/04_pretrain/configs/mini64.json"),
    )
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_training_config(args.config)
    path = convert_checkpoint(config, args.weights, args.output)
    print(verify_conversion(config, args.weights, path))


if __name__ == "__main__":
    main()
