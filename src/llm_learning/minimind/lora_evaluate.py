"""阶段 6 Pretrain、Full SFT 与 LoRA 固定生成对比入口。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .evaluate import evaluate_checkpoints
from .lora_config import load_lora_config
from .lora_train import build_lora_training_config
from .sft_evaluate import render_chat_prompt


def load_stage5_generation(path: Path) -> dict[str, Any]:
    """读取并验证可复用的 Pretrain 与 Full SFT 固定生成。"""
    if not path.is_file():
        raise FileNotFoundError(f"找不到阶段 5 固定生成：{path}")
    result = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "inspection_prompts",
        "generation_config",
        "pretrain",
        "sft",
    }
    missing = sorted(required.difference(result))
    if missing:
        raise ValueError(f"stage 5 generation is missing fields: {missing}")

    prompts = list(result["generation_config"]["prompts"])
    for model_name in ["pretrain", "sft"]:
        modes = result[model_name].get("modes", {})
        if set(modes) != {"greedy", "sampling"}:
            raise ValueError(f"{model_name} generation modes do not match")
        for mode_name, generations in modes.items():
            observed = [item.get("prompt") for item in generations]
            if observed != prompts:
                raise ValueError(
                    f"{model_name} {mode_name} prompts do not match"
                )
    return result


def run_three_way_generation(
    config_path: Path,
    stage5_generation_path: Path,
    lora_weights_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    """复用前两组生成，只对 LoRA merged weights 运行同一口径。"""
    if not lora_weights_path.is_file():
        raise FileNotFoundError(f"找不到 LoRA merged weights：{lora_weights_path}")
    lora_config = load_lora_config(config_path)
    training_config = build_lora_training_config(lora_config)
    stage5 = load_stage5_generation(stage5_generation_path)
    generation_config = dict(stage5["generation_config"])
    lora_result = evaluate_checkpoints(
        training_config,
        [lora_weights_path],
        generation_config,
        prompt_renderer=render_chat_prompt,
    )
    result = {
        "profile": lora_config.profile,
        "inspection_prompts": stage5["inspection_prompts"],
        "generation_config": generation_config,
        "pretrain": stage5["pretrain"],
        "full_sft": stage5["sft"],
        "lora": lora_result["checkpoints"][0],
        "comparison_scope": (
            "Pretrain 使用普通文本 prompt；Full SFT 与 LoRA 使用相同 chat "
            "template。三者使用相同 user prompts 与 decoding 参数。"
        ),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("docs/stages/06_lora/configs/rank16.json"),
    )
    parser.add_argument(
        "--stage5-generation",
        type=Path,
        default=Path("outputs/minimind/stage5/mini64/generation.json"),
    )
    parser.add_argument(
        "--lora-weights",
        type=Path,
        default=Path(
            "checkpoints/minimind/stage6/rank16/"
            "merged_weights_step_56480.pth"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/minimind/stage6/rank16/generation.json"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_three_way_generation(
        args.config,
        args.stage5_generation,
        args.lora_weights,
        args.output,
    )
    print(args.output)


if __name__ == "__main__":
    main()
