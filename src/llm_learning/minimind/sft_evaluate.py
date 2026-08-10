"""阶段 5 的 Pretrain/SFT 固定生成对比入口。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .evaluate import (
    evaluate_checkpoints,
    load_generation_config,
)
from .sft_config import SFTConfig, load_sft_config
from .sft_data import load_sft_json_dataset


def render_chat_prompt(prompt: str, tokenizer: Any) -> str:
    """把单轮 user prompt 渲染为官方 assistant generation prompt。"""
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
        open_thinking=False,
    )


def inspection_prompts(
    config: SFTConfig,
    manifest_path: Path,
    count: int,
) -> list[dict[str, Any]]:
    """读取固定教学子集开头的 user 问题。"""
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    row_ids = manifest["split"]["inspection_row_ids"][:count]
    records = load_sft_json_dataset(config.data_path)
    prompts = []
    for row_id in row_ids:
        conversations = records[row_id]["conversations"]
        user_messages = [
            message
            for message in conversations
            if message.get("role") == "user" and message.get("content")
        ]
        if not user_messages:
            continue
        prompts.append(
            {
                "row_id": row_id,
                "prompt": str(user_messages[-1]["content"]),
            }
        )
    return prompts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("docs/stages/05_sft/configs/mini64.json"),
    )
    parser.add_argument(
        "--generation-config",
        type=Path,
        default=Path("docs/stages/05_sft/configs/generation.json"),
    )
    parser.add_argument("--pretrain-checkpoint", type=Path, required=True)
    parser.add_argument("--sft-checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_sft_config(args.config)
    generation = load_generation_config(args.generation_config)
    subset_prompts = inspection_prompts(
        config,
        args.manifest,
        int(generation["inspection_prompt_count"]),
    )
    prompts = list(generation["prompts"]) + [
        item["prompt"] for item in subset_prompts
    ]
    run_config = {**generation, "prompts": prompts}
    pretrain_result = evaluate_checkpoints(
        config,
        [args.pretrain_checkpoint],
        run_config,
    )
    sft_result = evaluate_checkpoints(
        config,
        [args.sft_checkpoint],
        run_config,
        prompt_renderer=render_chat_prompt,
    )
    result = {
        "profile": config.profile,
        "inspection_prompts": subset_prompts,
        "generation_config": run_config,
        "pretrain": pretrain_result["checkpoints"][0],
        "sft": sft_result["checkpoints"][0],
        "comparison_scope": (
            "Pretrain 使用普通文本 prompt；SFT 使用官方 chat template。"
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(args.output)


if __name__ == "__main__":
    main()
