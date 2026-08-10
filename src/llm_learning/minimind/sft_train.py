"""阶段 5 MiniMind 单卡 SFT 入口。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .sft_config import load_sft_config
from .sft_data import (
    build_sft_dataset,
    create_sft_split_manifest,
    load_sft_json_dataset,
    resolve_sft_split_row_ids,
)
from .train import run_training


def run_sft_training(
    config_path: Path,
    resume_path: Path | None = None,
    stop_after_step: int | None = None,
) -> dict[str, Any]:
    """从配置文件启动或恢复一次 SFT。"""
    config = load_sft_config(config_path)
    return run_training(
        config,
        resume_path,
        stop_after_step=stop_after_step,
        dataset_factory=build_sft_dataset,
        manifest_factory=create_sft_split_manifest,
        split_resolver=resolve_sft_split_row_ids,
        records_loader=load_sft_json_dataset,
        initial_weights_path=config.initial_weights_path,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("docs/stages/05_sft/configs/overfit100.json"),
    )
    parser.add_argument("--resume", type=Path)
    parser.add_argument(
        "--stop-after-step",
        type=int,
        help="在指定 optimizer step 完成评估、导出和 checkpoint 后停止",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_sft_training(args.config, args.resume, args.stop_after_step)
    print(json.dumps(result["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
