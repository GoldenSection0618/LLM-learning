"""调用冻结版本 lm-evaluation-harness 的七任务评估入口。"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


def load_lm_eval_config(path: Path) -> dict[str, Any]:
    """读取固定七任务配置。"""
    config = json.loads(path.read_text(encoding="utf-8"))
    if len(config["tasks"]) != 7:
        raise ValueError("MiniMind baseline must contain exactly seven tasks")
    if config.get("trust_remote_code") is not True:
        raise ValueError("frozen dataset scripts require trust_remote_code=true")
    return config


def build_lm_eval_command(
    model_dir: Path,
    task_dir: Path,
    output_path: Path,
    config: dict[str, Any],
) -> list[str]:
    """构造标准 hf backend 命令，不实现任务或评分器。"""
    installed_version = importlib.metadata.version("lm_eval")
    if installed_version != config["lm_eval_version"]:
        raise RuntimeError(
            "lm_eval 版本不匹配："
            f"期望 {config['lm_eval_version']}，实际 {installed_version}"
        )
    command = [
        sys.executable,
        "-m",
        "lm_eval",
        "--model",
        "hf",
        "--model_args",
        f"pretrained={model_dir.resolve()},dtype=float16",
        "--tasks",
        ",".join(config["tasks"]),
        "--include_path",
        str(task_dir.resolve()),
        "--batch_size",
        str(config["batch_size"]),
        "--device",
        str(config["device"]),
        "--output_path",
        str(output_path.resolve()),
        "--log_samples",
    ]
    if config.get("apply_chat_template") is True:
        command.append("--apply_chat_template")
    return command


def run_lm_eval(
    model_dir: Path,
    task_dir: Path,
    output_path: Path,
    config: dict[str, Any],
) -> None:
    """运行冻结的七任务套件并保留标准输出。"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        build_lm_eval_command(model_dir, task_dir, output_path, config),
        check=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--task-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("docs/stages/04_pretrain/configs/lm_eval.json"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_lm_eval_config(args.config)
    run_lm_eval(args.model, args.task_dir, args.output, config)


if __name__ == "__main__":
    main()
