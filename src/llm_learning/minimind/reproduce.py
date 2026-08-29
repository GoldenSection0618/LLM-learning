"""阶段 9 MiniMind 固定 validation 与 greedy generation 复现入口。"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from llm_learning.experiment_record import (
    compare_reproduction_results,
    current_git_commit,
    write_manifest,
)

from .evaluate import generate_one, load_generation_config, load_native_checkpoint
from .sft_config import load_sft_config
from .sft_data import (
    build_sft_dataset,
    load_sft_json_dataset,
    resolve_sft_split_row_ids,
)
from .sft_evaluate import render_chat_prompt
from .train import evaluate_loss, resolve_device


def load_reproduction_config(path: Path) -> dict[str, Any]:
    """读取阶段 9 固定复现配置。"""
    config = json.loads(path.read_text(encoding="utf-8"))
    if len(config["run_outputs"]) != 2:
        raise ValueError("reproduction config requires exactly two run outputs")
    return config


def run_fixed_evaluation(config_path: Path, run_index: int) -> dict[str, Any]:
    """执行一次固定 validation 和 greedy generation。"""
    reproduction = load_reproduction_config(config_path)
    if run_index not in {1, 2}:
        raise ValueError("run_index must be 1 or 2")
    model_config = load_sft_config(Path(reproduction["model_config"]))
    device = resolve_device(model_config.device)
    model, tokenizer = load_native_checkpoint(
        model_config,
        Path(reproduction["checkpoint"]),
        device,
    )
    records = load_sft_json_dataset(model_config.data_path)
    data_manifest = json.loads(
        Path(reproduction["data_manifest"]).read_text(encoding="utf-8")
    )
    _, validation_ids = resolve_sft_split_row_ids(data_manifest)
    validation_ids = validation_ids[: int(reproduction["validation_rows"])]
    validation_dataset = build_sft_dataset(
        model_config,
        records,
        validation_ids,
        tokenizer,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=model_config.batch_size,
        shuffle=False,
        num_workers=model_config.num_workers,
        pin_memory=device.type == "cuda",
    )
    generation = load_generation_config(Path(reproduction["generation_config"]))
    started = time.perf_counter()
    validation = evaluate_loss(
        model,
        validation_loader,
        device,
        model_config.dtype,
    )
    greedy = [
        generate_one(
            model,
            tokenizer,
            prompt,
            generation["greedy"],
            device,
            render_chat_prompt,
        )
        for prompt in generation["prompts"]
    ]
    result = {
        "profile": reproduction["profile"],
        "run_index": run_index,
        "validation_loss": validation["loss"],
        "perplexity": validation["perplexity"],
        "validation_tokens": validation["tokens"],
        "greedy_token_ids": [item["token_ids"] for item in greedy],
        "greedy_text": [item["text"] for item in greedy],
        "wall_time_seconds": time.perf_counter() - started,
    }
    output_path = Path(reproduction["run_outputs"][run_index - 1])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    manifest = {
        "experiment": reproduction["profile"],
        "git_commit": current_git_commit(Path.cwd()),
        "command": (
            "python -m llm_learning.minimind.reproduce run "
            f"--config {config_path} --run-index {run_index}"
        ),
        "model": {
            "name": "MiniMind 64M Full SFT",
            "checkpoint": reproduction["checkpoint"],
            "trainable_parameters": 0,
        },
        "dataset": {
            "name": str(model_config.data_path),
            "revision": data_manifest.get("raw_sha256"),
            "split": f"fixed validation {len(validation_ids)} rows",
        },
        "seed": int(reproduction["seed"]),
        "hardware": {
            "device": str(device),
            "gpu": (
                torch.cuda.get_device_name(device.index or 0)
                if device.type == "cuda"
                else "CPU"
            ),
            "dtype": model_config.dtype,
        },
        "training": {
            "effective_batch_size": 0,
            "total_tokens": 0,
            "peak_memory_bytes": 0,
            "wall_time_seconds": 0,
        },
        "evaluation": result,
        "artifacts": {
            "metrics": str(output_path),
            "checkpoint": reproduction["checkpoint"],
            "generation": str(output_path),
        },
    }
    write_manifest(output_path.with_name(f"manifest_{run_index}.json"), manifest)
    print(output_path)
    return result


def compare_runs(config_path: Path, output: Path) -> dict[str, Any]:
    """比较配置指定的两次运行。"""
    config = load_reproduction_config(config_path)
    first, second = [
        json.loads(Path(path).read_text(encoding="utf-8"))
        for path in config["run_outputs"]
    ]
    comparison = compare_reproduction_results(
        first,
        second,
        numeric_fields=config["numeric_fields"],
        exact_fields=config["exact_fields"],
        absolute_tolerance=float(config["absolute_tolerance"]),
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(comparison, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(output)
    return comparison


def parse_args() -> argparse.Namespace:
    """解析 run / compare 子命令。"""
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("run")
    run.add_argument("--config", type=Path, required=True)
    run.add_argument("--run-index", type=int, required=True)
    compare = subparsers.add_parser("compare")
    compare.add_argument("--config", type=Path, required=True)
    compare.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "run":
        run_fixed_evaluation(args.config, args.run_index)
    else:
        compare_runs(args.config, args.output)


if __name__ == "__main__":
    main()
