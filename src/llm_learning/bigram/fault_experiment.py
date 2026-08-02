from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

from .config import TrainConfig
from .train import train


def run_experiment(
    output_root: Path,
    checkpoint_root: Path,
    steps: int,
    device: str,
) -> None:
    base = replace(
        TrainConfig(),
        max_steps=steps,
        eval_interval=max(1, steps // 4),
        eval_batches=10,
        generate_tokens=200,
        device=device,
    )
    experiments = {
        "control": base.learning_rate,
        # 只放大学习率，用单变量实验观察训练失稳。
        "learning_rate_too_large": 100.0,
    }
    results: dict[str, dict[str, object]] = {}

    for name, learning_rate in experiments.items():
        config = replace(
            base,
            learning_rate=learning_rate,
            output_dir=output_root / name,
            checkpoint_dir=checkpoint_root / name,
        )
        result = train(config)
        history = result["history"]
        results[name] = {
            "learning_rate": learning_rate,
            "initial_validation_loss": history[0]["validation_loss"],
            "final_validation_loss": history[-1]["validation_loss"],
            "generated_text": result["generated_text"],
        }

    output_root.mkdir(parents=True, exist_ok=True)
    result_path = output_root / "fault_experiment.json"
    result_path.write_text(
        json.dumps(results, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    report = [
        "# 错误实验：学习率过大",
        "",
        "对照组和错误组使用相同的数据、随机种子、模型和训练步数。",
        "实验只改变学习率。",
        "",
        "| 实验 | 学习率 | 初始验证损失 | 最终验证损失 |",
        "| --- | ---: | ---: | ---: |",
    ]

    for name, values in results.items():
        report.append(
            f"| {name} | {values['learning_rate']} | "
            f"{values['initial_validation_loss']:.4f} | "
            f"{values['final_validation_loss']:.4f} |"
        )
    report.extend(
        [
            "",
            "生成样本和原始数值保存在 `fault_experiment.json`。",
            "过大的学习率会造成更高或更不稳定的损失，并降低生成质量。",
        ]
    )
    (output_root / "fault_report.md").write_text(
        "\n".join(report) + "\n",
        encoding="utf-8",
    )
    print(f"fault_experiment={result_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="比较正常学习率和过大学习率")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs/bigram_fault"),
    )
    parser.add_argument(
        "--checkpoint-root",
        type=Path,
        default=Path("checkpoints/bigram_fault"),
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=200,
    )
    parser.add_argument(
        "--device",
        default="auto",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_experiment(
        output_root=args.output_root,
        checkpoint_root=args.checkpoint_root,
        steps=args.steps,
        device=args.device,
    )


if __name__ == "__main__":
    main()
