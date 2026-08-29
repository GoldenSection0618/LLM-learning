"""阶段 9 的轻量运行清单与复现结果比较工具。"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any, Iterable, Mapping


REQUIRED_MANIFEST_FIELDS = {
    "experiment",
    "git_commit",
    "command",
    "model",
    "dataset",
    "seed",
    "hardware",
    "training",
    "evaluation",
    "artifacts",
}


def current_git_commit(repository: Path) -> str:
    """读取当前 repository 的 commit。"""
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def validate_manifest(manifest: Mapping[str, Any]) -> None:
    """检查运行清单是否包含阶段 9 固定字段。"""
    missing = REQUIRED_MANIFEST_FIELDS.difference(manifest)
    if missing:
        raise ValueError(f"run manifest is missing fields: {sorted(missing)}")
    if not isinstance(manifest["seed"], int):
        raise TypeError("run manifest seed must be an integer")


def write_manifest(path: Path, manifest: Mapping[str, Any]) -> None:
    """校验并写入一份运行清单。"""
    validate_manifest(manifest)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def scalar_series(records: Iterable[Mapping[str, Any]]) -> list[tuple[int, str, float]]:
    """把 JSONL 指标转换为 TensorBoard 所需的 step/name/value。"""
    series = []
    ignored = {"step", "optimizer_step", "epoch", "next_batch"}
    for record in records:
        step = int(record.get("optimizer_step", record.get("step", 0)))
        for name, value in record.items():
            if name in ignored or isinstance(value, bool):
                continue
            if isinstance(value, int | float):
                series.append((step, name, float(value)))
    return series


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    """读取非空 JSONL 记录。"""
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def write_tensorboard(metrics_path: Path, log_dir: Path) -> int:
    """将已有 JSONL scalar metrics 写入本地 TensorBoard。"""
    from torch.utils.tensorboard import SummaryWriter

    values = scalar_series(load_jsonl(metrics_path))
    with SummaryWriter(log_dir=str(log_dir)) as writer:
        for step, name, value in values:
            writer.add_scalar(name, value, step)
    return len(values)


def compare_reproduction_results(
    first: Mapping[str, Any],
    second: Mapping[str, Any],
    *,
    numeric_fields: Iterable[str],
    exact_fields: Iterable[str],
    absolute_tolerance: float,
) -> dict[str, Any]:
    """按数值容差与精确字段比较两次固定评估。"""
    numeric = {}
    for field in numeric_fields:
        difference = abs(float(first[field]) - float(second[field]))
        numeric[field] = {
            "difference": difference,
            "matches": difference <= absolute_tolerance,
        }
    exact = {
        field: {"matches": first[field] == second[field]}
        for field in exact_fields
    }
    return {
        "numeric": numeric,
        "exact": exact,
        "reproduced": all(item["matches"] for item in numeric.values())
        and all(item["matches"] for item in exact.values()),
    }


def parse_args() -> argparse.Namespace:
    """解析 JSONL 转 TensorBoard 的命令行参数。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--log-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    scalar_count = write_tensorboard(args.metrics, args.log_dir)
    print(f"TensorBoard scalars: {scalar_count}")
    print(f"log directory: {args.log_dir}")


if __name__ == "__main__":
    main()
