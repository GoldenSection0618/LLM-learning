"""为 MiniMind 七任务生成注入 Dataset revision 的本地 task 目录。"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.resources
import json
import shutil
from pathlib import Path
from typing import Any

from .lm_eval import load_lm_eval_config

TASK_MANIFEST_NAME = "stage4_task_manifest.json"


def inject_dataset_revision(
    source: str,
    revision: str,
    trust_remote_code: bool = True,
) -> tuple[str, bool]:
    """在含 dataset_path 的 task YAML 中固定 revision 与 script 授权。"""
    lines = source.splitlines(keepends=True)
    if not any(line.startswith("dataset_path:") for line in lines):
        return source, False

    dataset_kwargs_index = next(
        (
            index
            for index, line in enumerate(lines)
            if line.startswith("dataset_kwargs:")
        ),
        None,
    )
    if dataset_kwargs_index is None:
        dataset_path_index = next(
            index for index, line in enumerate(lines) if line.startswith("dataset_path:")
        )
        lines.insert(
            dataset_path_index + 1,
            "dataset_kwargs:\n"
            f"  revision: {revision}\n"
            f"  trust_remote_code: {str(trust_remote_code).lower()}\n",
        )
        return "".join(lines), True

    block_end = dataset_kwargs_index + 1
    revision_index = None
    trust_remote_code_index = None
    while block_end < len(lines):
        line = lines[block_end]
        if line.strip() and not line.startswith((" ", "\t", "#")):
            break
        if line.startswith("  revision:"):
            revision_index = block_end
        if line.startswith("  trust_remote_code:"):
            trust_remote_code_index = block_end
        block_end += 1
    if revision_index is not None:
        lines[revision_index] = f"  revision: {revision}\n"
    if trust_remote_code_index is not None:
        lines[trust_remote_code_index] = (
            f"  trust_remote_code: {str(trust_remote_code).lower()}\n"
        )
    additions = []
    if revision_index is None:
        additions.append(f"  revision: {revision}\n")
    if trust_remote_code_index is None:
        additions.append(
            f"  trust_remote_code: {str(trust_remote_code).lower()}\n"
        )
    lines[dataset_kwargs_index + 1 : dataset_kwargs_index + 1] = additions
    return "".join(lines), True


def sha256_directory(path: Path, excluded: set[str] | None = None) -> str:
    """按相对路径与文件内容计算目录身份。"""
    excluded = excluded or set()
    digest = hashlib.sha256()
    for file_path in sorted(item for item in path.rglob("*") if item.is_file()):
        relative_path = file_path.relative_to(path).as_posix()
        if (
            relative_path in excluded
            or "__pycache__" in file_path.parts
            or file_path.suffix == ".pyc"
        ):
            continue
        digest.update(relative_path.encode("utf-8"))
        digest.update(b"\0")
        digest.update(file_path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def reuse_existing_task_directory(
    destination: Path,
    config: dict[str, Any],
) -> dict[str, Any]:
    """校验并复用已经生成的 task 目录。"""
    if not destination.is_dir():
        raise ValueError(f"task 输出路径不是目录：{destination}")
    manifest_path = destination / TASK_MANIFEST_NAME
    if not manifest_path.is_file():
        raise ValueError(f"task 输出目录缺少 manifest：{manifest_path}")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as error:
        raise ValueError(f"task manifest 无法读取：{manifest_path}") from error

    for key in [
        "lm_eval_version",
        "lm_eval_commit",
        "trust_remote_code",
        "tasks",
        "task_sources",
    ]:
        if manifest.get(key) != config[key]:
            raise ValueError(f"现有 task 目录的 {key} 与配置不匹配")
    actual_sha256 = sha256_directory(destination, {TASK_MANIFEST_NAME})
    if manifest.get("task_directory_sha256") != actual_sha256:
        raise ValueError("现有 task 目录内容与 manifest hash 不匹配")
    return manifest


def prepare_task_directory(
    destination: Path,
    config: dict[str, Any],
) -> dict[str, Any]:
    """复制上游 task 配置，仅注入冻结的 Dataset revision。"""
    installed_version = importlib.metadata.version("lm_eval")
    if installed_version != config["lm_eval_version"]:
        raise RuntimeError(
            "lm_eval 版本不匹配："
            f"期望 {config['lm_eval_version']}，实际 {installed_version}"
        )
    if destination.exists():
        return reuse_existing_task_directory(destination, config)

    upstream_root = Path(str(importlib.resources.files("lm_eval") / "tasks"))
    temporary = destination.with_name(destination.name + ".tmp")
    if temporary.exists():
        raise FileExistsError(f"task 临时目录已经存在：{temporary}")
    temporary.mkdir(parents=True)
    modified_files: dict[str, list[str]] = {}
    try:
        for task_name, source_name in config["task_sources"].items():
            source_path = upstream_root / source_name
            target_path = temporary / source_name
            if not target_path.exists():
                shutil.copytree(
                    source_path,
                    target_path,
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
                )
            changed = []
            for task_path in sorted(item for item in target_path.rglob("*") if item.is_file()):
                try:
                    original = task_path.read_text(encoding="utf-8")
                except UnicodeDecodeError:
                    continue
                rendered, was_changed = inject_dataset_revision(
                    original,
                    config["tasks"][task_name],
                    config["trust_remote_code"],
                )
                if was_changed:
                    task_path.write_text(rendered, encoding="utf-8")
                    changed.append(str(task_path.relative_to(temporary)))
            if not changed:
                raise ValueError(f"task source 中没有可注入的 dataset_path：{source_name}")
            modified_files[task_name] = changed

        manifest = {
            "lm_eval_version": config["lm_eval_version"],
            "lm_eval_commit": config["lm_eval_commit"],
            "trust_remote_code": config["trust_remote_code"],
            "tasks": config["tasks"],
            "task_sources": config["task_sources"],
            "modified_files": modified_files,
        }
        manifest["task_directory_sha256"] = sha256_directory(temporary)
        (temporary / TASK_MANIFEST_NAME).write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary.replace(destination)
        return manifest
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("docs/stages/04_pretrain/configs/lm_eval.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/minimind/stage4/lm_eval_tasks"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_lm_eval_config(args.config)
    manifest = prepare_task_directory(args.output, config)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
