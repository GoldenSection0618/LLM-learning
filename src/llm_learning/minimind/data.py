"""阶段 4 固定数据下载、切分与预训练样本编码。"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping, Sequence, TypeAlias, cast

import torch
from huggingface_hub import hf_hub_download
from torch.utils.data import Dataset, Sampler

from .config import PretrainConfig, load_pretrain_config

if TYPE_CHECKING:
    from datasets import Dataset as HuggingFaceDataset
else:
    HuggingFaceDataset = Any


RecordCollection: TypeAlias = (
    Sequence[Mapping[str, Any]] | HuggingFaceDataset
)


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """流式计算文件 SHA-256。"""
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while chunk := file.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    """计算规范 JSON 的 SHA-256。"""
    rendered = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(rendered).hexdigest()


@dataclass(frozen=True)
class DataLock:
    """固定 Hugging Face 数据文件的上游身份。"""

    repo_id: str
    repo_type: str
    revision: str
    filename: str
    size: int
    sha256: str

    @classmethod
    def load(cls, path: Path) -> "DataLock":
        return cls(**json.loads(path.read_text(encoding="utf-8")))


def download_locked_file(lock: DataLock, destination: Path) -> Path:
    """下载并验证固定文件，目标异常时拒绝覆盖。"""
    if destination.is_file():
        if destination.stat().st_size != lock.size:
            raise ValueError(f"现有数据文件大小不匹配：{destination}")
        if sha256_file(destination) != lock.sha256:
            raise ValueError(f"现有数据文件 SHA-256 不匹配：{destination}")
        return destination

    last_error: Exception | None = None
    for attempt in range(3):
        try:
            cached_path = Path(
                hf_hub_download(
                    repo_id=lock.repo_id,
                    repo_type=lock.repo_type,
                    revision=lock.revision,
                    filename=lock.filename,
                )
            )
            break
        except Exception as error:
            last_error = error
            if attempt < 2:
                time.sleep(2**attempt)
    else:
        raise RuntimeError("固定数据文件下载连续失败") from last_error
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = destination.with_suffix(destination.suffix + ".tmp")
    shutil.copy2(cached_path, temporary_path)
    if temporary_path.stat().st_size != lock.size:
        temporary_path.unlink(missing_ok=True)
        raise ValueError("下载文件大小与 data lock 不匹配")
    actual_sha256 = sha256_file(temporary_path)
    if actual_sha256 != lock.sha256:
        temporary_path.unlink(missing_ok=True)
        raise ValueError(
            f"下载文件 SHA-256 不匹配：期望 {lock.sha256}，实际 {actual_sha256}"
        )
    temporary_path.replace(destination)
    return destination


def tokenizer_identity(tokenizer_dir: Path, tokenizer: Any) -> dict[str, Any]:
    """记录 tokenizer 文件与 token ID 约定。"""
    files = {}
    for name in ["tokenizer.json", "tokenizer_config.json"]:
        path = tokenizer_dir / name
        files[name] = sha256_file(path)
    return {
        "files": files,
        "vocab_size": len(tokenizer),
        "pad_token_id": tokenizer.pad_token_id,
        "bos_token_id": tokenizer.bos_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }


class MiniMindPretrainDataset(Dataset[dict[str, torch.Tensor]]):
    """按固定 row ID 将官方 text schema 编码为预训练样本。"""

    def __init__(
        self,
        records: RecordCollection,
        row_ids: Sequence[int],
        tokenizer: Any,
        sequence_length: int,
    ) -> None:
        self.records = records
        self.row_ids = list(row_ids)
        self.tokenizer = tokenizer
        self.sequence_length = sequence_length

    def __len__(self) -> int:
        return len(self.row_ids)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        row_id = self.row_ids[index]
        sample = self.records[row_id]
        token_ids = self.tokenizer(
            str(sample["text"]),
            add_special_tokens=False,
            max_length=self.sequence_length - 2,
            truncation=True,
        ).input_ids
        token_ids = [self.tokenizer.bos_token_id, *token_ids, self.tokenizer.eos_token_id]
        padding_length = self.sequence_length - len(token_ids)
        input_ids = token_ids + [self.tokenizer.pad_token_id] * padding_length
        labels = input_ids.copy()
        for position in range(len(token_ids), self.sequence_length):
            labels[position] = -100
        attention_mask = [1] * len(token_ids) + [0] * padding_length
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.bool),
            "row_id": torch.tensor(row_id, dtype=torch.long),
        }


class EpochBatchSampler(Sampler[list[int]]):
    """按 epoch seed 生成可重建并支持跳过 batch 的顺序。"""

    def __init__(
        self,
        dataset_size: int,
        batch_size: int,
        seed: int,
        epoch: int,
        start_batch: int = 0,
    ) -> None:
        self.dataset_size = dataset_size
        self.batch_size = batch_size
        self.seed = seed
        self.epoch = epoch
        self.start_batch = start_batch

    @property
    def total_batches(self) -> int:
        return (self.dataset_size + self.batch_size - 1) // self.batch_size

    def __len__(self) -> int:
        return max(0, self.total_batches - self.start_batch)

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        order = torch.randperm(self.dataset_size, generator=generator).tolist()
        batches = [
            order[start : start + self.batch_size]
            for start in range(0, self.dataset_size, self.batch_size)
        ]
        yield from batches[self.start_batch :]


def load_json_dataset(path: Path) -> RecordCollection:
    """使用 Hugging Face Datasets 读取官方 JSONL。"""
    from datasets import load_dataset

    return cast(
        RecordCollection,
        load_dataset("json", data_files=str(path), split="train"),
    )


def create_split_manifest(
    config: PretrainConfig,
    records: RecordCollection,
    raw_sha256: str,
    dataset_fingerprint: str | None,
    tokenizer_info: dict[str, Any],
) -> dict[str, Any]:
    """创建过拟合或正式训练使用的确定性 split 清单。"""
    row_count = len(records)
    if config.overfit_rows:
        if row_count < config.overfit_rows:
            raise ValueError("数据行数小于 overfit_rows")
        train_row_ids = list(range(config.overfit_rows))
        validation_row_ids = train_row_ids
        split_rule = "first_rows_for_overfit"
    else:
        if row_count <= config.validation_rows:
            raise ValueError("数据行数不足以划分 validation")
        generator = torch.Generator().manual_seed(config.seed)
        order = torch.randperm(row_count, generator=generator).tolist()
        validation_row_ids = order[: config.validation_rows]
        validation_set = set(validation_row_ids)
        train_row_ids = [index for index in range(row_count) if index not in validation_set]
        split_rule = "seeded_validation_then_train_complement"

    split_identity = {
        "seed": config.seed,
        "rule": split_rule,
        "row_count": row_count,
        "train_count": len(train_row_ids),
        "validation_row_ids": validation_row_ids,
        "overfit_row_ids": train_row_ids if config.overfit_rows else [],
    }
    return {
        "data_path": str(config.data_path),
        "raw_sha256": raw_sha256,
        "dataset_fingerprint": dataset_fingerprint,
        "tokenizer": tokenizer_info,
        "split": split_identity,
        "split_sha256": sha256_json(split_identity),
    }


def resolve_split_row_ids(manifest: dict[str, Any]) -> tuple[list[int], list[int]]:
    """根据紧凑 split 清单恢复训练与验证 row ID。"""
    split = manifest["split"]
    validation_row_ids = list(split["validation_row_ids"])
    if split["rule"] == "first_rows_for_overfit":
        train_row_ids = list(split["overfit_row_ids"])
    else:
        validation_set = set(validation_row_ids)
        train_row_ids = [
            index
            for index in range(int(split["row_count"]))
            if index not in validation_set
        ]
    return train_row_ids, validation_row_ids


def save_manifest(path: Path, manifest: dict[str, Any]) -> None:
    """保存运行时数据清单。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--lock",
        type=Path,
        default=Path("docs/stages/04_pretrain/configs/data_lock.json"),
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("docs/stages/04_pretrain/configs/overfit256.json"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    lock = DataLock.load(args.lock)
    config = load_pretrain_config(args.config)
    path = download_locked_file(lock, config.data_path)
    print(path)


if __name__ == "__main__":
    main()
