from __future__ import annotations

import hashlib
import json
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import Dataset

TINY_SHAKESPEARE_URL = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/"
    "6f9487a/data/tinyshakespeare/input.txt"
)

# 固定数据版本和完整哈希，保证每次实验使用同一份语料。
TINY_SHAKESPEARE_SHA256 = (
    "86c4e6aa9db7c042ec79f339dcb96d42"
    "b0075e16b8fc2e86bf0ca57e2dc565ed"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_tiny_shakespeare(path: Path) -> Path:
    """下载固定版本的 Tiny Shakespeare，并验证完整哈希。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        digest = sha256_file(path)

        if digest == TINY_SHAKESPEARE_SHA256:
            return path

    temporary_path = path.with_suffix(path.suffix + ".part")
    try:
        # 先验证临时文件，再原子覆盖缺失或损坏的正式文件。
        urllib.request.urlretrieve(TINY_SHAKESPEARE_URL, temporary_path)
        digest = sha256_file(temporary_path)

        if digest != TINY_SHAKESPEARE_SHA256:
            raise ValueError(
                f"Downloaded dataset has an unexpected SHA-256: {digest}"
            )

        temporary_path.replace(path)
    finally:
        # 下载或校验失败时清理残留，已有正式文件保持不变。
        temporary_path.unlink(missing_ok=True)

    return path


def split_token_ids(
    token_ids: list[int],
    train_fraction: float = 0.9,
) -> tuple[list[int], list[int], int]:
    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must be between 0 and 1")

    split_index = int(len(token_ids) * train_fraction)

    # 按原始顺序切分，验证集来自训练集之后的文本。
    return token_ids[:split_index], token_ids[split_index:], split_index


@dataclass(frozen=True)
class DataManifest:
    source_url: str
    sha256: str
    total_characters: int
    train_characters: int
    validation_characters: int
    split_index: int
    train_fraction: float
    vocab_size: int

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.__dict__, indent=2) + "\n",
            encoding="utf-8",
        )


class NextTokenDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """返回定长输入，以及向后错开一个 token 的目标序列。"""

    def __init__(self, token_ids: list[int], block_size: int) -> None:
        if block_size < 1:
            raise ValueError("block_size must be positive")

        if len(token_ids) <= block_size:
            raise ValueError("The token sequence must be longer than block_size")

        self.tokens = torch.tensor(token_ids, dtype=torch.long)
        self.block_size = block_size

    def __len__(self) -> int:
        # 每个起点都需要 block_size 个输入和紧随其后的一个目标。
        return len(self.tokens) - self.block_size

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        # targets 相对 inputs 左移一个 token，形状都为 [T]。
        inputs = self.tokens[index : index + self.block_size]
        targets = self.tokens[index + 1 : index + self.block_size + 1]
        return inputs, targets
