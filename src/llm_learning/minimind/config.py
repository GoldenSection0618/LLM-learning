"""阶段 4 MiniMind 预训练配置。"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class PretrainConfig:
    """固定一次 MiniMind 预训练运行的全部核心参数。"""

    profile: str
    source_dir: Path
    data_path: Path
    output_dir: Path
    checkpoint_dir: Path
    seed: int
    device: str
    dtype: str
    hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    sequence_length: int
    batch_size: int
    accumulation_steps: int
    epochs: int
    max_optimizer_steps: int | None
    learning_rate: float
    minimum_lr_ratio: float
    gradient_clip: float
    eval_interval: int
    checkpoint_interval: int
    num_workers: int
    overfit_rows: int
    validation_rows: int
    periodic_validation_rows: int
    overfit_loss_target: float | None

    def __post_init__(self) -> None:
        positive_values = {
            "hidden_size": self.hidden_size,
            "num_hidden_layers": self.num_hidden_layers,
            "num_attention_heads": self.num_attention_heads,
            "num_key_value_heads": self.num_key_value_heads,
            "sequence_length": self.sequence_length,
            "batch_size": self.batch_size,
            "accumulation_steps": self.accumulation_steps,
            "epochs": self.epochs,
            "eval_interval": self.eval_interval,
            "checkpoint_interval": self.checkpoint_interval,
        }
        for name, value in positive_values.items():
            if value < 1:
                raise ValueError(f"{name} must be positive")
        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError("hidden_size must be divisible by num_attention_heads")
        if self.num_attention_heads % self.num_key_value_heads != 0:
            raise ValueError(
                "num_attention_heads must be divisible by num_key_value_heads"
            )
        if self.dtype not in {"float32", "float16", "bfloat16"}:
            raise ValueError("dtype must be float32, float16, or bfloat16")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if not 0 < self.minimum_lr_ratio <= 1:
            raise ValueError("minimum_lr_ratio must be in (0, 1]")
        if self.gradient_clip <= 0:
            raise ValueError("gradient_clip must be positive")
        if self.max_optimizer_steps is not None and self.max_optimizer_steps < 1:
            raise ValueError("max_optimizer_steps must be positive when provided")
        count_values = {
            "num_workers": self.num_workers,
            "overfit_rows": self.overfit_rows,
            "validation_rows": self.validation_rows,
            "periodic_validation_rows": self.periodic_validation_rows,
        }
        for name, value in count_values.items():
            if value < 0:
                raise ValueError(f"{name} cannot be negative")
        if bool(self.overfit_rows) == bool(self.validation_rows):
            raise ValueError("enable exactly one of overfit_rows or validation_rows")
        if self.periodic_validation_rows > self.validation_rows:
            raise ValueError(
                "periodic_validation_rows cannot exceed validation_rows"
            )
        if self.overfit_loss_target is not None and not self.overfit_rows:
            raise ValueError("overfit_loss_target requires overfit_rows")

    @property
    def effective_batch_size(self) -> int:
        """返回单卡运行中每次参数更新包含的序列数。"""
        return self.batch_size * self.accumulation_steps

    def model_kwargs(self, vocab_size: int) -> dict[str, Any]:
        """构造官方 MiniMindConfig 使用的字段。"""
        return {
            "hidden_size": self.hidden_size,
            "num_hidden_layers": self.num_hidden_layers,
            "num_attention_heads": self.num_attention_heads,
            "num_key_value_heads": self.num_key_value_heads,
            "vocab_size": vocab_size,
            "max_position_embeddings": max(32768, self.sequence_length),
            "flash_attn": True,
            "use_moe": False,
            "dropout": 0.0,
        }

    def to_dict(self) -> dict[str, Any]:
        """转换为可以写入 JSON 的字典。"""
        values = asdict(self)
        for name in ["source_dir", "data_path", "output_dir", "checkpoint_dir"]:
            values[name] = str(values[name])
        return values


def load_pretrain_config(path: Path) -> PretrainConfig:
    """从 JSON 文件加载阶段 4 配置。"""
    values = json.loads(path.read_text(encoding="utf-8"))
    for name in ["source_dir", "data_path", "output_dir", "checkpoint_dir"]:
        values[name] = Path(values[name])
    return PretrainConfig(**values)


def load_training_config(path: Path) -> PretrainConfig:
    """根据配置字段加载 Pretrain 或 SFT 配置。"""
    values = json.loads(path.read_text(encoding="utf-8"))
    if "initial_weights_path" in values:
        from .sft_config import load_sft_config

        return load_sft_config(path)
    return load_pretrain_config(path)
