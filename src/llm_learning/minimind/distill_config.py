"""阶段 8 序列级与 logit distillation 配置。"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class SequenceTeacherConfig:
    """固定 LM Studio Teacher 生成参数。"""

    dataset_id: str
    dataset_config: str
    dataset_split: str
    dataset_revision: str
    seed: int
    development_rows: int
    teaching_rows: int
    api_base: str
    model_name_contains: str
    temperature: float
    top_k: int
    top_p: float
    presence_penalty: float
    max_tokens: int
    request_timeout_seconds: int
    split_manifest_path: Path
    raw_output_path: Path
    verified_sft_path: Path

    def __post_init__(self) -> None:
        if self.development_rows < 1 or self.teaching_rows < 1:
            raise ValueError("split sizes must be positive")
        if self.temperature < 0:
            raise ValueError("temperature cannot be negative")
        if self.top_k < 1:
            raise ValueError("top_k must be positive")
        if not 0 < self.top_p <= 1:
            raise ValueError("top_p must be in (0, 1]")
        if self.max_tokens < 1 or self.request_timeout_seconds < 1:
            raise ValueError("generation limits must be positive")


@dataclass(frozen=True)
class LogitDistillationConfig:
    """固定一次 MiniMind CE / KD 对照。"""

    profile: str
    base_sft_config: Path
    source_dir: Path
    data_path: Path
    student_weights_path: Path
    teacher_weights_path: Path
    output_dir: Path
    checkpoint_dir: Path
    seed: int
    device: str
    dtype: str
    sequence_length: int
    batch_size: int
    accumulation_steps: int
    epochs: int
    learning_rate: float
    minimum_lr_ratio: float
    gradient_clip: float
    train_rows: int
    validation_rows: int
    eval_interval: int
    checkpoint_interval: int
    num_workers: int
    ce_weight: float
    temperature: float
    kl_direction: str

    def __post_init__(self) -> None:
        positive = {
            "sequence_length": self.sequence_length,
            "batch_size": self.batch_size,
            "accumulation_steps": self.accumulation_steps,
            "epochs": self.epochs,
            "train_rows": self.train_rows,
            "validation_rows": self.validation_rows,
            "eval_interval": self.eval_interval,
            "checkpoint_interval": self.checkpoint_interval,
        }
        for name, value in positive.items():
            if value < 1:
                raise ValueError(f"{name} must be positive")
        if not 0 <= self.ce_weight <= 1:
            raise ValueError("ce_weight must be in [0, 1]")
        if self.temperature <= 0:
            raise ValueError("temperature must be positive")
        if self.kl_direction not in {"forward", "reverse"}:
            raise ValueError("kl_direction must be 'forward' or 'reverse'")

    @property
    def effective_batch_size(self) -> int:
        """返回每次 optimizer step 累计的序列数。"""
        return self.batch_size * self.accumulation_steps

    def model_kwargs(self, vocab_size: int, *, use_moe: bool) -> dict[str, Any]:
        """构造本阶段固定的 64M Dense / 198M MoE 结构。"""
        return {
            "hidden_size": 768,
            "num_hidden_layers": 8,
            "num_attention_heads": 8,
            "num_key_value_heads": 4,
            "vocab_size": vocab_size,
            "max_position_embeddings": 32768,
            "flash_attn": True,
            "use_moe": use_moe,
            "dropout": 0.0,
        }


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_sequence_teacher_config(path: Path) -> SequenceTeacherConfig:
    """读取序列级 Teacher 配置。"""
    values = _load(path)
    values["api_base"] = os.environ.get(
        "LM_STUDIO_BASE_URL",
        values["api_base"],
    )
    for name in ["split_manifest_path", "raw_output_path", "verified_sft_path"]:
        values[name] = Path(values[name])
    return SequenceTeacherConfig(**values)


def load_logit_distillation_config(path: Path) -> LogitDistillationConfig:
    """读取 logit distillation 配置。"""
    values = _load(path)
    for name in [
        "source_dir",
        "base_sft_config",
        "data_path",
        "student_weights_path",
        "teacher_weights_path",
        "output_dir",
        "checkpoint_dir",
    ]:
        values[name] = Path(values[name])
    return LogitDistillationConfig(**values)
