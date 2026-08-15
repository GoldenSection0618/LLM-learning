"""阶段 6 MiniMind LoRA 对照配置。"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class LoRAConfig:
    """固定 LoRA 专属参数及其阶段 5 基线引用。"""

    profile: str
    base_sft_config: Path
    base_weights_path: Path
    base_weights_sha256: str
    rank: int
    target_modules: tuple[str, ...]
    learning_rate: float
    stop_after_step: int
    output_dir: Path
    checkpoint_dir: Path

    def __post_init__(self) -> None:
        if self.rank < 1:
            raise ValueError("rank must be positive")
        if not self.target_modules:
            raise ValueError("target_modules cannot be empty")
        if len(set(self.target_modules)) != len(self.target_modules):
            raise ValueError("target_modules cannot contain duplicates")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.stop_after_step < 1:
            raise ValueError("stop_after_step must be positive")
        if len(self.base_weights_sha256) != 64:
            raise ValueError("base_weights_sha256 must contain 64 characters")

    def to_dict(self) -> dict[str, object]:
        """转换为可以写入运行记录的字典。"""
        values = asdict(self)
        for name in [
            "base_sft_config",
            "base_weights_path",
            "output_dir",
            "checkpoint_dir",
        ]:
            values[name] = str(values[name])
        values["target_modules"] = list(self.target_modules)
        return values


def load_lora_config(path: Path) -> LoRAConfig:
    """从 JSON 文件加载阶段 6 配置。"""
    values = json.loads(path.read_text(encoding="utf-8"))
    for name in [
        "base_sft_config",
        "base_weights_path",
        "output_dir",
        "checkpoint_dir",
    ]:
        values[name] = Path(values[name])
    values["target_modules"] = tuple(values["target_modules"])
    return LoRAConfig(**values)
