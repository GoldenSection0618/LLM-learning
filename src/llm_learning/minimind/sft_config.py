"""阶段 5 MiniMind SFT 配置。"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .config import PretrainConfig


@dataclass(frozen=True)
class SFTConfig(PretrainConfig):
    """固定一次 MiniMind SFT 运行的核心参数。"""

    initial_weights_path: Path
    inspection_rows: int
    add_system_ratio: float
    empty_think_ratio: float

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.inspection_rows < 1:
            raise ValueError("inspection_rows must be positive")
        for name, value in {
            "add_system_ratio": self.add_system_ratio,
            "empty_think_ratio": self.empty_think_ratio,
        }.items():
            if not 0 <= value <= 1:
                raise ValueError(f"{name} must be in [0, 1]")
        if self.overfit_rows and self.overfit_rows > self.inspection_rows:
            raise ValueError("overfit_rows cannot exceed inspection_rows")

    def to_dict(self) -> dict[str, object]:
        values = super().to_dict()
        values["initial_weights_path"] = str(self.initial_weights_path)
        return values


def load_sft_config(path: Path) -> SFTConfig:
    """从 JSON 文件加载阶段 5 配置。"""
    values = json.loads(path.read_text(encoding="utf-8"))
    for name in [
        "source_dir",
        "data_path",
        "output_dir",
        "checkpoint_dir",
        "initial_weights_path",
    ]:
        values[name] = Path(values[name])
    return SFTConfig(**values)
