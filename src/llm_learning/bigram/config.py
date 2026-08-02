from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class TrainConfig:
    # 数据与训练产物的位置。
    data_path: Path = Path("data/tinyshakespeare/input.txt")
    output_dir: Path = Path("outputs/bigram")
    checkpoint_dir: Path = Path("checkpoints/bigram")
    # 最小训练实验的核心超参数。
    train_fraction: float = 0.9
    block_size: int = 128
    batch_size: int = 64
    max_steps: int = 2_000
    eval_interval: int = 200
    eval_batches: int = 50
    learning_rate: float = 1e-2
    seed: int = 1337
    generate_tokens: int = 500
    device: str = "auto"

    def to_dict(self) -> dict[str, object]:
        values = asdict(self)
        # Path 转成字符串后才能直接写入 JSON。
        values["data_path"] = str(self.data_path)
        values["output_dir"] = str(self.output_dir)
        values["checkpoint_dir"] = str(self.checkpoint_dir)
        return values
