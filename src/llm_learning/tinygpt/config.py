from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class ModelConfig:
    """TinyGPT 的结构配置。"""

    vocab_size: int
    block_size: int = 128
    d_model: int = 256
    num_heads: int = 4
    num_layers: int = 4
    dropout: float = 0.1

    def __post_init__(self) -> None:
        if self.vocab_size < 1:
            raise ValueError("vocab_size must be positive")
        if self.block_size < 1:
            raise ValueError("block_size must be positive")
        if self.d_model < 1:
            raise ValueError("d_model must be positive")
        if self.num_heads < 1:
            raise ValueError("num_heads must be positive")
        if self.num_layers < 1:
            raise ValueError("num_layers must be positive")
        if self.d_model % self.num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")

    @property
    def head_dim(self) -> int:
        return self.d_model // self.num_heads

    def to_dict(self) -> dict[str, int | float]:
        return asdict(self)


@dataclass(frozen=True)
class TrainConfig:
    """TinyGPT 的数据与训练配置。"""

    data_path: Path = Path("data/tinyshakespeare/input.txt")
    output_dir: Path = Path("outputs/tinygpt")
    train_fraction: float = 0.9
    block_size: int = 128
    batch_size: int = 64
    max_steps: int = 2_000
    eval_interval: int = 200
    eval_batches: int = 20
    learning_rate: float = 3e-4
    d_model: int = 256
    num_heads: int = 4
    num_layers: int = 4
    dropout: float = 0.1
    seed: int = 2026
    generate_tokens: int = 300
    device: str = "auto"

    def __post_init__(self) -> None:
        if not 0.0 < self.train_fraction < 1.0:
            raise ValueError("train_fraction must be between 0 and 1")
        if self.batch_size < 1:
            raise ValueError("batch_size must be positive")
        if self.max_steps < 1:
            raise ValueError("max_steps must be positive")
        if self.eval_interval < 1:
            raise ValueError("eval_interval must be positive")
        if self.eval_batches < 1:
            raise ValueError("eval_batches must be positive")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.generate_tokens < 0:
            raise ValueError("generate_tokens cannot be negative")

        # 复用模型配置中的结构约束，包括 block_size 和 head 划分。
        self.model_config(vocab_size=1)

    def model_config(self, vocab_size: int) -> ModelConfig:
        return ModelConfig(
            vocab_size=vocab_size,
            block_size=self.block_size,
            d_model=self.d_model,
            num_heads=self.num_heads,
            num_layers=self.num_layers,
            dropout=self.dropout,
        )

    def to_dict(self) -> dict[str, object]:
        values = asdict(self)
        values["data_path"] = str(self.data_path)
        values["output_dir"] = str(self.output_dir)
        return values
