from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from .model import BigramLanguageModel


def save_checkpoint(
    path: Path,
    model: BigramLanguageModel,
    optimizer: torch.optim.Optimizer,
    step: int,
    config: dict[str, object],
    tokenizer: dict[str, list[str]],
    history: list[dict[str, float | int]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    # 先写临时文件，再原子替换，避免中断时留下半个 checkpoint。
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    cuda_rng_state = (
        torch.cuda.get_rng_state_all()
        if torch.cuda.is_available()
        else None
    )

    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "step": step,
            "config": config,
            "tokenizer": tokenizer,
            "history": history,
            # 保存随机数状态，恢复训练后可以延续原来的随机序列。
            "cpu_rng_state": torch.get_rng_state(),
            "cuda_rng_state": cuda_rng_state,
        },
        temporary_path,
    )
    temporary_path.replace(path)


def load_checkpoint(
    path: Path,
    model: BigramLanguageModel,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
) -> dict[str, Any]:
    # map_location 让同一份 checkpoint 可以在 CPU 或 GPU 上恢复。
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])

    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer"])

    torch.set_rng_state(checkpoint["cpu_rng_state"].cpu())
    cuda_rng_state = checkpoint.get("cuda_rng_state")

    if cuda_rng_state is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([state.cpu() for state in cuda_rng_state])

    return checkpoint
