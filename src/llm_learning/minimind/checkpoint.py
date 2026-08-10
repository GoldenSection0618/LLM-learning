"""MiniMind 阶段 4 的可恢复训练状态。"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import torch
from torch import nn


RUNTIME_PATH_FIELDS = {
    "source_dir",
    "data_path",
    "output_dir",
    "checkpoint_dir",
    "initial_weights_path",
}


def training_config_identity(config: dict[str, Any]) -> dict[str, Any]:
    """排除不影响训练数值的本地路径。"""
    return {
        name: value
        for name, value in config.items()
        if name not in RUNTIME_PATH_FIELDS
    }


def capture_rng_state() -> dict[str, Any]:
    """捕获 Python、CPU 与 CUDA 随机数状态。"""
    return {
        "python": random.getstate(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": (
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        ),
    }


def restore_rng_state(state: dict[str, Any]) -> None:
    """恢复 checkpoint 中的随机数状态。"""
    random.setstate(state["python"])
    torch.set_rng_state(state["torch_cpu"].cpu())
    cuda_states = state.get("torch_cuda")
    if cuda_states is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([item.cpu() for item in cuda_states])


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.GradScaler,
    training_state: dict[str, Any],
    config: dict[str, Any],
    tokenizer: dict[str, Any],
    data_manifest: dict[str, Any],
    history: list[dict[str, Any]],
) -> None:
    """原子保存只能位于 optimizer 边界的完整训练状态。"""
    if training_state.get("microbatches_in_update", 0) != 0:
        raise ValueError("checkpoint cannot contain a partial accumulation window")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "training_state": training_state,
            "config": config,
            "tokenizer": tokenizer,
            "data_manifest": data_manifest,
            "history": history,
            "rng_state": capture_rng_state(),
        },
        temporary_path,
    )
    temporary_path.replace(path)


def load_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.GradScaler,
    expected_config: dict[str, Any],
    expected_tokenizer: dict[str, Any],
    expected_data_manifest: dict[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    """验证运行身份后恢复模型、优化器、scaler 与 RNG。"""
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if training_config_identity(checkpoint["config"]) != training_config_identity(
        expected_config
    ):
        raise ValueError("checkpoint config does not match the current run")
    if checkpoint["tokenizer"] != expected_tokenizer:
        raise ValueError("checkpoint tokenizer does not match the current run")
    if checkpoint["data_manifest"]["split_sha256"] != expected_data_manifest[
        "split_sha256"
    ]:
        raise ValueError("checkpoint data split does not match the current run")
    if checkpoint["data_manifest"]["raw_sha256"] != expected_data_manifest[
        "raw_sha256"
    ]:
        raise ValueError("checkpoint data file does not match the current run")
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    scaler.load_state_dict(checkpoint["scaler"])
    restore_rng_state(checkpoint["rng_state"])
    return checkpoint


def export_model_weights(path: Path, model: nn.Module) -> None:
    """原子导出供生成与格式转换使用的 FP16 CPU 权重。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    state_dict = {
        name: tensor.detach().half().cpu()
        for name, tensor in model.state_dict().items()
    }
    torch.save(state_dict, temporary_path)
    temporary_path.replace(path)
