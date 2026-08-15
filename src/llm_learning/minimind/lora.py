"""阶段 6 使用的最小 LoRA 注入与 adapter 操作。"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import torch
from torch import nn


class LoRAAdapter(nn.Module):
    """计算低秩增量 B(A(x))，不包含原 Linear 分支。"""

    def __init__(self, in_features: int, out_features: int, rank: int) -> None:
        super().__init__()
        if rank < 1:
            raise ValueError("rank must be positive")
        self.rank = rank
        self.A = nn.Linear(in_features, rank, bias=False)
        self.B = nn.Linear(rank, out_features, bias=False)
        nn.init.normal_(self.A.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.B.weight)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.B(self.A(inputs))


def _target_linear_modules(
    model: nn.Module,
    target_module_names: Iterable[str],
) -> list[tuple[str, nn.Linear]]:
    target_names = tuple(dict.fromkeys(target_module_names))
    if not target_names:
        raise ValueError("target_module_names cannot be empty")
    matches = [
        (name, module)
        for name, module in model.named_modules()
        if isinstance(module, nn.Linear)
        and name.rsplit(".", 1)[-1] in target_names
    ]
    matched_leaf_names = {name.rsplit(".", 1)[-1] for name, _ in matches}
    missing = sorted(set(target_names) - matched_leaf_names)
    if missing:
        raise ValueError(f"target modules were not found: {missing}")
    return matches


def apply_lora(
    model: nn.Module,
    rank: int,
    target_module_names: Iterable[str],
) -> tuple[str, ...]:
    """为指定名称的 Linear 注入 LoRA 分支并返回完整模块名。"""
    matches = _target_linear_modules(model, target_module_names)
    injected_names = []
    for name, module in matches:
        if hasattr(module, "lora"):
            raise ValueError(f"LoRA is already attached to {name}")
        adapter = LoRAAdapter(
            module.in_features,
            module.out_features,
            rank,
        ).to(device=module.weight.device, dtype=module.weight.dtype)
        original_forward = module.forward

        def forward_with_lora(
            inputs: torch.Tensor,
            base_forward: Any = original_forward,
            lora_adapter: LoRAAdapter = adapter,
        ) -> torch.Tensor:
            return base_forward(inputs) + lora_adapter(inputs)

        module.add_module("lora", adapter)
        setattr(module, "_lora_original_forward", original_forward)
        module.forward = forward_with_lora
        injected_names.append(name)
    return tuple(injected_names)


def lora_target_names(model: nn.Module) -> tuple[str, ...]:
    """返回已经注入 LoRA 的 Linear 完整名称。"""
    return tuple(
        name
        for name, module in model.named_modules()
        if isinstance(module, nn.Linear) and hasattr(module, "lora")
    )


def freeze_non_lora_parameters(model: nn.Module) -> None:
    """冻结原模型，只保留 LoRA A、B 为可训练参数。"""
    for parameter in model.parameters():
        parameter.requires_grad = False
    for name, parameter in model.named_parameters():
        if ".lora." in name:
            parameter.requires_grad = True


def count_parameters(model: nn.Module) -> dict[str, int]:
    """统计总参数、可训练参数与 LoRA 参数。"""
    return {
        "total": sum(parameter.numel() for parameter in model.parameters()),
        "trainable": sum(
            parameter.numel()
            for parameter in model.parameters()
            if parameter.requires_grad
        ),
        "lora": sum(
            parameter.numel()
            for name, parameter in model.named_parameters()
            if ".lora." in name
        ),
    }


def adapter_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    """复制只包含 LoRA A、B 的 state dict。"""
    return {
        name: tensor.detach().cpu().clone()
        for name, tensor in model.state_dict().items()
        if ".lora." in name
    }


def load_adapter_state_dict(
    model: nn.Module,
    state_dict: Mapping[str, torch.Tensor],
) -> None:
    """严格验证名称和 shape 后加载 adapter 参数。"""
    expected = adapter_state_dict(model)
    expected_names = set(expected)
    received_names = set(state_dict)
    if expected_names != received_names:
        missing = sorted(expected_names - received_names)
        unexpected = sorted(received_names - expected_names)
        raise ValueError(
            f"adapter keys do not match; missing={missing}, unexpected={unexpected}"
        )
    parameters = dict(model.named_parameters())
    with torch.no_grad():
        for name, value in state_dict.items():
            parameter = parameters[name]
            if parameter.shape != value.shape:
                raise ValueError(
                    f"adapter shape does not match for {name}: "
                    f"expected {tuple(parameter.shape)}, got {tuple(value.shape)}"
                )
            parameter.copy_(value.to(parameter.device, parameter.dtype))


def save_adapter_checkpoint(
    path: Path,
    model: nn.Module,
    metadata: Mapping[str, Any],
) -> None:
    """原子保存 adapter 参数及其 Base、rank 和 target 身份。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "format": "minimind_lora_adapter_v1",
            "metadata": dict(metadata),
            "state_dict": {
                name: tensor.half()
                for name, tensor in adapter_state_dict(model).items()
            },
        },
        temporary_path,
    )
    temporary_path.replace(path)


def load_adapter_checkpoint(
    path: Path,
    model: nn.Module,
    expected_metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """验证 adapter 文件身份并严格加载 A、B。"""
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("format") != "minimind_lora_adapter_v1":
        raise ValueError("unsupported adapter checkpoint format")
    metadata = checkpoint.get("metadata")
    if not isinstance(metadata, dict):
        raise ValueError("adapter checkpoint metadata is missing")
    if expected_metadata is not None and metadata != dict(expected_metadata):
        raise ValueError("adapter checkpoint metadata does not match")
    state_dict = checkpoint.get("state_dict")
    if not isinstance(state_dict, dict):
        raise ValueError("adapter checkpoint state_dict is missing")
    load_adapter_state_dict(model, state_dict)
    return metadata


def merge_lora_(model: nn.Module) -> tuple[str, ...]:
    """把 B @ A 加回原 Linear，并移除 LoRA 分支。"""
    merged_names = []
    for name, module in list(model.named_modules()):
        if not isinstance(module, nn.Linear) or not hasattr(module, "lora"):
            continue
        adapter = getattr(module, "lora")
        if not isinstance(adapter, LoRAAdapter):
            raise TypeError(f"unsupported LoRA module on {name}")
        with torch.no_grad():
            increment = adapter.B.weight @ adapter.A.weight
            module.weight.add_(increment.to(module.weight.dtype))
        module.forward = getattr(module, "_lora_original_forward")
        delattr(module, "_lora_original_forward")
        delattr(module, "lora")
        merged_names.append(name)
    return tuple(merged_names)
