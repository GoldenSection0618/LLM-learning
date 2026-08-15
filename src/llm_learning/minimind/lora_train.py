"""阶段 6 MiniMind rank-16 LoRA 单卡训练入口。"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import torch
from torch import nn

from .checkpoint import export_model_weights
from .data import sha256_file
from .lora import (
    apply_lora,
    count_parameters,
    freeze_non_lora_parameters,
    merge_lora_,
    save_adapter_checkpoint,
)
from .lora_config import LoRAConfig, load_lora_config
from .sft_config import SFTConfig, load_sft_config
from .sft_data import (
    build_sft_dataset,
    create_sft_split_manifest,
    load_sft_json_dataset,
    resolve_sft_split_row_ids,
)
from .train import (
    autocast_context,
    build_model_and_tokenizer,
    cuda_device_index,
    make_grad_scaler,
    resolve_device,
    run_training,
    seed_everything,
)


def build_lora_training_config(config: LoRAConfig) -> SFTConfig:
    """把阶段 5 数据配置与阶段 6 训练预算合成一次运行配置。"""
    base = load_sft_config(config.base_sft_config)
    return replace(
        base,
        profile=config.profile,
        output_dir=config.output_dir,
        checkpoint_dir=config.checkpoint_dir,
        epochs=1,
        max_optimizer_steps=config.stop_after_step,
        learning_rate=config.learning_rate,
        initial_weights_path=config.base_weights_path,
    )


def lora_run_identity(config: LoRAConfig) -> dict[str, Any]:
    """返回会影响 adapter 数值或结构的固定身份。"""
    return {
        "kind": "lora",
        "rank": config.rank,
        "target_module_names": list(config.target_modules),
        "base_weights_sha256": config.base_weights_sha256,
    }


def verify_base_weights(config: LoRAConfig) -> None:
    """确认训练从配置指定的 Base checkpoint 开始。"""
    if not config.base_weights_path.is_file():
        raise FileNotFoundError(
            f"找不到 Base checkpoint：{config.base_weights_path}"
        )
    actual = sha256_file(config.base_weights_path)
    if actual != config.base_weights_sha256:
        raise ValueError("Base checkpoint does not match the LoRA config")


def prepare_lora_model(
    model: nn.Module,
    config: LoRAConfig,
) -> dict[str, Any]:
    """向真实 MiniMind 注入 LoRA、冻结 Base 并验证目标范围。"""
    target_names = apply_lora(model, config.rank, config.target_modules)
    num_layers = int(model.config.num_hidden_layers)
    expected_count = num_layers * len(config.target_modules)
    if len(target_names) != expected_count:
        raise ValueError(
            "LoRA target count does not match the model config: "
            f"expected {expected_count}, got {len(target_names)}"
        )
    freeze_non_lora_parameters(model)
    counts = count_parameters(model)
    if counts["trainable"] != counts["lora"]:
        raise ValueError("non-LoRA parameters remain trainable")
    result = {
        "rank": config.rank,
        "target_modules": list(target_names),
        "target_module_count": len(target_names),
        "total_parameter_count": counts["total"],
        "trainable_parameter_count": counts["trainable"],
        "lora_parameter_count": counts["lora"],
    }
    print(
        f"[lora] targets={len(target_names)} "
        f"trainable_parameters={counts['trainable']:,}",
        flush=True,
    )
    for name in target_names:
        print(f"[lora] target={name}", flush=True)
    return result


def merge_and_export(
    model: nn.Module,
    tokenizer: Any,
    path: Path,
) -> dict[str, Any]:
    """比较 merge 前后 logits，并导出普通 MiniMind 权重。"""
    device = next(model.parameters()).device
    input_ids = tokenizer(
        "LoRA merge consistency check",
        add_special_tokens=False,
        return_tensors="pt",
    ).input_ids.to(device)
    model.eval()
    with torch.no_grad():
        before = model(input_ids, use_cache=False).logits.float().cpu()
    merged_names = merge_lora_(model)
    with torch.no_grad():
        after = model(input_ids, use_cache=False).logits.float().cpu()
    maximum_difference = float((before - after).abs().max().item())
    if maximum_difference > 1e-4:
        raise ValueError(
            "merge changed logits beyond tolerance: "
            f"max_abs_diff={maximum_difference}"
        )
    export_model_weights(path, model)
    return {
        "target_modules": list(merged_names),
        "max_logit_abs_diff": maximum_difference,
        "tolerance": 1e-4,
        "merged_weights": str(path),
    }


def run_lora_training(
    config_path: Path,
    resume_path: Path | None = None,
    stop_after_step: int | None = None,
) -> dict[str, Any]:
    """从阶段 4 Base weights 启动或恢复一次正式 LoRA SFT。"""
    lora_config = load_lora_config(config_path)
    training_config = build_lora_training_config(lora_config)
    verify_base_weights(lora_config)
    adapter_metadata: dict[str, Any] = {
        "base_weights_sha256": lora_config.base_weights_sha256,
        "rank": lora_config.rank,
        "target_modules": [],
    }

    def setup(model: nn.Module) -> dict[str, Any]:
        result = prepare_lora_model(model, lora_config)
        adapter_metadata["target_modules"] = result["target_modules"]
        return result

    def export_adapter(path: Path, model: nn.Module) -> None:
        save_adapter_checkpoint(path, model, adapter_metadata)

    result = run_training(
        training_config,
        resume_path,
        stop_after_step=stop_after_step,
        dataset_factory=build_sft_dataset,
        manifest_factory=create_sft_split_manifest,
        split_resolver=resolve_sft_split_row_ids,
        records_loader=load_sft_json_dataset,
        initial_weights_path=lora_config.base_weights_path,
        model_setup=setup,
        weights_exporter=export_adapter,
        weights_prefix="adapter",
        run_metadata=lora_run_identity(lora_config),
    )
    final_step = int(result["training_state"]["optimizer_step"])
    adapter_path = (
        training_config.checkpoint_dir / f"adapter_step_{final_step}.pth"
    )
    merged_path = (
        training_config.checkpoint_dir
        / f"merged_weights_step_{final_step}.pth"
    )
    merge_result = merge_and_export(
        result["model"],
        result["tokenizer"],
        merged_path,
    )
    summary = result["summary"]
    summary["adapter"] = str(adapter_path)
    summary["merge_validation"] = merge_result
    (training_config.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return result


def run_capacity_check(config_path: Path) -> dict[str, Any]:
    """用正式 shape 完成一次 LoRA forward、backward 与 AdamW step。"""
    lora_config = load_lora_config(config_path)
    config = build_lora_training_config(lora_config)
    verify_base_weights(lora_config)
    seed_everything(config.seed)
    device = resolve_device(config.device)
    if device.type == "cuda":
        torch.cuda.init()
        torch.cuda.reset_peak_memory_stats(cuda_device_index(device))

    print(f"[capacity] loading model on {device}", flush=True)
    model, tokenizer = build_model_and_tokenizer(config, device)
    state_dict = torch.load(
        lora_config.base_weights_path,
        map_location=device,
        weights_only=True,
    )
    model.load_state_dict(state_dict, strict=True)
    setup = prepare_lora_model(model, lora_config)
    parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    optimizer = torch.optim.AdamW(parameters, lr=config.learning_rate)
    scaler = make_grad_scaler(device, config.dtype)
    input_ids = torch.randint(
        0,
        len(tokenizer),
        (config.batch_size, config.sequence_length),
        device=device,
    )
    labels = input_ids.clone()
    model.train()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    with autocast_context(device, config.dtype):
        output = model(input_ids, labels=labels, use_cache=False)
        if output.loss is None:
            raise RuntimeError("capacity check loss was not produced")
    scaler.scale(output.loss).backward()
    scaler.unscale_(optimizer)
    gradient_norm = torch.nn.utils.clip_grad_norm_(
        parameters,
        config.gradient_clip,
    )
    scaler.step(optimizer)
    scaler.update()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    summary = {
        "device": str(device),
        "dtype": config.dtype,
        "batch_size": config.batch_size,
        "sequence_length": config.sequence_length,
        "loss": float(output.loss.detach().item()),
        "gradient_norm": float(gradient_norm.item()),
        "step_seconds": elapsed,
        "peak_memory_bytes": (
            int(torch.cuda.max_memory_allocated(cuda_device_index(device)))
            if device.type == "cuda"
            else 0
        ),
        **setup,
    }
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("docs/stages/06_lora/configs/rank16.json"),
    )
    parser.add_argument("--resume", type=Path)
    parser.add_argument(
        "--stop-after-step",
        type=int,
        help="在指定 optimizer step 完成评估和 checkpoint 后停止",
    )
    parser.add_argument(
        "--capacity-check",
        action="store_true",
        help="只执行一次正式 shape 的参数更新，不写训练产物",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.capacity_check:
        if args.resume is not None or args.stop_after_step is not None:
            raise ValueError("capacity check cannot be combined with resume or stop")
        summary = run_capacity_check(args.config)
    else:
        result = run_lora_training(
            args.config,
            args.resume,
            args.stop_after_step,
        )
        summary = result["summary"]
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
