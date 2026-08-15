"""阶段 8 MiniMind CE / logit distillation 单卡训练入口。"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from .checkpoint import export_model_weights
from .data import EpochBatchSampler, sha256_json
from .distill_config import (
    LogitDistillationConfig,
    load_logit_distillation_config,
)
from .distillation import (
    distillation_losses,
    masked_cross_entropy,
    masked_kl_divergence,
)
from .inspect import load_official_model_module
from .sft_config import load_sft_config
from .sft_data import MiniMindSFTDataset, load_sft_json_dataset
from .train import (
    append_metric,
    autocast_context,
    cosine_learning_rate,
    cuda_device_index,
    make_grad_scaler,
    resolve_device,
    seed_everything,
)


def load_weights(model: torch.nn.Module, path: Path) -> None:
    """加载 MiniMind 直接 state dict 或含 model 字段的 checkpoint。"""
    if not path.exists():
        raise FileNotFoundError(path)
    state = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    model.load_state_dict(state, strict=True)


def fixed_row_ids(
    row_count: int,
    *,
    seed: int,
    train_rows: int,
    validation_rows: int,
) -> tuple[list[int], list[int]]:
    """固定 CE / KD 共用的 train 与 validation row ID。"""
    if train_rows + validation_rows > row_count:
        raise ValueError("distillation subset exceeds dataset size")
    order = torch.randperm(
        row_count,
        generator=torch.Generator().manual_seed(seed),
    ).tolist()
    validation_ids = order[:validation_rows]
    train_ids = order[validation_rows : validation_rows + train_rows]
    return train_ids, validation_ids


def build_models(config: LogitDistillationConfig, device: torch.device):
    """加载同 tokenizer 的 Dense Student 与 MoE Teacher。"""
    module = load_official_model_module(config.source_dir)
    tokenizer = AutoTokenizer.from_pretrained(config.source_dir / "model")
    student_config = module.MiniMindConfig(
        **config.model_kwargs(len(tokenizer), use_moe=False)
    )
    teacher_config = module.MiniMindConfig(
        **config.model_kwargs(len(tokenizer), use_moe=True)
    )
    student = module.MiniMindForCausalLM(student_config)
    load_weights(student, config.student_weights_path)
    student.to(device)

    teacher = module.MiniMindForCausalLM(teacher_config)
    load_weights(teacher, config.teacher_weights_path)
    teacher.requires_grad_(False)
    teacher.eval()
    teacher.to(device)
    return student, teacher, tokenizer


@torch.no_grad()
def evaluate(
    student: torch.nn.Module,
    teacher: torch.nn.Module,
    loader: DataLoader,
    config: LogitDistillationConfig,
    device: torch.device,
) -> dict[str, float | int]:
    """在固定 validation 上统一汇总 CE 与两个方向的 KL。"""
    student.eval()
    ce_sum = 0.0
    forward_kl_sum = 0.0
    reverse_kl_sum = 0.0
    token_count = 0
    for batch in loader:
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)
        with autocast_context(device, config.dtype):
            student_logits = student(input_ids, use_cache=False).logits
            teacher_logits = teacher(input_ids, use_cache=False).logits
            ce_loss = masked_cross_entropy(student_logits, labels)
            forward_kl = masked_kl_divergence(
                student_logits,
                teacher_logits,
                labels,
                config.temperature,
                direction="forward",
                scale_by_temperature=False,
            )
            reverse_kl = masked_kl_divergence(
                student_logits,
                teacher_logits,
                labels,
                config.temperature,
                direction="reverse",
                scale_by_temperature=False,
            )
        tokens = int(labels[..., 1:].ne(-100).sum().item())
        ce_sum += ce_loss.item() * tokens
        forward_kl_sum += forward_kl.item() * tokens
        reverse_kl_sum += reverse_kl.item() * tokens
        token_count += tokens
    student.train()
    ce = ce_sum / token_count
    forward_kl_value = forward_kl_sum / token_count
    reverse_kl_value = reverse_kl_sum / token_count
    forward_kd_loss = forward_kl_value * config.temperature**2
    reverse_kd_loss = reverse_kl_value * config.temperature**2
    kd = (
        forward_kd_loss
        if config.kl_direction == "forward"
        else reverse_kd_loss
    )
    total = config.ce_weight * ce + (1 - config.ce_weight) * kd
    return {
        "loss": total,
        "ce_loss": ce,
        "kd_loss": kd,
        "forward_kl": forward_kl_value,
        "reverse_kl": reverse_kl_value,
        "forward_kd_loss": forward_kd_loss,
        "reverse_kd_loss": reverse_kd_loss,
        "perplexity": math.exp(ce),
        "tokens": token_count,
    }


def save_training_checkpoint(
    path: Path,
    *,
    student: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.GradScaler,
    step: int,
    history: list[dict[str, Any]],
    config: LogitDistillationConfig,
    split_identity: str,
) -> None:
    """在 optimizer step 边界保存可恢复状态。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".pt.tmp")
    torch.save(
        {
            "model": student.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "step": step,
            "history": history,
            "profile": config.profile,
            "split_identity": split_identity,
            "rng_state": torch.get_rng_state(),
            "cuda_rng_state": (
                torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
            ),
        },
        temporary,
    )
    temporary.replace(path)


def run_distillation(
    config: LogitDistillationConfig,
    *,
    resume_path: Path | None = None,
) -> dict[str, Any]:
    """执行一组 CE / KD 对照并返回运行摘要。"""
    seed_everything(config.seed)
    device = resolve_device(config.device)
    if device.type == "cuda":
        torch.cuda.init()
        torch.cuda.reset_peak_memory_stats(cuda_device_index(device))
    student, teacher, tokenizer = build_models(config, device)
    records = load_sft_json_dataset(config.data_path)
    train_ids, validation_ids = fixed_row_ids(
        len(records),
        seed=config.seed,
        train_rows=config.train_rows,
        validation_rows=config.validation_rows,
    )
    split_identity = sha256_json(
        {"seed": config.seed, "train": train_ids, "validation": validation_ids}
    )
    base_sft = load_sft_config(config.base_sft_config)
    dataset_config = replace(
        base_sft,
        data_path=config.data_path,
        sequence_length=config.sequence_length,
    )
    train_dataset = MiniMindSFTDataset(records, train_ids, tokenizer, dataset_config)
    validation_dataset = MiniMindSFTDataset(
        records,
        validation_ids,
        tokenizer,
        dataset_config,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=device.type == "cuda",
    )
    updates_per_epoch = math.ceil(
        math.ceil(len(train_dataset) / config.batch_size)
        / config.accumulation_steps
    )
    planned_steps = updates_per_epoch * config.epochs
    optimizer = torch.optim.AdamW(student.parameters(), lr=config.learning_rate)
    scaler = make_grad_scaler(device, config.dtype)
    step = 0
    history: list[dict[str, Any]] = []
    if resume_path is not None:
        checkpoint = torch.load(resume_path, map_location=device, weights_only=False)
        if checkpoint["profile"] != config.profile:
            raise ValueError("resume profile does not match config")
        if checkpoint["split_identity"] != split_identity:
            raise ValueError("resume split does not match config")
        student.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scaler.load_state_dict(checkpoint["scaler"])
        step = int(checkpoint["step"])
        history = list(checkpoint["history"])
        torch.set_rng_state(checkpoint["rng_state"].cpu())
        if checkpoint["cuda_rng_state"] is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(checkpoint["cuda_rng_state"])

    metrics_path = config.output_dir / "metrics.jsonl"
    config.output_dir.mkdir(parents=True, exist_ok=True)
    if step == 0:
        metrics_path.unlink(missing_ok=True)
        initial = {"step": 0, **evaluate(student, teacher, validation_loader, config, device)}
        history.append(initial)
        append_metric(metrics_path, initial)
    print(
        f"[setup] profile={config.profile} device={device} dtype={config.dtype} "
        f"ce_weight={config.ce_weight} temperature={config.temperature} "
        f"kl_direction={config.kl_direction}",
        flush=True,
    )
    print(
        f"[setup] train_rows={len(train_ids)} validation_rows={len(validation_ids)} "
        f"optimizer_steps={planned_steps} "
        f"optimizer_steps_per_epoch={updates_per_epoch}",
        flush=True,
    )

    start_time = time.perf_counter()
    optimizer.zero_grad(set_to_none=True)
    for epoch in range(config.epochs):
        sampler = EpochBatchSampler(
            dataset_size=len(train_dataset),
            batch_size=config.batch_size,
            seed=config.seed,
            epoch=epoch,
            start_batch=0,
        )
        loader = DataLoader(
            train_dataset,
            batch_sampler=sampler,
            num_workers=config.num_workers,
            pin_memory=device.type == "cuda",
        )
        for batch_index, batch in enumerate(loader):
            target_step = epoch * updates_per_epoch + (
                batch_index // config.accumulation_steps
            ) + 1
            if target_step <= step:
                continue
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)
            with autocast_context(device, config.dtype):
                student_logits = student(input_ids, use_cache=False).logits
                with torch.no_grad():
                    teacher_logits = (
                        None
                        if config.ce_weight == 1
                        else teacher(input_ids, use_cache=False).logits
                    )
                losses = distillation_losses(
                    student_logits,
                    teacher_logits,
                    labels,
                    ce_weight=config.ce_weight,
                    temperature=config.temperature,
                    kl_direction=config.kl_direction,
                )
                scaled_loss = losses.total / config.accumulation_steps
            scaler.scale(scaled_loss).backward()
            end_of_window = (
                (batch_index + 1) % config.accumulation_steps == 0
                or batch_index + 1 == len(loader)
            )
            if not end_of_window:
                continue
            learning_rate = cosine_learning_rate(
                step,
                planned_steps,
                config.learning_rate,
                config.minimum_lr_ratio,
            )
            for group in optimizer.param_groups:
                group["lr"] = learning_rate
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(student.parameters(), config.gradient_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            step += 1
            epoch_step = math.ceil((batch_index + 1) / config.accumulation_steps)
            if step == 1 or step % 10 == 0:
                loss_parts = []
                if losses.ce is not None:
                    loss_parts.append(f"ce={losses.ce.item():.4f}")
                if losses.kd is not None:
                    loss_parts.append(f"kd={losses.kd.item():.4f}")
                print(
                    f"step={step}/{planned_steps} epoch={epoch + 1}/{config.epochs} "
                    f"epoch_step={epoch_step}/{updates_per_epoch} "
                    f"loss={losses.total.item():.4f} {' '.join(loss_parts)} "
                    f"lr={learning_rate:.2e}",
                    flush=True,
                )
            if step % config.eval_interval == 0 or step == planned_steps:
                metric = {
                    "step": step,
                    **evaluate(student, teacher, validation_loader, config, device),
                }
                history.append(metric)
                append_metric(metrics_path, metric)
                print(
                    f"[eval] step={step} ce={metric['ce_loss']:.4f} "
                    f"forward_kl={metric['forward_kl']:.4f} "
                    f"reverse_kl={metric['reverse_kl']:.4f} "
                    f"kd_loss={metric['kd_loss']:.4f} "
                    f"ppl={metric['perplexity']:.2f}",
                    flush=True,
                )
            if step % config.checkpoint_interval == 0 or step == planned_steps:
                save_training_checkpoint(
                    config.checkpoint_dir / "latest.pt",
                    student=student,
                    optimizer=optimizer,
                    scaler=scaler,
                    step=step,
                    history=history,
                    config=config,
                    split_identity=split_identity,
                )
        if step >= planned_steps:
            break

    weights_path = config.checkpoint_dir / f"weights_step_{step}.pth"
    export_model_weights(weights_path, student)
    final_evaluation = evaluate(
        student,
        teacher,
        validation_loader,
        config,
        device,
    )
    result = {
        "profile": config.profile,
        "device": str(device),
        "dtype": config.dtype,
        "ce_weight": config.ce_weight,
        "temperature": config.temperature,
        "kl_direction": config.kl_direction,
        "train_rows": len(train_ids),
        "validation_rows": len(validation_ids),
        "effective_batch_size": config.effective_batch_size,
        "completed_optimizer_steps": step,
        "final_evaluation": final_evaluation,
        "peak_memory_bytes": (
            torch.cuda.max_memory_allocated(cuda_device_index(device))
            if device.type == "cuda"
            else 0
        ),
        "wall_time_seconds": time.perf_counter() - start_time,
        "weights": str(weights_path),
    }
    (config.output_dir / "summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_distillation(
        load_logit_distillation_config(args.config),
        resume_path=args.resume,
    )


if __name__ == "__main__":
    main()
