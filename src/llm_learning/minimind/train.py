"""阶段 4 MiniMind 单卡预训练入口。"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

from .checkpoint import export_model_weights, load_checkpoint, save_checkpoint
from .config import PretrainConfig, load_pretrain_config
from .data import (
    EpochBatchSampler,
    MiniMindPretrainDataset,
    RecordCollection,
    create_split_manifest,
    load_json_dataset,
    resolve_split_row_ids,
    save_manifest,
    sha256_file,
    tokenizer_identity,
)
from .inspect import MINIMIND_REVISION, load_official_model_module


def resolve_device(requested: str) -> torch.device:
    """解析 auto/CPU/CUDA 设备并验证可用性。"""
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("请求使用 CUDA，但当前环境没有可用 CUDA 设备")
    return device


def cuda_device_index(device: torch.device) -> int:
    """返回 CUDA 统计 API 接受的整数设备索引。"""
    if device.type != "cuda":
        raise ValueError("device is not CUDA")
    return device.index if device.index is not None else torch.cuda.current_device()


def seed_everything(seed: int) -> None:
    """固定 Python、CPU 与 CUDA 随机数。"""
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def cosine_learning_rate(
    optimizer_step: int,
    total_optimizer_steps: int,
    base_learning_rate: float,
    minimum_ratio: float,
) -> float:
    """按 optimizer step 从基础学习率余弦衰减到固定比例。"""
    if total_optimizer_steps < 1:
        raise ValueError("total_optimizer_steps must be positive")
    if total_optimizer_steps == 1:
        progress = 0.0
    else:
        progress = optimizer_step / (total_optimizer_steps - 1)
    progress = min(max(progress, 0.0), 1.0)
    multiplier = minimum_ratio + (1.0 - minimum_ratio) * 0.5 * (
        1.0 + math.cos(math.pi * progress)
    )
    return base_learning_rate * multiplier


def autocast_context(device: torch.device, dtype: str):
    """为当前设备和配置创建 autocast 上下文。"""
    if device.type != "cuda" or dtype == "float32":
        return nullcontext()
    torch_dtype = torch.bfloat16 if dtype == "bfloat16" else torch.float16
    return torch.autocast("cuda", dtype=torch_dtype)


def make_grad_scaler(device: torch.device, dtype: str) -> torch.GradScaler:
    """只为 CUDA FP16 启用 GradScaler。"""
    enabled = device.type == "cuda" and dtype == "float16"
    return torch.GradScaler("cuda", enabled=enabled)


def valid_label_count(labels: torch.Tensor) -> int:
    """统计模型内部 shift 后真正参与 loss 的 token。"""
    return int(labels[..., 1:].ne(-100).sum().item())


@torch.no_grad()
def evaluate_loss(
    model: nn.Module,
    loader: Iterable[dict[str, torch.Tensor]],
    device: torch.device,
    dtype: str,
) -> dict[str, float | int]:
    """按有效 token 数汇总 loss 与 perplexity。"""
    was_training = model.training
    model.eval()
    loss_sum = 0.0
    token_count = 0
    for batch in loader:
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)
        with autocast_context(device, dtype):
            output = model(input_ids, labels=labels, use_cache=False)
        if output.loss is None:
            raise RuntimeError("evaluation loss was not produced")
        valid_tokens = valid_label_count(labels)
        loss_sum += output.loss.item() * valid_tokens
        token_count += valid_tokens
    model.train(was_training)
    if token_count == 0:
        raise ValueError("evaluation contains no valid labels")
    mean_loss = loss_sum / token_count
    return {
        "loss": mean_loss,
        "perplexity": math.exp(mean_loss),
        "tokens": token_count,
    }


def append_metric(path: Path, metric: dict[str, Any]) -> None:
    """追加一行结构化训练指标。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(metric, ensure_ascii=False) + "\n")


def rewrite_metrics(path: Path, history: list[dict[str, Any]]) -> None:
    """按 checkpoint 历史重建 metrics，避免 resume 后重复。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for metric in history:
            file.write(json.dumps(metric, ensure_ascii=False) + "\n")


def make_eval_loader(
    dataset: Dataset,
    batch_size: int,
    num_workers: int,
    device: torch.device,
) -> DataLoader:
    """创建固定顺序且不丢样本的评估 Loader。"""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        drop_last=False,
    )


def total_optimizer_steps(config: PretrainConfig, train_rows: int) -> int:
    """根据数据量、epoch 和梯度累积计算正式更新次数。"""
    batches_per_epoch = math.ceil(train_rows / config.batch_size)
    updates_per_epoch = math.ceil(batches_per_epoch / config.accumulation_steps)
    derived = updates_per_epoch * config.epochs
    if config.max_optimizer_steps is not None:
        return min(derived, config.max_optimizer_steps)
    return derived


def compact_manifest(manifest: dict[str, Any]) -> dict[str, Any]:
    """返回适合写入运行记录和 checkpoint 的数据身份。"""
    return manifest


def build_model_and_tokenizer(config: PretrainConfig, device: torch.device):
    """从固定官方源码创建随机初始化模型与官方 tokenizer。"""
    module = load_official_model_module(config.source_dir)
    tokenizer_dir = config.source_dir / "model"
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
    model_config = module.MiniMindConfig(**config.model_kwargs(len(tokenizer)))
    model = module.MiniMindForCausalLM(model_config).to(device)
    return model, tokenizer


DatasetFactory = Callable[
    [Any, RecordCollection, Sequence[int], Any],
    Dataset,
]
ManifestFactory = Callable[..., dict[str, Any]]
SplitResolver = Callable[[dict[str, Any]], tuple[list[int], list[int]]]
RecordsLoader = Callable[[Path], RecordCollection]
ModelSetup = Callable[[nn.Module], Mapping[str, Any] | None]
WeightsExporter = Callable[[Path, nn.Module], None]


def build_pretrain_dataset(
    config: PretrainConfig,
    records: RecordCollection,
    row_ids: Sequence[int],
    tokenizer: Any,
) -> Dataset:
    """构造阶段 4 的 next-token Dataset。"""
    return MiniMindPretrainDataset(
        records,
        row_ids,
        tokenizer,
        config.sequence_length,
    )


def run_training(
    config: PretrainConfig,
    resume_path: Path | None = None,
    *,
    stop_after_step: int | None = None,
    dataset_factory: DatasetFactory = build_pretrain_dataset,
    manifest_factory: ManifestFactory = create_split_manifest,
    split_resolver: SplitResolver = resolve_split_row_ids,
    records_loader: RecordsLoader = load_json_dataset,
    initial_weights_path: Path | None = None,
    model_setup: ModelSetup | None = None,
    weights_exporter: WeightsExporter = export_model_weights,
    weights_prefix: str = "weights",
    run_metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """执行一次固定配置的单卡 causal LM 训练。"""
    seed_everything(config.seed)
    device = resolve_device(config.device)
    print(
        f"[setup] profile={config.profile} device={device} dtype={config.dtype}",
        flush=True,
    )
    if not config.data_path.is_file():
        raise FileNotFoundError(
            f"找不到训练数据：{config.data_path}；先运行 llm_learning.minimind.data"
        )
    config.output_dir.mkdir(parents=True, exist_ok=True)
    config.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    if device.type == "cuda":
        torch.cuda.init()
        torch.cuda.reset_peak_memory_stats(cuda_device_index(device))

    print("[setup] loading model and tokenizer", flush=True)
    model, tokenizer = build_model_and_tokenizer(config, device)
    initial_weights_sha256 = None
    if initial_weights_path is not None and resume_path is None:
        if not initial_weights_path.is_file():
            raise FileNotFoundError(f"找不到初始模型权重：{initial_weights_path}")
        print(f"[setup] loading initial weights: {initial_weights_path}", flush=True)
        state_dict = torch.load(
            initial_weights_path,
            map_location=device,
            weights_only=True,
        )
        model.load_state_dict(state_dict, strict=True)
        initial_weights_sha256 = sha256_file(initial_weights_path)
    model_setup_info: dict[str, Any] = {}
    if model_setup is not None:
        setup_result = model_setup(model)
        if setup_result is not None:
            model_setup_info = dict(setup_result)
    trainable_parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    if not trainable_parameters:
        raise ValueError("model contains no trainable parameters")
    if not weights_prefix:
        raise ValueError("weights_prefix cannot be empty")

    tokenizer_info = tokenizer_identity(config.source_dir / "model", tokenizer)
    print(f"[setup] loading dataset: {config.data_path}", flush=True)
    records = records_loader(config.data_path)
    manifest = manifest_factory(
        config=config,
        records=records,
        raw_sha256=sha256_file(config.data_path),
        dataset_fingerprint=getattr(records, "_fingerprint", None),
        tokenizer_info=tokenizer_info,
    )
    if initial_weights_path is not None:
        manifest["initial_weights"] = {
            "path": str(initial_weights_path),
            "sha256": (
                initial_weights_sha256
                if initial_weights_sha256 is not None
                else sha256_file(initial_weights_path)
            ),
        }
    train_row_ids, validation_row_ids = split_resolver(manifest)
    save_manifest(config.output_dir / "data_manifest.json", manifest)
    runtime_config = config.to_dict()
    if run_metadata is not None:
        runtime_config["training_extension"] = dict(run_metadata)
    (config.output_dir / "config.json").write_text(
        json.dumps(runtime_config, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    train_dataset = dataset_factory(config, records, train_row_ids, tokenizer)
    validation_dataset = dataset_factory(
        config,
        records,
        validation_row_ids,
        tokenizer,
    )
    if config.periodic_validation_rows:
        periodic_row_ids = validation_row_ids[: config.periodic_validation_rows]
        periodic_dataset = dataset_factory(
            config,
            records,
            periodic_row_ids,
            tokenizer,
        )
    else:
        periodic_dataset = validation_dataset
    periodic_loader = make_eval_loader(
        periodic_dataset,
        config.batch_size,
        config.num_workers,
        device,
    )
    final_validation_loader = make_eval_loader(
        validation_dataset,
        config.batch_size,
        config.num_workers,
        device,
    )

    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=config.learning_rate,
    )
    scaler = make_grad_scaler(device, config.dtype)
    planned_steps = total_optimizer_steps(config, len(train_dataset))
    optimizer_steps_per_epoch = math.ceil(
        math.ceil(len(train_dataset) / config.batch_size)
        / config.accumulation_steps
    )
    if stop_after_step is not None and not 1 <= stop_after_step <= planned_steps:
        raise ValueError(
            f"stop_after_step must be between 1 and {planned_steps}"
        )
    print(
        f"[setup] train_rows={len(train_dataset)} "
        f"validation_rows={len(validation_dataset)} "
        f"optimizer_steps={planned_steps} "
        f"optimizer_steps_per_epoch={optimizer_steps_per_epoch}",
        flush=True,
    )
    early_stop = (
        f"eval_loss<={config.overfit_loss_target:.4f}"
        if config.overfit_loss_target is not None
        else "disabled"
    )
    print(
        f"[setup] epochs={config.epochs} early_stop={early_stop} "
        f"stop_after_step={stop_after_step or 'disabled'}",
        flush=True,
    )
    history: list[dict[str, Any]] = []
    training_state = {
        "epoch": 0,
        "next_batch": 0,
        "optimizer_step": 0,
        "trained_tokens": 0,
        "microbatches_in_update": 0,
        "stop_reason": None,
    }
    latest_checkpoint = config.checkpoint_dir / "latest.pt"
    metrics_path = config.output_dir / "metrics.jsonl"

    if resume_path is not None:
        print(f"[setup] resuming checkpoint: {resume_path}", flush=True)
        checkpoint = load_checkpoint(
            path=resume_path,
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            expected_config=runtime_config,
            expected_tokenizer=tokenizer_info,
            expected_data_manifest=compact_manifest(manifest),
            device=device,
        )
        history = list(checkpoint["history"])
        training_state = dict(checkpoint["training_state"])
        training_state.setdefault("stop_reason", None)
        rewrite_metrics(metrics_path, history)
    else:
        metrics_path.unlink(missing_ok=True)
        print("[eval] measuring initial loss", flush=True)
        initial_eval = evaluate_loss(model, periodic_loader, device, config.dtype)
        metric = {
            "kind": "eval",
            "scope": "train" if config.overfit_rows else "validation_subset",
            "optimizer_step": 0,
            **initial_eval,
        }
        history.append(metric)
        append_metric(metrics_path, metric)
        weights_exporter(
            config.checkpoint_dir / f"{weights_prefix}_step_0.pth",
            model,
        )
        print(
            f"[eval] step=0 loss={initial_eval['loss']:.4f} "
            f"ppl={initial_eval['perplexity']:.2f}",
            flush=True,
        )

    model.train()
    if (
        training_state["optimizer_step"] >= planned_steps
        and training_state.get("stop_reason") is None
    ):
        training_state["stop_reason"] = "planned_steps"
    stop_requested = training_state.get("stop_reason") is not None
    run_start = time.perf_counter()
    for epoch in range(training_state["epoch"], config.epochs):
        if stop_requested:
            break
        start_batch = (
            training_state["next_batch"]
            if epoch == training_state["epoch"]
            else 0
        )
        batch_sampler = EpochBatchSampler(
            dataset_size=len(train_dataset),
            batch_size=config.batch_size,
            seed=config.seed,
            epoch=epoch,
            start_batch=start_batch,
        )
        train_loader = DataLoader(
            train_dataset,
            batch_sampler=batch_sampler,
            num_workers=config.num_workers,
            pin_memory=device.type == "cuda",
        )
        total_batches = batch_sampler.total_batches
        microbatches_in_update = 0
        group_size = 0
        group_loss_sum = 0.0
        group_tokens = 0
        group_start = time.perf_counter()

        for local_batch_index, batch in enumerate(train_loader):
            absolute_batch_index = start_batch + local_batch_index
            if microbatches_in_update == 0:
                group_size = min(
                    config.accumulation_steps,
                    total_batches - absolute_batch_index,
                )
                group_start = time.perf_counter()

            input_ids = batch["input_ids"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)
            with autocast_context(device, config.dtype):
                output = model(input_ids, labels=labels, use_cache=False)
                loss = output.loss
                if loss is None:
                    raise RuntimeError("training loss was not produced")
                backward_loss = loss / group_size
            scaler.scale(backward_loss).backward()
            valid_tokens = valid_label_count(labels)
            group_loss_sum += loss.detach().item() * valid_tokens
            group_tokens += valid_tokens
            microbatches_in_update += 1
            training_state["next_batch"] = absolute_batch_index + 1
            training_state["microbatches_in_update"] = microbatches_in_update

            if microbatches_in_update != group_size:
                continue

            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(
                trainable_parameters,
                config.gradient_clip,
            )
            grad_norm = grad_norm.item()
            optimizer_step = training_state["optimizer_step"]
            learning_rate = cosine_learning_rate(
                optimizer_step,
                planned_steps,
                config.learning_rate,
                config.minimum_lr_ratio,
            )
            for parameter_group in optimizer.param_groups:
                parameter_group["lr"] = learning_rate
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

            optimizer_step += 1
            epoch_step = math.ceil(
                training_state["next_batch"] / config.accumulation_steps
            )
            training_state["optimizer_step"] = optimizer_step
            training_state["trained_tokens"] += group_tokens
            training_state["microbatches_in_update"] = 0
            elapsed = max(time.perf_counter() - group_start, 1e-9)
            peak_memory = (
                int(torch.cuda.max_memory_allocated(cuda_device_index(device)))
                if device.type == "cuda"
                else 0
            )
            train_metric = {
                "kind": "train",
                "epoch": epoch,
                "next_batch": training_state["next_batch"],
                "optimizer_step": optimizer_step,
                "loss": group_loss_sum / group_tokens,
                "learning_rate": learning_rate,
                "gradient_norm": grad_norm,
                "tokens": group_tokens,
                "trained_tokens": training_state["trained_tokens"],
                "tokens_per_second": group_tokens / elapsed,
                "peak_memory_bytes": peak_memory,
            }
            history.append(train_metric)
            append_metric(metrics_path, train_metric)

            microbatches_in_update = 0
            group_loss_sum = 0.0
            group_tokens = 0

            should_evaluate = optimizer_step % config.eval_interval == 0
            should_evaluate = should_evaluate or optimizer_step == planned_steps
            if should_evaluate:
                evaluation = evaluate_loss(
                    model,
                    periodic_loader,
                    device,
                    config.dtype,
                )
                eval_metric = {
                    "kind": "eval",
                    "scope": (
                        "train" if config.overfit_rows else "validation_subset"
                    ),
                    "optimizer_step": optimizer_step,
                    **evaluation,
                }
                history.append(eval_metric)
                append_metric(metrics_path, eval_metric)
                print(
                    f"step={optimizer_step}/{planned_steps} "
                    f"epoch={epoch + 1}/{config.epochs} "
                    f"epoch_step={epoch_step}/{optimizer_steps_per_epoch} "
                    f"train_loss={train_metric['loss']:.4f} "
                    f"eval_loss={evaluation['loss']:.4f} "
                    f"ppl={evaluation['perplexity']:.2f} "
                    f"lr={learning_rate:.2e}",
                    flush=True,
                )
                if (
                    config.overfit_loss_target is not None
                    and evaluation["loss"] <= config.overfit_loss_target
                ):
                    stop_requested = True
                    training_state["stop_reason"] = "overfit_loss_target"
                    print(
                        f"[stop] step={optimizer_step} "
                        "reason=overfit_loss_target "
                        f"eval_loss={evaluation['loss']:.4f} "
                        f"target={config.overfit_loss_target:.4f}",
                        flush=True,
                    )
            elif optimizer_step == 1 or optimizer_step % 10 == 0:
                print(
                    f"step={optimizer_step}/{planned_steps} "
                    f"epoch={epoch + 1}/{config.epochs} "
                    f"epoch_step={epoch_step}/{optimizer_steps_per_epoch} "
                    f"train_loss={train_metric['loss']:.4f} "
                    f"lr={learning_rate:.2e} "
                    f"tokens/s={train_metric['tokens_per_second']:.0f}",
                    flush=True,
                )

            if optimizer_step == max(1, planned_steps // 2):
                weights_exporter(
                    config.checkpoint_dir
                    / f"{weights_prefix}_step_{optimizer_step}.pth",
                    model,
                )
            if optimizer_step % config.checkpoint_interval == 0:
                save_checkpoint(
                    latest_checkpoint,
                    model,
                    optimizer,
                    scaler,
                    training_state,
                    runtime_config,
                    tokenizer_info,
                    compact_manifest(manifest),
                    history,
                )
            if stop_after_step is not None and optimizer_step >= stop_after_step:
                stop_requested = True
                training_state["stop_reason"] = "stop_after_step"
                print(
                    f"[stop] step={optimizer_step} reason=stop_after_step",
                    flush=True,
                )
            if optimizer_step >= planned_steps or stop_requested:
                stop_requested = True
                if training_state.get("stop_reason") is None:
                    training_state["stop_reason"] = "planned_steps"
                break

        if training_state["next_batch"] >= total_batches:
            training_state["epoch"] = epoch + 1
            training_state["next_batch"] = 0
        if stop_requested:
            break

    final_step = training_state["optimizer_step"]
    final_scope = "train_full" if config.overfit_rows else "validation_full"
    existing_final = next(
        (
            metric
            for metric in reversed(history)
            if metric.get("kind") == "eval"
            and metric.get("scope") == final_scope
            and metric.get("optimizer_step") == final_step
        ),
        None,
    )
    if existing_final is None:
        final_evaluation = evaluate_loss(
            model,
            final_validation_loader,
            device,
            config.dtype,
        )
        final_metric = {
            "kind": "eval",
            "scope": final_scope,
            "optimizer_step": final_step,
            **final_evaluation,
        }
        history.append(final_metric)
        append_metric(metrics_path, final_metric)
    else:
        final_evaluation = {
            "loss": existing_final["loss"],
            "perplexity": existing_final["perplexity"],
            "tokens": existing_final["tokens"],
        }
    weights_exporter(
        config.checkpoint_dir / f"{weights_prefix}_step_{final_step}.pth",
        model,
    )
    save_checkpoint(
        latest_checkpoint,
        model,
        optimizer,
        scaler,
        training_state,
        runtime_config,
        tokenizer_info,
        compact_manifest(manifest),
        history,
    )

    summary = {
        "source_revision": MINIMIND_REVISION,
        "profile": config.profile,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "trainable_parameter_count": sum(
            parameter.numel()
            for parameter in model.parameters()
            if parameter.requires_grad
        ),
        "device": str(device),
        "device_name": (
            torch.cuda.get_device_name(cuda_device_index(device))
            if device.type == "cuda"
            else "CPU"
        ),
        "dtype": config.dtype,
        "effective_batch_size": config.effective_batch_size,
        "planned_optimizer_steps": planned_steps,
        "completed_optimizer_steps": final_step,
        "completed_epochs": training_state["epoch"],
        "stop_reason": training_state.get("stop_reason"),
        "trained_tokens": training_state["trained_tokens"],
        "final_evaluation": final_evaluation,
        "peak_memory_bytes": (
            int(torch.cuda.max_memory_allocated(cuda_device_index(device)))
            if device.type == "cuda"
            else 0
        ),
        "wall_time_seconds": time.perf_counter() - run_start,
        "checkpoint": str(latest_checkpoint),
        "initial_weights": (
            manifest.get("initial_weights") if initial_weights_path is not None else None
        ),
        "model_setup": model_setup_info or None,
    }
    (config.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return {
        "model": model,
        "tokenizer": tokenizer,
        "history": history,
        "summary": summary,
        "training_state": training_state,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("docs/stages/04_pretrain/configs/overfit256.json"),
    )
    parser.add_argument("--resume", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_pretrain_config(args.config)
    result = run_training(config, args.resume)
    print(json.dumps(result["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
