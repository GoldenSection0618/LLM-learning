from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from llm_learning.bigram.data import (
    TINY_SHAKESPEARE_URL,
    DataManifest,
    NextTokenDataset,
    download_tiny_shakespeare,
    sha256_file,
    split_token_ids,
)
from llm_learning.bigram.tokenizer import CharacterTokenizer
from llm_learning.bigram.train import (
    append_metric,
    infinite_batches,
    make_evaluation_loader,
    make_loader,
    resolve_device,
    seed_everything,
)

from .config import TrainConfig
from .model import TinyGPT


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    batches: int,
    device: torch.device,
) -> float:
    """在固定样本上计算平均 loss。"""
    was_training = model.training
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    iterator = infinite_batches(loader)

    for _ in range(batches):
        inputs, targets = next(iterator)
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        _, loss = model(inputs, targets)

        if loss is None:
            raise RuntimeError("Evaluation loss was not produced")

        token_count = targets.numel()
        total_loss += loss.item() * token_count
        total_tokens += token_count

    model.train(was_training)
    return total_loss / total_tokens


def print_trace(trace: dict[str, tuple[int, ...]]) -> None:
    """按 forward 顺序打印关键张量 shape。"""
    print("forward tensor trace:")
    for name, shape in trace.items():
        print(f"  {name:<32} {shape}")


def train(config: TrainConfig) -> dict[str, object]:
    seed_everything(config.seed)
    device = resolve_device(config.device)

    download_tiny_shakespeare(config.data_path)
    text = config.data_path.read_text(encoding="utf-8")
    tokenizer = CharacterTokenizer.from_text(text)
    token_ids = tokenizer.encode(text)
    train_ids, validation_ids, split_index = split_token_ids(
        token_ids,
        train_fraction=config.train_fraction,
    )

    config.output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer.save(config.output_dir / "tokenizer.json")
    manifest = DataManifest(
        source_url=TINY_SHAKESPEARE_URL,
        sha256=sha256_file(config.data_path),
        total_characters=len(token_ids),
        train_characters=len(train_ids),
        validation_characters=len(validation_ids),
        split_index=split_index,
        train_fraction=config.train_fraction,
        vocab_size=tokenizer.vocab_size,
    )
    manifest.save(config.output_dir / "data_manifest.json")

    model_config = config.model_config(tokenizer.vocab_size)
    experiment_config = {
        "training": config.to_dict(),
        "model": model_config.to_dict(),
    }
    (config.output_dir / "config.json").write_text(
        json.dumps(experiment_config, indent=2) + "\n",
        encoding="utf-8",
    )

    train_dataset = NextTokenDataset(train_ids, config.block_size)
    validation_dataset = NextTokenDataset(
        validation_ids,
        config.block_size,
    )
    train_loader = make_loader(
        dataset=train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        device=device,
        seed=config.seed,
    )
    train_evaluation_loader = make_evaluation_loader(
        dataset=train_dataset,
        batch_size=config.batch_size,
        batches=config.eval_batches,
        seed=config.seed + 1,
        device=device,
    )
    validation_loader = make_evaluation_loader(
        dataset=validation_dataset,
        batch_size=config.batch_size,
        batches=config.eval_batches,
        seed=config.seed + 2,
        device=device,
    )
    full_validation_loader = DataLoader(
        validation_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
        drop_last=False,
    )
    train_iterator = infinite_batches(train_loader)

    model = TinyGPT(model_config).to(device)
    parameter_count = model.count_parameters()
    if not 1_000_000 <= parameter_count <= 10_000_000:
        raise ValueError(
            "The stage 2 model must contain between 1M and 10M parameters"
        )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
    )
    print(f"device={device}")
    print(f"parameters={parameter_count:,}")

    # 使用真实 batch 记录一次计划要求的完整 forward 张量流。
    trace_inputs, trace_targets = next(iter(train_evaluation_loader))
    trace_inputs = trace_inputs[:2].to(device)
    trace_targets = trace_targets[:2].to(device)
    model.eval()
    with torch.no_grad():
        _, _, trace = model.forward_with_trace(
            trace_inputs,
            trace_targets,
        )
    model.train()
    print_trace(trace)
    (config.output_dir / "forward_trace.json").write_text(
        json.dumps(trace, indent=2) + "\n",
        encoding="utf-8",
    )

    metrics_path = config.output_dir / "metrics.jsonl"
    metrics_path.unlink(missing_ok=True)
    history: list[dict[str, float | int]] = []

    for step in range(1, config.max_steps + 1):
        inputs, targets = next(train_iterator)
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        _, loss = model(inputs, targets)

        if loss is None:
            raise RuntimeError("Training loss was not produced")

        loss.backward()
        optimizer.step()

        should_evaluate = step == 1 or step % config.eval_interval == 0
        should_evaluate = should_evaluate or step == config.max_steps

        if should_evaluate:
            train_loss = evaluate(
                model=model,
                loader=train_evaluation_loader,
                batches=config.eval_batches,
                device=device,
            )
            validation_loss = evaluate(
                model=model,
                loader=validation_loader,
                batches=config.eval_batches,
                device=device,
            )
            metric: dict[str, float | int] = {
                "step": step,
                "train_loss": train_loss,
                "validation_loss": validation_loss,
            }
            history.append(metric)
            append_metric(metrics_path, metric)
            print(
                f"step={step:5d} train_loss={train_loss:.4f} "
                f"validation_loss={validation_loss:.4f}"
            )

    full_validation_loss = evaluate(
        model=model,
        loader=full_validation_loader,
        batches=len(full_validation_loader),
        device=device,
    )
    final_metrics = {
        "step": config.max_steps,
        "train_subset_loss": history[-1]["train_loss"],
        "validation_subset_loss": history[-1]["validation_loss"],
        "full_validation_loss": full_validation_loss,
    }
    (config.output_dir / "final_metrics.json").write_text(
        json.dumps(final_metrics, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"full_validation_loss={full_validation_loss:.4f}")

    model_path = config.output_dir / "model.pt"
    torch.save(
        {
            "model": model.state_dict(),
            "model_config": model_config.to_dict(),
            "tokenizer": tokenizer.to_dict(),
        },
        model_path,
    )

    generation_device = "cuda" if device.type == "cuda" else "cpu"
    generator = torch.Generator(device=generation_device)
    generator.manual_seed(config.seed + 3)
    start_token_id = tokenizer.encode("\n")[0]
    start_tokens = torch.tensor(
        [[start_token_id]],
        device=device,
    )
    generated = model.generate(
        start_tokens,
        max_new_tokens=config.generate_tokens,
        generator=generator,
    )
    generated_text = tokenizer.decode(generated[0].tolist())
    (config.output_dir / "generated.txt").write_text(
        generated_text,
        encoding="utf-8",
    )
    print(f"model={model_path}")
    print(f"generated_text={config.output_dir / 'generated.txt'}")

    return {
        "model": model,
        "optimizer": optimizer,
        "tokenizer": tokenizer,
        "history": history,
        "trace": trace,
        "full_validation_loss": full_validation_loss,
        "generated_text": generated_text,
        "device": device,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="训练阶段 2 的 TinyGPT",
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=TrainConfig.data_path,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=TrainConfig.output_dir,
    )
    parser.add_argument(
        "--block-size",
        type=int,
        default=TrainConfig.block_size,
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=TrainConfig.batch_size,
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=TrainConfig.max_steps,
    )
    parser.add_argument(
        "--eval-interval",
        type=int,
        default=TrainConfig.eval_interval,
    )
    parser.add_argument(
        "--eval-batches",
        type=int,
        default=TrainConfig.eval_batches,
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=TrainConfig.learning_rate,
    )
    parser.add_argument(
        "--d-model",
        type=int,
        default=TrainConfig.d_model,
    )
    parser.add_argument(
        "--num-heads",
        type=int,
        default=TrainConfig.num_heads,
    )
    parser.add_argument(
        "--num-layers",
        type=int,
        default=TrainConfig.num_layers,
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=TrainConfig.dropout,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=TrainConfig.seed,
    )
    parser.add_argument(
        "--generate-tokens",
        type=int,
        default=TrainConfig.generate_tokens,
    )
    parser.add_argument(
        "--device",
        default=TrainConfig.device,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = replace(
        TrainConfig(),
        data_path=args.data_path,
        output_dir=args.output_dir,
        block_size=args.block_size,
        batch_size=args.batch_size,
        max_steps=args.max_steps,
        eval_interval=args.eval_interval,
        eval_batches=args.eval_batches,
        learning_rate=args.learning_rate,
        d_model=args.d_model,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        dropout=args.dropout,
        seed=args.seed,
        generate_tokens=args.generate_tokens,
        device=args.device,
    )
    train(config)


if __name__ == "__main__":
    main()
