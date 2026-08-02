from __future__ import annotations

import argparse
import json
import random
from dataclasses import replace
from pathlib import Path
from typing import Iterator

import torch
from torch.utils.data import DataLoader, Dataset, Subset

from .checkpoint import load_checkpoint, save_checkpoint
from .config import TrainConfig
from .data import (
    TINY_SHAKESPEARE_URL,
    DataManifest,
    NextTokenDataset,
    download_tiny_shakespeare,
    sha256_file,
    split_token_ids,
)
from .model import BigramLanguageModel
from .tokenizer import CharacterTokenizer


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    device = torch.device(requested)

    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    return device


def seed_everything(seed: int) -> None:
    # 同时固定 Python、CPU 和所有 CUDA 设备的随机序列。
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_loader(
    dataset: Dataset[tuple[torch.Tensor, torch.Tensor]],
    batch_size: int,
    shuffle: bool,
    device: torch.device,
    seed: int | None = None,
) -> DataLoader[tuple[torch.Tensor, torch.Tensor]]:
    generator = None

    if shuffle:
        if seed is None:
            raise ValueError("seed is required when shuffle is enabled")

        # 独立 generator 让训练样本的打乱顺序可以复现。
        generator = torch.Generator().manual_seed(seed)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
        num_workers=0,
        pin_memory=device.type == "cuda",
        # 固定 batch 形状，舍弃最后一个不足 batch_size 的批次。
        drop_last=True,
    )


def make_evaluation_loader(
    dataset: Dataset[tuple[torch.Tensor, torch.Tensor]],
    batch_size: int,
    batches: int,
    seed: int,
    device: torch.device,
) -> DataLoader[tuple[torch.Tensor, torch.Tensor]]:
    sample_count = batch_size * batches

    if sample_count > len(dataset):
        raise ValueError("Evaluation subset is larger than the dataset")

    # 只随机选择一次索引，后续评估始终复用同一批样本。
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(
        len(dataset),
        generator=generator,
    )[:sample_count].tolist()
    fixed_subset = Subset(dataset, indices)

    return make_loader(
        dataset=fixed_subset,
        batch_size=batch_size,
        shuffle=False,
        device=device,
    )


def infinite_batches(
    loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
    while True:
        yield from loader


@torch.no_grad()
def evaluate(
    model: BigramLanguageModel,
    loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    batches: int,
    device: torch.device,
) -> float:
    # 评估期间关闭梯度，并在结束后恢复模型原来的训练状态。
    was_training = model.training
    model.eval()
    losses: list[float] = []
    iterator = infinite_batches(loader)

    for _ in range(batches):
        inputs, targets = next(iterator)
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        _, loss = model(inputs, targets)
        if loss is None:
            raise RuntimeError("Evaluation loss was not produced")

        losses.append(loss.item())

    model.train(was_training)
    return sum(losses) / len(losses)


def append_metric(path: Path, metric: dict[str, float | int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(metric) + "\n")


def train(config: TrainConfig, resume: Path | None = None) -> dict[str, object]:
    seed_everything(config.seed)
    device = resolve_device(config.device)

    download_tiny_shakespeare(config.data_path)
    text = config.data_path.read_text(encoding="utf-8")
    tokenizer = CharacterTokenizer.from_text(text)
    token_ids = tokenizer.encode(text)

    # 先编码整篇文本，再按时间顺序切出训练集和验证集。
    train_ids, validation_ids, split_index = split_token_ids(
        token_ids,
        train_fraction=config.train_fraction,
    )

    config.output_dir.mkdir(parents=True, exist_ok=True)
    config.checkpoint_dir.mkdir(parents=True, exist_ok=True)
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
    (config.output_dir / "config.json").write_text(
        json.dumps(config.to_dict(), indent=2) + "\n",
        encoding="utf-8",
    )

    train_dataset = NextTokenDataset(train_ids, config.block_size)
    validation_dataset = NextTokenDataset(validation_ids, config.block_size)

    train_loader = make_loader(
        dataset=train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        device=device,
        seed=config.seed,
    )

    # 固定评估子集，让不同 step 的 loss 可以直接比较。
    train_eval_loader = make_evaluation_loader(
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
    train_iterator = infinite_batches(train_loader)

    model = BigramLanguageModel(tokenizer.vocab_size).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    start_step = 0
    history: list[dict[str, float | int]] = []

    if resume is not None:
        checkpoint = load_checkpoint(
            path=resume,
            model=model,
            optimizer=optimizer,
            device=device,
        )

        # token ID 映射不同会让已学习的 embedding 权重失去含义。
        if checkpoint["tokenizer"] != tokenizer.to_dict():
            raise ValueError("Checkpoint tokenizer does not match the dataset")

        start_step = int(checkpoint["step"])
        history = list(checkpoint["history"])
        print(f"resumed checkpoint={resume} step={start_step}")

    metrics_path = config.output_dir / "metrics.jsonl"
    if resume is None:
        metrics_path.unlink(missing_ok=True)

    latest_checkpoint = config.checkpoint_dir / "latest.pt"
    model.train()
    for step in range(start_step + 1, config.max_steps + 1):
        inputs, targets = next(train_iterator)
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        # 一个标准训练步：清梯度、forward、backward、更新参数。
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
                loader=train_eval_loader,
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
            save_checkpoint(
                path=latest_checkpoint,
                model=model,
                optimizer=optimizer,
                step=step,
                config=config.to_dict(),
                tokenizer=tokenizer.to_dict(),
                history=history,
            )

    generation_device = "cuda" if device.type == "cuda" else "cpu"

    # 使用独立随机数生成器，让采样文本可以复现。
    generation_generator = torch.Generator(
        device=generation_device,
    )
    generation_generator.manual_seed(config.seed + 3)

    start_token_id = tokenizer.encode("\n")[0]
    start_token = torch.tensor(
        [[start_token_id]],
        device=device,
    )

    generated_tokens = model.generate(
        start_token,
        max_new_tokens=config.generate_tokens,
        generator=generation_generator,
    )
    generated_ids = generated_tokens[0].tolist()
    generated_text = tokenizer.decode(generated_ids)

    (config.output_dir / "generated.txt").write_text(
        generated_text,
        encoding="utf-8",
    )
    print(f"generated_text={config.output_dir / 'generated.txt'}")
    print(generated_text)

    return {
        "model": model,
        "optimizer": optimizer,
        "tokenizer": tokenizer,
        "history": history,
        "checkpoint": latest_checkpoint,
        "generated_text": generated_text,
        "device": device,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="训练 Bigram 语言模型",
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
        "--checkpoint-dir",
        type=Path,
        default=TrainConfig.checkpoint_dir,
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
    parser.add_argument(
        "--resume",
        type=Path,
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = replace(
        TrainConfig(),
        data_path=args.data_path,
        output_dir=args.output_dir,
        checkpoint_dir=args.checkpoint_dir,
        block_size=args.block_size,
        batch_size=args.batch_size,
        max_steps=args.max_steps,
        eval_interval=args.eval_interval,
        eval_batches=args.eval_batches,
        learning_rate=args.learning_rate,
        seed=args.seed,
        generate_tokens=args.generate_tokens,
        device=args.device,
    )
    train(
        config=config,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()
