"""阶段 4 MiniMind 固定生成与 checkpoint 对比入口。"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Callable

import torch

from .config import PretrainConfig, load_pretrain_config
from .train import build_model_and_tokenizer, resolve_device


def load_generation_config(path: Path) -> dict[str, Any]:
    """读取并校验固定生成配置。"""
    config = json.loads(path.read_text(encoding="utf-8"))
    if not config.get("prompts"):
        raise ValueError("generation config must contain prompts")
    for name in ["greedy", "sampling"]:
        required = {
            "do_sample",
            "temperature",
            "top_k",
            "top_p",
            "max_new_tokens",
            "eos_token_id",
            "seed",
        }
        missing = required.difference(config[name])
        if missing:
            raise ValueError(f"{name} is missing fields: {sorted(missing)}")
    return config


def filter_logits(logits: torch.Tensor, top_k: int, top_p: float) -> torch.Tensor:
    """应用 top-k 与 nucleus 过滤，保留至少一个候选 token。"""
    filtered = logits.clone()
    if top_k > 0:
        top_k = min(top_k, filtered.shape[-1])
        threshold = torch.topk(filtered, top_k, dim=-1).values[..., -1, None]
        filtered.masked_fill_(filtered < threshold, -torch.inf)
    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(filtered, descending=True)
        cumulative = torch.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
        remove = cumulative > top_p
        remove[..., 1:] = remove[..., :-1].clone()
        remove[..., 0] = False
        remove_original_order = torch.zeros_like(remove).scatter(
            -1,
            sorted_indices,
            remove,
        )
        filtered.masked_fill_(remove_original_order, -torch.inf)
    return filtered


@torch.inference_mode()
def generate_one(
    model: torch.nn.Module,
    tokenizer: Any,
    prompt: str,
    generation: dict[str, Any],
    device: torch.device,
    prompt_renderer: Callable[[str, Any], str] | None = None,
) -> dict[str, Any]:
    """使用独立随机数生成器完成一次可复现的增量生成。"""
    rendered_prompt = (
        prompt_renderer(prompt, tokenizer)
        if prompt_renderer is not None
        else tokenizer.bos_token + prompt
    )
    input_ids = tokenizer(rendered_prompt, return_tensors="pt").input_ids.to(device)
    prompt_length = input_ids.shape[1]
    generated = input_ids
    past_key_values = None
    generator = torch.Generator(device=device).manual_seed(int(generation["seed"]))
    start = time.perf_counter()

    for _ in range(int(generation["max_new_tokens"])):
        current_input = generated if past_key_values is None else generated[:, -1:]
        output = model(
            current_input,
            past_key_values=past_key_values,
            use_cache=True,
        )
        logits = output.logits[:, -1, :]
        if generation["do_sample"]:
            temperature = float(generation["temperature"])
            if temperature <= 0:
                raise ValueError("sampling temperature must be positive")
            logits = filter_logits(
                logits / temperature,
                int(generation["top_k"]),
                float(generation["top_p"]),
            )
            probabilities = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(
                probabilities,
                num_samples=1,
                generator=generator,
            )
        else:
            next_token = logits.argmax(dim=-1, keepdim=True)
        generated = torch.cat([generated, next_token], dim=-1)
        past_key_values = output.past_key_values
        if next_token.item() == int(generation["eos_token_id"]):
            break

    elapsed = max(time.perf_counter() - start, 1e-9)
    continuation = generated[0, prompt_length:].cpu()
    stopped_on_eos = bool(
        continuation.numel()
        and continuation[-1].item() == int(generation["eos_token_id"])
    )
    return {
        "prompt": prompt,
        "prompt_tokens": prompt_length,
        "generated_tokens": int(continuation.numel()),
        "tokens_per_second": float(continuation.numel()) / elapsed,
        "stopped_on_eos": stopped_on_eos,
        "token_ids": continuation.tolist(),
        "text": tokenizer.decode(continuation, skip_special_tokens=True),
    }


def load_native_checkpoint(
    config: PretrainConfig,
    weights_path: Path,
    device: torch.device,
):
    """按训练配置重建模型并严格加载模型权重。"""
    model, tokenizer = build_model_and_tokenizer(config, device)
    state_dict = torch.load(weights_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model, tokenizer


def evaluate_checkpoints(
    config: PretrainConfig,
    checkpoint_paths: list[Path],
    generation_config: dict[str, Any],
    prompt_renderer: Callable[[str, Any], str] | None = None,
) -> dict[str, Any]:
    """用相同 prompts 与生成参数比较一组 checkpoint。"""
    device = resolve_device(config.device)
    results = []
    for checkpoint_path in checkpoint_paths:
        model, tokenizer = load_native_checkpoint(config, checkpoint_path, device)
        checkpoint_result = {
            "checkpoint": str(checkpoint_path),
            "modes": {},
        }
        for mode in ["greedy", "sampling"]:
            checkpoint_result["modes"][mode] = [
                generate_one(
                    model,
                    tokenizer,
                    prompt,
                    generation_config[mode],
                    device,
                    prompt_renderer,
                )
                for prompt in generation_config["prompts"]
            ]
        results.append(checkpoint_result)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return {
        "profile": config.profile,
        "device": str(device),
        "generation_config": generation_config,
        "checkpoints": results,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("docs/stages/04_pretrain/configs/mini64.json"),
    )
    parser.add_argument(
        "--generation-config",
        type=Path,
        default=Path("docs/stages/04_pretrain/configs/generation.json"),
    )
    parser.add_argument("--checkpoint", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_pretrain_config(args.config)
    generation_config = load_generation_config(args.generation_config)
    result = evaluate_checkpoints(config, args.checkpoint, generation_config)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(args.output)


if __name__ == "__main__":
    main()
