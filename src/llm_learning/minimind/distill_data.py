"""阶段 8 GSM8K 固定划分、Teacher 生成与自动验证。"""

from __future__ import annotations

import argparse
import json
import random
import urllib.error
import urllib.request
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping, Sequence, cast

from .distill_config import SequenceTeacherConfig, load_sequence_teacher_config

if TYPE_CHECKING:
    from datasets import Dataset as HuggingFaceDataset
else:
    HuggingFaceDataset = Any


SYSTEM_PROMPT = (
    "Solve the math problem step by step. End with the final numerical answer "
    "inside \\boxed{}."
)


def build_gsm8k_split(
    row_count: int,
    *,
    seed: int,
    development_rows: int,
    teaching_rows: int,
) -> dict[str, Any]:
    """以原始 row ID 生成一次性固定划分。"""
    if development_rows + teaching_rows > row_count:
        raise ValueError("GSM8K rows are insufficient for the requested split")
    order = list(range(row_count))
    random.Random(seed).shuffle(order)
    development_ids = order[:development_rows]
    training_ids = order[development_rows:]
    return {
        "version": 1,
        "seed": seed,
        "rule": "shuffle_train_once_then_take_development_prefix",
        "row_count": row_count,
        "development_row_ids": development_ids,
        "training_row_ids": training_ids,
        "teaching_100_row_ids": training_ids[:teaching_rows],
    }


def gsm8k_gold_answer(answer: str) -> str:
    """提取 GSM8K `####` 后的最终数值。"""
    marker = "####"
    if marker not in answer:
        raise ValueError("GSM8K answer does not contain the final-answer marker")
    return answer.rsplit(marker, maxsplit=1)[1].strip().replace(",", "")


def api_json(
    url: str,
    *,
    timeout: int,
    payload: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """访问 LM Studio OpenAI-compatible JSON API。"""
    body = None if payload is None else json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="GET" if body is None else "POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.URLError as error:
        raise ConnectionError(f"cannot reach LM Studio API: {url}") from error


def resolve_teacher_model(config: SequenceTeacherConfig) -> str:
    """从 LM Studio 已加载模型中选取配置指定的 Teacher。"""
    response = api_json(
        f"{config.api_base.rstrip('/')}/models",
        timeout=config.request_timeout_seconds,
    )
    model_ids = [str(item["id"]) for item in response.get("data", [])]
    matches = [
        model_id
        for model_id in model_ids
        if config.model_name_contains.lower() in model_id.lower()
    ]
    if len(matches) != 1:
        raise RuntimeError(
            "expected exactly one loaded Teacher containing "
            f"{config.model_name_contains!r}; received {model_ids}"
        )
    return matches[0]


def generate_teacher_response(
    question: str,
    *,
    row_id: int,
    model_id: str,
    config: SequenceTeacherConfig,
) -> dict[str, Any]:
    """请求一条 Teacher response 并保留 API 用量。"""
    payload = {
        "model": model_id,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ],
        "temperature": config.temperature,
        "top_k": config.top_k,
        "top_p": config.top_p,
        "presence_penalty": config.presence_penalty,
        "max_tokens": config.max_tokens,
        "seed": config.seed + row_id,
        "stream": False,
    }
    response = api_json(
        f"{config.api_base.rstrip('/')}/chat/completions",
        timeout=config.request_timeout_seconds,
        payload=payload,
    )
    choice = response["choices"][0]
    message = choice["message"]
    content = str(message.get("content") or "").strip()
    reasoning = str(
        message.get("reasoning_content") or message.get("reasoning") or ""
    ).strip()
    finish_reason = choice.get("finish_reason")
    return {
        "response": content,
        "reasoning_content": reasoning,
        "finish_reason": finish_reason,
        "complete": finish_reason == "stop" and bool(content),
        "usage": response.get("usage", {}),
    }


def verify_math_answer(gold_answer: str, response: str) -> bool:
    """使用 Math-Verify 比较 GSM8K gold 与 Teacher response。"""
    try:
        from math_verify import parse, verify
        from math_verify.parser import ExprExtractionConfig, LatexExtractionConfig
    except ImportError as error:
        raise RuntimeError(
            "math-verify is required; update the llm environment first"
        ) from error

    gold = parse(
        gold_answer,
        extraction_config=[ExprExtractionConfig()],
    )
    prediction = parse(
        response,
        extraction_config=[
            LatexExtractionConfig(boxed_match_priority=0),
            ExprExtractionConfig(),
        ],
    )
    return bool(gold and prediction and verify(gold, prediction))


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _append_jsonl(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(value, ensure_ascii=False) + "\n")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    """读取已生成的 JSONL；文件不存在时返回空列表。"""
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def load_gsm8k_dataset(config: SequenceTeacherConfig) -> HuggingFaceDataset:
    """加载支持 `len` 和按 row ID 索引的普通 Hugging Face Dataset。"""
    from datasets import Dataset, load_dataset

    loaded = load_dataset(
        config.dataset_id,
        config.dataset_config,
        split=config.dataset_split,
        revision=config.dataset_revision,
    )
    if not isinstance(loaded, Dataset):
        raise TypeError("GSM8K split must load as a non-streaming Dataset")
    return cast(HuggingFaceDataset, loaded)


def prepare_split(config: SequenceTeacherConfig) -> dict[str, Any]:
    """下载 GSM8K metadata 并写入固定 row ID 清单。"""
    dataset = load_gsm8k_dataset(config)
    split = build_gsm8k_split(
        len(dataset),
        seed=config.seed,
        development_rows=config.development_rows,
        teaching_rows=config.teaching_rows,
    )
    split.update(
        {
            "dataset_id": config.dataset_id,
            "dataset_config": config.dataset_config,
            "dataset_split": config.dataset_split,
            "dataset_revision": config.dataset_revision,
            "dataset_fingerprint": getattr(dataset, "_fingerprint", None),
        }
    )
    _write_json(config.split_manifest_path, split)
    return split


def generate_verified_data(config: SequenceTeacherConfig) -> dict[str, Any]:
    """逐条生成、验证并导出 MiniMind SFT 数据。"""
    if not config.split_manifest_path.exists():
        raise FileNotFoundError("prepare the GSM8K split before generation")
    split = json.loads(config.split_manifest_path.read_text(encoding="utf-8"))
    dataset = load_gsm8k_dataset(config)
    if getattr(dataset, "_fingerprint", None) != split["dataset_fingerprint"]:
        raise RuntimeError("GSM8K dataset fingerprint changed")

    model_id = resolve_teacher_model(config)
    existing = load_jsonl(config.raw_output_path)
    completed_ids = {int(row["row_id"]) for row in existing}
    requested_ids: Sequence[int] = split["teaching_100_row_ids"]
    for index, row_id in enumerate(requested_ids, start=1):
        if row_id in completed_ids:
            continue
        record = dataset[row_id]
        generated = generate_teacher_response(
            str(record["question"]),
            row_id=row_id,
            model_id=model_id,
            config=config,
        )
        gold = gsm8k_gold_answer(str(record["answer"]))
        result = {
            "row_id": row_id,
            "question": record["question"],
            "gold_answer": gold,
            "teacher_model": model_id,
            "response": generated["response"],
            "reasoning_content": generated["reasoning_content"],
            "finish_reason": generated["finish_reason"],
            "usage": generated["usage"],
            "complete": generated["complete"],
            "verified": (
                bool(generated["complete"])
                and verify_math_answer(gold, str(generated["response"]))
            ),
        }
        _append_jsonl(config.raw_output_path, result)
        print(
            f"[{index}/{len(requested_ids)}] row_id={row_id} "
            f"verified={result['verified']} finish={result['finish_reason']}",
            flush=True,
        )

    results = load_jsonl(config.raw_output_path)
    by_id = {int(row["row_id"]): row for row in results}
    ordered = [by_id[row_id] for row_id in requested_ids if row_id in by_id]
    verified = [row for row in ordered if row["verified"]]
    config.verified_sft_path.parent.mkdir(parents=True, exist_ok=True)
    with config.verified_sft_path.open("w", encoding="utf-8") as file:
        for row in verified:
            sft_record = {
                "conversations": [
                    {"role": "user", "content": row["question"]},
                    {"role": "assistant", "content": row["response"]},
                ]
            }
            file.write(json.dumps(sft_record, ensure_ascii=False) + "\n")
    return {
        "teacher_model": model_id,
        "requested": len(requested_ids),
        "completed": len(ordered),
        "verified": len(verified),
        "raw_output": str(config.raw_output_path),
        "verified_sft": str(config.verified_sft_path),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["prepare", "generate"])
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("docs/stages/08_distillation/configs/sequence_teacher.json"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_sequence_teacher_config(args.config)
    result = (
        prepare_split(config)
        if args.command == "prepare"
        else generate_verified_data(config)
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
