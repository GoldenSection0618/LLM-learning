"""阶段 5 的确定性对话渲染、response mask 与数据划分。"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

from .data import (
    DataLock,
    RecordCollection,
    download_locked_file,
    save_manifest,
    sha256_file,
    sha256_json,
    tokenizer_identity,
)
from .sft_config import SFTConfig, load_sft_config


SYSTEM_PROMPTS = (
    "你是一个知识丰富的AI，尽力为用户提供准确的信息。",
    "你是minimind，一个小巧但有用的语言模型。",
    "你是一个专业的AI助手，请提供有价值的回答。",
    "你是minimind，请尽力帮助用户解决问题。",
    "你是一个可靠的AI，请给出准确的回答。",
    "You are a helpful AI assistant.",
    "You are minimind, a lightweight intelligent assistant.",
    "You are a friendly chatbot. Please answer the user's questions carefully.",
    "You are a knowledgeable AI. Try your best to provide accurate information.",
    "You are minimind, a small but useful language model.",
)


def load_sft_json_dataset(path: Path) -> RecordCollection:
    """使用固定完整 schema 读取字段随样本变化的 SFT JSONL。"""
    from datasets import Features, Value, load_dataset

    features = Features(
        {
            "conversations": [
                {
                    "role": Value("string"),
                    "content": Value("string"),
                    "reasoning_content": Value("string"),
                    "tools": Value("string"),
                    "tool_calls": Value("string"),
                }
            ]
        }
    )
    return load_dataset(
        "json",
        data_files=str(path),
        split="train",
        features=features,
    )


def parse_json_value(value: Any) -> Any:
    """解析数据文件中以字符串保存的 JSON 字段。"""
    if not isinstance(value, str) or not value:
        return value
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def prepare_messages(
    conversations: Sequence[Mapping[str, Any]],
    *,
    seed: int,
    row_id: int,
    add_system_ratio: float,
) -> tuple[list[dict[str, Any]], Any, random.Random]:
    """按 row ID 确定性复现官方消息预处理。"""
    generator = random.Random((seed << 32) + row_id)
    messages = [dict(message) for message in conversations]
    has_tools = any(message.get("tools") for message in messages)
    if (
        messages
        and messages[0].get("role") != "system"
        and not has_tools
        and generator.random() < add_system_ratio
    ):
        messages.insert(
            0,
            {
                "role": "system",
                "content": generator.choice(SYSTEM_PROMPTS),
            },
        )

    tools = None
    for message in messages:
        if message.get("role") == "system" and message.get("tools"):
            tools = parse_json_value(message["tools"])
        if message.get("tool_calls"):
            message["tool_calls"] = parse_json_value(message["tool_calls"])
    return messages, tools, generator


def render_conversation(
    conversations: Sequence[Mapping[str, Any]],
    tokenizer: Any,
    *,
    seed: int,
    row_id: int,
    add_system_ratio: float,
    empty_think_ratio: float,
) -> tuple[list[dict[str, Any]], str]:
    """使用官方 chat template 渲染一条确定性对话。"""
    messages, tools, generator = prepare_messages(
        conversations,
        seed=seed,
        row_id=row_id,
        add_system_ratio=add_system_ratio,
    )
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
        tools=tools,
    )
    empty_think = "<think>\n\n</think>\n\n"
    if empty_think in prompt and generator.random() > empty_think_ratio:
        prompt = prompt.replace(empty_think, "")
    return messages, prompt


def assistant_marker_ids(tokenizer: Any) -> tuple[list[int], list[int]]:
    """返回官方 response mask 使用的 assistant 起止标记。"""
    start = tokenizer(
        f"{tokenizer.bos_token}assistant\n",
        add_special_tokens=False,
    ).input_ids
    end = tokenizer(
        f"{tokenizer.eos_token}\n",
        add_special_tokens=False,
    ).input_ids
    return list(start), list(end)


def classify_chat_roles(input_ids: Sequence[int], tokenizer: Any) -> list[str]:
    """按 chat template 标记每个 token 所在的消息 role。"""
    role_markers = {
        role: list(
            tokenizer(
                f"{tokenizer.bos_token}{role}\n",
                add_special_tokens=False,
            ).input_ids
        )
        for role in ["system", "user", "assistant", "tool"]
    }
    _, end_ids = assistant_marker_ids(tokenizer)
    roles = ["special"] * len(input_ids)
    current_role: str | None = None
    index = 0
    while index < len(input_ids):
        matched_role = next(
            (
                role
                for role, marker in role_markers.items()
                if list(input_ids[index : index + len(marker)]) == marker
            ),
            None,
        )
        if matched_role is not None:
            marker = role_markers[matched_role]
            roles[index : index + len(marker)] = [matched_role] * len(marker)
            current_role = matched_role
            index += len(marker)
            continue
        if list(input_ids[index : index + len(end_ids)]) == end_ids:
            roles[index : index + len(end_ids)] = [
                current_role or "special"
            ] * len(end_ids)
            current_role = None
            index += len(end_ids)
            continue
        roles[index] = current_role or "special"
        index += 1
    return roles


def create_token_inspection(
    encoded: Mapping[str, Any],
    tokenizer: Any,
) -> list[dict[str, Any]]:
    """用原文 offset 展开 labels 与 causal shift 后的 target。"""
    input_ids = list(encoded["input_ids"])
    labels = list(encoded["labels"])
    unpadded_length = int(encoded["unpadded_length"])
    prompt = str(encoded["prompt"])
    prompt_encoding = tokenizer(
        prompt,
        add_special_tokens=False,
        return_offsets_mapping=True,
    )
    prompt_ids = list(prompt_encoding.input_ids)
    offsets = list(prompt_encoding.offset_mapping)
    if prompt_ids[:unpadded_length] != input_ids[:unpadded_length]:
        raise ValueError("inspection tokenization does not match encoded input_ids")
    roles = classify_chat_roles(input_ids, tokenizer)
    segments = []
    for role, label in zip(roles, labels):
        if role != "assistant":
            segments.append(role)
        elif label == -100:
            segments.append("assistant_marker")
        else:
            segments.append("assistant_response")

    _, assistant_end_ids = assistant_marker_ids(tokenizer)
    for index in range(len(input_ids) - len(assistant_end_ids) + 1):
        if input_ids[index : index + len(assistant_end_ids)] != assistant_end_ids:
            continue
        end_positions = range(index, index + len(assistant_end_ids))
        if any(labels[position] != -100 for position in end_positions):
            segments[index : index + len(assistant_end_ids)] = [
                "assistant_end_marker"
            ] * len(assistant_end_ids)

    def source_span(position: int) -> tuple[str, tuple[int, int]]:
        start, end = offsets[position]
        return prompt[start:end], (int(start), int(end))

    rows = []
    for position in range(max(0, unpadded_length - 1)):
        next_position = position + 1
        shifted_target = labels[next_position]
        input_text, input_offset = source_span(position)
        next_text, next_offset = source_span(next_position)
        rows.append(
            {
                "position": position,
                "input_token": input_text,
                "input_offset": input_offset,
                "input_token_id": input_ids[position],
                "input_role": roles[position],
                "input_segment": segments[position],
                "stored_label": labels[position],
                "next_token": next_text,
                "next_offset": next_offset,
                "next_token_id": input_ids[next_position],
                "next_role": roles[next_position],
                "next_segment": segments[next_position],
                "shifted_target": shifted_target,
                "computes_loss": shifted_target != -100,
            }
        )
    return rows


def generate_response_labels(
    input_ids: Sequence[int],
    assistant_start_ids: Sequence[int],
    assistant_end_ids: Sequence[int],
) -> list[int]:
    """只保留 assistant 内容及其结束标记对应的 labels。"""
    labels = [-100] * len(input_ids)
    index = 0
    while index < len(input_ids):
        if list(input_ids[index : index + len(assistant_start_ids)]) != list(
            assistant_start_ids
        ):
            index += 1
            continue
        start = index + len(assistant_start_ids)
        end = start
        while end < len(input_ids):
            if list(input_ids[end : end + len(assistant_end_ids)]) == list(
                assistant_end_ids
            ):
                break
            end += 1
        supervised_end = min(end + len(assistant_end_ids), len(input_ids))
        for position in range(start, supervised_end):
            labels[position] = int(input_ids[position])
        index = supervised_end
    return labels


def encode_sft_sample(
    record: Mapping[str, Any],
    row_id: int,
    tokenizer: Any,
    config: SFTConfig,
) -> dict[str, Any]:
    """把一条 conversations 记录编码为 SFT 输入与 labels。"""
    conversations = record.get("conversations")
    if not isinstance(conversations, Sequence) or isinstance(conversations, str):
        raise ValueError(f"row {row_id} 缺少 conversations 列表")
    messages, prompt = render_conversation(
        conversations,
        tokenizer,
        seed=config.seed,
        row_id=row_id,
        add_system_ratio=config.add_system_ratio,
        empty_think_ratio=config.empty_think_ratio,
    )
    untruncated_ids = list(tokenizer(prompt, add_special_tokens=False).input_ids)
    input_ids = untruncated_ids[: config.sequence_length]
    padding_length = config.sequence_length - len(input_ids)
    input_ids += [tokenizer.pad_token_id] * padding_length
    assistant_start_ids, assistant_end_ids = assistant_marker_ids(tokenizer)
    labels = generate_response_labels(
        input_ids,
        assistant_start_ids,
        assistant_end_ids,
    )
    supervised_tokens = sum(label != -100 for label in labels[1:])
    return {
        "messages": messages,
        "prompt": prompt,
        "input_ids": input_ids,
        "labels": labels,
        "supervised_tokens": supervised_tokens,
        "truncated": len(untruncated_ids) > config.sequence_length,
        "unpadded_length": min(len(untruncated_ids), config.sequence_length),
    }


class MiniMindSFTDataset(Dataset[dict[str, torch.Tensor]]):
    """按固定 row ID 构造 response-only labels。"""

    def __init__(
        self,
        records: RecordCollection,
        row_ids: Sequence[int],
        tokenizer: Any,
        config: SFTConfig,
    ) -> None:
        self.records = records
        self.row_ids = list(row_ids)
        self.tokenizer = tokenizer
        self.config = config

    def __len__(self) -> int:
        return len(self.row_ids)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        row_id = self.row_ids[index]
        encoded = encode_sft_sample(
            self.records[row_id],
            row_id,
            self.tokenizer,
            self.config,
        )
        return {
            "input_ids": torch.tensor(encoded["input_ids"], dtype=torch.long),
            "labels": torch.tensor(encoded["labels"], dtype=torch.long),
            "row_id": torch.tensor(row_id, dtype=torch.long),
            "supervised_tokens": torch.tensor(
                encoded["supervised_tokens"],
                dtype=torch.long,
            ),
            "truncated": torch.tensor(encoded["truncated"], dtype=torch.bool),
        }


def create_sft_split_manifest(
    config: SFTConfig,
    records: RecordCollection,
    raw_sha256: str,
    dataset_fingerprint: str | None,
    tokenizer_info: dict[str, Any],
) -> dict[str, Any]:
    """固定教学子集、正式 validation 与完整训练集合。"""
    row_count = len(records)
    required = config.inspection_rows + config.validation_rows
    if row_count < required:
        raise ValueError("SFT 数据行数不足以创建固定划分")
    order = torch.randperm(
        row_count,
        generator=torch.Generator().manual_seed(config.seed),
    ).tolist()
    inspection_row_ids = order[: config.inspection_rows]
    if config.overfit_rows:
        train_row_ids = inspection_row_ids[: config.overfit_rows]
        validation_row_ids = train_row_ids
        rule = "seeded_rows_for_overfit"
    else:
        validation_row_ids = order[
            config.inspection_rows : config.inspection_rows + config.validation_rows
        ]
        validation_set = set(validation_row_ids)
        train_row_ids = [row_id for row_id in range(row_count) if row_id not in validation_set]
        rule = "seeded_inspection_then_validation_then_train_complement"
    split = {
        "seed": config.seed,
        "rule": rule,
        "row_count": row_count,
        "train_count": len(train_row_ids),
        "validation_row_ids": validation_row_ids,
        "inspection_row_ids": inspection_row_ids,
        "overfit_row_ids": train_row_ids if config.overfit_rows else [],
    }
    return {
        "data_path": str(config.data_path),
        "raw_sha256": raw_sha256,
        "dataset_fingerprint": dataset_fingerprint,
        "tokenizer": tokenizer_info,
        "split": split,
        "split_sha256": sha256_json(split),
    }


def resolve_sft_split_row_ids(
    manifest: dict[str, Any],
) -> tuple[list[int], list[int]]:
    """从 SFT manifest 恢复 train 与 validation row ID。"""
    split = manifest["split"]
    validation_row_ids = list(split["validation_row_ids"])
    if split["rule"] == "seeded_rows_for_overfit":
        return list(split["overfit_row_ids"]), validation_row_ids
    validation_set = set(validation_row_ids)
    train_row_ids = [
        row_id
        for row_id in range(split["row_count"])
        if row_id not in validation_set
    ]
    return train_row_ids, validation_row_ids


def build_sft_dataset(
    config: SFTConfig,
    records: RecordCollection,
    row_ids: Sequence[int],
    tokenizer: Any,
) -> Dataset:
    """构造训练循环使用的 SFT Dataset。"""
    return MiniMindSFTDataset(records, row_ids, tokenizer, config)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--lock",
        type=Path,
        default=Path("docs/stages/05_sft/configs/data_lock.json"),
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("docs/stages/05_sft/configs/overfit100.json"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_sft_config(args.config)
    lock = DataLock.load(args.lock)
    data_path = download_locked_file(lock, config.data_path)
    records = load_sft_json_dataset(data_path)
    tokenizer = AutoTokenizer.from_pretrained(config.source_dir / "model")
    manifest = create_sft_split_manifest(
        config,
        records,
        sha256_file(data_path),
        getattr(records, "_fingerprint", None),
        tokenizer_identity(config.source_dir / "model", tokenizer),
    )
    destination = config.output_dir / "data_manifest.json"
    save_manifest(destination, manifest)
    print(destination)


if __name__ == "__main__":
    main()
