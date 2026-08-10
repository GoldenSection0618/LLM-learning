import json
from dataclasses import replace
from pathlib import Path

from transformers import AutoTokenizer

from llm_learning.minimind.config import load_training_config
from llm_learning.minimind.lm_eval import build_lm_eval_command
from llm_learning.minimind.sft_config import SFTConfig, load_sft_config
from llm_learning.minimind.sft_data import (
    MiniMindSFTDataset,
    create_sft_split_manifest,
    create_token_inspection,
    encode_sft_sample,
    load_sft_json_dataset,
    render_conversation,
    resolve_sft_split_row_ids,
)
from llm_learning.minimind.train import total_optimizer_steps


TOKENIZER_DIR = Path("third_party/minimind/model")


def make_config(**overrides):
    values = {
        "profile": "sft_test",
        "source_dir": Path("third_party/minimind"),
        "data_path": Path("data/test_sft.jsonl"),
        "output_dir": Path("outputs/test_sft"),
        "checkpoint_dir": Path("checkpoints/test_sft"),
        "seed": 42,
        "device": "cpu",
        "dtype": "float32",
        "hidden_size": 64,
        "num_hidden_layers": 1,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "sequence_length": 64,
        "batch_size": 2,
        "accumulation_steps": 1,
        "epochs": 1,
        "max_optimizer_steps": None,
        "learning_rate": 1e-5,
        "minimum_lr_ratio": 0.1,
        "gradient_clip": 1.0,
        "eval_interval": 10,
        "checkpoint_interval": 10,
        "num_workers": 0,
        "overfit_rows": 0,
        "validation_rows": 2,
        "periodic_validation_rows": 1,
        "overfit_loss_target": None,
        "initial_weights_path": Path("checkpoints/pretrain.pth"),
        "inspection_rows": 3,
        "add_system_ratio": 0.0,
        "empty_think_ratio": 0.2,
    }
    values.update(overrides)
    return SFTConfig(**values)


def simple_record():
    return {
        "conversations": [
            {"role": "system", "content": "保持简洁。"},
            {"role": "user", "content": "一加一等于几？"},
            {"role": "assistant", "content": "等于二。"},
        ]
    }


def test_sft_config_is_loaded_by_generic_loader():
    path = Path("docs/stages/05_sft/configs/overfit100.json")
    config = load_sft_config(path)
    assert isinstance(config, SFTConfig)
    assert isinstance(load_training_config(path), SFTConfig)
    assert config.epochs == 100
    assert total_optimizer_steps(config, train_rows=100) == 1000


def test_sft_json_loader_uses_complete_schema(tmp_path):
    path = tmp_path / "mixed.jsonl"
    rows = [
        {
            "conversations": [
                {"role": "user", "content": "普通问题", "reasoning_content": ""}
            ]
        },
        {
            "conversations": [
                {
                    "role": "system",
                    "content": "工具问题",
                    "reasoning_content": "",
                    "tools": "[]",
                    "tool_calls": "[]",
                }
            ]
        },
    ]
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )
    dataset = load_sft_json_dataset(path)
    assert len(dataset) == 2
    assert set(dataset[0]["conversations"][0]) == {
        "role",
        "content",
        "reasoning_content",
        "tools",
        "tool_calls",
    }


def test_response_mask_supervises_assistant_content_and_eos_only():
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_DIR)
    encoded = encode_sft_sample(simple_record(), 7, tokenizer, make_config())
    rows = create_token_inspection(encoded, tokenizer)

    assert any(row["next_role"] == "user" for row in rows)
    assert all(
        not row["computes_loss"]
        for row in rows
        if row["next_role"] in {"system", "user"}
    )
    assert any(
        row["computes_loss"] and row["next_segment"] == "assistant_response"
        for row in rows
    )
    assert all(
        not row["computes_loss"]
        for row in rows
        if row["next_segment"] == "assistant_marker"
    )
    assert any(
        row["computes_loss"]
        and row["next_segment"] == "assistant_end_marker"
        for row in rows
    )
    assert all("�" not in row["input_token"] for row in rows)
    active_ids = [label for label in encoded["labels"] if label != -100]
    active_text = tokenizer.decode(active_ids, skip_special_tokens=False)
    assert "等于二" in active_text
    assert tokenizer.eos_token in active_text


def test_rendering_augmentation_is_deterministic_per_row():
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_DIR)
    conversations = simple_record()["conversations"][1:]
    first = render_conversation(
        conversations,
        tokenizer,
        seed=42,
        row_id=11,
        add_system_ratio=1.0,
        empty_think_ratio=0.2,
    )
    second = render_conversation(
        conversations,
        tokenizer,
        seed=42,
        row_id=11,
        add_system_ratio=1.0,
        empty_think_ratio=0.2,
    )
    assert first == second
    assert first[0][0]["role"] == "system"


def test_truncation_is_reported_and_padding_stays_ignored():
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_DIR)
    complete = encode_sft_sample(simple_record(), 4, tokenizer, make_config())
    first_active = next(
        index for index, label in enumerate(complete["labels"]) if label != -100
    )
    short_config = replace(make_config(), sequence_length=first_active + 2)
    truncated = encode_sft_sample(simple_record(), 4, tokenizer, short_config)
    assert truncated["truncated"] is True
    assert truncated["supervised_tokens"] == 2

    dataset = MiniMindSFTDataset(
        [simple_record()],
        [0],
        tokenizer,
        make_config(),
    )
    sample = dataset[0]
    assert sample["input_ids"].shape == sample["labels"].shape == (64,)
    padding = sample["input_ids"].eq(tokenizer.pad_token_id)
    assert sample["labels"][padding].eq(-100).all()


def test_sft_split_is_deterministic_and_keeps_inspection_rows_in_full_train():
    records = [{"conversations": []} for _ in range(12)]
    config = make_config()
    first = create_sft_split_manifest(config, records, "raw", "fp", {"v": 1})
    second = create_sft_split_manifest(config, records, "raw", "fp", {"v": 1})
    train_ids, validation_ids = resolve_sft_split_row_ids(first)
    inspection_ids = first["split"]["inspection_row_ids"]
    assert first == second
    assert len(validation_ids) == 2
    assert set(train_ids).isdisjoint(validation_ids)
    assert set(inspection_ids).issubset(train_ids)


def test_lm_eval_command_applies_chat_template_for_sft(tmp_path):
    config = {
        "lm_eval_version": "0.4.12",
        "tasks": ["piqa"],
        "batch_size": 1,
        "device": "cpu",
        "apply_chat_template": True,
    }
    command = build_lm_eval_command(
        tmp_path / "model",
        tmp_path / "tasks",
        tmp_path / "results",
        config,
    )
    assert "--apply_chat_template" in command
