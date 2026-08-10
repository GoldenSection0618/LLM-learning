import json
import random
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from llm_learning.minimind.checkpoint import load_checkpoint, save_checkpoint
from llm_learning.minimind.config import PretrainConfig
from llm_learning.minimind.data import (
    EpochBatchSampler,
    MiniMindPretrainDataset,
    create_split_manifest,
    resolve_split_row_ids,
    sha256_json,
)
from llm_learning.minimind.evaluate import filter_logits, generate_one
from llm_learning.minimind.inspect import load_official_model_module
from llm_learning.minimind.prepare_lm_eval import (
    TASK_MANIFEST_NAME,
    inject_dataset_revision,
    reuse_existing_task_directory,
    sha256_directory,
)
from llm_learning.minimind.train import (
    cosine_learning_rate,
    evaluate_loss,
    total_optimizer_steps,
)


class FakeTokenizer:
    bos_token_id = 1
    eos_token_id = 2
    pad_token_id = 0
    bos_token = "<bos>"

    def __call__(self, text, **kwargs):
        if kwargs.get("return_tensors") == "pt":
            return SimpleNamespace(input_ids=torch.tensor([[1, 3]], dtype=torch.long))
        maximum = kwargs["max_length"]
        return SimpleNamespace(input_ids=[3, 4, 5, 6][:maximum])

    def decode(self, token_ids, skip_special_tokens=True):
        values = token_ids.tolist() if isinstance(token_ids, torch.Tensor) else token_ids
        return " ".join(str(value) for value in values)


def make_config(**overrides):
    values = {
        "profile": "test",
        "source_dir": Path("third_party/minimind"),
        "data_path": Path("data/test.jsonl"),
        "output_dir": Path("outputs/test"),
        "checkpoint_dir": Path("checkpoints/test"),
        "seed": 42,
        "device": "cpu",
        "dtype": "float32",
        "hidden_size": 64,
        "num_hidden_layers": 1,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "sequence_length": 8,
        "batch_size": 3,
        "accumulation_steps": 2,
        "epochs": 2,
        "max_optimizer_steps": None,
        "learning_rate": 5e-4,
        "minimum_lr_ratio": 0.1,
        "gradient_clip": 1.0,
        "eval_interval": 10,
        "checkpoint_interval": 10,
        "num_workers": 0,
        "overfit_rows": 2,
        "validation_rows": 0,
        "periodic_validation_rows": 0,
        "overfit_loss_target": 1.0,
    }
    values.update(overrides)
    return PretrainConfig(**values)


def test_cosine_learning_rate_uses_optimizer_step_endpoints():
    assert cosine_learning_rate(0, 5, 1.0, 0.1) == pytest.approx(1.0)
    assert cosine_learning_rate(4, 5, 1.0, 0.1) == pytest.approx(0.1)
    assert cosine_learning_rate(0, 1, 1.0, 0.1) == pytest.approx(1.0)


def test_total_optimizer_steps_includes_tail_update():
    config = make_config(
        overfit_rows=0,
        validation_rows=1,
        overfit_loss_target=None,
    )
    assert total_optimizer_steps(config, train_rows=10) == 4


def test_dataset_adds_special_tokens_and_masks_only_padding():
    dataset = MiniMindPretrainDataset(
        records=[{"text": "ignored"}],
        row_ids=[0],
        tokenizer=FakeTokenizer(),
        sequence_length=8,
    )
    sample = dataset[0]
    assert sample["input_ids"].tolist() == [1, 3, 4, 5, 6, 2, 0, 0]
    assert sample["labels"].tolist() == [1, 3, 4, 5, 6, 2, -100, -100]
    assert sample["attention_mask"].tolist() == [True] * 6 + [False, False]
    assert sample["row_id"].item() == 0


def test_epoch_batch_sampler_can_reconstruct_and_skip_batches():
    complete = list(EpochBatchSampler(10, 3, seed=7, epoch=2))
    repeated = list(EpochBatchSampler(10, 3, seed=7, epoch=2))
    resumed = list(EpochBatchSampler(10, 3, seed=7, epoch=2, start_batch=2))
    assert complete == repeated
    assert resumed == complete[2:]
    assert sorted(index for batch in complete for index in batch) == list(range(10))


def test_split_manifest_is_deterministic_and_recoverable():
    records = [{"text": str(index)} for index in range(12)]
    config = make_config(
        overfit_rows=0,
        validation_rows=3,
        periodic_validation_rows=2,
        overfit_loss_target=None,
    )
    first = create_split_manifest(config, records, "raw", "fingerprint", {"v": 1})
    second = create_split_manifest(config, records, "raw", "fingerprint", {"v": 1})
    train_ids, validation_ids = resolve_split_row_ids(first)
    assert first == second
    assert len(train_ids) == 9
    assert len(validation_ids) == 3
    assert set(train_ids).isdisjoint(validation_ids)
    assert first["split_sha256"] == sha256_json(first["split"])


class FixedLossModel(torch.nn.Module):
    def forward(self, input_ids, labels=None, use_cache=False):
        loss = input_ids[:, 0].float().mean()
        return SimpleNamespace(loss=loss)


def test_evaluate_loss_is_weighted_by_valid_token_count():
    loader = [
        {
            "input_ids": torch.tensor([[2, 0, 0]]),
            "labels": torch.tensor([[1, 2, -100]]),
        },
        {
            "input_ids": torch.tensor([[4, 0, 0]]),
            "labels": torch.tensor([[1, 2, 3]]),
        },
    ]
    result = evaluate_loss(FixedLossModel(), loader, torch.device("cpu"), "float32")
    assert result["tokens"] == 3
    assert result["loss"] == pytest.approx((2.0 * 1 + 4.0 * 2) / 3)


def test_right_padding_without_attention_mask_preserves_valid_logits():
    module = load_official_model_module(Path("third_party/minimind"))
    torch.manual_seed(4)
    config = module.MiniMindConfig(
        hidden_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        vocab_size=32,
        max_position_embeddings=16,
        flash_attn=True,
        dropout=0.0,
    )
    model = module.MiniMindForCausalLM(config).eval()
    padded = torch.tensor([[1, 5, 6, 2, 0], [1, 7, 8, 9, 2]])
    with torch.no_grad():
        batch_logits = model(padded).logits
        first_logits = model(padded[:1, :4]).logits
        second_logits = model(padded[1:, :5]).logits
    torch.testing.assert_close(batch_logits[0, :4], first_logits[0], atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(batch_logits[1, :5], second_logits[0], atol=1e-5, rtol=1e-5)


def make_checkpoint_parts(seed):
    torch.manual_seed(seed)
    model = torch.nn.Linear(3, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    scaler = torch.GradScaler("cuda", enabled=False)
    return model, optimizer, scaler


def take_update(model, optimizer, inputs):
    optimizer.zero_grad(set_to_none=True)
    model(inputs).square().mean().backward()
    optimizer.step()


def test_checkpoint_roundtrip_restores_exact_continuation_and_rng(tmp_path):
    inputs = torch.tensor([[0.2, -0.1, 0.5]])
    reference, reference_optimizer, _ = make_checkpoint_parts(9)
    take_update(reference, reference_optimizer, inputs)
    take_update(reference, reference_optimizer, inputs)

    model, optimizer, scaler = make_checkpoint_parts(9)
    take_update(model, optimizer, inputs)
    checkpoint_path = tmp_path / "latest.pt"
    training_state = {
        "epoch": 0,
        "next_batch": 1,
        "optimizer_step": 1,
        "trained_tokens": 3,
        "microbatches_in_update": 0,
    }
    random.seed(123)
    torch.manual_seed(123)
    save_checkpoint(
        checkpoint_path,
        model,
        optimizer,
        scaler,
        training_state,
        {"profile": "test", "source_dir": "previous/location/minimind"},
        {"hash": "tokenizer"},
        {
            "raw_sha256": "raw",
            "split_sha256": "split",
            "dataset_fingerprint": "old-library-fingerprint",
        },
        [{"loss": 1.0}],
    )
    expected_python_random = random.random()
    expected_torch_random = torch.rand(1)

    restored, restored_optimizer, restored_scaler = make_checkpoint_parts(99)
    checkpoint = load_checkpoint(
        checkpoint_path,
        restored,
        restored_optimizer,
        restored_scaler,
        {"profile": "test", "source_dir": "third_party/minimind"},
        {"hash": "tokenizer"},
        {
            "raw_sha256": "raw",
            "split_sha256": "split",
            "dataset_fingerprint": "new-library-fingerprint",
        },
        torch.device("cpu"),
    )
    assert checkpoint["training_state"] == training_state
    assert random.random() == expected_python_random
    torch.testing.assert_close(torch.rand(1), expected_torch_random)
    take_update(restored, restored_optimizer, inputs)
    for actual, expected in zip(restored.parameters(), reference.parameters()):
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)


def test_checkpoint_rejects_partial_accumulation_window(tmp_path):
    model, optimizer, scaler = make_checkpoint_parts(1)
    with pytest.raises(ValueError, match="partial accumulation"):
        save_checkpoint(
            tmp_path / "bad.pt",
            model,
            optimizer,
            scaler,
            {"microbatches_in_update": 1},
            {},
            {},
            {},
            [],
        )


class SamplingModel(torch.nn.Module):
    def forward(self, input_ids, past_key_values=None, use_cache=True):
        logits = torch.zeros(input_ids.shape[0], input_ids.shape[1], 8)
        logits[..., 3:7] = torch.tensor([1.0, 0.8, 0.6, 0.4])
        return SimpleNamespace(logits=logits, past_key_values=("cached",))


def test_sampling_generation_uses_reproducible_independent_generator():
    generation = {
        "do_sample": True,
        "temperature": 0.85,
        "top_k": 4,
        "top_p": 0.95,
        "max_new_tokens": 6,
        "eos_token_id": 2,
        "seed": 2026,
    }
    torch.manual_seed(7)
    before = torch.get_rng_state()
    first = generate_one(
        SamplingModel(), FakeTokenizer(), "prompt", generation, torch.device("cpu")
    )
    after = torch.get_rng_state()
    second = generate_one(
        SamplingModel(), FakeTokenizer(), "prompt", generation, torch.device("cpu")
    )
    assert first["token_ids"] == second["token_ids"]
    torch.testing.assert_close(before, after)


def test_filter_logits_keeps_only_requested_top_k():
    logits = torch.tensor([[1.0, 3.0, 2.0, 0.0]])
    filtered = filter_logits(logits, top_k=2, top_p=1.0)
    assert torch.isfinite(filtered).tolist() == [[False, True, True, False]]


def test_inject_dataset_revision_adds_dataset_kwargs():
    source = "task: piqa\ndataset_path: baber/piqa\ntest_split: validation\n"
    rendered, changed = inject_dataset_revision(source, "142f6d7")
    assert changed
    assert (
        "dataset_path: baber/piqa\ndataset_kwargs:\n"
        "  revision: 142f6d7\n"
        "  trust_remote_code: true\n"
    ) in rendered


def test_inject_dataset_revision_preserves_existing_kwargs():
    source = (
        "dataset_path: ceval/ceval-exam\n"
        "dataset_kwargs:\n"
        "  trust_remote_code: true\n"
        "test_split: test\n"
    )
    rendered, changed = inject_dataset_revision(source, "617524a")
    assert changed
    assert "dataset_kwargs:\n  revision: 617524a\n" in rendered
    assert "  trust_remote_code: true\n" in rendered
    assert rendered.count("revision:") == 1


def test_existing_lm_eval_tasks_are_reused_only_when_hash_matches(tmp_path):
    destination = tmp_path / "tasks"
    destination.mkdir()
    (destination / "task.yaml").write_text("task: example\n", encoding="utf-8")
    config = {
        "lm_eval_version": "0.4.12",
        "lm_eval_commit": "revision",
        "trust_remote_code": True,
        "tasks": {"example": "dataset-revision"},
        "task_sources": {"example": "example"},
    }
    manifest = {
        **config,
        "modified_files": {"example": ["task.yaml"]},
        "task_directory_sha256": sha256_directory(destination),
    }
    (destination / TASK_MANIFEST_NAME).write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )

    assert reuse_existing_task_directory(destination, config) == manifest

    (destination / "task.yaml").write_text("task: changed\n", encoding="utf-8")
    with pytest.raises(ValueError, match="manifest hash"):
        reuse_existing_task_directory(destination, config)
