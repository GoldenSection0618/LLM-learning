"""阶段 8 蒸馏数据与 loss 测试。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from llm_learning.minimind.distill_config import (
    load_logit_distillation_config,
    load_sequence_teacher_config,
)
from llm_learning.minimind.distill_data import (
    build_gsm8k_split,
    generate_teacher_response,
    gsm8k_gold_answer,
)
from llm_learning.minimind.distill_train import fixed_row_ids
from llm_learning.minimind.distillation import (
    distillation_losses,
    masked_kl_divergence,
)


ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = ROOT / "docs/stages/08_distillation/configs"


def test_gsm8k_split_is_deterministic_and_disjoint() -> None:
    first = build_gsm8k_split(
        7473,
        seed=42,
        development_rows=500,
        teaching_rows=100,
    )
    second = build_gsm8k_split(
        7473,
        seed=42,
        development_rows=500,
        teaching_rows=100,
    )
    assert first == second
    assert len(first["development_row_ids"]) == 500
    assert len(first["training_row_ids"]) == 6973
    assert len(first["teaching_100_row_ids"]) == 100
    assert set(first["development_row_ids"]).isdisjoint(
        first["training_row_ids"]
    )


def test_gsm8k_gold_answer_extracts_final_value() -> None:
    assert gsm8k_gold_answer("work\n#### 1,234") == "1234"
    with pytest.raises(ValueError):
        gsm8k_gold_answer("missing marker")


def test_distillation_loss_uses_only_supervised_shifted_positions() -> None:
    torch.manual_seed(7)
    student = torch.randn(1, 4, 5, requires_grad=True)
    teacher = torch.randn(1, 4, 5)
    labels = torch.tensor([[-100, -100, 3, 4]])
    losses = distillation_losses(
        student,
        teacher,
        labels,
        ce_weight=0.5,
        temperature=1.5,
    )
    assert losses.supervised_tokens == 2
    losses.total.backward()
    assert student.grad is not None
    assert torch.count_nonzero(student.grad[:, 0]).item() == 0
    assert torch.count_nonzero(student.grad[:, 1]).item() > 0
    assert torch.count_nonzero(student.grad[:, 2]).item() > 0
    assert torch.count_nonzero(student.grad[:, 3]).item() == 0


def test_temperature_squared_keeps_kd_scale_comparable() -> None:
    student = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
    teacher = torch.tensor([[[0.0, 1.0], [1.0, 0.0]]])
    labels = torch.tensor([[-100, 1]])
    low = distillation_losses(
        student,
        teacher,
        labels,
        ce_weight=0.0,
        temperature=1.0,
    )
    high = distillation_losses(
        student,
        teacher,
        labels,
        ce_weight=0.0,
        temperature=8.0,
    )
    assert low.kd is not None
    assert high.kd is not None
    assert low.kd.item() > 0
    assert high.kd.item() > 0
    assert high.kd.item() == pytest.approx(low.kd.item(), rel=0.4)


def test_zero_weight_loss_branch_is_not_computed() -> None:
    student = torch.randn(1, 2, 4)
    teacher = torch.randn(1, 2, 4)
    labels = torch.tensor([[-100, 2]])

    ce_only = distillation_losses(
        student,
        None,
        labels,
        ce_weight=1.0,
        temperature=1.5,
    )
    kd_only = distillation_losses(
        student,
        teacher,
        labels,
        ce_weight=0.0,
        temperature=1.5,
    )

    assert ce_only.ce is not None
    assert ce_only.kd is None
    assert kd_only.ce is None
    assert kd_only.kd is not None


def test_forward_and_reverse_kl_use_the_requested_distribution_order() -> None:
    student_probabilities = torch.tensor([0.45, 0.45, 0.10])
    teacher_probabilities = torch.tensor([0.80, 0.15, 0.05])
    student = (
        student_probabilities.log().reshape(1, 1, 3).repeat(1, 2, 1)
        .clone()
        .requires_grad_()
    )
    teacher = teacher_probabilities.log().reshape(1, 1, 3).repeat(1, 2, 1)
    labels = torch.tensor([[-100, 0]])

    forward = masked_kl_divergence(
        student, teacher, labels, 1.0, direction="forward"
    )
    reverse = masked_kl_divergence(
        student, teacher, labels, 1.0, direction="reverse"
    )

    expected_forward = torch.sum(
        teacher_probabilities
        * (teacher_probabilities.log() - student_probabilities.log())
    )
    expected_reverse = torch.sum(
        student_probabilities
        * (student_probabilities.log() - teacher_probabilities.log())
    )
    assert forward.item() == pytest.approx(expected_forward.item())
    assert reverse.item() == pytest.approx(expected_reverse.item())
    assert forward.item() != pytest.approx(reverse.item())
    reverse.backward()
    assert student.grad is not None


def test_unknown_kl_direction_is_rejected() -> None:
    logits = torch.zeros(1, 2, 3)
    labels = torch.tensor([[-100, 0]])
    with pytest.raises(ValueError, match="direction"):
        masked_kl_divergence(
            logits,
            logits,
            labels,
            1.0,
            direction="sideways",
        )


def test_temperature_scaling_is_reported_separately_from_raw_kl() -> None:
    student = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
    teacher = torch.tensor([[[0.0, 1.0], [1.0, 0.0]]])
    labels = torch.tensor([[-100, 1]])
    raw = masked_kl_divergence(
        student,
        teacher,
        labels,
        2.0,
        scale_by_temperature=False,
    )
    scaled = masked_kl_divergence(student, teacher, labels, 2.0)
    assert scaled.item() == pytest.approx(raw.item() * 4)


def test_all_logit_configs_share_data_and_row_selection() -> None:
    configs = [
        load_logit_distillation_config(CONFIG_DIR / name)
        for name in [
            "ce.json",
            "kd.json",
            "reverse_kd.json",
            "ce_kd.json",
            "ce_kd_t2.json",
        ]
    ]
    shared = {
        (
            config.data_path,
            config.seed,
            config.train_rows,
            config.validation_rows,
            config.student_weights_path,
        )
        for config in configs
    }
    assert len(shared) == 1
    assert [config.ce_weight for config in configs] == [1.0, 0.0, 0.0, 0.5, 0.5]
    assert [config.temperature for config in configs] == [1.5, 1.5, 1.5, 1.5, 2.0]
    assert [config.kl_direction for config in configs] == [
        "forward",
        "forward",
        "reverse",
        "forward",
        "forward",
    ]


def test_fixed_logit_split_is_disjoint() -> None:
    train, validation = fixed_row_ids(
        10_000,
        seed=42,
        train_rows=4096,
        validation_rows=256,
    )
    assert len(train) == 4096
    assert len(validation) == 256
    assert set(train).isdisjoint(validation)


def test_sequence_config_does_not_store_local_model_path() -> None:
    path = CONFIG_DIR / "sequence_teacher.json"
    config = load_sequence_teacher_config(path)
    raw = json.loads(path.read_text(encoding="utf-8"))
    assert config.model_name_contains == "qwen3.5-9b"
    assert not any("D:\\" in str(value) or "/mnt/d/" in str(value) for value in raw.values())


def test_sequence_config_accepts_runtime_api_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LM_STUDIO_BASE_URL", "http://windows-host:1234/v1")
    config = load_sequence_teacher_config(CONFIG_DIR / "sequence_teacher.json")
    assert config.api_base == "http://windows-host:1234/v1"


def test_teacher_response_keeps_reasoning_separate_from_content(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = load_sequence_teacher_config(CONFIG_DIR / "sequence_teacher.json")
    captured: dict[str, object] = {}

    def fake_api_json(url: str, *, timeout: int, payload=None):
        captured.update({"url": url, "timeout": timeout, "payload": payload})
        return {
            "choices": [
                {
                    "message": {
                        "reasoning": "12 / 3 = 4",
                        "content": "The answer is \\boxed{4}.",
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {"completion_tokens": 12},
        }

    monkeypatch.setattr(
        "llm_learning.minimind.distill_data.api_json",
        fake_api_json,
    )
    result = generate_teacher_response(
        "How many?",
        row_id=17,
        model_id="qwen/qwen3.5-9b",
        config=config,
    )
    assert result["response"] == "The answer is \\boxed{4}."
    assert result["reasoning_content"] == "12 / 3 = 4"
    assert result["finish_reason"] == "stop"
    assert result["complete"] is True
    assert captured["url"] == "http://localhost:1234/v1/chat/completions"
    payload = captured["payload"]
    assert isinstance(payload, dict)
    assert payload["seed"] == 59
    assert payload["max_tokens"] == 2048
    assert payload["top_k"] == 20
    assert payload["presence_penalty"] == 1.5


def test_truncated_teacher_response_is_incomplete(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = load_sequence_teacher_config(CONFIG_DIR / "sequence_teacher.json")

    def fake_api_json(url: str, *, timeout: int, payload=None):
        return {
            "choices": [
                {
                    "message": {
                        "reasoning_content": "The draft contains \\boxed{4}.",
                        "content": "",
                    },
                    "finish_reason": "length",
                }
            ],
            "usage": {"completion_tokens": 2048},
        }

    monkeypatch.setattr(
        "llm_learning.minimind.distill_data.api_json",
        fake_api_json,
    )
    result = generate_teacher_response(
        "How many?",
        row_id=17,
        model_id="qwen/qwen3.5-9b",
        config=config,
    )
    assert result["response"] == ""
    assert result["finish_reason"] == "length"
    assert result["complete"] is False
