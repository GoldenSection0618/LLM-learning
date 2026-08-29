"""阶段 9 运行清单测试。"""

from __future__ import annotations

import pytest

from llm_learning.experiment_record import (
    compare_reproduction_results,
    scalar_series,
    validate_manifest,
)


def test_manifest_requires_fixed_fields() -> None:
    with pytest.raises(ValueError, match="missing fields"):
        validate_manifest({"experiment": "incomplete"})


def test_scalar_series_uses_optimizer_step() -> None:
    records = [{"optimizer_step": 3, "loss": 1.25, "kind": "train"}]
    assert scalar_series(records) == [(3, "loss", 1.25)]


def test_reproduction_comparison_separates_numeric_and_exact_fields() -> None:
    first = {"loss": 1.0, "greedy_token_ids": [1, 2, 3]}
    second = {"loss": 1.0000004, "greedy_token_ids": [1, 2, 3]}
    result = compare_reproduction_results(
        first,
        second,
        numeric_fields=["loss"],
        exact_fields=["greedy_token_ids"],
        absolute_tolerance=1e-6,
    )
    assert result["reproduced"] is True
