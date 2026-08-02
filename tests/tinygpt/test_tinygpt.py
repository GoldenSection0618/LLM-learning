from __future__ import annotations

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from llm_learning.tinygpt.config import ModelConfig
from llm_learning.tinygpt.model import (
    CausalSelfAttention,
    TinyGPT,
)
from llm_learning.tinygpt.train import evaluate


def make_small_config(
    vocab_size: int = 7,
    block_size: int = 8,
    d_model: int = 24,
    num_heads: int = 4,
    num_layers: int = 2,
    dropout: float = 0.0,
) -> ModelConfig:
    return ModelConfig(
        vocab_size=vocab_size,
        block_size=block_size,
        d_model=d_model,
        num_heads=num_heads,
        num_layers=num_layers,
        dropout=dropout,
    )


def test_model_config_rejects_incompatible_head_count() -> None:
    try:
        make_small_config(d_model=22)
    except ValueError:
        pass
    else:
        raise AssertionError("d_model must divide evenly across heads")


def test_attention_mask_is_lower_triangular() -> None:
    attention = CausalSelfAttention(make_small_config())
    expected = torch.tril(torch.ones(8, 8, dtype=torch.bool))
    assert torch.equal(attention.causal_mask, expected)


def test_forward_shapes_and_trace() -> None:
    torch.manual_seed(1)
    model = TinyGPT(make_small_config())
    input_ids = torch.tensor(
        [
            [0, 1, 2, 3, 4],
            [4, 3, 2, 1, 0],
        ]
    )
    targets = torch.tensor(
        [
            [1, 2, 3, 4, 5],
            [3, 2, 1, 0, 6],
        ]
    )

    logits, loss, trace = model.forward_with_trace(input_ids, targets)

    assert logits.shape == (2, 5, 7)
    assert loss is not None and loss.ndim == 0
    assert trace["input_ids"] == (2, 5)
    assert trace["embedding"] == (2, 5, 24)
    assert trace["block_0.attention.queries"] == (2, 4, 5, 6)
    assert trace["block_0.attention.scores"] == (2, 4, 5, 5)
    assert trace["hidden_states"] == (2, 5, 24)
    assert trace["logits"] == (2, 5, 7)


def test_future_tokens_cannot_change_earlier_logits() -> None:
    torch.manual_seed(2)
    model = TinyGPT(make_small_config())
    model.eval()

    original = torch.tensor([[0, 1, 2, 3, 4]])
    future_changed = torch.tensor([[0, 1, 2, 6, 5]])
    original_logits, _ = model(original)
    changed_logits, _ = model(future_changed)

    assert torch.allclose(
        original_logits[:, :3],
        changed_logits[:, :3],
    )
    assert not torch.allclose(
        original_logits[:, 3:],
        changed_logits[:, 3:],
    )


def test_generate_crops_context_to_block_size() -> None:
    torch.manual_seed(3)
    model = TinyGPT(make_small_config(block_size=4))
    prompt = torch.tensor([[0, 1, 2, 3]])
    generated = model.generate(
        prompt,
        max_new_tokens=5,
        sample=False,
    )
    assert generated.shape == (1, 9)


def test_default_model_is_within_stage_parameter_budget() -> None:
    config = ModelConfig(vocab_size=65)
    model = TinyGPT(config)
    assert 1_000_000 <= model.count_parameters() <= 10_000_000


def test_one_batch_overfit_reduces_loss() -> None:
    torch.manual_seed(4)
    model = TinyGPT(make_small_config(vocab_size=3))
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-2)
    inputs = torch.tensor(
        [
            [0, 1, 2, 0, 1, 2],
            [1, 2, 0, 1, 2, 0],
        ]
    )
    targets = torch.tensor(
        [
            [1, 2, 0, 1, 2, 0],
            [2, 0, 1, 2, 0, 1],
        ]
    )

    _, initial_loss = model(inputs, targets)
    assert initial_loss is not None

    for _ in range(40):
        optimizer.zero_grad(set_to_none=True)
        _, loss = model(inputs, targets)
        assert loss is not None
        loss.backward()
        optimizer.step()

    _, final_loss = model(inputs, targets)
    assert final_loss is not None
    assert final_loss.item() < initial_loss.item()


def test_evaluation_weights_batches_by_token_count() -> None:
    class MeanTargetModel(nn.Module):
        def forward(
            self,
            inputs: torch.Tensor,
            targets: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            return inputs, targets.float().mean()

    inputs = torch.zeros((3, 2), dtype=torch.long)
    targets = torch.tensor(
        [
            [1, 1],
            [1, 1],
            [5, 5],
        ]
    )
    loader = DataLoader(
        TensorDataset(inputs, targets),
        batch_size=2,
        shuffle=False,
    )
    model = MeanTargetModel()

    loss = evaluate(
        model=model,
        loader=loader,
        batches=len(loader),
        device=torch.device("cpu"),
    )

    assert abs(loss - 7 / 3) < 1e-6
