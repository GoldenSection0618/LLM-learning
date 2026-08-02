from __future__ import annotations

from pathlib import Path

import pytest
import torch

from llm_learning.bigram import data as data_module
from llm_learning.bigram.checkpoint import load_checkpoint, save_checkpoint
from llm_learning.bigram.data import (
    NextTokenDataset,
    split_token_ids,
)
from llm_learning.bigram.model import BigramLanguageModel
from llm_learning.bigram.tokenizer import CharacterTokenizer
from llm_learning.bigram.train import make_evaluation_loader


def test_valid_existing_dataset_is_reused(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "input.txt"
    path.write_bytes(b"valid data")
    monkeypatch.setattr(
        data_module,
        "sha256_file",
        lambda _: data_module.TINY_SHAKESPEARE_SHA256,
    )

    def unexpected_download(*_: object) -> None:
        raise AssertionError("A valid existing dataset must not be downloaded")

    monkeypatch.setattr(
        data_module.urllib.request,
        "urlretrieve",
        unexpected_download,
    )

    assert data_module.download_tiny_shakespeare(path) == path
    assert path.read_bytes() == b"valid data"


def test_invalid_existing_dataset_is_replaced(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "input.txt"
    path.write_bytes(b"damaged data")
    prefix_only_digest = (
        data_module.TINY_SHAKESPEARE_SHA256[:7] + "0" * 57
    )

    def fake_sha256(candidate: Path) -> str:
        if candidate.suffix == ".part":
            return data_module.TINY_SHAKESPEARE_SHA256
        return prefix_only_digest

    def fake_download(_: str, destination: Path) -> None:
        destination.write_bytes(b"replacement data")

    monkeypatch.setattr(data_module, "sha256_file", fake_sha256)
    monkeypatch.setattr(
        data_module.urllib.request,
        "urlretrieve",
        fake_download,
    )

    data_module.download_tiny_shakespeare(path)

    assert path.read_bytes() == b"replacement data"
    assert not path.with_suffix(".txt.part").exists()


def test_failed_replacement_preserves_existing_dataset(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "input.txt"
    path.write_bytes(b"damaged data")

    def fake_download(_: str, destination: Path) -> None:
        destination.write_bytes(b"invalid replacement")

    monkeypatch.setattr(data_module, "sha256_file", lambda _: "0" * 64)
    monkeypatch.setattr(
        data_module.urllib.request,
        "urlretrieve",
        fake_download,
    )

    with pytest.raises(ValueError, match="Downloaded dataset"):
        data_module.download_tiny_shakespeare(path)

    assert path.read_bytes() == b"damaged data"
    assert not path.with_suffix(".txt.part").exists()


def test_character_tokenizer_round_trip(tmp_path: Path) -> None:
    tokenizer = CharacterTokenizer.from_text("cabca\n")
    encoded = tokenizer.encode("cab\n")
    assert tokenizer.decode(encoded) == "cab\n"

    path = tmp_path / "tokenizer.json"
    tokenizer.save(path)
    assert CharacterTokenizer.load(path) == tokenizer

    try:
        tokenizer.decode([-1])
    except ValueError:
        pass
    else:
        raise AssertionError("A negative token ID must be rejected")


def test_ordered_split() -> None:
    train, validation, split_index = split_token_ids(list(range(10)), 0.9)
    assert split_index == 9
    assert train == list(range(9))
    assert validation == [9]


def test_next_token_dataset_shifts_one_token() -> None:
    dataset = NextTokenDataset([0, 1, 2, 3, 4], block_size=3)
    inputs, targets = dataset[1]
    assert inputs.tolist() == [1, 2, 3]
    assert targets.tolist() == [2, 3, 4]


def test_bigram_forward_and_generation() -> None:
    model = BigramLanguageModel(vocab_size=5)
    inputs = torch.tensor([[0, 1, 2]])
    targets = torch.tensor([[1, 2, 3]])
    logits, loss = model(inputs, targets)
    assert logits.shape == (1, 3, 5)
    assert loss is not None and loss.ndim == 0

    generated = model.generate(
        inputs[:, :1],
        max_new_tokens=4,
        sample=False,
    )
    assert generated.shape == (1, 5)


def test_evaluation_loader_reuses_fixed_samples() -> None:
    dataset = NextTokenDataset(
        list(range(20)),
        block_size=2,
    )
    loader = make_evaluation_loader(
        dataset=dataset,
        batch_size=2,
        batches=2,
        seed=1337,
        device=torch.device("cpu"),
    )

    first_pass = list(loader)
    second_pass = list(loader)

    assert len(first_pass) == 2
    assert len(second_pass) == 2

    for first_batch, second_batch in zip(first_pass, second_pass):
        first_inputs, first_targets = first_batch
        second_inputs, second_targets = second_batch

        assert torch.equal(first_inputs, second_inputs)
        assert torch.equal(first_targets, second_targets)


def test_checkpoint_round_trip(tmp_path: Path) -> None:
    torch.manual_seed(1)
    model = BigramLanguageModel(vocab_size=4)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
    path = tmp_path / "checkpoint.pt"
    original = model.token_embedding_table.weight.detach().clone()

    save_checkpoint(
        path,
        model,
        optimizer,
        step=7,
        config={"seed": 1},
        tokenizer={"characters": ["a", "b", "c", "d"]},
        history=[
            {
                "step": 7,
                "train_loss": 1.0,
                "validation_loss": 1.1,
            }
        ],
    )
    with torch.no_grad():
        model.token_embedding_table.weight.zero_()

    checkpoint = load_checkpoint(
        path=path,
        model=model,
        optimizer=optimizer,
        device=torch.device("cpu"),
    )
    assert checkpoint["step"] == 7
    assert torch.equal(model.token_embedding_table.weight, original)
