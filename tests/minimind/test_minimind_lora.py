from copy import deepcopy
import json
from pathlib import Path

import pytest
import torch
from torch import nn

from llm_learning.minimind.checkpoint import load_checkpoint, save_checkpoint
from llm_learning.minimind.lora import (
    adapter_state_dict,
    apply_lora,
    count_parameters,
    freeze_non_lora_parameters,
    load_adapter_checkpoint,
    load_adapter_state_dict,
    lora_target_names,
    merge_lora_,
    save_adapter_checkpoint,
)
from llm_learning.minimind.lora_config import load_lora_config
from llm_learning.minimind import lora_evaluate
from llm_learning.minimind.lora_train import (
    build_lora_training_config,
    prepare_lora_model,
)
from llm_learning.minimind.inspect import load_official_model_module


class TinyAttention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.q_proj = nn.Linear(4, 4, bias=False)
        self.k_proj = nn.Linear(4, 2, bias=False)
        self.o_proj = nn.Linear(4, 4, bias=False)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.o_proj(self.q_proj(inputs))


def make_model() -> nn.Module:
    torch.manual_seed(7)
    return nn.Sequential(TinyAttention())


def test_rank16_config_is_fixed() -> None:
    config = load_lora_config(
        Path("docs/stages/06_lora/configs/rank16.json")
    )
    assert config.rank == 16
    assert config.target_modules == ("q_proj", "o_proj")
    assert config.stop_after_step == 56480

    training_config = build_lora_training_config(config)
    assert training_config.epochs == 1
    assert training_config.max_optimizer_steps == 56480
    assert training_config.learning_rate == 1e-4
    assert training_config.initial_weights_path == config.base_weights_path


def test_apply_lora_targets_only_requested_linear_layers() -> None:
    model = make_model()
    original_output = model(torch.randn(2, 4))
    names = apply_lora(model, rank=2, target_module_names=("q_proj", "o_proj"))
    adapted_output = model(torch.randn(2, 4))

    assert names == ("0.q_proj", "0.o_proj")
    assert lora_target_names(model) == names
    assert not hasattr(model[0].k_proj, "lora")
    assert original_output.shape == adapted_output.shape == (2, 4)


def test_zero_initialized_b_preserves_initial_output() -> None:
    model = make_model()
    inputs = torch.randn(3, 4)
    expected = model(inputs).detach().clone()
    apply_lora(model, rank=2, target_module_names=("q_proj", "o_proj"))
    actual = model(inputs).detach()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_optimizer_step_changes_only_lora_parameters() -> None:
    model = make_model()
    apply_lora(model, rank=2, target_module_names=("q_proj", "o_proj"))
    freeze_non_lora_parameters(model)
    counts = count_parameters(model)
    base_before = {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
        if ".lora." not in name
    }
    adapter_before = adapter_state_dict(model)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=0.1,
    )

    loss = model(torch.randn(3, 4)).square().mean()
    loss.backward()
    optimizer.step()

    assert counts["trainable"] == counts["lora"] == 32
    for name, parameter in model.named_parameters():
        if name in base_before:
            torch.testing.assert_close(parameter, base_before[name], rtol=0, atol=0)
    assert any(
        not torch.equal(value, adapter_before[name])
        for name, value in adapter_state_dict(model).items()
    )


def test_adapter_round_trip_and_merge_preserve_output() -> None:
    base_model = make_model()
    base_state = deepcopy(base_model.state_dict())
    apply_lora(base_model, rank=2, target_module_names=("q_proj", "o_proj"))
    with torch.no_grad():
        for name, parameter in base_model.named_parameters():
            if name.endswith("lora.B.weight"):
                parameter.normal_(mean=0.0, std=0.1)
    state = adapter_state_dict(base_model)
    inputs = torch.randn(3, 4)
    expected = base_model(inputs).detach().clone()

    reloaded = make_model()
    reloaded.load_state_dict(base_state)
    apply_lora(reloaded, rank=2, target_module_names=("q_proj", "o_proj"))
    load_adapter_state_dict(reloaded, state)
    torch.testing.assert_close(reloaded(inputs), expected)

    merged_names = merge_lora_(reloaded)
    assert merged_names == ("0.q_proj", "0.o_proj")
    assert lora_target_names(reloaded) == ()
    torch.testing.assert_close(reloaded(inputs), expected, rtol=1e-5, atol=1e-6)


def test_adapter_checkpoint_validates_metadata(tmp_path: Path) -> None:
    source = make_model()
    apply_lora(source, rank=2, target_module_names=("q_proj", "o_proj"))
    metadata = {
        "base_weights_sha256": "a" * 64,
        "rank": 2,
        "target_modules": ["0.q_proj", "0.o_proj"],
    }
    path = tmp_path / "adapter.pth"
    save_adapter_checkpoint(path, source, metadata)

    target = make_model()
    apply_lora(target, rank=2, target_module_names=("q_proj", "o_proj"))
    loaded_metadata = load_adapter_checkpoint(path, target, metadata)

    assert loaded_metadata == metadata
    for name, value in adapter_state_dict(source).items():
        torch.testing.assert_close(
            adapter_state_dict(target)[name],
            value,
            rtol=1e-3,
            atol=1e-5,
        )
    with pytest.raises(ValueError, match="metadata does not match"):
        load_adapter_checkpoint(path, target, {**metadata, "rank": 4})


def test_real_minimind_injection_selects_q_and_o_projection() -> None:
    module = load_official_model_module(Path("third_party/minimind"))
    model_config = module.MiniMindConfig(
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        vocab_size=128,
        max_position_embeddings=128,
        flash_attn=True,
        use_moe=False,
        dropout=0.0,
    )
    model = module.MiniMindForCausalLM(model_config)
    config = load_lora_config(
        Path("docs/stages/06_lora/configs/rank16.json")
    )

    result = prepare_lora_model(model, config)

    assert result["target_module_count"] == 4
    assert result["trainable_parameter_count"] == 8192
    assert all(
        name.endswith(("q_proj", "o_proj"))
        for name in result["target_modules"]
    )


def test_lora_resume_restores_optimizer_and_continues_identically(
    tmp_path: Path,
) -> None:
    def adapted_model() -> nn.Module:
        model = make_model()
        apply_lora(model, rank=2, target_module_names=("q_proj", "o_proj"))
        freeze_non_lora_parameters(model)
        return model

    def optimizer_for(model: nn.Module) -> torch.optim.Optimizer:
        return torch.optim.AdamW(
            [parameter for parameter in model.parameters() if parameter.requires_grad],
            lr=0.01,
        )

    def step(model: nn.Module, optimizer: torch.optim.Optimizer) -> None:
        optimizer.zero_grad(set_to_none=True)
        model(torch.ones(3, 4)).square().mean().backward()
        optimizer.step()

    source = adapted_model()
    source_optimizer = optimizer_for(source)
    source_scaler = torch.GradScaler("cuda", enabled=False)
    step(source, source_optimizer)
    path = tmp_path / "latest.pt"
    training_state = {
        "epoch": 0,
        "next_batch": 1,
        "optimizer_step": 1,
        "trained_tokens": 12,
        "microbatches_in_update": 0,
    }
    config = {"profile": "test", "training_extension": {"rank": 2}}
    tokenizer = {"name": "test"}
    manifest = {"split_sha256": "split", "raw_sha256": "raw"}
    save_checkpoint(
        path,
        source,
        source_optimizer,
        source_scaler,
        training_state,
        config,
        tokenizer,
        manifest,
        [],
    )

    reloaded = adapted_model()
    reloaded_optimizer = optimizer_for(reloaded)
    reloaded_scaler = torch.GradScaler("cuda", enabled=False)
    checkpoint = load_checkpoint(
        path,
        reloaded,
        reloaded_optimizer,
        reloaded_scaler,
        config,
        tokenizer,
        manifest,
        torch.device("cpu"),
    )
    assert checkpoint["training_state"]["optimizer_step"] == 1

    step(source, source_optimizer)
    step(reloaded, reloaded_optimizer)
    for name, value in adapter_state_dict(source).items():
        torch.testing.assert_close(adapter_state_dict(reloaded)[name], value)


def stage5_generation_fixture() -> dict:
    prompts = ["问题一", "问题二"]

    def checkpoint(name: str) -> dict:
        return {
            "checkpoint": name,
            "modes": {
                mode: [{"prompt": prompt, "text": f"{name}-{mode}"} for prompt in prompts]
                for mode in ["greedy", "sampling"]
            },
        }

    return {
        "inspection_prompts": [{"row_id": 1, "prompt": "问题二"}],
        "generation_config": {
            "prompts": prompts,
            "greedy": {},
            "sampling": {},
        },
        "pretrain": checkpoint("pretrain"),
        "sft": checkpoint("full_sft"),
    }


def test_three_way_generation_reuses_stage5_and_runs_only_lora(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stage5_path = tmp_path / "stage5.json"
    stage5_path.write_text(
        json.dumps(stage5_generation_fixture(), ensure_ascii=False),
        encoding="utf-8",
    )
    lora_weights = tmp_path / "merged.pth"
    lora_weights.touch()
    output = tmp_path / "three_way.json"
    observed: dict = {}

    def fake_evaluate(config, paths, generation_config, prompt_renderer=None):
        observed["paths"] = paths
        observed["prompts"] = generation_config["prompts"]
        observed["renderer"] = prompt_renderer
        return {
            "checkpoints": [
                {
                    "checkpoint": str(paths[0]),
                    "modes": {"greedy": [], "sampling": []},
                }
            ]
        }

    monkeypatch.setattr(lora_evaluate, "evaluate_checkpoints", fake_evaluate)
    result = lora_evaluate.run_three_way_generation(
        Path("docs/stages/06_lora/configs/rank16.json"),
        stage5_path,
        lora_weights,
        output,
    )

    assert observed["paths"] == [lora_weights]
    assert observed["prompts"] == ["问题一", "问题二"]
    assert observed["renderer"] is lora_evaluate.render_chat_prompt
    assert result["full_sft"]["checkpoint"] == "full_sft"
    assert result["lora"]["checkpoint"] == str(lora_weights)
    assert json.loads(output.read_text(encoding="utf-8")) == result


def test_stage5_generation_rejects_prompt_mismatch(tmp_path: Path) -> None:
    result = stage5_generation_fixture()
    result["sft"]["modes"]["greedy"][0]["prompt"] = "不同问题"
    path = tmp_path / "stage5.json"
    path.write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")

    with pytest.raises(ValueError, match="prompts do not match"):
        lora_evaluate.load_stage5_generation(path)


def test_missing_target_and_wrong_adapter_shape_are_rejected() -> None:
    model = make_model()
    with pytest.raises(ValueError, match="were not found"):
        apply_lora(model, rank=2, target_module_names=("missing_proj",))

    apply_lora(model, rank=2, target_module_names=("q_proj", "o_proj"))
    state = adapter_state_dict(model)
    first_name = next(iter(state))
    state[first_name] = torch.zeros(1)
    with pytest.raises(ValueError, match="shape does not match"):
        load_adapter_state_dict(model, state)
