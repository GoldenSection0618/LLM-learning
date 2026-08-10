"""MiniMind 阶段检查工具测试。"""

from types import SimpleNamespace

import torch
from torch import nn

from llm_learning.minimind.inspect import inspect_attention_variant


class FakeConfig:
    """提供检查工具所需的最小配置接口。"""

    def __init__(self, **kwargs):
        for name, value in kwargs.items():
            setattr(self, name, value)
        self.head_dim = self.hidden_size // self.num_attention_heads


class FakeAttention(nn.Module):
    """保留 Q/K/V 投影 shape 的最小 attention。"""

    def __init__(self, config: FakeConfig):
        super().__init__()
        self.q_proj = nn.Linear(
            config.hidden_size,
            config.num_attention_heads * config.head_dim,
            bias=False,
        )
        self.k_proj = nn.Linear(
            config.hidden_size,
            config.num_key_value_heads * config.head_dim,
            bias=False,
        )
        self.v_proj = nn.Linear(
            config.hidden_size,
            config.num_key_value_heads * config.head_dim,
            bias=False,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        self.q_proj(hidden_states)
        self.k_proj(hidden_states)
        self.v_proj(hidden_states)
        return hidden_states


class FakeLayer(nn.Module):
    """提供与官方 block 一致的 hook 输出接口。"""

    def __init__(self, config: FakeConfig):
        super().__init__()
        self.self_attn = FakeAttention(config)

    def forward(self, hidden_states: torch.Tensor):
        return self.self_attn(hidden_states), None


class FakeModelBody(nn.Module):
    """提供 embedding 与 block 列表。"""

    def __init__(self, config: FakeConfig):
        super().__init__()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            [FakeLayer(config) for _ in range(config.num_hidden_layers)]
        )


class FakeCausalLM(nn.Module):
    """模拟官方模型的 logits 与 cache 返回值。"""

    def __init__(self, config: FakeConfig):
        super().__init__()
        self.config = config
        self.model = FakeModelBody(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(
        self,
        input_ids: torch.Tensor,
        past_key_values=None,
        use_cache: bool = False,
    ):
        hidden_states = self.model.embed_tokens(input_ids)
        for layer in self.model.layers:
            hidden_states, _ = layer(hidden_states)
        logits = self.lm_head(hidden_states)

        past_length = 0
        if past_key_values is not None:
            past_length = past_key_values[0][0].shape[1]
        total_length = past_length + input_ids.shape[1]
        if use_cache:
            cache = [
                (
                    torch.zeros(
                        input_ids.shape[0],
                        total_length,
                        self.config.num_key_value_heads,
                        self.config.head_dim,
                    ),
                    torch.zeros(
                        input_ids.shape[0],
                        total_length,
                        self.config.num_key_value_heads,
                        self.config.head_dim,
                    ),
                )
                for _ in self.model.layers
            ]
        else:
            cache = [None for _ in self.model.layers]
        return SimpleNamespace(logits=logits, past_key_values=cache)


FAKE_MODULE = SimpleNamespace(
    MiniMindConfig=FakeConfig,
    MiniMindForCausalLM=FakeCausalLM,
)


def test_inspection_uses_observed_projection_and_layer_shapes():
    """检查结果应来自模型实际产生的 Tensor shape。"""
    result = inspect_attention_variant(FAKE_MODULE, num_key_value_heads=2, seed=2026)

    assert result["input_shape"] == [2, 8]
    assert result["device"] == "cpu"
    assert result["device_name"] == "CPU"
    assert result["dtype"] == "torch.float32"
    assert result["embedding_shape"] == [2, 8, 256]
    assert result["query_shape_before_transpose"] == [2, 8, 4, 64]
    assert result["key_shape_before_repeat"] == [2, 8, 2, 64]
    assert result["value_shape_before_repeat"] == [2, 8, 2, 64]
    assert result["layer_output_shapes"] == {
        f"layer_{index}": [2, 8, 256] for index in range(4)
    }
    assert result["prefix_key_shape"] == [2, 7, 2, 64]
    assert result["incremental_key_shape"] == [2, 8, 2, 64]
    assert result["cache_vs_full_last_logit_max_abs_diff"] < 2e-6
