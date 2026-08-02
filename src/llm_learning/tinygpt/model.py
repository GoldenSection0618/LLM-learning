from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from .config import ModelConfig

ShapeTrace = dict[str, tuple[int, ...]]


class CausalSelfAttention(nn.Module):
    """让每个位置只关注自己和它左侧的 token。"""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.num_heads = config.num_heads
        self.head_dim = config.head_dim
        self.d_model = config.d_model

        # 一次线性变换同时计算 Q、K、V，最后再沿特征维切成三份。
        self.qkv_projection = nn.Linear(
            config.d_model,
            3 * config.d_model,
        )
        self.output_projection = nn.Linear(
            config.d_model,
            config.d_model,
        )
        self.attention_dropout = nn.Dropout(config.dropout)
        self.output_dropout = nn.Dropout(config.dropout)

        # 固定长度样本没有 padding，此处只需要 causal mask。
        # 下三角为 True，表示当前位置可以读取对应的 key。
        causal_mask = torch.tril(
            torch.ones(
                config.block_size,
                config.block_size,
                dtype=torch.bool,
            )
        )
        self.register_buffer(
            "causal_mask",
            causal_mask,
            persistent=False,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        trace: ShapeTrace | None = None,
        trace_prefix: str = "attention",
    ) -> torch.Tensor:
        batch_size, sequence_length, _ = hidden_states.shape

        qkv = self.qkv_projection(hidden_states)
        queries, keys, values = qkv.chunk(3, dim=-1)

        # [B, T, D] 拆成 H 个头，并把头维移到序列维前面。
        queries = queries.view(
            batch_size,
            sequence_length,
            self.num_heads,
            self.head_dim,
        ).transpose(1, 2)
        keys = keys.view(
            batch_size,
            sequence_length,
            self.num_heads,
            self.head_dim,
        ).transpose(1, 2)
        values = values.view(
            batch_size,
            sequence_length,
            self.num_heads,
            self.head_dim,
        ).transpose(1, 2)

        # 每个 query 与全部 key 做点积，得到 [B, H, T, T]。
        attention_scores = queries @ keys.transpose(-2, -1)
        attention_scores = attention_scores / math.sqrt(self.head_dim)

        allowed_positions = self.causal_mask[
            :sequence_length,
            :sequence_length,
        ]
        attention_scores = attention_scores.masked_fill(
            ~allowed_positions,
            float("-inf"),
        )
        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_weights = self.attention_dropout(attention_weights)

        # 每行注意力权重对 V 加权求和，得到每个位置的新表示。
        context = attention_weights @ values
        context = context.transpose(1, 2).contiguous()
        context = context.view(
            batch_size,
            sequence_length,
            self.d_model,
        )
        output = self.output_projection(context)
        output = self.output_dropout(output)

        if trace is not None:
            trace[f"{trace_prefix}.queries"] = tuple(queries.shape)
            trace[f"{trace_prefix}.keys"] = tuple(keys.shape)
            trace[f"{trace_prefix}.values"] = tuple(values.shape)
            trace[f"{trace_prefix}.scores"] = tuple(attention_scores.shape)
            trace[f"{trace_prefix}.weights"] = tuple(attention_weights.shape)
            trace[f"{trace_prefix}.output"] = tuple(output.shape)

        return output


class FeedForward(nn.Module):
    """在每个序列位置上独立运行的两层 MLP。"""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        hidden_size = 4 * config.d_model
        self.layers = nn.Sequential(
            nn.Linear(config.d_model, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, config.d_model),
            nn.Dropout(config.dropout),
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.layers(hidden_states)


class TransformerBlock(nn.Module):
    """一个带 Pre-LN 和两条残差连接的 Transformer block。"""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.attention_norm = nn.LayerNorm(config.d_model)
        self.attention = CausalSelfAttention(config)
        self.mlp_norm = nn.LayerNorm(config.d_model)
        self.mlp = FeedForward(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        trace: ShapeTrace | None = None,
        trace_prefix: str = "block",
    ) -> torch.Tensor:
        # 残差连接保留进入子层前的信息，并叠加子层学到的变化。
        attention_input = self.attention_norm(hidden_states)
        hidden_states = hidden_states + self.attention(
            attention_input,
            trace=trace,
            trace_prefix=f"{trace_prefix}.attention",
        )

        mlp_input = self.mlp_norm(hidden_states)
        hidden_states = hidden_states + self.mlp(mlp_input)

        if trace is not None:
            trace[f"{trace_prefix}.hidden_states"] = tuple(
                hidden_states.shape
            )

        return hidden_states


class TinyGPT(nn.Module):
    """使用绝对位置 embedding 的最小 Decoder-only Transformer。"""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config

        self.token_embedding = nn.Embedding(
            config.vocab_size,
            config.d_model,
        )
        self.position_embedding = nn.Embedding(
            config.block_size,
            config.d_model,
        )
        self.embedding_dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(config)
                for _ in range(config.num_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(
            config.d_model,
            config.vocab_size,
            bias=False,
        )

        self.apply(self._initialize_weights)

    @staticmethod
    def _initialize_weights(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def _forward(
        self,
        input_ids: torch.Tensor,
        targets: torch.Tensor | None,
        trace: ShapeTrace | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [B, T]")

        _, sequence_length = input_ids.shape
        if sequence_length > self.config.block_size:
            raise ValueError(
                "sequence length exceeds the configured block_size"
            )
        if targets is not None and targets.shape != input_ids.shape:
            raise ValueError("targets must have the same shape as input_ids")

        positions = torch.arange(
            sequence_length,
            device=input_ids.device,
        )
        token_embeddings = self.token_embedding(input_ids)
        position_embeddings = self.position_embedding(positions)
        hidden_states = token_embeddings + position_embeddings
        hidden_states = self.embedding_dropout(hidden_states)

        if trace is not None:
            trace["input_ids"] = tuple(input_ids.shape)
            trace["token_embedding"] = tuple(token_embeddings.shape)
            trace["position_embedding"] = tuple(position_embeddings.shape)
            trace["embedding"] = tuple(hidden_states.shape)

        for index, block in enumerate(self.blocks):
            hidden_states = block(
                hidden_states,
                trace=trace,
                trace_prefix=f"block_{index}",
            )

        hidden_states = self.final_norm(hidden_states)
        logits = self.lm_head(hidden_states)
        loss = None

        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, self.config.vocab_size),
                targets.reshape(-1),
            )

        if trace is not None:
            trace["hidden_states"] = tuple(hidden_states.shape)
            trace["logits"] = tuple(logits.shape)

        return logits, loss

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        return self._forward(
            input_ids=input_ids,
            targets=targets,
            trace=None,
        )

    def forward_with_trace(
        self,
        input_ids: torch.Tensor,
        targets: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None, ShapeTrace]:
        """执行同一条 forward，并额外返回关键张量的 shape。"""
        trace: ShapeTrace = {}
        logits, loss = self._forward(
            input_ids=input_ids,
            targets=targets,
            trace=trace,
        )
        return logits, loss, trace

    def count_parameters(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        sample: bool = True,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        if temperature <= 0:
            raise ValueError("temperature must be positive")

        generated = input_ids
        was_training = self.training
        self.eval()

        for _ in range(max_new_tokens):
            # 模型只能接收 block_size 范围内的最近上下文。
            model_input = generated[:, -self.config.block_size :]
            logits, _ = self(model_input)
            next_token_logits = logits[:, -1, :] / temperature

            if sample:
                probabilities = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(
                    probabilities,
                    num_samples=1,
                    generator=generator,
                )
            else:
                next_token = torch.argmax(
                    next_token_logits,
                    dim=-1,
                    keepdim=True,
                )

            generated = torch.cat((generated, next_token), dim=1)

        self.train(was_training)
        return generated
