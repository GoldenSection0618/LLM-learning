from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class BigramLanguageModel(nn.Module):
    """只根据当前 token 预测下一个 token。"""

    def __init__(self, vocab_size: int) -> None:
        super().__init__()
        self.vocab_size = vocab_size

        # 每个 token ID 直接查出下一 token 的 V 个预测分数。
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        # input_ids [B, T] 经过查表得到 logits [B, T, V]。
        logits = self.token_embedding_table(input_ids)
        loss = None

        if targets is not None:
            # 把 B、T 合并，让交叉熵同时计算所有位置并取平均。
            loss = F.cross_entropy(
                logits.reshape(-1, self.vocab_size),
                targets.reshape(-1),
            )
        return logits, loss

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

        for _ in range(max_new_tokens):
            # Bigram 只依赖最后一个 token，历史序列用于保存生成结果。
            logits, _ = self(generated[:, -1:])
            next_token_logits = logits[:, -1, :] / temperature
            if sample:
                # 在词表维度归一化，再按概率抽取下一个 token。
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

        return generated
