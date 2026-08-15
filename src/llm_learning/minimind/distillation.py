"""阶段 8 的 logit distillation 核心计算。"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class DistillationLosses:
    """保留 CE、KD 与混合 loss，便于分别观察。"""

    total: torch.Tensor
    ce: torch.Tensor | None
    kd: torch.Tensor | None
    supervised_tokens: int


def masked_cross_entropy(
    student_logits: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    """在 causal shift 后的有效 label 位置计算 CE。"""
    shifted_logits = student_logits[..., :-1, :].contiguous()
    shifted_labels = labels[..., 1:].contiguous()
    return F.cross_entropy(
        shifted_logits.reshape(-1, shifted_logits.size(-1)),
        shifted_labels.reshape(-1),
        ignore_index=-100,
    )


def masked_kl_divergence(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    labels: torch.Tensor,
    temperature: float,
    *,
    direction: str = "forward",
    scale_by_temperature: bool = True,
) -> torch.Tensor:
    """只在 causal shift 后的有效 label 位置计算指定方向的 KL。"""
    if student_logits.shape != teacher_logits.shape:
        raise ValueError("Teacher and Student logits must have the same shape")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if direction not in {"forward", "reverse"}:
        raise ValueError("direction must be 'forward' or 'reverse'")

    shifted_student = student_logits[..., :-1, :].contiguous()
    shifted_teacher = teacher_logits[..., :-1, :].contiguous()
    shifted_labels = labels[..., 1:].contiguous()
    valid = shifted_labels.ne(-100).reshape(-1)
    if not torch.any(valid):
        raise ValueError("batch contains no supervised tokens")

    vocab_size = shifted_student.size(-1)
    student_rows = shifted_student.reshape(-1, vocab_size)[valid]
    teacher_rows = shifted_teacher.reshape(-1, vocab_size)[valid]
    teacher_probabilities = F.softmax(
        teacher_rows.float() / temperature,
        dim=-1,
    )
    teacher_log_probabilities = F.log_softmax(
        teacher_rows.float() / temperature,
        dim=-1,
    )
    student_probabilities = F.softmax(
        student_rows.float() / temperature,
        dim=-1,
    )
    student_log_probabilities = F.log_softmax(
        student_rows.float() / temperature,
        dim=-1,
    )
    if direction == "forward":
        per_token = torch.sum(
            teacher_probabilities
            * (teacher_log_probabilities - student_log_probabilities),
            dim=-1,
        )
    else:
        per_token = torch.sum(
            student_probabilities
            * (student_log_probabilities - teacher_log_probabilities),
            dim=-1,
        )
    divergence = per_token.mean()
    return divergence * temperature**2 if scale_by_temperature else divergence


def distillation_losses(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor | None,
    labels: torch.Tensor,
    *,
    ce_weight: float,
    temperature: float,
    kl_direction: str = "forward",
) -> DistillationLosses:
    """计算 `ce_weight * CE + (1 - ce_weight) * KD`。"""
    if not 0 <= ce_weight <= 1:
        raise ValueError("ce_weight must be in [0, 1]")
    shifted_labels = labels[..., 1:].contiguous()
    ce = None if ce_weight == 0 else masked_cross_entropy(student_logits, labels)
    if ce_weight == 1:
        kd = None
    else:
        if teacher_logits is None:
            raise ValueError("teacher_logits are required when KD weight is positive")
        kd = masked_kl_divergence(
            student_logits,
            teacher_logits,
            labels,
            temperature,
            direction=kl_direction,
        )
    if ce is None:
        assert kd is not None
        total = kd
    elif kd is None:
        total = ce
    else:
        total = ce_weight * ce + (1 - ce_weight) * kd
    return DistillationLosses(
        total=total,
        ce=ce,
        kd=kd,
        supervised_tokens=int(shifted_labels.ne(-100).sum().item()),
    )
