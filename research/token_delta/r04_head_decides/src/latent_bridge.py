"""Trainable map from one frozen head's hidden vector to frozen writer soft tokens.

The writer receives only the prefix + the constant assistant-start marker. No
original question, target answer, length flag, format flag or reasoning trace.
This module is independent of Hugging Face so tests run on a CPU-only install.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class LatentBridge(nn.Module):
    def __init__(self, head_size: int = 2560, writer_size: int = 1024,
                 plan_tokens: int = 4, inner_size: int = 512) -> None:
        super().__init__()
        if min(head_size, writer_size, plan_tokens, inner_size) <= 0:
            raise ValueError("all bridge dimensions must be positive")
        self.head_size = head_size
        self.writer_size = writer_size
        self.plan_tokens = plan_tokens
        self.project = nn.Sequential(
            nn.LayerNorm(head_size),
            nn.Linear(head_size, inner_size),
            nn.GELU(),
            nn.Linear(inner_size, writer_size * plan_tokens),
        )
        self.slot_bias = nn.Parameter(torch.zeros(plan_tokens, writer_size))

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        if hidden.ndim != 2 or hidden.size(-1) != self.head_size:
            raise ValueError(f"expected [batch, {self.head_size}] teacher states")
        bsz = hidden.size(0)
        raw = self.project(hidden.float()).view(bsz, self.plan_tokens, self.writer_size)
        return 0.1 * raw + self.slot_bias.unsqueeze(0)


def assemble_writer_embeddings(soft_plan: torch.Tensor,
                               start_ids: torch.Tensor,
                               answer_ids: torch.Tensor,
                               token_embedding: nn.Module) -> torch.Tensor:
    """Embed constant assistant-start and teacher SUPERVISION ONLY (no question)."""
    if soft_plan.ndim != 3:
        raise ValueError("soft_plan must be [batch, slots, writer_hidden]")
    if start_ids.ndim != 2 or answer_ids.ndim != 2:
        raise ValueError("start_ids and answer_ids must be [batch, tokens]")
    if start_ids.size(0) != soft_plan.size(0) or answer_ids.size(0) != soft_plan.size(0):
        raise ValueError("batch dimensions must match")
    if start_ids.size(1) < 1 or answer_ids.size(1) < 1:
        raise ValueError("must have at least one start and target token")
    start_emb = token_embedding(start_ids)
    target_emb = token_embedding(answer_ids)
    return torch.cat([soft_plan.to(start_emb.dtype), start_emb, target_emb], dim=1)


def answer_cross_entropy(logits: torch.Tensor, answer_ids: torch.Tensor,
                         plan_tokens: int, start_tokens: int) -> torch.Tensor:
    """Causal alignment: answer token #0 is predicted at (plan+header-1).

    Logits at the last answer embedding are intentionally discarded, as they
    predict the *next* token, not a token within the answer supervision.
    """
    if logits.ndim != 3 or answer_ids.ndim != 2:
        raise ValueError("invalid logits or target rank")
    if plan_tokens < 1 or start_tokens < 1:
        raise ValueError("prefix/header must contain tokens")
    start = plan_tokens + start_tokens - 1
    end = start + answer_ids.size(1)
    if logits.size(1) < end:
        raise ValueError("logits do not span all shifted target positions")
    selected = logits[:, start:end, :].float()
    if selected.size(1) != answer_ids.size(1):
        raise AssertionError("causal shift mismatch")
    return F.cross_entropy(selected.reshape(-1, selected.size(-1)),
                           answer_ids.reshape(-1), reduction="mean")