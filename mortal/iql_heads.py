"""Heads for Implicit Q-Learning built on top of the shared Brain encoder."""

from __future__ import annotations

import torch
from torch import nn
from .libriichi.consts import ACTION_SPACE


class IQLValueHead(nn.Module):
    """Value head mirroring the ACH policy head geometry with scalar output."""

    def __init__(self, *, hidden_size: int = 1024):
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.act = nn.ReLU()
        self.head = nn.Linear(256, 1)
        nn.init.constant_(self.head.bias, 0.0)

    def forward(self, phi: torch.Tensor) -> torch.Tensor:
        x = self.act(self.bn1(self.fc1(phi)))
        return self.head(x).squeeze(-1)


class IQLQHead(nn.Module):
    """Q head producing per-action values."""

    def __init__(self, *, hidden_size: int = 1024):
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.act = nn.ReLU()
        self.head = nn.Linear(256, ACTION_SPACE)
        nn.init.constant_(self.head.bias, 0.0)

    def forward(self, phi: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        x = self.act(self.bn1(self.fc1(phi)))
        q = self.head(x)
        if mask is not None:
            q = q.masked_fill(~mask, -torch.inf)
        return q
