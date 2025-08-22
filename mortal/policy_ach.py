import torch
from torch import nn
from .libriichi.consts import ACTION_SPACE


class PolicyNet(nn.Module):
    def __init__(self, *, hidden_size: int = 1024):
        super().__init__()
        # Simple linear head from Brain features to action logits
        self.head = nn.Linear(hidden_size, ACTION_SPACE)
        nn.init.constant_(self.head.bias, 0.0)

    def forward(self, phi: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # Returns masked logits; invalid actions set to -inf
        logits = self.head(phi)
        logits = logits.masked_fill(~mask, -torch.inf)
        return logits


class ValueHead(nn.Module):
    def __init__(self, *, hidden_size: int = 1024):
        super().__init__()
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, phi: torch.Tensor) -> torch.Tensor:
        return self.head(phi).squeeze(-1)

