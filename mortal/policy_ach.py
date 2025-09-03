import torch
from torch import nn
from .libriichi.consts import ACTION_SPACE


class PolicyNet(nn.Module):
    def __init__(self, *, hidden_size: int = 1024):
        super().__init__()
        # Two-layer head: hidden -> 256 -> ACTION_SPACE with BN + ReLU
        self.fc1 = nn.Linear(hidden_size, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.act = nn.ReLU()
        self.head = nn.Linear(256, ACTION_SPACE)
        nn.init.constant_(self.head.bias, 0.0)

    def forward(self, phi: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # Returns masked logits; invalid actions set to -inf
        x = self.act(self.bn1(self.fc1(phi)))
        logits = self.head(x)
        logits = logits.masked_fill(~mask, -torch.inf)
        return logits


class ValueHead(nn.Module):
    def __init__(self, *, hidden_size: int = 1024):
        super().__init__()
        # Two-layer head: hidden -> 256 -> 1 with BN + ReLU, tanh output in [-3, 3]
        self.fc1 = nn.Linear(hidden_size, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.act = nn.ReLU()
        self.head = nn.Linear(256, 1)
        nn.init.constant_(self.head.bias, 0.0)

    def forward(self, phi: torch.Tensor) -> torch.Tensor:
        x = self.act(self.bn1(self.fc1(phi)))
        v = self.head(x).squeeze(-1)
        return 3.0 * torch.tanh(v)
