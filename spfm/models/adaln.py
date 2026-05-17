import torch
import torch.nn as nn


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class AdaLN(nn.Module):
    def __init__(self, hidden_size: int, cond_dim: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_dim, 2 * hidden_size, bias=True),
        )
        self.norm = nn.RMSNorm(hidden_size, eps=1e-6)
        nn.init.zeros_(self.proj[1].weight)
        nn.init.zeros_(self.proj[1].bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        shift, scale = self.proj(cond).chunk(2, dim=1)
        return modulate(self.norm(x), shift, scale)
