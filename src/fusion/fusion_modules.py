"""Gated auxiliary residual with an exact zero-initialized fallback."""
import torch
from torch import nn


class GatedResidualFusion(nn.Module):
    def __init__(self, channels, zero_init=True):
        super().__init__()
        self.proj = nn.Conv2d(channels, channels, 1)
        self.gate = nn.Sequential(nn.Conv2d(2 * channels, 1, 1), nn.Sigmoid())
        nn.init.zeros_(self.gate[0].weight)
        nn.init.zeros_(self.gate[0].bias)  # sigmoid(0) = 0.5, residual still exactly zero
        if zero_init:
            nn.init.zeros_(self.proj.weight)
            nn.init.zeros_(self.proj.bias)

    def forward(self, rgb, ir):
        return rgb + self.gate(torch.cat((rgb, ir), 1)) * self.proj(ir)
