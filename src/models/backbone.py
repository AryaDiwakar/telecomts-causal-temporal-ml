"""Temporal backbone for TAGN v3: TCN + Transformer hybrid.

Designed to beat the standalone TCN baseline by combining:
  - dilated causal convolutions (local multi-scale temporal patterns)
  - multi-head self-attention (long-range dependencies)
  - residual connections + layer norm (stable training)

Usage:
    from src.models.backbone import TemporalBackbone
    b = TemporalBackbone(n_channels=18, hidden=64, n_heads=4, n_layers=2)
    out = b(x)  # x: (B, T, C) -> (B, hidden)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class _CausalConv1d(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, dilation):
        super().__init__()
        self.pad = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size, dilation=dilation, padding=0)

    def forward(self, x):  # x: (B, C, T)
        x = F.pad(x, (self.pad, 0))
        return self.conv(x)


class _TCNBlock(nn.Module):
    def __init__(self, channels, kernel_size=3, dilation=1, dropout=0.1):
        super().__init__()
        self.conv1 = _CausalConv1d(channels, channels, kernel_size, dilation)
        self.conv2 = _CausalConv1d(channels, channels, kernel_size, dilation)
        self.norm1 = nn.LayerNorm(channels)
        self.norm2 = nn.LayerNorm(channels)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):  # (B, T, C)
        h = x.transpose(1, 2)  # (B, C, T)
        h = self.conv1(h).transpose(1, 2)
        h = self.norm1(h + x)
        h = h.transpose(1, 2)
        h = self.conv2(h).transpose(1, 2)
        h = self.norm2(h + h)
        h = self.drop(h)
        return h


class _TransformerBlock(nn.Module):
    def __init__(self, hidden, n_heads, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(hidden, n_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(hidden)
        self.ff = nn.Sequential(
            nn.Linear(hidden, hidden * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden * 4, hidden),
            nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(hidden)

    def forward(self, x, mask=None):
        a, _ = self.attn(x, x, x, attn_mask=mask, need_weights=False)
        x = self.norm1(x + a)
        x = self.norm2(x + self.ff(x))
        return x


class TemporalBackbone(nn.Module):
    """TCN + Transformer hybrid producing a single vector per window."""

    def __init__(self, n_channels=18, hidden=64, n_heads=4, n_tcn_layers=3,
                 n_transformer_layers=2, kernel_size=3, dropout=0.1):
        super().__init__()
        self.proj = nn.Linear(n_channels, hidden)
        self.tcn = nn.ModuleList([
            _TCNBlock(hidden, kernel_size, dilation=2 ** i, dropout=dropout)
            for i in range(n_tcn_layers)
        ])
        self.transformer = nn.ModuleList([
            _TransformerBlock(hidden, n_heads, dropout=dropout)
            for _ in range(n_transformer_layers)
        ])
        self.pool = nn.Linear(hidden, 1)
        self.out_norm = nn.LayerNorm(hidden)

    def forward(self, x):  # x: (B, T, C)
        h = self.proj(x)  # (B, T, H)
        for block in self.tcn:
            h = block(h)
        for block in self.transformer:
            h = block(h)
        w = torch.softmax(self.pool(h), dim=1)  # (B, T, 1)
        out = (h * w).sum(dim=1)  # (B, H)
        return self.out_norm(out)
