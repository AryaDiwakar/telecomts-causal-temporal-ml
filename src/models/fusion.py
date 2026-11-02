"""Multimodal observability fusion for TAGN v3 (Phase 2).

Fuses three modalities per window:
  - KPI time series (via TemporalBackbone)
  - graph topology (adjacency + edge strengths)
  - scenario context (zone/mobility/app embedding)

Uses cross-modal attention + late fusion + modality dropout for robustness.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.backbone import TemporalBackbone


class ModalityDropout(nn.Module):
    def __init__(self, p=0.15):
        super().__init__()
        self.p = p

    def forward(self, x):
        if not self.training or self.p <= 0:
            return x
        mask = (torch.rand(x.size(0), 1, device=x.device) > self.p).float()
        return x * mask


class CrossModalAttention(nn.Module):
    def __init__(self, hidden, n_heads=4, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(hidden, n_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(hidden)

    def forward(self, q, kv):
        # q: (B,1,H), kv: (B,M,H)
        out, _ = self.attn(q, kv, kv, need_weights=False)
        return self.norm(q + out).squeeze(1)


class MultimodalFusion(nn.Module):
    def __init__(self, n_channels=18, hidden=64, n_heads=4, n_context=8,
                 n_graph_nodes=16, dropout=0.1, modality_dropout=0.15):
        super().__init__()
        self.backbone = TemporalBackbone(n_channels=n_channels, hidden=hidden, n_heads=n_heads, dropout=dropout)
        self.graph_proj = nn.Sequential(nn.Linear(n_graph_nodes * n_graph_nodes, hidden), nn.GELU())
        self.ctx_proj = nn.Sequential(nn.Linear(n_context, hidden), nn.GELU())
        self.cross_kpi_graph = CrossModalAttention(hidden, n_heads, dropout)
        self.cross_kpi_ctx = CrossModalAttention(hidden, n_heads, dropout)
        self.fuse = nn.Sequential(nn.Linear(hidden * 3, hidden), nn.GELU(), nn.LayerNorm(hidden))
        self.mod_drop = ModalityDropout(modality_dropout)

    def forward(self, x, graph_adj, context):
        # x: (B,T,C); graph_adj: (B,N,N); context: (B,Cctx)
        h_kpi = self.backbone(x)  # (B,H)
        h_graph = self.graph_proj(graph_adj.flatten(1))  # (B,H)
        h_ctx = self.ctx_proj(context)  # (B,H)
        h_kpi = self.mod_drop(h_kpi)
        h_graph = self.mod_drop(h_graph)
        h_ctx = self.mod_drop(h_ctx)
        q = h_kpi.unsqueeze(1)
        kv = torch.stack([h_graph, h_ctx], dim=1)
        fused_cross = self.cross_kpi_graph(q, kv)  # (B,H) using graph as kv (approx)
        # also cross with context
        kv2 = torch.stack([h_kpi, h_graph], dim=1)
        fused_ctx = self.cross_kpi_ctx(h_ctx.unsqueeze(1), kv2)
        out = self.fuse(torch.cat([h_kpi, fused_cross, fused_ctx], dim=-1))
        return out
