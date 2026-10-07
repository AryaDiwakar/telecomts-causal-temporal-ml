"""Tabular-signal + dynamic-graph network (TAGN) for TelecomTS.

Composition (CPU-friendly):
  1. Depthwise temporal conv over each KPI channel (local dynamics).
  2. Per-step GAT over the channel-level adjacency learned by
     ``src.models.graphs`` (prototype graph per chunk => dynamic topology).
  3. Temporal aggregation via GRU + self-attention pooling to classify the
     chunk and emit per-node (per-KPI) root-cause scores.
  4. Causal per-step hazard head for early warning (onset within horizon).

Heads:
  anomaly      : binary (chunk-level)
  type         : 11-way (anomalous only)
  root_cause   : 16 multi-label logits against affected-KPI template
  early_warning: per-step hazard, masked pre-onset-only

Usage (import):
    from src.models.tagn import TagnModel, model_config
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "processed"

N_NUMERIC = 16
N_PROTO = 2


def model_config():
    """Flat config derived from configs/model.yaml."""
    cfg_path = ROOT / "configs" / "model.yaml"
    import yaml

    cfg = yaml.safe_load(cfg_path.read_text())
    t = cfg["temporal_encoder"]
    g = cfg["gat"]
    tr = cfg["training"]
    device_name = os.getenv("TAGN_DEVICE", cfg["common"].get("device", "auto"))
    if device_name == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(device_name)
    input_dim = int(cfg["common"].get("input_dim", N_NUMERIC + N_PROTO))
    n_kpis = int(cfg["common"].get("n_kpis", N_NUMERIC))
    if input_dim != n_kpis + N_PROTO:
        raise ValueError(
            f"model input_dim={input_dim} must equal n_kpis + {N_PROTO} "
            f"({n_kpis + N_PROTO})"
        )
    return {
        "input_dim": input_dim,
        "n_kpis": n_kpis,
        "d": t["hidden_dim"],
        "n_layers_t": t["n_layers"],
        "heads_t": t["n_heads"],
        "gat_heads": g["n_heads"],
        "gat_layers": g["n_layers"],
        "dropout": cfg["common"]["dropout"],
        "device": device,
        "batch_size": tr["batch_size"],
        "lr": tr["learning_rate"],
        "weight_decay": tr["weight_decay"],
        "epochs": tr["epochs"],
        "patience": tr["early_stopping_patience"],
        "grad_clip": tr["grad_clip"],
        "loss_weights": tr["loss_weights"],
    }


def load_graphs() -> tuple[torch.Tensor, torch.LongTensor]:
    gz = np.load(PROCESSED / "graph_prototypes.npz")
    prototypes = torch.tensor(gz["prototypes"], dtype=torch.float32)  # [K,16,16]
    chunk_proto = torch.tensor(gz["chunk_prototypes"], dtype=torch.long)
    if prototypes.ndim != 3 or prototypes.shape[1:] != (N_NUMERIC, N_NUMERIC):
        raise ValueError(f"invalid graph prototype shape: {tuple(prototypes.shape)}")
    if not torch.isfinite(prototypes).all():
        raise ValueError("graph prototypes contain non-finite values")
    if (chunk_proto < 0).any() or (chunk_proto >= len(prototypes)).any():
        raise ValueError("chunk graph assignments contain an invalid prototype id")
    return prototypes, chunk_proto


class DilatedCausalConv(nn.Module):
    def __init__(self, in_channels: int, d: int, kernel: int = 5, layers: int = 2, dilation_growth: int = 2):
        super().__init__()
        self.blocks = nn.ModuleList()
        dim = in_channels
        for i in range(layers):
            self.blocks.append(
                nn.Sequential(
                    nn.Conv1d(dim, d, kernel_size=kernel, padding=(kernel - 1) * (dilation_growth ** i),
                              dilation=dilation_growth ** i),
                    nn.GELU(),
                    nn.Dropout(0.1),
                )
            )
            dim = d

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, C]; causal (trims the right-side padding of every block)
        h = x.transpose(1, 2)
        for blk in self.blocks:
            h = blk(h)[:, :, :-blk[0].padding[0]]
        return h.transpose(1, 2)  # [B, T, d]


class NodeEncoder(nn.Module):
    """Causal temporal encoding with a distinct embedding for every KPI node."""

    def __init__(self, d: int, kernel: int = 5, layers: int = 2, dilation_growth: int = 2):
        super().__init__()
        self.d = d
        self.blocks = nn.ModuleList()
        dim = N_NUMERIC
        for i in range(layers):
            dil = dilation_growth ** i
            pad = (kernel - 1) * dil
            self.blocks.append(
                nn.Conv1d(
                    dim,
                    N_NUMERIC * d,
                    kernel_size=kernel,
                    padding=pad,
                    dilation=dil,
                    groups=N_NUMERIC,
                )
            )
            dim = N_NUMERIC * d
        self.ln = nn.LayerNorm(d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, 16] -> grouped temporal conv -> [B, T, 16, d]
        h = x.transpose(1, 2)
        for blk in self.blocks:
            h = blk(h)[:, :, :-blk.padding[0]]
        h = h.view(x.shape[0], N_NUMERIC, self.d, x.shape[1]).permute(0, 3, 1, 2)
        return self.ln(h)


class GATLayer(nn.Module):
    def __init__(self, d: int, n_heads: int):
        super().__init__()
        if d % n_heads:
            raise ValueError(f"hidden dimension {d} must be divisible by {n_heads} heads")
        self.n_heads = n_heads
        self.d = d
        self.dh = d // n_heads
        self.q = nn.Linear(d, d)
        self.k = nn.Linear(d, d)
        self.v = nn.Linear(d, d)
        self.g = nn.Parameter(torch.zeros(n_heads))
        self.drop = nn.Dropout(0.1)
        self.out = nn.Linear(d, d)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        # x: [B, T, 16, d], adj: [B, 16, 16]
        B, T, N, _ = x.shape
        q = self.q(x).view(B * T, N, self.n_heads, self.dh)
        k = self.k(x).view(B * T, N, self.n_heads, self.dh)
        v = self.v(x).view(B * T, N, self.n_heads, self.dh)
        att = torch.einsum("bihd,bjhd->bijh", q, k) / (self.dh ** 0.5)
        att = att + self.g.view(1, 1, 1, -1)  # GATv2-style hyper-embedding
        adjb = adj.unsqueeze(1).expand(-1, T, -1, -1).reshape(B * T, N, N)
        # A self-loop keeps isolated graph nodes from receiving a uniform
        # average over every node after the softmax mask is applied.
        self_loop = torch.eye(N, device=adj.device, dtype=torch.bool).unsqueeze(0)
        mask = ((adjb > 1e-6) | self_loop).unsqueeze(-1)
        att = att.masked_fill(mask == 0, float("-1e9"))
        att = F.softmax(att, dim=2)
        out = torch.einsum("bijh,bjhd->bihd", self.drop(att), v).reshape(B * T, N, self.d)
        return x + self.out(out).view(B, T, N, self.d)


class TagnModel(nn.Module):
    def __init__(self, cfg: dict, n_types: int = 11, variant: str = "full"):
        super().__init__()
        self.cfg = cfg
        self.variant = variant
        self.n_kpis = int(cfg.get("n_kpis", N_NUMERIC))
        self.input_dim = int(cfg.get("input_dim", self.n_kpis + N_PROTO))
        if self.n_kpis != N_NUMERIC:
            raise ValueError(f"TAGN currently supports {N_NUMERIC} KPI nodes, got {self.n_kpis}")
        if self.input_dim != self.n_kpis + N_PROTO:
            raise ValueError("input_dim must contain numeric KPIs plus two protocol channels")
        d = cfg["d"]
        self.in_proj = nn.Sequential(nn.Linear(self.input_dim, d), nn.GELU())
        self.tconv = DilatedCausalConv(d, d, layers=cfg["n_layers_t"])
        if variant != "no_nodeconv":
            self.node_enc = NodeEncoder(d, layers=cfg["gat_layers"])
        else:
            self.node_enc = nn.Sequential(
                nn.Conv1d(N_NUMERIC, d, kernel_size=1), nn.GELU()
            )
        if variant != "no_graph":
            self.gats = nn.ModuleList(
                [GATLayer(d if i == 0 else d, cfg["gat_heads"])
                 for i in range(cfg["gat_layers"])]
            )
        self.fuse = nn.Linear(d + d, d)
        self.gru = nn.GRU(d, d, batch_first=True, num_layers=1)
        self.pool = nn.Linear(d, d)
        self.anom_head = nn.Linear(d, 1)
        self.type_head = nn.Linear(d, n_types)
        self.kpi_head = nn.Linear(d, 1)
        self.ew_head = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, 1))
        self.drop = nn.Dropout(cfg["dropout"])
        self.prototypes: nn.Parameter | None = None

    def set_prototypes(self, prototypes: torch.Tensor) -> None:
        if prototypes.ndim != 3 or prototypes.shape[1:] != (self.n_kpis, self.n_kpis):
            raise ValueError(
                f"expected graph prototypes [K,{self.n_kpis},{self.n_kpis}], "
                f"got {tuple(prototypes.shape)}"
            )
        if not torch.isfinite(prototypes).all():
            raise ValueError("graph prototypes contain non-finite values")
        device = next(self.parameters()).device
        prototypes = prototypes.detach().clone().to(device)
        self.prototypes = nn.Parameter(prototypes, requires_grad=False)
        # static-graph variant: single mean adjacency used for every chunk
        # (contiguous repeat, NOT expand: an expanded param shares memory and
        # breaks load_state_dict's copy_ with an overlap error).
        self.static_adj = nn.Parameter(
            prototypes.mean(dim=0, keepdim=True).repeat(len(prototypes), 1, 1),
            requires_grad=False,
        )

    def graph_for(self, chunk_proto: torch.Tensor) -> torch.Tensor:
        if self.prototypes is None:
            raise RuntimeError("call set_prototypes first")
        adj = self.prototypes[chunk_proto]  # [B,16,16]
        if self.variant == "static_graph":
            adj = self.static_adj[chunk_proto]
        if self.variant == "undirected":
            adj = 0.5 * (adj + adj.transpose(1, 2))
        return adj

    def forward(self, x: torch.Tensor, chunk_proto: torch.Tensor) -> dict:
        # x: [B, T, 18] (16 scaled numeric + 2 protocol codes)
        if x.ndim != 3 or x.shape[-1] != self.input_dim:
            raise ValueError(f"expected x with shape [B,T,{self.input_dim}], got {tuple(x.shape)}")
        if chunk_proto.ndim != 1 or chunk_proto.shape[0] != x.shape[0]:
            raise ValueError("chunk_proto must be a [B] tensor aligned with x")
        B, T, _ = x.shape
        adj = self.graph_for(chunk_proto)
        prot = x[:, :, N_NUMERIC:N_NUMERIC + 2].mean(dim=1, keepdim=True).expand(-1, T, -1)

        t_in = torch.cat([x[:, :, :N_NUMERIC], prot], dim=-1)
        h0 = self.in_proj(t_in)                         # [B, T, d]
        hh = self.tconv(self.drop(h0)) + h0             # residual
        xnode = x[:, :, :N_NUMERIC]                     # [B, T, 16]
        if self.variant == "no_nodeconv":
            node_feats = self.node_enc(xnode.transpose(1, 2)).transpose(1, 2)  # [B,T,d]
            node_feats = node_feats.unsqueeze(2).expand(-1, -1, N_NUMERIC, -1)
        else:
            node_feats = self.node_enc(xnode)           # [B, T, 16, d]
        if self.variant != "no_graph":
            for gat in self.gats:
                node_feats = gat(node_feats, adj)
        # fuse: pool nodes per step -> context
        ctx = torch.cat([hh, node_feats.mean(dim=2)], dim=-1)
        fused = self.drop(F.gelu(self.fuse(ctx)))       # [B, T, d]
        h, _ = self.gru(fused)                           # [B, T, d]

        # chunk pooling (attention over time)
        attn = torch.softmax(self.pool(h), dim=1)        # [B, T, d]
        pooled = (attn * h).sum(dim=1)                   # [B, d]

        logits = {
            "anomaly": self.anom_head(self.drop(pooled)).squeeze(-1),
            "type": self.type_head(pooled),
            "kpi": self.kpi_head(node_feats.mean(dim=1)).squeeze(-1),  # [B,16] per-node score
            "hazard": self.ew_head(h).squeeze(-1),                     # [B,T]
        }
        # explicit template-consistent root-cause: map type logits -> template mask
        return logits, node_feats, h


if __name__ == "__main__":
    mcfg = model_config()
    m = TagnModel(mcfg)
    p, cp = load_graphs()
    m.set_prototypes(p)
    out, nf, h = m(torch.randn(4, 128, 18), cp[:4])
    print({k: tuple(v.shape) for k, v in out.items()})
    n_params = sum(pp.numel() for pp in m.parameters())
    print("params:", n_params)
