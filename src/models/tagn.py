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
import torch.autograd as autograd

ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "processed"

N_NUMERIC = 16
N_PROTO = 2


class _GradReverse(autograd.Function):
    @staticmethod
    def forward(ctx, x, lam):
        ctx.lam = lam
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lam * grad_output, None


def grad_reverse(x, lam):
    if lam <= 0.0:
        return x
    return _GradReverse.apply(x, lam)


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
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            device = torch.device("mps")
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


# Chunk-level statistics, in the exact order used by
# ``src.features.classical.extract_features`` so the summary branch has feature
# parity with the classical baseline (which is why that baseline is strong).
SUMMARY_STATS = (
    "mean", "std", "min", "max", "median", "p10", "p90",
    "slope", "energy", "zero_frac", "iqr",
)
# classical stats + last level + endpoint change
STATS_PER_KPI = len(SUMMARY_STATS) + 2
N_PROTO_SUMMARY = N_PROTO * 2  # first timestep + window mean, per protocol channel
SUMMARY_DIM = N_NUMERIC * STATS_PER_KPI + N_PROTO_SUMMARY


def chunk_summary(
    x: torch.Tensor, target_dim: int | None = None
) -> torch.Tensor:
    """Per-chunk statistical summary of a window.

    x: [B, T, C] -> [B, SUMMARY_DIM]. Matches ``extract_features`` so the
    anomaly head can express the classical baseline as a special case.

    The final-dimension guard applies to the canonical 16-KPI schema. Pass
    ``target_dim`` when summarizing a different observation space (e.g. the
    K=10 shared cross-vendor space in exp_010) to keep the invariant explicit.
    """
    if x.ndim != 3:
        raise ValueError(f"expected [B,T,C], got {tuple(x.shape)}")
    num = x[:, :, :N_NUMERIC]
    T = num.shape[1]
    t = torch.arange(T, dtype=x.dtype, device=x.device)
    tc = t - t.mean()
    slope_denom = (tc * tc).sum().clamp_min(1e-8)
    centered = num - num.mean(dim=1, keepdim=True)
    slope = (centered * tc.view(1, -1, 1)).sum(dim=1) / slope_denom
    q = torch.quantile(
        num, torch.tensor([0.1, 0.5, 0.9], dtype=x.dtype, device=x.device), dim=1
    )
    p10, median, p90 = q[0], q[1], q[2]
    cols = (
        num.mean(dim=1),
        num.std(dim=1, unbiased=False),
        num.amin(dim=1),
        num.amax(dim=1),
        median,
        p10,
        p90,
        slope,
        (num * num).sum(dim=1),
        (num == 0).to(x.dtype).mean(dim=1),
        p90 - p10,
        num[:, -1],
        num[:, -1] - num[:, 0],
    )
    proto = x[:, :, N_NUMERIC:N_NUMERIC + N_PROTO]
    # Keep the first 176 values byte-for-byte compatible with the classical
    # feature order (all statistics for KPI 0, then KPI 1, ...). The extra
    # endpoint features are appended after that anchor rather than interleaved
    # into it, so parity is testable and the neural model can add signal.
    base = torch.stack(cols[: len(SUMMARY_STATS)], dim=-1).reshape(x.shape[0], -1)
    extra = torch.stack(cols[len(SUMMARY_STATS):], dim=-1).reshape(x.shape[0], -1)
    summary = torch.cat(
        [base, extra, x[:, 0, N_NUMERIC:N_NUMERIC + N_PROTO], proto.mean(dim=1)], dim=-1
    )
    expect = SUMMARY_DIM if target_dim is None else target_dim
    if summary.shape[-1] != expect:
        raise ValueError(f"summary dim {summary.shape[-1]} != {expect}")
    return summary


class CausalChannelNorm(nn.Module):
    """LayerNorm over channels at each time step, without future leakage."""

    def __init__(self, d: int):
        super().__init__()
        self.norm = nn.LayerNorm(d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x.transpose(1, 2)).transpose(1, 2)


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
                    CausalChannelNorm(d),
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
        self.norm = nn.LayerNorm(d)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        # x: [B, T, 16, d], adj: [B, 16, 16]
        B, T, N, _ = x.shape
        q = self.q(x).view(B * T, N, self.n_heads, self.dh)
        k = self.k(x).view(B * T, N, self.n_heads, self.dh)
        v = self.v(x).view(B * T, N, self.n_heads, self.dh)
        att = torch.einsum("bihd,bjhd->bijh", q, k) / (self.dh ** 0.5)
        att = att + self.g.view(1, 1, 1, -1)  # GATv2-style hyper-embedding
        adjb = adj.unsqueeze(1).expand(-1, T, -1, -1).reshape(B * T, N, N)
        # ``adj[source, target]`` is the documented convention.  A target
        # must aggregate messages from its source, hence transpose here.
        incoming = adjb.transpose(1, 2)
        # A self-loop keeps isolated graph nodes from receiving a uniform
        # average over every node after the softmax mask is applied.
        self_loop = torch.eye(N, device=adj.device, dtype=torch.bool).unsqueeze(0)
        mask = ((incoming > 1e-6) | self_loop).unsqueeze(-1)
        # Retain the inferred edge strength instead of using the graph as a
        # binary mask.  The log transform prevents a few strong edges from
        # overwhelming the learned attention scores.
        edge_bias = torch.log1p(incoming.clamp_min(0.0)).unsqueeze(-1)
        att = att + edge_bias
        att = att.masked_fill(mask == 0, float("-1e9"))
        att = F.softmax(att, dim=2)
        out = torch.einsum("bijh,bjhd->bihd", self.drop(att), v).reshape(B * T, N, self.d)
        return self.norm(x + self.out(out).view(B, T, N, self.d))


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
        # The classical baseline is strong because it exposes distributional
        # shifts (spread, level, endpoint change) explicitly. The summary branch
        # therefore carries the *same* statistics extract_features() computes,
        # so the classical model is expressible rather than unbeatable.
        self.summary_proj = nn.Sequential(
            nn.Linear(SUMMARY_DIM, d), nn.GELU(), nn.Dropout(cfg["dropout"])
        )
        # Direct linear readout over the summary: this path alone reproduces
        # logistic regression on the classical features, so the temporal and
        # graph branches only have to add value on top of it.
        self.summary_head = nn.Linear(SUMMARY_DIM, 1)
        self.summary_residual = nn.Sequential(
            nn.Linear(SUMMARY_DIM, d), nn.GELU(), nn.Dropout(cfg["dropout"]),
            nn.Linear(d, 1),
        )
        self.summary_type_head = nn.Linear(SUMMARY_DIM, n_types)
        # Per-channel location/scale of the summary, fitted on the training
        # split only. `energy` reaches ~4e6 while `slope` is ~2, so raw inputs
        # would swamp summary_proj at init. Buffers (not Parameters) so they
        # follow the module to the right device and ride along in checkpoints.
        self.register_buffer("summary_loc", torch.zeros(SUMMARY_DIM))
        self.register_buffer("summary_scale", torch.ones(SUMMARY_DIM))
        self.anom_head = nn.Linear(d, 1)
        self.type_head = nn.Linear(d, n_types)
        self.kpi_head = nn.Linear(d, 1)
        self.ew_head = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, 1))
        self.rca_adv = nn.Sequential(
            nn.Linear(N_NUMERIC, d), nn.GELU(), nn.Linear(d, n_types)
        )
        self.ident_head = nn.Sequential(
            nn.Linear(2, d), nn.GELU(), nn.Linear(d, 1)
        )
        self.adv_lambda = float(cfg.get("grl_lambda", 0.0))
        # Start the anomaly readouts at exactly zero: with a default init the
        # trunk path fits the training split first, which collapses
        # sigmoid(logit)-y and starves summary_head of gradient. Starting from
        # zero lets the linear path converge before the trunk contributes.
        for head in (self.anom_head, self.summary_head, self.summary_type_head):
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)
        nn.init.zeros_(self.summary_residual[-1].weight)
        nn.init.zeros_(self.summary_residual[-1].bias)
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

    ENCODER_MODULES = ("in_proj", "tconv", "node_enc", "gats", "fuse", "gru", "pool")

    def calibrate_summary(self, X: torch.Tensor, batch: int = 4096) -> None:
        """Fit the summary normalization on the training split only.

        An invertible per-channel affine map, so any linear function of the
        raw summary is still a linear function of the normalized one: the
        classical baseline remains expressible through ``summary_head``.
        """
        n = 0
        total = torch.zeros(SUMMARY_DIM, dtype=torch.float64)
        total_sq = torch.zeros(SUMMARY_DIM, dtype=torch.float64)
        with torch.no_grad():
            for i in range(0, len(X), batch):
                # MPS has no float64; accumulate the summary moments on CPU
                s = chunk_summary(X[i:i + batch]).cpu().double()
                total += s.sum(dim=0)
                total_sq += (s * s).sum(dim=0)
                n += len(s)
        mean = total / max(1, n)
        std = (total_sq / max(1, n) - mean * mean).clamp_min(0.0).sqrt().clamp_min(1e-6)
        self.summary_loc.copy_(mean.to(self.summary_loc.dtype))
        self.summary_scale.copy_(std.to(self.summary_scale.dtype))

    def encoder_state(self) -> dict:
        """State dict of the shared trunk only (pretrained by exp_008)."""
        out = {}
        for name in self.ENCODER_MODULES:
            mod = getattr(self, name, None)
            if mod is None:
                continue
            for k, v in mod.state_dict().items():
                out[f"{name}.{k}"] = v
        return out

    def load_encoder_state(self, state: dict) -> int:
        """Load a trunk-only checkpoint; returns the number of tensors applied."""
        own = self.state_dict()
        n = 0
        for k, v in state.items():
            if k in own and own[k].shape == v.shape and own[k].dtype == v.dtype:
                own[k].copy_(v)
                n += 1
        if n == 0:
            raise ValueError("no encoder tensors matched the current model")
        self.load_state_dict(own, strict=True)
        return n

    def encode(self, x: torch.Tensor, chunk_proto: torch.Tensor) -> dict:
        """Shared trunk. Returns pooled/h/node_feats used by every head."""
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
        return {"pooled": pooled, "h": h, "node_feats": node_feats, "fused": fused}

    def forward(self, x: torch.Tensor, chunk_proto: torch.Tensor) -> dict:
        enc = self.encode(x, chunk_proto)
        pooled, h, node_feats = enc["pooled"], enc["h"], enc["node_feats"]

        summary_raw = chunk_summary(x)
        # Normalize once and use everywhere. `energy` reaches ~4e6 while
        # `slope` is ~2; with raw values Adam gives every weight the same
        # step size regardless of input scale, so w_energy and w_slope grow
        # together and the logit reaches ~240 after a single epoch. A fixed
        # per-channel affine map (fitted on train only) keeps both branches
        # on the same footing while preserving the linear span, so the
        # classical baseline stays expressible.
        summary = (summary_raw - self.summary_loc) / self.summary_scale
        # Summary statistics are injected only into chunk-level heads. The
        # per-step hazard head remains strictly causal.
        head_context = pooled + self.summary_proj(summary)

        kpi_logits = self.kpi_head(node_feats.mean(dim=1)).squeeze(-1)  # [B,16]
        logits = {
            "anomaly": (
                self.summary_head(summary)
                + self.summary_residual(summary)
                + float(self.cfg.get("anomaly_trunk_weight", 1.0))
                * self.anom_head(self.drop(head_context))
            ).squeeze(-1),
            "type": self.type_head(head_context) + self.summary_type_head(summary),
            "kpi": kpi_logits,  # [B,16] per-node score
            "hazard": self.ew_head(h).squeeze(-1),                     # [B,T]
        }
        try:
            logits["rca_adv"] = self.rca_adv(
                grad_reverse(kpi_logits, self.adv_lambda)
            )
        except Exception:
            pass
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
