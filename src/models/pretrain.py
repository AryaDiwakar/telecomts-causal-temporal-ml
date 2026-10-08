"""Self-supervised pretraining for the TAGN trunk (Bet 1).

Why: TelecomTS has 32k windows but only 1235 anomalous ones (~3.9%). A
supervised TAGN is label-starved, which is why a linear baseline on
hand-crafted features currently wins. This module exploits the ~96%
unlabelled telemetry that no prior TelecomTS baseline uses.

Three pretext objectives, all label-free:

  1. masked-KPI reconstruction : drop 35% of the 16 KPI channels per window,
                                 rebuild them from the remaining channels and
                                 their temporal context -> forces cross-KPI and
                                 cross-time structure into the encoder.
  2. next-step forecasting     : predict x[t+1] from the GRU state at t
                                 -> forces causal dynamics, not memorisation.
  3. contrastive (NT-Xent)     : two augmented views of the same window align
                                 in projection space, other windows repel
                                 -> makes the pooled embedding invariant to
                                 noise/channel-drop/time-warp jitter.

Only the trunk (``TagnModel.ENCODER_MODULES``) is trained; task heads stay
uninitialised until fine-tuning in exp_002.

Usage:
    from src.models.pretrain import TagnPretrain, pretrain_config
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.tagn import N_NUMERIC, TagnModel

ROOT = Path(__file__).resolve().parents[2]


def pretrain_config() -> dict:
    """Defaults, overridable via configs/pretrain.yaml."""
    import yaml

    path = ROOT / "configs" / "pretrain.yaml"
    if not path.exists():
        return dict(DEFAULTS)
    cfg = dict(DEFAULTS)
    for k, v in (yaml.safe_load(path.read_text()) or {}).items():
        if isinstance(v, dict):
            cfg.setdefault(k, {}).update(v)
        else:
            cfg[k] = v
    return cfg


DEFAULTS = {
    "epochs": 12,
    "batch_size": 128,
    "lr": 3.0e-4,
    "weight_decay": 1.0e-4,
    "grad_clip": 1.0,
    "mask_ratio": 0.35,
    "noise_std": 0.05,
    "channel_drop": 0.15,
    "temperature": 0.2,
    "proj_dim": 128,
    "weights": {"recon": 1.0, "forecast": 0.5, "contrastive": 0.5},
    "patience": 4,
    "device": "auto",
}


def resolve_device(name: str = "auto") -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# --------------------------------------------------------------------------- #
# augmentations (input-level, label-free)
# --------------------------------------------------------------------------- #
def augment(x: torch.Tensor, rng: np.random.Generator, noise_std: float, drop: float) -> torch.Tensor:
    """Channel dropout + gaussian noise. x: [B, T, 18] -> same shape."""
    B, T, C = x.shape
    out = x.clone()
    keep = torch.rand(B, N_NUMERIC, device=x.device) >= drop
    keep = keep.unsqueeze(1).expand(-1, T, -1)            # [B,T,16]
    out[:, :, :N_NUMERIC] = out[:, :, :N_NUMERIC] * keep
    if noise_std > 0:
        out[:, :, :N_NUMERIC] = out[:, :, :N_NUMERIC] + torch.randn(
            B, T, N_NUMERIC, device=x.device
        ) * noise_std
    return out


def sample_mask(batch: int, t: int, ratio: float, device, rng: np.random.Generator) -> torch.Tensor:
    """Per-window random subset of KPI channels, >=1 always masked. [B,T,16] bool."""
    n = max(1, int(round(ratio * N_NUMERIC)))
    mask = torch.zeros(batch, N_NUMERIC, dtype=torch.bool, device=device)
    for b in range(batch):
        idx = rng.choice(N_NUMERIC, size=n, replace=False)
        mask[b, idx] = True
    return mask.unsqueeze(1).expand(-1, t, -1)


# --------------------------------------------------------------------------- #
# pretrainer
# --------------------------------------------------------------------------- #
class TagnPretrain(nn.Module):
    """Wraps a TAGN trunk with pretext heads. Discarded after pretraining."""

    def __init__(self, tagn: TagnModel, cfg: dict | None = None):
        super().__init__()
        self.cfg = dict(DEFAULTS) if cfg is None else {**DEFAULTS, **cfg}
        self.tagn = tagn
        d = tagn.cfg["d"]

        self.mask_token = nn.Parameter(torch.zeros(1, 1, N_NUMERIC))
        nn.init.normal_(self.mask_token, std=0.02)

        self.recon_head = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, N_NUMERIC))
        self.forecast_head = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, N_NUMERIC))
        self.proj = nn.Sequential(
            nn.Linear(d, self.cfg["proj_dim"]), nn.GELU(),
            nn.Linear(self.cfg["proj_dim"], self.cfg["proj_dim"]),
        )
        # learned graph is a trunk concern; pretraining reuses the same prototypes
        self._has_protos = tagn.prototypes is not None

    # -- masking ---------------------------------------------------------- #
    def _mask_input(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = x.clone()
        tok = self.mask_token.to(x.dtype).expand(-1, x.shape[1], -1)
        x[:, :, :N_NUMERIC] = torch.where(mask, tok, x[:, :, :N_NUMERIC])
        return x

    # -- losses ----------------------------------------------------------- #
    @staticmethod
    def nt_xent(z1: torch.Tensor, z2: torch.Tensor, temperature: float) -> torch.Tensor:
        z1 = F.normalize(z1, dim=-1)
        z2 = F.normalize(z2, dim=-1)
        B = z1.shape[0]
        z = torch.cat([z1, z2], dim=0)                       # [2B, d]
        sim = (z @ z.t()) / temperature                      # [2B, 2B]
        sim.fill_diagonal_(float("-inf"))
        tgt = torch.arange(2 * B, device=z.device)
        tgt = (tgt + B) % (2 * B)                            # partner index
        return F.cross_entropy(sim, tgt)

    def forward(
        self,
        x: torch.Tensor,
        chunk_proto: torch.Tensor,
        rng: np.random.Generator,
    ) -> dict:
        B, T, _ = x.shape
        w = self.cfg["weights"]

        # view A: augmented + channel-masked, used for recon/forecast
        xa = augment(x, rng, self.cfg["noise_std"], self.cfg["channel_drop"])
        mask = sample_mask(B, T, self.cfg["mask_ratio"], x.device, rng)
        xa_m = self._mask_input(xa, mask)
        enc_a = self.tagn.encode(xa_m, chunk_proto)

        # view B: independently augmented, used for the contrastive pairing
        xb = augment(x, rng, self.cfg["noise_std"], self.cfg["channel_drop"])
        enc_b = self.tagn.encode(xb, chunk_proto)

        # 1. masked-KPI reconstruction (loss only where channels were hidden)
        raw = self.recon_head(enc_a["node_feats"])                       # [B,T,16,16]
        k = torch.arange(N_NUMERIC, device=x.device)
        pred = raw[:, :, k, k]                                           # node k -> KPI k
        tgt = x[:, :, :N_NUMERIC]
        denom = mask.float().sum().clamp_min(1.0)
        l_recon = ((pred - tgt) ** 2 * mask.float()).sum() / denom

        # 2. next-step forecasting from the GRU state at t
        l_fc = F.mse_loss(self.forecast_head(enc_a["h"][:, :-1]), tgt[:, 1:])

        # 3. contrastive
        l_ctr = self.nt_xent(self.proj(enc_a["pooled"]), self.proj(enc_b["pooled"]),
                             self.cfg["temperature"])

        total = w["recon"] * l_recon + w["forecast"] * l_fc + w["contrastive"] * l_ctr
        return {"loss": total, "recon": l_recon.detach(), "forecast": l_fc.detach(),
                "contrastive": l_ctr.detach()}


def build_pretrainer(cfg_model: dict, prototypes: torch.Tensor, pcfg: dict,
                     device: torch.device) -> TagnPretrain:
    tagn = TagnModel(cfg_model)
    tagn.set_prototypes(prototypes)
    pre = TagnPretrain(tagn, pcfg).to(device)
    return pre


def pretrained_encoder_path() -> Path:
    return ROOT / "checkpoints" / "exp_008" / "tagn_encoder_pretrained.pt"


def save_encoder(pre: TagnPretrain, extra: dict | None = None) -> Path:
    path = pretrained_encoder_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "encoder_state": pre.tagn.encoder_state(),
        "cfg": {k: v for k, v in pre.tagn.cfg.items() if k != "device"},
        "pretrain_cfg": {k: v for k, v in pre.cfg.items() if k != "device"},
    }
    if extra:
        payload.update(extra)
    torch.save(payload, path)
    return path
