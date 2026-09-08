"""
DH-KV: Dual-Heuristic KV-Cache — STAC Integration.
===================================================

Drop-in replacement for STAC's HeavyHittersKV that replaces H2O token
scoring with MoGA (Motion-Gated Attention).

Architecture:
    s_i^H2O = γ · s_i^{t-1} + Σ_j α_j,i           (unchanged H2O base)
    g_i     = MoGA_gate(token_3d_pos_i, vehicle_state)  (physics-informed)
    s_i^DH  = s_i^H2O · g_i                        (gated score)
    M_anchor = Top-K tokens by s_i^DH              (same selection)

Also replaces the fixed window_size with speed-adaptive s(v).

Usage (in stream_session.py):
    from stac.dhkv_kv import DHKVManager
    kv_kwargs = self.register_kv_mgr(mode, images, DHKVManager, **kv_kwargs)

Copyright (c) 2026 DH-KV Authors.
"""

import logging
import math
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from .kv_manager import KVManager
from .h2o import HeavyHittersKV

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════
# MoGA Heuristic (inlined — no external dependency)
# ═══════════════════════════════════════════════════════════════════

def _adaptive_window(v_ms: float, s_min: int = 2, s_max: int = 12,
                     v_ref: float = 30.0) -> int:
    """Speed-adaptive sliding window: s(v) ∈ [s_min, s_max]."""
    v_clipped = max(v_ms, 0.0)
    s = s_min + (s_max - s_min) * v_clipped / v_ref
    return max(s_min, min(s_max, int(round(s))))


def _estimate_vehicle_state_from_poses(
    positions_3d: torch.Tensor,   # [H, T_live, 3] per-head token positions
    slot_idx: int = 0,
) -> Tuple[float, float, float]:
    """Estimate vehicle state from token position changes.

    In a real deployment, this would come from VGGT pose estimates.
    Here we use the average token displacement as a proxy.

    Returns:
        v: speed proxy (m/s equivalent)
        omega: turn rate proxy (rad/s equivalent)
        theta_cam: heading proxy (radians)
    """
    # Use coarse heuristics based on token positions
    # In practice, these come from VGGT's pose_enc output
    # For now: extract heading from mean token position
    if positions_3d.numel() == 0:
        return 0.0, 0.0, 0.0

    # Average over heads and tokens
    mean_pos = positions_3d.mean(dim=(0, 1))  # [3]

    # Heading: angle in XZ plane of mean forward tokens
    theta_cam = float(torch.atan2(mean_pos[0], mean_pos[2]).item())

    # Speed proxy: magnitude of mean position (distance from origin)
    v = float(torch.norm(mean_pos).item())

    # Turn proxy: lateral displacement of recent tokens
    # (simplified — in real impl, this comes from pose differences)
    omega = 0.0
    if positions_3d.shape[1] > 4:
        recent = positions_3d[:, -4:, :].mean(dim=(0, 1))  # [3]
        older = positions_3d[:, :4, :].mean(dim=(0, 1))    # [3]
        lateral_recent = float((-recent[0] * math.sin(theta_cam) +
                                 recent[2] * math.cos(theta_cam)).item())
        lateral_older = float((-older[0] * math.sin(theta_cam) +
                                older[2] * math.cos(theta_cam)).item())
        omega = (lateral_recent - lateral_older) / 4.0  # proxy

    return v, omega, theta_cam


def _moga_gate_vectorized(
    theta_i: torch.Tensor,     # [H, T] per-head token angles in XZ plane
    d_i: torch.Tensor,         # [H, T] per-head token distances
    theta_cam: float,
    v: float,
    omega: float,
    alpha: float = 2.0,
    beta: float = 0.3,
    g_min: float = 0.25,
) -> torch.Tensor:
    """Vectorized MoGA gate computation for [H, T] token arrays.

    g_i = g_forward · g_turn · g_speed
    """
    # g_forward: [H, T]
    cos_diff = torch.cos(theta_i - theta_cam)
    mask_forward = cos_diff > 0
    gf = torch.full_like(cos_diff, g_min)
    if mask_forward.any():
        gf[mask_forward] = g_min + (1.0 - g_min) * (cos_diff[mask_forward] ** alpha)

    # g_turn: [H, T]
    if abs(omega) < 1e-6:
        gt = torch.ones_like(theta_i)
    else:
        lateral = torch.sin(theta_i - theta_cam)
        gain = beta * (1.0 if omega > 0 else -1.0) * lateral * abs(omega) / 0.5
        gt = 1.0 + gain

    # g_speed: [H, T]
    d_0, d_min_val, v_ref = 50.0, 10.0, 30.0
    d_effective = torch.full_like(d_i, d_min_val)
    if mask_forward.any():
        scale = max(v / v_ref, 0.2)
        d_effective[mask_forward] = d_0 * scale
    gs = torch.exp(-d_i / d_effective.clamp(min=0.1))

    return gf * gt * gs


# ═══════════════════════════════════════════════════════════════════
# DH-KV Manager
# ═══════════════════════════════════════════════════════════════════

class DHKVManager(HeavyHittersKV):
    """
    DH-KV: MoGA-gated H2O KV cache manager.

    Inherits all H2O infrastructure (append_kv, decode_sparse_attn, _parallel_prune_kv)
    and overrides _update_scores to apply the MoGA gate, plus adds adaptive window.

    Extra __init__ params:
        use_moga: bool = True          — enable MoGA gating
        moga_alpha: float = 2.0        — forward cone sharpness
        moga_beta: float = 0.3         — turn sensitivity
        moga_g_min: float = 0.25       — floor weight
        adaptive_window_enabled: bool = True
        token_positions_3d: Optional[Tensor] — [S, H, T, 3] or None
    """

    def __init__(self, *args,
                 use_moga: bool = True,
                 moga_alpha: float = 2.0,
                 moga_beta: float = 0.3,
                 moga_g_min: float = 0.25,
                 adaptive_window_enabled: bool = True,
                 **kwargs):
        # Extract DH-KV params before passing to parent
        self.use_moga = bool(use_moga)
        self.moga_alpha = float(moga_alpha)
        self.moga_beta = float(moga_beta)
        self.moga_g_min = float(moga_g_min)
        self.adaptive_window_enabled = bool(adaptive_window_enabled)

        # Token 3D positions: stored externally, updated each frame
        self._token_positions_3d: Optional[torch.Tensor] = None  # [L_eff, H, T, 3]
        self._vehicle_v: float = 0.0
        self._vehicle_omega: float = 0.0
        self._vehicle_theta_cam: float = 0.0

        super().__init__(*args, **kwargs)

        if type(self) is DHKVManager:
            self._log_dhkv_registration()

    def _log_dhkv_registration(self):
        logger.info(
            "[DH-KV] MoGA enabled=%s  α=%.2f  β=%.2f  g_min=%.2f  adaptive_window=%s",
            self.use_moga, self.moga_alpha, self.moga_beta,
            self.moga_g_min, self.adaptive_window_enabled,
        )

    # ── Public API for updating token positions ──

    def update_token_positions(
        self,
        positions_3d: torch.Tensor,   # [S, token_per_frame, 3]
        valid_mask: torch.Tensor,     # [S, token_per_frame]
    ):
        """Update the 3D world positions of tokens for MoGA gating.

        Called once per frame after depth unprojection.

        Args:
            positions_3d: [S, token_per_frame, 3] world positions
            valid_mask:   [S, token_per_frame] boolean mask
        """
        S = positions_3d.shape[0]
        H = self.num_heads

        # Expand to [L_eff, H, T_buffer, 3] if needed
        # For now, store the latest positions per slot
        if self._token_positions_3d is None:
            self._token_positions_3d = torch.zeros(
                self._L_eff, H, self.reserved_buffer_token_size, 3,
                dtype=torch.float32, device=positions_3d.device,
            )

        # Place positions at the current offset
        for slot_idx in range(self._L_eff):
            offset = self._offset_hot[slot_idx]
            new_start = offset - S * H if offset >= S * H else 0  # simplified
            # Store per-slot
            for s in range(min(S, self.reserved_buffer_token_size // H)):
                for h in range(H):
                    idx = offset + s * H + h
                    if idx < self.reserved_buffer_token_size:
                        self._token_positions_3d[slot_idx, h, idx, :] = positions_3d[s, :, :].mean(dim=0)

        # Estimate vehicle state from positions
        if positions_3d.numel() > 0:
            # Use first slot for vehicle estimation
            mean_pos = positions_3d[positions_3d.abs().sum(dim=-1) > 0].mean(dim=(0, 1))
            if mean_pos.numel() >= 3:
                self._vehicle_theta_cam = float(torch.atan2(mean_pos[0], mean_pos[2]).item())
                self._vehicle_v = float(torch.norm(mean_pos).item())

    def get_adaptive_window_size(self) -> int:
        """Return the current adaptive window size in frames."""
        if not self.adaptive_window_enabled:
            return self.recent_size  # fallback to configured window
        return _adaptive_window(self._vehicle_v)

    # ── Override: apply MoGA gate to scores ──

    def _update_scores(self, scores: Optional[torch.Tensor] = None,
                       slot_idx: int = 0):
        """Override: compute H2O scores then apply MoGA gate."""
        if scores is None:
            return

        # Call parent H2O score update
        super()._update_scores(scores, slot_idx)

        if not self.use_moga:
            return

        # Apply MoGA gate to the accumulated scores
        T_live = self._offset_hot[slot_idx]
        if T_live == 0:
            return

        H = self.num_heads

        # Get token positions for this slot
        if self._token_positions_3d is not None and T_live > 0:
            pos = self._token_positions_3d[slot_idx, :, :T_live, :]  # [H, T, 3]

            # Compute per-token angles and distances
            theta_i = torch.atan2(pos[..., 0], pos[..., 2])  # [H, T]
            d_i = torch.norm(pos, dim=-1)                     # [H, T]

            # Compute gate
            gate = _moga_gate_vectorized(
                theta_i, d_i,
                theta_cam=self._vehicle_theta_cam,
                v=self._vehicle_v,
                omega=self._vehicle_omega,
                alpha=self.moga_alpha,
                beta=self.moga_beta,
                g_min=self.moga_g_min,
            )  # [H, T]

            # Apply gate to scores
            scores_hot = self._scores_hot[slot_idx]  # [H, buffer_T]
            scores_hot[:, :T_live].mul_(gate)

    def reset(self):
        super().reset()
        self._token_positions_3d = None
        self._vehicle_v = 0.0
        self._vehicle_omega = 0.0
        self._vehicle_theta_cam = 0.0
