"""
VoCacheVoxelKV — Unified DH-KV + VoCACHE STAC Integration (Optimized)
======================================================================
DH-KV's MoGA motion gating (temporal KV-cache) combined with a 16× head-pooled
voxel pool (CompactMerger) in a single drop-in replacement for STAC's STACVoxelKV.

Usage in stream_session.py:
    from stac.vocache_kv import VoCacheVoxelKV
    kv_kwargs = self.register_kv_mgr(mode, images, VoCacheVoxelKV, **kv_kwargs)
"""

import logging, os, sys
from pathlib import Path
from typing import Optional, Dict, Tuple
import numpy as np
import torch
import torch.nn.functional as F

from .stac_voxel import STACVoxelKV
from .merger import VoxelKVMerger
from .compact_merger import CompactMerger
from causalvggt.utils.rotation import quat_to_mat

logger = logging.getLogger(__name__)

# Module-level diagnostic counter for MoGA gating (readable from benchmarks)
_MOGA_FIRED_TOTAL = {"n": 0}
# Module-level diagnostic: spatial admission rejected tokens
_MOGA_ADMIT_TOTAL = {"rejected": 0, "total": 0, "calls": 0}
# Diagnostic: why admission doesn't fire
_MOGA_DIAG = {}


# ═══════════════════════════════════════════════════════════════════
# MoGA Heuristics (inlined from dhkv_kv.py)
# ═══════════════════════════════════════════════════════════════════

def _moga_gate_vectorized(theta_i, d_i, theta_cam, v, omega,
                           alpha=2.0, beta=0.3, g_min=0.25,
                           d_ref=150.0):
    """Gentle MoGA gate. Re-ranks tokens (forward keeps, behind/far down-
    weights) but never crushes scores below g_min, so it nudges WHICH tokens
    stay hot without destroying the K/V they carry.
    """
    theta_i = theta_i.float()
    d_i = d_i.float()
    cos_diff = torch.cos(theta_i - theta_cam)
    mask_forward = cos_diff > 0
    gf = torch.full_like(cos_diff, g_min)
    if mask_forward.any():
        gf[mask_forward] = g_min + (1.0 - g_min) * (cos_diff[mask_forward] ** alpha)
    # gentle distance term: even far tokens keep >= g_min weight
    gs = torch.full_like(d_i, 1.0)
    if mask_forward.any():
        # ahead: mild decay from 1.0 down to ~0.6 at d_ref
        gs[mask_forward] = g_min + (1.0 - g_min) / (1.0 + d_i[mask_forward] / d_ref)
    else:
        # behind: keep at a moderate floor (not fully crushed)
        gs = torch.full_like(d_i, max(g_min, 0.5))
    return gf * gs


# ═══════════════════════════════════════════════════════════════════
# VoCacheVoxelKV (Optimized)
# ═══════════════════════════════════════════════════════════════════

class VoCacheVoxelKV(STACVoxelKV):
    """DH-KV + VoCache Voxel KV Manager.

    DH-KV: MoGA gating applied to H2O scores in _update_scores
    VoCache: 16× head-pooled voxel pool (CompactMerger)

    Extra params:
        use_moga: toggle MoGA gating
        moga_alpha, moga_beta, moga_g_min: MoGA gate hyper-params
        enable_ctrv_retrieval: use CTRV extrapolated frustum for voxel retrieval
        use_compact_kv: use 16× head-pooled CompactMerger storage
    """

    def __init__(self, *args,
                 use_moga: bool = True,
                 enable_ctrv_retrieval: bool = True,
                 use_compact_kv: bool = True,
                 moga_alpha: float = 2.0,
                 moga_beta: float = 0.3,
                 moga_g_min: float = 0.25,
                 moga_score_gate: bool = False,
                 moga_admit: bool = True,
                 moga_admit_dist: float = 200.0,
                 **kwargs):
        self.use_moga = bool(use_moga)
        self.moga_alpha = float(moga_alpha)
        self.moga_beta = float(moga_beta)
        self.moga_g_min = float(moga_g_min)
        self.moga_score_gate = bool(moga_score_gate)
        self.moga_admit = bool(moga_admit)
        self.moga_admit_dist = float(moga_admit_dist)
        self._enable_ctrv_retrieval = bool(enable_ctrv_retrieval)
        self._use_compact_kv = bool(use_compact_kv)

        self._vehicle_v = 0.0
        self._vehicle_omega = 0.0
        self._vehicle_theta_cam = 0.0
        self._vehicle_positions = []
        self._cam_R = torch.eye(3, dtype=torch.float32)   # world->cam rotation
        self._cam_t = torch.zeros(3, dtype=torch.float32)  # cam pos in world
        self._prev_cam_t = None
        # 16× head-pooled storage: CompactMerger wraps merger with L eff heads
        if self._use_compact_kv and hasattr(self, '_vk_conf'):
            conf = dict(self._vk_conf)
            conf.pop('num_heads', None)  # CompactMerger computes its own
            self._vstore = CompactMerger(
                num_layers=self._L_eff, num_heads=self.num_heads,
                **conf)  # CTRV history: list of (t, 4x4 pose)
        self._token_positions_3d = None
        self._frame_count = 0
        self._prev_theta_cam = 0.0
        self._last_gate = None

        super().__init__(*args, **kwargs)

        if type(self) is VoCacheVoxelKV:
            logger.info("[VoCache] DH-KV=%s α=%.2f β=%.2f g_min=%.2f ctrv=%s compact=%s",
                       self.use_moga, self.moga_alpha, self.moga_beta,
                       self.moga_g_min, self._enable_ctrv_retrieval,
                       self._use_compact_kv)

    # ═══════════════════════════════════════════════════════════
    # DH-KV: MoGA motion gating (temporal)
    # ═══════════════════════════════════════════════════════════

    def _update_scores(self, scores=None, slot_idx=0):
        super()._update_scores(scores, slot_idx)
        if not self.use_moga or not self.moga_score_gate or self._token_positions_3d is None:
            return
        T_live = self._offset_hot[slot_idx]
        if T_live == 0:
            return
        pos = self._token_positions_3d[slot_idx, :, :T_live, :]  # world frame
        if pos.abs().sum() == 0:
            return
        # Transform world -> ego camera frame.
        # Extrinsic convention: p_cam = R @ p_world + T, so the camera center
        # in world is C = -R^T @ T, and p_cam = R @ (p_world - C).
        R = self._cam_R                                  # [3,3] world->cam rotation
        T = self._cam_t                                  # [3]   extrinsic translation
        C = -R.T @ T                                     # camera center in world
        p_rel = pos - C.view(1, 1, 3)                    # [H,T,3]
        p_cam = torch.einsum('ij,htj->hti', R, p_rel)    # [H,T,3]
        theta_i = torch.atan2(p_cam[..., 0], p_cam[..., 2])
        # world units -> meters (depth convention: raw × 30 = meters)
        d_i = torch.norm(p_cam, dim=-1) * 30.0
        # theta_cam = 0 because p_cam is already in the ego frame (bearing
        # relative to camera forward). v, omega still come from real poses.
        gate_now = _moga_gate_vectorized(
            theta_i, d_i, 0.0, self._vehicle_v, self._vehicle_omega,
            self.moga_alpha, self.moga_beta, self.moga_g_min)  # [H, T_live]
        # Telescoping: multiply by g_now/g_prev so the TOTAL multiplier over
        # all calls equals g_now (not g^k). This is exactly s^DH = s^H2O · g
        # with no compounding across forward passes.
        if self._last_gate is not None:
            # pad previous gate with 1.0 for newly-appended tokens
            prev = self._last_gate
            if prev.shape[-1] < T_live:
                pad = torch.ones_like(gate_now[:, :T_live - prev.shape[-1]])
                prev = torch.cat([prev, pad], dim=-1)
            if prev.shape == gate_now.shape:
                gate_eff = gate_now / prev.clamp(min=1e-6)
            else:
                gate_eff = gate_now
        else:
            gate_eff = gate_now
        self._scores_hot[slot_idx, :, :T_live].mul_(gate_eff.to(self._scores_hot.dtype))
        self._last_gate = gate_now.detach().clone()
        _MOGA_FIRED_TOTAL["n"] += 1

    def _apply_keep_and_compact_parallel(self, keep_idx_lh: torch.Tensor) -> bool:
        """Compact token positions (and the telescoping gate) in sync with the
        hot cache so MoGA's per-token positions stay aligned after pruning.

        CRITICAL: compact AFTER super(), because super()._apply... calls
        _pool_drops_impl which reads positions at the PRE-compact T offset.
        """
        L = self._L_eff
        has_pos = self.use_moga and self._token_positions_3d is not None
        old_pos = self._token_positions_3d if has_pos else None
        old_gate = self._last_gate if (has_pos and self._last_gate is not None) else None
        out = super()._apply_keep_and_compact_parallel(keep_idx_lh)
        if has_pos and old_pos is not None:
            Tn = int(self._offset_hot[0])  # post-compact
            T_old = old_pos.shape[2]
            if Tn > 0 and Tn < T_old:
                k = keep_idx_lh[:L, :, :Tn].unsqueeze(-1).expand(-1, -1, -1, 3)
                comp = torch.gather(old_pos[:L, :, :T_old, :], 2, k)  # [L,H,Tn,3]
            else:
                comp = old_pos[:L, :, :Tn, :]
            full = torch.zeros(L, self.num_heads, self.reserved_buffer_token_size, 3,
                               device=self.device, dtype=torch.float32)
            full[:, :, :Tn, :] = comp
            self._token_positions_3d = full
            if old_gate is not None and old_gate.ndim == 2:
                # _last_gate is [H, T_old]
                gk = keep_idx_lh[0, :, :Tn] if Tn < old_gate.shape[1] else None
                if gk is not None:
                    self._last_gate = torch.gather(old_gate[:self.num_heads, :T_old], 1, gk)
        return out

    def update_camera_pose(self, pose_enc):
        """Feed the REAL ego camera pose from StreamVGGT's pose head.

        pose_enc: [B, S, 9] or [S, 9] — (T[3], quat[4], fov[2]).
        Extracts world->cam rotation R, cam translation t, and estimates
        ego v, omega, theta_cam from consecutive camera translations.
        """
        pe = pose_enc.detach()
        if pe.dim() == 3:
            pe = pe[0]                    # [S, 9]
        pe = pe.float().to(self.device)
        if pe.numel() == 0:
            return
        last = pe[-1]                     # [9]
        T = last[:3]                      # camera position in world
        quat = last[3:7]                  # XYZW scalar-last
        R = quat_to_mat(quat.reshape(1, 4))[0]   # [3,3] world->cam rotation
        self._cam_R = R.contiguous()
        self._cam_t = T.contiguous()

        # Ego velocity from consecutive camera positions (world units; ×30 -> m)
        if self._prev_cam_t is not None and pe.shape[0] >= 2:
            prev_T = self._prev_cam_t
            disp = T - prev_T
            self._vehicle_v = float(torch.norm(disp).item()) * 30.0 * 30.0  # m/s @30fps
            # heading from translation direction (ego frame: forward = +z of cam)
            heading = float(torch.atan2(disp[0].item(), disp[2].item()))
            dtheta = heading - self._prev_theta_cam
            dtheta = (dtheta + np.pi) % (2 * np.pi) - np.pi
            self._vehicle_omega = dtheta
            self._vehicle_theta_cam = heading
        else:
            self._vehicle_v = 0.0
            self._vehicle_omega = 0.0
            self._vehicle_theta_cam = float(torch.atan2(T[0].item(), T[2].item()))
        self._prev_cam_t = T.detach().clone()
        self._prev_theta_cam = self._vehicle_theta_cam

    def append_positions(self, new_positions: torch.Tensor, new_pos_mask: torch.Tensor):
        """Store per-token 3D world positions (MoGA needs per-token geometry)."""
        self._append_called = getattr(self, '_append_called', 0) + 1
        if self.use_moga:
            if self._token_positions_3d is None:
                self._token_positions_3d = torch.zeros(
                    self._L_eff, self.num_heads, self.reserved_buffer_token_size, 3,
                    dtype=torch.float32, device=self.device)
            old_T = int(self._offset_voxel[0])
            T_new = new_positions.shape[0]
            if T_new > 0:
                end = min(old_T + T_new, self.reserved_buffer_token_size)
                n = end - old_T
                if n > 0:
                    pos = new_positions[:n].to(self.device, dtype=torch.float32)  # [n,3]
                    self._token_positions_3d[:, :, old_T:end, :] = (
                        pos.view(1, 1, n, 3).expand(self._L_eff, self.num_heads, n, 3))
                    # DIAGNOSTIC
                    if int(os.environ.get('MOGA_DEBUG', '0')):
                        logger.info("[VoCache] stored %d positions at offset %d", n, old_T)
        super().append_positions(new_positions, new_pos_mask)

    # ═══════════════════════════════════════════════════════════
    # VoCACHE: CTRV Frustum Prediction for Retrieval
    # ═══════════════════════════════════════════════════════════

    def _ctrv_extrapolate_pose(self, dt=1.0/30.0):
        """Extrapolate next camera pose using CTRV motion model."""
        if len(self._vehicle_positions) < 1:
            return None
        _, last_pose = self._vehicle_positions[-1]
        if isinstance(last_pose, torch.Tensor):
            last_pose = last_pose.cpu().numpy() if last_pose.is_cuda else last_pose
        last_pose = np.array(last_pose)
        
        v = self._vehicle_v; omega = self._vehicle_omega; theta = self._vehicle_theta_cam
        
        # CTRV extrapolation
        if abs(omega) < 1e-6:
            dx = v * np.cos(theta) * dt
            dz = v * np.sin(theta) * dt
            dtheta = 0.0
        else:
            dx = (v / omega) * (np.sin(theta + omega * dt) - np.sin(theta))
            dz = (v / omega) * (np.cos(theta) - np.cos(theta + omega * dt))
            dtheta = omega * dt
        
        next_pose = last_pose.copy()
        next_pose[:3, 3] += np.array([dx, 0.0, dz])
        # Rotate around Y axis
        cy, sy = np.cos(dtheta), np.sin(dtheta)
        R_delta = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
        next_pose[:3, :3] = R_delta @ last_pose[:3, :3]
        return torch.from_numpy(next_pose).float().to(self.device)

    def _query_frustum_voxels(self, pose, dist_thres=0.1):
        """Query voxels near the frustum of predicted camera pose."""
        if not hasattr(self, '_vm') or self._vm is None:
            return None
        # Get camera center from pose
        cam_center = pose[:3, 3] if isinstance(pose, torch.Tensor) else pose[:3, 3]
        if isinstance(cam_center, np.ndarray):
            cam_center = torch.from_numpy(cam_center).float().to(self.device)
        cam_center = cam_center.reshape(1, 3)
        
        # Query voxels within a radius (frustum approximation)
        radius = 20.0  # meters — broad enough for forward visibility
        neighbor_ids, _, found = self._vm.neighbors(
            positions=cam_center,
            radius_m=radius,
            include_self=False
        )
        if found and neighbor_ids.numel() > 0:
            return neighbor_ids[neighbor_ids >= 0].unique()
        return None

    def _search_recent_neighbors(self, slot_idx=0, dist_thres=0.1):
        """CTRV-aware override: predict next-frame frustum voxels for retrieval."""
        if not self._enable_ctrv_retrieval:
            return super()._search_recent_neighbors(slot_idx, dist_thres)
        
        # Try CTRV prediction first
        next_pose = self._ctrv_extrapolate_pose()
        if next_pose is not None:
            frustum_voxels = self._query_frustum_voxels(next_pose, dist_thres)
            if frustum_voxels is not None and frustum_voxels.numel() > 0:
                # Also include recent neighbor voxels for robustness
                super()._search_recent_neighbors(slot_idx, dist_thres)
                recent = self._recent_voxel_ids
                if recent is not None and recent.numel() > 0:
                    self._recent_voxel_ids = torch.cat([frustum_voxels, recent]).unique()
                    self._ctrv_predicted_voxels = frustum_voxels  # for telemetry
                    return
        
        # Fallback to parent
        super()._search_recent_neighbors(slot_idx, dist_thres)
        self._ctrv_predicted_voxels = None

    # ═══════════════════════════════════════════════════════════
    # VoCache: head-pooled drop path via CompactMerger
    # ═══════════════════════════════════════════════════════════

    def _pool_drops_vectorized_parallel(self, keep_idx_lh):
        try:
            self._pool_drops_impl(keep_idx_lh)
        except Exception as e:
            logger.warning("[VoCache] Fallback to H2O: %s", e)
            super()._pool_drops_vectorized_parallel(keep_idx_lh)

    def _pool_drops_impl(self, keep_idx_lh):
        """Drop path feeding the 16× head-pooled CompactMerger.

        Replicates base extraction, then insert_and_merge (which head-pools
        internally). When `use_moga` + `moga_admit` are enabled, tokens that
        are behind the ego (or beyond `moga_admit_dist`) are given VX=-1 so
        the merger SKIPS storing them in the voxel pool entirely — a real
        spatial memory saving (not just a score re-ranking).
        """
        L, H, D = self._L_eff, self.num_heads, self.head_dim
        T = self._offset_hot[0]
        if T == 0:
            return super()._pool_drops_vectorized_parallel(keep_idx_lh)

        keep_mask = torch.zeros(L, H, T, dtype=torch.bool, device=self.device)
        keep_mask.scatter_(2, keep_idx_lh, True)
        drop_mask_all = ~keep_mask
        T_d = int(drop_mask_all.sum(dim=2)[0, 0].item())
        if T_d == 0:
            return super()._pool_drops_vectorized_parallel(keep_idx_lh)

        Kh = self.key_cache_hot[:, :, :T, :]
        Vh = self.value_cache_hot[:, :, :T, :]
        Sh = self._scores_hot[:, :, :T]
        Ih = self._token_indices_hot[:, :, :T].to(self.device)
        VXh = self._token_voxel_indices[:, :, :T].to(self.device)

        S_drop = Sh[drop_mask_all].view(L, H, T_d).float()
        K_drop = Kh[drop_mask_all].view(L, H, T_d, D)
        V_drop = Vh[drop_mask_all].view(L, H, T_d, D)
        I_drop = Ih[drop_mask_all].view(L, H, T_d)
        VX_drop = VXh[drop_mask_all].view(L, H, T_d).to(torch.int32)

        # ── Spatial admission: skip storing behind/far tokens (real memory win) ──
        _MOGA_DIAG.update({
            'use_moga': self.use_moga,
            'moga_admit': self.moga_admit,
            'has_pos': self._token_positions_3d is not None,
            'T': T, 'T_d': T_d,
            'append_called': getattr(self, '_append_called', 0),
        })
        if self.use_moga and self.moga_admit and self._token_positions_3d is not None:
            admit = self._compute_admit_mask(keep_idx_lh, T, T_d, L, H)
            if admit is not None:
                rej = int((~admit).sum().item())
                _MOGA_ADMIT_TOTAL["rejected"] += rej
                _MOGA_ADMIT_TOTAL["total"] += int(admit.numel())
                _MOGA_ADMIT_TOTAL["calls"] += 1
                VX_drop = torch.where(admit.view(L, H, T_d),
                                      VX_drop,
                                      torch.full_like(VX_drop, -1))
        # ── Insert (CompactMerger head-pools internally) ──
        num_voxels = self._vm.get_voxel_keys().shape[0]
        if self._vk_conf["backend"] == "cuda":
            mode = "cuda"
        else:
            mode = "default"
        if hasattr(self._vm, '_voxel_zones') and self._vm._voxel_zones.numel() > 0:
            self._vstore.update_voxel_zones(self._vm._voxel_zones)
        # technique C: give the store the frame clock so it can fit/apply the
        # time deformation (no-op unless VOCACHE_DEFORM_RANK>0)
        if hasattr(self._vstore, 'set_time'):
            self._vstore.set_time(self._frame_count)
        self._vstore.insert_and_merge(
            K_new=K_drop.view(L * H, T_d, D),
            V_new=V_drop.view(L * H, T_d, D),
            S_new=S_drop.view(L * H, T_d),
            I_new=I_drop.view(L * H, T_d),
            VX_new=VX_drop.view(L * H, T_d),
            num_voxels=num_voxels, mode=mode)
        self._frame_count += 1

    def _compute_admit_mask(self, keep_idx_lh, T, T_d, L, H):
        """DH-KV admission mask [L,H,T_d]: True = store in voxel pool.
        z > 0 (forward) → admit. z ≤ 0 (behind ego) → reject (VX=-1).
        """
        try:
            if self._token_positions_3d is None:
                return None
            keep_mask = torch.zeros(L, H, T, dtype=torch.bool, device=self.device)
            keep_mask.scatter_(2, keep_idx_lh, True)
            drop_mask_all = ~keep_mask                                   # [L,H,T]
            pos_all = self._token_positions_3d[:, :, :T, :]              # [L,H,T,3]
            pos_drop = pos_all[drop_mask_all].view(L, H, T_d, 3)         # [L,H,T_d,3]
            R = self._cam_R.to(pos_drop.device)
            TT = self._cam_t.to(pos_drop.device)
            C = -R.T @ TT
            p_rel = pos_drop - C.view(1, 1, 1, 3)
            p_cam = torch.einsum('ij,lhtj->lhti', R, p_rel)              # [L,H,T_d,3]
            z = p_cam[..., 2]
            admit = z > 0
            return admit.to(torch.bool)
        except Exception as e:
            _MOGA_DIAG.update({'admit_err': str(e)})
            return None

    def _retrieve_kv(self, layer_idx: int, **kwargs):
        """Deprecated: retrieval now goes through retrieve_kv_parallel."""
        if not self._use_compact_kv:
            return super()._retrieve_kv(layer_idx, **kwargs)
        # Fallback to parallel path
        return self.retrieve_kv_parallel(**kwargs)[:2]

    def reset(self):
        super().reset()
        self._vehicle_v = 0.0
        self._vehicle_omega = 0.0
        self._vehicle_theta_cam = 0.0
        self._cam_R = torch.eye(3, dtype=torch.float32)
        self._cam_t = torch.zeros(3, dtype=torch.float32)
        self._prev_cam_t = None
        self._token_positions_3d = None
        self._frame_count = 0
        self._prev_theta_cam = 0.0
        self._last_gate = None
        self._vehicle_positions = []
        # 16× head-pooled storage: CompactMerger wraps merger with L eff heads
        if self._use_compact_kv and hasattr(self, '_vk_conf'):
            conf = dict(self._vk_conf)
            conf.pop('num_heads', None)  # CompactMerger computes its own
            self._vstore = CompactMerger(
                num_layers=self._L_eff, num_heads=self.num_heads,
                **conf)
