"""
KGTCVoxelKV — KGTC + MLA streaming KV manager (drop-in for VoCacheVoxelKV).

Replaces the growing voxel pool with a BOUNDED KGTC macro-voxel cache:
  - MLA head-pooling (CompactMerger logic) is kept inside KGTCStore.
  - Dropped tokens are routed by a kinematic gate and absorbed into macro-voxels
    via geodesic (feature x kinematic) affinity + SparseMax + capacity-gate EMA.
  - Retrieval returns the (bounded) macro-voxel K/V, so attention sees a
    constant-size persistent memory -> voxel memory is constant (a few MB) no
    matter how long the stream runs.

Usage in stream_session.py (mode='vocache', use_kgtc=True):
    from stac.kgtc_kv import KGTCVoxelKV
    kv_cls = KGTCVoxelKV if is_kgtc else VoCacheVoxelKV
"""
import logging
import numpy as np
import torch

from .vocache_kv import VoCacheVoxelKV
from .kgtc_store import KGTCStore

logger = logging.getLogger(__name__)


class KGTCVoxelKV(VoCacheVoxelKV):
    def __init__(self, *args,
                 use_kgtc: bool = True,
                 kgtc_n_macro: int = 1024,
                 kgtc_gamma: float = 3.0,
                 kgtc_threshold: float = 0.1,
                 kgtc_alpha_scale: float = 0.3,
                 kgtc_gz: float = 2.0,
                 kgtc_gf: float = 3.0,
                 kgtc_z_ref: float = 0.6,
                 kgtc_f7_ref: float = 0.15,
                 **kwargs):
        self.use_kgtc = bool(use_kgtc)
        self.kgtc_n_macro = int(kgtc_n_macro)
        self.kgtc_gamma = float(kgtc_gamma)
        self.kgtc_threshold = float(kgtc_threshold)
        self.kgtc_alpha_scale = float(kgtc_alpha_scale)
        self.kgtc_gz = float(kgtc_gz)
        self.kgtc_gf = float(kgtc_gf)
        self.kgtc_z_ref = float(kgtc_z_ref)
        self.kgtc_f7_ref = float(kgtc_f7_ref)
        # force position tracking so KGTC gets real kinematics
        kwargs.setdefault('use_moga', True)
        kwargs.setdefault('moga_score_gate', False)
        kwargs.setdefault('moga_admit', False)
        kwargs.setdefault('use_compact_kv', True)
        super().__init__(*args, **kwargs)
        self._kgtc_store_built = False
        self._build_kgtc_store()

    def _build_kgtc_store(self):
        if self.use_kgtc and hasattr(self, '_vk_conf'):
            conf = dict(self._vk_conf)
            conf.pop('num_heads', None)
            self._vstore = KGTCStore(
                num_layers=self._L_eff, num_heads=self.num_heads,
                head_dim=self.head_dim,
                n_macro=self.kgtc_n_macro,
                gamma=self.kgtc_gamma,
                threshold=self.kgtc_threshold,
                alpha_scale=self.kgtc_alpha_scale,
                gz=self.kgtc_gz, gf=self.kgtc_gf,
                z_ref=self.kgtc_z_ref, f7_ref=self.kgtc_f7_ref,
                dtype=self.dtype, device=self.device)
            self._kgtc_store_built = True
            logger.info("[KGTC] macro-voxel store: Bm=%d x L=%d x D=%d "
                        "= %.2f MB (constant)",
                        self.kgtc_n_macro, self._L_eff, self.head_dim,
                        self._vstore.memory()['buffer_alloc_mem'])

    def reset(self):
        super().reset()
        self._build_kgtc_store()

    # ── KGTC drop path: reuse VoCache extraction + real kinematics ─────────
    def append_positions(self, new_positions: torch.Tensor, new_pos_mask: torch.Tensor):
        """Lightweight position tracking — NO BinaryVoxel table (KGTC doesn't
        need spatial voxels, and we must not grow the voxel hash with frames).
        Keeps _token_positions_3d (for kinematics) and the T==TV invariant.
        """
        self._append_called = getattr(self, '_append_called', 0) + 1
        # store positions (aligned with hot cache)
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
                pos = new_positions[:n].to(self.device, dtype=torch.float32)
                self._token_positions_3d[:, :, old_T:end, :] = (
                    pos.view(1, 1, n, 3).expand(self._L_eff, self.num_heads, n, 3))
        # admit all tokens (VX=0) and keep the T == TV invariant
        total = min(old_T + T_new, self.reserved_buffer_token_size)
        n = total - old_T
        if n > 0:
            self._token_voxel_indices[:, :, old_T:total] = 0
        for l in range(self._L_eff):
            self._offset_voxel[l] = total

    def _pool_drops_impl(self, keep_idx_lh):
        if not self.use_kgtc:
            return super()._pool_drops_impl(keep_idx_lh)
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

        K_drop = Kh[drop_mask_all].view(L, H, T_d, D)
        V_drop = Vh[drop_mask_all].view(L, H, T_d, D)
        S_drop = Sh[drop_mask_all].view(L, H, T_d).float()
        I_drop = Ih[drop_mask_all].view(L, H, T_d)
        VX_drop = VXh[drop_mask_all].view(L, H, T_d).to(torch.int32)

        # DH-KV admission (optional): behind/far tokens -> VX=-1 (skipped)
        if self.use_moga and self.moga_admit and self._token_positions_3d is not None:
            admit = self._compute_admit_mask(keep_idx_lh, T, T_d, L, H)
            if admit is not None:
                VX_drop = torch.where(admit.view(L, H, T_d), VX_drop,
                                      torch.full_like(VX_drop, -1))

        # real kinematics for KGTC routing / barrier
        pos_drop = kin_drop = None
        if self._token_positions_3d is not None:
            pos_all = self._token_positions_3d[:, :, :T, :]
            pos_drop = pos_all[drop_mask_all].view(L, H, T_d, 3)[0, 0]  # [T_d,3]
            R = self._cam_R.to(pos_drop.device)
            TT = self._cam_t.to(pos_drop.device)
            C = -R.T @ TT
            p_rel = pos_drop - C.view(1, 3)
            p_cam = torch.einsum('ij,tj->ti', R, p_rel)
            z_m = p_cam[:, 2] * 30.0
            z_n = (z_m.clamp(0, 120) / 120.0).float()
            f7 = torch.full_like(z_n, 0.05)
            v_n = float(min(max(self._vehicle_v / 30.0, 0.0), 1.0))
            o_n = float(((self._vehicle_omega + np.pi) / (2 * np.pi)) % 1.0)
            kin_drop = torch.stack([torch.full_like(z_n, v_n),
                                    torch.full_like(z_n, o_n),
                                    z_n, f7], dim=-1)  # [T_d,4]

        num_voxels = self._vm.get_voxel_keys().shape[0]
        if self._vk_conf["backend"] == "cuda":
            mode = "cuda"
        else:
            mode = "default"
        self._vstore.insert_and_merge(
            K_new=K_drop.view(L * H, T_d, D),
            V_new=V_drop.view(L * H, T_d, D),
            S_new=S_drop.view(L * H, T_d),
            I_new=I_drop.view(L * H, T_d),
            VX_new=VX_drop.view(L * H, T_d),
            num_voxels=num_voxels, mode=mode,
            positions=pos_drop, kinematics=kin_drop)
        self._frame_count += 1

    # ── retrieval: macro-voxel K/V straight into the per-layer cache ───────
    def retrieve_kv_parallel(self, **kwargs):
        if not self.use_kgtc:
            return super().retrieve_kv_parallel(**kwargs)
        K_ret, V_ret, M_ret, B_ret = self._vstore.retrieve(
            voxel_ids=None, return_buf=kwargs.get('return_buf', False),
            retrieve_size=self.retrieval_token_size)
        H = self.num_heads
        for slot_idx in range(self._L_eff):
            if not self._voxelize_list[slot_idx]:
                continue
            self._clear_retrieval_cache(slot_idx)
            Tm = K_ret.shape[1]
            if self.retrieval_token_size >= 0:
                Tm = min(Tm, self.retrieval_token_size)
            if Tm == 0:
                continue
            self.key_cache_retrieval[slot_idx] = (
                K_ret[slot_idx * H:(slot_idx + 1) * H, :Tm, :].contiguous())
            self.value_cache_retrieval[slot_idx] = (
                V_ret[slot_idx * H:(slot_idx + 1) * H, :Tm, :].contiguous())
            self._bias_retrieval[slot_idx] = (
                B_ret[slot_idx * H:(slot_idx + 1) * H, :Tm].contiguous())
            self._offset_retrieval[slot_idx] = Tm
        del K_ret, V_ret, M_ret, B_ret
        return 0, 0
