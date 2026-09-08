"""
CompactMerger: wraps VoxelKVMerger for 16× head-pooled storage.
- Insert: pools H heads → stores L rows/voxel (not L*H)
- Retrieve: fetches L rows → broadcasts to L*H
- Parent code sees standard [L*H, T, D] interface, never knows about pooling.
"""
import torch
import logging
from typing import Dict

logger = logging.getLogger(__name__)


class CompactMerger:
    """Wrapper around VoxelKVMerger that stores head-pooled K,V.

    External interface:  [L*num_heads, T, D]  (same as parent expects)
    Internal storage:    [L, T, D]            (16× fewer rows)
    """

    def __init__(self, num_layers: int, num_heads: int, **merger_kwargs):
        self._L = num_layers
        self._H = num_heads
        self._LH = num_layers * num_heads

        # Internal merger: stores only L "heads" (one per layer)
        internal_kwargs = merger_kwargs.copy()
        internal_kwargs["num_heads"] = num_layers  # 24, not 384
        from stac.merger import VoxelKVMerger
        self._merger = VoxelKVMerger(**internal_kwargs)

        # Forward compat attributes
        self.num_heads = self._LH
        self.head_dim = internal_kwargs.get("head_dim", 64)

    def insert_and_merge(self, K_new, V_new, S_new, I_new, VX_new,
                         num_voxels=4096, mode="default"):
        """
        Insert head-pooled K,V into internal merger.

        Input:  [L*H, T_d, D] or [H, T_d, D]
        Stored: [L, T_d, D]  (mean across heads)
        S,I,VX: use head-0 values per layer (duplicated to H)
        """
        L, H = self._L, self._H

        # Handle both [L*H, T_d, D] and [H, T_d, D] formats
        if K_new.dim() == 2:
            K_new = K_new.unsqueeze(0)
            V_new = V_new.unsqueeze(0)
            S_new = S_new.unsqueeze(0)
            I_new = I_new.unsqueeze(0)
            VX_new = VX_new.unsqueeze(0)

        E, T_d, D = K_new.shape

        if E == L * H:
            # Standard format: reshape to [L, H, T_d, D], pool to [L, T_d, D]
            K_compact = K_new.view(L, H, T_d, D).mean(dim=1)  # [L, T_d, D]
            V_compact = V_new.view(L, H, T_d, D).mean(dim=1)
            S_compact = S_new.view(L, H, T_d).mean(dim=1)     # [L, T_d]
            I_compact = I_new.view(L, H, T_d)[:, 0, :]         # [L, T_d]
            VX_compact = VX_new.view(L, H, T_d)[:, 0, :]       # [L, T_d]
        elif E == L:
            # Already head-pooled
            K_compact = K_new
            V_compact = V_new
            S_compact = S_new
            I_compact = I_new
            VX_compact = VX_new
        else:
            # Unknown format — pass through
            return self._merger.insert_and_merge(
                K_new=K_new, V_new=V_new, S_new=S_new,
                I_new=I_new, VX_new=VX_new,
                num_voxels=num_voxels, mode=mode)

        return self._merger.insert_and_merge(
            K_new=K_compact, V_new=V_compact, S_new=S_compact,
            I_new=I_compact, VX_new=VX_compact,
            num_voxels=num_voxels, mode=mode)

    def retrieve(self, voxel_ids, return_buf=False, retrieve_size=-1):
        """
        Retrieve head-pooled K,V and broadcast to H heads.

        Internal: [L, Tm, D]  →  Output: [L*H, Tm, D]
        """
        L, H, LH = self._L, self._H, self._LH

        K_ret, V_ret, M_ret, B_ret = self._merger.retrieve(
            voxel_ids=voxel_ids, return_buf=return_buf,
            retrieve_size=retrieve_size)

        if K_ret.shape[1] == 0:
            return K_ret, V_ret, M_ret, B_ret

        # Broadcast from [L, Tm, D] to [L*H, Tm, D]
        Tm = K_ret.shape[1]
        D = K_ret.shape[2]

        K_broadcast = K_ret.unsqueeze(1).expand(-1, H, -1, -1).reshape(LH, Tm, D)
        V_broadcast = V_ret.unsqueeze(1).expand(-1, H, -1, -1).reshape(LH, Tm, D)

        # Broadcast bias — repeat L→L*H
        if B_ret.numel() > 0:
            B_broadcast = B_ret.unsqueeze(1).expand(-1, H, -1).reshape(LH, Tm)
        else:
            B_broadcast = B_ret

        # M_ret: repeat per-layer
        if M_ret.numel() > 0:
            M_broadcast = M_ret.unsqueeze(1).expand(-1, H, -1).reshape(LH, Tm)
        else:
            M_broadcast = M_ret

        return K_broadcast, V_broadcast, M_broadcast, B_broadcast

    # ── Pass-through methods ──
    def update_voxel_zones(self, voxel_zones):
        if hasattr(self._merger, 'update_voxel_zones'):
            self._merger.update_voxel_zones(voxel_zones)

    def take_diagnostics(self):
        if hasattr(self._merger, 'take_diagnostics'):
            return self._merger.take_diagnostics()
        return []

    def info(self) -> Dict:
        if hasattr(self._merger, 'info'):
            return self._merger.info()
        return {}

    def reset(self):
        if hasattr(self._merger, 'reset'):
            self._merger.reset()

    def __getattr__(self, name):
        """Forward all other attributes to internal merger."""
        return getattr(self._merger, name)
