"""
CompactMerger: wraps VoxelKVMerger for 16× head-pooled storage.
- Insert: pools H heads → stores L rows/voxel (not L*H)
- Retrieve: fetches L rows → broadcasts to L*H
- Parent code sees standard [L*H, T, D] interface, never knows about pooling.
"""
import math
import os
import torch
import logging
from typing import Dict

logger = logging.getLogger(__name__)


def _time_basis(t: float, n_harm: int, period: float) -> torch.Tensor:
    """ψ(t) = [1, cos(2πt/P), sin(2πt/P), cos(2πt/2P), ...] with 1+2*n_harm terms.

    Column 0 is the constant term, so a fitted coefficient matrix's row 0 IS the
    canonical / time-invariant component and the remaining rows are the temporal
    deformation. n_harm=0 therefore degenerates to the current time-blind store.
    """
    cols = [1.0]
    for k in range(1, int(n_harm) + 1):
        w = 2.0 * math.pi * t / (period * k)
        cols.append(math.cos(w))
        cols.append(math.sin(w))
    return torch.tensor(cols, dtype=torch.float32)


class CompactMerger:
    """Wrapper around VoxelKVMerger that stores head-pooled K,V.

    External interface:  [L*num_heads, T, D]  (same as parent expects)
    Internal storage:    [L, T, D]            (16× fewer rows)

    --- Technique C (experimental, env-gated) ---------------------------------
    Time-deformation of the retrieved store. The store is otherwise time-blind:
    content written at frame t is returned unchanged forever. With
    VOCACHE_DEFORM_RANK>0 the retrieved rows get a time-conditioned correction

        K_read(t) = K_stored + W_Kᵀ ψ(t)          (same for V)

    where W_K ∈ [L, B, D] is fitted online (LMS on the per-frame per-layer mean
    of the compacted features) and B = 1+2·rank. Row 0 of W is the canonical term.

    Scope note: this is applied per *internal row* (= per layer, all retrieved
    tokens alike), because VoxelKVMerger.retrieve re-sorts its pivots by weight
    and compacts them, so there is no reliable row→voxel-id mapping to correct
    per voxel. It therefore probes "does time-conditioning the store help at all",
    not the full per-voxel version (that one lives in VoCache/core/kgtc_deform.py).
    Env-gated so it can be A/B'd without threading a flag through run_model.
    ---------------------------------------------------------------------------
    """

    def __init__(self, num_layers: int, num_heads: int, **merger_kwargs):
        self._L = num_layers
        self._H = num_heads
        self._LH = num_layers * num_heads

        # ── technique C config (env-gated; popped so VoxelKVMerger never sees it)
        # VOCACHE_DEFORM_RANK = R >= 1 : B = 1+2R time-basis terms (canonical + R harmonics)
        # VOCACHE_DEFORM_RANK = -1     : B = 1, i.e. a TIME-INVARIANT per-layer offset.
        #                                This is the control that separates "the time basis
        #                                helps" from "any per-layer re-centering helps".
        # VOCACHE_DEFORM_RANK = 0      : off (baseline)
        self._deform_rank = int(os.environ.get('VOCACHE_DEFORM_RANK', '0'))
        self._deform_period = float(os.environ.get('VOCACHE_DEFORM_PERIOD', '50'))
        self._deform_lr = float(os.environ.get('VOCACHE_DEFORM_LR', '0.2'))
        merger_kwargs.pop('deform_rank', None)
        merger_kwargs.pop('deform_period', None)
        merger_kwargs.pop('deform_lr', None)
        self._deform_on = self._deform_rank != 0
        self._n_harm = max(0, self._deform_rank)
        self._B = 1 + 2 * self._n_harm
        self._W_K = None
        self._W_V = None
        self._t = 0.0

        # Internal merger: stores only L "heads" (one per layer)
        internal_kwargs = merger_kwargs.copy()
        internal_kwargs["num_heads"] = num_layers  # 24, not 384
        from stac.merger import VoxelKVMerger
        self._merger = VoxelKVMerger(**internal_kwargs)

        # Forward compat attributes
        self.num_heads = self._LH
        self.head_dim = internal_kwargs.get("head_dim", 64)

    # ── technique C: time-conditioned store ───────────────────────────────
    def set_time(self, t: float):
        """Tell the store which frame it is (called once per frame, causally)."""
        self._t = float(t)

    def _psi(self, dtype, device):
        return _time_basis(self._t, self._n_harm, self._deform_period).to(
            dtype=dtype, device=device)

    def _fit_deform(self, K_compact, V_compact):
        """LMS on the per-layer mean of the compacted features (no backprop)."""
        L, D = K_compact.shape[0], K_compact.shape[2]
        if self._W_K is None:
            self._W_K = torch.zeros(L, self._B, D, dtype=torch.float32,
                                    device=K_compact.device)
            self._W_V = torch.zeros(L, self._B, D, dtype=torch.float32,
                                    device=K_compact.device)
        psi = self._psi(torch.float32, K_compact.device)          # [B]
        for W, X in ((self._W_K, K_compact), (self._W_V, V_compact)):
            tgt = X.detach().float().mean(dim=1)                  # [L,D]
            pred = torch.einsum('b,lbd->ld', psi, W)              # [L,D]
            err = pred - tgt
            W -= self._deform_lr * psi.view(1, self._B, 1) * err.unsqueeze(1)


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

        if self._deform_on:
            self._fit_deform(K_compact, V_compact)

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

        # ── technique C: time-conditioned read (see class docstring) ──
        if self._deform_on and self._W_K is not None:
            psi = self._psi(K_ret.dtype, K_ret.device)
            dK = torch.einsum('b,lbd->ld', psi, self._W_K.to(K_ret.dtype))
            dV = torch.einsum('b,lbd->ld', psi, self._W_V.to(V_ret.dtype))
            if dK.shape[0] == K_ret.shape[0]:
                K_ret = K_ret + dK.unsqueeze(1)
                V_ret = V_ret + dV.unsqueeze(1)

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
