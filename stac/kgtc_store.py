"""
KGTCStore — KGTC-backed bounded K/V store (drop-in for CompactMerger).

MLA head-pooling (CompactMerger) + KGTC geodesic condensation:
  - Incoming dropped K/V are head-pooled [L*H,T,D] -> [L,T,D]  (MLA).
  - Compressible tokens are absorbed into a BOUNDED set of macro-voxels
    [Bm, L, D] via a joint feature-kinematic (geodesic) affinity and SparseMax,
    guarded by a kinematic penalty (deformation barrier) and an adaptive
    capacity gate (EMA).
  - The macro-voxel cache is CONSTANT size (Bm x L x D x 2 x 2 bytes) regardless
    of stream length -> replaces the growing voxel pool.

Interface matches CompactMerger / VoxelKVMerger:
    insert_and_merge(...), retrieve(...), update_voxel_zones(...),
    info(), memory(), num_heads, head_dim
"""
import logging
import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


def _geo_router(kinematics, gz=2.0, gf=3.0, z_ref=0.6, f7_ref=0.15):
    """Geometric routing prior on normalized kinematics [v,omega,z,f7].
    p = sigmoid(gz*(z - z_ref) - gf*(f7 - f7_ref));  p>=0.5 -> compressible.
    """
    z = kinematics[..., 2]
    f7 = kinematics[..., 3]
    logit = gz * (z - z_ref) - gf * (f7 - f7_ref)
    return torch.sigmoid(logit)


class KGTCStore:
    def __init__(self, num_layers: int, num_heads: int,
                 head_dim: int = 64,
                 n_macro: int = 1024,
                 gamma: float = 3.0,        # spatial penalty sharpness
                 threshold: float = 0.1,    # SparseMax threshold
                 alpha_scale: float = 0.3,  # capacity-gate EMA scale
                 gz: float = 2.0, gf: float = 3.0,
                 z_ref: float = 0.6, f7_ref: float = 0.15,
                 pos_scale: float = 4.0,    # spatial penalty length (STAC units)
                 dtype=torch.bfloat16, device='cuda',
                 **kwargs):
        self._L = int(num_layers)
        self._H = int(num_heads)
        self._LH = self._L * self._H
        self.head_dim = int(head_dim)
        self.num_heads = self._LH          # external interface expects L*H
        self.dtype = dtype
        self.device = device

        self.n_macro = int(n_macro)
        self.gamma = float(gamma)
        self.threshold = float(threshold)
        self.alpha_scale = float(alpha_scale)
        self.gz, self.gf = float(gz), float(gf)
        self.z_ref, self.f7_ref = float(z_ref), float(f7_ref)
        self.pos_scale = float(pos_scale)
        self.reset()

    def reset(self):
        Bm, L, D = self.n_macro, self._L, self.head_dim
        dev = self.device
        self.macro_K = torch.zeros(Bm, L, D, device=dev, dtype=torch.float32)
        self.macro_V = torch.zeros(Bm, L, D, device=dev, dtype=torch.float32)
        self.macro_pos = torch.zeros(Bm, 3, device=dev, dtype=torch.float32)
        self.macro_f7 = torch.zeros(Bm, 1, device=dev, dtype=torch.float32)
        self.macro_cnt = torch.zeros(Bm, device=dev, dtype=torch.float32)
        self._n_absorbed = 0
        self._n_tokens = 0
        self._n_absorb_calls = 0

    # ── insert: MLA pool + KGTC absorb ─────────────────────────────────────
    @torch.no_grad()
    def insert_and_merge(self, K_new, V_new, S_new, I_new, VX_new,
                         num_voxels=4096, mode='default', **kwargs):
        """
        K_new: [L*H, T_d, D] or [L, T_d, D]  (head-pooled by caller or here)
        kwargs may carry:
            positions:  [T_d, 3] world positions of the dropped tokens
            kinematics: [T_d, 4] = [v, omega, z_n, f7]
        """
        # --- MLA head-pooling (same as CompactMerger) ---
        L, H, D = self._L, self._H, self.head_dim
        if K_new.dim() == 2:
            K_new = K_new.unsqueeze(0); V_new = V_new.unsqueeze(0)
        E, T_d, Dk = K_new.shape
        if E == L * H:
            K_p = K_new.view(L, H, T_d, D).mean(dim=1)   # [L,T_d,D]
            V_p = V_new.view(L, H, T_d, D).mean(dim=1)
            VX_p = (VX_new.view(L, H, T_d)[0, 0] if VX_new is not None else None)  # [T_d]
        elif E == L:
            K_p, V_p = K_new, V_new
            VX_p = (VX_new[0] if VX_new is not None else None)   # [T_d]
        else:
            return 0, 0
        T_d = K_p.shape[1]
        if T_d == 0:
            return 0, 0

        positions = kwargs.get('positions')      # [T_d,3] or None
        kinematics = kwargs.get('kinematics')    # [T_d,4] or None

        # --- admission: skip rejected tokens (VX < 0), e.g. DH-KV behind-ego ---
        if VX_p is not None:
            vx = VX_p.to(self.device).reshape(T_d)
            admitted = (vx >= 0)
        else:
            admitted = torch.ones(T_d, dtype=torch.bool, device=self.device)

        # --- router: compressible vs discard ---
        if kinematics is not None and kinematics.shape[0] == T_d:
            p = _geo_router(kinematics.to(self.device).float(),
                            self.gz, self.gf, self.z_ref, self.f7_ref)
            comp = (p >= 0.5) & admitted
        else:
            comp = admitted
        T_c = int(comp.sum().item())
        self._n_tokens += T_d
        if T_c == 0:
            return 0, 0

        K_c = K_p[:, comp].transpose(0, 1).float().contiguous()   # [T_c,L,D]
        V_c = V_p[:, comp].transpose(0, 1).float().contiguous()
        pos_c = positions[comp].float().to(self.device) if positions is not None else None

        Bm = self.n_macro
        eps = 1e-6
        # feature affinity: cosine over flattened per-layer K [T_c, L*D]
        feat_c = K_c.reshape(T_c, L * D)
        feat_m = self.macro_K.reshape(Bm, L * D)
        aff = F.normalize(feat_m, dim=-1) @ F.normalize(feat_c, dim=-1).t()  # [Bm,T_c]

        # kinematic penalty (deformation barrier) — spatial proximity
        penalty = torch.ones(Bm, T_c, device=self.device)
        if pos_c is not None and self.macro_cnt.sum() > 0:
            d2 = ((self.macro_pos.unsqueeze(1) - pos_c.unsqueeze(0)) ** 2).sum(-1)
            penalty = torch.exp(-self.gamma * d2 / (self.pos_scale ** 2))

        joint = aff * penalty
        A = F.relu(joint - self.threshold)
        A = A / (A.sum(dim=0, keepdim=True) + eps)     # SparseMax over macro
        # A: [Bm, T_c]  (each token's absorption weights sum to 1 over macro)

        # adaptive capacity gate (EMA)
        alpha = self.alpha_scale * 0.5                  # fixed gate (sig(0)=0.5)
        upd_K = torch.einsum('mt,tld->mld', A, K_c)     # [Bm, L, D]
        upd_V = torch.einsum('mt,tld->mld', A, V_c)
        self.macro_K = (1 - alpha) * self.macro_K + alpha * upd_K
        self.macro_V = (1 - alpha) * self.macro_V + alpha * upd_V
        if pos_c is not None:
            upd_pos = torch.einsum('mt,td->md', A, pos_c)  # [Bm,3]
            self.macro_pos = (1 - alpha) * self.macro_pos + alpha * upd_pos
        cnt = A.sum(dim=1)                              # [Bm]
        self.macro_cnt = self.macro_cnt + cnt
        self._n_absorbed += T_c
        self._n_absorb_calls += 1
        return T_c, int((cnt > 0).sum().item())

    # ── retrieve: broadcast macro-voxels to [L*H, Bm, D] ──────────────────
    @torch.no_grad()
    def retrieve(self, voxel_ids=None, return_buf=False, retrieve_size=-1):
        L, H, Bm, D = self._L, self._H, self.n_macro, self.head_dim
        active = self.macro_cnt > 0                     # [Bm]
        # per-layer macro K/V -> broadcast to H heads
        K_m = self.macro_K.transpose(0, 1).to(self.dtype)   # [L, Bm, D] bf16
        V_m = self.macro_V.transpose(0, 1).to(self.dtype)
        K_ret = K_m.unsqueeze(1).expand(L, H, Bm, D).reshape(L * H, Bm, D)
        V_ret = V_m.unsqueeze(1).expand(L, H, Bm, D).reshape(L * H, Bm, D)
        # mask inactive macro-voxels with -inf bias (all heads)
        bias = torch.where(active, torch.zeros_like(self.macro_cnt),
                           torch.full_like(self.macro_cnt, -float('inf')))
        B_ret = bias.unsqueeze(0).expand(L * H, Bm).contiguous()   # [LH,Bm]
        M_ret = torch.ones(L * H, Bm, device=self.device, dtype=torch.bool)
        if retrieve_size is not None and retrieve_size >= 0:
            Tm = min(Bm, int(retrieve_size))
            K_ret = K_ret[:, :Tm]; V_ret = V_ret[:, :Tm]
            B_ret = B_ret[:, :Tm]; M_ret = M_ret[:, :Tm]
        return K_ret, V_ret, M_ret, B_ret

    def update_voxel_zones(self, voxel_zones):
        pass  # no spatial zones in KGTC store

    def step_frame(self):
        pass

    def check_and_evict(self):
        return 0

    def memory(self):
        bytes_all = (self.macro_K.numel() + self.macro_V.numel()
                     + self.macro_pos.numel() + self.macro_f7.numel()
                     + self.macro_cnt.numel()) * 4
        mb = bytes_all / (1024.0 ** 2)
        return {
            "buffer_used_mem": mb,
            "buffer_alloc_mem": mb,
            "pivot_used_mem": 0.0,
            "pivot_alloc_mem": 0.0,
        }

    def info(self):
        active = int((self.macro_cnt > 0).sum().item())
        return {
            "n_macro": self.n_macro,
            "macro_active": active,
            "macro_bytes_mb": self.memory()["buffer_used_mem"],
            "n_tokens_seen": self._n_tokens,
            "n_tokens_absorbed": self._n_absorbed,
            "absorb_calls": self._n_absorb_calls,
        }
