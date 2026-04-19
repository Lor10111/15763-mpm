"""
Pure-PyTorch SVD replacement for warp_svd.
Drops the warp dependency entirely while keeping the same interface:
    SVD()(F) -> U, sigma, Vh
where
    F     : (N, 3, 3) deformation gradient tensor
    U     : (N, 3, 3)  left singular vectors,  det(U)  = +1
    sigma : (N, 3)     singular values (may be negative after sign fix)
    Vh    : (N, 3, 3)  right singular vectors transposed, det(Vh) = +1

Sign convention matches the original warp kernel (last column / row flipped
to enforce proper rotations) so downstream CorotatedElasticity is unchanged.
"""

import torch
import torch.nn as nn
from torch import Tensor


class SVD(nn.Module):
    def forward(self, F: Tensor):   # F: (N, 3, 3)
        U, S, Vh = torch.linalg.svd(F)   # S >= 0

        # ── Ensure det(U) = +1 ──────────────────────────────────────────
        det_U  = torch.linalg.det(U)          # (N,)
        sign_U = torch.sign(det_U)            # ±1
        U  = U.clone()
        U[:, :, 2]  = U[:, :, 2]  * sign_U.unsqueeze(-1)
        S  = S.clone()
        S[:, 2]     = S[:, 2]     * sign_U

        # ── Ensure det(Vh) = +1 ─────────────────────────────────────────
        det_Vh  = torch.linalg.det(Vh)        # (N,)
        sign_Vh = torch.sign(det_Vh)          # ±1
        Vh = Vh.clone()
        Vh[:, 2, :] = Vh[:, 2, :] * sign_Vh.unsqueeze(-1)
        S[:, 2]     = S[:, 2]     * sign_Vh

        return U, S, Vh
