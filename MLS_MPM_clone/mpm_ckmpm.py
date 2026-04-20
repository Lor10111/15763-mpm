"""
CK-MPM Python/PyTorch implementation
Kernel:    SmoothLinear (C2-compact, radius 1 cell)
           N(x)  = (1 - |x|) + (1/2pi)*sin(2pi*|x|)
           N'(x) = sgn(x)*(cos(2pi*|x|) - 1)
Dual-grid: two grids offset by 0.0 and 0.25*dx.
           P2G scatters to both; G2P averages from both.
Stencil:   2x2x2 = 8 nodes (vs 3x3x3 = 27 in MLS-MPM).
"""
import math
import torch
from torch import Tensor
from typing import List, Callable


def _w1(x: Tensor) -> Tensor:
    """SmoothLinear weight: N(x) = (1-|x|) + sin(2pi|x|)/(2pi), support |x|<=1"""
    ax = x.abs().clamp(max=1.0)   # hard zero outside support
    return (1.0 - ax) + (1.0 / (2.0 * math.pi)) * torch.sin(2.0 * math.pi * ax)

def _dw1(x: Tensor) -> Tensor:
    """SmoothLinear gradient: N'(x) = sgn(x)*(cos(2pi|x|) - 1), support |x|<=1"""
    ax = x.abs()
    inside = (ax <= 1.0).float()
    return inside * torch.sign(x) * (torch.cos(2.0 * math.pi * ax) - 1.0)


class CKMPMSolver:
    """
    CK-MPM solver. API-compatible with MPMSolver from mpm_pytorch.
    self.grid_mv is always a plain Tensor so boundary_condition.py
    boolean-mask writes work unchanged.
    """

    def __init__(
        self,
        init_pos: Tensor,
        rho: float = 1000.0,
        num_grids: int = 40,
        dt: float = 5e-5,
        gravity: List[float] = [0.0, 0.0, -9.8],
        clip_bound: float = 0.5,
        damping: float = 1.0,
        enable_train: bool = False,
        device: torch.device = torch.device("cuda"),
    ):
        self.num_grids  = num_grids
        self.dt         = dt
        self.dx         = 1.0 / num_grids
        self.inv_dx     = float(num_grids)
        self.gravity    = torch.tensor(gravity, device=device, dtype=torch.float32)
        self.clip_bound = clip_bound * self.dx
        self.damping    = damping
        self.device     = device

        self.n_particles = init_pos.shape[0]
        self.vol         = (self.dx ** 3) / 8.0
        self.p_mass      = rho * self.vol

        # Dual-grid offsets (fractional cells): 0.0 and 0.25
        self._grid_offset = [0.0, 0.25]

        n3 = num_grids ** 3
        self._grid_mv = [torch.zeros(n3, 3, device=device) for _ in range(2)]
        self._grid_m  = [torch.zeros(n3,    device=device) for _ in range(2)]

        # Public grid_mv — plain tensor, swapped to each internal grid during BCs
        self.grid_mv = self._grid_mv[0]
        self.grid_m  = self._grid_m[0]

        # Integer node positions for boundary_condition.py
        r = torch.arange(num_grids, device=device)
        gx, gy, gz = torch.meshgrid(r, r, r, indexing='ij')
        self.grid_x = torch.stack([gx, gy, gz], dim=-1).reshape(-1, 3).float()

        # 2x2x2 stencil offsets {0,1}^3
        self.offset = torch.tensor(
            [[i, j, k] for i in range(2) for j in range(2) for k in range(2)],
            device=device, dtype=torch.float32)   # [8, 3]

        self.post_grid_process: List[Callable] = []
        self.time = 0.0

    def _stencil(self, x: Tensor, g: int):
        """2x2x2 stencil for grid g. Returns weight[N,8], dweight[N,8,3], flat[N*8], dpos[N,8,3]"""
        off  = self._grid_offset[g]
        px   = x * self.inv_dx - off
        base = px.floor().long().clamp(0, self.num_grids - 2)
        fx   = px - base.float()          # fractional part in [0, 1)

        w_list, dw_list = [], []
        for o in self.offset:             # o in {0,1}^3
            rel = fx - o.unsqueeze(0)     # [N, 3]
            wx = _w1(rel[:,0]); wy = _w1(rel[:,1]); wz = _w1(rel[:,2])
            w_list.append(wx * wy * wz)
            dw_list.append(torch.stack([
                _dw1(rel[:,0])*wy*wz,
                wx*_dw1(rel[:,1])*wz,
                wx*wy*_dw1(rel[:,2]),
            ], dim=1))

        weight  = torch.stack(w_list,  dim=1)                       # [N, 8]
        dweight = torch.stack(dw_list, dim=1) * self.inv_dx         # [N, 8, 3]

        n = self.num_grids
        nidx = (base.unsqueeze(1) + self.offset.unsqueeze(0).long()).clamp(0, n-1)
        flat = (nidx[:,:,0]*n*n + nidx[:,:,1]*n + nidx[:,:,2]).reshape(-1)
        dpos = (self.offset.unsqueeze(0) - fx.unsqueeze(1)) * self.dx  # [N,8,3]
        return weight, dweight, flat, dpos

    def _apply_bcs(self, g: int):
        """Swap grid_mv to internal grid g, run all BC hooks, save back."""
        self.grid_mv = self._grid_mv[g]
        for op in self.post_grid_process:
            op(self)
        self._grid_mv[g] = self.grid_mv

    def __call__(self, x, v, C, F, stress):
        return self.p2g2p(x, v, C, F, stress)

    def p2g2p(self, x: Tensor, v: Tensor, C: Tensor, F: Tensor, stress: Tensor):
        dt     = self.dt
        vol    = self.vol
        p_mass = self.p_mass

        # Zero both grids
        for g in range(2):
            self._grid_mv[g].zero_()
            self._grid_m[g].zero_()

        # ---- P2G → both grids ----
        for g in range(2):
            weight, dweight, flat, dpos = self._stencil(x, g)
            stress_t = -dt * vol * torch.einsum('bij,bnj->bni', stress, dweight)
            apic_t   = p_mass * weight.unsqueeze(2) * (
                v.unsqueeze(1) + torch.einsum('bij,bnj->bni', C, dpos))
            self._grid_mv[g].index_add_(0, flat, (stress_t + apic_t).reshape(-1, 3))
            self._grid_m[g].index_add_(0, flat, (weight * p_mass).reshape(-1))

        # ---- Grid normalise + gravity ----
        # Use a larger mass threshold than MLS-MPM because the CK 2x2x2 stencil
        # can leave boundary cells with tiny but non-zero mass, causing mv/m → inf.
        for g in range(2):
            sel = self._grid_m[g] > 1e-10
            self._grid_mv[g][sel] = self.damping * (
                self._grid_mv[g][sel] / self._grid_m[g][sel].unsqueeze(1)
                + dt * self.gravity)
            # Zero out any nodes that didn't meet threshold (avoids stale values)
            self._grid_mv[g][~sel] = 0.0

        # ---- Boundary conditions ----
        for g in range(2):
            self._apply_bcs(g)

        # ---- G2P ← average both grids ----
        v_new = torch.zeros_like(v)
        C_new = torch.zeros_like(C)
        dF    = torch.zeros(self.n_particles, 3, 3, device=self.device)

        for g in range(2):
            weight, dweight, flat, dpos = self._stencil(x, g)
            v_g = self._grid_mv[g].index_select(0, flat).reshape(-1, 8, 3)
            v_new += 0.5 * (weight.unsqueeze(2) * v_g).sum(dim=1)
            C_new += 0.5 * 4.0 * self.inv_dx**2 * (
                weight.unsqueeze(2).unsqueeze(3) *
                torch.einsum('bni,bnj->bnij', v_g, dpos)).sum(dim=1)
            dF    += 0.5 * dt * torch.einsum('bni,bnj->bnij', v_g, dweight).sum(dim=1)

        x = (x + v_new * dt).clamp(self.clip_bound, 1.0 - self.clip_bound)
        F = F + torch.bmm(dF, F)

        if not torch.isfinite(F).all():
            # print(f"[CK-MPM] nan in F after bmm — clamping")
            F = torch.nan_to_num(F, nan=1.0, posinf=1.0, neginf=-1.0)

        self.time += dt
        self.grid_mv = self._grid_mv[0]
        return x, v_new, C_new, F
