"""How long does STP construction take, and how does it scale with path count?

No GPU needed: this is pure Python calling into cueq's descriptor structure.
"""

import itertools
import sys
import time

import torch

torch.serialization.add_safe_globals([slice])
sys.path.insert(0, "/home/jerry528/mace-ace-pr124/mace")
import cuequivariance as cue
from e3nn import o3

from mace.tools.cg import U_matrix_real

torch.set_default_dtype(torch.float64)

print(f"{'case':<26} {'paths':>7} {'build s':>9} {'us/path':>9}")
print("-" * 56)
for max_ell, max_m_ell, nu, label in (
    (2, 1, 2, "max_ell=2,m_ell=1,nu=2"),
    (2, 1, 3, "max_ell=2,m_ell=1,nu=3"),
    (3, 2, 2, "max_ell=3,m_ell=2,nu=2"),
    (3, 2, 3, "max_ell=3,m_ell=2,nu=3"),
):
    Vr = o3.Irreps.spherical_harmonics(max_ell, p=-1)
    Vm = o3.Irreps.spherical_harmonics(max_m_ell, p=-1)
    Us = U_matrix_real(
        irreps_in=Vr,
        irreps_out=o3.Irreps("0e"),
        correlation=nu,
        dtype=torch.float64,
        use_cueq_cg=False,
    )[-1]
    Um = U_matrix_real(
        irreps_in=Vm,
        irreps_out=o3.Irreps("0e"),
        correlation=nu,
        dtype=torch.float64,
        use_cueq_cg=False,
    )[-1]
    dr, dm, P, Q = Vr.dim, Vm.dim, Us.shape[-1], Um.shape[-1]
    ch = 128
    t0 = time.perf_counter()
    stp = cue.SegmentedTensorProduct.from_subscripts(",".join(["u"] * (nu + 2)))
    for _ in range(P * Q):
        stp.add_segment(0, (ch,))
    for op in range(1, nu + 1):
        for _ in range(dr * dm):
            stp.add_segment(op, (ch,))
    stp.add_segment(nu + 1, (ch,))
    nzr = [
        (ri, k, float(Us[ri + (k,)]))
        for ri in itertools.product(range(dr), repeat=nu)
        for k in range(P)
        if abs(float(Us[ri + (k,)])) > 1e-12
    ]
    nzm = [
        (mi, q, float(Um[mi + (q,)]))
        for mi in itertools.product(range(dm), repeat=nu)
        for q in range(Q)
        if abs(float(Um[mi + (q,)])) > 1e-12
    ]
    n = 0
    for ri, k, cr in nzr:
        for mi, q, cm in nzm:
            stp.add_path(
                k * Q + q, *[ri[t] * dm + mi[t] for t in range(nu)], 0, c=cr * cm
            )
            n += 1
    dt = time.perf_counter() - t0
    print(f"{label:<26} {n:>7,} {dt:9.1f} {1e6*dt/max(n,1):9.1f}")
