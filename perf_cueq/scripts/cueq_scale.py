"""Does cueq backward cost scale pathologically with path count?

Forward is fine at 29,299 paths (2.8 s incl. compile). If backward is fine at 830 paths
but not at 29,299, the problem is the path count, which is forced by the torch binding
requiring zero/one-dimensional operands (so the paired (l,l') blocks must be flattened
into unit segments, one path per nonzero CG entry).
"""

import itertools
import sys
import time

import torch

torch.serialization.add_safe_globals([slice])
sys.path.insert(0, "/home/jerry528/mace-ace-pr124/mace")
import cuequivariance as cue
import cuequivariance_torch as cuet
from e3nn import o3

from mace.tools.cg import U_matrix_real

torch.set_default_dtype(torch.float64)
DEV = "cuda"
T0 = time.perf_counter()


def log(m):
    print(f"[{time.perf_counter()-T0:7.1f}s] {m}", flush=True)


for max_ell, max_m_ell, nu, ch, B in (
    (2, 1, 2, 64, 256),
    (2, 1, 3, 64, 256),
    (3, 2, 2, 64, 256),
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
    for ri, k, cr in nzr:
        for mi, q, cm in nzm:
            stp.add_path(
                k * Q + q, *[ri[t] * dm + mi[t] for t in range(nu)], 0, c=cr * cm
            )
    poly = cue.SegmentedPolynomial(
        inputs=tuple(stp.operands[: nu + 1]),
        outputs=(stp.operands[nu + 1],),
        operations=[(cue.Operation([0] + [1] * nu + [nu + 1]), stp)],
    )
    mod = cuet.SegmentedPolynomial(poly, math_dtype=torch.float64).to(DEV)
    g = torch.Generator(device=DEV).manual_seed(0)
    W = torch.randn(B, P * Q * ch, device=DEV, generator=g)
    A = torch.randn(B, dr * dm * ch, device=DEV, generator=g)
    t = time.perf_counter()
    _ = mod([W] + [A] * nu)[0]
    torch.cuda.synchronize()
    tf = time.perf_counter() - t
    Wg = W.clone().requires_grad_(True)
    Ag = A.clone().requires_grad_(True)
    t = time.perf_counter()
    mod([Wg] + [Ag] * nu)[0].sum().backward()
    torch.cuda.synchronize()
    tb = time.perf_counter() - t
    log(f"paths={stp.num_paths:>6}  first fwd {tf:7.2f}s   first bwd {tb:8.2f}s")
