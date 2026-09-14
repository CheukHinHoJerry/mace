"""Product-group STP, verified against the einsum reference."""

import itertools
import sys

import numpy as np
import torch

torch.serialization.add_safe_globals([slice])
sys.path.insert(0, "/home/jerry528/mace-ace-pr124/mace")
import cuequivariance as cue
import cuequivariance_torch as cuet
from e3nn import o3

from mace.tools.cg import U_matrix_real

torch.set_default_dtype(torch.float64)


def build_stp(MAX_ELL, MAX_M_ELL, NU, CH):
    Vr = o3.Irreps.spherical_harmonics(MAX_ELL, p=-1)
    Vm = o3.Irreps.spherical_harmonics(MAX_M_ELL, p=-1)
    Us = U_matrix_real(
        irreps_in=Vr,
        irreps_out=o3.Irreps("0e"),
        correlation=NU,
        dtype=torch.float64,
        use_cueq_cg=False,
    )[-1]
    Um = U_matrix_real(
        irreps_in=Vm,
        irreps_out=o3.Irreps("0e"),
        correlation=NU,
        dtype=torch.float64,
        use_cueq_cg=False,
    )[-1]
    dr, dm, P, Q = Vr.dim, Vm.dim, Us.shape[-1], Um.shape[-1]
    D = dr * dm
    stp = cue.SegmentedTensorProduct.from_subscripts(",".join(["u"] * (NU + 2)))
    for _ in range(P * Q):
        stp.add_segment(0, (CH,))  # weights, one per (k,q)
    for op in range(1, NU + 1):
        for _ in range(D):
            stp.add_segment(op, (CH,))  # paired components
    stp.add_segment(NU + 1, (CH,))  # scalar output
    nzr = [
        (ri, k, float(Us[ri + (k,)]))
        for ri in itertools.product(range(dr), repeat=NU)
        for k in range(P)
        if abs(float(Us[ri + (k,)])) > 1e-12
    ]
    nzm = [
        (mi, q, float(Um[mi + (q,)]))
        for mi in itertools.product(range(dm), repeat=NU)
        for q in range(Q)
        if abs(float(Um[mi + (q,)])) > 1e-12
    ]
    for ri, k, cr in nzr:
        for mi, q, cm in nzm:
            stp.add_path(
                k * Q + q, *[ri[t] * dm + mi[t] for t in range(NU)], 0, c=cr * cm
            )
    return stp, Us, Um, dr, dm, P, Q, D


for MAX_ELL, MAX_M_ELL, NU in ((2, 1, 2), (2, 1, 3)):
    CH, B = 4, 6
    stp, Us, Um, dr, dm, P, Q, D = build_stp(MAX_ELL, MAX_M_ELL, NU, CH)
    print(
        f"max_ell={MAX_ELL} max_m_ell={MAX_M_ELL} nu={NU}: D={D} weights={P*Q} "
        f"paths={stp.num_paths}"
    )
    poly = cue.SegmentedPolynomial(
        inputs=(stp.operands[0],) + tuple(stp.operands[1 : NU + 1]),
        outputs=(stp.operands[NU + 1],),
        operations=[(cue.Operation([0] + [1] * NU + [NU + 1]), stp)],
    )
    mod = cuet.SegmentedPolynomial(poly, math_dtype=torch.float64)
    g = torch.Generator().manual_seed(0)
    W = torch.randn(B, P * Q, CH, generator=g).reshape(B, -1)
    A = torch.randn(B, D, CH, generator=g).reshape(B, -1)
    got = mod([W] + [A] * NU)[0].reshape(B, CH)
    # reference einsum, same convention
    Ar = A.reshape(B, dr, dm, CH)
    Wr = W.reshape(B, P, Q, CH)
    eq = {2: "ijk,lmq,bkqa,bila,bjma->ba", 3: "ijfk,lmgq,bkqa,bila,bjma,bfga->ba"}[NU]
    ref = torch.einsum(eq, Us, Um, Wr, *([Ar] * NU))
    print(
        f"   max|cueq - einsum| = {float((got-ref).abs().max()):.3e}  "
        f"(scale {float(ref.abs().max()):.3e})"
    )
