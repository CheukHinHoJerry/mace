"""Discriminate WHY cueq's double backward disagrees with einsum.

Three axes:
  encoding : shared   -> Operation([0,1,1,...,1,out]); cueq sees one buffer repeated nu times
             distinct -> Operation([0,1,2,...,nu,out]); each slot its own buffer, torch accumulates
  method   : whatever cuet.SegmentedPolynomial accepts (kernel vs pure-pytorch 'naive')
  order    : first backward vs double backward

einsum is the reference: plain autograd through torch.einsum is correct by construction.
"""

import itertools
import os
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
MAX_ELL = int(os.environ.get("MAX_ELL", 2))
MAX_MELL = int(os.environ.get("MAX_MELL", 1))
NU = int(os.environ.get("NU", 3))
CH = int(os.environ.get("CH", 32))
B = int(os.environ.get("BATCH", 64))
METHOD = os.environ.get("CUEQ_METHOD", "")
T0 = time.perf_counter()


def log(m):
    print(f"[{time.perf_counter()-T0:7.1f}s] {m}", flush=True)


log(
    f"cueq {cue.__version__} | max_ell={MAX_ELL} m_ell={MAX_MELL} nu={NU} ch={CH} B={B} method={METHOD or 'default'}"
)
Vr = o3.Irreps.spherical_harmonics(MAX_ELL, p=-1)
Vm = o3.Irreps.spherical_harmonics(MAX_MELL, p=-1)
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


def build(encoding):
    stp = cue.SegmentedTensorProduct.from_subscripts(",".join(["u"] * (NU + 2)))
    for _ in range(P * Q):
        stp.add_segment(0, (CH,))
    for op in range(1, NU + 1):
        for _ in range(dr * dm):
            stp.add_segment(op, (CH,))
    stp.add_segment(NU + 1, (CH,))
    for ri, k, cr in nzr:
        for mi, q, cm in nzm:
            stp.add_path(
                k * Q + q, *[ri[t] * dm + mi[t] for t in range(NU)], 0, c=cr * cm
            )
    buffers = [0] + [1] * NU if encoding == "shared" else list(range(NU + 1))
    poly = cue.SegmentedPolynomial(
        inputs=tuple(stp.operands[: NU + 1]),
        outputs=(stp.operands[NU + 1],),
        operations=[(cue.Operation(buffers + [NU + 1]), stp)],
    )
    kw = {"math_dtype": torch.float64}
    if METHOD:
        try:
            return (
                cuet.SegmentedPolynomial(poly, method=METHOD, **kw).to(DEV),
                stp.num_paths,
            )
        except TypeError:
            log("  (this cueq build takes no method= arg; using default)")
    return cuet.SegmentedPolynomial(poly, **kw).to(DEV), stp.num_paths


g = torch.Generator(device=DEV).manual_seed(0)
W0 = torch.randn(B, P * Q * CH, device=DEV, generator=g)
A0 = torch.randn(B, dr * dm * CH, device=DEV, generator=g)
UsD, UmD = Us.to(DEV), Um.to(DEV)
EQ = "ijfk,lmgq,bkqa,bila,bjma,bfga->ba"  # nu=3


def run(fn, second):
    W = W0.clone().requires_grad_(True)
    A = A0.clone().requires_grad_(True)
    out = fn(W, A)
    if not second:
        out.sum().backward()
    else:
        (gA,) = torch.autograd.grad(out.sum(), A, create_graph=True)
        (gA**2).sum().backward()
    return A.grad.clone(), W.grad.clone()


ref = {
    s: run(
        lambda W, A: torch.einsum(
            EQ, UsD, UmD, W.reshape(B, P, Q, CH), *([A.reshape(B, dr, dm, CH)] * NU)
        ),
        s,
    )
    for s in (False, True)
}
log("einsum reference done")

for encoding in ("shared", "distinct"):
    t = time.perf_counter()
    mod, npaths = build(encoding)
    bt = time.perf_counter() - t
    # 'shared' hands cueq the identical tensor nu times; 'distinct' hands it nu separate
    # graph nodes of the same values, so torch -- not cueq -- accumulates the slot gradients.
    fn = (
        (lambda W, A: mod([W] + [A] * NU)[0].reshape(B, CH))
        if encoding == "shared"
        else (lambda W, A: mod([W] + [A * 1.0 for _ in range(NU)])[0].reshape(B, CH))
    )
    log(f"--- encoding={encoding}  paths={npaths}  build={bt:.1f}s")
    with torch.no_grad():
        fc = fn(W0.clone(), A0.clone())
        fe = torch.einsum(
            EQ, UsD, UmD, W0.reshape(B, P, Q, CH), *([A0.reshape(B, dr, dm, CH)] * NU)
        )
        fd = float((fc - fe).abs().max())
        fs = float(fe.abs().max())
        log(
            f"    FORWARD          max|cueq-einsum|={fd:.3e}  rel={fd/fs:.3e}  "
            f"{'OK' if fd/fs < 1e-10 else '** MISMATCH **'}"
        )
    for second in (False, True):
        tag = "double" if second else "first "
        try:
            t = time.perf_counter()
            ga, gw = run(fn, second)
            torch.cuda.synchronize()
            d = max(
                float((ga - ref[second][0]).abs().max()),
                float((gw - ref[second][1]).abs().max()),
            )
            scale = max(
                float(ref[second][0].abs().max()), float(ref[second][1].abs().max())
            )
            log(
                f"    {tag} bwd  {time.perf_counter()-t:7.1f}s  max|cueq-einsum|={d:.3e}  rel={d/scale:.3e}  "
                f"{'OK' if d/scale < 1e-10 else '** MISMATCH **'}"
            )
        except Exception as e:
            log(f"    {tag} bwd  ** FAILED ** {type(e).__name__}: {str(e)[:160]}")
