"""Backward + DOUBLE backward through cuet.SegmentedPolynomial, with stage timing.

Prints each stage as it starts so a long stall is attributable rather than mysterious.
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


def log(msg):
    print(f"[{time.perf_counter()-T0:8.1f}s] {msg}", flush=True)


max_ell, max_m_ell, nu, ch, B = 3, 1, 3, 128, 432  # PRODUCTION truncation
log("building CG bases")
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

log("building STP")
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
        stp.add_path(k * Q + q, *[ri[t] * dm + mi[t] for t in range(nu)], 0, c=cr * cm)
log(f"STP built: {stp.num_paths} paths")

poly = cue.SegmentedPolynomial(
    inputs=tuple(stp.operands[: nu + 1]),
    outputs=(stp.operands[nu + 1],),
    operations=[(cue.Operation([0] + [1] * nu + [nu + 1]), stp)],
)
log("constructing cuet.SegmentedPolynomial (this is where JIT happens)")
import os

METHOD = os.environ.get("CUEQ_METHOD", "uniform_1d")
mod = cuet.SegmentedPolynomial(poly, method=METHOD, math_dtype=torch.float64).to(DEV)
log(f"module ready (cueq {cue.__version__}, method={METHOD})")

Us, Um = Us.to(DEV), Um.to(DEV)
g = torch.Generator(device=DEV).manual_seed(0)
W0 = torch.randn(B, P * Q * ch, device=DEV, generator=g)
A0 = torch.randn(B, dr * dm * ch, device=DEV, generator=g)
eq = "ijfk,lmgq,bkqa,bila,bjma,bfga->ba"

log("first cueq forward (kernel compile lands here if lazy)")
_ = mod([W0] + [A0] * nu)[0]
torch.cuda.synchronize()
log("forward done")


def grads(kind, second):
    W = W0.clone().requires_grad_(True)
    A = A0.clone().requires_grad_(True)
    out = (
        mod([W] + [A] * nu)[0].reshape(B, ch)
        if kind == "cueq"
        else torch.einsum(
            eq, Us, Um, W.reshape(B, P, Q, ch), *([A.reshape(B, dr, dm, ch)] * nu)
        )
    )
    if not second:
        out.sum().backward()
    else:
        (gA,) = torch.autograd.grad(out.sum(), A, create_graph=True)
        (gA**2).sum().backward()
    return A.grad.clone(), W.grad.clone()


for second in (False, True):
    tag = "DOUBLE backward" if second else "first backward"
    for kind in ("cueq", "einsum"):
        log(f"{tag} / {kind} ...")
        try:
            ga, gw = grads(kind, second)
            torch.cuda.synchronize()
            log(f"{tag} / {kind} OK  |gA|={float(ga.abs().max()):.3e}")
            globals()[f"{kind}_{second}"] = (ga, gw)
        except Exception as e:
            log(f"{tag} / {kind} ** FAILED ** {type(e).__name__}: {str(e)[:140]}")
    a, b = globals().get(f"cueq_{second}"), globals().get(f"einsum_{second}")
    if a and b:
        d = max(float((a[0] - b[0]).abs().max()), float((a[1] - b[1]).abs().max()))
        log(
            f"{tag}: max|cueq - einsum| = {d:.3e}  {'OK' if d<1e-8 else '** MISMATCH **'}"
        )
