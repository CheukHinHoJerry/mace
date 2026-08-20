"""Steady-state timing of the 'distinct' encoding vs einsum, after JIT warmup.

The discriminator's 27s/108s were first-call compile. What matters for training is the
warm cost of forward, forward+backward, and forward+backward+double-backward (the force loss).
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
MAX_ELL = int(os.environ.get("MAX_ELL", 3))
MAX_MELL = int(os.environ.get("MAX_MELL", 2))
NU = int(os.environ.get("NU", 3))
CH = int(os.environ.get("CH", 128))
B = int(os.environ.get("BATCH", 432))
REP = int(os.environ.get("REP", 5))
T0 = time.perf_counter()


def log(m):
    print(f"[{time.perf_counter()-T0:7.1f}s] {m}", flush=True)


log(
    f"cueq {cue.__version__} | max_ell={MAX_ELL} m_ell={MAX_MELL} nu={NU} ch={CH} B={B}"
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
log(f"dr={dr} dm={dm} P={P} Q={Q}")
stp = cue.SegmentedTensorProduct.from_subscripts(",".join(["u"] * (NU + 2)))
for _ in range(P * Q):
    stp.add_segment(0, (CH,))
for op in range(1, NU + 1):
    for _ in range(dr * dm):
        stp.add_segment(op, (CH,))
stp.add_segment(NU + 1, (CH,))
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
t = time.perf_counter()
for ri, k, cr in nzr:
    for mi, q, cm in nzm:
        stp.add_path(
            k * Q + q, *[ri[t2] * dm + mi[t2] for t2 in range(NU)], 0, c=cr * cm
        )
log(f"STP {stp.num_paths} paths, build {time.perf_counter()-t:.1f}s")
poly = cue.SegmentedPolynomial(
    inputs=tuple(stp.operands[: NU + 1]),
    outputs=(stp.operands[NU + 1],),
    operations=[(cue.Operation(list(range(NU + 1)) + [NU + 1]), stp)],
)  # DISTINCT buffers
mod = cuet.SegmentedPolynomial(poly, math_dtype=torch.float64).to(DEV)
UsD, UmD = Us.to(DEV), Um.to(DEV)
g = torch.Generator(device=DEV).manual_seed(0)
W0 = torch.randn(B, P * Q * CH, device=DEV, generator=g)
A0 = torch.randn(B, dr * dm * CH, device=DEV, generator=g)
EQ = "ijfk,lmgq,bkqa,bila,bjma,bfga->ba"
f_cueq = lambda W, A: mod([W] + [A * 1.0 for _ in range(NU)])[0].reshape(B, CH)
f_eins = lambda W, A: torch.einsum(
    EQ, UsD, UmD, W.reshape(B, P, Q, CH), *([A.reshape(B, dr, dm, CH)] * NU)
)


def once(fn, order):
    W = W0.clone().requires_grad_(True)
    A = A0.clone().requires_grad_(True)
    out = fn(W, A)
    if order == 0:
        return out
    if order == 1:
        out.sum().backward()
        return A.grad
    (gA,) = torch.autograd.grad(out.sum(), A, create_graph=True)
    (gA**2).sum().backward()
    return A.grad


log("correctness (warm-up call doubles as the check)")
for order, name in ((0, "forward"), (1, "backward"), (2, "double bwd")):
    try:
        a = once(f_cueq, order)
        b = once(f_eins, order)
        torch.cuda.synchronize()
        d = float((a - b).abs().max())
        s = float(b.abs().max())
        log(f"  {name:11s} rel={d/s:.3e}  {'OK' if d/s<1e-10 else '** MISMATCH **'}")
    except Exception as e:
        import traceback

        log(f"  {name:11s} FAILED {type(e).__name__}")
        log(traceback.format_exc())
log("warm timings (median of %d)" % REP)
for order, name in ((0, "forward"), (1, "fwd+bwd"), (2, "fwd+bwd+double")):
    row = []
    for fn, tag in ((f_cueq, "cueq"), (f_eins, "einsum")):
        ts = []
        for _ in range(REP):
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            t = time.perf_counter()
            try:
                once(fn, order)
                torch.cuda.synchronize()
            except Exception as e:
                import traceback

                ts = None
                log(f"  {name} {tag} FAILED {type(e).__name__}")
                log(traceback.format_exc())
                break
            ts.append(time.perf_counter() - t)
        if ts is None:
            row.append(None)
            continue
        ts.sort()
        row.append((ts[len(ts) // 2], torch.cuda.max_memory_allocated() / 2**30))
    if all(r is not None for r in row):
        (tc, mc), (te, me) = row
        log(
            f"  {name:15s} cueq {tc*1e3:8.1f} ms {mc:6.2f} GB | einsum {te*1e3:8.1f} ms {me:6.2f} GB | "
            f"speedup {te/tc:5.2f}x  mem {me/mc:6.1f}x"
        )
