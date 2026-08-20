"""cueq SegmentedPolynomial vs the reference einsum for the paired non-SOC contraction."""

import itertools
import statistics
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


def build(max_ell, max_m_ell, nu, ch):
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
    return stp, poly, Us.to(DEV), Um.to(DEV), dr, dm, P, Q


def bench(fn, n=5):
    for _ in range(2):
        fn()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    ts = []
    for _ in range(n):
        torch.cuda.synchronize()
        t = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        ts.append((time.perf_counter() - t) * 1e3)
    return statistics.median(ts), torch.cuda.max_memory_allocated() / 1e9


for max_ell, max_m_ell, nu, ch, B in ((2, 1, 3, 64, 432), (3, 2, 3, 128, 432)):
    stp, poly, Us, Um, dr, dm, P, Q = build(max_ell, max_m_ell, nu, ch)
    print(
        f"\n=== max_ell={max_ell} max_m_ell={max_m_ell} nu={nu} ch={ch} nodes={B} "
        f"| STP paths={stp.num_paths} ==="
    )
    try:
        mod = cuet.SegmentedPolynomial(poly, math_dtype=torch.float64).to(DEV)
    except Exception as e:
        print(f"  cueq module build FAILED: {type(e).__name__}: {str(e)[:110]}")
        continue
    g = torch.Generator(device=DEV).manual_seed(0)
    W = torch.randn(B, P * Q * ch, device=DEV, generator=g)
    A = torch.randn(B, dr * dm * ch, device=DEV, generator=g)
    Ar, Wr = A.reshape(B, dr, dm, ch), W.reshape(B, P, Q, ch)
    eq = {3: "ijfk,lmgq,bkqa,bila,bjma,bfga->ba"}[nu]

    def f_cueq():
        return mod([W] + [A] * nu)[0]

    def f_eins():
        return torch.einsum(eq, Us, Um, Wr, Ar, Ar, Ar)

    try:
        d = float((f_cueq().reshape(B, ch) - f_eins()).abs().max())
        print(f"  max|cueq - einsum| = {d:.3e}")
    except Exception as e:
        print(f"  cueq RUN FAILED: {type(e).__name__}: {str(e)[:110]}")
        continue
    tc, mc = bench(f_cueq)
    te, me = bench(f_eins)
    print(f"  cueq   fwd {tc:8.2f} ms   peak {mc:6.2f} GB")
    print(f"  einsum fwd {te:8.2f} ms   peak {me:6.2f} GB   -> cueq is {te/tc:.2f}x")
