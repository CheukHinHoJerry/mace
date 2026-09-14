"""Three ways to compute pooled_A[i,k,l,p] = sum_{e: recv(e)=i} r[e,k,l] * m[e,k,p].

 (1) naive        : materialise A_msg [E,C,d_r,d_m] then scatter_sum   <- what ships now
 (2) checkpointed : same, but recompute in backward (torch.utils.checkpoint)
 (3) fused        : custom autograd, chunked; A_msg NEVER exists in forward OR backward
The backward of (3) needs no A_msg because
     dL/dr[e,k,l] = sum_p g[recv(e),k,l,p] m[e,k,p]
     dL/dm[e,k,p] = sum_l g[recv(e),k,l,p] r[e,k,l]
"""

import os
import time

import torch
from torch.utils.checkpoint import checkpoint

dev = os.environ.get("BENCH_DEV", "cuda")
torch.set_default_dtype(torch.float64)


def naive(r, m, recv, n):
    a = torch.einsum("ekl,ekp->eklp", r, m)
    return torch.zeros(n, *a.shape[1:], dtype=a.dtype, device=a.device).index_add_(
        0, recv, a
    )


class Fused(torch.autograd.Function):
    @staticmethod
    def forward(ctx, r, m, recv, n, chunk):
        out = r.new_zeros(n, r.shape[1], r.shape[2], m.shape[2])
        for s in range(0, r.shape[0], chunk):
            e = slice(s, s + chunk)
            out.index_add_(0, recv[e], torch.einsum("ekl,ekp->eklp", r[e], m[e]))
        ctx.save_for_backward(r, m, recv)
        ctx.chunk = chunk
        return out

    @staticmethod
    def backward(ctx, g):
        r, m, recv = ctx.saved_tensors
        gr, gm = torch.zeros_like(r), torch.zeros_like(m)
        for s in range(0, r.shape[0], ctx.chunk):
            e = slice(s, s + ctx.chunk)
            ge = g[recv[e]]  # [chunk,C,d_r,d_m]
            gr[e] = torch.einsum("eklp,ekp->ekl", ge, m[e])
            gm[e] = torch.einsum("eklp,ekl->ekp", ge, r[e])
        return gr, gm, None, None, None


def run(fn, r, m, recv, n, tag):
    r = r.detach().requires_grad_(True)
    m = m.detach().requires_grad_(True)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    out = fn(r, m, recv, n)
    out.square().sum().backward()
    torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) * 1e3
    return dt, torch.cuda.max_memory_allocated() / 1e9, r.grad.clone(), m.grad.clone()


E, C, DR, DM, N = 16956, 128, 16, 9, 432
g = torch.Generator(device=dev).manual_seed(0)
r = torch.randn(E, C, DR, device=dev, generator=g)
m = torch.randn(E, C, DM, device=dev, generator=g)
recv = torch.randint(0, N, (E,), device=dev, generator=g)
print(
    f"E={E} C={C} d_r={DR} d_m={DM} N={N}  (A_msg would be "
    f"{E*C*DR*DM*8/1e9:.2f} GB, pooled_A {N*C*DR*DM*8/1e9:.3f} GB)\n"
)
print(f"{'variant':<28} {'fwd+bwd ms':>11} {'peak GB':>9} {'max|dgrad|':>12}")
t, mem, gr0, gm0 = run(naive, r, m, recv, N, "naive")
print(f"{'(1) naive einsum+scatter':<28} {t:11.1f} {mem:9.2f} {'--':>12}")
for ck in (8192, 2048):
    t, mem, gr, gm = run(
        lambda a, b, c, d, ck=ck: Fused.apply(a, b, c, d, ck), r, m, recv, N, "fused"
    )
    d = max(float((gr - gr0).abs().max()), float((gm - gm0).abs().max()))
    print(f"{f'(3) fused chunk={ck}':<28} {t:11.1f} {mem:9.2f} {d:12.2e}")
t, mem, gr, gm = run(
    lambda a, b, c, d: checkpoint(naive, a, b, c, d, use_reentrant=False),
    r,
    m,
    recv,
    N,
    "ckpt",
)
d = max(float((gr - gr0).abs().max()), float((gm - gm0).abs().max()))
print(f"{'(2) checkpointed':<28} {t:11.1f} {mem:9.2f} {d:12.2e}")
