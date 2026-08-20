"""Go/no-go: backward and DOUBLE backward through cuet.SegmentedPolynomial.

Training's loss is on forces, which are themselves autograd derivatives, so the op must
be twice differentiable. Also checks gradient agreement with the einsum reference.
"""

exec(
    open("/storage/data/jerry528/nonsoc_repro/bench/cueq_bench.py")
    .read()
    .split("def bench(")[0]
)
import statistics
import time


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


max_ell, max_m_ell, nu, ch, B = 3, 2, 3, 128, 432
stp, poly, Us, Um, dr, dm, P, Q = build(max_ell, max_m_ell, nu, ch)
mod = cuet.SegmentedPolynomial(poly, math_dtype=torch.float64).to(DEV)
g = torch.Generator(device=DEV).manual_seed(0)
W0 = torch.randn(B, P * Q * ch, device=DEV, generator=g)
A0 = torch.randn(B, dr * dm * ch, device=DEV, generator=g)
eq = "ijfk,lmgq,bkqa,bila,bjma,bfga->ba"
print(f"paths={stp.num_paths}  nodes={B} ch={ch}\n")


def grads(kind, second):
    W = W0.clone().requires_grad_(True)
    A = A0.clone().requires_grad_(True)
    if kind == "cueq":
        out = mod([W] + [A] * nu)[0].reshape(B, ch)
    else:
        out = torch.einsum(
            eq, Us, Um, W.reshape(B, P, Q, ch), *([A.reshape(B, dr, dm, ch)] * nu)
        )
    if not second:
        out.sum().backward()
        return A.grad.clone(), W.grad.clone()
    # double backward: differentiate a function OF the first derivative
    (gA,) = torch.autograd.grad(out.sum(), A, create_graph=True)
    (gA**2).sum().backward()
    return A.grad.clone(), W.grad.clone()


for second in (False, True):
    tag = "DOUBLE backward (force-like)" if second else "first backward"
    try:
        ga_c, gw_c = grads("cueq", second)
        ga_e, gw_e = grads("einsum", second)
        d = max(float((ga_c - ga_e).abs().max()), float((gw_c - gw_e).abs().max()))
        print(
            f"  {tag:<30} max|cueq - einsum| = {d:.3e}   "
            f"{'OK' if d < 1e-8 else '** MISMATCH **'}"
        )
    except Exception as e:
        print(f"  {tag:<30} ** FAILED ** {type(e).__name__}: {str(e)[:120]}")

print()


def fb(kind):
    def f():
        W = W0.clone().requires_grad_(True)
        A = A0.clone().requires_grad_(True)
        out = (
            mod([W] + [A] * nu)[0]
            if kind == "cueq"
            else torch.einsum(
                eq, Us, Um, W.reshape(B, P, Q, ch), *([A.reshape(B, dr, dm, ch)] * nu)
            )
        )
        out.sum().backward()

    return f


tc, mc = bench(fb("cueq"))
te, me = bench(fb("einsum"))
print(f"  cueq   fwd+bwd {tc:8.2f} ms   peak {mc:6.2f} GB")
print(
    f"  einsum fwd+bwd {te:8.2f} ms   peak {me:6.2f} GB   -> cueq {te/tc:.2f}x faster, "
    f"{me/mc:.0f}x less memory"
)
