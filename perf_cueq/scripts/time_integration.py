"""Time the INTEGRATED contraction: use_cueq=True vs the chunked einsum baseline.

The einsum side runs WITH chunking, because that is the production configuration -- an
unchunked baseline OOMs at 432 nodes and would not be the thing cueq has to beat.
Reports forward, fwd+bwd, and fwd+bwd+double (the force loss needs the second derivative).
"""

import os
import statistics
import sys
import time

import torch
from e3nn import o3

sys.path.insert(0, "/home/jerry528/mace-ace-pr124/mace")
from mace.modules.symmetric_contraction_nonsoc import NonSOCSymmetricContraction

T0 = time.time()


def log(m):
    print(f"[{time.time()-T0:8.1f}s] {m}", flush=True)


DEV = "cuda"
DT = torch.float64 if os.environ.get("DTYPE", "64") == "64" else torch.float32
torch.set_default_dtype(DT)

MAX_ELL = int(os.environ.get("MAX_ELL", 3))
MAX_MELL = int(os.environ.get("MAX_MELL", 2))
CH = int(os.environ.get("CH", 128))
B = int(os.environ.get("BATCH", 432))
NU = int(os.environ.get("NU", 3))
NEL = int(os.environ.get("NEL", 2))
CHUNK = int(os.environ.get("CHUNK", 250))
REP = int(os.environ.get("REP", 5))

irreps_in = o3.Irreps(
    "+".join(f"{CH}x{l}{'e' if l%2==0 else 'o'}" for l in range(MAX_ELL + 1))
)
magmom_irreps = o3.Irreps(
    "+".join(f"1x{l}{'e' if l%2==0 else 'o'}" for l in range(MAX_MELL + 1))
)
dr = sum(2 * l + 1 for l in range(MAX_ELL + 1))
dm = sum(2 * l + 1 for l in range(MAX_MELL + 1))
log(
    f"max_ell={MAX_ELL} m_ell={MAX_MELL} nu={NU} ch={CH} B={B} dtype={DT} chunk={CHUNK} rep={REP}"
)
log(f"method={os.environ.get('CUEQ_METHOD','<default>')}  dr={dr} dm={dm}")


def build(use_cueq, chunk):
    torch.manual_seed(1234)
    return (
        NonSOCSymmetricContraction(
            irreps_in=irreps_in,
            irreps_out=o3.Irreps(f"{CH}x0e"),
            correlation=NU,
            num_elements=NEL,
            magmom_irreps=magmom_irreps,
            internal_weights=True,
            shared_weights=True,
            chunk_size=chunk,
            center_spin_coupling=True,
            use_cueq=use_cueq,
        )
        .to(DEV)
        .to(DT)
    )


t = time.time()
ref = build(False, CHUNK)
log(f"built einsum (chunk={CHUNK}) in {time.time()-t:.1f}s")
t = time.time()
cq = build(True, None)
log(
    f"built cueq in {time.time()-t:.1f}s "
    + ", ".join(
        f"nu{n}={cq.contractions[0].cueq_contractions[f'nu{n}'].num_paths}p"
        for n in range(1, NU + 1)
    )
)

y = torch.nn.functional.one_hot(torch.randint(0, NEL, (B,), device=DEV), NEL).to(DT)
mc = torch.randn(B, dm, device=DEV, dtype=DT)


def once(mod, order):
    x = torch.randn(B, CH, dr, dm, device=DEV, dtype=DT, requires_grad=(order > 0))
    out = mod(x, y, mc)
    if order == 0:
        return
    loss = out.square().sum()
    g = torch.autograd.grad(loss, x, create_graph=(order > 1))
    if order > 1:
        torch.autograd.grad(g[0].square().sum(), [p for p in mod.parameters()])


for name, order in (("forward", 0), ("fwd+bwd", 1), ("fwd+bwd+double", 2)):
    for tag, mod in (("cueq", cq), ("einsum", ref)):
        try:
            t = time.time()
            once(mod, order)
            torch.cuda.synchronize()
            log(f"  {name:15s} {tag:6s} warm-up/compile {time.time()-t:7.1f}s")
            ts = []
            for _ in range(REP):
                torch.cuda.synchronize()
                t = time.time()
                once(mod, order)
                torch.cuda.synchronize()
                ts.append(time.time() - t)
            log(
                f"  {name:15s} {tag:6s} median {statistics.median(ts)*1e3:9.2f} ms   "
                f"(min {min(ts)*1e3:.2f})"
            )
        except Exception as e:
            import traceback

            log(f"  {name:15s} {tag:6s} FAILED {type(e).__name__}")
            log(traceback.format_exc())
        torch.cuda.empty_cache()
log("done")
