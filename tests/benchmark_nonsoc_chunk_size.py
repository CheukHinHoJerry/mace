#!/usr/bin/env python3
"""Benchmark node-chunk size for the non-SOC symmetric contraction on the current GPU.

Builds the contraction the bader per-layer model uses at its non-SOC layer
(irreps_in = 128x{0e,1o,2e,3o}, irrep_out = 0e, correlation 3, 89 elements, magmom
1x0e+1x1o), float64 to match --default_dtype, and times a fwd+bwd step for a sweep of
chunk sizes at a few realistic node counts N. Prints ms/step and peak GB; flags OOM.

Run on an H100:  srun --jobid <ALLOC> --overlap --pty bash -lc '... python tests/benchmark_nonsoc_chunk_size.py'
"""
import time

import torch
from e3nn import o3

from mace.modules.symmetric_contraction_nonsoc import NonSOCContraction

torch.set_default_dtype(torch.float64)  # training runs --default_dtype=float64
DEV = "cuda"

c = NonSOCContraction(
    irreps_in=o3.Irreps("128x0e+128x1o+128x2e+128x3o"),
    irrep_out=o3.Irreps("0e"),
    correlation=3,
    num_elements=89,
    magmom_irreps=o3.Irreps("1x0e+1x1o"),
).to(DEV)
S = c.U_tensors(1).shape[0]
M = c.U_magmom_tensors(1).shape[0]
NF = 128
print(f"device={torch.cuda.get_device_name(0)}  spatial(s)={S}  magmom(m)={M}  nf={NF}  dtype=float64")
print(f"per-node joint scratch ~ nf*s^k*m^k floats; this is what chunking bounds.\n")

CHUNKS = [125, 250, 500, 750, 1000, 1500, 2000, 4000, None]
NODES = [1000, 2000, 4000]


def run(N, chunk, iters=12, warmup=3):
    c.chunk_size = chunk
    x = torch.randn(N, NF, S, M, device=DEV, requires_grad=True)
    y = torch.nn.functional.one_hot(torch.randint(0, 89, (N,), device=DEV), 89).to(
        torch.get_default_dtype()
    )
    torch.cuda.reset_peak_memory_stats()
    for _ in range(warmup):
        out = c(x, y)
        out.pow(2).sum().backward()
        x.grad = None
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(iters):
        out = c(x, y)
        out.pow(2).sum().backward()
        x.grad = None
    torch.cuda.synchronize()
    ms = (time.time() - t0) / iters * 1e3
    peak = torch.cuda.max_memory_allocated() / 1e9
    del x, y
    torch.cuda.empty_cache()
    return ms, peak


for N in NODES:
    print(f"=== N = {N} nodes ===")
    best = (None, 1e18)
    for chunk in CHUNKS:
        label = "single-shot" if chunk is None else f"chunk={chunk}"
        try:
            ms, peak = run(N, chunk)
            tag = ""
            if ms < best[1]:
                best = (chunk, ms)
            print(f"  {label:14s}: {ms:8.1f} ms/step   peak {peak:6.2f} GB")
        except RuntimeError as e:
            torch.cuda.empty_cache()
            msg = "OOM" if "out of memory" in str(e).lower() else str(e)[:60]
            print(f"  {label:14s}: {msg}")
    print(f"  -> fastest: chunk={best[0]} ({best[1]:.1f} ms/step)\n")
