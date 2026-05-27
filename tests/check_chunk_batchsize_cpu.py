"""CPU check: the node-chunked non-SOC contraction is batch-size independent.

The contraction is a per-node map: output[i] must depend only on (x[i], y[i]), never on
how many nodes share the batch or how they are chunked. We compute a per-node reference
(each node alone) and assert that every grouping -- different total batch sizes, different
chunk_size -- reproduces it exactly. Run: python tests/check_chunk_batchsize_cpu.py
"""
import torch
from e3nn import o3

from mace.modules.symmetric_contraction_nonsoc import NonSOCContraction


def main():
    torch.set_default_dtype(torch.float64)
    torch.manual_seed(0)
    nf = 8
    c = NonSOCContraction(
        irreps_in=o3.Irreps(f"{nf}x0e+{nf}x1o+{nf}x2e"),
        irrep_out=o3.Irreps("0e"),
        correlation=3,
        num_elements=2,
        magmom_irreps=o3.Irreps("1x0e+1x1o"),
    ).to("cpu")
    s = c.U_tensors(1).shape[0]
    m = c.U_magmom_tensors(1).shape[0]

    N = 137  # deliberately not a multiple of any chunk/batch below
    x = torch.randn(N, nf, s, m)
    y = torch.nn.functional.one_hot(torch.randint(0, 2, (N,)), 2).double()

    # Per-node reference: each node computed entirely alone (batch size 1, no chunking).
    c.chunk_size = None
    ref = torch.cat([c(x[i : i + 1], y[i : i + 1]) for i in range(N)], dim=0)

    worst = 0.0
    for chunk_size in [None, 1, 13, 64, 1000]:
        c.chunk_size = chunk_size
        for batch in [1, 5, 32, 50, 100, 137]:
            # Process the full set in dataloader-style batches of `batch`, concatenate.
            out = torch.cat(
                [c(x[i : i + batch], y[i : i + batch]) for i in range(0, N, batch)],
                dim=0,
            )
            d = (out - ref).abs().max().item()
            worst = max(worst, d)
            assert torch.allclose(out, ref, atol=1e-10), (
                f"MISMATCH chunk_size={chunk_size} batch={batch}: max|diff|={d}"
            )
    print(f"device=cpu N={N} s={s} m={m}")
    print(f"OK: per-node output identical across all chunk_size x batch_size combos")
    print(f"worst max|diff| vs per-node reference = {worst:.3e}")


if __name__ == "__main__":
    main()
