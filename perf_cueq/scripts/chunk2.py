exec(
    open("/storage/data/jerry528/nonsoc_repro/bench/gpu_bench.py")
    .read()
    .split("with default_dtype")[0]
)
from mace.modules.symmetric_contraction_nonsoc import NonSOCContraction

with default_dtype(torch.float64):
    NELEM = 89
    pos = bulk_bcc(432)
    batch, n, ne = make_batch(pos, NELEM)
    print(f"merged contraction, {n} atoms / {ne} edges, max_m_ell=2\n")
    print(f"{'chunk_size':>10} {'fwd ms':>9} {'bwd ms':>9} {'peak GB':>9}")
    for chunk in (250, 128, 64, 32, 16, 8):
        m = build(NELEM, chunk=chunk)
        try:
            f, b, mem = run(m, batch, nrep=3)
            print(f"{chunk:>10} {f:9.1f} {b:9.1f} {mem:9.2f}")
        except torch.cuda.OutOfMemoryError:
            print(f"{chunk:>10} {'OOM':>9}")
            torch.cuda.empty_cache()
        del m
        torch.cuda.empty_cache()
