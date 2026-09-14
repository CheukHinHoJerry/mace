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
    print(f"{n} atoms / {ne} edges, max_m_ell=2\n")
    print(f"{'variant':<34} {'fwd ms':>9} {'bwd ms':>9} {'peak GB':>9}")
    for label, chunk, recomp in (
        ("baseline chunk=250", 250, False),
        ("recompute chunk=250", 250, True),
        ("recompute chunk=64", 64, True),
        ("recompute chunk=32", 32, True),
    ):
        m = build(NELEM, chunk=chunk)
        for mod in m.modules():
            if isinstance(mod, NonSOCContraction):
                mod.recompute_chunks = recomp
        try:
            f, b, mem = run(m, batch, nrep=3)
            print(f"{label:<34} {f:9.1f} {b:9.1f} {mem:9.2f}")
        except torch.cuda.OutOfMemoryError:
            print(f"{label:<34} {'OOM':>9}")
            torch.cuda.empty_cache()
        del m
        torch.cuda.empty_cache()
