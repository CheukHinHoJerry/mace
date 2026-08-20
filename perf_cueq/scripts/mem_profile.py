"""Where does the 60 GB actually go? Record the allocator history and rank by size."""

import collections
import pickle
import sys

sys.path.insert(0, "/storage/data/jerry528/nonsoc_repro/bench")
exec(
    open("/storage/data/jerry528/nonsoc_repro/bench/gpu_bench.py")
    .read()
    .split("with default_dtype")[0]
)

with default_dtype(torch.float64):
    NELEM = 89
    pos = bulk_bcc(432)
    batch, n, ne = make_batch(pos, NELEM)
    model = build(NELEM)
    # warm up outside the recording
    out = model(batch, training=True, compute_force=False, compute_stress=False)
    out["energy"].sum().backward()
    model.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    torch.cuda.empty_cache()

    torch.cuda.memory._record_memory_history(max_entries=200000)
    out = model(batch, training=True, compute_force=False, compute_stress=False)
    torch.cuda.synchronize()
    # snapshot at END OF FORWARD: the autograd graph is fully live here, which is where
    # the peak sits. After backward the graph is freed and the snapshot shows nothing.
    snap = torch.cuda.memory._snapshot()
    fwd_peak = torch.cuda.max_memory_allocated() / 1e9
    out["energy"].sum().backward()
    torch.cuda.synchronize()
    print(f"forward-end allocated = {fwd_peak:.2f} GB")
    torch.cuda.memory._record_memory_history(enabled=None)
    print(
        f"peak = {torch.cuda.max_memory_allocated()/1e9:.2f} GB, {n} atoms, {ne} edges\n"
    )

    # aggregate live allocations by the frame that created them
    by_site = collections.defaultdict(lambda: [0, 0])
    for seg in snap["segments"]:
        for blk in seg["blocks"]:
            if blk["state"] != "active_allocated":
                continue
            frames = blk.get("frames") or []
            site = "<no frames>"
            for f in frames:
                fn = f.get("filename", "")
                if "mace/modules" in fn or "mace/tools" in fn:
                    site = f"{fn.split('mace/')[-1]}:{f.get('line')} {f.get('name')}"
                    break
            else:
                if frames:
                    f = frames[0]
                    site = f"{f.get('filename','?').split('/')[-1]}:{f.get('line')} {f.get('name')}"
            by_site[site][0] += blk["size"]
            by_site[site][1] += 1
    tot = sum(v[0] for v in by_site.values())
    print(f"{'live allocation site':<74} {'GB':>7} {'n':>6}")
    print("-" * 90)
    for site, (sz, cnt) in sorted(by_site.items(), key=lambda kv: -kv[1][0])[:18]:
        print(f"{site[:74]:<74} {sz/1e9:7.2f} {cnt:6d}")
    print("-" * 90)
    print(f"{'TOTAL live':<74} {tot/1e9:7.2f}")
