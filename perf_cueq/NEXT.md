# Next steps (written 2026-08-19 19:0x, resume at 20:19 PDT)

## How to run: PERSISTENT allocation, not one-shot srun
One-shot `srun` dies whenever the driving session is interrupted (it killed job 2188290
mid-benchmark). Allocate once and reuse it -- also keeps the JIT/kernel cache warm, which
matters because the first `SegmentedPolynomial` call at 29,299 paths costs ~155 s to compile.

    salloc --partition=gpu -w gpu01 --gres=gpu:2 --cpus-per-task=8 --mem=200G --time=4:00:00
    srun --jobid=<ID> bash scripts/run_cueq.sh scripts/warm_bench.py

`--gres=gpu:2` is REQUIRED: gpu01's device 0 is faulty and every script here pins
CUDA_VISIBLE_DEVICES=1, so a 1-GPU allocation fails with "No CUDA GPUs are available".

## Status
DONE  cueq's repeated-buffer gradients are wrong (rel 2.9e+01 first order, 8.7e+01 second).
      Fix: one buffer per contraction slot, PyTorch accumulates. Exact at both orders
      (2.5e-16 / 3.6e-16). See README section 5.
DONE  Product-group decomposition verified exact on CPU (rel 1.0e-15), including folding
      the centre factor v[b,Q] into the weights.
DONE  mace/modules/cueq_nonsoc.py -- PairedContractionCueq + build_paired_contractions.
DONE  Construction wired into NonSOCContraction.__init__ behind use_cueq.
DONE  Production-scale forward: rel 1.369e-14 (max_ell=3, m_ell=2, ch=128, B=432).

TODO  1. `use_cueq` arg on NonSOCContraction.__init__ + thread from NonSOCSymmetricContraction.
      2. `_forward_cueq` branch (fold v into weights, permute x to (b,(i,l),a)).
         Falls back to einsum when center_spin_coupling=True but m_center is None -- the
         merged magmom basis is sliced in that case and would not match the built STP.
      3. END-TO-END VALIDATION (the goal): energy, forces AND magforces, cueq vs einsum,
         to numerical precision. Forces exercise double backward -- the whole point.
      4. Warm timings at production scale (job 2188291) to decide if this ships at all.

## CORRECTION: there is no use_cueq_cg bug (I read a stale worktree)

An earlier draft of this file claimed `use_cueq_cg=True` was hardcoded. WRONG -- that was
read from the prototype worktree at 509d855, which is 9 commits behind the live checkout
(78191e3). All three sites already pass `use_cueq_cg=False`, deliberately, since da82c19
"fix(nonsoc): span the full non-SOC invariant class".

The reason is worth knowing for THIS work: cueq's `compute_U_cueq` returns
`reduced_symmetric_tensor_product_basis`, a reduced basis of Sym^nu of ONE space. Pairing two
of those spans only Sym^nu(V_r) (x) Sym^nu(V_m) and DROPS the mixed Young sectors of
Sym^nu(V_r (x) V_m) = (+)_lambda S^lambda(V_r) (x) S^lambda(V_m). Measured 61 -> 40 paths at
max_ell=3,max_m_ell=2. The STP benchmarks here correctly use use_cueq_cg=False, so the paired
bases are the full unsymmetrised coupling trees. DO NOT "optimise" this to True.

LESSON: /storage/data/jerry528/nonsoc_repro/l1fix is a STALE worktree. Read
/home/jerry528/mace-ace-pr124/mace (branch nonsoc-minimal) for current code.

## Also corrected: L>0 hidden irreps are supported now

Commit 5cc1107 "keep the spin factor spin-pure, enabling L>0 hidden irreps" restricts
conv_tp_m's first input to the l=0 part of the node features, and REMOVED the lmax>0
rejection guard. Earlier statements in this session that L=1 is rejected / that the fix is
unapplied were based on the stale worktree and are wrong.
TODO verify empirically at HEAD: build 128x0e+128x1o and check E(Rr,m)=E(r,Sm)=E(-r,m)=E(r,-m).
     The docstring at extensions.py:2204 still says "Requires scalar hidden_irreps" -- likely
     stale after 5cc1107; confirm and fix if so.

## TODO: does a newer cuequivariance behave better?

Test cueq 0.11.1 (already installed at /storage/data/jerry528/cueq_test_011; 0.5.1 is at
/storage/data/jerry528/cueq_test) for:
  1. Is the repeated-buffer gradient bug FIXED in 0.11.1? Run scripts/grad_matrix.py under it
     -- if `shared` now matches, report the 0.5.1 bug upstream; if not, report it against 0.11.1.
  2. Is it faster / does it compile faster? The 155 s first-call compile at 29,299 paths is the
     main practical cost, and 0.11.1 has a `method=` arg (naive | uniform_1d | ...) that 0.5.1
     lacks -- worth sweeping.
  3. `method=naive` is pure PyTorch: useful to separate an encoding bug from a CUDA-kernel bug.
Runner: scripts/run011_naive.sh / run011_uniform.sh already point at the 0.11.1 venv.

## RESULT 2026-08-19 ~19:45 — production warm benchmark FAILED (job 2188291)

At `max_ell=3 m_ell=2 nu=3 ch=128 B=432` (dr=16 dm=9 P=23 Q=11, 29,299 STP paths), cueq 0.5.1:

- forward           OK, rel=1.369e-14
- backward          **RuntimeError in `cuequivariance_ops/cuda/equivariance/tensor_product_uniform_1d_jit...`**
- double backward   same RuntimeError
- einsum reference  CUDA OOM (21.83 GiB) — harness ran with chunk_size=None

**No timings were produced. The speedup question is still completely open.**

Compile cost measured: ~36 min single-threaded (99.6% of one core, 37 GB RSS) before the
backward even attempted to run. That is a shipping concern on its own, separate from speed.

The error text was truncated to 120 chars by warm_bench.py's own logging — now patched to
print the full traceback, so a rerun is not wasted. **Get the full traceback before theorising
about the cause.**

Next, cheapest-first:
1. `scripts/validate_integration.py` at SMALL scale (float64) — does the backward work at all?
   If small passes and production fails, it is a scale/JIT limit, not a wrong decomposition.
2. Re-run production with the traceback patch to see the actual kernel error.
3. cueq 0.11.1 (`/storage/data/jerry528/cueq_test_011`): does it fix this AND the
   repeated-buffer gradient bug? Sweep `method=naive` vs `uniform_1d` — the failing kernel IS
   uniform_1d, so `naive` may sidestep it.

## VALIDATED 2026-08-19 ~20:50 — integration is numerically exact (cueq 0.11.1, method=naive)

`scripts/validate_integration.py`, float64, max_ell=2 m_ell=1 nu=3 ch=16 B=12, centre-spin ON.
Weights copied tensor-by-tensor from the einsum reference (NOT load_state_dict — cueq >= 0.11
registers STP coefficients as buffers, so strict loading fails on legitimately-extra keys).

    forward                   rel=3.217e-15
    d loss/dx                 rel=2.601e-16
    d loss/d m_center         rel=2.572e-16
    d loss/d w[0..2]          rel=0.000e+00 .. 2.706e-16
    d2 loss/dx2 (force-like)  rel=5.020e-16      <-- the one the force loss needs
    d2 loss/dx dw[0..2]       rel=0.000e+00 .. 3.458e-16

Settles three things:
- the paired product-group decomposition is correct;
- `_forward_cueq` is correct (v folded into weights + the (b,a,i,l)->(b,(i,l),a) permute);
- passing the SAME `x_flat` object into nu DISTINCT buffers is safe. The earlier gradient bug
  was a repeated *buffer* in the Operation, not a repeated tensor object. No clone needed.

STP build cost at this tiny scale: 35 s (nu1=4, nu2=90, nu3=5063 paths).

### Harness bugs found and fixed (both mine, not the backend's)
- `m_center` is the FULL dm-dimensional flat spherical embedding (block s at offset s**2),
  because centre_cols = cat([ones, m_center]) is indexed as `1 + s**2 + comp`. I had passed
  dm-1 columns, which read out of bounds -> device-side assert.
- `srun` without `--ntasks=1` launched TWO tasks that competed for the GPU and doubled all
  output. Always pass `--ntasks=1` when using `--overlap` against the persistent allocation.

### Still open
- Does the FAST kernel (uniform_1d) agree, and does it survive at production scale? The
  production failure WAS in uniform_1d, and `naive` is a different code path, so this result
  does NOT clear the kernel.
- No speedup number exists yet. Nothing about performance has been measured successfully.

## 2026-08-19 ~21:15 — no-centre branch also exact; kernel COMPILE TIME is the real risk

`CENTER=0` (naive, same shapes): all checks exact, best 4.6e-16 / worst 3.968e-16 on the
force-like double backward. So BOTH branches of `_forward_cueq` are validated — including my
masking substitution (v=0 on the s>=1 columns) for the reference's column-slicing.

### method=naive vs method=uniform_1d (cueq 0.11.1, IDENTICAL tiny shapes, 5063 paths at nu=3)

| stage          | naive  | uniform_1d |
|----------------|--------|------------|
| forward        |  0.4 s |    5.9 s   |
| first backward |  1.5 s |  341 s     |
| double backward|  1.6 s |  >12 min, still compiling at 43 min CPU-time / 35 GB RSS |

Both are numerically exact where they completed. `uniform_1d` forward+backward agree to
~1e-16..3e-15, same as naive.

**The blocker is now compile time, not correctness.** And it does NOT amortise across runs:
`~/.cache/cuequivariance-triton` stays EMPTY (0 files), because uniform_1d compiles through
NVRTC, not Triton. `CUEQ_TRITON_CACHE_DIR` therefore does not help this path.

Extrapolation to production is bad: 29,299 paths vs 5,063 here, and 0.5.1 already spent 36 min
before its backward FAILED outright. RSS also climbs steadily during the compile (14 -> 35 GB),
so an OOM during compilation is a live risk at production scale.

### Consequence for the shipping decision
- `naive` is exact and compiles instantly, but is pure PyTorch — it is NOT obviously faster
  than the chunked einsum. Needs an actual timing before it can be called a win.
- `uniform_1d` is the only candidate for a real speedup, but may be uncompilable at the shapes
  we care about. A one-off 40-min compile is acceptable for a multi-day training run ONLY if it
  (a) completes, (b) fits in memory, and (c) is then actually faster.
- Nothing has been timed successfully yet at ANY scale. The speedup remains unmeasured.

## 2026-08-19 ~21:40 — uniform_1d (fast kernel) is EXACT under 0.11.1

Same tiny shapes, `CUEQ_METHOD=uniform_1d`, cueq 0.11.1, float64, centre-spin ON:

    forward                   rel=3.088e-15
    d loss/dx                 rel=1.301e-16
    d loss/d m_center         rel=2.572e-16
    d loss/d w[0..2]          rel=0.000e+00 .. 2.556e-16
    d2 loss/dx2 (force-like)  rel=4.462e-16
    d2 loss/dx dw[0..2]       rel=0.000e+00 .. 6.051e-16
    ALL CHECKS PASSED

**Answer to "does a newer cuequivariance behave better?": YES, and it is required.**
0.5.1's backward RuntimeError in `tensor_product_uniform_1d_jit` does not reproduce on 0.11.1.
0.11.1 also adds the `method=` argument, which 0.5.1 lacks entirely -- `cueq_nonsoc.py` now
passes it only when the installed build's signature has it, and reads a `CUEQ_METHOD` default
from the environment.

Caveat: this was verified at 5,063 paths, NOT at the 29,299-path production shape where 0.5.1
failed. Version and scale are still confounded. Do not claim 0.11.1 fixes production until the
production shape actually compiles.

Compile cost, uniform_1d, tiny shapes: forward 5.9 s, first backward 341 s, double backward
1632 s. Total 33 min. All three are exact; none is cached to disk.

### Both correctness questions are now CLOSED
The decomposition, the integration, and both `_forward_cueq` branches are exact at both
derivative orders under both methods. What remains is entirely a performance/viability
question: is it FASTER than the chunked einsum, and does the production shape compile at all?

## 2026-08-19 ~22:10 — build-time bug fixed; `naive` looks NON-VIABLE at production scale

### Bug in my own code (fixed)
`_nonzeros(u_magmom, nu)` sat in the INNER loop header of the path-construction double loop,
so it was recomputed once per spatial nonzero (353x at production), and it indexed the tensor
one Python-level element at a time. Now hoisted and vectorised via `.nonzero()`.

### Path counts are NOT the problem (measured, production shape max_ell=3 / m_ell=2)
    nu=1: U_spatial(16,1)        nnz=1     U_magmom(9,1)       nnz=1    -> 1 path
    nu=2: U_spatial(16,16,4)     nnz=16    U_magmom(9,9,3)     nnz=9    -> 144 paths
    nu=3: U_spatial(16,16,16,23) nnz=353   U_magmom(9,9,9,11)  nnz=83   -> 29,299 paths
29,299 is modest. There is no combinatorial explosion; the STP itself is small.

### `method=naive` at production scale: >11 min single-threaded and 21 GB RSS, still inside
`cuet.SegmentedPolynomial(...)` CONSTRUCTION -- it had not even reached a forward pass.
Instrumented with `CUEQ_NONSOC_VERBOSE=1` (prints per-nu stp_build vs poly_build) to confirm
which phase, but the shape of it is clear: naive appears to unroll the polynomial into a torch
graph that does not scale to 29k paths. Exact, but likely unusable here.

That leaves `uniform_1d` as the ONLY candidate for a production speedup. Submitted as detached
sbatch job 2188294 (12 h limit, NOT gpu01, `CUEQ_NONSOC_VERBOSE=1`) -> results/time_uniform_prod.txt.
Detached deliberately: an earlier srun-attached benchmark was killed as collateral from a
session interrupt.

### Still no speedup number. Every performance claim remains unmeasured.

## 2026-08-19 ~22:45 — CORRECTION: production is 552,798 paths, not 29,299

The 29,299 figure above is WRONG for the real module. It came from a standalone measurement
using the RAW magmom basis `U_magmom_{nu}` (9,9,9,11). The module actually contracts against
`U_magmom_merged_{nu}`, the merged Q layout, which stacks the s=0 columns TOGETHER with the
centre-spin s=1 and s=2 blocks. Measured from the running job:

    nu=1 paths=9        stp_build=0.0s   poly_build=0.0s
    nu=2 paths=1,328    stp_build=0.1s   poly_build=0.0s
    nu=3 paths=552,798  stp_build=464.2s poly_build=5.1s

So ~19x more paths than I estimated, and the earlier "29,299 is modest, no combinatorial
explosion" conclusion does not hold at production. `stp_build` (my Python loop calling
`stp.add_path` 552,798 times) is now the dominant construction cost at 7.7 min, even after
the vectorisation fix -- worth optimising if this ships, e.g. a batched path-add API.

Note the 0.5.1 production benchmark that "failed after 36 min" was also at 552,798 paths, not
29,299. Its 36 min was mostly this same STP build plus the kernel compile.

### `method=naive` at production: KILLED, non-viable
Reached 156 GB RSS and 42 min still inside `SegmentedPolynomial` construction, never running a
forward. Killed before it took the node down. naive is exact but cannot build this polynomial.
`uniform_1d` builds the same polynomial in 5.1s (poly_build) -- the two methods are not
remotely comparable at scale.

`uniform_1d` production job 2188294 is past construction (469.7s total) and into kernel compile.

## 2026-08-19 ~23:40 — VERDICT: cueq does NOT scale to this problem's path count

Job 2188294, `uniform_1d`, production shape (max_ell=3 m_ell=2 ch=128 B=432, float64):

    State=OUT_OF_MEMORY  MaxRSS=196 GB  Elapsed=1:49:23  ExitCode 0:125

It died while still compiling the FORWARD kernel. It never reached the backward, never reached
the double backward, and produced ZERO timings. Construction itself was fine (469.7 s).

Combined with `method=naive` (killed at 156 GB, 42 min, still inside SegmentedPolynomial
construction, also never reached a forward), BOTH cueq methods fail at 552,798 paths:

| method     | outcome at production shape                                  |
|------------|--------------------------------------------------------------|
| naive      | 156 GB RSS, never finished CONSTRUCTION                       |
| uniform_1d | 196 GB RSS OOM after 1h49m, never finished FORWARD COMPILE    |

**There is still no speedup number, and on this evidence there may never be one.** The
integration is numerically exact but currently unusable at the shapes we actually train at.

### This is NOT a "0.5.1 vs 0.11.1" problem
0.11.1 genuinely fixed the backward correctness bug (verified exact at 5,063 paths). The
remaining wall is compile time/memory vs path count, and it is present in 0.11.1 too.

### Where the 552,798 comes from, and the one idea left
paths(nu=3) = (spatial nnz 353) x (MERGED magmom nnz). The merged Q layout stacks s=0, s=1 and
s=2 into ONE contraction -- that was the right call for the einsum (one contraction instead of
one per s) but it is what multiplies the STP path count.

Idea worth testing: split back into ONE STP PER CENTRE-SPIN SECTOR s. Total path count is
unchanged, so this only helps if compile cost is SUPERLINEAR in paths. Job 2188295 (scaling
sweep over max_ell/m_ell, gpu01) is measuring exactly that curve. If compile is ~linear,
splitting buys nothing and cueq should be abandoned for this contraction.

### Fallback that already works
The chunked einsum path is correct, fits in memory, and is what the model trains with today.
Nothing here threatens it; the cueq backend is opt-in via `use_cueq=False` by default.

## 2026-08-20 ~00:00 — FIRST REAL TIMINGS: the speedup is ~1.0x. cueq is not worth it.

Job 2188295, `uniform_1d`, cueq 0.11.1, float64, ch=128 B=432, einsum baseline CHUNKED (250),
at max_ell=2 / m_ell=1 => nu3 paths=5,063 (stp_build 0.3 s, poly_build 0.0 s):

| stage   | cueq median | einsum median | speedup | cueq compile |
|---------|-------------|---------------|---------|--------------|
| forward |    8.17 ms  |     9.18 ms   |  1.12x  |     7.8 s    |
| fwd+bwd |   16.97 ms  |    17.45 ms   |  1.03x  |   270.3 s    |

**This is the number the whole exercise was for, and it is ~1.0x.** fwd+bwd -- the thing
training actually does -- is 3% faster, for 270 s of one-time compile. The forward-only 12% is
irrelevant since training never runs forward alone.

Note the einsum baseline compiles in 0.0-0.1 s and has no memory cliff.

### Overall conclusion so far
- correctness: DONE, exact to machine precision at both derivative orders, both branches.
- production scale: cueq cannot run at all (OOM at 196 GB / 156 GB, both methods).
- small scale where it does run: ~1.03x on fwd+bwd.

Unless the remaining sweep points (max_ell=3/m_ell=1, max_ell=2/m_ell=2) show the gap WIDENING
sharply with path count, the honest recommendation is: **do not ship the cueq backend**; keep
it opt-in and default-off, and keep the chunked einsum as the production path.

Still pending in the sweep: double backward at this shape, then the two larger shapes.

## 2026-08-20 ~00:20 — the double backward is fatal even at the SMALLEST config

Job 2188295, max_ell=2 / m_ell=1, only 5,063 paths, but production batch/width (B=432, ch=128):
the `fwd+bwd+double` compile has now run **1h54m** at 58 GB RSS and climbing, 451% CPU, and has
not produced a timing. The per-config `timeout 7200` will cut it off at 2h.

Compare: the SAME 5,063 paths compiled the double backward in 1632 s at B=12 / ch=16. So the
compile cost blows up with batch and channel width too, not just path count.

**This kills the last remaining escape route.** I had floated splitting the merged Q layout
into one STP per centre-spin sector s, on the theory that compile cost is superlinear in paths.
It is superlinear -- but that idea cannot work, because even 5,063 paths (roughly 1/100th of
production's 552,798, and far smaller than any per-sector split would give) ALREADY fails to
compile the double backward at production batch/width. There is no path-count reduction that
reaches a viable regime.

And the double backward is not optional: the training loss is on FORCES, which are themselves
autograd derivatives.

# ============================ FINAL VERDICT ============================
# The cueq backend is CORRECT and NOT VIABLE. Do not ship it as the default.
#
#   correctness   exact to machine precision, both derivative orders, both
#                 _forward_cueq branches, cueq 0.11.1 (0.5.1's backward is broken)
#   speed         1.03x on fwd+bwd where it runs at all (1.12x forward-only, irrelevant)
#   production    cannot run: OOM at 196 GB (uniform_1d) / 156 GB (naive)
#   double bwd    >2 h compile at production batch/width even at 5,063 paths
#
# Keep `use_cueq=False` as the default. Keep the chunked einsum as the production path.
# The backend is worth KEEPING in-tree, opt-in and validated, so that if a future cueq
# release fixes compile scaling the work is already done and proven.
# =======================================================================

## 2026-08-20 ~00:35 — the trend REVERSES: cueq gets slower as paths grow

Config 1 (max_ell=2/m_ell=1) double backward: killed by the 2 h `timeout`, no result.

Config 2, max_ell=3 / m_ell=1, nu3 paths=21,533 (stp_build 1.6 s, poly_build 0.3 s):

    forward   cueq  30.77 ms   einsum 12.70 ms   ->  cueq 2.42x SLOWER   (compile 54.5 s)

Scaling of cueq forward vs the chunked einsum:

| nu=3 paths | cueq forward |
|------------|--------------|
|      5,063 | 1.12x FASTER |
|     21,533 | 2.42x SLOWER |
|    552,798 | OOM, cannot run at all |

This answers the one question left open in the verdict above. The 1.03x-1.12x at 5,063 paths
was not a small win waiting to grow with scale -- it is the peak, and the curve turns against
cueq immediately after. Production is 552,798 paths, ~26x past the point where cueq is already
2.4x slower.

**The FINAL VERDICT above stands, now supported by measurement rather than extrapolation.**

### Regression check: the production path is untouched
`use_cueq` defaults to False at both call sites (symmetric_contraction_nonsoc.py:135 and :216).
    pytest test_nonsoc_center_spin.py test_nonsoc_chunking.py test_nonsoc_equivariance.py
           test_nonsoc_parity.py  ->  132 passed, 1 skipped (332 s, CPU)
