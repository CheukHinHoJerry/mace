# Non-SOC performance investigation: where the memory goes, and can cuEquivariance help

Branch `nonsoc-minimal`. All numbers H100 NVL, float64, 432 atoms / 16956 edges,
128 channels, max_ell=3, max_m_ell=2, correlation=3, unless stated otherwise.

## 1. There is exactly ONE bottleneck

Allocation profile at end of forward (`scripts/mem_profile.py`), 47.06 GB live:

| site                                                | GB    | share |
|-----------------------------------------------------|-------|-------|
| **`symmetric_contraction_nonsoc.py:495`** (the contraction) | **44.90** | **95%** |
| `irreps_tools.py:107`                                | 0.44  | 0.9%  |
| `blocks.py:1303`                                     | 0.26  | 0.6%  |
| `blocks.py:778`                                      | 0.15  | 0.3%  |
| `blocks.py:2335` (the A_msg region)                  | 0.12  | 0.3%  |
| everything else                                      | <0.1  | --    |

"Optimise the bottlenecks one by one" has ONE entry. The next largest site is 100x
smaller. Screening the rest of the model for cueq opportunities is not worthwhile.

## 2. Two things that did NOT work, and why

**Fusing the per-edge outer product (`scripts/fused_scatter.py`).** `A_msg` is the largest
single tensor I could see (2.5 GB), so I wrote a custom autograd Function that never
materialises it. In isolation it is excellent:

| variant                   | fwd+bwd  | peak    |
|---------------------------|----------|---------|
| naive einsum + scatter     | 248.5 ms | 8.28 GB |
| **fused, chunk=2048**      | **12.5 ms** | **2.50 GB** |
| torch.utils.checkpoint     | 1455.4 ms| 9.18 GB |

End-to-end it changed **nothing**: 108.1 -> 109.5 ms forward, 60.04 -> 59.92 GB peak.
`A_msg` is only ~14% of the peak. REVERTED. The lesson is that the largest visible tensor
was not the peak driver, and only the allocator profile settled it.

**Lowering `chunk_size`.** Memory plateaus at ~48 GB no matter how small the chunk
(`scripts/chunk2.py`). Chunking bounds the TRANSIENT, not the retained autograd graph:
every chunk's intermediates are still saved.

## 3. What did work: recomputing chunks in backward

`scripts/ckpt.py`. Opt-in `recompute_chunks` flag on `NonSOCContraction`:

| variant             | fwd      | bwd      | peak       |
|---------------------|----------|----------|------------|
| baseline chunk=250  | 109.5 ms | 304.9 ms | **59.93 GB** |
| recompute chunk=250 | 110.2 ms | 423.5 ms | 46.03 GB   |
| recompute chunk=64  | 114.6 ms | 435.7 ms | 13.48 GB   |
| recompute chunk=32  | 114.5 ms | 442.3 ms | **7.88 GB**  |

**7.6x less memory for +43% backward.** Gradients bit-identical (0.000e+00) for an energy
loss AND a force loss, so double backward is safe. Currently uncommitted.

## 4. cuEquivariance: the contraction as a product-group SegmentedPolynomial

`scripts/cueq_bench.py`. The paired index `(l_spatial, l'_spin)` is treated as ONE irrep
label of `O(3)_space x SO(3)_spin`; segments are the paired components and each path's
coefficient is the PRODUCT of the spatial and magnetic CG entries:

```python
stp = cue.SegmentedTensorProduct.from_subscripts(",".join(["u"]*(nu+2)))
for _ in range(P*Q):        stp.add_segment(0, (ch,))       # weights, one per (k,q)
for op in range(1, nu+1):
    for _ in range(dr*dm):  stp.add_segment(op, (ch,))      # PAIRED components
stp.add_segment(nu+1, (ch,))                                 # scalar output per channel

for ri, k, cr in nzr:                    # nonzeros of the spatial CG
    for mi, q, cm in nzm:                # nonzeros of the magnetic CG
        stp.add_path(k*Q+q, *[ri[t]*dm+mi[t] for t in range(nu)], 0, c=cr*cm)

poly = cue.SegmentedPolynomial(inputs=..., outputs=...,
        operations=[(cue.Operation([0]+[1]*nu+[nu+1]), stp)])
```

The joint permutation symmetry is preserved because the SAME operand is repeated `nu`
times (`[1]*nu`), exactly as in the einsum.

### Path counts (`scripts/cueq_paths.py`)

cueq emits one path per nonzero CG entry. Validated: `nnz(Us)=353` matches cueq's own
`num_paths=353` for the same irreps at degree 3.

| case                   | nu | nnz(Us) | nnz(Um) | STP paths |
|------------------------|----|---------|---------|-----------|
| max_ell=2, m_ell=1     | 3  | 83      | 10      | 830       |
| **max_ell=3, m_ell=2** | 3  | 353     | 83      | **29,299** |

83x more than the single-group case cueq is tuned for. I expected this to sink the idea.
It did not.

### Measured (forward only, so far)

| max_ell=3, m_ell=2, nu=3 | forward   | peak    |
|--------------------------|-----------|---------|
| **cueq STP**             | **38.40 ms** | **0.21 GB** |
| einsum (ships today)     | 101.47 ms | 50.09 GB |
| ratio                    | **2.64x faster** | **239x less memory** |

Agreement with the einsum reference: 2.98e-12.

## 5. The gate: RESOLVED. cueq mishandles a REPEATED operand; give each slot its own buffer.

The training loss is on FORCES, which are autograd derivatives, so the contraction must be
TWICE differentiable. Measured (`scripts/cueq_grad_l1.py`, then `scripts/grad_matrix.py`):

The original encoding declares `nu+1` input buffers but wires the operation as
`Operation([0] + [1]*nu + [out])` -- buffer 1 repeated `nu` times, buffers 2..nu unused. That
hands the slot-gradient accumulation to cueq, and cueq gets it WRONG:

| encoding                                   | forward | first bwd | double bwd |
|--------------------------------------------|---------|-----------|------------|
| `shared`   `Operation([0,1,1,...,1,out])`   | ok      | rel 2.9e+01 ** | rel 8.7e+01 ** |
| `distinct` `Operation([0,1,2,...,nu,out])`  | ok      | rel 2.5e-16 OK | rel 3.6e-16 OK |

(max_ell=2, m_ell=1, nu=3, ch=32, B=64, cueq 0.5.1, float64.)

**The fix is to pass each contraction slot as its own buffer** and let PyTorch -- not cueq --
accumulate the `nu` slot gradients (feed `A*1.0` per slot so each is a distinct graph node).
Both derivative orders then agree with the einsum reference to machine precision.

Note the `shared` encoding is wrong at FIRST order too. An earlier run at max_ell=3,m_ell=1
happened to show 1.137e-13 first-backward agreement; that was a coincidence of that
truncation, not a working first derivative. Do not trust a single-configuration gradient check.

The product-group decomposition itself is exact and was never the problem -- verified
independently on CPU: folding the centre factor `v[b,Q]` into the weights and flattening
`(i,l) -> i*dm+l` with coefficient `Us*Um` reproduces the production einsum to rel 1.0e-15.

Also open:
* STP construction is slow (29,299 Python `add_path` calls). Build-time only, but the
  descriptor must be cached, not rebuilt.
* Integration is not started. The benchmark contracts a bare `A`; the real path also
  carries the centre-moment factor `v[b,Q]`, species-gathered weights, and three nu terms.
* `cuequivariance-ops-torch-cu12` is NOT in the training env. It lives in an isolated venv
  (`/storage/data/jerry528/cueq_test`) so the running job's environment stays untouched.
  Training with cueq needs it installed there -- a separate decision.

## Environments

| purpose | path |
|---------|------|
| cueq kernels (isolated) | `/storage/data/jerry528/cueq_test/bin/python` |
| training env (untouched)| `/storage/data/jerry528/uv/environments/mace-ace-pr124/bin/python` |
| runner (sets LD_LIBRARY_PATH, pins the good GPU) | `scripts/run_cueq.sh` |

## Hardware note

**gpu01's device 0 is faulty.** `cuda:0` raises "CUDA-capable device(s) is/are busy or
unavailable" while `cuda:1` works; `nvidia-smi` reports the node idle, 0 MiB used, Default
compute mode. It killed two training submissions before I isolated it
(`scripts/cudatest.py`). All scripts here pin `CUDA_VISIBLE_DEVICES=1` on gpu01.
