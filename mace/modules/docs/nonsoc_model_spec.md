# The non-SOC magnetic MACE model: symmetry specification

This document states, precisely, which group the non-SOC magnetic model represents and
which representation each tensor slot carries. It exists because the model has a failure
mode that **improves the reported validation error while silently breaking the symmetry**,
so nothing in a training log flags it. See "Failure modes" below.

Normative statements use MUST / MUST NOT. Every claim marked (T) has a corresponding
executable test in `tests/extensions/magnetic/`.


## 1. The group

The non-SOC model represents

    G_nonSOC = O(3)_space  x  SO(3)_spin  x  Z2^T

i.e. space and spin rotate **independently**. The three generators act on a configuration
of positions `r_i` and magnetic moments `m_i` as

| generator          | action                                  |
|--------------------|-----------------------------------------|
| `R in O(3)_space`  | `r_i -> R r_i`,  `m_i -> m_i`           |
| `S in SO(3)_spin`  | `r_i -> r_i`,    `m_i -> S m_i`         |
| `T in Z2^T`        | `r_i -> r_i`,    `m_i -> -m_i`          |

The energy MUST be invariant under all three, independently:

    E(R r, m) = E(r, S m) = E(r, -m) = E(r, m)                                        (T)

These are FOUR separate identities (spatial rotation, spatial inversion, spin rotation,
time reversal). Checking a joint rotation `E(R r, R m) = E(r, m)` is NOT sufficient and
will pass on a model that is broken -- see 5.1.

Contrast with the diagonal / SOC model (`MagneticMACE`), which represents only the
**diagonal** subgroup: space and spin rotate together, `(r, m) -> (R r, det(R) R m)`.
There `r . m` is a legal invariant. Here it is not.


## 2. Parity slots: the two models mean different things by `o`

e3nn tracks exactly ONE Z2 parity label per irrep. The two models spend it differently.
This is the single most confusing point in the codebase and MUST NOT be unified.

| model                | magmom irreps                    | the `e`/`o` label denotes      |
|----------------------|----------------------------------|--------------------------------|
| non-SOC (factorised) | `0e + 1o + 2e + 3o + ...` (p=-1) | **time reversal** `T: m -> -m` |
| diagonal / SOC       | `0e + 1e + 2e + 3e + ...` (p=+1) | **spatial inversion** (axial)  |

Reasoning:

* In the non-SOC model the spin axis is a **separate representation space** from the
  spatial axis. Spatial inversion does not act on it at all (`P: (r,m) -> (-r,m)`), so the
  parity slot is free, and it is used to track time reversal. Under `T`, a rank-l spin
  tensor picks up `(-1)^l`, which is exactly `p=-1` spherical harmonics: `0e+1o+2e+3o`.
* In the diagonal/SOC model space and spin are the same rotation, and `m` is an **axial**
  vector: `m -> det(R) R m`. The parity slot must therefore carry spatial inversion, and
  axial vectors are inversion-even: `0e+1e+2e+3e` (`p=+1`).

The two constructors MUST remain separate. A single global "magnetic irreps" helper shared
by both models is a bug, not a simplification.


## 3. Slot assignment in the interaction block

`MagneticRealAgnosticNonSpinOrbitCoupledDensityInteractionBlock` builds two factors and
outer-products them:

    r_msg = conv_tp_r(node_feats[sender], edge_attrs,          w_r)   # SPATIAL factor
    m_msg = conv_tp_m(node_feats_scalar,  magmom_node_attrs,   w_m)   # SPIN factor
    A_msg = einsum("bkl,bkp->bklp", r_msg, m_msg)

Index `l` is a **spatial** irrep index; index `p` is a **spin** irrep index. The symmetric
contraction reduces `l` against the spatial CG basis and `p` against the magmom CG basis.

### 3.1 The spin-purity rule (normative)

> The spin factor `m_msg` MUST be built only from quantities that are **invariant under
> `O(3)_space`**. Concretely, `conv_tp_m`'s first input MUST be restricted to the `l = 0`
> part of `node_feats`.

Why: `conv_tp_m`'s output is consumed as the spin axis `p`. Any spatial angular content
entering it is subsequently reduced against the **magmom** CG basis -- i.e. a spatial index
is contracted as though it were a spin index. That is a category error, and it produces
exactly the SOC-type invariants the non-SOC model is defined to exclude:

    node 1o  (x)  magmom 1o  ->  0e          is   r . m          <- spin-orbit coupling

Such a term is invariant under a JOINT rotation but not under rotating space alone, so it
breaks `E(R r, m) = E(r, m)` while leaving `E(R r, R m) = E(r, m)` intact.

Implementation:

```python
self.node_feats_scalar_irreps = o3.Irreps(
    [(mul, ir) for mul, ir in self.node_feats_irreps if ir.l == 0]
)
self.n_node_feats_scalar = self.node_feats_scalar_irreps.dim
...
m_msg = self.conv_tp_m(
    node_feats[sender][:, : self.n_node_feats_scalar], magmom_node_attrs[sender], w_m
)
```

The contiguous slice `[:, :n_scalar]` is valid only because MACE lists scalars first; this
is asserted at construction, not assumed.

**Backward compatibility.** For scalar `hidden_irreps` the restriction selects the whole
tensor and the slice is the identity, so the rule is a strict no-op. Verified bit-identical
(`hidden_irreps=16x0e`, float64): both `81408` parameters, `E = 1.13880285927894542297e+00`
and the same forces to all 20 digits, with and without the restriction. Every existing
non-SOC checkpoint is scalar-hidden and is therefore unaffected (T).

### 3.2 What the rule does NOT cost

The restriction applies to the **spin factor only**. Specifically:

* `conv_tp_r` still receives the FULL `node_feats` including `l > 0`. Spatial angular
  resolution is untouched.
* `magmom_node_attrs` still enter `conv_tp_m` at full `max_m_ell`. Spin angular structure
  is untouched; scalars only channel-mix them.
* The block still emits `l > 0` node features to the next layer.

Verified numerically (probe at 16 channels, `hidden_irreps = 16x0e+16x1o`, `max_m_ell=2`):

| tensor            | irreps                          | lmax |
|-------------------|---------------------------------|------|
| `conv_tp_r` in 1  | `16x0e+16x1o`                   | 1    |
| `r_msg`           | `16x0e+16x0e+16x1o+16x1o+16x1o` | 1    |
| `conv_tp_m` in 1  | `16x0e`                         | 0    |
| `conv_tp_m` in 2  | `1x0e+1x1o+1x2e`                | 2    |
| block output      | `16x0e+16x1o`                   | 1    |

What IS removed is the set of cross terms `(node l>0) (x) (magmom l>0)`. Those are SOC
couplings and MUST NOT exist in this model. Removing them does not reduce expressiveness
within the non-SOC hypothesis class; it restores that class. Genuine coupling between
directional geometry and magnetism requires the diagonal/SOC model.

### 3.3 The check is on the INPUT, not the target

A magnetic layer's `node_feats` is the PREVIOUS layer's output. A model whose first layer
is a plain `RealAgnosticDensityInteractionBlock` emitting `128x0e+128x1o` feeds lmax=1 into
a magnetic second layer even when that layer's own `target_irreps` is scalar. Any guard or
assertion MUST inspect `inter.node_feats_irreps`, never `inter.target_irreps`.


## 4. Joint slot symmetrisation

The symmetric contraction symmetrises the paired slots jointly:

    C_bar = (1/nu!) sum_pi (P_pi^space (x) P_pi^mag) C

ONE permutation `pi` is applied to BOTH angular factors, because slot `k` of the product
carries the pair `(l_k, p_k)`; permuting the spatial factors alone is not a symmetry of the
object being contracted.

Consequences for testing:

* A **generic paired probe** `X[n,i,j]` measures the jointly symmetrised object.
* A **factorised probe** `x_i * y_j` is strictly weaker and can miss admitted paths (T).
* Slot antisymmetry does NOT imply a dead coupling: with a single channel the determinant
  identity `eps_ijf eps_lmg A_kil A_kjm A_kfg = 6 det(A_k)` is nonzero. Concluding "this
  path vanishes by antisymmetry" without a multichannel numerical check is invalid (T).


## 5. Failure modes

### 5.1 Non-equivariant L > 0 (historical: `MATPES_V2_BADER_NONSOC_L1`)

Trained 2026-07-17 on branch `magnetic-pr-noSOC` @ `2120657`, which passed the full
`node_feats` into `conv_tp_m` (no rule 3.1). Config: `interaction_first=
'RealAgnosticDensityInteractionBlock'`, `hidden_irreps='128x0e+128x1o'`, so magnetic layer
1 received lmax=1 -- precisely the 3.3 case.

Measured on that model: the magnetic message shifted by ~7.6 under a pure spatial rotation
that must leave it unchanged, and the identities of section 1 broke at 1e-3 to 1e-1.

**The reported errors got BETTER**, against an otherwise identical L=0 run on the same
train/valid files:

| metric            | L=0 `FIXED` | L=1 (broken) |
|-------------------|-------------|--------------|
| MAE_E (meV/atom)  | 28.8        | 21.36        |
| MAE_F (meV/A)     | 137.5       | 98.61        |
| MAE_stress        | 4.6         | 3.73         |

This is the reason this document exists. The illegal `r . m` paths buy fitting power by
discarding the symmetry, and a validation curve cannot distinguish that from progress.
**Validation error is not evidence of correctness for a symmetry-constrained model.**
(The two runs also differed in `batch_size`, 100 vs 32, so it was not even controlled.)

### 5.2 Confusing the two parity conventions

Applying the axial `0e+1e+2e` convention to the non-SOC model, or `0e+1o+2e` to the
diagonal model, changes which paths are admitted. Parameter counts may still match, so a
count comparison does NOT detect it. Enumerate the contraction paths before and after any
parity change and evaluate every newly admitted path on random multichannel inputs (T).

### 5.3 Conditional binding of the layer-0 interaction

`i0 = self.interactions[0]` MUST be bound unconditionally. It was previously bound inside
`if not first_is_magnetic:` while being read further down for every layer
(`radial_MLP=getattr(i0, ...)`), so ANY configuration with a magnetic first interaction
block raised `UnboundLocalError` at construction. Every run to date used a plain
`RealAgnosticDensityInteractionBlock` first, so the path was never exercised.

Lesson: a configuration that no existing run happens to use is still a supported
configuration, and the test matrix MUST cover magnetic-first as well as plain-first.

### 5.4 Block offsets under `reshape_irreps`

`reshape_irreps` factors multiplicity into the channel axis, so block offsets MUST stride
by `ir.dim`, not `mul * ir.dim`. The wrong stride stays equivariant at 128 channels but
leaves all but one block weight without gradient -- it trains, converges more slowly, and
looks merely "worse", not "broken".


## 6. Acceptance tests

A change to any magnetic path is not complete until these pass.

1. **Four identities** (section 1), each separately, to ~1e-9 in float64, for every
   supported `hidden_irreps` including `lmax > 0`. A joint-rotation test does not substitute.
2. **Spin purity**: the magnetic message `m_msg` is unchanged (bitwise, float64) under a
   pure spatial rotation of the positions.
3. **Path enumeration**: contraction paths before/after the change, with every newly
   admitted path evaluated on random multichannel inputs.
4. **Non-degenerate fixtures**: `n_edges != n_nodes != n_graphs`. A fixture with
   `n_edges == n_nodes` has previously let a shape bug pass by coincidence.

Test 1 is cheap and MUST run in CI. The L1 episode cost a full 100-epoch training run that
produced a checkpoint nobody could use; the test that would have caught it takes seconds.
