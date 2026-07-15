"""Regression tests for the non-SOC magnetic MACE L=1 path and the block-weight fix.

Covers behaviour introduced in the sandbox work (see SANDBOX_CHANGES.md):
  1. the scalar (L=0) non-SOC magnetic model is rotation- and spin-invariant;
  2. a per-layer L=1 model ("8x0e+8x1o | 8x0e") builds and stays invariant at trained scale;
  3. the block-weight fix (correct cumsum(ir.dim) offsets) leaves NO dead W_block params;
  4. the fix auto-resolves from the layer: OFF on the scalar path (byte-identical), ON on
     the L>0 path -- inferred from node_feats.lmax, no flag to set.

These construct the model directly (no data pipeline, no external mace tree) so they are
self-contained and fast. They skip if the magnetic non-SOC extension is unavailable.
"""
import numpy as np
import pytest
import torch
from e3nn import o3
from scipy.spatial.transform import Rotation as Rot

pytest.importorskip("mace.modules.extensions")
from mace.modules import interaction_classes  # noqa: E402

try:
    from mace.modules.extensions import MagneticNonSOCScaleShiftMACE
except ImportError:  # pragma: no cover - feature not present on this branch
    pytest.skip("MagneticNonSOCScaleShiftMACE unavailable", allow_module_level=True)

torch.set_default_dtype(torch.float64)

_RMAX = 4.0
_POS = np.array([[0.0, -1.6, 0.0], [1.3, 0.0, 0.0], [0.0, 1.1, 0.4]])
_MAG = np.array([[0.0, 0.0, 2.2], [0.2, 0.0, 2.1], [0.0, 0.1, 2.0]])
_ROTS = [
    Rot.from_euler("xyz", a, degrees=True).as_matrix()
    for a in ([37, -19, 54], [90, 0, 0], [10, 20, 30], [180, 45, 0])
]


def _batch(pos, mag):
    n = len(pos)
    s, d = [], []
    for i in range(n):
        for j in range(n):
            if i != j and np.linalg.norm(pos[i] - pos[j]) < _RMAX:
                s.append(j)
                d.append(i)
    ei = torch.tensor([s, d])
    return {
        "positions": torch.tensor(pos, requires_grad=True),
        "cell": torch.zeros(1, 3, 3),
        "batch": torch.zeros(n, dtype=torch.long),
        "ptr": torch.tensor([0, n]),
        "node_attrs": torch.ones(n, 1),
        "edge_index": ei,
        "magmom": torch.tensor(mag),
        "unit_shifts": torch.zeros(ei.shape[1], 3),
        "shifts": torch.zeros(ei.shape[1], 3),
    }


def _build(hidden, scale=1.0, seed=1):
    torch.manual_seed(seed)
    model = MagneticNonSOCScaleShiftMACE(
        r_max=_RMAX,
        num_bessel=4,
        num_polynomial_cutoff=4,
        max_ell=2,
        interaction_cls=interaction_classes[
            "MagneticRealAgnosticNonSpinOrbitCoupledDensityInteractionBlock"
        ],
        interaction_cls_first=interaction_classes["RealAgnosticDensityInteractionBlock"],
        contraction_cls_first="SymmetricContraction",
        contraction_cls="NonSOCSymmetricContraction",
        num_interactions=2,
        num_elements=1,
        hidden_irreps=hidden,
        MLP_irreps=o3.Irreps("4x0e"),
        atomic_energies=np.zeros(1),
        avg_num_neighbors=2.0,
        atomic_numbers=[26],
        correlation=2,
        gate=torch.nn.functional.silu,
        atomic_inter_shift=0.0,
        atomic_inter_scale=1.0,
        m_max=[3.0],
        num_mag_radial_basis=4,
        num_mag_radial_basis_one_body=4,
        max_m_ell=1,
        use_magmom_one_body=False,
    )
    if scale != 1.0:
        with torch.no_grad():
            for p in model.parameters():
                p.mul_(scale)
    return model


def _energy(model, pos, mag):
    out = model(
        _batch(pos, mag),
        training=False,
        compute_force=False,
        compute_stress=False,
        compute_magforces=False,
    )
    return float(out["energy"])


def _nonsoc_block(model):
    for mod in model.modules():
        if type(mod).__name__.endswith("NonSpinOrbitCoupledDensityInteractionBlock"):
            return mod
    raise AssertionError("non-SOC interaction block not found")


# --------------------------------------------------------------------------------------
# 1. scalar (L=0) invariance — the production path must keep working
# --------------------------------------------------------------------------------------
def test_scalar_energy_is_spatial_and_spin_invariant():
    model = _build(["8x0e", "8x0e"], scale=0.1)
    base = _energy(model, _POS, _MAG)
    assert np.isfinite(base)
    spatial = max(abs(_energy(model, (R @ _POS.T).T, _MAG) - base) for R in _ROTS)
    spin = max(abs(_energy(model, _POS, (R @ _MAG.T).T) - base) for R in _ROTS)
    denom = abs(base) + 1e-30
    assert spatial / denom < 1e-9, f"scalar spatial non-invariance {spatial/denom:.2e}"
    assert spin / denom < 1e-9, f"scalar spin non-invariance {spin/denom:.2e}"


# --------------------------------------------------------------------------------------
# 2. per-layer L=1 model builds and is invariant at trained scale
# --------------------------------------------------------------------------------------
def test_l1_model_builds_and_is_invariant():
    model = _build(["8x0e+8x1o", "8x0e"], scale=0.1)  # first layer carries the vector
    base = _energy(model, _POS, _MAG)
    assert np.isfinite(base)
    spatial = max(abs(_energy(model, (R @ _POS.T).T, _MAG) - base) for R in _ROTS)
    spin = max(abs(_energy(model, _POS, (R @ _MAG.T).T) - base) for R in _ROTS)
    denom = abs(base) + 1e-30
    assert spatial / denom < 1e-9, f"L=1 spatial non-invariance {spatial/denom:.2e}"
    assert spin / denom < 1e-9, f"L=1 spin non-invariance {spin/denom:.2e}"


# --------------------------------------------------------------------------------------
# 3. block-weight fix leaves no dead params (revives the DDP find_unused hazard)
# --------------------------------------------------------------------------------------
def test_block_weight_fix_has_no_dead_params():
    model = _build(["8x0e+8x1o", "8x0e"], scale=1.0)
    blk = _nonsoc_block(model)
    assert blk._fix_block_offsets is True  # auto-on for the L>0 path

    model(
        _batch(_POS, _MAG),
        training=True,
        compute_force=False,
        compute_stress=False,
        compute_magforces=False,
    )["energy"].sum().backward()

    dead = []
    total = 0
    for i, row in enumerate(blk.linear_block_weight_list):
        for j, W in enumerate(row):
            total += 1
            if W.grad is None or float(W.grad.norm()) == 0.0:
                dead.append((i, j))
    assert total > 1, "expected multiple (l,l') blocks for the L=1 message"
    assert not dead, f"fixed offsets should leave NO dead W_block params; dead={dead}"


def test_legacy_offsets_have_dead_params():
    """Documents the pre-existing bug: legacy offsets strand most block weights.

    The fix auto-enables on the L>0 path, so we override the inferred attribute to force
    the pre-fix (legacy) offsets and pin the dead-parameter behaviour it produced.
    """
    model = _build(["8x0e+8x1o", "8x0e"], scale=1.0)
    blk = _nonsoc_block(model)
    assert blk._fix_block_offsets is True  # inferred ON for L>0
    blk._fix_block_offsets = False  # force pre-fix legacy offsets

    model(
        _batch(_POS, _MAG),
        training=True,
        compute_force=False,
        compute_stress=False,
        compute_magforces=False,
    )["energy"].sum().backward()

    dead = sum(
        1
        for row in blk.linear_block_weight_list
        for W in row
        if W.grad is None or float(W.grad.norm()) == 0.0
    )
    assert dead > 0, "legacy offsets are expected to strand some W_block params"


# --------------------------------------------------------------------------------------
# 4. fix auto-resolves from the layer: OFF on scalar path, ON on the L>0 path
# --------------------------------------------------------------------------------------
def test_fix_auto_resolves_by_layer_type():
    scalar_blk = _nonsoc_block(_build(["8x0e", "8x0e"]))
    l1_blk = _nonsoc_block(_build(["8x0e+8x1o", "8x0e"]))
    assert scalar_blk._fix_block_offsets is False, "scalar path must stay on legacy (byte-identical)"
    assert l1_blk._fix_block_offsets is True, "L>0 path must auto-enable the fix"
