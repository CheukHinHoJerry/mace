"""O(3) equivariance of the magnetic models: magnetic moments are AXIAL vectors.

Regression test for ACEsuit/mace#1647. Magnetic moments are pseudovectors: under an
orthogonal transform R the positions map as r -> R r but the moments as m -> det(R) R m,
so a physically correct energy satisfies

    E(R r, det(R) R m) == E(r, m)   for every R in O(3).

The spherical harmonics of m must therefore be labelled with parity (+1)**l
(0e + 1e + 2e + ...), not the (-1)**l of a polar vector. The numerical values of the
harmonics are the same either way, but the irrep labels drive the parity selection rules
of every downstream tensor product; labelling them 1o admits spin-space couplings that
break inversion symmetry (observed: ~45% relative energy error under inversion).

Proper rotations are included as a control: they pass under either labelling, so a failure
confined to the improper operations isolates the parity bookkeeping.
"""
import numpy as np
import pytest
import torch
from e3nn import o3
from scipy.spatial.transform import Rotation as Rot

pytest.importorskip("mace.modules.extensions")
from mace.modules import interaction_classes  # noqa: E402
from mace.modules.extensions import MagneticScaleShiftMACE  # noqa: E402


@pytest.fixture(autouse=True)
def _float64():
    """Run every test in this module in float64, and restore the previous dtype.

    These are exact-symmetry assertions at the 1e-10 level, so they need double precision.
    The dtype is global state that mace.cli.run_train mutates (to float32 by default), and
    the training tests in this directory go through that CLI -- so the dtype must be pinned
    per test rather than at import, and restored so this module cannot leak state either.
    """
    prev = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    yield
    torch.set_default_dtype(prev)


_RMAX = 4.0
_POS = np.array([[0.0, -1.6, 0.0], [1.3, 0.0, 0.0], [0.0, 1.1, 0.4]])
_MAG = np.array([[0.1, 0.3, 2.2], [0.2, -0.4, 2.1], [-0.3, 0.1, 2.0]])
_TOL = 1e-10

_PROPER = [
    ("rot(37,-19,54)", Rot.from_euler("xyz", [37, -19, 54], degrees=True).as_matrix()),
    ("rot(90,0,0)", Rot.from_euler("xyz", [90, 0, 0], degrees=True).as_matrix()),
]
_IMPROPER = [
    ("inversion", -np.eye(3)),
    ("mirror z", np.diag([1.0, 1.0, -1.0])),
    ("mirror x", np.diag([-1.0, 1.0, 1.0])),
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


def _build(hidden, seed=1):
    torch.manual_seed(seed)
    blk = interaction_classes[
        "MagneticRealAgnosticSpinOrbitCoupledDensityInteractionBlock"
    ]
    return MagneticScaleShiftMACE(
        r_max=_RMAX,
        num_bessel=4,
        num_polynomial_cutoff=4,
        max_ell=2,
        m_max=[3.0],
        num_mag_radial_basis=4,
        max_m_ell=1,
        interaction_cls=blk,
        interaction_cls_first=blk,
        num_interactions=2,
        num_elements=1,
        hidden_irreps=o3.Irreps(hidden),
        MLP_irreps=o3.Irreps("4x0e"),
        atomic_energies=np.zeros(1),
        avg_num_neighbors=2.0,
        atomic_numbers=[26],
        correlation=2,
        gate=torch.nn.functional.silu,
        atomic_inter_shift=0.0,
        atomic_inter_scale=1.0,
        use_magmom_one_body=False,
    )


def _energy(model, pos, mag):
    return float(
        model(
            _batch(pos, mag),
            training=False,
            compute_force=False,
            compute_stress=False,
        )["energy"]
    )


@pytest.mark.parametrize("hidden", ["8x0e", "8x0e+8x1o"])
@pytest.mark.parametrize("name,R", _PROPER + _IMPROPER)
def test_energy_is_o3_invariant_with_axial_moments(hidden, name, R):
    """E(Rr, det(R) Rm) == E(r, m) for proper AND improper R."""
    model = _build(hidden)
    base = _energy(model, _POS, _MAG)
    det = float(np.linalg.det(R))
    moved = _energy(model, _POS @ R.T, det * (_MAG @ R.T))
    assert abs(moved - base) < _TOL, (
        f"{name} (det={det:+.0f}) changed the energy by {abs(moved - base):.3e} "
        f"with hidden_irreps={hidden}; magmom harmonics must carry axial parity"
    )


def test_magmom_harmonics_are_labelled_even_parity():
    """The magmom attribute irreps must be 0e+1e+..., never 0e+1o+..."""
    model = _build("8x0e+8x1o")
    seen = 0
    for mod in model.modules():
        irreps = getattr(mod, "magmom_node_attrs_irreps", None)
        if irreps is None:
            continue
        seen += 1
        for mul_ir in o3.Irreps(irreps):
            assert mul_ir.ir.p == 1, (
                f"magmom irrep {mul_ir} has odd parity; magnetic moments are axial "
                f"vectors and their harmonics must all be even (ACEsuit/mace#1647)"
            )
    assert seen > 0, "no module exposed magmom_node_attrs_irreps"


def test_moments_still_couple_after_the_parity_fix():
    """Tightening the selection rules must not leave the model blind to magnetism."""
    model = _build("8x0e+8x1o")
    base = _energy(model, _POS, _MAG)
    assert abs(_energy(model, _POS, _MAG * 0.5) - base) > _TOL, "no |m| response"
    assert abs(_energy(model, _POS, np.zeros_like(_MAG)) - base) > _TOL, "no on/off response"
    flipped = _MAG.copy()
    flipped[0] = -flipped[0]
    assert abs(_energy(model, _POS, flipped) - base) > _TOL, "no relative-orientation response"
    # spin-orbit character: rotating the spins alone (lattice fixed) must change E
    spun = _MAG @ Rot.from_euler("y", 90, degrees=True).as_matrix().T
    assert abs(_energy(model, _POS, spun) - base) > _TOL, "lost magnetocrystalline anisotropy"
