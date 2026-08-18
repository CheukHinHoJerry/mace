import numpy as np
import pytest
import torch
from e3nn import o3
from scipy.spatial.transform import Rotation as Rot

from mace.modules import interaction_classes
from mace.modules.extensions import MagneticNonSOCScaleShiftMACE
from mace.tools.torch_tools import default_dtype

# ----------------------------------------------------------
# Shared geometry and rotations
# ----------------------------------------------------------
_RMAX = 4.0
_TOL = 1e-12
_POS = np.array([[0.0, -1.6, 0.0], [1.3, 0.0, 0.0], [0.0, 1.1, 0.4]])
_MAG = np.array([[0.1, 0.3, 2.2], [0.2, -0.4, 2.1], [-0.3, 0.1, 2.0]])
_R = Rot.from_euler("xyz", [11, 73, -40], degrees=True).as_matrix()
_S = Rot.from_euler("xyz", [37, -19, 54], degrees=True).as_matrix()


# ----------------------------------------------------------
# Helpers
# ----------------------------------------------------------
def _batch(pos, mag):
    """Single-structure batch with a hand-built neighbor list (no PBC edges)."""
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


def _build(max_ell=2, hidden="8x0e", correlation=2):
    """Small non-SOC model: plain first block, magnetic non-SOC second block."""
    torch.manual_seed(1)
    return MagneticNonSOCScaleShiftMACE(
        r_max=_RMAX,
        num_bessel=4,
        num_polynomial_cutoff=4,
        max_ell=max_ell,
        interaction_cls=interaction_classes[
            "MagneticRealAgnosticNonSpinOrbitCoupledDensityInteractionBlock"
        ],
        interaction_cls_first=interaction_classes[
            "RealAgnosticDensityInteractionBlock"
        ],
        contraction_cls_first="SymmetricContraction",
        contraction_cls="NonSOCSymmetricContraction",
        num_interactions=2,
        num_elements=1,
        hidden_irreps=o3.Irreps(hidden),
        MLP_irreps=o3.Irreps("4x0e"),
        atomic_energies=np.zeros(1),
        avg_num_neighbors=2.0,
        atomic_numbers=[26],
        correlation=correlation,
        gate=torch.nn.functional.silu,
        atomic_inter_shift=0.0,
        atomic_inter_scale=1.0,
        m_max=[3.0],
        num_mag_radial_basis=4,
        num_mag_radial_basis_one_body=4,
        max_m_ell=1,
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


# ----------------------------------------------------------
# O(3)_space x O(3)_spin equivariance tests
# ----------------------------------------------------------
@pytest.mark.parametrize("max_ell", [1, 2, 3])
@pytest.mark.parametrize(
    "name,rot_pos,rot_mag",
    [
        ("rotate space alone", _R, np.eye(3)),
        ("rotate spins alone", np.eye(3), _S),
        ("rotate both, differently", _R, _S),
        ("invert space alone", -np.eye(3), np.eye(3)),
        ("reverse moments alone", np.eye(3), -np.eye(3)),
        ("invert space and reverse moments", -np.eye(3), -np.eye(3)),
        ("rotate both and reverse moments", _R, -_S),
    ],
)
def test_spin_and_space_rotate_independently(max_ell, name, rot_pos, rot_mag):
    """Space and spin rotate INDEPENDENTLY, so the enforced group is O(3) x O(3).

    Independently is the operative word. The magmom angular parts contract among
    themselves to a spin scalar before meeting the spatial part, so the energy
    sees the moments only through spin-rotation invariants and every one of

        E(Rr, m)  E(r, Sm)  E(Rr, Sm)  E(-r, m)  E(r, -m)

    equals E(r, m) for arbitrary independent R and S. Contrast the spin-orbit
    coupled MagneticMACE, where only the JOINT rotation is a symmetry.

    Inverting space leaves the moments alone because they are axial: under an
    orthogonal transform they map m -> det(R) R m, so P: (r, m) -> (-r, m).
    Reversing the moments alone is time reversal, under which the energy is even
    at zero field.
    """
    with default_dtype(torch.float64):
        model = _build(max_ell=max_ell)
        base = _energy(model, _POS, _MAG)
        moved = _energy(model, _POS @ rot_pos.T, _MAG @ rot_mag.T)
        assert (
            abs(moved - base) < _TOL
        ), f"{name} changed the energy by {abs(moved - base):.3e} at max_ell={max_ell}"


# ----------------------------------------------------------
# Block-weight offsets
# ----------------------------------------------------------
@pytest.mark.parametrize("max_ell", [2, 3])
def test_block_offsets_index_the_angular_axis(max_ell):
    """Block offsets must be cumsum(ir.dim); cumsum(mul * ir.dim) overruns the axis."""
    with default_dtype(torch.float64):
        model = _build(max_ell=max_ell)
        block = model.interactions[1]
        for attr in ("_r_msg_irreps", "_m_msg_irreps"):
            irreps = o3.Irreps(getattr(block, attr))
            angular = sum(ir.dim for _, ir in irreps)
            good = np.cumsum([0] + [ir.dim for _, ir in irreps])
            assert (
                good[-1] == angular
            ), f"{attr}: correct offsets must span the angular axis"
            bad = np.cumsum([0] + [mul * ir.dim for mul, ir in irreps])
            assert bad[-1] > angular, (
                f"{attr}: the legacy offsets should overrun the axis -- if they no "
                f"longer do, this test has stopped guarding anything"
            )
        assert not hasattr(block, "_fix_block_offsets"), (
            "correct block offsets are unconditional; there should be no flag selecting "
            "the legacy cumsum(mul * ir.dim) behaviour"
        )


def test_no_dead_block_weights():
    """Every per-block weight must actually receive a gradient.

    Checked through a real backward pass rather than by recomputing the offsets,
    so it fails if the forward stops using them. Under the legacy stride the
    first block swallows the whole angular axis and the rest slice empty ranges,
    leaving their parameters at initialisation.
    """
    with default_dtype(torch.float64):
        model = _build(max_ell=3)
        block = model.interactions[1]
        weights = [w for row in block.linear_block_weight_list for w in row]
        # pylint: disable=protected-access
        n_r = len(o3.Irreps(block._r_msg_irreps))
        n_m = len(o3.Irreps(block._m_msg_irreps))
        assert (
            len(weights) == n_r * n_m
        ), f"expected {n_r * n_m} block weights, found {len(weights)}"

        model.zero_grad(set_to_none=True)
        out = model(
            _batch(_POS, _MAG), training=True, compute_force=False, compute_stress=False
        )
        out["energy"].sum().backward()

        starved = [
            i
            for i, w in enumerate(weights)
            if w.grad is None or not bool(w.grad.abs().sum() > 0)
        ]
        assert not starved, (
            f"{len(starved)} of {len(weights)} block weights received no gradient "
            f"(indices {starved}); they slice empty ranges and train as if absent"
        )
