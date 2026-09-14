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
@pytest.mark.parametrize("hidden", ["8x0e", "8x0e+8x1o", "8x0e+8x1o+8x2e"])
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
def test_spin_and_space_rotate_independently(max_ell, hidden, name, rot_pos, rot_mag):
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
        model = _build(max_ell=max_ell, hidden=hidden)
        base = _energy(model, _POS, _MAG)
        moved = _energy(model, _POS @ rot_pos.T, _MAG @ rot_mag.T)
        assert abs(moved - base) < _TOL, (
            f"{name} changed the energy by {abs(moved - base):.3e} at "
            f"max_ell={max_ell}, hidden={hidden}"
        )


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


# ----------------------------------------------------------
# Spin purity of the magnetic factor
# ----------------------------------------------------------
@pytest.mark.parametrize("hidden", ["8x0e", "8x0e+8x1o", "8x0e+8x1o+8x2e"])
def test_magnetic_message_is_spin_pure(hidden):
    """conv_tp_m's output is the SPIN axis of A_msg, so a spatial rotation must not move it.

    This is the property the energy identities rest on, tested directly rather than
    through the energy: if spatial angular content reaches the spin factor it is later
    reduced against the magmom CG basis, which admits SOC-type invariants such as
    node 1o (x) magmom 1o -> 0e, i.e. r.m. Those survive a JOINT rotation and so are
    invisible to a joint-rotation check, but they break E(Rr, m) = E(r, m).
    """
    with default_dtype(torch.float64):
        model = _build(max_ell=2, hidden=hidden)
        block = model.interactions[1]
        captured = {}

        def hook(_module, _inputs, output):
            captured.setdefault("m_msg", []).append(output.detach().clone())

        handle = block.conv_tp_m.register_forward_hook(hook)
        try:
            _energy(model, _POS, _MAG)
            _energy(model, _POS @ _R.T, _MAG)
        finally:
            handle.remove()

        base, rotated = captured["m_msg"]
        drift = float((rotated - base).abs().max())
        assert drift < _TOL, (
            f"the magnetic message moved by {drift:.3e} under a pure spatial rotation "
            f"at hidden={hidden}; the spin factor is carrying spatial content"
        )


def test_scalar_restriction_is_a_noop_for_scalar_features():
    """Restricting conv_tp_m to the l=0 features must not change a scalar-hidden model.

    Existing non-SOC checkpoints are all scalar-hidden, so this is the backward
    compatibility guarantee: for hidden="8x0e" the restriction selects everything and
    the slice is the identity.
    """
    with default_dtype(torch.float64):
        model = _build(max_ell=2, hidden="8x0e")
        block = model.interactions[1]
        assert block.n_node_feats_scalar == o3.Irreps(block.node_feats_irreps).dim, (
            "for scalar node features the restriction must select the whole tensor, "
            "otherwise previously trained models change numerically"
        )
        assert o3.Irreps(block.conv_tp_m.irreps_in1) == o3.Irreps(
            block.node_feats_irreps
        )


def test_scalars_are_listed_first():
    """forward() takes a contiguous scalar slice, which is only valid if scalars lead."""
    with default_dtype(torch.float64):
        for hidden in ("8x0e", "8x0e+8x1o", "8x0e+8x1o+8x2e"):
            block = _build(max_ell=2, hidden=hidden).interactions[1]
            irreps = o3.Irreps(block.node_feats_irreps)
            assert (
                irreps[0].ir.l == 0
            ), f"{hidden}: scalars must come first, got {irreps}"


def test_spatial_path_keeps_full_angular_resolution():
    """The restriction applies to the SPIN factor only; geometry keeps its l > 0 features."""
    with default_dtype(torch.float64):
        block = _build(max_ell=2, hidden="8x0e+8x1o").interactions[1]
        assert o3.Irreps(block.conv_tp_r.irreps_in1).lmax == 1, (
            "conv_tp_r must still receive the full node features; restricting the "
            "spatial path would cost real angular resolution"
        )
        assert o3.Irreps(block.conv_tp_m.irreps_in1).lmax == 0
        assert o3.Irreps(block.conv_tp_m.irreps_in2).lmax > 0, (
            "the magmom attributes must still enter at full max_m_ell; the spin angular "
            "structure is not what gets restricted"
        )
