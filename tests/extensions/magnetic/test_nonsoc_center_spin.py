import numpy as np
import pytest
import torch
from e3nn import o3
from scipy.spatial.transform import Rotation as Rot

from mace.modules import interaction_classes
from mace.modules.extensions import MagneticNonSOCScaleShiftMACE
from mace.modules.symmetric_contraction_nonsoc import NonSOCContraction
from mace.tools.torch_tools import default_dtype

# ----------------------------------------------------------
# Shared geometry
# ----------------------------------------------------------
_RMAX = 4.0
_TOL = 1e-12
_S = Rot.from_euler("xyz", [23, -51, 17], degrees=True).as_matrix()
_R = Rot.from_euler("xyz", [11, 73, -40], degrees=True).as_matrix()
_DIMER = np.array([[0.0, 0.0, 0.0], [2.1, 0.0, 0.0]])
_MAG2 = np.array([[0.3, 1.4, 0.7], [1.1, -0.5, 0.9]])


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


def _build(correlation=2, max_m_ell=2, hidden="8x0e"):
    torch.manual_seed(1)
    return MagneticNonSOCScaleShiftMACE(
        r_max=_RMAX,
        num_bessel=4,
        num_polynomial_cutoff=4,
        max_ell=2,
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
        max_m_ell=max_m_ell,
        use_magmom_one_body=False,
    )


def _disable_center_spin(model):
    for mod in model.modules():
        if isinstance(mod, NonSOCContraction):
            mod.center_spin_coupling = False
    return model


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
# The centre moment must couple to its neighbours
# ----------------------------------------------------------
@pytest.mark.parametrize("correlation", [1, 2, 3])
@pytest.mark.parametrize("max_m_ell", [1, 2])
def test_dimer_energy_depends_on_the_relative_spin_angle(correlation, max_m_ell):
    """In a dimer the only spin invariant besides the norms is m_0 . m_1.

    So an energy that does not move when ONE moment is rotated cannot represent
    J_ij(R) m_i . m_j -- the ordinary exchange term. Without the centre-spin channel
    this difference is not merely small, it is exactly zero: the neighbour magmom slots
    are contracted to a spin scalar and m_i re-enters only through invariants of |m_i|.
    """
    with default_dtype(torch.float64):
        model = _build(correlation=correlation, max_m_ell=max_m_ell)
        base = _energy(model, _DIMER, _MAG2)
        turned = _MAG2.copy()
        turned[0] = _MAG2[0] @ _S.T
        assert abs(_energy(model, _DIMER, turned) - base) > 1e-9, (
            f"rotating m_0 alone left the dimer energy unchanged at correlation="
            f"{correlation}, max_m_ell={max_m_ell}; the centre moment is not coupling "
            f"to its neighbours"
        )


def test_without_the_channel_the_dimer_is_exactly_blind():
    """Guards the premise of the test above: the gap it detects is exact, not numerical."""
    with default_dtype(torch.float64):
        model = _disable_center_spin(_build())
        base = _energy(model, _DIMER, _MAG2)
        turned = _MAG2.copy()
        turned[0] = _MAG2[0] @ _S.T
        assert _energy(model, _DIMER, turned) == base


# ----------------------------------------------------------
# The channel must not buy expressivity by breaking the group
# ----------------------------------------------------------
@pytest.mark.parametrize("correlation", [1, 2, 3])
@pytest.mark.parametrize(
    "name,rot_pos,rot_mag",
    [
        ("rotate space alone", _R, np.eye(3)),
        ("rotate spins alone", np.eye(3), _S),
        ("rotate both, differently", _R, _S),
        ("invert space alone", -np.eye(3), np.eye(3)),
        ("reverse moments alone", np.eye(3), -np.eye(3)),
    ],
)
def test_center_spin_channel_preserves_the_group(correlation, name, rot_pos, rot_mag):
    """O(3)_space x SO(3)_spin x Z2^T must survive the new coupling.

    Time reversal is the one worth stating: the coupling is [Gamma_i (x) M_i]_0 with
    Gamma_i built from the neighbour moments. Because the e3nn parity slot tracks T in
    this model, requesting a "1o" output admits only T-odd neighbour combinations, and
    M_i is itself T-odd, so the product is T-even.
    """
    with default_dtype(torch.float64):
        model = _build(correlation=correlation)
        base = _energy(model, _DIMER, _MAG2)
        moved = _energy(model, _DIMER @ rot_pos.T, _MAG2 @ rot_mag.T)
        assert (
            abs(moved - base) < _TOL
        ), f"{name} changed the energy by {abs(moved-base):.3e}"


# ----------------------------------------------------------
# It must actually span the Heisenberg term, not merely react to the angle
# ----------------------------------------------------------
def test_heisenberg_dimer_is_representable():
    """Fit E = J(r) m_0 . m_1 on dimers with unit moments.

    Unit moments make the |m| channels carry no information, so the target is pure
    exchange. Without the centre-spin channel the best achievable fit is the mean over
    relative angle, i.e. the full target variance.
    """
    with default_dtype(torch.float64):
        rng = np.random.RandomState(0)
        cfgs, tgt = [], []
        for _ in range(60):
            r = rng.uniform(1.6, 3.4)
            u = rng.randn(3)
            u /= np.linalg.norm(u)
            v = rng.randn(3)
            v /= np.linalg.norm(v)
            cfgs.append((np.array([[0.0, 0.0, 0.0], [r, 0.0, 0.0]]), np.stack([u, v])))
            tgt.append(np.exp(-1.2 * r) * float(u @ v))
        target = torch.tensor(np.array(tgt))
        variance = float(np.var(tgt))

        model = _build(correlation=2, hidden="16x0e")
        opt = torch.optim.Adam(model.parameters(), lr=1e-2)
        for _ in range(150):
            opt.zero_grad()
            pred = torch.stack(
                [
                    model(
                        _batch(p, m),
                        training=True,
                        compute_force=False,
                        compute_stress=False,
                    )["energy"].squeeze()
                    for p, m in cfgs
                ]
            )
            loss = ((pred - target) ** 2).mean()
            loss.backward()
            opt.step()
        assert float(loss) < 0.25 * variance, (
            f"final MSE {float(loss):.3e} is not well below the target variance "
            f"{variance:.3e}; the exchange term is not being represented"
        )


# ----------------------------------------------------------
# Which centre-spin orders exist
# ----------------------------------------------------------
@pytest.mark.parametrize("max_m_ell", [1, 2, 3])
def test_centre_spin_orders_follow_max_m_ell(max_m_ell):
    """s runs 1..max_m_ell: the centre attributes stop there, so higher s has nothing to
    close against. s=1 carries Heisenberg, s=2 the biquadratic order, and so on."""
    with default_dtype(torch.float64):
        model = _build(correlation=2, max_m_ell=max_m_ell)
        contraction = [m for m in model.modules() if isinstance(m, NonSOCContraction)][
            0
        ]
        assert contraction.center_spin_orders == list(range(1, max_m_ell + 1))
        for s_order in contraction.center_spin_orders:
            for nu in (1, 2):
                buffers = dict(contraction.named_buffers())
                assert f"U_matrix_magmom_s{s_order}_{nu}" in buffers


def test_correlation_above_three_is_rejected():
    """The contraction equations and the completeness verification both stop at nu = 3."""
    with default_dtype(torch.float64):
        with pytest.raises(ValueError, match="correlation <= 3"):
            _build(correlation=4)
