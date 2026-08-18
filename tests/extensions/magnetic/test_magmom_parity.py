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


def _build(hidden, seed=1, enforce_time_reversal=False):
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
        enforce_time_reversal=enforce_time_reversal,
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


# --------------------------------------------------------------------------------------
# Capacity: the axial relabelling changes WHICH spin-space paths exist. These pin down the
# consequence, because it is the reason a previous attempt at this fix was abandoned.
# --------------------------------------------------------------------------------------
def _spin_space_paths(target, magmom_parity, max_m_ell=1):
    """The (spatial (x) magmom -> out) coupling paths the interaction block can build."""
    from mace.modules.blocks import tp_out_irreps_with_instructions

    tgt = o3.Irreps(target)
    mag = o3.Irreps.spherical_harmonics(max_m_ell, p=magmom_parity)
    mid, ins = tp_out_irreps_with_instructions(tgt, mag, tgt)
    return {(str(tgt[i[0]].ir), str(mag[i[1]].ir), str(mid[i[2]].ir)) for i in ins}


def test_axial_moments_need_an_axial_channel_to_stay_expressive():
    """The moment direction must have SOME route into the equivariant channels.

    With polar-labelled moments a scalar feature could be promoted to a vector by the
    moment (0e x m -> 1o) and read back out (1o x m -> 0e). Labelling the moment axial --
    which is what the physics requires -- removes both unless the hidden irreps carry an
    axial (1e) channel for those outputs to land in. This is the degree of freedom that
    makes the parity fix look lossy if 1e is omitted; it is recovered by including it.
    """
    l1 = lambda paths: {p for p in paths if p[1].startswith("1")}

    polar_only = l1(_spin_space_paths("8x0e+8x1o", -1))
    axial_only = l1(_spin_space_paths("8x0e+8x1o", +1))
    axial_plus = l1(_spin_space_paths("8x0e+8x1o+8x1e", +1))

    # without an axial channel the moment direction is squeezed to a single route
    assert len(axial_only) < len(polar_only), (
        "expected the axial relabelling to prune spin-space paths when hidden_irreps "
        "carries no 1e channel"
    )
    # adding one restores at least as many routes as the (incorrect) polar labelling had
    assert len(axial_plus) >= len(polar_only), (
        f"adding a 1e channel should restore the lost capacity: "
        f"polar={sorted(polar_only)} vs axial+1e={sorted(axial_plus)}"
    )
    # and specifically restores both directions of flow the polar labelling had
    assert ("0e", "1e", "1e") in axial_plus, "moment cannot seed an equivariant channel"
    assert ("1e", "1e", "0e") in axial_plus, "moment cannot be read back into a scalar"


@pytest.mark.parametrize("hidden", ["8x0e+8x1o", "8x0e+8x1o+8x1e"])
def test_o3_invariance_holds_with_an_axial_channel(hidden):
    """Adding the 1e channel must not reintroduce the inversion-symmetry breaking."""
    model = _build(hidden)
    base = _energy(model, _POS, _MAG)
    for name, R in _IMPROPER:
        det = float(np.linalg.det(R))
        moved = _energy(model, _POS @ R.T, det * (_MAG @ R.T))
        assert abs(moved - base) < _TOL, f"{name} broke O(3) with hidden_irreps={hidden}"


@pytest.mark.parametrize("hidden", ["8x0e+8x1o", "8x0e+8x1o+8x1e"])
def test_chirality_and_anisotropy_survive(hidden):
    """Two responses a magnetic model must keep: coupling of the spins to the lattice,
    and sensitivity to the handedness of a non-coplanar spin texture.

    The umbrella states below share every pairwise dot product and the same cone angle to
    z, differing only in the sign of the scalar spin chirality S1.(S2 x S3) -- which is a
    true scalar for axial moments, so it is allowed to enter the energy and must not be
    silently forbidden by the parity change.
    """
    model = _build(hidden)

    def umbrella(sign, theta=50.0):
        th = np.radians(theta)
        phis = [0, 120, 240][::sign]
        v = [
            [np.sin(th) * np.cos(np.radians(p)), np.sin(th) * np.sin(np.radians(p)), np.cos(th)]
            for p in phis
        ]
        return 2.2 * np.array(v)

    assert abs(_energy(model, _POS, umbrella(1)) - _energy(model, _POS, umbrella(-1))) > _TOL, (
        "model is blind to the handedness of a non-coplanar spin texture"
    )
    spun = _MAG @ Rot.from_euler("y", 90, degrees=True).as_matrix().T
    assert abs(_energy(model, _POS, spun) - _energy(model, _POS, _MAG)) > _TOL, (
        "model lost magnetocrystalline anisotropy"
    )


# --------------------------------------------------------------------------------------
# Gradient covariance. Energy invariance alone does not exercise the gradient path, and
# the two gradients transform DIFFERENTLY: forces are polar, magnetic forces are axial.
# --------------------------------------------------------------------------------------
def _energy_and_grads(model, pos, mag):
    batch = _batch(pos, mag)
    batch["magmom"] = batch["magmom"].clone().requires_grad_(True)
    out = model(batch, training=True, compute_force=True, compute_stress=False)
    (dedm,) = torch.autograd.grad(out["energy"].sum(), batch["magmom"], retain_graph=True)
    return out["forces"].detach().numpy(), dedm.detach().numpy()


@pytest.mark.parametrize("name,R", _PROPER + _IMPROPER)
def test_forces_are_polar_and_magforces_are_axial(name, R):
    """Under r -> Rr, m -> det(R) Rm:  F -> R F  but  dE/dm -> det(R) R dE/dm.

    Differentiating the invariance E(Rr, det(R)Rm) = E(r, m) gives exactly these two laws,
    so they are implied by the energy test -- but only if the gradient path is wired the
    same way as the forward path, which is what this checks.
    """
    model = _build("8x0e+8x1o")
    det = float(np.linalg.det(R))
    f0, g0 = _energy_and_grads(model, _POS, _MAG)
    f1, g1 = _energy_and_grads(model, _POS @ R.T, det * (_MAG @ R.T))
    assert np.abs(f1 - f0 @ R.T).max() < 1e-9, f"forces not polar under {name}"
    assert np.abs(g1 - det * (g0 @ R.T)).max() < 1e-9, f"magforces not axial under {name}"


# --------------------------------------------------------------------------------------
# Physically meaningful magnetic states, rather than random moments.
# --------------------------------------------------------------------------------------
def _states():
    n = len(_POS)
    z = np.tile([0.0, 0.0, 2.2], (n, 1))
    fm = z.copy()
    afm = z.copy()
    afm[1::2] *= -1.0
    # 120-degree coplanar order on the first three sites (the Mn3X D-phase motif)
    tri = np.array(
        [[np.cos(np.radians(a)), np.sin(np.radians(a)), 0.0] for a in (0, 120, 240)]
    )
    tri = 2.2 * np.vstack([tri, [[0.0, 0.0, 1.0]]])[:n]
    # flat spin spiral along x
    q = 1.3
    spiral = 2.2 * np.array(
        [[np.cos(q * p[0]), np.sin(q * p[0]), 0.0] for p in _POS]
    )
    return {"ferromagnetic": fm, "antiferromagnetic": afm,
            "120-degree": tri, "spin spiral": spiral}


@pytest.mark.parametrize("state", sorted(_states()))
@pytest.mark.parametrize("name,R", _IMPROPER)
def test_o3_invariance_on_physical_magnetic_states(state, name, R):
    """The improper-operation invariance must hold for real magnetic orders, not just
    random moments -- collinear FM/AFM, 120-degree noncollinear, and a spin spiral."""
    model = _build("8x0e+8x1o")
    mag = _states()[state]
    base = _energy(model, _POS, mag)
    det = float(np.linalg.det(R))
    moved = _energy(model, _POS @ R.T, det * (mag @ R.T))
    assert abs(moved - base) < _TOL, f"{name} broke O(3) for the {state} state"


# --------------------------------------------------------------------------------------
# Time reversal. A SECOND symmetry, independent of O(3): e3nn tracks rotations and
# inversion, not m -> -m, so terms like (r_i x r_j).m_k are legitimate O(3) scalars that
# parity bookkeeping admits even though they are time-reversal odd. With no external
# field the magnetic energy must be even in the moments, so those terms are spurious.
# --------------------------------------------------------------------------------------
@pytest.mark.parametrize("hidden", ["8x0e+8x1o", "8x0e+8x1o+8x1e"])
def test_time_reversal_is_exact_when_enforced(hidden):
    """E(r, -m) == E(r, m) to machine precision with enforce_time_reversal=True."""
    model = _build(hidden, enforce_time_reversal=True)
    assert abs(_energy(model, _POS, -_MAG) - _energy(model, _POS, _MAG)) < _TOL


@pytest.mark.parametrize("hidden", ["8x0e+8x1o", "8x0e+8x1o+8x1e"])
def test_time_reversal_is_violated_by_default(hidden):
    """Without the flag the odd sector is present -- the flag is doing real work.

    Guards against the symmetrisation silently becoming a no-op (e.g. if the sign buffer
    were all ones), which would make the test above pass for the wrong reason.
    """
    model = _build(hidden, enforce_time_reversal=False)
    assert abs(_energy(model, _POS, -_MAG) - _energy(model, _POS, _MAG)) > _TOL


def test_time_reversal_signs_follow_minus_one_to_the_l():
    """Y_l(-m) = (-1)**l Y_l(m): even-l blocks keep sign, odd-l blocks flip."""
    model = _build("8x0e+8x1o", enforce_time_reversal=True)
    lmax = model.mag_solid_harmoics.SH.l_max()
    expected = np.concatenate(
        [np.full(2 * l + 1, (-1.0) ** l) for l in range(lmax + 1)]
    )
    assert np.allclose(model.magmom_tr_signs.numpy(), expected)


@pytest.mark.parametrize("name,R", _IMPROPER)
def test_time_reversal_and_o3_hold_together(name, R):
    """Enforcing one symmetry must not break the other."""
    model = _build("8x0e+8x1o", enforce_time_reversal=True)
    base = _energy(model, _POS, _MAG)
    det = float(np.linalg.det(R))
    assert abs(_energy(model, _POS @ R.T, det * (_MAG @ R.T)) - base) < _TOL, name
    assert abs(_energy(model, _POS, -_MAG) - base) < _TOL


def test_time_reversal_keeps_the_even_physics():
    """The projection removes only the odd sector.

    Exchange (m_i.m_j), anisotropy ((m.n)^2), Dzyaloshinskii-Moriya (D.(m_i x m_j)) and
    the scalar spin chirality are all EVEN in total degree and must survive. This is why
    the fix cannot be "drop the odd-l harmonics": m_i.m_j needs l=1 on both sites.
    """
    model = _build("8x0e+8x1o", enforce_time_reversal=True)
    base = _energy(model, _POS, _MAG)

    def umbrella(sign, theta=50.0):
        th = np.radians(theta)
        v = [
            [np.sin(th) * np.cos(np.radians(p)), np.sin(th) * np.sin(np.radians(p)), np.cos(th)]
            for p in [0, 120, 240][::sign]
        ]
        return 2.2 * np.array(v)

    rot = _MAG.copy()
    rot[0] = Rot.from_euler("z", 90, degrees=True).as_matrix() @ rot[0]
    checks = {
        "exchange": abs(_energy(model, _POS, rot) - base),
        "anisotropy": abs(
            _energy(model, _POS, _MAG @ Rot.from_euler("y", 90, degrees=True).as_matrix().T) - base
        ),
        "chirality": abs(_energy(model, _POS, umbrella(1)) - _energy(model, _POS, umbrella(-1))),
        "|m| response": abs(_energy(model, _POS, _MAG * 0.5) - base),
    }
    dead = [k for k, v in checks.items() if v <= _TOL]
    assert not dead, f"time-reversal projection also removed: {dead} ({checks})"


@pytest.mark.parametrize("name,R", _PROPER + _IMPROPER)
def test_gradients_stay_covariant_with_time_reversal_on(name, R):
    """Symmetrising before get_outputs must leave the gradient laws intact."""
    model = _build("8x0e+8x1o", enforce_time_reversal=True)
    det = float(np.linalg.det(R))
    f0, g0 = _energy_and_grads(model, _POS, _MAG)
    f1, g1 = _energy_and_grads(model, _POS @ R.T, det * (_MAG @ R.T))
    assert np.abs(f1 - f0 @ R.T).max() < 1e-9, f"forces not polar under {name}"
    assert np.abs(g1 - det * (g0 @ R.T)).max() < 1e-9, f"magforces not axial under {name}"
