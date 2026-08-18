import itertools
import math

import numpy as np
import pytest
import torch
from e3nn import o3
from scipy.spatial.transform import Rotation as Rot

from mace.modules.symmetric_contraction_nonsoc import _joint_symmetrised_probe, _rank
from mace.tools.cg import U_matrix_real
from mace.tools.torch_tools import default_dtype

from .test_nonsoc_equivariance import _build, _energy

# ----------------------------------------------------------
# Coupling bases
# ----------------------------------------------------------
# The spatial and spin angular axes are kept separate, so the model implements
# O(3)_space x O(3)_spin and the parity slot on the SPIN axis denotes time
# reversal (m -> -m) rather than spatial inversion.
SPATIAL = "0e+1o+2e"
MAG_POLAR, MAG_AXIAL = "0e+1o", "0e+1e"


def _U(irreps, out, nu):
    """Explicit (non-reduced) CG basis.

    use_cueq_cg=False is deliberate: with cuequivariance installed the basis is returned
    already reduced to a minimal spanning set, which changes the PATH COUNTS these tests
    reason about (at nu=3 both magmom labels collapse to 2 paths). The parity and
    time-reversal structure is a property of the mathematical basis, so it is pinned here
    to stay independent of whether an accelerator happens to be installed.
    """
    return U_matrix_real(
        irreps_in=o3.Irreps(irreps),
        irreps_out=o3.Irreps(out),
        correlation=nu,
        dtype=torch.float64,
        use_cueq_cg=False,
    )[-1]


def _rand_state(seed=0, n=6):
    g = torch.Generator().manual_seed(seed)
    return (
        torch.randn(n, 3, dtype=torch.float64, generator=g) * 2.0,
        torch.randn(n, 3, dtype=torch.float64, generator=g) * 0.5,
    )


# ----------------------------------------------------------
# Joint symmetrisation of the combined basis
# ----------------------------------------------------------
@pytest.mark.parametrize("nu", [2, 3])
def test_joint_symmetrisation_matches_explicit_permutation_sum(nu):
    """Probing with nu identical paired copies equals the explicitly symmetrised basis.

    Cbar = (1/nu!) sum_pi (P_pi^r (x) P_pi^m) C, i.e. ONE permutation applied to
    both angular factors at once. Contracting against nu identical copies of a
    probe already projects onto that symmetric part, so the cheap probe and the
    dense construction must report the same rank.
    """
    with default_dtype(torch.float64):
        Ur, Um = _U(SPATIAL, "0e", nu), _U(MAG_AXIAL, "0e", nu)
        dr, dm = Ur.shape[0], Um.shape[0]

        dense = []
        for a in range(Ur.shape[-1]):
            for b in range(Um.shape[-1]):
                eq = {2: "ab,cd->acbd", 3: "abc,def->adbecf"}[nu]
                C = torch.einsum(eq, Ur[..., a], Um[..., b]).reshape(*([dr * dm] * nu))
                Cb = sum(C.permute(pi) for pi in itertools.permutations(range(nu)))
                dense.append((Cb / math.factorial(nu)).reshape(-1))
        dense_rank = _rank(torch.stack(dense).T)

        probe_rank = _rank(_joint_symmetrised_probe(Ur, Um, nu).flatten(1))
        assert probe_rank == dense_rank, (
            f"generic paired probe reports rank {probe_rank}, explicit joint "
            f"symmetrisation {dense_rank}; the probe is not measuring the jointly "
            f"symmetrised object"
        )


def test_factorised_probe_is_strictly_weaker():
    """A factorised probe x_i * y_j symmetrises the factors independently and under-reports.

    This is why coupling paths may never be pruned by examining the magmom factor
    on its own.
    """
    with default_dtype(torch.float64):
        nu = 3
        Ur, Um = _U(SPATIAL, "0e", nu), _U(MAG_AXIAL, "0e", nu)
        joint = _rank(_joint_symmetrised_probe(Ur, Um, nu).flatten(1))

        g = torch.Generator().manual_seed(0)
        x = torch.randn(400, Ur.shape[0], 1, generator=g)
        y = torch.randn(400, 1, Um.shape[0], generator=g)
        eq = "abcp,defq,nad,nbe,ncf->npq"
        factorised = _rank(torch.einsum(eq, Ur, Um, *([x * y] * nu)).flatten(1))

        assert factorised < joint, (
            "a factorised probe did not under-report the rank, so this test cannot "
            "detect the isolated-symmetrisation error it exists to guard against"
        )


# ----------------------------------------------------------
# The determinant channel
# ----------------------------------------------------------
def test_determinant_identity_is_generically_nonzero_with_one_channel():
    """eps_ijf eps_lmg A_kil A_kjm A_kfg = 6 det A_k, nonzero with one channel k.

    The same channel sits in every slot and the contraction still does not
    vanish, so slot antisymmetry alone does not make a coupling dead.
    """
    with default_dtype(torch.float64):
        g = torch.Generator().manual_seed(0)
        A = torch.randn(16, 3, 3, generator=g)
        eps = torch.zeros(3, 3, 3)
        for p in itertools.permutations(range(3)):
            eps[p] = np.sign(np.linalg.det(np.eye(3)[list(p)]))
        lhs = torch.einsum("ijf,lmg,kil,kjm,kfg->k", eps, eps, A, A, A)
        rhs = 6.0 * torch.linalg.det(A)
        assert torch.allclose(lhs, rhs, atol=1e-10)
        assert lhs.abs().max() > 1.0, "determinant channel came out numerically zero"


def test_determinant_channel_needs_a_pseudoscalar_spatial_partner():
    """det A_k is P-odd, so its spatial partner lives in 0o; the 0e target has no such path.

    This, and not slot antisymmetry, is why the coupling cannot appear in the
    model, whose spatial target is 0e.
    """

    def antisym_content(irreps, out):
        U = _U(irreps, out, 3)
        n = 0
        for p in range(U.shape[-1]):
            C = U[..., p]
            asym = sum(
                np.sign(np.linalg.det(np.eye(3)[list(pi)])) * C.permute(pi)
                for pi in itertools.permutations(range(3))
            )
            n += int(asym.abs().max() > 1e-9)
        return U.shape[-1], n

    with default_dtype(torch.float64):
        n_even, asym_even = antisym_content(SPATIAL, "0e")
        n_odd, asym_odd = antisym_content(SPATIAL, "0o")
        assert (
            asym_even == 0
        ), f"0e target unexpectedly has {asym_even} antisymmetric path(s)"
        assert (
            asym_odd >= 1
        ), "0o target should carry the pseudoscalar (determinant) path"
        assert n_even > 0 and n_odd > 0


# ----------------------------------------------------------
# Selection rules on the spin axis
# ----------------------------------------------------------
@pytest.mark.parametrize("nu", [2, 3])
def test_polar_spin_axis_encodes_the_time_reversal_rule(nu):
    """With "o" read as T-odd on the spin axis, the parity arithmetic is the T rule.

    At nu = 3 the axial label admits one extra coupling, l' = (1, 1, 1), which is
    T-odd and which the polar labelling excludes for free.
    """
    with default_dtype(torch.float64):
        n_polar = _U(MAG_POLAR, "0e", nu).shape[-1]
        n_axial = _U(MAG_AXIAL, "0e", nu).shape[-1]
        if nu == 3:
            assert n_axial == n_polar + 1, (
                f"expected the axial label to admit exactly one extra (T-forbidden) "
                f"coupling, got {n_axial} against {n_polar}"
            )
        else:
            assert n_axial == n_polar


@pytest.mark.parametrize("nu", [2, 3])
def test_the_surplus_axial_coupling_is_the_antisymmetric_one(nu):
    """The excluded coupling is antisymmetric under slot exchange (the m.(m x m) family).

    A labelling statement only -- antisymmetry does not by itself make a coupling
    dead, as the determinant tests above show.
    """
    if nu != 3:
        pytest.skip("the surplus coupling first appears at correlation 3")
    with default_dtype(torch.float64):
        U_ax = _U(MAG_AXIAL, "0e", nu)
        n_asym = 0
        for q in range(U_ax.shape[-1]):
            C = U_ax[..., q]
            acc = torch.zeros_like(C)
            for pi in itertools.permutations(range(3)):
                sgn = np.sign(np.linalg.det(np.eye(3)[list(pi)]))
                acc = acc + sgn * C.permute(pi)
            n_asym += int(acc.abs().max() > 1e-9)
        assert (
            n_asym == 1
        ), f"expected exactly one antisymmetric magnetic coupling, found {n_asym}"


@pytest.mark.parametrize("nu", [2, 3])
def test_joint_basis_rank_is_label_independent(nu):
    """Both labels span the same invariant space, decided by rank, not parameter count."""
    with default_dtype(torch.float64):
        Ur = _U(SPATIAL, "0e", nu)
        axial = _joint_symmetrised_probe(Ur, _U(MAG_AXIAL, "0e", nu), nu).flatten(1)
        polar = _joint_symmetrised_probe(Ur, _U(MAG_POLAR, "0e", nu), nu).flatten(1)
        ra, rp = _rank(axial), _rank(polar)
        union = _rank(torch.cat([axial, polar], dim=1))
        assert (
            ra == rp == union
        ), f"nu={nu}: rank(axial)={ra}, rank(polar)={rp}, rank(union)={union}"


# ----------------------------------------------------------
# Model-level symmetries
# ----------------------------------------------------------
@pytest.mark.parametrize("correlation", [2, 3])
@pytest.mark.parametrize(
    "name, transform",
    [
        # E(Rr, m) = E(r, m)   -- spatial rotation, spin untouched
        ("spatial_rotation", lambda r, m, R, S: (r @ R.T, m)),
        # E(r, Sm) = E(r, m)   -- spin rotation, positions untouched
        ("spin_rotation", lambda r, m, R, S: (r, m @ S.T)),
        # E(-r, m) = E(r, m)   -- inversion P: (r, m) -> (-r, m), the moments are AXIAL
        ("spatial_inversion", lambda r, m, R, S: (-r, m)),
        # E(r, -m) = E(r, m)   -- time reversal T: m -> -m
        ("time_reversal", lambda r, m, R, S: (r, -m)),
        # all of them at once, with R and S drawn INDEPENDENTLY
        ("independent_R_and_S", lambda r, m, R, S: (-r @ R.T, -m @ S.T)),
    ],
)
def test_the_four_nonsoc_identities(correlation, name, transform):
    """E(Rr,m) = E(r,Sm) = E(-r,m) = E(r,-m) = E(r,m) for independent R, S in SO(3).

    Together these certify O(3)_space x O(3)_spin rather than only the diagonal
    subgroup, which is the defining property of the non-SOC model.
    """
    with default_dtype(torch.float64):
        torch.manual_seed(0)
        model = _build(max_ell=2, correlation=correlation)
        pos, mag = _rand_state()
        R = torch.tensor(Rot.random(rng=1).as_matrix(), dtype=torch.float64)
        S = torch.tensor(Rot.random(rng=2).as_matrix(), dtype=torch.float64)
        e0 = _energy(model, pos, mag)
        e1 = _energy(model, *transform(pos, mag, R, S))
        assert (
            abs(e0 - e1) < 1e-9
        ), f"{name}: violated by {abs(e0 - e1):.3e} (E = {e0:.6f})"


@pytest.mark.parametrize("correlation", [2, 3])
def test_energy_is_invariant_under_spatial_inversion(correlation):
    """P: r -> -r with the moments untouched, since they are axial."""
    with default_dtype(torch.float64):
        torch.manual_seed(0)
        model = _build(max_ell=2, correlation=correlation)
        pos, mag = _rand_state()
        e0, e1 = _energy(model, pos, mag), _energy(model, -pos, mag)
        assert abs(e0 - e1) < 1e-10, f"P-invariance violated by {abs(e0 - e1):.3e}"


@pytest.mark.parametrize("correlation", [2, 3])
def test_energy_is_invariant_under_time_reversal(correlation):
    """T: m -> -m with the positions untouched."""
    with default_dtype(torch.float64):
        torch.manual_seed(0)
        model = _build(max_ell=2, correlation=correlation)
        pos, mag = _rand_state()
        e0, e1 = _energy(model, pos, mag), _energy(model, pos, -mag)
        assert abs(e0 - e1) < 1e-10, f"T-invariance violated by {abs(e0 - e1):.3e}"


@pytest.mark.parametrize("correlation", [2, 3])
def test_space_and_spin_rotate_independently(correlation):
    """Invariance under O(3)_space x O(3)_spin, not merely the diagonal subgroup."""
    with default_dtype(torch.float64):
        torch.manual_seed(0)
        model = _build(max_ell=2, correlation=correlation)
        pos, mag = _rand_state()
        Rs = torch.tensor(Rot.random(rng=1).as_matrix(), dtype=torch.float64)
        Rm = torch.tensor(Rot.random(rng=2).as_matrix(), dtype=torch.float64)
        e0 = _energy(model, pos, mag)
        e1 = _energy(model, pos @ Rs.T, mag @ Rm.T)
        assert (
            abs(e0 - e1) < 1e-9
        ), f"independent rotations violated by {abs(e0 - e1):.3e}"
