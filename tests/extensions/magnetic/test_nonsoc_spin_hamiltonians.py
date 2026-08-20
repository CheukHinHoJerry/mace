import itertools

import numpy as np
import pytest
import torch
from e3nn import o3

from mace.tools.cg import U_matrix_real
from mace.tools.torch_tools import default_dtype

# Physical fitting tests for the centre-spin sectors.
#
# The basis is fitted by LINEAR least squares rather than by training, so what is measured
# is the span of  B^(nu,s) = [ Gamma^(nu,s) (x) Y_s(m_i) ]_0  and not optimiser behaviour.
# Columns are the free parameters of the deployed architecture -- one per (spatial path,
# centre-spin order s, magnetic path) -- with the 2s+1 spin components already summed
# against the centre moment, which is the weight sharing the model enforces.
_IRR_S = {0: "0e", 1: "1o", 2: "2e"}


def _sph(l_max, vecs, normalize):
    return o3.spherical_harmonics(
        o3.Irreps.spherical_harmonics(l_max, p=-1),
        torch.as_tensor(vecs),
        normalize=normalize,
        normalization="component",
    )


def _design(pos, mags, max_ell, max_m_ell, correlation, sectors):
    """Design matrix for the centre atom (index 0) over a batch of spin configurations."""
    pos = torch.as_tensor(pos)
    nbr = list(range(1, pos.shape[0]))
    rvec = pos[nbr] - pos[0]
    y_spatial = _sph(max_ell, rvec, normalize=True)
    # SOLID harmonics on the magnetic side, matching mag_solid_harmoics in the model, so
    # the basis is a genuine function of the vector m and d/dm is meaningful.
    a_pooled = torch.stack(
        [
            torch.einsum(
                "ja,jb->ab", y_spatial, _sph(max_m_ell, m[nbr], normalize=False)
            )
            for m in mags
        ]
    )
    centre = torch.stack([_sph(max_m_ell, m[0:1], normalize=False)[0] for m in mags])
    v_r = o3.Irreps.spherical_harmonics(max_ell, p=-1)
    v_m = o3.Irreps.spherical_harmonics(max_m_ell, p=-1)
    cols = []
    for nu in range(1, correlation + 1):
        u_r = U_matrix_real(
            irreps_in=v_r,
            irreps_out=o3.Irreps("0e"),
            correlation=nu,
            dtype=torch.float64,
            use_cueq_cg=False,
        )[-1]
        slots_r, slots_m = "abc"[:nu], "ghi"[:nu]
        for s_order in sectors:
            u_m = U_matrix_real(
                irreps_in=v_m,
                irreps_out=o3.Irreps(_IRR_S[s_order]),
                correlation=nu,
                dtype=torch.float64,
                use_cueq_cg=False,
            )[-1]
            lead = u_m.dim() - 1 - nu
            probes = ",".join(f"n{slots_r[t]}{slots_m[t]}" for t in range(nu))
            lhs_m = ("C" if lead else "") + slots_m + "q"
            out = "nkq" + ("C" if lead else "")
            val = torch.einsum(
                f"{slots_r}k,{lhs_m},{probes}->{out}", u_r, u_m, *([a_pooled] * nu)
            )
            if lead:
                lo, hi = s_order * s_order, (s_order + 1) * (s_order + 1)
                val = torch.einsum("nkqC,nC->nkq", val, centre[:, lo:hi])
            cols.append(val.reshape(len(mags), -1))
    return torch.cat(cols, dim=1)


def _fit(phi, y):
    """Half train, half test; returns test RMSE relative to the target's own spread."""
    n = phi.shape[0] // 2
    coef = torch.linalg.lstsq(
        phi[:n], y[:n].unsqueeze(-1), driver="gelsd"
    ).solution.squeeze(-1)
    rel = float(((phi[n:] @ coef - y[n:]) ** 2).mean().sqrt()) / float(y.std())
    return rel, coef


def _spins(rng, n_cfg, n_atom, unit=True):
    out = []
    for _ in range(n_cfg):
        m = torch.tensor(rng.randn(n_atom, 3))
        out.append(m / m.norm(dim=-1, keepdim=True) if unit else m * 0.9)
    return out


_DIMER = np.array([[0.0, 0.0, 0.0], [2.1, 0.0, 0.0]])
_TRIMER = np.array([[0.0, 0.0, 0.0], [2.1, 0.0, 0.0], [0.7, 1.9, 0.0]])


def _dot(a, b):
    return torch.stack([x @ y for x, y in zip(a, b)])


@pytest.mark.parametrize(
    "name,needs_s",
    [("heisenberg", 1), ("legendre_p2", 2), ("biquadratic", 2)],
)
def test_two_spin_targets_need_exactly_their_sector(name, needs_s):
    """Each spin Hamiltonian is fitted exactly by its sector and NOT by lower ones.

    Failing below the required s is the point: it shows the sector is necessary, not
    merely sufficient. Without it the fit is no better than predicting the mean.
    """
    with default_dtype(torch.float64):
        rng = np.random.RandomState(0)
        mags = _spins(rng, 400, 2)
        x = _dot([m[0] for m in mags], [m[1] for m in mags])
        target = {
            "heisenberg": x,
            "legendre_p2": 0.5 * (3 * x * x - 1),
            "biquadratic": x * x,
        }[name]
        below, _ = _fit(_design(_DIMER, mags, 2, 2, 2, tuple(range(needs_s))), target)
        at, _ = _fit(_design(_DIMER, mags, 2, 2, 2, tuple(range(needs_s + 1))), target)
        assert at < 1e-12, f"{name} not represented with s<={needs_s}: rel {at:.2e}"
        assert below > 1e-3, (
            f"{name} was already representable with s<{needs_s} (rel {below:.2e}); "
            f"the s={needs_s} sector is then not what carries it"
        )


def test_heisenberg_magnetic_force_matches_the_analytic_form():
    """After fitting E = J m_i.m_j, -dE/dm_i must equal -J m_j."""
    with default_dtype(torch.float64):
        rng = np.random.RandomState(1)
        mags = _spins(rng, 400, 2, unit=False)
        coupling = 0.734
        target = coupling * _dot([m[0] for m in mags], [m[1] for m in mags])
        _, coef = _fit(_design(_DIMER, mags, 2, 2, 2, (0, 1)), target)

        probe = [m.clone().requires_grad_(True) for m in mags[:48]]
        energy = _design(_DIMER, probe, 2, 2, 2, (0, 1)) @ coef
        force = -torch.autograd.grad(energy.sum(), probe)[0]
        reference = -coupling * mags[0][1]
        assert float((force[0] - reference).abs().max()) < 1e-10


def test_three_spin_centre_dependence_needs_s2():
    """(m_i.m_j)(m_i.m_k) is quadratic in the CENTRE moment, so it needs s=2."""
    with default_dtype(torch.float64):
        rng = np.random.RandomState(2)
        mags = _spins(rng, 400, 3)
        centre, first, second = ([m[i] for m in mags] for i in range(3))
        target = _dot(centre, first) * _dot(centre, second)
        assert _fit(_design(_TRIMER, mags, 2, 2, 2, (0, 1)), target)[0] > 1e-3
        assert _fit(_design(_TRIMER, mags, 2, 2, 2, (0, 1, 2)), target)[0] < 1e-12


def test_neighbour_only_invariant_is_blind_to_the_centre_spin():
    """m_j.m_k lives entirely in s=0, so the fitted model must ignore m_i completely."""
    with default_dtype(torch.float64):
        rng = np.random.RandomState(3)
        mags = _spins(rng, 400, 3)
        target = _dot([m[1] for m in mags], [m[2] for m in mags])
        rel, coef = _fit(_design(_TRIMER, mags, 2, 2, 2, (0,)), target)
        assert rel < 1e-12
        rot = torch.tensor(
            o3.rand_matrix(dtype=torch.float64).numpy()  # proper rotation
        )
        turned = [torch.cat([(m[0] @ rot.T).unsqueeze(0), m[1:]]) for m in mags[:64]]
        base = _design(_TRIMER, mags[:64], 2, 2, 2, (0,)) @ coef
        moved = _design(_TRIMER, turned, 2, 2, 2, (0,)) @ coef
        assert float((moved - base).abs().max()) < 1e-12


def test_correlation_three_invariant_needs_nu_three():
    """(m_i.m_j)(m_k.m_l) is degree 3 in the neighbour moments, so nu=2 cannot reach it."""
    with default_dtype(torch.float64):
        rng = np.random.RandomState(4)
        pos = np.array(
            [[0.0, 0.0, 0.0], [2.1, 0.0, 0.0], [0.7, 1.9, 0.0], [-0.6, 0.8, 1.7]]
        )
        mags = _spins(rng, 500, 4)
        target = _dot([m[0] for m in mags], [m[1] for m in mags]) * _dot(
            [m[2] for m in mags], [m[3] for m in mags]
        )
        assert _fit(_design(pos, mags, 2, 2, 2, (0, 1, 2)), target)[0] > 1e-3
        assert _fit(_design(pos, mags, 2, 2, 3, (0, 1, 2)), target)[0] < 1e-10
