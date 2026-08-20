import itertools
import math

import pytest
import torch
from e3nn import o3

from mace.tools.cg import U_matrix_real
from mace.modules.symmetric_contraction_nonsoc import (
    _joint_symmetrised_probe,
    _rank,
)
from mace.tools.torch_tools import default_dtype

# True dimension of the jointly symmetrised equivariant space, target (spatial 0e,
# spin trivial i.e. T-EVEN). Computed independently three ways and cross-checked:
#   * O(3) x O(3) character integration over the Cauchy decomposition
#         Sym^nu(V_r (x) V_m) = (+)_{lambda |- nu} S^lambda(V_r) (x) S^lambda(V_m)
#   * an explicit Reynolds operator on representation matrices, using no CG at all
#   * the rank of the shipped basis itself
# Parity MUST be carried on BOTH sides: a spin l=0 piece that is T-ODD is not the trivial
# representation, and counting it inflates these numbers (13/34/61 -> 16/45/92).
_TRUE_JOINT_DIM = {
    ("0e+1o+2e", "0e+1o"): {2: 6, 3: 13},
    ("0e+1o+2e", "0e+1o+2e"): {2: 9, 3: 34},
    ("0e+1o+2e+3o", "0e+1o+2e"): {2: 12, 3: 61},
}


def _basis(irreps, correlation):
    return U_matrix_real(
        irreps_in=o3.Irreps(irreps),
        irreps_out=o3.Irreps("0e"),
        correlation=correlation,
        dtype=torch.float64,
        use_cueq_cg=False,
    )[-1]


def _symmetrise(U, correlation):
    """Project the slot indices onto the symmetric subspace and orthonormalise.

    This is what cue.reduced_symmetric_tensor_product_basis returns a basis of, so it
    stands in for the cuequivariance path without requiring it to be installed.
    """
    slots = list(range(U.dim() - 1))
    acc = torch.zeros_like(U)
    for perm in itertools.permutations(slots):
        acc = acc + U.permute(*perm, U.dim() - 1)
    acc = acc / math.factorial(correlation)
    flat = acc.reshape(-1, acc.shape[-1])
    left, sing, _ = torch.linalg.svd(flat, full_matrices=False)
    keep = int((sing > 1e-9 * max(1.0, float(sing[0]))).sum())
    return left[:, :keep].reshape(*acc.shape[:-1], keep).contiguous()


@pytest.mark.parametrize("correlation", [2, 3])
@pytest.mark.parametrize("spatial,magmom", list(_TRUE_JOINT_DIM))
def test_paired_basis_spans_the_whole_joint_space(spatial, magmom, correlation):
    """The factorised basis must span all of Sym^nu(V_r (x) V_m), not a subspace of it.

    The spatial and magmom bases are built separately and paired, which is complete only
    because each is the FULL UNSYMMETRISED coupling-tree basis: contracting against nu
    copies of the same A then projects the product onto the jointly symmetric subspace.
    A regression to symmetric-per-space bases -- for instance a silent fall back to
    cuequivariance inside U_matrix_real -- shows up here as a rank deficit.
    """
    with default_dtype(torch.float64):
        u_spatial = _basis(spatial, correlation)
        u_magmom = _basis(magmom, correlation)
        rank = _rank(
            _joint_symmetrised_probe(u_spatial, u_magmom, correlation).flatten(1)
        )
        expected = _TRUE_JOINT_DIM[(spatial, magmom)][correlation]
        assert rank == expected, (
            f"joint rank {rank} != {expected} for V_r={spatial}, V_m={magmom}, "
            f"nu={correlation}; the paired basis no longer spans the joint space"
        )


def test_separately_symmetric_bases_would_lose_channels():
    """Guards the premise: pairing two per-space SYMMETRIC bases is strictly weaker.

    At nu=2 nothing is lost, because Lambda^2(V_m) contains no spin scalar for
    multiplicity-one magmom irreps. The loss appears at nu=3 through the mixed
    lambda=(2,1) sector, so a nu=2 test cannot detect it.
    """
    with default_dtype(torch.float64):
        spatial, magmom = "0e+1o+2e", "0e+1o"
        for correlation, expect_loss in ((2, False), (3, True)):
            u_spatial = _basis(spatial, correlation)
            u_magmom = _basis(magmom, correlation)
            full = _rank(
                _joint_symmetrised_probe(u_spatial, u_magmom, correlation).flatten(1)
            )
            sym = _rank(
                _joint_symmetrised_probe(
                    _symmetrise(u_spatial, correlation),
                    _symmetrise(u_magmom, correlation),
                    correlation,
                ).flatten(1)
            )
            assert full == _TRUE_JOINT_DIM[(spatial, magmom)][correlation]
            if expect_loss:
                assert sym < full, (
                    "symmetric-per-space bases should lose the mixed Young sectors at "
                    f"nu={correlation}, but matched the full basis at rank {sym}"
                )
            else:
                assert sym == full


def test_contraction_does_not_ask_for_the_cueq_basis():
    """The non-SOC contraction must never request the cuequivariance CG basis.

    compute_U_cueq returns a per-space SYMMETRIC basis, which would make the model's
    hypothesis class depend on whether cuequivariance is installed.
    """
    import inspect

    from mace.modules import symmetric_contraction_nonsoc as mod

    src = inspect.getsource(mod.NonSOCContraction)
    assert "use_cueq_cg=True" not in src, (
        "NonSOCContraction requests the cueq CG basis; that silently drops the mixed "
        "Young-symmetry sectors when cuequivariance is installed"
    )


# Joint rank retained per centre-spin sector, max_ell=3 / max_m_ell=2 / nu=3.
# Verified two independent ways: S_3 isotypic multiplicities from O(3) characters
# (n_r^lambda . n_m^lambda summed over lambda), and the rank of the shipped bases.
#   s=0: 8(5)+7(3)+1(0) = 61      s=1: 8(5)+7(7)+1(2) = 91
#   s=2: 8(6)+7(8)+1(1) = 105     total 257
_CENTRE_SECTOR_DIM = {"0e": 61, "1o": 91, "2e": 105}


@pytest.mark.parametrize("spin_out,expected", sorted(_CENTRE_SECTOR_DIM.items()))
def test_centre_spin_sectors_span_their_joint_space(spin_out, expected):
    """Each centre-spin order must retain its full jointly symmetrised rank.

    The magmom basis for order s carries 2s+1 free spin indices, later closed against the
    centre moment's own l=s block, so the rank is taken with that index treated as part of
    the observation rather than summed away.
    """
    with default_dtype(torch.float64):
        u_spatial = _basis("0e+1o+2e+3o", 3)
        u_magmom = U_matrix_real(
            irreps_in=o3.Irreps("0e+1o+2e"),
            irreps_out=o3.Irreps(spin_out),
            correlation=3,
            dtype=torch.float64,
            use_cueq_cg=False,
        )[-1]
        lead = u_magmom.dim() - 4
        gen = torch.Generator().manual_seed(0)
        n_r, n_m = u_spatial.shape[-1], u_magmom.shape[-1]
        probe = torch.randn(
            max(512, 4 * n_r * n_m),
            u_spatial.shape[0],
            u_magmom.shape[lead],
            generator=gen,
            dtype=torch.float64,
        )
        if lead:
            vals = torch.einsum(
                "abcp,jghiq,nag,nbh,nci->npqj", u_spatial, u_magmom, probe, probe, probe
            )
            flat = vals.permute(0, 3, 1, 2).reshape(-1, n_r * n_m)
        else:
            vals = torch.einsum(
                "abcp,ghiq,nag,nbh,nci->npq", u_spatial, u_magmom, probe, probe, probe
            )
            flat = vals.reshape(-1, n_r * n_m)
        assert _rank(flat) == expected
