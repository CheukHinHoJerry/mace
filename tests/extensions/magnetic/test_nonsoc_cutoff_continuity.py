"""Energy continuity at r_max for the non-SOC magnetic interaction blocks.

The radial weights of MagneticRealAgnosticNonSpinOrbitCoupledDensityInteractionBlock are
an MLP of [edge_feats, magmom_feats_j]. edge_feats vanish at r_max but the neighbour's
moment features do not, so a neighbour crossing r_max at nonzero moment changes the
energy by a finite step. The Envelope variant multiplies both weight sets and the magmom
edge density by the polynomial envelope and is continuous; it has the same parameters.
"""

import numpy as np
import pytest
import torch
from e3nn import o3

from mace.modules import interaction_classes
from mace.modules.extensions import MagneticNonSOCScaleShiftMACE
from mace.tools.torch_tools import default_dtype

_RMAX = 4.0
_OLD = "MagneticRealAgnosticNonSpinOrbitCoupledDensityInteractionBlock"
_NEW = "MagneticRealAgnosticNonSpinOrbitCoupledDensityEnvelopeInteractionBlock"
# Atoms 0 and 1 are fixed neighbours; atom 2 sits on the x axis at distance d from atom 0
# and stays more than r_max from atom 1, so only the 0-2 edge crosses the cutoff.
_POS01 = np.array([[0.0, 0.0, 0.0], [-2.0, 0.3, 0.1]])
_MAG = np.array([[0.1, 0.2, 2.2], [0.2, -0.3, 2.1], [-0.1, 0.1, 2.0]])  # ferromagnetic


def _batch(pos, mag):
    n = len(pos)
    s, d = [], []
    for i in range(n):
        for j in range(n):
            if i != j and np.linalg.norm(pos[i] - pos[j]) < _RMAX:
                s.append(j)
                d.append(i)
    ei = torch.tensor([s, d], dtype=torch.long)
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


def _build(name, seed=1):
    torch.manual_seed(seed)
    return MagneticNonSOCScaleShiftMACE(**_build_kwargs(name))


def _build_kwargs(name):
    return dict(
        r_max=_RMAX,
        num_bessel=4,
        num_polynomial_cutoff=4,
        max_ell=2,
        interaction_cls=interaction_classes[name],
        interaction_cls_first=interaction_classes["RealAgnosticDensityInteractionBlock"],
        contraction_cls_first="SymmetricContraction",
        contraction_cls="NonSOCSymmetricContraction",
        num_interactions=2,
        num_elements=1,
        hidden_irreps=o3.Irreps("8x0e+8x1o"),
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


def _energy(model, d, mag=_MAG):
    pos = np.vstack([_POS01, [[d, 0.0, 0.0]]])
    out = model(
        _batch(pos, mag), training=False, compute_force=False, compute_stress=False
    )
    return float(out["energy"])


def _step(model, mag=_MAG, eps=1e-9):
    """E(atom 2 just inside r_max) - E(just outside)."""
    return _energy(model, _RMAX - eps, mag) - _energy(model, _RMAX + eps, mag)


def test_old_block_steps_at_cutoff():
    """Documents the bug: a finite step as a moment-carrying neighbour crosses r_max."""
    with default_dtype(torch.float64):
        step = _step(_build(_OLD))
    assert abs(step) > 1e-6, step


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_envelope_block_is_continuous_at_cutoff(seed):
    with default_dtype(torch.float64):
        model = _build(_NEW, seed)
        assert abs(_step(model)) < 1e-10
        # also with zero moments and with antiparallel moments on the crossing atom
        assert abs(_step(model, _MAG * 0.0)) < 1e-10
        mag = _MAG.copy()
        mag[2] *= -1
        assert abs(_step(model, mag)) < 1e-10


def test_envelope_block_force_is_continuous_at_cutoff():
    """The polynomial envelope (p=4) also takes the force on the crossing atom to 0."""
    with default_dtype(torch.float64):
        model = _build(_NEW)
        pos = np.vstack([_POS01, [[_RMAX - 1e-6, 0.0, 0.0]]])
        f = model(_batch(pos, _MAG), training=False, compute_force=True)["forces"]
        pos_far = np.vstack([_POS01, [[_RMAX + 1e-6, 0.0, 0.0]]])
        f_far = model(_batch(pos_far, _MAG), training=False, compute_force=True)[
            "forces"
        ]
    assert torch.allclose(f, f_far, atol=1e-8), (f - f_far).abs().max()


def test_envelope_block_has_old_parameter_names_and_loads_old_weights():
    """Same parameters, names and shapes: an old checkpoint loads strictly by name."""
    with default_dtype(torch.float64):
        old, new = _build(_OLD), _build(_NEW, seed=7)
        sd_old, sd_new = old.state_dict(), new.state_dict()
        assert sd_old.keys() == sd_new.keys()
        assert all(sd_old[k].shape == sd_new[k].shape for k in sd_old)
        new.load_state_dict(sd_old, strict=True)
        # with the same weights the models still differ, through the envelope only
        assert _energy(old, 3.0) != _energy(new, 3.0)
        assert not getattr(old.interactions[1], "apply_envelope", False)


def test_envelope_is_outside_the_radial_basis():
    """Envelope models use the raw radial basis and apply u(r) once, in the blocks."""
    with default_dtype(torch.float64):
        old, new = _build(_OLD), _build(_NEW)
        r = torch.tensor([[1.0], [2.5], [_RMAX - 1e-3]])
        args = (torch.ones(3, 1), torch.zeros(2, 3, dtype=torch.long), old.atomic_numbers)
        feats_old, cut_old = old.radial_embedding(r, *args)
        feats_new, cut_new = new.radial_embedding(r, *args)
    assert old.radial_embedding.apply_cutoff and not new.radial_embedding.apply_cutoff
    assert cut_old is None
    assert torch.allclose(feats_old, feats_new * cut_new)


def test_envelope_block_cannot_be_mixed_with_block_without_cutoff():
    """Every interaction must apply u(r) once the radial basis no longer carries it."""
    with default_dtype(torch.float64), pytest.raises(ValueError, match="cutoff"):
        MagneticNonSOCScaleShiftMACE(
            **{
                **_build_kwargs(_NEW),
                "interaction_cls_first": interaction_classes[_OLD],
            }
        )


def test_envelope_block_requires_cutoff():
    with default_dtype(torch.float64):
        model = _build(_NEW)
        block = model.interactions[1]
        n, e = 3, 2
        with pytest.raises(ValueError, match="cutoff"):
            block(
                node_attrs=torch.ones(n, 1),
                node_feats=torch.zeros(n, 8),
                edge_attrs=torch.zeros(e, 9),
                edge_feats=torch.zeros(e, 4),
                edge_index=torch.tensor([[0, 1], [1, 0]]),
                magmom_node_inv_feats=torch.zeros(n, 4),
                magmom_node_attrs=torch.zeros(n, 4),
            )


def test_pooled_outer_product_matches_per_edge_scatter():
    from mace.modules.blocks import pooled_outer_product
    from mace.tools.scatter import scatter_sum

    torch.manual_seed(0)
    with default_dtype(torch.float64):
        n_nodes, n_edges = 7, 40
        receiver = torch.randint(0, n_nodes - 1, (n_edges,))  # last node has no edges
        r = torch.randn(n_edges, 3, 4, requires_grad=True)
        m = torch.randn(n_edges, 3, 5, requires_grad=True)
        ref = scatter_sum(
            torch.einsum("ekl,ekp->eklp", r, m), receiver, dim=0, dim_size=n_nodes
        )
        out = pooled_outer_product(r, m, receiver, n_nodes)
        assert torch.allclose(out, ref, atol=1e-13)
        w = torch.randn_like(ref)
        g_ref = torch.autograd.grad((ref * w).sum(), (r, m), create_graph=True)
        g_out = torch.autograd.grad((out * w).sum(), (r, m), create_graph=True)
        for a, b in zip(g_ref, g_out):
            assert torch.allclose(a, b, atol=1e-13)
        assert pooled_outer_product(r[:0], m[:0], receiver[:0], n_nodes).abs().sum() == 0


def test_model_energy_and_forces_same_with_receiver_pooling():
    from mace.modules.blocks import (
        MagneticRealAgnosticNonSpinOrbitCoupledDensityEnvelopeInteractionBlock as B,
    )

    pos = np.array([[0, 0, 0], [-2.0, 0.3, 0.1], [1.5, 0.4, -0.2], [0.3, 1.7, 0.5]])
    mag = np.vstack([_MAG, [[0.3, -0.2, 1.9]]])
    out = {}
    with default_dtype(torch.float64):
        model = _build(_NEW)
        for flag in (False, True):
            B.pool_by_receiver = flag
            o = model(_batch(pos, mag), training=True, compute_magforces=True)
            out[flag] = [o[k].detach() for k in ("energy", "forces", "magforces")]
    B.pool_by_receiver = True
    for a, b in zip(out[False], out[True]):
        assert torch.allclose(a, b, atol=1e-12)


def test_magmom_attrs_by_sat_rescales_spin_features_only():
    """m_eff / m_sat: solid harmonics scale by m_sat^-l; the model stays spin-rotation
    invariant and the flag is off by default."""
    with default_dtype(torch.float64):
        assert not _build(_NEW).magmom_attrs_by_sat
        kw = _build_kwargs(_NEW)
        torch.manual_seed(1)
        model = MagneticNonSOCScaleShiftMACE(**{**kw, "magmom_attrs_by_sat": True})
        pos = np.vstack([_POS01, [[3.0, 0.0, 0.0]]])
        e = _energy(model, 3.0)
        rot = torch.tensor(
            [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=torch.float64
        ).numpy()
        out = model(_batch(pos, _MAG @ rot.T), training=False, compute_force=False)
        assert float(out["energy"].detach()) == pytest.approx(e, abs=1e-10)
        with pytest.raises(ValueError, match="magmom_sat_scale"):
            MagneticNonSOCScaleShiftMACE(
                **{**kw, "magmom_attrs_by_sat": True, "magmom_sat_scale": 0.0}
            )
