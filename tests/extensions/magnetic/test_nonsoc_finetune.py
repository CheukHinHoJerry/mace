"""Fine-tuning from a MagneticNonSOCScaleShiftMACE foundation.

Covers the three pieces run_train relies on when --foundation_model is a non-SOC
checkpoint: extract_config_mace_model reports the magnetic hyperparameters, the
inherited --m_max is keyed by atomic number so it survives any element table, and
load_foundations_elements transfers every weight so that each head of the new
(multihead) model reproduces the foundation energies and forces exactly.
"""

import argparse

import numpy as np
import pytest
import torch
from e3nn import o3

from mace.modules import MagneticNonSOCScaleShiftMACE, interaction_classes
from mace.tools.finetuning_utils import load_foundations_elements
from mace.tools.multihead_tools import inherit_magnetic_hyperparameters_from_foundation
from mace.tools.scripts_utils import extract_config_mace_model, resolve_m_max
from mace.tools.torch_tools import default_dtype
from mace.tools.utils import AtomicNumberTable

_RMAX = 3.0
_ZS = [26, 28]


def _build(heads=None, atomic_numbers=None, seed=1, use_one_body=True):
    torch.manual_seed(seed)
    zs = list(atomic_numbers or _ZS)
    heads = list(heads or ["Default"])
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
        num_elements=len(zs),
        hidden_irreps=o3.Irreps("8x0e"),
        MLP_irreps=o3.Irreps("4x0e"),
        atomic_energies=np.zeros((len(heads), len(zs))),
        avg_num_neighbors=2.0,
        atomic_numbers=zs,
        correlation=2,
        gate=torch.nn.functional.silu,
        atomic_inter_shift=[0.0] * len(heads),
        atomic_inter_scale=[1.0] * len(heads),
        m_max=[2.5, 1.5][: len(zs)],
        num_mag_radial_basis=4,
        num_mag_radial_basis_one_body=3,
        max_m_ell=1,
        use_magmom_one_body=use_one_body,
        heads=heads,
    )


def _randomize_(model, seed):
    """Foundation weights must not be at their init (the one-body head starts at
    zero, for example), otherwise a broadcast that silently skips a tensor passes."""
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in model.parameters():
            p.copy_(torch.randn(p.shape, generator=g, dtype=p.dtype))


def _batch(head, n_heads, seed=0, n=6):
    g = torch.Generator().manual_seed(seed)
    pos = torch.randn(n, 3, dtype=torch.float64, generator=g) * 1.2
    mag = torch.randn(n, 3, dtype=torch.float64, generator=g)
    species = torch.arange(n) % len(_ZS)
    s, d = [], []
    for i in range(n):
        for j in range(n):
            if i != j and torch.linalg.norm(pos[i] - pos[j]) < _RMAX:
                s.append(j)
                d.append(i)
    ei = torch.tensor([s, d])
    return {
        "positions": pos.clone().requires_grad_(True),
        "cell": torch.zeros(1, 3, 3),
        "batch": torch.zeros(n, dtype=torch.long),
        "ptr": torch.tensor([0, n]),
        "node_attrs": torch.nn.functional.one_hot(species, len(_ZS)).to(torch.float64),
        "edge_index": ei,
        "magmom": mag,
        "unit_shifts": torch.zeros(ei.shape[1], 3),
        "shifts": torch.zeros(ei.shape[1], 3),
        "head": torch.full((1,), head, dtype=torch.long),
        "node_heads": torch.full((n,), head, dtype=torch.long),
    }


def _eval(model, batch):
    out = model(batch, training=False, compute_force=True, compute_stress=False)
    return out["energy"].detach(), out["forces"].detach()


def test_extract_config_reports_nonsoc_magnetic_hyperparameters():
    with default_dtype(torch.float64):
        foundation = _build()
        config = extract_config_mace_model(foundation)
    assert "error" not in config
    assert config["m_max"] == [2.5, 1.5]
    assert config["max_m_ell"] == 1
    assert config["num_mag_radial_basis"] == 4
    assert config["use_magmom_one_body"] is True
    assert config["num_mag_radial_basis_one_body"] == 3
    assert config["one_body_spectral_degree"] == 0.0
    assert config["magmom_sat_scale"] == 1.0


def test_inherited_m_max_is_keyed_by_atomic_number():
    with default_dtype(torch.float64):
        foundation = _build()
    args = argparse.Namespace(m_max=None, max_m_ell=None, num_mag_radial_basis=None)
    inherited = inherit_magnetic_hyperparameters_from_foundation(args, foundation)
    assert inherited["max_m_ell"] == 1
    assert inherited["num_mag_radial_basis"] == 4
    assert inherited["num_mag_radial_basis_one_body"] == 3
    assert args.use_magmom_one_body is True
    assert args.magmom_sat_scale == 1.0
    # same table, subset, and re-ordered/larger tables all resolve without a length error
    assert resolve_m_max(args.m_max, [26, 28]) == [2.5, 1.5]
    assert resolve_m_max(args.m_max, [28]) == [1.5]
    assert resolve_m_max(args.m_max, [1, 26, 28]) == [1.0, 2.5, 1.5]


@pytest.mark.parametrize("heads", [["Default"], ["Default", "pt_head"]])
def test_every_head_reproduces_the_foundation(heads):
    with default_dtype(torch.float64):
        foundation = _build()
        _randomize_(foundation, seed=7)
        model = _build(heads=heads, seed=2)
        model = load_foundations_elements(
            model,
            foundation,
            AtomicNumberTable(_ZS),
            load_readout=True,
            max_L=0,
            default_dtype=torch.float64,
        )
        e_ref, f_ref = _eval(foundation, _batch(head=0, n_heads=1))
        for h in range(len(heads)):
            e, f = _eval(model, _batch(head=h, n_heads=len(heads)))
            assert torch.allclose(e, e_ref, atol=1e-10, rtol=0), (heads, h, e, e_ref)
            assert torch.allclose(f, f_ref, atol=1e-10, rtol=0), (heads, h)


def test_element_subset_is_refused():
    with default_dtype(torch.float64):
        foundation = _build()
        model = _build(atomic_numbers=[26])
        with pytest.raises(ValueError, match="foundation_model_elements"):
            load_foundations_elements(
                model, foundation, AtomicNumberTable([26]), load_readout=True, max_L=0
            )


def test_architecture_mismatch_is_reported_not_skipped():
    with default_dtype(torch.float64):
        foundation = _build()
        model = _build(use_one_body=False)  # one-body tensors absent in the new model
        with pytest.raises(ValueError, match="onebody_magmombasis_coeffs"):
            load_foundations_elements(
                model, foundation, AtomicNumberTable(_ZS), load_readout=True, max_L=0
            )
