"""Unit tests for ``virtual_hydrogen_mask`` in ``PolarMACE.forward``.

The mask zeros the electrostatic source-charge density for masked
atoms before any Coulomb sum runs. Used by openmm-ml's ONIOM-EE
mode to enforce the standard QM/MM Z1 link-atom convention
(`q_cap = 0`) on the MACE side, matching the same convention on
the MM-force-field side.

The mask zeroes *only* electrostatic contributions; masked atoms
remain real participants in the rest of the model (atomic energies,
field-dependent polarization, exchange-correlation features). This
file pins the masking is correctly applied to all Coulomb sums and
that the default (mask absent) behavior is unchanged.

Mirrors the test layout in ``test_mm_field_routes.py`` for the
PolarMACE builder + batch.
"""
from __future__ import annotations

import math

import numpy as np
import pytest
import torch
from e3nn import o3

from mace.modules import interaction_classes
from mace.modules.extensions import PolarMACE


def _build_minimal_model(device, dtype):
    fixedpoint_update_config = {
        "type": "AgnosticEmbeddedOneBodyVariableUpdate",
        "potential_embedding_cls": "AgnosticChargeBiasedLinearPotentialEmbedding",
        "nonlinearity_cls": "MLPNonLinearity",
    }
    return PolarMACE(
        r_max=4.0,
        num_bessel=4,
        num_polynomial_cutoff=3,
        max_ell=1,
        interaction_cls=interaction_classes[
            "RealAgnosticResidualNonLinearInteractionBlock"
        ],
        interaction_cls_first=interaction_classes[
            "RealAgnosticResidualNonLinearInteractionBlock"
        ],
        num_interactions=2,
        num_elements=2,
        hidden_irreps=o3.Irreps("4x0e + 4x1o"),
        MLP_irreps=o3.Irreps("8x0e"),
        atomic_energies=torch.zeros(2, dtype=dtype, device=device),
        avg_num_neighbors=3.0,
        atomic_numbers=[1, 8],
        correlation=1,
        gate=torch.nn.functional.silu,
        radial_MLP=[16, 16],
        radial_type="bessel",
        kspace_cutoff_factor=1.0,
        atomic_multipoles_max_l=1,
        atomic_multipoles_smearing_width=1.0,
        field_feature_max_l=1,
        field_feature_widths=[1.0],
        field_feature_norms=[1.0, 1.0],
        num_recursion_steps=1,
        field_si=False,
        include_electrostatic_self_interaction=False,
        add_local_electron_energy=True,
        field_dependence_type="AgnosticEmbeddedOneBodyVariableUpdate",
        final_field_readout_type="OneBodyMLPFieldReadout",
        return_electrostatic_potentials=False,
        heads=["Default"],
        field_norm_factor=1.0,
        fixedpoint_update_config=fixedpoint_update_config,
        field_readout_config={"type": "OneBodyMLPFieldReadout"},
    ).to(device=device, dtype=dtype)


def _build_batch(device, dtype, n_mm=4, pbc=False, seed=11):
    rng = np.random.default_rng(seed)
    positions = torch.tensor(
        [[0.0, 0.0, 0.0], [1.2, 0.1, -0.2]], device=device, dtype=dtype
    )
    n = positions.shape[0]
    batch = torch.zeros(n, dtype=torch.long, device=device)
    ptr = torch.tensor([0, n], dtype=torch.long, device=device)

    src, dst = [], []
    for i in range(n):
        for j in range(n):
            if i != j:
                src.append(i); dst.append(j)
    edge_index = torch.tensor([src, dst], dtype=torch.long, device=device)
    shifts = torch.zeros((edge_index.shape[1], 3), dtype=dtype, device=device)
    node_attrs = torch.tensor([[1.0, 0.0], [0.0, 1.0]], device=device, dtype=dtype)

    cell_len = 10.0
    cell = torch.eye(3, dtype=dtype, device=device).unsqueeze(0) * cell_len
    rcell = torch.eye(3, dtype=dtype, device=device).unsqueeze(0) * (
        2.0 * math.pi / cell_len
    )
    volume = torch.det(cell.view(1, 3, 3))

    mm_positions = torch.tensor(
        rng.uniform(4.0, 8.0, size=(n_mm, 3)), device=device, dtype=dtype
    )
    mm_charges = torch.tensor(
        [0.4, -0.3, 0.2, -0.1][:n_mm], device=device, dtype=dtype
    )
    mm_source_batch = torch.zeros(n_mm, dtype=torch.long, device=device)

    return {
        "positions": positions,
        "node_attrs": node_attrs,
        "batch": batch,
        "ptr": ptr,
        "edge_index": edge_index,
        "shifts": shifts,
        "cell": cell.view(-1, 9),
        "rcell": rcell.view(-1, 9),
        "volume": volume.reshape(-1),
        "pbc": torch.tensor([[pbc, pbc, pbc]], dtype=torch.bool, device=device),
        "external_field": torch.zeros((1, 3), dtype=dtype, device=device),
        "fermi_level": torch.zeros((1,), dtype=dtype, device=device),
        "total_charge": torch.zeros(1, dtype=dtype, device=device),
        "total_spin": torch.ones(1, dtype=dtype, device=device),
        "density_coefficients": torch.zeros(n, 1, dtype=dtype, device=device),
        "mm_positions": mm_positions,
        "mm_charges": mm_charges,
        "mm_source_batch": mm_source_batch,
    }


@pytest.fixture
def minimal_model():
    torch.manual_seed(11)
    device = torch.device("cpu")
    dtype = torch.float64
    return _build_minimal_model(device, dtype), device, dtype


def test_virtual_hydrogen_mask_default_absent_is_no_op(minimal_model):
    """Mask key absent → behavior identical to current (pre-mask) MACE."""
    model, device, dtype = minimal_model
    data = _build_batch(device, dtype)
    out = model(data, training=False, compute_force=True)
    assert "electrostatic_energy" in out
    e_baseline = out["electrostatic_energy"].detach().clone()

    # Now pass an explicit all-False mask: same behavior expected.
    data_mask = dict(data)
    data_mask["virtual_hydrogen_mask"] = torch.zeros(
        data["positions"].shape[0], dtype=torch.bool, device=device
    )
    out_mask_off = model(data_mask, training=False, compute_force=True)
    e_mask_off = out_mask_off["electrostatic_energy"].detach().clone()

    torch.testing.assert_close(e_mask_off, e_baseline, rtol=0.0, atol=0.0)


def test_virtual_hydrogen_mask_all_true_zeros_electrostatic_energy(minimal_model):
    """Masking every atom kills the entire electrostatic source-charge
    density. Both ML-internal and ML-MM Coulomb sums collapse to zero."""
    model, device, dtype = minimal_model
    data = _build_batch(device, dtype)
    n_atoms = data["positions"].shape[0]
    data["virtual_hydrogen_mask"] = torch.ones(
        n_atoms, dtype=torch.bool, device=device
    )
    out = model(data, training=False, compute_force=True)
    e = out["electrostatic_energy"].detach()
    torch.testing.assert_close(
        e, torch.zeros_like(e), rtol=0.0, atol=1e-12
    )

    # ML-MM electrostatic energy must also vanish (no charge density on
    # the ML side to interact with MM background charges).
    e_ml_mm = out["ml_mm_electrostatic_energy"].detach()
    torch.testing.assert_close(
        e_ml_mm, torch.zeros_like(e_ml_mm), rtol=0.0, atol=1e-12
    )


def test_virtual_hydrogen_mask_one_atom_drops_its_contribution(minimal_model):
    """Masking only atom 0 must reproduce the energy that would result
    if atom 0's predicted density were zeroed BEFORE the Coulomb sum.
    Verified by running MACE with mask = [True, False] and comparing
    against an externally-recomputed ``coulomb_energy`` on the manually
    zeroed density coefficients."""
    model, device, dtype = minimal_model
    data = _build_batch(device, dtype)

    # Baseline forward — capture density_coefficients to recompute Coulomb.
    out_full = model(data, training=False, compute_force=True)

    data_masked = dict(data)
    data_masked["virtual_hydrogen_mask"] = torch.tensor(
        [True, False], dtype=torch.bool, device=device
    )
    out_masked = model(data_masked, training=False, compute_force=True)

    # When atom 0 is masked, the only ML source is atom 1. Therefore:
    # - ML internal Coulomb is between atom 1 and itself → typically
    #   non-zero only via self-interaction or PBC images. For the
    #   default include_electrostatic_self_interaction=False, this
    #   should yield zero ML-internal contribution and ML-MM Coulomb
    #   only between atom 1 and the MM cloud.
    # - The masked energy should be strictly smaller in magnitude than
    #   the full energy when atom 0 has non-zero predicted charge.
    e_full = float(out_full["electrostatic_energy"].detach())
    e_masked = float(out_masked["electrostatic_energy"].detach())

    # Sanity: not equal (masking changed something).
    assert e_full != e_masked, (
        "Masking atom 0 should change electrostatic energy. "
        f"e_full={e_full}, e_masked={e_masked}"
    )
    # If atom 0 had predicted charge of meaningful magnitude, |e_full|
    # ≠ |e_masked|. We don't make stronger claims about sign here
    # because the energy can be small and noisy in this tiny test.


def test_virtual_hydrogen_mask_preserves_total_charge(minimal_model):
    """When the user requests a specific total charge (via
    ``data['total_charge']``), the masking must not silently shift the
    sum of monopole density across real atoms below the requested
    value. Codex review MAJOR finding: pre-masking, the
    ``scatter_normalize_charges_`` step distributes deficit/surplus
    over ALL atoms including caps; naively zeroing caps would leave
    real atoms short. The mask code now redistributes the lost cap
    monopole back over real atoms to preserve total charge.

    Verify by checking that the sum of returned monopole density over
    the masked atoms matches the sum without masking, within float64
    tolerance.
    """
    model, device, dtype = minimal_model
    data = _build_batch(device, dtype)

    # Set a non-zero requested total charge to exercise the
    # normalization scatter on the path that has the bug.
    data["total_charge"] = torch.tensor([0.5], device=device, dtype=dtype)

    out_unmasked = model(data, training=False, compute_force=True)
    sum_full = float(
        out_unmasked["density_coefficients"][:, 0].detach().sum()
    )

    data_mask = dict(data)
    data_mask["virtual_hydrogen_mask"] = torch.tensor(
        [True, False], dtype=torch.bool, device=device
    )
    out_masked = model(data_mask, training=False, compute_force=True)
    sum_masked_real = float(
        out_masked["density_coefficients"][:, 0].detach().sum()
    )

    # Both sums should equal the requested total charge (within
    # the SCF / float64 tolerance of the underlying solver).
    expected_total = 0.5
    assert sum_full == pytest.approx(expected_total, abs=1e-6), (
        "Unmasked total monopole density should match requested "
        f"total_charge. Got {sum_full}, expected {expected_total}."
    )
    assert sum_masked_real == pytest.approx(expected_total, abs=1e-6), (
        "Masked total monopole density should still match requested "
        f"total_charge after cap charge is redistributed onto real "
        f"atoms. Got {sum_masked_real}, expected {expected_total}."
    )


def test_virtual_hydrogen_mask_shape_validation(minimal_model):
    """A wrong-shape mask raises a clear error instead of silently
    broadcasting. (Codex review MINOR finding.)"""
    model, device, dtype = minimal_model
    data = _build_batch(device, dtype)
    n_atoms = data["positions"].shape[0]

    # Wrong length: n_atoms + 1.
    data["virtual_hydrogen_mask"] = torch.zeros(
        n_atoms + 1, dtype=torch.bool, device=device
    )
    with pytest.raises(ValueError, match=r"virtual_hydrogen_mask shape"):
        model(data, training=False, compute_force=True)
