"""Equivalence tests for the two MM->QM electrostatic-field implementations.

PolarMACE keeps two implementations of the MM->QM field descriptor:

- ``_compute_mm_field_features``                 (legacy, concat-and-slice)
- ``_compute_mm_field_features_source_target``   (active, source-target)

They are mathematically equivalent (the legacy path zero-pads QM source rows
and slices the QM tail off the projection; source-target avoids that wasted
work). This file asserts that equivalence end-to-end on a small batch by
toggling ``model.mm_field_route`` and comparing energy, QM forces, and MM
forces in float64.

Run in float32 too with a looser tolerance to flag any pathological drift.
"""
from __future__ import annotations

import math

import numpy as np
import pytest
import torch
from e3nn import o3

from mace.modules import interaction_classes
from mace.modules.extensions import PolarMACE


def _build_minimal_model(device, dtype, *, include_dipole_mm_interaction=False):
    """A tiny PolarMACE matching the one in test_polar_models.py.

    Inlined here (not imported) so this file is self-contained and pytest
    can collect it without sibling-test path tricks.
    """
    num_elements = 2
    atomic_numbers = [1, 8]
    hidden_irreps = o3.Irreps("4x0e + 4x1o")
    MLP_irreps = o3.Irreps("8x0e")

    fixedpoint_update_config = {
        "type": "AgnosticEmbeddedOneBodyVariableUpdate",
        "potential_embedding_cls": "AgnosticChargeBiasedLinearPotentialEmbedding",
        "nonlinearity_cls": "MLPNonLinearity",
    }
    field_readout_config = {"type": "OneBodyMLPFieldReadout"}

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
        num_elements=num_elements,
        hidden_irreps=hidden_irreps,
        MLP_irreps=MLP_irreps,
        atomic_energies=torch.zeros(num_elements, dtype=dtype, device=device),
        avg_num_neighbors=3.0,
        atomic_numbers=atomic_numbers,
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
        include_dipole_mm_interaction=include_dipole_mm_interaction,
        add_local_electron_energy=True,
        field_dependence_type="AgnosticEmbeddedOneBodyVariableUpdate",
        final_field_readout_type="OneBodyMLPFieldReadout",
        return_electrostatic_potentials=False,
        heads=["Default"],
        field_norm_factor=1.0,
        fixedpoint_update_config=fixedpoint_update_config,
        field_readout_config=field_readout_config,
    ).to(device=device, dtype=dtype)


def _build_mm_batch(device, dtype, n_mm=8, seed=0):
    """A batched water-like 2-atom QM system + n_mm MM point charges.

    Layout matches the schema PolarMACE.forward expects: mm_positions,
    mm_charges, and mm_source_batch are explicit top-level keys.
    """
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

    # MM cloud placed in a 4-8 Å shell around the origin so it sits clearly
    # outside the 2-atom QM region.
    radii = 4.0 + 4.0 * rng.random(n_mm)
    directions = rng.standard_normal((n_mm, 3))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    mm_pos = directions * radii[:, None]
    mm_q = rng.uniform(-0.834, 0.417, size=n_mm)
    mm_positions = torch.tensor(mm_pos, device=device, dtype=dtype)
    mm_charges = torch.tensor(mm_q, device=device, dtype=dtype)
    mm_source_batch = torch.zeros(n_mm, dtype=torch.long, device=device)

    return {
        "positions": positions,
        "edge_index": edge_index,
        "shifts": shifts,
        "node_attrs": node_attrs,
        "batch": batch,
        "ptr": ptr,
        "cell": cell.view(-1, 9),
        "rcell": rcell.view(-1, 9),
        "volume": torch.ones((1,), dtype=dtype, device=device),
        "pbc": torch.zeros((1, 3), dtype=torch.bool, device=device),
        "external_field": torch.zeros((1, 3), dtype=dtype, device=device),
        "fermi_level": torch.zeros((1,), dtype=dtype, device=device),
        "total_charge": torch.zeros((1,), dtype=dtype, device=device),
        "total_spin": torch.zeros((1,), dtype=dtype, device=device),
        "density_coefficients": torch.zeros((n, 1), dtype=dtype, device=device),
        "mm_positions": mm_positions,
        "mm_charges": mm_charges,
        "mm_source_batch": mm_source_batch,
    }


def _run_route(model, data, route: str) -> dict:
    model.set_mm_field_route(route)
    out = model(data, training=False, compute_force=True)
    energy = out["energy"].detach().clone()
    forces = out["forces"].detach().clone()
    ml_mm_electrostatic_energy = out["ml_mm_electrostatic_energy"].detach().clone()
    ml_mm_dipole_energy = out["ml_mm_dipole_energy"].detach().clone()
    mm_forces = out["mm_forces"]
    assert mm_forces is not None, "expected mm_forces in output when MM input present"
    mm_forces = mm_forces.detach().clone()
    charges = out["charges"].detach().clone()
    density_coefficients = out["density_coefficients"].detach().clone()
    return {
        "energy": energy,
        "forces": forces,
        "mm_forces": mm_forces,
        "ml_mm_electrostatic_energy": ml_mm_electrostatic_energy,
        "ml_mm_dipole_energy": ml_mm_dipole_energy,
        "charges": charges,
        "density_coefficients": density_coefficients,
    }


def _direct_ml_mm_coulomb(charges, data) -> torch.Tensor:
    k_e = 14.3996454784255
    positions = data["positions"]
    mm_positions = data["mm_positions"]
    mm_charges = data["mm_charges"]
    batch = data["batch"]
    mm_batch = data["mm_source_batch"]

    rij = positions[:, None, :] - mm_positions[None, :, :]
    r = torch.linalg.norm(rij, dim=-1).clamp_min(1e-6)
    e_pair = k_e * charges[:, None] * mm_charges[None, :] / r
    e_pair = torch.where(batch[:, None] == mm_batch[None, :], e_pair, 0.0)
    e_atom = e_pair.sum(dim=1)
    return torch.zeros(
        int(batch.max().item()) + 1, dtype=positions.dtype, device=positions.device
    ).scatter_add_(0, batch, e_atom)


def _direct_ml_mm_dipole_energy(density_coefficients, data) -> torch.Tensor:
    k_e = 14.3996454784255
    positions = data["positions"]
    mm_positions = data["mm_positions"]
    mm_charges = data["mm_charges"]
    batch = data["batch"]
    mm_batch = data["mm_source_batch"]
    dipoles = density_coefficients[:, 1:4]

    rij = positions[:, None, :] - mm_positions[None, :, :]
    r = torch.linalg.norm(rij, dim=-1).clamp_min(1e-6)
    dipole_dot_r = torch.einsum("ik,ijk->ij", dipoles, rij)
    e_pair = -k_e * dipole_dot_r * mm_charges[None, :] / (r * r * r)
    e_pair = torch.where(batch[:, None] == mm_batch[None, :], e_pair, 0.0)
    e_atom = e_pair.sum(dim=1)
    return torch.zeros(
        int(batch.max().item()) + 1, dtype=positions.dtype, device=positions.device
    ).scatter_add_(0, batch, e_atom)


def test_default_route_is_source_target():
    """A freshly constructed PolarMACE picks the source-target path by default."""
    model = _build_minimal_model(torch.device("cpu"), torch.float64)
    assert model.mm_field_route == "source_target"


def test_set_mm_field_route_validates():
    model = _build_minimal_model(torch.device("cpu"), torch.float64)
    with pytest.raises(ValueError, match="mm_field_route"):
        model.set_mm_field_route("not_a_route")


@pytest.mark.parametrize(
    "dtype, atol",
    [
        (torch.float64, 1e-9),
        (torch.float32, 1e-3),
    ],
)
def test_mm_field_routes_energy_force_equivalence(dtype, atol):
    """Energy, QM forces, and MM forces must match between routes.

    Tolerances are tight in float64 (round-off only) and looser in float32 to
    accommodate order-of-summation drift between the two algorithms.
    """
    device = torch.device("cpu")
    torch.manual_seed(0)
    model = _build_minimal_model(device, dtype)
    data = _build_mm_batch(device, dtype)

    out_st = _run_route(model, data, "source_target")
    out_lg = _run_route(model, data, "legacy_concat")

    torch.testing.assert_close(out_st["energy"], out_lg["energy"], atol=atol, rtol=atol)
    torch.testing.assert_close(out_st["forces"], out_lg["forces"], atol=atol, rtol=atol)
    torch.testing.assert_close(
        out_st["mm_forces"], out_lg["mm_forces"], atol=atol, rtol=atol
    )
    torch.testing.assert_close(
        out_st["ml_mm_electrostatic_energy"],
        out_lg["ml_mm_electrostatic_energy"],
        atol=atol,
        rtol=atol,
    )


def test_mm_forces_nontrivial_so_test_is_meaningful():
    """Sanity check: the MM forces in this fixture are not all zero, so the
    equivalence test above isn't passing trivially on a degenerate case.
    """
    device = torch.device("cpu")
    model = _build_minimal_model(device, torch.float64)
    data = _build_mm_batch(device, torch.float64)
    out = _run_route(model, data, "source_target")
    assert out["mm_forces"].abs().max().item() > 1e-12


def test_ml_mm_electrostatic_energy_matches_direct_pair_sum():
    device = torch.device("cpu")
    model = _build_minimal_model(device, torch.float64)
    data = _build_mm_batch(device, torch.float64)
    out = _run_route(model, data, "source_target")
    expected = _direct_ml_mm_coulomb(out["charges"], data)
    torch.testing.assert_close(
        out["ml_mm_electrostatic_energy"], expected, atol=1e-9, rtol=1e-9
    )


def test_ml_mm_dipole_energy_matches_direct_pair_sum():
    device = torch.device("cpu")
    model = _build_minimal_model(
        device, torch.float64, include_dipole_mm_interaction=True
    )
    data = _build_mm_batch(device, torch.float64)
    out = _run_route(model, data, "source_target")
    expected_dipole = _direct_ml_mm_dipole_energy(out["density_coefficients"], data)
    expected_total = _direct_ml_mm_coulomb(out["charges"], data) + expected_dipole
    assert out["ml_mm_dipole_energy"].abs().max().item() > 1e-12
    torch.testing.assert_close(
        out["ml_mm_dipole_energy"], expected_dipole, atol=1e-9, rtol=1e-9
    )
    torch.testing.assert_close(
        out["ml_mm_electrostatic_energy"], expected_total, atol=1e-9, rtol=1e-9
    )
