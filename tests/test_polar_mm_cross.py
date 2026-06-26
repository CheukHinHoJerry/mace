"""Validate PolarMACE's direct ML×MM cross-Coulomb path vs the
concat-then-subtract reference.

The new helpers (`_assemble_rho_chunked`, `_pbc_mm_cross_energy`) are
exercised against the original `mixed_E - mm_E - ml_E` identity that
PolarMACE used to evaluate on the combined ML+MM source list. Chunked vs
unchunked ρ_MM assembly is also verified to be invariant (since
``assemble_fourier_series_batch`` is bilinear in the source coefficients,
per-chunk contributions sum exactly).
"""

from __future__ import annotations

import math

import pytest
import torch

graph_longrange = pytest.importorskip("graph_longrange")
from graph_longrange.energy import GTOElectrostaticEnergy  # noqa: E402
from graph_longrange.kspace import compute_k_vectors_flat  # noqa: E402

from mace.modules.extensions import PolarMACE  # noqa: E402

pytestmark = pytest.mark.filterwarnings("ignore::UserWarning")


def _mm_charge_features(q: torch.Tensor, m_dim: int) -> torch.Tensor:
    out = torch.zeros((q.numel(), m_dim), dtype=q.dtype, device=q.device)
    out[:, 0] = q
    return out


def _make_stub(chunk_size: int) -> object:
    """Minimal object that satisfies the attribute surface of
    ``_assemble_rho_chunked`` and ``_pbc_mm_cross_energy``.

    We bypass full PolarMACE construction because both methods only access
    ``self.coulomb_energy`` and ``self.mm_chunk_size``."""
    stub = type("Stub", (), {})()
    stub.coulomb_energy = GTOElectrostaticEnergy(
        density_max_l=1,
        density_smearing_width=0.5,
        kspace_cutoff=3.0,
        include_self_interaction=False,
        include_pbc_corrections=True,
    )
    stub.mm_chunk_size = chunk_size
    stub._assemble_rho_chunked = PolarMACE._assemble_rho_chunked.__get__(stub)
    stub._pbc_mm_cross_energy = PolarMACE._pbc_mm_cross_energy.__get__(stub)
    stub._pad_to_num_graphs = PolarMACE._pad_to_num_graphs  # staticmethod
    return stub


def _reference_cross(
    stub,
    ml_f, ml_r, ml_b,
    mm_f, mm_r, mm_b,
    k_vectors, k_norm2, k_vector_batch, k0_mask, volume, pbc,
):
    mixed_f = torch.cat([ml_f, mm_f], dim=0)
    mixed_r = torch.cat([ml_r, mm_r], dim=0)
    mixed_b = torch.cat([ml_b, mm_b], dim=0)
    mixed_E = stub.coulomb_energy(
        k_vectors=k_vectors, k_norm2=k_norm2, k_vector_batch=k_vector_batch,
        k0_mask=k0_mask, source_feats=mixed_f, node_positions=mixed_r,
        batch=mixed_b, volume=volume, pbc=pbc, force_pbc_evaluator=True,
    )
    mm_E = stub.coulomb_energy(
        k_vectors=k_vectors, k_norm2=k_norm2, k_vector_batch=k_vector_batch,
        k0_mask=k0_mask, source_feats=mm_f, node_positions=mm_r,
        batch=mm_b, volume=volume, pbc=pbc, force_pbc_evaluator=True,
    )
    ml_E = stub.coulomb_energy(
        k_vectors=k_vectors, k_norm2=k_norm2, k_vector_batch=k_vector_batch,
        k0_mask=k0_mask, source_feats=ml_f, node_positions=ml_r,
        batch=ml_b, volume=volume, pbc=pbc, force_pbc_evaluator=True,
    )
    return (mixed_E - mm_E) - ml_E


def _build_inputs(n_mm: int, multipoles: bool, seed: int):
    torch.manual_seed(seed)
    nml, m = 4, 4
    cell = torch.eye(3, dtype=torch.float64).unsqueeze(0) * 10.0
    rcell = 2 * math.pi * torch.linalg.inv(cell).transpose(-1, -2)
    volume = torch.det(cell)
    kv, k2, kb, k0 = compute_k_vectors_flat(3.0, cell, rcell)

    ml_r = (torch.rand(nml, 3, dtype=torch.float64) * 8 + 1).requires_grad_(True)
    mm_r = (torch.rand(n_mm, 3, dtype=torch.float64) * 8 + 1).requires_grad_(True)
    ml_f = torch.randn(nml, m, dtype=torch.float64).requires_grad_(True)
    ml_b = torch.zeros(nml, dtype=torch.long)
    mm_b = torch.zeros(n_mm, dtype=torch.long)
    if multipoles:
        mm_f = torch.randn(n_mm, m, dtype=torch.float64).requires_grad_(True)
    else:
        q = torch.randn(n_mm, dtype=torch.float64, requires_grad=True)
        mm_f = _mm_charge_features(q, m)
    return dict(
        ml_r=ml_r, mm_r=mm_r, ml_f=ml_f, mm_f=mm_f, ml_b=ml_b, mm_b=mm_b,
        kv=kv, k2=k2, kb=kb, k0=k0, volume=volume,
    )


@pytest.mark.parametrize(
    "pbc, multipoles, n_mm, chunk_size",
    [
        (torch.tensor([[True, True, True]]),  False, 10,   4096),
        (torch.tensor([[True, True, True]]),  True,  10,   4096),
        (torch.tensor([[True, True, False]]), False, 10,   4096),  # slab
        (torch.tensor([[True, True, True]]),  False, 128,  4096),  # no chunking (1 chunk)
        (torch.tensor([[True, True, True]]),  False, 128,  32),    # 4 chunks
        (torch.tensor([[True, True, True]]),  False, 128,  1),     # per-node
        (torch.tensor([[True, True, True]]),  True,  128,  37),    # uneven chunks
    ],
    ids=[
        "full_pbc_charges_n10_chunk4096",
        "full_pbc_multipoles_n10_chunk4096",
        "slab_pbc_charges_n10_chunk4096",
        "full_pbc_charges_n128_nochunk",
        "full_pbc_charges_n128_chunk32",
        "full_pbc_charges_n128_chunk1",
        "full_pbc_multipoles_n128_chunk37_uneven",
    ],
)
def test_mm_cross_matches_concat_minus_mm(pbc, multipoles, n_mm, chunk_size):
    torch.set_default_dtype(torch.float64)
    x = _build_inputs(n_mm=n_mm, multipoles=multipoles, seed=11)
    stub = _make_stub(chunk_size=chunk_size)

    cross_k, cross_corr = stub._pbc_mm_cross_energy(
        ml_features=x["ml_f"], ml_positions=x["ml_r"], ml_batch=x["ml_b"],
        mm_features=x["mm_f"], mm_positions=x["mm_r"], mm_batch=x["mm_b"],
        k_vectors=x["kv"], k_norm2=x["k2"], k_vector_batch=x["kb"],
        k0_mask=x["k0"], volume=x["volume"], pbc=pbc,
        mm_chunk_size=chunk_size,
    )
    cross_new = cross_k + cross_corr
    cross_ref = _reference_cross(
        stub,
        x["ml_f"], x["ml_r"], x["ml_b"],
        x["mm_f"], x["mm_r"], x["mm_b"],
        x["kv"], x["k2"], x["kb"], x["k0"], x["volume"], pbc,
    )

    g_new = torch.autograd.grad(
        cross_new.sum(), [x["ml_r"], x["mm_r"], x["ml_f"]],
        retain_graph=True, allow_unused=True,
    )
    g_ref = torch.autograd.grad(
        cross_ref.sum(), [x["ml_r"], x["mm_r"], x["ml_f"]],
        retain_graph=True, allow_unused=True,
    )

    ed = (cross_new - cross_ref).abs().max().item()
    gd = max(
        (a - b).abs().max().item()
        for a, b in zip(g_new, g_ref)
        if a is not None and b is not None
    )
    assert ed < 1e-9, f"energy mismatch: {ed:.3e}"
    assert gd < 1e-8, f"gradient mismatch: {gd:.3e}"


def test_mm_cross_handles_trailing_graph_without_mm():
    """Batched case where the trailing graph has ML but no MM nodes.

    `MonopoleDipoleCorrectionBlock.forward()` scatters without `dim_size`,
    so the raw `corr(MM)` tensor would be shorter than `volume`; the cross
    subtraction has to pad to `num_graphs` for the algebra to stay exact.
    """
    torch.set_default_dtype(torch.float64)
    torch.manual_seed(17)
    chunk_size = 4096
    nml_per, n_mm_g0, m = 3, 8, 4
    num_graphs = 2
    cell = (torch.eye(3, dtype=torch.float64).unsqueeze(0) * 10.0).repeat(
        num_graphs, 1, 1
    )
    rcell = 2 * math.pi * torch.linalg.inv(cell).transpose(-1, -2)
    volume = torch.det(cell)
    kv, k2, kb, k0 = compute_k_vectors_flat(3.0, cell, rcell)

    # ML in both graphs
    ml_r = (torch.rand(nml_per * num_graphs, 3, dtype=torch.float64) * 8 + 1).requires_grad_(True)
    ml_f = torch.randn(nml_per * num_graphs, m, dtype=torch.float64).requires_grad_(True)
    ml_b = torch.cat([torch.full((nml_per,), g, dtype=torch.long) for g in range(num_graphs)])

    # MM only in graph 0
    mm_r = (torch.rand(n_mm_g0, 3, dtype=torch.float64) * 8 + 1).requires_grad_(True)
    q = torch.randn(n_mm_g0, dtype=torch.float64, requires_grad=True)
    mm_f = _mm_charge_features(q, m)
    mm_b = torch.zeros(n_mm_g0, dtype=torch.long)

    pbc = torch.tensor([[True, True, True]], dtype=torch.bool).repeat(num_graphs, 1)
    stub = _make_stub(chunk_size=chunk_size)

    cross_k, cross_corr = stub._pbc_mm_cross_energy(
        ml_features=ml_f, ml_positions=ml_r, ml_batch=ml_b,
        mm_features=mm_f, mm_positions=mm_r, mm_batch=mm_b,
        k_vectors=kv, k_norm2=k2, k_vector_batch=kb, k0_mask=k0,
        volume=volume, pbc=pbc, mm_chunk_size=chunk_size,
    )
    cross_new = cross_k + cross_corr

    cross_ref = _reference_cross(
        stub, ml_f, ml_r, ml_b, mm_f, mm_r, mm_b,
        kv, k2, kb, k0, volume, pbc,
    )

    assert cross_new.shape == volume.shape
    assert cross_ref.shape == volume.shape
    # Graph 1 has no MM nodes; the cross must be exactly zero there.
    assert cross_new[1].abs().item() < 1e-12
    assert cross_ref[1].abs().item() < 1e-12

    ed = (cross_new - cross_ref).abs().max().item()
    assert ed < 1e-9, f"energy mismatch: {ed:.3e}"
