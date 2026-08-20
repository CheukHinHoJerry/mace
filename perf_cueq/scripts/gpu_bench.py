"""GPU benchmark: fixed non-SOC model vs the original architecture (centre-spin off).

Production-ish truncation: 128x0e, correlation 3, max_ell 3, max_m_ell 2, 89 elements.
Also sweeps chunk_size, which is the open memory item.
"""

import statistics
import sys
import time

import numpy as np
import torch

torch.serialization.add_safe_globals([slice])
sys.path.insert(0, "/home/jerry528/mace-ace-pr124/mace")
from e3nn import o3

from mace.modules import interaction_classes as IC
from mace.modules.extensions import MagneticNonSOCScaleShiftMACE
from mace.modules.symmetric_contraction_nonsoc import NonSOCContraction
from mace.tools.torch_tools import default_dtype

DEV = "cuda"
RMAX = 5.0


def bulk_bcc(n):
    a = 2.87
    cells = []
    r = int(np.ceil((n / 2) ** (1 / 3)))
    for i in range(r):
        for j in range(r):
            for k in range(r):
                cells += [
                    [i * a, j * a, k * a],
                    [(i + 0.5) * a, (j + 0.5) * a, (k + 0.5) * a],
                ]
    return np.array(cells[:n])


def make_batch(pos, nelem):
    n = len(pos)
    d = np.linalg.norm(pos[:, None, :] - pos[None, :, :], axis=-1)
    src, dst = np.where((d < RMAX) & (d > 1e-8))
    ei = torch.tensor(np.stack([src, dst]), dtype=torch.long, device=DEV)
    na = torch.zeros(n, nelem, device=DEV, dtype=torch.get_default_dtype())
    na[:, 0] = 1.0
    return (
        {
            "positions": torch.tensor(
                pos, device=DEV, dtype=torch.get_default_dtype(), requires_grad=True
            ),
            "cell": torch.zeros(1, 3, 3, device=DEV, dtype=torch.get_default_dtype()),
            "batch": torch.zeros(n, dtype=torch.long, device=DEV),
            "ptr": torch.tensor([0, n], device=DEV),
            "node_attrs": na,
            "edge_index": ei,
            "magmom": torch.randn(n, 3, device=DEV, dtype=torch.get_default_dtype())
            * 0.8,
            "unit_shifts": torch.zeros(
                ei.shape[1], 3, device=DEV, dtype=torch.get_default_dtype()
            ),
            "shifts": torch.zeros(
                ei.shape[1], 3, device=DEV, dtype=torch.get_default_dtype()
            ),
        },
        n,
        ei.shape[1],
    )


def build(nelem, corr=3, max_ell=3, max_m_ell=2, hidden="128x0e", chunk=250):
    torch.manual_seed(1)
    m = MagneticNonSOCScaleShiftMACE(
        r_max=RMAX,
        num_bessel=8,
        num_polynomial_cutoff=5,
        max_ell=max_ell,
        interaction_cls=IC[
            "MagneticRealAgnosticNonSpinOrbitCoupledDensityInteractionBlock"
        ],
        interaction_cls_first=IC["RealAgnosticDensityInteractionBlock"],
        contraction_cls_first="SymmetricContraction",
        contraction_cls="NonSOCSymmetricContraction",
        num_interactions=2,
        num_elements=nelem,
        hidden_irreps=o3.Irreps(hidden),
        MLP_irreps=o3.Irreps("16x0e"),
        atomic_energies=np.zeros(nelem),
        avg_num_neighbors=14.0,
        atomic_numbers=list(range(1, nelem + 1)),
        correlation=corr,
        gate=torch.nn.functional.silu,
        atomic_inter_shift=[0.0],
        atomic_inter_scale=[1.0],
        m_max=[3.0] * nelem,
        num_mag_radial_basis=8,
        num_mag_radial_basis_one_body=8,
        max_m_ell=max_m_ell,
        use_magmom_one_body=False,
        heads=["Default"],
    ).to(DEV)
    for mod in m.modules():
        if isinstance(mod, NonSOCContraction):
            mod.chunk_size = chunk
    return m


def run(model, batch, nrep=4):
    def once(bwd):
        out = model(batch, training=bwd, compute_force=False, compute_stress=False)
        e = out["energy"].sum()
        if bwd:
            e.backward()
        return e

    for _ in range(2):
        model.zero_grad(set_to_none=True)
        once(True)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    fs, bs = [], []
    for _ in range(nrep):
        model.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        t = time.perf_counter()
        with torch.no_grad():
            once(False)
        torch.cuda.synchronize()
        fs.append((time.perf_counter() - t) * 1e3)
        torch.cuda.synchronize()
        t = time.perf_counter()
        once(True)
        torch.cuda.synchronize()
        bs.append((time.perf_counter() - t) * 1e3)
    return (
        statistics.median(fs),
        statistics.median(bs),
        torch.cuda.max_memory_allocated() / 1e9,
    )


with default_dtype(torch.float64):
    NELEM = 89
    print(f"H100, float64, 128x0e, corr=3, max_ell=3, max_m_ell=2, {NELEM} elements\n")
    for natoms in (128, 432):
        pos = bulk_bcc(natoms)
        batch, n, ne = make_batch(pos, NELEM)
        print(f"--- {n} atoms, {ne} edges ---")
        print(
            f"{'variant':<28} {'fwd ms':>9} {'bwd ms':>9} {'peak GB':>9} {'params':>12}"
        )
        for label, centre in (
            ("ORIGINAL (centre off)", False),
            ("FIXED (centre on)", True),
        ):
            m = build(NELEM)
            if not centre:
                for mod in m.modules():
                    if isinstance(mod, NonSOCContraction):
                        mod.center_spin_coupling = False
            try:
                f, b, mem = run(m, batch)
                p = sum(q.numel() for q in m.parameters())
                print(f"{label:<28} {f:9.1f} {b:9.1f} {mem:9.2f} {p:12,}")
            except torch.cuda.OutOfMemoryError:
                print(f"{label:<28} {'OOM':>9}")
            del m
            torch.cuda.empty_cache()
        print()
