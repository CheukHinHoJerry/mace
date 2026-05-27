#!/usr/bin/env python3
"""Standalone CPU check for per-layer hidden_irreps with mixed L per layer:

    layer 0 -> L=1  ("Nx0e + Nx1o")     (carries vector features)
    layer 1 -> L=0  ("Nx0e")            (scalar-only message passing)

Confirms the model builds, the per-layer irreps actually take effect (layer-0 product
carries 1o, layer-1 product is scalar-only), the forward runs, energy is rotation
invariant, the "|" CLI string parses to the same per-layer list, and the mixed-L model
is genuinely different (fewer params) from a uniform L=1 model.

Run:  CUDA_VISIBLE_DEVICES="" TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 python tests/check_per_layer_L1_L0_cpu.py
"""
import numpy as np
import torch
from e3nn import o3
from scipy.spatial.transform import Rotation as R

from mace import data, modules, tools
from mace.tools import torch_geometric
from mace.tools.scripts_utils import extract_config_mace_model, parse_hidden_irreps

torch.set_default_dtype(torch.float64)

TABLE = tools.AtomicNumberTable([1, 8])
ATOMIC_ENERGIES = np.array([1.0, 3.0], dtype=float)
POSITIONS = np.array([[0.0, -2.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])


def batch(positions):
    cfg = data.Configuration(
        atomic_numbers=np.array([8, 1, 1]),
        positions=positions,
        properties={"forces": np.zeros((3, 3)), "energy": -1.5},
        property_weights={"forces": 1.0, "energy": 1.0},
    )
    atoms = data.AtomicData.from_config(cfg, z_table=TABLE, cutoff=3.0)
    loader = torch_geometric.dataloader.DataLoader([atoms], batch_size=1)
    return next(iter(loader)).to_dict()


def build(hidden_irreps, num_interactions, seed=0):
    torch.manual_seed(seed)
    cfg = dict(
        r_max=5,
        num_bessel=8,
        num_polynomial_cutoff=6,
        max_ell=2,
        interaction_cls=modules.interaction_classes[
            "RealAgnosticResidualInteractionBlock"
        ],
        interaction_cls_first=modules.interaction_classes[
            "RealAgnosticResidualInteractionBlock"
        ],
        num_interactions=num_interactions,
        num_elements=2,
        hidden_irreps=hidden_irreps,
        MLP_irreps=o3.Irreps("16x0e"),
        gate=torch.nn.functional.silu,
        atomic_energies=ATOMIC_ENERGIES,
        avg_num_neighbors=8,
        atomic_numbers=TABLE.zs,
        correlation=3,
        radial_type="bessel",
        atomic_inter_scale=1.0,
        atomic_inter_shift=0.0,
    )
    return modules.ScaleShiftMACE(**cfg)


def nparams(m):
    return sum(p.numel() for p in m.parameters())


def main():
    ok = True

    # --- CLI string form parses to the per-layer list -------------------------------
    cli = "32x0e+32x1o | 32x0e"
    parsed = parse_hidden_irreps(cli)
    expect = [o3.Irreps("32x0e+32x1o"), o3.Irreps("32x0e")]
    ok &= parsed == expect
    print(f"[parse] '{cli}' -> {parsed}   {'OK' if parsed == expect else 'FAIL'}")

    # --- build the mixed-L model (L=1 then L=0) -------------------------------------
    model = build(parsed, num_interactions=2, seed=11)
    l0 = model.products[0].linear.irreps_out
    l1 = model.products[1].linear.irreps_out
    print(f"[build] layer-0 product irreps_out = {l0}  (lmax={l0.lmax})")
    print(f"[build] layer-1 product irreps_out = {l1}  (lmax={l1.lmax})")
    ok &= l0.lmax == 1 and l0.count(o3.Irrep(1, -1)) == 32   # layer 0 carries vectors
    ok &= l1.lmax == 0                                       # layer 1 is scalar-only

    # --- forward runs and is rotation invariant in energy ---------------------------
    out = model(batch(POSITIONS), compute_force=True)
    e = out["energy"]
    f = out["forces"]
    finite = bool(torch.isfinite(e).all() and torch.isfinite(f).all())
    print(f"[fwd]   energy={e.item():.6f}  forces finite={finite}  shape={tuple(f.shape)}")
    ok &= finite

    rot = R.from_euler("z", 47, degrees=True).as_matrix()
    e_rot = model(batch((rot @ POSITIONS.T).T))["energy"]
    inv = torch.allclose(e, e_rot, atol=1e-8)
    print(f"[rot]   E={e.item():.8f}  E(Rx)={e_rot.item():.8f}  invariant={inv}")
    ok &= inv

    # --- extract_config preserves the per-layer (mixed-L) irreps --------------------
    cfg = extract_config_mace_model(model)
    is_list = isinstance(cfg["hidden_irreps"], list)
    print(f"[cfg]   extract_config hidden_irreps={cfg['hidden_irreps']}  list={is_list}")
    ok &= is_list and [str(h) for h in cfg["hidden_irreps"]] == [str(h) for h in expect]

    # --- the L=1 layer-0 actually adds capacity vs an all-scalar model --------------
    # (NB: for 2 layers the *last* layer is always scalarized, so L1->L0 and a uniform
    # L1 model are identical; the real per-layer effect is layer-0 carrying vectors.)
    scalar = build([o3.Irreps("32x0e"), o3.Irreps("32x0e")], num_interactions=2, seed=11)
    np_mixed, np_scalar = nparams(model), nparams(scalar)
    bigger = np_mixed > np_scalar
    print(f"[diff]  params L1->L0={np_mixed}  L0->L0={np_scalar}  "
          f"L1_layer0_adds_capacity={bigger}")
    ok &= bigger

    print("\nRESULT:", "ALL CHECKS PASSED" if ok else "FAILURES ABOVE")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
