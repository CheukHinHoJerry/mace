#!/usr/bin/env python3
"""Generic init/forward smoke test for magnetic non-SOC MACE models."""

from __future__ import annotations

import argparse
from typing import Any

import numpy as np
import torch
from e3nn import o3

from mace import data, modules, tools
from mace.tools import torch_geometric


DEFAULT_MODEL = "MagneticSolidHarmonicsNonSpinOrbitCoupledWithOneBodyMultiSpeciesGinzburgSelfMagmomScaleShiftMACE"
DEFAULT_INTERACTION_FIRST = "RealAgnosticDensityInteractionBlock"
DEFAULT_INTERACTION = "MagneticRealAgnosticNonSpinOrbitCoupledDensityInteractionBlock"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--interaction_first",
        "--interaction-cls-first",
        dest="interaction_first",
        default=DEFAULT_INTERACTION_FIRST,
    )
    parser.add_argument(
        "--interaction",
        "--interaction-cls",
        dest="interaction",
        default=DEFAULT_INTERACTION,
    )
    parser.add_argument("--hidden-irreps", default="32x0e + 32x1o")
    parser.add_argument("--mlp-irreps", default="16x0e")
    parser.add_argument(
        "--contraction_cls_first",
        "--contraction-cls-first",
        dest="contraction_cls_first",
        default="SymmetricContraction",
    )
    parser.add_argument(
        "--contraction_cls",
        "--contraction-cls",
        dest="contraction_cls",
        default="NonSOCSymmetricContraction",
    )
    parser.add_argument("--num-interactions", type=int, default=2)
    parser.add_argument("--correlation", type=int, default=3)
    parser.add_argument("--r-max", type=float, default=4.5)
    parser.add_argument("--num-bessel", type=int, default=10)
    parser.add_argument("--num-polynomial-cutoff", type=int, default=5)
    parser.add_argument("--max-ell", type=int, default=3)
    parser.add_argument("--avg-num-neighbors", type=float, default=28.243721185921906)
    parser.add_argument("--num-elements", type=int, default=2)
    parser.add_argument("--atomic-numbers", default="1,8")
    parser.add_argument("--atomic-energies", default="0.0,0.0")
    parser.add_argument("--atomic-inter-scale", default="1.0")
    parser.add_argument("--atomic-inter-shift", default="0.0")
    parser.add_argument("--m-max", default="10.0,10.0")
    parser.add_argument("--num-mag-radial-basis", type=int, default=8)
    parser.add_argument("--num-mag-radial-basis-one-body", type=int, default=10)
    parser.add_argument("--max-m-ell", type=int, default=3)
    parser.add_argument("--distance-transform", default="Agnesi")
    parser.add_argument("--cutoff", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def parse_float_list(text: str) -> list[float]:
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def parse_int_list(text: str) -> list[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def build_batch(cutoff: float):
    table = tools.AtomicNumberTable([1, 8])
    config = data.Configuration(
        atomic_numbers=np.array([8, 1, 1]),
        positions=np.array(
            [
                [0.0, -1.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
            ]
        ),
        forces=np.zeros((3, 3), dtype=float),
        energy=-1.5,
        charges=np.array([-2.0, 1.0, 1.0]),
        dipole=np.array([-1.5, 1.5, 2.0]),
        magmom=np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, -2.0, 0.0],
                [0.0, 0.0, -3.0],
            ]
        ),
    )
    atom_data = data.AtomicData.from_config(config, z_table=table, cutoff=cutoff)
    loader = torch_geometric.dataloader.DataLoader([atom_data], batch_size=1)
    return table, next(iter(loader))


def build_model(args: argparse.Namespace, table: tools.AtomicNumberTable) -> torch.nn.Module:
    model_cls = getattr(modules, args.model)
    interaction_cls_first = modules.interaction_classes[args.interaction_first]
    interaction_cls = modules.interaction_classes[args.interaction]
    atomic_numbers = parse_int_list(args.atomic_numbers)
    atomic_energies = np.array(parse_float_list(args.atomic_energies), dtype=float)
    atomic_inter_scale = parse_float_list(args.atomic_inter_scale)
    atomic_inter_shift = parse_float_list(args.atomic_inter_shift)
    m_max = parse_float_list(args.m_max)

    model_config: dict[str, Any] = dict(
        r_max=args.r_max,
        num_bessel=args.num_bessel,
        num_polynomial_cutoff=args.num_polynomial_cutoff,
        max_ell=args.max_ell,
        interaction_cls_first=interaction_cls_first,
        interaction_cls=interaction_cls,
        num_interactions=args.num_interactions,
        num_elements=args.num_elements,
        hidden_irreps=o3.Irreps(args.hidden_irreps),
        MLP_irreps=o3.Irreps(args.mlp_irreps),
        gate=torch.nn.functional.silu,
        atomic_energies=atomic_energies,
        avg_num_neighbors=args.avg_num_neighbors,
        atomic_numbers=atomic_numbers if atomic_numbers else table.zs,
        correlation=args.correlation,
        radial_type="bessel",
        contraction_cls_first=args.contraction_cls_first,
        contraction_cls=args.contraction_cls,
        atomic_inter_scale=atomic_inter_scale,
        atomic_inter_shift=atomic_inter_shift,
        m_max=m_max,
        num_mag_radial_basis=args.num_mag_radial_basis,
        max_m_ell=args.max_m_ell,
        num_mag_radial_basis_one_body=args.num_mag_radial_basis_one_body,
        heads=["default"],
        distance_transform=args.distance_transform,
    )
    return model_cls(**model_config)


def main() -> None:
    args = parse_args()

    torch.set_default_dtype(torch.float64)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    table, batch = build_batch(args.cutoff)
    model = build_model(args, table).to(args.device)
    batch_dict = {
        key: value.to(args.device) if hasattr(value, "to") else value
        for key, value in batch.to_dict().items()
    }

    outputs = model(
        batch_dict,
        training=False,
        compute_force=True,
        compute_stress=True,
        compute_virials=True,
        compute_magforces=True,
    )

    print("initialized=True")
    print(f"model={args.model}")
    print(f"interaction_first={args.interaction_first}")
    print(f"interaction={args.interaction}")
    print(f"hidden_irreps={args.hidden_irreps}")
    print(f"contraction_cls_first={args.contraction_cls_first}")
    print(f"contraction_cls={args.contraction_cls}")

    nonsoc_modules = [
        (name, mod)
        for name, mod in model.named_modules()
        if mod.__class__.__name__ == "NonSOCContraction"
    ]
    print(f"n_nonsoc_contractions={len(nonsoc_modules)}")
    for name, mod in nonsoc_modules:
        print(f"{name} irrep_out={mod.irrep_out} correlation={mod.correlation}")

    for key in ("energy", "forces", "stress", "virials", "magforces", "node_feats"):
        if key in outputs:
            value = outputs[key]
            shape = tuple(value.shape) if hasattr(value, "shape") else type(value)
            print(f"{key}_shape={shape}")

    if "energy" in outputs:
        print(f"energy0={float(outputs['energy'][0])}")


if __name__ == "__main__":
    main()
