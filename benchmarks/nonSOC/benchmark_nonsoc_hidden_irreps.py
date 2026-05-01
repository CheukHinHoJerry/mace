#!/usr/bin/env python3
"""Benchmark non-SOC MACE model forward time for L=0 and L=1 across k values."""

from __future__ import annotations

import argparse
import statistics
import time
import warnings
from contextlib import contextmanager
from typing import Any, Iterator

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
    parser.add_argument("--k-values", default="16,32,64,128")
    parser.add_argument("--l-values", default="0,1")
    parser.add_argument("--modes", default="reference,optimized,sparse")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeat", type=int, default=50)
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


def hidden_irreps_for(k: int, ell_mode: int) -> str:
    if ell_mode == 0:
        return f"{k}x0e"
    if ell_mode == 1:
        return f"{k}x0e + {k}x1o"
    raise ValueError(f"Unsupported L value: {ell_mode}")


def build_model(args: argparse.Namespace, table: tools.AtomicNumberTable, hidden_irreps: str) -> torch.nn.Module:
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
        hidden_irreps=o3.Irreps(hidden_irreps),
        MLP_irreps=o3.Irreps("16x0e"),
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


@contextmanager
def contraction_mode(model: torch.nn.Module, mode: str) -> Iterator[None]:
    touched = []
    for _, mod in model.named_modules():
        if mod.__class__.__name__ != "NonSOCContraction":
            continue
        if not hasattr(mod, "forward_reference"):
            continue
        touched.append((mod, mod.forward))
        if mode == "reference":
            mod.forward = mod.forward_reference
        elif mode == "optimized":
            mod.forward = mod.forward_optimized
        elif mode == "sparse":
            mod.forward = mod.forward_sparse
        else:
            raise ValueError(f"Unknown mode: {mode}")
    try:
        yield
    finally:
        for mod, old_forward in touched:
            mod.forward = old_forward


def synchronize_if_needed(device: str) -> None:
    if device.startswith("cuda"):
        torch.cuda.synchronize()


def benchmark(fn, warmup: int, repeat: int, device: str) -> tuple[float, float]:
    for _ in range(warmup):
        fn()
    synchronize_if_needed(device)

    if device.startswith("cuda"):
        timings_ms = []
        for _ in range(repeat):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            fn()
            end.record()
            torch.cuda.synchronize()
            timings_ms.append(start.elapsed_time(end))
        return statistics.mean(timings_ms) / 1e3, statistics.pstdev(timings_ms) / 1e3

    timings_s = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn()
        timings_s.append(time.perf_counter() - t0)
    return statistics.mean(timings_s), statistics.pstdev(timings_s)


def main() -> None:
    args = parse_args()

    warnings.filterwarnings(
        "ignore",
        message="The TorchScript type system doesn't support instance-level annotations",
        category=UserWarning,
    )
    torch.set_default_dtype(torch.float64)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    k_values = parse_int_list(args.k_values)
    l_values = parse_int_list(args.l_values)
    modes = [item.strip() for item in args.modes.split(",") if item.strip()]

    table, batch = build_batch(args.cutoff)
    batch_dict = {
        key: value.to(args.device) if hasattr(value, "to") else value
        for key, value in batch.to_dict().items()
    }

    print(f"model={args.model}")
    print(f"interaction_first={args.interaction_first}")
    print(f"interaction={args.interaction}")
    print(f"contraction_cls_first={args.contraction_cls_first}")
    print(f"contraction_cls={args.contraction_cls}")
    print(f"device={args.device}")
    print("L k mode init_ok forward_ok mean_ms std_ms energy0 n_nonsoc")

    for ell_mode in l_values:
        for k in k_values:
            hidden_irreps = hidden_irreps_for(k, ell_mode)
            try:
                torch.manual_seed(args.seed)
                np.random.seed(args.seed)
                model = build_model(args, table, hidden_irreps).to(args.device)
                model.eval()
                init_ok = True
                n_nonsoc = sum(
                    1 for _, mod in model.named_modules() if mod.__class__.__name__ == "NonSOCContraction"
                )
            except Exception as exc:  # noqa: BLE001
                for mode in modes:
                    print(f"{ell_mode} {k} {mode} False False nan nan init_error:{exc!s} 0")
                continue

            for mode in modes:
                def run():
                    with contraction_mode(model, mode):
                        return model(
                            batch_dict,
                            training=False,
                            compute_force=False,
                            compute_stress=False,
                            compute_virials=False,
                            compute_hessian=False,
                            compute_magforces=False,
                        )

                try:
                    out = run()
                    energy0 = float(out["energy"][0]) if "energy" in out else float("nan")
                    mean_s, std_s = benchmark(run, args.warmup, args.repeat, args.device)
                    print(
                        f"{ell_mode} {k} {mode} {init_ok} True "
                        f"{mean_s*1e3:.3f} {std_s*1e3:.3f} {energy0:.12f} {n_nonsoc}"
                    )
                except Exception as exc:  # noqa: BLE001
                    print(
                        f"{ell_mode} {k} {mode} {init_ok} False nan nan "
                        f"forward_error:{exc!s} {n_nonsoc}"
                    )


if __name__ == "__main__":
    main()
