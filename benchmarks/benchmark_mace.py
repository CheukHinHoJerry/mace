import torch
import numpy as np
from tqdm import tqdm
import ase.io
from e3nn import o3

from mace import modules, tools, data
from mace.data import config_from_atoms, AtomicData
from mace.tools import torch_geometric
from mace.modules.utils import get_edge_vectors_and_lengths
from mace.cli.convert_e3nn_cueq import run as convert_e3nn_cueq

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_dtype(torch.float64)

# === Load structure and prepare batch ===
table = tools.AtomicNumberTable([26])
atomic_energies = np.array([0.0], dtype=float)

at = ase.io.read("/home/coder/project/magnetic-mace/data/ace_ralf_data_Fe/exp_usual_mace/defects_ats.xyz", "-1")
configs = [config_from_atoms(at)]
magmoms = torch.tensor(at.arrays["dft_magmom"], dtype=torch.float64, requires_grad=True)
configs[0].magmom = magmoms

atomic_data_list = [AtomicData.from_config(cfg, z_table=table, cutoff=5.0) for cfg in configs]
batch = next(iter(torch_geometric.dataloader.DataLoader(atomic_data_list, batch_size=1)))
batch = batch.to(device)

# === Result container ===
results = []

def benchmark_mace_interaction_product(model, batch, iterations=20):
    interaction_times = [[], []]
    product_times = [[], []]

    positions = batch["positions"]
    node_attrs = batch["node_attrs"]
    magmom = batch["magmom"]
    edge_index = batch["edge_index"]
    shifts = batch["shifts"]
    sender, receiver = edge_index

    node_feats = model.node_embedding(node_attrs)
    vectors, lengths = get_edge_vectors_and_lengths(positions, edge_index, shifts)
    edge_attrs = model.spherical_harmonics(vectors)
    edge_feats = model.radial_embedding(lengths, node_attrs, edge_index, model.atomic_numbers)

    for _ in range(iterations):
        nf = node_feats.clone()
        for layer_idx in range(2):
            interaction = model.interactions[layer_idx]
            product = model.products[layer_idx]

            torch.cuda.synchronize()
            t0 = torch.cuda.Event(enable_timing=True)
            t1 = torch.cuda.Event(enable_timing=True)
            t0.record()

            nf, sc = interaction(
                node_attrs=node_attrs,
                node_feats=nf,
                edge_attrs=edge_attrs,
                edge_feats=edge_feats,
                edge_index=edge_index,
            )

            t1.record()
            torch.cuda.synchronize()
            interaction_times[layer_idx].append(t0.elapsed_time(t1))  # ms

            t0.record()
            nf = product(
                node_feats=nf,
                sc=sc,
                node_attrs=node_attrs,
            )
            t1.record()
            torch.cuda.synchronize()
            product_times[layer_idx].append(t0.elapsed_time(t1))  # ms

    return {
        "interaction_avg": [np.mean(t) for t in interaction_times],
        "interaction_std": [np.std(t) for t in interaction_times],
        "product_avg": [np.mean(t) for t in product_times],
        "product_std": [np.std(t) for t in product_times],
    }

# === Loop over different hidden_irreps ===
irreps_list = ["128x0e", "128x0e+128x1o", "128x0e+128x1o+128x2e"]

for HI in irreps_list:
    print(f"\n--- Benchmarking for hidden_irreps = {HI} ---")

    model_config = dict(
        r_max=5,
        num_bessel=8,
        num_polynomial_cutoff=5,
        max_ell=3,
        interaction_cls_first=modules.interaction_classes[
            "RealAgnosticDensityInteractionBlock"
        ],
        interaction_cls=modules.interaction_classes[
            "RealAgnosticDensityResidualInteractionBlock"
        ],
        num_interactions=2,
        num_elements=1,
        hidden_irreps=o3.Irreps(HI),
        MLP_irreps=o3.Irreps("16x0e"),
        gate=torch.nn.functional.silu,
        atomic_energies=atomic_energies,
        avg_num_neighbors=8,
        atomic_numbers=table.zs,
        correlation=3,
        radial_type="bessel",
        contraction_cls_first="SymmetricContraction",
        contraction_cls="SymmetricContraction",
        atomic_inter_scale=[1.0],
        atomic_inter_shift=[0.0, 0.1],
    )

    model = modules.ScaleShiftMACE(**model_config)
    model = convert_e3nn_cueq(model).to(device)

    # Warm-up
    for _ in tqdm(range(10), desc="Warming up"):
        model(batch)

    # Benchmark
    timing = benchmark_mace_interaction_product(model, batch, iterations=20)
    results.append((HI, timing))

# === Final Summary Table ===
print("\n=== Summary Table ===")
print("{:<35} {:>15} {:>15} {:>15} {:>15}".format(
    "Hidden Irreps", "Inter1 (ms)", "Prod1 (ms)", "Inter2 (ms)", "Prod2 (ms)"
))

for HI, timing in results:
    print("{:<35} {:>6.2f} ± {:<5.2f} {:>6.2f} ± {:<5.2f} {:>6.2f} ± {:<5.2f} {:>6.2f} ± {:<5.2f}".format(
        HI,
        timing["interaction_avg"][0], timing["interaction_std"][0],
        timing["product_avg"][0],     timing["product_std"][0],
        timing["interaction_avg"][1], timing["interaction_std"][1],
        timing["product_avg"][1],     timing["product_std"][1],
    ))
