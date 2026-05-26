"""Symmetry/equivariance check for magnetic MACE models on magnetic-pr.

A spin-orbit-COUPLED (SOC) model: energy changes when magmoms are rotated
independently of positions, but is invariant under a *joint* rotation of
positions+magmoms.

A NON-SOC model: energy is invariant under rotating magmoms independently
(only |m_i| magnitudes couple), and of course also under joint rotation.

Select the model via env vars:
    MODEL   (default MagneticScaleShiftMACE)
    IFIRST  (default MagneticRealAgnosticSpinOrbitCoupledDensityInteractionBlock)
    INTER   (default MagneticRealAgnosticResidueSpinOrbitCoupledDensityInteractionBlock)
    HIDDEN  (default "64x0e+64x1o")
    MAXMELL (default 3)
"""
import os
import numpy as np
import torch
from e3nn import o3
from scipy.spatial.transform import Rotation as R

from mace import data, modules, tools
from mace.tools import torch_geometric

torch.set_default_dtype(torch.float64)
torch.manual_seed(42)
np.random.seed(42)

MODEL = os.environ.get("MODEL", "MagneticScaleShiftMACE")
IFIRST = os.environ.get("IFIRST", "MagneticRealAgnosticSpinOrbitCoupledDensityInteractionBlock")
INTER = os.environ.get("INTER", "MagneticRealAgnosticResidueSpinOrbitCoupledDensityInteractionBlock")
HIDDEN = os.environ.get("HIDDEN", "64x0e+64x1o")
MAXMELL = int(os.environ.get("MAXMELL", "3"))

table = tools.AtomicNumberTable([1, 8])

model_config = dict(
    r_max=5.0,
    num_bessel=8,
    num_polynomial_cutoff=5,
    max_ell=3,
    interaction_cls_first=modules.interaction_classes[IFIRST],
    interaction_cls=modules.interaction_classes[INTER],
    num_interactions=2,
    num_elements=2,
    hidden_irreps=o3.Irreps(HIDDEN),
    MLP_irreps=o3.Irreps("16x0e"),
    atomic_energies=np.array([0.0, 0.0], dtype=float),
    avg_num_neighbors=8.0,
    atomic_numbers=table.zs,
    correlation=3,
    gate=torch.nn.functional.silu,
    radial_type="bessel",
    distance_transform="Agnesi",
    pair_repulsion=False,
    atomic_inter_scale=1.0,
    atomic_inter_shift=0.0,
    m_max=[3.0, 5.0],
    num_mag_radial_basis=8,
    max_m_ell=MAXMELL,
    num_mag_radial_basis_one_body=10,
    use_magmom_one_body=False,
    heads=["default"],
)

print(f"MODEL={MODEL}\nIFIRST={IFIRST}\nINTER={INTER}\nHIDDEN={HIDDEN} MAXMELL={MAXMELL}")
model = getattr(modules, MODEL)(**model_config)

positions = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
magmom = np.array([[1.0, 0.0, 0.0], [0.0, -2.0, 0.0], [0.0, 0.0, -3.0]])


from ase import Atoms
from mace.data.utils import KeySpecification, config_from_atoms

key_spec = KeySpecification(
    info_keys={"energy": "REF_energy"},
    arrays_keys={"forces": "REF_forces", "magmom": "REF_magmom"},
)


def make_config(pos, mag):
    atoms = Atoms(numbers=[8, 1, 1], positions=pos, cell=[12.0, 12.0, 12.0], pbc=False)
    atoms.info["REF_energy"] = -1.5
    atoms.new_array("REF_forces", np.zeros_like(pos))
    atoms.new_array("REF_magmom", np.asarray(mag, dtype=float))
    return config_from_atoms(atoms, key_specification=key_spec)


rot = R.from_euler("z", np.random.rand() * 360, degrees=True).as_matrix()
configs = [
    ("base", make_config(positions, magmom)),
    ("rot_magmom_only", make_config(positions, magmom @ rot.T)),
    ("rot_positions_only", make_config(positions @ rot.T, magmom)),
    ("rot_both", make_config(positions @ rot.T, magmom @ rot.T)),
]
atomic_data = [data.AtomicData.from_config(c, z_table=table, cutoff=5.0) for _, c in configs]
loader = torch_geometric.dataloader.DataLoader(atomic_data, batch_size=len(configs))
batch = next(iter(loader))
out = model(batch.to_dict(), training=False)
energies = out["energy"]
e0 = energies[0].item()
scale = max(abs(energies[0].item()), 1.0)  # relative tolerance: energies are unscaled/large
print("\n=== energy deltas vs base (relative ΔE/|E_base|) ===")
results = {}
for i, (name, _) in enumerate(configs):
    rel = abs(energies[i] - energies[0]).item() / scale
    results[name] = rel
    print(f"{name:20s} E={energies[i].item():.6f}  ΔE/|E|={rel:.3e}")

print("\n=== verdict (relative tol) ===")
tol = 1e-10        # machine-precision-ish invariance
changed = 1e-6     # a real, non-noise change
inv = lambda k: results[k] < tol
print(f"joint-rotation invariant (rot_both):       {inv('rot_both')}")
print(f"magmom-rotation invariant (rot_magmom):    {inv('rot_magmom_only')}")
print(f"position-rotation invariant (rot_pos):     {inv('rot_positions_only')}")
soc = (results["rot_magmom_only"] > changed) and inv("rot_both")
nonsoc = inv("rot_magmom_only") and inv("rot_positions_only") and inv("rot_both")
print(f"--> looks SOC (magmom rotation changes E): {soc}")
print(f"--> looks NON-SOC (magmom decoupled):      {nonsoc}")
