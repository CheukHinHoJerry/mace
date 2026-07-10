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

MODEL_PATH = os.environ.get("MODEL_PATH", "")
if MODEL_PATH:
    print(f"MODEL_PATH={MODEL_PATH}   (loading from disk; MODEL/IFIRST/INTER/HIDDEN ignored)")
    model = torch.load(MODEL_PATH, map_location="cpu")
    model = model.to(torch.float64) if hasattr(model, "to") else model
    model.eval()
    # Read species from the checkpoint so make_config uses matching atomic numbers.
    if hasattr(model, "atomic_numbers"):
        zs_loaded = model.atomic_numbers.cpu().tolist()
        table = tools.AtomicNumberTable(zs_loaded)
        print(f"loaded atomic_numbers = {zs_loaded}")
else:
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
# Independent rotations Q_pos != Q_mag (tests strict O(3) x O(3), not diagonal SO(3))
rot_pos = R.from_euler("z", np.random.rand() * 360, degrees=True).as_matrix()
rot_mag = R.from_euler("x", np.random.rand() * 360, degrees=True).as_matrix()
configs = [
    ("base", make_config(positions, magmom)),
    ("rot_magmom_only", make_config(positions, magmom @ rot.T)),
    ("rot_positions_only", make_config(positions @ rot.T, magmom)),
    ("rot_both", make_config(positions @ rot.T, magmom @ rot.T)),
    # Parity / inversion checks (improper elements of O(3)).
    # SOC (paper Eq. 8, polar-magmom convention): only inv_both is guaranteed.
    # Non-SOC (paper Eq. 9, independent O(3) x O(3)): all three should hold.
    ("inv_positions_only", make_config(-positions, magmom)),
    ("inv_magmom_only", make_config(positions, -magmom)),
    ("inv_both", make_config(-positions, -magmom)),
    # Independent rotations with different Q_pos and Q_mag.
    # Non-SOC MUST be invariant (Eq. 9). SOC breaks by large amount.
    ("indep_rotations_diff", make_config(positions @ rot_pos.T, magmom @ rot_mag.T)),
    # Independent improper element on positions + proper rotation on magmoms.
    # Non-SOC: still invariant. SOC: not guaranteed.
    ("indep_improper_pos", make_config(-positions @ rot_pos.T, magmom @ rot_mag.T)),
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
print(f"joint-rotation invariant  (rot_both):          {inv('rot_both')}")
print(f"magmom-rotation invariant (rot_magmom_only):   {inv('rot_magmom_only')}")
print(f"position-rotation invariant (rot_pos_only):    {inv('rot_positions_only')}")
print(f"joint-inversion invariant (inv_both):          {inv('inv_both')}")
print(f"magmom-inversion invariant (inv_magmom_only):  {inv('inv_magmom_only')}")
print(f"position-inversion invariant (inv_pos_only):   {inv('inv_positions_only')}")
print(f"indep-rotation invariant  (indep_rot_diff):    {inv('indep_rotations_diff')}")
print(f"indep-improper invariant  (indep_improper):    {inv('indep_improper_pos')}")
soc = (
    (results["rot_magmom_only"] > changed)
    and inv("rot_both")
    and inv("inv_both")
)
nonsoc = (
    inv("rot_magmom_only") and inv("rot_positions_only") and inv("rot_both")
    and inv("inv_magmom_only") and inv("inv_positions_only") and inv("inv_both")
    and inv("indep_rotations_diff") and inv("indep_improper_pos")
)
print(f"--> looks SOC (magmom rotation changes E, joint P/rot invariant): {soc}")
print(f"--> looks NON-SOC (fully decoupled O(3) x O(3)):                  {nonsoc}")


# ----------------------------------------------------------------------
# FM vs AFM diagnostic on collinear magmoms
# ----------------------------------------------------------------------
# Same geometry (O-H-H triangle); vary only the magmom pattern along z.
# For a full-O(3)x O(3) non-SOC model:
#   - fm_up == fm_dn   (global spin flip = single-Q'=P mag inversion)
#   - afm_pm == afm_mp (twin: same physical state, model must give equal E)
#   - fm != afm        (Heisenberg-like ordering distinction should survive)
# For a polar-mag SOC model:
#   - fm_up == fm_dn if the FM state happens to have joint-inversion symmetry
#     (proper rotation about x-axis maps up->down for both atoms), likely yes.
#   - afm_pm != afm_mp in general (T-symmetry not enforced -> twin split).
#   - fm != afm expected.
mag_configs = {
    "fm_up":     np.array([[0.,0., 1.], [0.,0., 1.], [0.,0., 1.]]),
    "fm_dn":     np.array([[0.,0.,-1.], [0.,0.,-1.], [0.,0.,-1.]]),
    "afm_hh_pm": np.array([[0.,0., 0.], [0.,0., 1.], [0.,0.,-1.]]),
    "afm_hh_mp": np.array([[0.,0., 0.], [0.,0.,-1.], [0.,0., 1.]]),
}
mag_atomic_data = [
    data.AtomicData.from_config(
        make_config(positions, m), z_table=table, cutoff=5.0
    )
    for m in mag_configs.values()
]
mag_batch = next(iter(
    torch_geometric.dataloader.DataLoader(mag_atomic_data, batch_size=len(mag_configs))
))
mag_out = model(mag_batch.to_dict(), training=False)
mag_energies = {name: e.item() for name, e in zip(mag_configs, mag_out["energy"])}

print("\n=== FM vs AFM diagnostic (collinear magmoms) ===")
for name, e in mag_energies.items():
    print(f"{name:12s} E = {e:+.6f}")
mag_scale = max(abs(mag_energies["fm_up"]), 1.0)
d_fm_twin = abs(mag_energies["fm_up"] - mag_energies["fm_dn"]) / mag_scale
d_afm_twin = abs(mag_energies["afm_hh_pm"] - mag_energies["afm_hh_mp"]) / mag_scale
d_fm_afm = abs(mag_energies["fm_up"] - mag_energies["afm_hh_pm"]) / mag_scale
print(f"\nFM twin split       |E(up,up) - E(dn,dn)| / |E|         = {d_fm_twin:.3e}")
print(f"AFM twin split      |E(0,+,-) - E(0,-,+)| / |E|         = {d_afm_twin:.3e}   <- non-SOC needs 0")
print(f"FM vs AFM distinct  |E(FM)    - E(AFM)| / |E|           = {d_fm_afm:.3e}   <- should be >0")
fm_twin_ok = d_fm_twin < tol
afm_twin_ok = d_afm_twin < tol
fm_vs_afm_ok = d_fm_afm > changed
print(f"--> FM twin symmetry (E(up,up)==E(dn,dn)):       {'OK' if fm_twin_ok else 'BROKEN'}")
print(f"--> AFM twin symmetry (E(0,+,-)==E(0,-,+)):      {'OK' if afm_twin_ok else 'BROKEN'}")
print(f"--> FM vs AFM distinguishable (arch has Heisenberg-like coupling): {'OK' if fm_vs_afm_ok else 'BROKEN'}")

# ----------------------------------------------------------------------
# Test 3: Force / magforce equivariance
# ----------------------------------------------------------------------
# Under position rotation R_pos: forces rotate by R_pos, magforces unchanged.
# Under magmom rotation R_mag: forces unchanged, magforces rotate by R_mag.
# This is the derivative-equivariance statement — E-invariance alone
# does not imply it (and training uses forces, so this is essential).
print("\n=== force / magforce equivariance ===")

def _eval_forces(pos, mag):
    """Return (forces, magforces) for a single config."""
    ad = data.AtomicData.from_config(
        make_config(pos, mag), z_table=table, cutoff=5.0
    )
    batch_local = next(iter(
        torch_geometric.dataloader.DataLoader([ad], batch_size=1)
    ))
    o = model(batch_local.to_dict(), training=True, compute_force=True, compute_magforces=True)
    return o["forces"].detach().numpy(), o["magforces"].detach().numpy()

R_pos_test = R.from_euler("z", 47.0, degrees=True).as_matrix()
R_mag_test = R.from_euler("y", 33.0, degrees=True).as_matrix()

F_base, MF_base = _eval_forces(positions, magmom)
# rotate positions only
F_rp, MF_rp = _eval_forces(positions @ R_pos_test.T, magmom)
# rotate magmoms only
F_rm, MF_rm = _eval_forces(positions, magmom @ R_mag_test.T)

F_scale = max(np.abs(F_base).max(), 1.0)
MF_scale = max(np.abs(MF_base).max(), 1.0)
dF_rp = np.abs(F_rp - F_base @ R_pos_test.T).max() / F_scale
dMF_rp = np.abs(MF_rp - MF_base).max() / MF_scale
dF_rm = np.abs(F_rm - F_base).max() / F_scale
dMF_rm = np.abs(MF_rm - MF_base @ R_mag_test.T).max() / MF_scale
print(f"under R_pos: |F_rot - R_pos F|/max|F|     = {dF_rp:.3e}   (must be ~0)")
print(f"under R_pos: |MF_rot - MF|/max|MF|        = {dMF_rp:.3e}   (must be ~0 for non-SOC)")
print(f"under R_mag: |F_rot - F|/max|F|           = {dF_rm:.3e}   (must be ~0 for non-SOC)")
print(f"under R_mag: |MF_rot - R_mag MF|/max|MF|  = {dMF_rm:.3e}   (must be ~0)")
force_ftol = 1e-8
force_equiv_ok = (
    dF_rp < force_ftol and dMF_rp < force_ftol
    and dF_rm < force_ftol and dMF_rm < force_ftol
)
print(f"--> force/magforce equivariance: {'OK' if force_equiv_ok else 'BROKEN'}")

# ----------------------------------------------------------------------
# Test 4: Isolated-atom direction independence
# ----------------------------------------------------------------------
# Under O(3) x O(3), the magmom on an isolated atom can be rotated freely
# (no other atom defines an anisotropy axis) so E must depend on |m| only.
# Non-SOC: exactly invariant. SOC: coupling to positions is trivial for
# an isolated atom but E may still leak direction dependence via cutoff /
# radial embedding — good sharp check.
print("\n=== isolated-atom direction independence ===")


def _iso_config(mag_vec, species=1):
    at = Atoms(numbers=[species], positions=[[0.0, 0.0, 0.0]],
               cell=[12.0, 12.0, 12.0], pbc=False)
    at.info["REF_energy"] = 0.0
    at.new_array("REF_forces", np.zeros((1, 3)))
    at.new_array("REF_magmom", np.asarray([mag_vec], dtype=float))
    return config_from_atoms(at, key_specification=key_spec)


iso_directions = {
    "iso_+x": [1.0, 0.0, 0.0],
    "iso_+y": [0.0, 1.0, 0.0],
    "iso_+z": [0.0, 0.0, 1.0],
    "iso_-z": [0.0, 0.0, -1.0],
    "iso_diag": [1.0, 1.0, 1.0] / np.sqrt(3),
}
iso_ad = [data.AtomicData.from_config(_iso_config(m), z_table=table, cutoff=5.0)
          for m in iso_directions.values()]
iso_batch = next(iter(
    torch_geometric.dataloader.DataLoader(iso_ad, batch_size=len(iso_ad))
))
iso_out = model(iso_batch.to_dict(), training=False)
iso_e = {name: e.item() for name, e in zip(iso_directions, iso_out["energy"])}
iso_scale = max(max(abs(v) for v in iso_e.values()), 1.0)
for name, e in iso_e.items():
    print(f"{name:10s} E = {e:+.6f}")
iso_ref = iso_e["iso_+z"]
iso_max_dev = max(abs(v - iso_ref) for v in iso_e.values()) / iso_scale
print(f"max deviation across directions / |E|:   {iso_max_dev:.3e}   (must be ~0)")
iso_ok = iso_max_dev < tol
print(f"--> isolated-atom direction independence: {'OK' if iso_ok else 'BROKEN'}")

# ----------------------------------------------------------------------
# Test 8: Non-collinear canting (0 / 90 / 180 degrees between H1 and H2)
# ----------------------------------------------------------------------
# Extends the FM/AFM diagnostic beyond aligned/antialigned collinear.
# All three should give DISTINCT energies — the arch must feel the full
# relative angle, not just sign(m_i . m_j). And 0-deg twin (FM up vs FM dn)
# equality should still hold as before.
print("\n=== canting (0 / 90 / 180 deg between H1 and H2 magmoms) ===")
canting_configs = {
    "cant_0deg":   np.array([[0.,0.,0.], [0.,0., 1.], [0.,0., 1.]]),  # FM
    "cant_90deg":  np.array([[0.,0.,0.], [0.,0., 1.], [1.,0., 0.]]),  # perpendicular
    "cant_180deg": np.array([[0.,0.,0.], [0.,0., 1.], [0.,0.,-1.]]),  # AFM
}
cant_ad = [data.AtomicData.from_config(make_config(positions, m), z_table=table, cutoff=5.0)
           for m in canting_configs.values()]
cant_batch = next(iter(
    torch_geometric.dataloader.DataLoader(cant_ad, batch_size=len(cant_ad))
))
cant_out = model(cant_batch.to_dict(), training=False)
cant_e = {name: e.item() for name, e in zip(canting_configs, cant_out["energy"])}
cant_scale = max(max(abs(v) for v in cant_e.values()), 1.0)
for name, e in cant_e.items():
    print(f"{name:12s} E = {e:+.6f}")
d_0_180 = abs(cant_e["cant_0deg"] - cant_e["cant_180deg"]) / cant_scale
d_0_90 = abs(cant_e["cant_0deg"] - cant_e["cant_90deg"]) / cant_scale
d_90_180 = abs(cant_e["cant_90deg"] - cant_e["cant_180deg"]) / cant_scale
print(f"|E(0 )-E(180)| / |E|  = {d_0_180:.3e}   (should be >0)")
print(f"|E(0 )-E(90 )| / |E|  = {d_0_90:.3e}   (should be >0)")
print(f"|E(90)-E(180)| / |E|  = {d_90_180:.3e}   (should be >0)")
canting_ok = d_0_180 > changed and d_0_90 > changed and d_90_180 > changed
print(f"--> arch resolves full canting angle (not just sign): {'OK' if canting_ok else 'BROKEN'}")

# Roll FM/AFM diagnostics into the non-SOC verdict:
# a correct non-SOC architecture must satisfy every symmetry AND still
# distinguish FM from AFM ordering (otherwise it can't represent Heisenberg).
nonsoc_full = (
    nonsoc and fm_twin_ok and afm_twin_ok and fm_vs_afm_ok
    and force_equiv_ok and iso_ok and canting_ok
)
print(f"\n--> NON-SOC (all symmetries + FM/AFM + forces + iso-atom + canting): {nonsoc_full}")
