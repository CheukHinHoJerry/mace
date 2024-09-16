###########################################################################################
# Data parsing utilities
# Authors: Ilyes Batatia, Gregor Simm and David Kovacs
# This program is distributed under the MIT License (see MIT.md)
###########################################################################################

import logging
import multiprocessing as mp
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import ase.data
import ase.io
import h5py
import numpy as np

from mace.tools import AtomicNumberTable
from tqdm import tqdm

Vector = np.ndarray  # [3,]
Positions = np.ndarray  # [..., 3]
Forces = np.ndarray  # [..., 3]
Stress = np.ndarray  # [6, ], [3,3], [9, ]
Virials = np.ndarray  # [6, ], [3,3], [9, ]
Charges = np.ndarray  # [..., 1]
Cell = np.ndarray  # [3,3]
Pbc = tuple  # (3,)

DEFAULT_CONFIG_TYPE = "Default"
DEFAULT_CONFIG_TYPE_WEIGHTS = {DEFAULT_CONFIG_TYPE: 1.0}


@dataclass
class Configuration:
    atomic_numbers: np.ndarray
    positions: Positions  # Angstrom
    energy: Optional[float] = None  # eV
    forces: Optional[Forces] = None  # eV/Angstrom
    stress: Optional[Stress] = None  # eV/Angstrom^3
    virials: Optional[Virials] = None  # eV
    dipole: Optional[Vector] = None  # Debye
    charges: Optional[Charges] = None  # atomic unit
    cell: Optional[Cell] = None
    pbc: Optional[Pbc] = None

    weight: float = 1.0  # weight of config in loss
    energy_weight: float = 1.0  # weight of config energy in loss
    forces_weight: float = 1.0  # weight of config forces in loss
    stress_weight: float = 1.0  # weight of config stress in loss
    virials_weight: float = 1.0  # weight of config virial in loss
    config_type: Optional[str] = DEFAULT_CONFIG_TYPE  # config_type of config
    head: Optional[str] = "Default"  # head used to compute the config


Configurations = List[Configuration]


def random_train_valid_split(
    items: Sequence, valid_fraction: float, seed: int
) -> Tuple[List, List]:
    assert 0.0 < valid_fraction < 1.0

    size = len(items)
    train_size = size - int(valid_fraction * size)

    indices = list(range(size))
    rng = np.random.default_rng(seed)
    rng.shuffle(indices)

    return (
        [items[i] for i in indices[:train_size]],
        [items[i] for i in indices[train_size:]],
    )


def config_from_atoms_list(
    atoms_list: List[ase.Atoms],
    energy_key="energy",
    forces_key="forces",
    stress_key="stress",
    virials_key="virials",
    dipole_key="dipole",
    charges_key="charges",
    head_key="head",
    config_type_weights: Dict[str, float] = None,
) -> Configurations:
    """Convert list of ase.Atoms into Configurations"""
    if config_type_weights is None:
        config_type_weights = DEFAULT_CONFIG_TYPE_WEIGHTS

    all_configs = []
    for atoms in atoms_list:
        all_configs.append(
            config_from_atoms(
                atoms,
                energy_key=energy_key,
                forces_key=forces_key,
                stress_key=stress_key,
                virials_key=virials_key,
                dipole_key=dipole_key,
                charges_key=charges_key,
                head_key=head_key,
                config_type_weights=config_type_weights,
            )
        )
    return all_configs


def config_from_atoms(
    atoms: ase.Atoms,
    energy_key="energy",
    forces_key="forces",
    stress_key="stress",
    virials_key="virials",
    dipole_key="dipole",
    charges_key="charges",
    head_key="head",
    config_type_weights: Dict[str, float] = None,
) -> Configuration:
    """Convert ase.Atoms to Configuration"""
    if config_type_weights is None:
        config_type_weights = DEFAULT_CONFIG_TYPE_WEIGHTS

    energy = atoms.info.get(energy_key, None)  # eV
    forces = atoms.arrays.get(forces_key, None)  # eV / Ang
    stress = atoms.info.get(stress_key, None)  # eV / Ang
    virials = atoms.info.get(virials_key, None)
    dipole = atoms.info.get(dipole_key, None)  # Debye
    # Charges default to 0 instead of None if not found
    charges = atoms.arrays.get(charges_key, np.zeros(len(atoms)))  # atomic unit
    atomic_numbers = np.array(
        [ase.data.atomic_numbers[symbol] for symbol in atoms.symbols]
    )
    pbc = tuple(atoms.get_pbc())
    cell = np.array(atoms.get_cell())
    config_type = atoms.info.get("config_type", "Default")
    weight = atoms.info.get("config_weight", 1.0) * config_type_weights.get(
        config_type, 1.0
    )
    energy_weight = atoms.info.get("config_energy_weight", 1.0)
    forces_weight = atoms.info.get("config_forces_weight", 1.0)
    stress_weight = atoms.info.get("config_stress_weight", 1.0)
    virials_weight = atoms.info.get("config_virials_weight", 1.0)

    head = atoms.info.get(head_key, "Default")

    # fill in missing quantities but set their weight to 0.0
    if energy is None:
        energy = 0.0
        energy_weight = 0.0
    if forces is None:
        forces = np.zeros(np.shape(atoms.positions))
        forces_weight = 0.0
    if stress is None:
        stress = np.zeros(6)
        stress_weight = 0.0
    if virials is None:
        virials = np.zeros((3, 3))
        virials_weight = 0.0
    if dipole is None:
        dipole = np.zeros(3)
        # dipoles_weight = 0.0

    return Configuration(
        atomic_numbers=atomic_numbers,
        positions=atoms.get_positions(),
        energy=energy,
        forces=forces,
        stress=stress,
        virials=virials,
        dipole=dipole,
        charges=charges,
        weight=weight,
        head=head,
        energy_weight=energy_weight,
        forces_weight=forces_weight,
        stress_weight=stress_weight,
        virials_weight=virials_weight,
        config_type=config_type,
        pbc=pbc,
        cell=cell,
    )


def test_config_types(
    test_configs: Configurations,
) -> List[Tuple[Optional[str], List[Configuration]]]:
    """Split test set based on config_type-s"""
    test_by_ct = []
    all_cts = []
    for conf in test_configs:
        config_type_name = conf.config_type + "_" + conf.head
        if config_type_name not in all_cts:
            all_cts.append(config_type_name)
            test_by_ct.append((config_type_name, [conf]))
        else:
            ind = all_cts.index(config_type_name)
            test_by_ct[ind][1].append(conf)
    return test_by_ct


def load_from_extxyzs(
    file_path: str,
    config_type_weights: Dict,
    energy_key: str = "energy",
    forces_key: str = "forces",
    stress_key: str = "stress",
    virials_key: str = "virials",
    dipole_key: str = "dipole",
    charges_key: str = "charges",
    head_key: str = "head",
) -> Tuple[Dict[int, float], Configurations]:
    atoms_list = atoms_from_oc20(file_path)

    atomic_energies_dict = {}

    heads = set()
    for atoms in atoms_list:
        heads.add(atoms.info.get("head", "Default"))
    heads = list(heads)

    configs = config_from_atoms_list(
        atoms_list,
        config_type_weights=config_type_weights,
        energy_key="energy",
        forces_key="forces",
        stress_key="stress",
        virials_key="virials",
        dipole_key="dipole",
        charges_key="charges",
        head_key="head",
    )
    return atomic_energies_dict, configs, heads

def load_from_xyz(
    file_path: str,
    config_type_weights: Dict,
    energy_key: str = "energy",
    forces_key: str = "forces",
    stress_key: str = "stress",
    virials_key: str = "virials",
    dipole_key: str = "dipole",
    charges_key: str = "charges",
    head_key: str = "head",
    extract_atomic_energies: bool = False,
    keep_isolated_atoms: bool = False,
) -> Tuple[Dict[int, float], Configurations]:
    atoms_list = ase.io.read(file_path, index=":")

    energy_from_calc = False
    forces_from_calc = False
    stress_from_calc = False

    ## Perform initial checks and log warnings
    if energy_key == "energy":
        logging.info(
            "Using energy_key 'energy' is unsafe, consider using a different key, rewriting energies to '_REF_energy'"
        )
        energy_from_calc = True
        energy_key = "_REF_energy"

    if forces_key == "forces":
        logging.info(
            "Using forces_key 'forces' is unsafe, consider using a different key, rewriting forces to '_REF_forces'"
        )
        forces_from_calc = True
        forces_key = "_REF_forces"

    if stress_key == "stress":
        logging.info(
            "Using stress_key 'stress' is unsafe, consider using a different key, rewriting stress to '_REF_stress'"
        )
        stress_from_calc = True
        stress_key = "_REF_stress"

    for atoms in atoms_list:
        if energy_from_calc:
            try:
                atoms.info["_REF_energy"] = atoms.get_potential_energy()
            except Exception as e:  # pylint: disable=W0703
                logging.warning(f"Failed to extract energy: {e}")
                atoms.info["_REF_energy"] = None

        if forces_from_calc:
            try:
                atoms.arrays["_REF_forces"] = atoms.get_forces()
            except Exception as e:  # pylint: disable=W0703
                logging.warning(f"Failed to extract forces: {e}")
                atoms.arrays["_REF_forces"] = None

        if stress_from_calc:
            try:
                atoms.info["_REF_stress"] = atoms.get_stress()
            except Exception as e:  # pylint: disable=W0703
                atoms.info["_REF_stress"] = None

    if not isinstance(atoms_list, list):
        atoms_list = [atoms_list]
    atomic_energies_dict = {}
    if extract_atomic_energies:
        atoms_without_iso_atoms = []

        for idx, atoms in enumerate(atoms_list):
            if atoms.info.get("config_type") == "IsolatedAtom":
                assert (
                    len(atoms) == 1
                ), f"Got config_type=IsolatedAtom for a config with len {len(atoms)}"
                if energy_key in atoms.info.keys():
                    head = atoms.info.get(head_key, "Default")
                    if head not in atomic_energies_dict:
                        atomic_energies_dict[head] = {}
                    atomic_energies_dict[head][atoms.get_atomic_numbers()[0]] = (
                        atoms.info[energy_key]
                    )
                else:
                    logging.warning(
                        f"Configuration '{idx}' is marked as 'IsolatedAtom' "
                        "but does not contain an energy. Zero energy will be used."
                    )
                    head = atoms.info.get(head_key, "Default")
                    if head not in atomic_energies_dict:
                        atomic_energies_dict[head] = {}
                    atomic_energies_dict[head][atoms.get_atomic_numbers()[0]] = (
                        np.zeros(1)
                    )
            else:
                atoms_without_iso_atoms.append(atoms)

        if len(atomic_energies_dict) > 0:
            logging.info("Using isolated atom energies from training file")
        if not keep_isolated_atoms:
            atoms_list = atoms_without_iso_atoms
    heads = set()
    for atoms in atoms_list:
        heads.add(atoms.info.get(head_key, "Default"))
    heads = list(heads)
    configs = config_from_atoms_list(
        atoms_list,
        config_type_weights=config_type_weights,
        energy_key=energy_key,
        forces_key=forces_key,
        stress_key=stress_key,
        virials_key=virials_key,
        dipole_key=dipole_key,
        charges_key=charges_key,
        head_key=head_key,
    )
    return atomic_energies_dict, configs, heads

def load_from_h5(
    file_path: str,
    config_type_weights: Dict,
    h5_positions_key: str = 'atXYZ',
    h5_numbers_key: str = "atNUM",
    h5_energy_key: str = 'ePBE0+MBD',
    h5_forces_key: str = 'totFOR',
) -> Tuple[Dict[int, float], Configurations]:
    atoms_list = atoms_from_hdf5_ani(
        file_path, 
        positions_key=h5_positions_key,
        numbers_key=h5_numbers_key,
        energy_key=h5_energy_key,
        forces_key=h5_forces_key
    )


    atomic_energies_dict = {}

    heads = set()
    for atoms in atoms_list:
        heads.add(atoms.info.get("head", "Default"))
    heads = list(heads)

    configs = config_from_atoms_list(
        atoms_list,
        config_type_weights=config_type_weights,
        energy_key="energy",
        forces_key="forces",
        stress_key="stress",
        virials_key="virials",
        dipole_key="dipole",
        charges_key="charges",
        head_key="head",
    )
    return atomic_energies_dict, configs, heads

def atoms_from_hdf5(file_path, positions_key='atXYZ', numbers_key="atNUM", energy_key='ePBE0+MBD', forces_key='totFOR'):
    aqm = h5py.File(file_path, "r")
    atoms_list = []
    AQMmol_ids = list(aqm.keys())

    for molid in AQMmol_ids:
        AQMconf_ids = list(aqm[molid].keys())
        for confid in AQMconf_ids:
            curr_cfg = aqm[molid][confid]
            atoms = ase.Atoms(positions=np.array(curr_cfg[positions_key]), numbers=np.array(curr_cfg[numbers_key]))
            atoms.info['energy'] = np.array(curr_cfg[energy_key]).item()
            atoms.arrays['forces'] = np.array(curr_cfg[forces_key])
            # atoms.info['head'] = "Default"
            atoms_list.append(atoms)

    return atoms_list

def atoms_from_hdf5_ani(file_path, positions_key='coordinates', numbers_key="species", energy_key='energies', forces_key='forces'):
    h5 = h5py.File(file_path, "r")
    atoms_list = []
    for num_atoms, properties in tqdm(h5.items()):        #Iterate thorugh like a dictionary
        coordinates = np.array(properties['coordinates'])     #Output of properties is of type h5py Dataset
        species = np.array(properties['species'])
        energies = np.array(properties['energies'])
        forces = np.array(properties['forces'])
        for c, s, e, f in zip(coordinates, species, energies, forces):
            atoms = ase.Atoms(positions=c, numbers=s)
            atoms.info['energy'] = e * ase.units.Hartree # convert to eV
            atoms.arrays['forces'] = f  * ase.units.Hartree # convert to eV
            atoms_list.append(atoms) 
    return atoms_list

def read_atoms_file(identifier):
    return ase.io.read(identifier, index=":")

def atoms_from_oc20(file_path, positions_key='coordinates', numbers_key="species", energy_key='energies', forces_key='forces'):
    filenames = [f for f in os.listdir(file_path) if f.endswith(".extxyz")]
    identifiers = [os.path.join(file_path, f) for f in filenames if f.endswith(".extxyz")]
    
    with mp.Pool(16) as pool:
        results = list(tqdm(pool.imap(read_atoms_file, identifiers), total=len(identifiers)))
    
    # Flatten the list of lists
    atoms_list = [atom for sublist in results for atom in sublist]
    
    return atoms_list


def compute_average_E0s( # TODO: compute species dependant
    collections_train: Configurations, z_table: AtomicNumberTable, heads: List[str]
) -> Dict[int, float]:
    """
    Function to compute the average interaction energy of each chemical element
    returns dictionary of E0s
    """
    len_train = len(collections_train)
    len_zs = len(z_table)
    atomic_energies_dict = {}
    for head in heads:
        A = np.zeros((len_train, len_zs))
        B = np.zeros(len_train)
        if head not in atomic_energies_dict:
            atomic_energies_dict[head] = {}
        for i in range(len_train):
            if collections_train[i].head != head:
                continue
            B[i] = collections_train[i].energy
            for j, z in enumerate(z_table.zs):
                A[i, j] = np.count_nonzero(collections_train[i].atomic_numbers == z)
        try:
            E0s = np.linalg.lstsq(A, B, rcond=None)[0]
            for i, z in enumerate(z_table.zs):
                atomic_energies_dict[head][z] = E0s[i]
        except np.linalg.LinAlgError:
            logging.warning(
                "Failed to compute E0s using least squares regression, using the same for all atoms"
            )
            for i, z in enumerate(z_table.zs):
                atomic_energies_dict[head][z] = 0.0
    return atomic_energies_dict


def save_dataset_as_HDF5(dataset: List, out_name: str) -> None:
    with h5py.File(out_name, "w") as f:
        for i, data in enumerate(dataset):
            grp = f.create_group(f"config_{i}")
            grp["num_nodes"] = data.num_nodes
            grp["edge_index"] = data.edge_index
            grp["positions"] = data.positions
            grp["shifts"] = data.shifts
            grp["unit_shifts"] = data.unit_shifts
            grp["cell"] = data.cell
            grp["node_attrs"] = data.node_attrs
            grp["weight"] = data.weight
            grp["energy_weight"] = data.energy_weight
            grp["forces_weight"] = data.forces_weight
            grp["stress_weight"] = data.stress_weight
            grp["virials_weight"] = data.virials_weight
            grp["forces"] = data.forces
            grp["energy"] = data.energy
            grp["stress"] = data.stress
            grp["virials"] = data.virials
            grp["dipole"] = data.dipole
            grp["charges"] = data.charges
            grp["head"] = data.head


def save_AtomicData_to_HDF5(data, i, h5_file) -> None:
    grp = h5_file.create_group(f"config_{i}")
    grp["num_nodes"] = data.num_nodes
    grp["edge_index"] = data.edge_index
    grp["positions"] = data.positions
    grp["shifts"] = data.shifts
    grp["unit_shifts"] = data.unit_shifts
    grp["cell"] = data.cell
    grp["node_attrs"] = data.node_attrs
    grp["weight"] = data.weight
    grp["energy_weight"] = data.energy_weight
    grp["forces_weight"] = data.forces_weight
    grp["stress_weight"] = data.stress_weight
    grp["virials_weight"] = data.virials_weight
    grp["forces"] = data.forces
    grp["energy"] = data.energy
    grp["stress"] = data.stress
    grp["virials"] = data.virials
    grp["dipole"] = data.dipole
    grp["charges"] = data.charges
    grp["head"] = data.head


def save_configurations_as_HDF5(configurations: Configurations, _, h5_file) -> None:
    grp = h5_file.create_group("config_batch_0")
    for j, config in enumerate(configurations):
        subgroup_name = f"config_{j}"
        subgroup = grp.create_group(subgroup_name)
        subgroup["atomic_numbers"] = write_value(config.atomic_numbers)
        subgroup["positions"] = write_value(config.positions)
        subgroup["energy"] = write_value(config.energy)
        subgroup["forces"] = write_value(config.forces)
        subgroup["stress"] = write_value(config.stress)
        subgroup["virials"] = write_value(config.virials)
        subgroup["head"] = write_value(config.head)
        subgroup["dipole"] = write_value(config.dipole)
        subgroup["charges"] = write_value(config.charges)
        subgroup["cell"] = write_value(config.cell)
        subgroup["pbc"] = write_value(config.pbc)
        subgroup["weight"] = write_value(config.weight)
        subgroup["energy_weight"] = write_value(config.energy_weight)
        subgroup["forces_weight"] = write_value(config.forces_weight)
        subgroup["stress_weight"] = write_value(config.stress_weight)
        subgroup["virials_weight"] = write_value(config.virials_weight)
        subgroup["config_type"] = write_value(config.config_type)


def write_value(value):
    return value if value is not None else "None"
