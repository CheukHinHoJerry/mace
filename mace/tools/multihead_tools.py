import argparse
import dataclasses
import logging
import os
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch

from mace.cli.fine_tuning_select import select_samples
from mace.tools.run_train_utils import normalize_file_paths
from mace.tools.scripts_utils import (
    SubsetCollection,
    dict_to_namespace,
    extract_config_mace_model,
    get_dataset_from_xyz,
    check_path_ase_read,
)


@dataclasses.dataclass
class HeadConfig:
    head_name: str
    train_file: Optional[Union[str, List[str]]] = None
    valid_file: Optional[Union[str, List[str]]] = None
    test_file: Optional[str] = None
    test_dir: Optional[str] = None
    E0s: Optional[Any] = None
    statistics_file: Optional[str] = None
    valid_fraction: Optional[float] = None
    config_type_weights: Optional[Dict[str, float]] = None
    energy_key: Optional[str] = None
    forces_key: Optional[str] = None
    stress_key: Optional[str] = None
    virials_key: Optional[str] = None
    dipole_key: Optional[str] = None
    charges_key: Optional[str] = None
    magmom_key: Optional[str] = None
    magforces_key: Optional[str] = None
    keep_isolated_atoms: Optional[bool] = None
    atomic_numbers: Optional[Union[List[int], List[str]]] = None
    mean: Optional[float] = None
    std: Optional[float] = None
    avg_num_neighbors: Optional[float] = None
    compute_avg_num_neighbors: Optional[bool] = None
    collections: Optional[SubsetCollection] = None
    train_loader: torch.utils.data.DataLoader = None
    z_table: Optional[Any] = None
    atomic_energies_dict: Optional[Dict[str, float]] = None


def dict_head_to_dataclass(
    head: Dict[str, Any], head_name: str, args: argparse.Namespace
) -> HeadConfig:
    r"""Convert head dictionary to HeadConfig dataclass."""

    return HeadConfig(
        head_name=head_name,
        train_file=head.get("train_file", args.train_file),
        valid_file=head.get("valid_file", args.valid_file),
        test_file=head.get("test_file", None),
        test_dir=head.get("test_dir", None),
        E0s=head.get("E0s", args.E0s),
        statistics_file=head.get("statistics_file", args.statistics_file),
        valid_fraction=head.get("valid_fraction", args.valid_fraction),
        config_type_weights=head.get("config_type_weights", args.config_type_weights),
        compute_avg_num_neighbors=head.get(
            "compute_avg_num_neighbors", args.compute_avg_num_neighbors
        ),
        atomic_numbers=head.get("atomic_numbers", args.atomic_numbers),
        mean=head.get("mean", args.mean),
        std=head.get("std", args.std),
        avg_num_neighbors=head.get("avg_num_neighbors", args.avg_num_neighbors),
        energy_key=head.get("energy_key", args.energy_key),
        forces_key=head.get("forces_key", args.forces_key),
        stress_key=head.get("stress_key", args.stress_key),
        virials_key=head.get("virials_key", args.virials_key),
        dipole_key=head.get("dipole_key", args.dipole_key),
        charges_key=head.get("charges_key", args.charges_key),
        magmom_key=head.get("magmom_key", args.magmom_key),
        magforces_key=head.get("magforces_key", args.magforces_key),
        keep_isolated_atoms=head.get("keep_isolated_atoms", args.keep_isolated_atoms),
    )


def prepare_default_head(args: argparse.Namespace) -> Dict[str, Any]:
    r"""Prepare a default head from args."""
    return {
        "default": {
            "train_file": args.train_file,
            "valid_file": args.valid_file,
            "test_file": args.test_file,
            "test_dir": args.test_dir,
            "E0s": args.E0s,
            "statistics_file": args.statistics_file,
            "valid_fraction": args.valid_fraction,
            "config_type_weights": args.config_type_weights,
            "energy_key": args.energy_key,
            "forces_key": args.forces_key,
            "stress_key": args.stress_key,
            "virials_key": args.virials_key,
            "dipole_key": args.dipole_key,
            "charges_key": args.charges_key,
            "keep_isolated_atoms": args.keep_isolated_atoms,
        }
    }


def assemble_mp_data(
    args: argparse.Namespace, tag: str, head_configs: List[HeadConfig]
) -> Dict[str, Any]:
    r"""Assemble Materials Project data for fine-tuning."""
    try:
        checkpoint_url = "https://github.com/ACEsuit/mace-mp/releases/download/mace_mp_0b/mp_traj_combined.xyz"
        descriptors_url = "https://github.com/ACEsuit/mace-mp/releases/download/mace_mp_0b/descriptors.npy"
        cache_dir = (
            Path(os.environ.get("XDG_CACHE_HOME", "~/")).expanduser() / ".cache/mace"
        )
        checkpoint_url_name = "".join(
            c for c in os.path.basename(checkpoint_url) if c.isalnum() or c in "_"
        )
        cached_dataset_path = f"{cache_dir}/{checkpoint_url_name}"
        descriptors_url_name = "".join(
            c for c in os.path.basename(descriptors_url) if c.isalnum() or c in "_"
        )
        cached_descriptors_path = f"{cache_dir}/{descriptors_url_name}"
        if not os.path.isfile(cached_dataset_path):
            os.makedirs(cache_dir, exist_ok=True)
            # download and save to disk
            logging.info("Downloading MP structures for finetuning")
            _, http_msg = urllib.request.urlretrieve(
                checkpoint_url, cached_dataset_path
            )
            if "Content-Type: text/html" in http_msg:
                raise RuntimeError(
                    f"Dataset download failed, please check the URL {checkpoint_url}"
                )
            logging.info(f"Materials Project dataset to {cached_dataset_path}")
        if not os.path.isfile(cached_descriptors_path):
            os.makedirs(cache_dir, exist_ok=True)
            # download and save to disk
            logging.info("Downloading MP descriptors for finetuning")
            _, http_msg = urllib.request.urlretrieve(
                descriptors_url, cached_descriptors_path
            )
            if "Content-Type: text/html" in http_msg:
                raise RuntimeError(
                    f"Descriptors download failed, please check the URL {descriptors_url}"
                )
            logging.info(f"Materials Project descriptors to {cached_descriptors_path}")
        dataset_mp = cached_dataset_path
        descriptors_mp = cached_descriptors_path
        msg = f"Using Materials Project dataset with {dataset_mp}"
        logging.info(msg)
        msg = f"Using Materials Project descriptors with {descriptors_mp}"
        logging.info(msg)

        # Use the first file if train_file is a list
        config_pt_paths = []
        for head in head_configs:
            if isinstance(head.train_file, list):
                config_pt_paths.append(head.train_file[0])
            else:
                config_pt_paths.append(head.train_file)

        args_samples = {
            "configs_pt": dataset_mp,
            "configs_ft": config_pt_paths,
            "num_samples": args.num_samples_pt,
            "seed": args.seed,
            "model": args.foundation_model,
            "head_pt": "pbe_mp",
            "head_ft": "Default",
            "weight_pt": args.weight_pt_head,
            "weight_ft": 1.0,
            "filtering_type": "combination",
            "output": f"mp_finetuning-{tag}.xyz",
            "descriptors": descriptors_mp,
            "subselect": args.subselect_pt,
            "device": args.device,
            "default_dtype": args.default_dtype,
        }
        select_samples(dict_to_namespace(args_samples))
        collections_mp, _ = get_dataset_from_xyz(
            work_dir=args.work_dir,
            train_path=f"mp_finetuning-{tag}.xyz",
            valid_path=None,
            valid_fraction=args.valid_fraction,
            config_type_weights=None,
            test_path=None,
            seed=args.seed,
            energy_key="energy",
            forces_key="forces",
            stress_key="stress",
            head_name="pt_head",
            virials_key=args.virials_key,
            dipole_key=args.dipole_key,
            charges_key=args.charges_key,
            keep_isolated_atoms=args.keep_isolated_atoms,
        )
        return collections_mp
    except Exception as exc:
        raise RuntimeError(
            "Model or descriptors download failed and no local model found"
        ) from exc


def prepare_custom_pt_head(
    args: argparse.Namespace,
    avg_num_neighbors: float,
) -> HeadConfig:
    r"""Prepare the replay head used for multihead fine-tuning with a custom foundation model."""
    pt_energy_key = args.pt_energy_key or args.energy_key
    pt_forces_key = args.pt_forces_key or args.forces_key
    pt_stress_key = args.pt_stress_key or args.stress_key
    pt_virials_key = args.pt_virials_key or args.virials_key
    pt_dipole_key = args.pt_dipole_key or args.dipole_key
    pt_charges_key = args.pt_charges_key or args.charges_key
    pt_magmom_key = args.pt_magmom_key or args.magmom_key

    logging.info(
        f"Using the following keys for pt_head: energy={pt_energy_key}, forces={pt_forces_key}, "
        f"stress={pt_stress_key}, virials={pt_virials_key}, dipole={pt_dipole_key}, charges={pt_charges_key}, magmom={pt_magmom_key}"
    )

    pt_train_file = normalize_file_paths(args.pt_train_file)
    pt_valid_file = normalize_file_paths(args.pt_valid_file) if args.pt_valid_file else None
    is_ase_readable = all(check_path_ase_read(f) for f in pt_train_file)

    head_config_pt = HeadConfig(
        head_name="pt_head",
        train_file=pt_train_file,
        valid_file=pt_valid_file,
        E0s="foundation",
        statistics_file=args.statistics_file,
        valid_fraction=args.valid_fraction,
        config_type_weights=None,
        energy_key=pt_energy_key,
        forces_key=pt_forces_key,
        stress_key=pt_stress_key,
        virials_key=pt_virials_key,
        dipole_key=pt_dipole_key,
        charges_key=pt_charges_key,
        magmom_key=pt_magmom_key,
        keep_isolated_atoms=args.keep_isolated_atoms,
        avg_num_neighbors=avg_num_neighbors,
        compute_avg_num_neighbors=False,
    )

    if is_ase_readable:
        collections, atomic_energies_dict = get_dataset_from_xyz(
            work_dir=args.work_dir,
            train_path=args.pt_train_file,
            valid_path=args.pt_valid_file,
            valid_fraction=args.valid_fraction,
            config_type_weights=None,
            test_path=None,
            seed=args.seed,
            energy_key=pt_energy_key,
            forces_key=pt_forces_key,
            stress_key=pt_stress_key,
            virials_key=pt_virials_key,
            dipole_key=pt_dipole_key,
            charges_key=pt_charges_key,
            magmom_key=pt_magmom_key,
            head_name="pt_head",
            keep_isolated_atoms=args.keep_isolated_atoms,
        )
        head_config_pt.collections = collections
        if atomic_energies_dict:
            head_config_pt.atomic_energies_dict = atomic_energies_dict
            logging.info(
                "Using atomic energies inferred from pt_train_file isolated atoms for pt_head."
            )
        logging.info(
            f"Loaded ASE readable pretraining data: train={len(collections.train)}, valid={len(collections.valid)}"
        )
    else:
        logging.info(
            f"Pretraining data file(s) will be loaded as LMDB/HDF5: {pt_train_file}"
        )

    return head_config_pt


def inherit_magnetic_hyperparameters_from_foundation(
    args: argparse.Namespace, model_foundation: torch.nn.Module
) -> Dict[str, Any]:
    r"""Copy magnetic basis hyperparameters from the foundation checkpoint onto args."""
    foundation_config = extract_config_mace_model(model_foundation)
    inherited_magnetic_args: Dict[str, Any] = {}

    foundation_m_max = foundation_config.get("m_max")
    if foundation_m_max is not None:
        if torch.is_tensor(foundation_m_max):
            foundation_m_max = foundation_m_max.detach().cpu().tolist()
        elif hasattr(foundation_m_max, "tolist"):
            foundation_m_max = foundation_m_max.tolist()
        args.m_max = foundation_m_max
        inherited_magnetic_args["m_max_len"] = len(foundation_m_max)

    foundation_max_m_ell = foundation_config.get("max_m_ell")
    if foundation_max_m_ell is not None:
        args.max_m_ell = int(foundation_max_m_ell)
        inherited_magnetic_args["max_m_ell"] = args.max_m_ell

    foundation_num_mag_radial_basis = foundation_config.get("num_mag_radial_basis")
    if foundation_num_mag_radial_basis is not None:
        args.num_mag_radial_basis = int(foundation_num_mag_radial_basis)
        inherited_magnetic_args["num_mag_radial_basis"] = args.num_mag_radial_basis

    foundation_num_mag_radial_basis_one_body = foundation_config.get(
        "num_mag_radial_basis_one_body"
    )
    if foundation_num_mag_radial_basis_one_body is not None:
        args.num_mag_radial_basis_one_body = int(
            foundation_num_mag_radial_basis_one_body
        )
        inherited_magnetic_args["num_mag_radial_basis_one_body"] = (
            args.num_mag_radial_basis_one_body
        )

    return inherited_magnetic_args
