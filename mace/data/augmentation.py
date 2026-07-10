import importlib

import numpy as np
import torch

from mace.tools import torch_geometric as tg_mace

# --- Optional torch_geometric support ---
if importlib.util.find_spec("torch_geometric") is not None:
    from torch_geometric.transforms import BaseTransform

    has_tg = True
else:
    has_tg = False

    # --- Minimal stub classes so code still runs ---
    class BaseTransform:
        """Fallback stub if torch_geometric is unavailable."""

        def forward(self, data):
            return data


def sample_rotation_matrix(device, dtype):
    """Sample a [3, 3] rotation matrix uniformly over SO(3) (Shoemake's method)."""
    # Random unit quaternion -> rotation matrix.
    u1, u2, u3 = torch.rand(3, device=device, dtype=dtype)
    q1 = torch.sqrt(1 - u1) * torch.sin(2 * np.pi * u2)
    q2 = torch.sqrt(1 - u1) * torch.cos(2 * np.pi * u2)
    q3 = torch.sqrt(u1) * torch.sin(2 * np.pi * u3)
    q4 = torch.sqrt(u1) * torch.cos(2 * np.pi * u3)

    R = torch.tensor(
        [
            [
                1 - 2 * (q3**2 + q4**2),
                2 * (q2 * q3 - q1 * q4),
                2 * (q2 * q4 + q1 * q3),
            ],
            [
                2 * (q2 * q3 + q1 * q4),
                1 - 2 * (q2**2 + q4**2),
                2 * (q3 * q4 - q1 * q2),
            ],
            [
                2 * (q2 * q4 - q1 * q3),
                2 * (q3 * q4 + q1 * q2),
                1 - 2 * (q2**2 + q3**2),
            ],
        ],
        device=device,
        dtype=dtype,
    )
    return R


def rotate_batch_dict(batch_dict, R, rotate_positions=True, rotate_magmom=True):
    """Copy of ``batch_dict`` with rotation ``R`` applied independently to the spatial
    quantities (positions, shifts, cell) and/or the magnetic moments. Rotate one or
    the other to probe the independent invariances E(Rx, m)=E(x, m) (positions only)
    and E(x, Rm)=E(x, m) (spins only). Rotated/copied tensors are detached fresh
    leaves so the copy is an independent model input."""
    rot = dict(batch_dict)
    Rt = R.t()
    pos = batch_dict["positions"].detach()
    rot["positions"] = pos @ Rt if rotate_positions else pos.clone()
    if batch_dict.get("shifts") is not None:
        s = batch_dict["shifts"].detach()
        rot["shifts"] = s @ Rt if rotate_positions else s
    if batch_dict.get("cell") is not None:  # [3 * n_graphs, 3], rows are lattice vectors
        c = batch_dict["cell"].detach()
        rot["cell"] = c @ Rt if rotate_positions else c
    if batch_dict.get("magmom") is not None:
        m = batch_dict["magmom"].detach()
        rot["magmom"] = m @ Rt if rotate_magmom else m.clone()
    if batch_dict.get("node_attrs") is not None:
        rot["node_attrs"] = batch_dict["node_attrs"].detach().clone()
    return rot


class Random3DRotation(BaseTransform):
    """
    Apply a random SO(3) rotation to all magnetic moments in a configuration.
    A single rotation is applied per structure, preserving relative orientation.
    """

    def forward(self, data):
        if hasattr(data, "magmom") and data.magmom is not None:
            R = sample_rotation_matrix(data.magmom.device, data.magmom.dtype)
            # === Apply to magmom (shape [N, 3])
            data.magmom = torch.matmul(data.magmom, R.T)
            if hasattr(data, "magforces") and data.magforces is not None:
                data.magforces = torch.matmul(data.magforces, R.T)

        return data


def create_random_rotation_loader(original_loader):
    """
    Create a new DataLoader with hemisphere rotation augmentation.

    Args:
        original_loader: Original PyTorch Geometric DataLoader

    Returns:
        New DataLoader with hemisphere rotation transform
    """
    if not has_tg:
        raise ImportError(
            "torch_geometric is required for DataLoader functionality.\n"
            "Install it via: pip install torch-geometric"
        )

    transform = Random3DRotation()

    # Apply transform to dataset
    dataset = original_loader.dataset

    # Create new dataset with transform
    class TransformedDataset:
        def __init__(self, original_dataset, transform):
            self.dataset = original_dataset
            self.transform = transform

        def __len__(self):
            return len(self.dataset)

        def __getitem__(self, idx):
            data = self.dataset[idx]
            return self.transform(data)

    transformed_dataset = TransformedDataset(dataset, transform)

    # Under distributed training the original loader has a DistributedSampler
    # already sharding the dataset per rank; replacing it with shuffle=True
    # would make every rank iterate the whole dataset, duplicating samples and
    # breaking the effective epoch size. Pass the sampler through in that case
    # (and don't also set shuffle=, which DataLoader forbids alongside a
    # sampler). Otherwise keep the previous shuffle-based behavior.
    sampler = getattr(original_loader, "sampler", None)
    is_distributed_sampler = isinstance(
        sampler, torch.utils.data.distributed.DistributedSampler
    )
    loader_kwargs = dict(
        batch_size=original_loader.batch_size,
        num_workers=original_loader.num_workers,
        pin_memory=original_loader.pin_memory,
        drop_last=original_loader.drop_last,
    )
    if is_distributed_sampler:
        loader_kwargs["sampler"] = sampler
    else:
        loader_kwargs["shuffle"] = not isinstance(
            sampler, torch.utils.data.SequentialSampler
        )

    new_loader = tg_mace.dataloader.DataLoader(transformed_dataset, **loader_kwargs)

    return new_loader
