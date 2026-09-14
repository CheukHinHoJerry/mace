import logging
from typing import Optional

import torch

from mace.tools.utils import AtomicNumberTable


def _copy_radial_weights(
    model: torch.nn.Module, model_foundations: torch.nn.Module
) -> None:
    """Copy the radial basis weights from the foundation model in-place.

    Preserves the target's buffer-vs-Parameter registration (BesselBasis /
    GaussianBasis expose ``*_weights`` as a Parameter only when ``trainable=True``;
    otherwise it is a register_buffer). Handles both radial classes.
    """
    dst = model.radial_embedding.bessel_fn
    src = model_foundations.radial_embedding.bessel_fn
    attr = {"BesselBasis": "bessel_weights", "GaussianBasis": "gaussian_weights"}.get(
        dst.__class__.__name__
    )
    if attr is None:
        return
    getattr(dst, attr).data.copy_(getattr(src, attr).data)


def load_foundations_elements(
    model: torch.nn.Module,
    model_foundations: torch.nn.Module,
    table: AtomicNumberTable,
    load_readout=False,
    use_shift=True,
    use_scale=True,
    max_L=2,
    default_dtype: Optional[torch.dtype] = None,
):
    """Dispatch loader: magnetic models use a specialized skip-tp / magmom-skip-tp
    layout; everything else falls through to the default (vanilla MACE) path.
    ``default_dtype`` is only meaningful for the default branch (the magnetic
    branch was ported from a pre-default_dtype cut of the paper repo)."""
    if model.__class__.__name__ == "MagneticNonSOCScaleShiftMACE":
        return load_foundations_elements_nonsoc(
            model,
            model_foundations,
            table,
            load_readout,
            use_shift,
            use_scale,
            default_dtype=default_dtype,
        )
    if "Magnetic" in str(model.__class__.__name__):
        return load_foundations_elements_magnetic(
            model, model_foundations, table, load_readout, use_shift, use_scale, max_L
        )
    return load_foundations_elements_default(
        model,
        model_foundations,
        table,
        load_readout,
        use_shift,
        use_scale,
        max_L,
        default_dtype=default_dtype,
    )


def _transfer_readouts(
    model: torch.nn.Module,
    model_foundations: torch.nn.Module,
    num_channels_foundation: int,
    model_heads,
) -> None:
    """Copy the (single-head) foundation readouts into every head of ``model``,
    with the fan-in rescaling that keeps each head equal to the foundation output."""
    for i, readout in enumerate(model.readouts):
        if readout.__class__.__name__ == "LinearReadoutBlock":
            model_readouts_zero_linear_weight = readout.linear.weight.clone()
            model_readouts_zero_linear_weight = (
                model_foundations.readouts[i]
                .linear.weight.view(num_channels_foundation, -1)
                .repeat(1, len(model_heads))
                .flatten()
                .clone()
            )
            readout.linear.weight = torch.nn.Parameter(
                model_readouts_zero_linear_weight
            )
        if readout.__class__.__name__ in [
            "NonLinearBiasReadoutBlock",
            "NonLinearReadoutBlock",
        ]:
            assert hasattr(readout, "linear_1") or hasattr(
                readout, "linear_mid"
            ), "Readout block must have linear_1 or linear_mid"
            if hasattr(readout, "linear_1"):
                shape_input_1 = (
                    model_foundations.readouts[i]
                    .linear_1.__dict__["irreps_out"]
                    .num_irreps
                )
                shape_output_1 = readout.linear_1.__dict__["irreps_out"].num_irreps
            else:
                raise ValueError("Readout block must have linear_1")
            if hasattr(readout, "linear_1"):
                model_readouts_one_linear_1_weight = readout.linear_1.weight.clone()
                model_readouts_one_linear_1_weight = (
                    model_foundations.readouts[i]
                    .linear_1.weight.view(num_channels_foundation, -1)
                    .repeat(1, len(model_heads))
                    .flatten()
                    .clone()
                )
                readout.linear_1.weight = torch.nn.Parameter(
                    model_readouts_one_linear_1_weight
                )
                if (
                    readout.linear_1.bias is not None
                    and readout.linear_1.bias.numel() > 0
                ):
                    model_readouts_one_linear_1_bias = (
                        model_foundations.readouts[i]
                        .linear_1.bias.view(-1)
                        .repeat(len(model_heads))
                        .clone()
                    )
                    readout.linear_1.bias = torch.nn.Parameter(
                        model_readouts_one_linear_1_bias
                    )
            if hasattr(readout, "linear_mid"):
                readout.linear_mid.weight = torch.nn.Parameter(
                    model_foundations.readouts[i]
                    .linear_mid.weight.view(
                        shape_input_1,
                        shape_input_1,
                    )
                    .repeat(len(model_heads), len(model_heads))
                    .flatten()
                    .clone()
                    / ((shape_input_1) / (shape_output_1)) ** 0.5
                )
                # if it has biases transfer them too
                if (
                    readout.linear_mid.bias is not None
                    and readout.linear_mid.bias.numel() > 0
                ):
                    readout.linear_mid.bias = torch.nn.Parameter(
                        model_foundations.readouts[i]
                        .linear_mid.bias.repeat(len(model_heads))
                        .clone()
                    )
            if hasattr(readout, "linear_2"):
                model_readouts_one_linear_2_weight = readout.linear_2.weight.clone()
                model_readouts_one_linear_2_weight = model_foundations.readouts[
                    i
                ].linear_2.weight.view(shape_input_1, -1).repeat(
                    len(model_heads), len(model_heads)
                ).flatten().clone() / (
                    ((shape_input_1) / (shape_output_1)) ** 0.5
                )
                readout.linear_2.weight = torch.nn.Parameter(
                    model_readouts_one_linear_2_weight
                )
                if (
                    readout.linear_2.bias is not None
                    and readout.linear_2.bias.numel() > 0
                ):
                    model_readouts_one_linear_2_bias = (
                        model_foundations.readouts[i]
                        .linear_2.bias.view(-1)
                        .repeat(len(model_heads))
                        .flatten()
                        .clone()
                    )
                    readout.linear_2.bias = torch.nn.Parameter(
                        model_readouts_one_linear_2_bias
                    )


def load_foundations_elements_default(
    model: torch.nn.Module,
    model_foundations: torch.nn.Module,
    table: AtomicNumberTable,
    load_readout=False,
    use_shift=True,
    use_scale=True,
    max_L=2,
    default_dtype: Optional[torch.dtype] = None,
):
    """
    Load the foundations of a model into a model for fine-tuning.
    """
    assert model_foundations.r_max == model.r_max
    z_table = AtomicNumberTable([int(z) for z in model_foundations.atomic_numbers])
    target_dtype = default_dtype or next(model.parameters()).dtype
    model_heads = model.heads
    new_z_table = table
    num_species_foundations = len(z_table.zs)
    num_channels_foundation = (
        model_foundations.node_embedding.linear.weight.shape[0]
        // num_species_foundations
    )
    indices_weights = [z_table.z_to_index(z) for z in new_z_table.zs]
    num_radial = model.radial_embedding.out_dim
    num_species = len(indices_weights)
    max_ell = model.spherical_harmonics._lmax  # pylint: disable=protected-access
    model.node_embedding.linear.weight = torch.nn.Parameter(
        model_foundations.node_embedding.linear.weight.view(
            num_species_foundations, -1
        )[indices_weights, :]
        .flatten()
        .clone()
        / (num_species_foundations / num_species) ** 0.5
    )
    if hasattr(model, "joint_embedding"):

        model_specs = model.joint_embedding.specs
        foundation_embedding = getattr(model_foundations, "joint_embedding", None)
        foundation_specs = foundation_embedding.specs if foundation_embedding else {}

        foundation_head_start = 0
        foundation_head_slices = {}
        for name, spec in foundation_specs.items():
            dim = spec["emb_dim"]
            foundation_head_slices[name] = slice(
                foundation_head_start, foundation_head_start + dim
            )
            foundation_head_start += dim

        if foundation_specs == {}:
            # if foundational model has no embeddings initialize all weights (small).
            for _, param in model.joint_embedding.named_parameters():
                torch.nn.init.uniform_(param.data, -0.05, 0.05)
        else:
            spec_names = []
            head_start = 0
            head_slices = {}
            # project[0] concatenates every spec in order, so the column offsets
            # must span all of them, matching or not (same as the foundation side).
            for name, spec in model_specs.items():
                dim = spec["emb_dim"]
                head_slices[name] = slice(head_start, head_start + dim)
                head_start += dim
            for embedding_spec in model_specs.items():

                spec_name = embedding_spec[0]
                submodule = model.joint_embedding.embedders[spec_name]
                model_params = dict(submodule.named_parameters())

                for param_name, param in model_params.items():
                    # embedding spec is the *exactly* the same as that in foundation model (ideal)
                    if embedding_spec in foundation_specs.items():

                        foundation_submodule = foundation_embedding.embedders[spec_name]
                        foundation_params = dict(
                            foundation_submodule.named_parameters()
                        )
                        param.data.copy_(foundation_params[param_name].data)
                    else:
                        torch.nn.init.uniform_(param.data, -0.05, 0.05)

                # Which specs the foundation can supply weights for.
                if embedding_spec in foundation_specs.items():
                    spec_names.append(spec_name)

            # update head.
            foundation_emd_head = model_foundations.joint_embedding.project
            foundation_head_data = foundation_emd_head[0].weight.data
            model_emb_head = model.joint_embedding.project
            head_emb_data = model_emb_head[0].weight.data

            # init and overwrite.
            torch.nn.init.uniform_(head_emb_data, -0.05, 0.05)
            for name in spec_names:
                head_emb_data[:, head_slices[name]] = foundation_head_data[
                    :, foundation_head_slices[name]
                ]

    if hasattr(model, "embedding_readout"):
        for (_, param_1), (_, param_2) in zip(
            model.embedding_readout.named_parameters(),
            model_foundations.embedding_readout.named_parameters(),
        ):
            param_1.data.copy_(
                param_2.data.reshape(-1, 1)
                .repeat(1, len(model_heads))
                .flatten()
                .clone()
            )
    _copy_radial_weights(model, model_foundations)
    for i in range(int(model.num_interactions)):
        model.interactions[i].linear_up.weight = torch.nn.Parameter(
            model_foundations.interactions[i].linear_up.weight.clone()
        )
        model.interactions[i].set_avg_num_neighbors(
            model_foundations.interactions[i].avg_num_neighbors
        )

        for (_, param_1), (_, param_2) in zip(
            model.interactions[i].conv_tp_weights.named_parameters(),
            model_foundations.interactions[i].conv_tp_weights.named_parameters(),
        ):
            if param_1.shape == param_2.shape:
                param_1.data.copy_(param_2.data)
            else:
                param_1.data.copy_(param_2.data[: (num_radial + 2 * num_species), ...])
        if hasattr(model.interactions[i], "linear"):
            model.interactions[i].linear.weight = torch.nn.Parameter(
                model_foundations.interactions[i].linear.weight.clone()
            )
        if hasattr(model.interactions[i], "linear_1"):
            model.interactions[i].linear_1.weight = torch.nn.Parameter(
                model_foundations.interactions[i].linear_1.weight.clone()
            )
        if hasattr(model.interactions[i], "linear_2"):
            model.interactions[i].linear_2.weight = torch.nn.Parameter(
                model_foundations.interactions[i].linear_2.weight.clone()
            )
        if hasattr(model.interactions[i], "linear_res"):
            model.interactions[i].linear_res.weight = torch.nn.Parameter(
                model_foundations.interactions[i].linear_res.weight.clone()
            )
        if hasattr(model.interactions[i], "source_embedding"):
            model.interactions[i].source_embedding.weight = torch.nn.Parameter(
                model_foundations.interactions[i]
                .source_embedding.weight.view(num_species_foundations, -1)[
                    indices_weights, :
                ]
                .flatten()
                .clone()
                / (num_species_foundations / num_species) ** 0.5
            )
        if hasattr(model.interactions[i], "target_embedding"):
            model.interactions[i].target_embedding.weight = torch.nn.Parameter(
                model_foundations.interactions[i]
                .target_embedding.weight.view(num_species_foundations, -1)[
                    indices_weights, :
                ]
                .flatten()
                .clone()
                / (num_species_foundations / num_species) ** 0.5
            )
        if hasattr(model.interactions[i], "alpha"):
            model.interactions[i].alpha = torch.nn.Parameter(
                model_foundations.interactions[i].alpha.clone()
            )
        if hasattr(model.interactions[i], "beta"):
            model.interactions[i].beta = torch.nn.Parameter(
                model_foundations.interactions[i].beta.clone()
            )
        if model.interactions[i].__class__.__name__ in [
            "RealAgnosticResidualInteractionBlock",
            "RealAgnosticDensityResidualInteractionBlock",
        ]:
            model.interactions[i].skip_tp.weight = torch.nn.Parameter(
                model_foundations.interactions[i]
                .skip_tp.weight.reshape(
                    num_channels_foundation,
                    num_species_foundations,
                    num_channels_foundation,
                )[:, indices_weights, :]
                .flatten()
                .clone()
                / (num_species_foundations / num_species) ** 0.5
            )
        elif model.interactions[i].__class__.__name__ in [
            "RealAgnosticResidualNonLinearInteractionBlock",
        ]:
            model.interactions[i].skip_tp.weight = torch.nn.Parameter(
                model_foundations.interactions[i].skip_tp.weight
            )
        else:
            model.interactions[i].skip_tp.weight = torch.nn.Parameter(
                model_foundations.interactions[i]
                .skip_tp.weight.reshape(
                    num_channels_foundation,
                    (max_ell + 1),
                    num_species_foundations,
                    num_channels_foundation,
                )[:, :, indices_weights, :]
                .flatten()
                .clone()
                / (num_species_foundations / num_species) ** 0.5
            )
        if hasattr(model.interactions[i], "density_fn"):
            for (_, param_1), (_, param_2) in zip(
                model.interactions[i].density_fn.named_parameters(),
                model_foundations.interactions[i].density_fn.named_parameters(),
            ):
                param_1.data.copy_(param_2.data)

    # Transferring products
    for i, product in enumerate(model.products):
        indices_weights_prod = indices_weights
        if hasattr(product, "use_agnostic_product"):
            if product.use_agnostic_product:
                indices_weights_prod = [0]
        max_range = max_L + 1 if i < len(model.products) - 1 else 1
        for j in range(max_range):  # Assuming 3 contractions in symmetric_contractions
            product.symmetric_contractions.contractions[j].weights_max = (
                torch.nn.Parameter(
                    model_foundations.products[i]
                    .symmetric_contractions.contractions[j]
                    .weights_max[indices_weights_prod, :, :]
                    .clone()
                )
            )

            target_weights = product.symmetric_contractions.contractions[j].weights
            source_weights = (
                model_foundations.products[i]
                .symmetric_contractions.contractions[j]
                .weights
            )
            for k, _ in enumerate(target_weights):
                target_weights[k] = torch.nn.Parameter(
                    source_weights[k][indices_weights_prod, :, :].clone()
                )
        product.linear.weight = torch.nn.Parameter(
            model_foundations.products[i].linear.weight.clone()
        )

    if load_readout:
        _transfer_readouts(
            model, model_foundations, num_channels_foundation, model_heads
        )
    _handled_attrs = {"interactions", "products", "readouts"}
    for attr_name, module in model.named_children():
        if attr_name in _handled_attrs:
            continue
        if attr_name not in dict(model_foundations.named_children()):
            continue
        submodules = (
            list(zip(module, model_foundations.__dict__["_modules"][attr_name]))
            if isinstance(module, torch.nn.ModuleList)
            else [(module, getattr(model_foundations, attr_name))]
        )
        for sub_new, sub_found in submodules:
            for emb_name in ("source_embedding", "target_embedding"):
                if not hasattr(sub_new, emb_name):
                    continue
                emb_new = getattr(sub_new, emb_name)
                emb_found = getattr(sub_found, emb_name)
                if (
                    hasattr(emb_new, "weight")
                    and hasattr(emb_found, "weight")
                    and emb_found.weight.shape[0]
                    == num_species_foundations * num_channels_foundation
                    and emb_new.weight.shape[0] == num_species * num_channels_foundation
                ):
                    emb_new.weight = torch.nn.Parameter(
                        emb_found.weight.view(num_species_foundations, -1)[
                            indices_weights, :
                        ]
                        .flatten()
                        .clone()
                        / (num_species_foundations / num_species) ** 0.5
                    )

    if getattr(model_foundations, "scale_shift", None) is not None and hasattr(
        model, "scale_shift"
    ):
        if use_scale:
            model.scale_shift.scale = model_foundations.scale_shift.scale.repeat(
                len(model_heads)
            ).clone()
        if use_shift:
            model.scale_shift.shift = model_foundations.scale_shift.shift.repeat(
                len(model_heads)
            ).clone()

    model_state = model.state_dict()
    foundation_state = model_foundations.state_dict()
    for name, param in foundation_state.items():
        if name not in model_state:
            continue
        if not load_readout and name.startswith("readouts."):
            continue
        # The joint embedding is transferred spec by spec above. A blanket copy
        # would re-align the head columns positionally and silently undo it
        # whenever both heads happen to have the same shape.
        if name.startswith("joint_embedding."):
            continue
        if model_state[name].shape != param.shape:
            continue
        model_state[name].copy_(param)

    model.to(target_dtype)

    return model


# Per-head tensors of the non-SOC model that a single-head foundation must be
# broadcast into (head axis in parentheses), on top of the readouts and scale_shift.
_NONSOC_PER_HEAD_TENSORS = {
    "onebody_magmombasis_coeffs": -1,  # (num_elements, num_basis, heads)
    "one_body_magmom_const_correction": -1,  # (num_elements, heads)
}


def load_foundations_elements_nonsoc(
    model: torch.nn.Module,
    model_foundations: torch.nn.Module,
    table: AtomicNumberTable,
    load_readout=False,
    use_shift=True,
    use_scale=True,
    default_dtype: Optional[torch.dtype] = None,
):
    """Load a MagneticNonSOCScaleShiftMACE foundation into a fine-tuning model.

    The non-SOC layers (plain ``RealAgnosticDensityInteractionBlock`` first, then the
    ``MagneticRealAgnosticNonSpinOrbitCoupledDensityInteractionBlock`` with the merged
    ``NonSOCSymmetricContraction`` weights) share none of the tensor layout the SOC
    loader assumes, so this loader copies every tensor by name instead. That is only
    exact when the fine-tuning model keeps the foundation element table, hence the
    check below: with a smaller table every element-indexed tensor would be silently
    skipped and the model would start from random weights.

    Head handling for multihead fine-tuning: the readouts, ``scale_shift`` and the
    one-body magmom tensors of the (single-head) foundation are repeated into every
    head so each head initially reproduces the foundation energies. The per-head
    ``atomic_energies`` are deliberately left alone: run_train assembles them per head
    (E0s of the fine-tuning data, ``--E0s=foundation`` for the pt_head) before the
    model is built.
    """
    assert model_foundations.r_max == model.r_max
    foundation_zs = [int(z) for z in model_foundations.atomic_numbers]
    if [int(z) for z in table.zs] != foundation_zs:
        raise ValueError(
            "Fine-tuning a MagneticNonSOCScaleShiftMACE foundation requires the "
            "foundation element table: the non-SOC contraction weights are indexed by "
            f"element and are not re-mapped. Got {len(table.zs)} elements "
            f"{list(table.zs)} but the foundation has {len(foundation_zs)}; pass "
            "--foundation_model_elements=True."
        )
    foundation_heads = list(getattr(model_foundations, "heads", ["Default"]))
    if len(foundation_heads) != 1:
        raise ValueError(
            "Fine-tuning from a multi-head MagneticNonSOCScaleShiftMACE foundation is "
            f"not supported (foundation heads: {foundation_heads})."
        )
    model_heads = model.heads
    n_heads = len(model_heads)
    target_dtype = default_dtype or next(model.parameters()).dtype

    if hasattr(model, "m_max") and not torch.equal(
        model.m_max.cpu().to(torch.float64),
        model_foundations.m_max.cpu().to(torch.float64),
    ):
        logging.warning(
            "The --m_max of the fine-tuning model differs from the foundation's; the "
            "foundation values are kept (its magnetic basis was trained with them)."
        )

    skip_prefixes = ("readouts.", "scale_shift.")
    skip_names = {"atomic_energies_fn.atomic_energies"}
    model_state = model.state_dict()
    foundation_state = model_foundations.state_dict()
    transferred, skipped = [], []
    for name, param in foundation_state.items():
        if name.startswith(skip_prefixes) or name in skip_names:
            continue
        if name not in model_state:
            skipped.append(f"{name} (absent in the new model)")
            continue
        target = model_state[name]
        head_axis = _NONSOC_PER_HEAD_TENSORS.get(name)
        if target.shape == param.shape:
            target.copy_(param)
        elif (
            head_axis is not None
            and param.shape[head_axis] == 1
            and target.shape[head_axis] == n_heads
            and target.shape[:head_axis] == param.shape[:head_axis]
        ):
            target.copy_(param.expand(target.shape))
        else:
            skipped.append(f"{name} {tuple(param.shape)} -> {tuple(target.shape)}")
            continue
        transferred.append(name)
    if skipped:
        raise ValueError(
            "Foundation tensors could not be transferred into the fine-tuning model "
            "(architecture mismatch between --foundation_model and the CLI flags?): "
            + "; ".join(skipped)
        )

    if getattr(model_foundations, "scale_shift", None) is not None and hasattr(
        model, "scale_shift"
    ):
        if use_scale:
            model.scale_shift.scale = model_foundations.scale_shift.scale.repeat(
                n_heads
            ).clone()
        if use_shift:
            model.scale_shift.shift = model_foundations.scale_shift.shift.repeat(
                n_heads
            ).clone()

    if load_readout:
        num_channels_foundation = model_foundations.node_embedding.linear.weight.shape[
            0
        ] // len(foundation_zs)
        _transfer_readouts(
            model, model_foundations, num_channels_foundation, model_heads
        )

    logging.info(
        f"Transferred {len(transferred)} tensors from the non-SOC foundation model "
        f"into {n_heads} head(s)"
    )
    model.to(target_dtype)
    return model


def load_foundations_elements_magnetic(
    model: torch.nn.Module,
    model_foundations: torch.nn.Module,
    table: AtomicNumberTable,
    load_readout=False,
    use_shift=True,
    use_scale=True,
    max_L=2,
):
    """
    Load the foundations of a model into a model for fine-tuning.
    """
    assert model_foundations.r_max == model.r_max
    z_table = AtomicNumberTable([int(z) for z in model_foundations.atomic_numbers])
    model_heads = model.heads
    new_z_table = table
    num_species_foundations = len(z_table.zs)

    num_channels_foundation = (
        model_foundations.node_embedding.linear.weight.shape[0]
        // num_species_foundations
    )
    indices_weights = [z_table.z_to_index(z) for z in new_z_table.zs]
    num_radial = model.radial_embedding.out_dim
    num_mag_radial = model.mag_radial_embedding.num_basis
    num_species = len(indices_weights)
    model.node_embedding.linear.weight = torch.nn.Parameter(
        model_foundations.node_embedding.linear.weight.view(
            num_species_foundations, -1
        )[indices_weights, :]
        .flatten()
        .clone()
        / (num_species_foundations / num_species) ** 0.5
    )
    _copy_radial_weights(model, model_foundations)

    for i in range(int(model.num_interactions)):
        model.interactions[i].linear_up.weight = torch.nn.Parameter(
            model_foundations.interactions[i].linear_up.weight.clone()
        )
        model.interactions[i].set_avg_num_neighbors(
            model_foundations.interactions[i].avg_num_neighbors
        )
        for j in range(4):  # Assuming 4 layers in conv_tp_weights,
            layer_name = f"layer{j}"
            if j == 0:
                getattr(model.interactions[i].conv_tp_weights, layer_name).weight = (
                    torch.nn.Parameter(
                        getattr(
                            model_foundations.interactions[i].conv_tp_weights,
                            layer_name,
                        )
                        .weight[: num_radial + num_mag_radial, :]
                        .clone()
                    )
                )
            else:
                getattr(model.interactions[i].conv_tp_weights, layer_name).weight = (
                    torch.nn.Parameter(
                        getattr(
                            model_foundations.interactions[i].conv_tp_weights,
                            layer_name,
                        ).weight.clone()
                    )
                )

        # conv_tp_weights_magmom
        for j in range(1):  # Assuming 4 layers in conv_tp_weights,
            layer_name = f"layer{j}"
            if j == 0:
                getattr(
                    model.interactions[i].conv_tp_weights_magmom, layer_name
                ).weight = torch.nn.Parameter(
                    getattr(
                        model_foundations.interactions[i].conv_tp_weights_magmom,
                        layer_name,
                    ).weight.clone()
                )
            else:
                getattr(
                    model.interactions[i].conv_tp_weights_magmom, layer_name
                ).weight = torch.nn.Parameter(
                    getattr(
                        model_foundations.interactions[i].conv_tp_weights_magmom,
                        layer_name,
                    ).weight.clone()
                )

        model.interactions[i].magmom_linear.weight = torch.nn.Parameter(
            model_foundations.interactions[i].magmom_linear.weight.clone()
        )
        if model.interactions[i].__class__.__name__ in [
            "MagneticRealAgnosticSpinOrbitCoupledDensityInteractionBlock",
        ]:
            model.interactions[i].magmom_skip_tp.weight = torch.nn.Parameter(
                model_foundations.interactions[i]
                .magmom_skip_tp.weight.flatten()
                .clone()
            )
        else:
            model.interactions[i].skip_tp.weight = torch.nn.Parameter(
                model_foundations.interactions[i].skip_tp.weight.flatten().clone()
            )
        if model.interactions[i].__class__.__name__ in [
            "MagneticRealAgnosticSpinOrbitCoupledDensityInteractionBlock",
            "MagneticRealAgnosticResidueSpinOrbitCoupledDensityInteractionBlock",
        ]:
            # Assuming only 1 layer in density_fn
            getattr(model.interactions[i].density_fn, "layer0").weight = (
                torch.nn.Parameter(
                    getattr(
                        model_foundations.interactions[i].density_fn,
                        "layer0",
                    ).weight.clone()
                )
            )
    # Transferring products
    for i in range(2):  # Assuming 2 products modules
        max_range = max_L + 1 if i == 0 else 1
        for j in range(max_range):  # Assuming 3 contractions in symmetric_contractions
            model.products[i].symmetric_contractions.contractions[j].weights_max = (
                torch.nn.Parameter(
                    model_foundations.products[i]
                    .symmetric_contractions.contractions[j]
                    .weights_max[indices_weights, :, :]
                    .clone()
                )
            )

            for k in range(2):  # Assuming 2 weights in each contraction
                model.products[i].symmetric_contractions.contractions[j].weights[k] = (
                    torch.nn.Parameter(
                        model_foundations.products[i]
                        .symmetric_contractions.contractions[j]
                        .weights[k][indices_weights, :, :]
                        .clone()
                    )
                )

        model.products[i].conv_tp.weight = torch.nn.Parameter(
            model_foundations.products[i].conv_tp.weight.clone()
        )
        for j in range(4):  # Assuming 4 layers in conv_tp_weights,
            layer_name = f"layer{j}"
            if j == 0:
                getattr(model.products[i].conv_tp_weights, layer_name).weight = (
                    torch.nn.Parameter(
                        getattr(
                            model_foundations.products[i].conv_tp_weights,
                            layer_name,
                        )
                        .weight[:num_mag_radial, :]
                        .clone()
                    )
                )
            else:
                getattr(model.products[i].conv_tp_weights, layer_name).weight = (
                    torch.nn.Parameter(
                        getattr(
                            model_foundations.products[i].conv_tp_weights,
                            layer_name,
                        ).weight.clone()
                    )
                )
        model.products[i].linear_ori.weight = torch.nn.Parameter(
            model_foundations.products[i].linear_ori.weight.clone()
        )
        model.products[i].linear.weight = torch.nn.Parameter(
            model_foundations.products[i].linear.weight.clone()
        )

    if load_readout:
        # Transferring readouts
        model_readouts_zero_linear_weight = model.readouts[0].linear.weight.clone()
        model_readouts_zero_linear_weight = (
            model_foundations.readouts[0]
            .linear.weight.view(num_channels_foundation, -1)
            .repeat(1, len(model_heads))
            .flatten()
            .clone()
        )
        model.readouts[0].linear.weight = torch.nn.Parameter(
            model_readouts_zero_linear_weight
        )

        shape_input_1 = (
            model_foundations.readouts[1].linear_1.__dict__["irreps_out"].num_irreps
        )
        shape_output_1 = model.readouts[1].linear_1.__dict__["irreps_out"].num_irreps
        model_readouts_one_linear_1_weight = model.readouts[1].linear_1.weight.clone()
        model_readouts_one_linear_1_weight = (
            model_foundations.readouts[1]
            .linear_1.weight.view(num_channels_foundation, -1)
            .repeat(1, len(model_heads))
            .flatten()
            .clone()
        )
        model.readouts[1].linear_1.weight = torch.nn.Parameter(
            model_readouts_one_linear_1_weight
        )
        model_readouts_one_linear_2_weight = model.readouts[1].linear_2.weight.clone()
        model_readouts_one_linear_2_weight = model_foundations.readouts[
            1
        ].linear_2.weight.view(shape_input_1, -1).repeat(
            len(model_heads), len(model_heads)
        ).flatten().clone() / (
            ((shape_input_1) / (shape_output_1)) ** 0.5
        )
        model.readouts[1].linear_2.weight = torch.nn.Parameter(
            model_readouts_one_linear_2_weight
        )
    if model_foundations.scale_shift is not None:
        if use_scale:
            model.scale_shift.scale = model_foundations.scale_shift.scale.repeat(
                len(model_heads)
            ).clone()
        if use_shift:
            model.scale_shift.shift = model_foundations.scale_shift.shift.repeat(
                len(model_heads)
            ).clone()
    return model


def load_foundations(
    model,
    model_foundations,
    include_readouts: bool = False,
):
    model_state = model.state_dict()
    foundation_state = model_foundations.state_dict()
    for name, param in foundation_state.items():
        if name not in model_state:
            continue
        if not include_readouts and name.startswith("readouts."):
            continue
        if model_state[name].shape != param.shape:
            continue
        model_state[name].copy_(param)
    return model


def load_foundations_mdp(
    model: torch.nn.Module,
    model_foundations: torch.nn.Module,
    table: AtomicNumberTable,
    max_L: int = 2,
):
    """
    Transfer weights from a pretrained AtomicDielectricMACE to a new one,
    with species remapping for a (possibly smaller) element set.

    Unlike load_foundations_elements, this handles higher-order irreps
    in skip_tp and transfers all angular momentum channels in products.
    """
    assert model_foundations.r_max == model.r_max
    z_table = AtomicNumberTable([int(z) for z in model_foundations.atomic_numbers])
    num_species_foundations = len(z_table.zs)
    num_channels_foundation = (
        model_foundations.node_embedding.linear.weight.shape[0]
        // num_species_foundations
    )
    indices_weights = [z_table.z_to_index(z) for z in table.zs]
    num_species = len(indices_weights)
    num_radial = model.radial_embedding.out_dim
    species_scale = (num_species_foundations / num_species) ** 0.5

    # --- Node embedding: extract rows for target species ---
    model.node_embedding.linear.weight = torch.nn.Parameter(
        model_foundations.node_embedding.linear.weight.view(
            num_species_foundations, -1
        )[indices_weights, :]
        .flatten()
        .clone()
        / species_scale
    )

    # --- Radial embedding ---
    _copy_radial_weights(model, model_foundations)

    # --- Interactions ---
    for i in range(int(model.num_interactions)):
        model.interactions[i].linear_up.weight = torch.nn.Parameter(
            model_foundations.interactions[i].linear_up.weight.clone()
        )
        model.interactions[i].set_avg_num_neighbors(
            model_foundations.interactions[i].avg_num_neighbors
        )

        for (_, param_1), (_, param_2) in zip(
            model.interactions[i].conv_tp_weights.named_parameters(),
            model_foundations.interactions[i].conv_tp_weights.named_parameters(),
        ):
            if param_1.shape == param_2.shape:
                param_1.data.copy_(param_2.data)
            else:
                param_1.data.copy_(param_2.data[: (num_radial + 2 * num_species), ...])
        if hasattr(model.interactions[i], "linear"):
            model.interactions[i].linear.weight = torch.nn.Parameter(
                model_foundations.interactions[i].linear.weight.clone()
            )
        if hasattr(model.interactions[i], "linear_1"):
            model.interactions[i].linear_1.weight = torch.nn.Parameter(
                model_foundations.interactions[i].linear_1.weight.clone()
            )
        if hasattr(model.interactions[i], "linear_2"):
            model.interactions[i].linear_2.weight = torch.nn.Parameter(
                model_foundations.interactions[i].linear_2.weight.clone()
            )
        if hasattr(model.interactions[i], "linear_res"):
            model.interactions[i].linear_res.weight = torch.nn.Parameter(
                model_foundations.interactions[i].linear_res.weight.clone()
            )
        if hasattr(model.interactions[i], "source_embedding"):
            model.interactions[i].source_embedding.weight = torch.nn.Parameter(
                model_foundations.interactions[i]
                .source_embedding.weight.view(num_species_foundations, -1)[
                    indices_weights, :
                ]
                .flatten()
                .clone()
                / species_scale
            )
        if hasattr(model.interactions[i], "target_embedding"):
            model.interactions[i].target_embedding.weight = torch.nn.Parameter(
                model_foundations.interactions[i]
                .target_embedding.weight.view(num_species_foundations, -1)[
                    indices_weights, :
                ]
                .flatten()
                .clone()
                / species_scale
            )
        if hasattr(model.interactions[i], "alpha"):
            model.interactions[i].alpha = torch.nn.Parameter(
                model_foundations.interactions[i].alpha.clone()
            )
        if hasattr(model.interactions[i], "beta"):
            model.interactions[i].beta = torch.nn.Parameter(
                model_foundations.interactions[i].beta.clone()
            )
        # skip_tp: use general reshape [-1, N_sp, N_ch] to handle higher-order irreps
        if model.interactions[i].__class__.__name__ in [
            "RealAgnosticResidualNonLinearInteractionBlock",
        ]:
            model.interactions[i].skip_tp.weight = torch.nn.Parameter(
                model_foundations.interactions[i].skip_tp.weight
            )
        else:
            foundation_skip = model_foundations.interactions[i].skip_tp.weight
            rest_dim = foundation_skip.numel() // (
                num_species_foundations * num_channels_foundation
            )
            model.interactions[i].skip_tp.weight = torch.nn.Parameter(
                foundation_skip.reshape(
                    rest_dim, num_species_foundations, num_channels_foundation
                )[:, indices_weights, :]
                .flatten()
                .clone()
                / species_scale
            )
        if hasattr(model.interactions[i], "density_fn"):
            for (_, param_1), (_, param_2) in zip(
                model.interactions[i].density_fn.named_parameters(),
                model_foundations.interactions[i].density_fn.named_parameters(),
            ):
                param_1.data.copy_(param_2.data)

    # --- Products: transfer ALL angular momentum channels (not just L=0 for last) ---
    for i, product in enumerate(model.products):
        indices_weights_prod = indices_weights
        if hasattr(product, "use_agnostic_product"):
            if product.use_agnostic_product:
                indices_weights_prod = [0]
        # MDP readouts use all irreps, so always transfer all contractions
        max_range = max_L + 1
        for j in range(max_range):
            product.symmetric_contractions.contractions[j].weights_max = (
                torch.nn.Parameter(
                    model_foundations.products[i]
                    .symmetric_contractions.contractions[j]
                    .weights_max[indices_weights_prod, :, :]
                    .clone()
                )
            )
            target_weights = product.symmetric_contractions.contractions[j].weights
            source_weights = (
                model_foundations.products[i]
                .symmetric_contractions.contractions[j]
                .weights
            )
            for k, _ in enumerate(target_weights):
                target_weights[k] = torch.nn.Parameter(
                    source_weights[k][indices_weights_prod, :, :].clone()
                )
        product.linear.weight = torch.nn.Parameter(
            model_foundations.products[i].linear.weight.clone()
        )

    # --- Readouts: copy matching params by name+shape (species-independent) ---
    model_state = model.state_dict()
    foundation_state = model_foundations.state_dict()
    for name, param in foundation_state.items():
        if not name.startswith("readouts."):
            continue
        if name not in model_state:
            continue
        if model_state[name].shape != param.shape:
            continue
        model_state[name].copy_(param)

    return model
