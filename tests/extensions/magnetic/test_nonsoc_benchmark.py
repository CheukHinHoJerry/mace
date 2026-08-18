import ase.io
import numpy as np
import pytest
import torch
from e3nn import o3
from scipy.spatial.transform import Rotation as Rot

from mace.cli.run_train import run as mace_run
from mace.modules import interaction_classes
from mace.modules.extensions import MagneticNonSOCScaleShiftMACE
from mace.tools.arg_parser import build_default_arg_parser
from mace.tools.torch_tools import default_dtype

from .test_nonsoc_equivariance import _batch

_RMAX = 6.0


# ----------------------------------------------------------
# Reference configuration
# ----------------------------------------------------------
# A plain (non-magnetic) interaction block followed by the magnetic non-SOC
# block, with SCALAR hidden irreps. The scalar hidden irreps are load-bearing:
# layer 1 receives hidden_irreps as its node features, and the magnetic message
# is built by conv_tp_m = TensorProduct(node_feats_irreps, magmom_node_attrs_irreps)
# whose output becomes the spin axis of A_msg. With scalar node features that
# product is a channel mix of the magmom attributes, so the spin axis is spin
# only; with lmax > 0 node features it also carries spatial content.


def build_reference(hidden="16x0e", correlation=3, max_ell=3, max_m_ell=1):
    """The reference non-SOC architecture, at a channel count small enough for tests."""
    torch.manual_seed(1)
    return MagneticNonSOCScaleShiftMACE(
        r_max=_RMAX,
        num_bessel=8,
        num_polynomial_cutoff=5,
        max_ell=max_ell,
        interaction_cls=interaction_classes[
            "MagneticRealAgnosticNonSpinOrbitCoupledDensityInteractionBlock"
        ],
        interaction_cls_first=interaction_classes[
            "RealAgnosticDensityInteractionBlock"
        ],
        contraction_cls_first="SymmetricContraction",
        contraction_cls="NonSOCSymmetricContraction",
        num_interactions=2,
        num_elements=1,
        hidden_irreps=(hidden if isinstance(hidden, list) else o3.Irreps(hidden)),
        MLP_irreps=o3.Irreps("16x0e"),
        atomic_energies=np.zeros(1),
        avg_num_neighbors=8.0,
        atomic_numbers=[26],
        correlation=correlation,
        gate=torch.nn.functional.silu,
        atomic_inter_shift=0.0,
        atomic_inter_scale=1.0,
        m_max=[3.0],
        num_mag_radial_basis=8,
        num_mag_radial_basis_one_body=8,
        max_m_ell=max_m_ell,
        use_magmom_one_body=False,
    )


def _state(seed=0, n=8):
    g = torch.Generator().manual_seed(seed)
    return (
        torch.randn(n, 3, dtype=torch.float64, generator=g) * 2.5,
        torch.randn(n, 3, dtype=torch.float64, generator=g) * 0.6,
    )


def _energy(model, pos, mag):
    return float(
        model(
            _batch(pos, mag), training=False, compute_force=False, compute_stress=False
        )["energy"]
    )


# ----------------------------------------------------------
# The reference architecture is correct
# ----------------------------------------------------------
@pytest.mark.parametrize(
    "name, transform",
    [
        ("E(Rr,m)", lambda r, m, R, S: (r @ R.T, m)),
        ("E(r,Sm)", lambda r, m, R, S: (r, m @ S.T)),
        ("E(-r,m)", lambda r, m, R, S: (-r, m)),
        ("E(r,-m)", lambda r, m, R, S: (r, -m)),
        ("independent R and S", lambda r, m, R, S: (-r @ R.T, -m @ S.T)),
    ],
)
@pytest.mark.parametrize("max_m_ell", [1, 3])
def test_reference_satisfies_the_four_identities(name, transform, max_m_ell):
    """E(Rr,m) = E(r,Sm) = E(-r,m) = E(r,-m) = E(r,m) on the supported configuration."""
    with default_dtype(torch.float64):
        model = build_reference(max_m_ell=max_m_ell)
        pos, mag = _state()
        R = torch.tensor(Rot.random(rng=1).as_matrix(), dtype=torch.float64)
        S = torch.tensor(Rot.random(rng=2).as_matrix(), dtype=torch.float64)
        e0 = _energy(model, pos, mag)
        e1 = _energy(model, *transform(pos, mag, R, S))
        assert (
            abs(e0 - e1) < 1e-9
        ), f"{name}: violated by {abs(e0 - e1):.3e} (E = {e0:.6f})"


def test_reference_magnetic_layer_is_on_the_spin_pure_path():
    """The magnetic layer must stay on the scalar-node path, which keeps its spin axis pure."""
    with default_dtype(torch.float64):
        model = build_reference()
        mag_layers = [b for b in model.interactions if hasattr(b, "conv_tp_m")]
        assert mag_layers, "reference should contain a magnetic non-SOC interaction"
        for blk in mag_layers:
            assert o3.Irreps(blk.node_feats_irreps).lmax == 0, (
                f"magnetic layer receives node_feats={blk.node_feats_irreps} with "
                f"lmax > 0; the spin axis of A_msg would carry spatial content"
            )
            assert blk._project_messages is False  # pylint: disable=protected-access


def test_magnetic_message_is_spin_pure_in_the_reference():
    """Under a purely SPATIAL rotation the magnetic message must not change at all."""
    with default_dtype(torch.float64):
        model = build_reference()
        blk = [b for b in model.interactions if hasattr(b, "conv_tp_m")][0]
        grabbed = {}
        handle = blk.conv_tp_m.register_forward_hook(
            lambda mod, inp, out: grabbed.__setitem__("m", out.detach().clone())
        )
        try:
            pos, mag = _state()
            R = torch.tensor(Rot.random(rng=1).as_matrix(), dtype=torch.float64)
            _energy(model, pos, mag)
            m0 = grabbed["m"]
            _energy(model, pos @ R.T, mag)
            m1 = grabbed["m"]
        finally:
            handle.remove()
        d = (m0 - m1).abs().max().item()
        assert (
            d < 1e-12
        ), f"magnetic message changed by {d:.3e} under a spatial rotation"


# ----------------------------------------------------------
# Non-scalar hidden irreps are rejected
# ----------------------------------------------------------
@pytest.mark.parametrize(
    "hidden",
    [
        "16x0e+16x1o",
        "16x0e+16x1o+16x2e",
        # per-layer: an L>0 FIRST layer still feeds L>0 into the magnetic layer,
        # even though that layer's own target is scalar
        ["16x0e+16x1o", "16x0e"],
    ],
)
def test_nonscalar_hidden_irreps_are_rejected(hidden):
    """Only scalar hidden features are supported, and the model must say so loudly.

    A magnetic non-SOC layer builds its magnetic message from the node features
    and the magmom attributes, then uses it as the spin axis of A_msg. Non-scalar
    node features put spatial content on that axis, which the contraction reduces
    against the magmom CG basis, and the energy stops being invariant under a
    spatial rotation.

    Checking only the final target would not catch this: with two interactions the
    last layer is forced to scalars anyway while the preceding one still emits
    lmax > 0 features into the next conv_tp_m.
    """
    with default_dtype(torch.float64):
        with pytest.raises(AssertionError, match="SCALAR node features"):
            build_reference(hidden=hidden)


# ----------------------------------------------------------
# Training CLI
# ----------------------------------------------------------
_nonsoc_params = {
    "name": "MACE",
    "valid_fraction": 0.05,
    "energy_weight": 1.0,
    "forces_weight": 10.0,
    "model": "MagneticNonSOCScaleShiftMACE",
    "interaction_first": "RealAgnosticDensityInteractionBlock",
    "interaction": "MagneticRealAgnosticNonSpinOrbitCoupledDensityInteractionBlock",
    "hidden_irreps": "32x0e",
    "r_max": 3.5,
    "m_max": 10.0,
    "batch_size": 5,
    "max_num_epochs": 2,
    "device": "cpu",
    "seed": 5,
    "energy_key": "REF_energy",
    "forces_key": "REF_forces",
    "magmom_key": "REF_magmom",
    "eval_interval": 1,
    "max_m_ell": 1,
    "correlation": 2,
}


@pytest.fixture(name="nonsoc_configs")
def fixture_nonsoc_configs():
    """Tiny synthetic magnetic dataset, enough to drive the training CLI."""
    from ase.atoms import Atoms

    base = Atoms(
        numbers=[26, 26],
        positions=[[0, 0, 0], [0, 0, 2.0]],
        cell=[6.0] * 3,
        pbc=[True] * 3,
    )
    isolated = Atoms(numbers=[26], positions=[[0, 0, 0]], cell=[6] * 3)
    isolated.info["REF_energy"] = 0.0
    isolated.info["config_type"] = "IsolatedAtom"
    isolated.arrays["REF_magmom"] = np.array([[0.0, 0.0, 2.2]])

    rng = np.random.default_rng(5)
    configs = [isolated]
    for _ in range(20):
        c = base.copy()
        c.positions += rng.normal(0, 0.05, size=c.positions.shape)
        c.info["REF_energy"] = rng.normal(0.0, 0.01)
        c.new_array("REF_forces", rng.normal(0, 0.01, size=c.positions.shape))
        c.new_array("REF_magmom", np.tile([[0.0, 0.0, 2.2]], (len(c), 1)))
        configs.append(c)
    return configs


def test_run_train_nonsoc_mace(tmp_path, nonsoc_configs):
    """The model must be reachable from run_train, not only by direct construction.

    Guards both registration points: the --model choices in arg_parser and the
    branch in model_script_utils._build_model.
    """
    ase.io.write(tmp_path / "fit.xyz", nonsoc_configs)

    params = _nonsoc_params.copy()
    params["checkpoints_dir"] = str(tmp_path)
    params["model_dir"] = str(tmp_path)
    params["results_dir"] = str(tmp_path)
    params["log_dir"] = str(tmp_path)
    params["train_file"] = str(tmp_path / "fit.xyz")

    args = build_default_arg_parser().parse_args(
        [f"--{k}={v}" if v is not None else f"--{k}" for k, v in params.items()]
    )
    assert args.model == "MagneticNonSOCScaleShiftMACE"

    mace_run(args)
    assert (tmp_path / "MACE.model").exists(), "training produced no model file"


# ----------------------------------------------------------
# One-body optimizer param group
# ----------------------------------------------------------
@pytest.mark.parametrize(
    "lr_factor, weight_decay", [(1.0, 0.0), (0.1, 0.05), (0.5, 0.0), (1.0, 0.2)]
)
def test_one_body_param_group_honours_lr_and_weight_decay(lr_factor, weight_decay):
    """The one-body head gets its own weight decay and optional LR scaling.

    It is a small per-element energy offset that can outrun the rest of the
    model, so it is separable from the global optimizer settings.
    """
    import argparse

    from mace.tools.scripts_utils import get_params_options

    with default_dtype(torch.float64):
        model = build_reference(hidden="16x0e")
        model.use_magmom_one_body = True
        model.onebody_magmombasis_coeffs = torch.nn.Parameter(torch.zeros(1, 8, 1))

        # take every default from the real parser, so this does not drift as
        # get_params_options grows new knobs
        # pylint: disable=protected-access
        defaults = {a.dest: a.default for a in build_default_arg_parser()._actions}
        defaults.pop("help", None)
        args = argparse.Namespace(**defaults)
        args.lr = 0.01
        args.train_one_body_contribution = True
        args.one_body_lr_factor = lr_factor
        args.one_body_weight_decay = weight_decay
        options = get_params_options(args, model)
        groups = {g.get("name"): g for g in options["params"] if isinstance(g, dict)}
        assert "onebody_magmombasis_coeffs" in groups, "one-body group missing"
        group = groups["onebody_magmombasis_coeffs"]

        assert group["weight_decay"] == weight_decay
        if lr_factor == 1.0:
            assert "lr" not in group, "LR should be left to the global setting when 1.0"
        else:
            assert group["lr"] == pytest.approx(lr_factor * args.lr)
