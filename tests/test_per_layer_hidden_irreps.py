"""Per-layer hidden_irreps: a model may take one Irreps per interaction layer.

Guarantees: (1) a single Irreps / a uniform list reproduce the legacy model byte-for-byte;
(2) a genuinely per-layer model builds, runs, and stays rotation-invariant; (3) the config
extracted from a per-layer model rebuilds an identical model and survives JSON round-trip.
"""
import numpy as np
import torch
from e3nn import o3
from scipy.spatial.transform import Rotation as R

from mace import data, modules, tools
from mace.tools import torch_geometric
from mace.tools.scripts_utils import (
    convert_from_json_format,
    convert_to_json_format,
    extract_config_mace_model,
    parse_hidden_irreps,
)

torch.set_default_dtype(torch.float64)

table = tools.AtomicNumberTable([1, 8])
atomic_energies = np.array([1.0, 3.0], dtype=float)
_positions = np.array([[0.0, -2.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
_rot = R.from_euler("z", 47, degrees=True).as_matrix()


def _config(positions):
    return data.Configuration(
        atomic_numbers=np.array([8, 1, 1]),
        positions=positions,
        properties={"forces": np.zeros((3, 3)), "energy": -1.5},
        property_weights={"forces": 1.0, "energy": 1.0},
    )


def _batch(positions):
    atoms = data.AtomicData.from_config(_config(positions), z_table=table, cutoff=3.0)
    loader = torch_geometric.dataloader.DataLoader([atoms], batch_size=1)
    return next(iter(loader)).to_dict()


def _base_config(hidden_irreps, num_interactions):
    return dict(
        r_max=5,
        num_bessel=8,
        num_polynomial_cutoff=6,
        max_ell=2,
        interaction_cls=modules.interaction_classes[
            "RealAgnosticResidualInteractionBlock"
        ],
        interaction_cls_first=modules.interaction_classes[
            "RealAgnosticResidualInteractionBlock"
        ],
        num_interactions=num_interactions,
        num_elements=2,
        hidden_irreps=hidden_irreps,
        MLP_irreps=o3.Irreps("16x0e"),
        gate=torch.nn.functional.silu,
        atomic_energies=atomic_energies,
        avg_num_neighbors=8,
        atomic_numbers=table.zs,
        correlation=3,
        radial_type="bessel",
        atomic_inter_scale=1.0,
        atomic_inter_shift=0.0,
    )


def _build(hidden_irreps, num_interactions, seed=0):
    torch.manual_seed(seed)
    return modules.ScaleShiftMACE(**_base_config(hidden_irreps, num_interactions))


def test_parse_hidden_irreps():
    assert parse_hidden_irreps("64x0e+64x1o") == o3.Irreps("64x0e+64x1o")
    assert parse_hidden_irreps(o3.Irreps("64x0e")) == o3.Irreps("64x0e")
    out = parse_hidden_irreps("128x0e+128x1o | 64x0e")
    assert out == [o3.Irreps("128x0e+128x1o"), o3.Irreps("64x0e")]
    assert parse_hidden_irreps(["32x0e", "16x0e"]) == [
        o3.Irreps("32x0e"),
        o3.Irreps("16x0e"),
    ]


def test_uniform_list_matches_single_irreps():
    """A single Irreps and the equivalent uniform list build byte-identical models."""
    n = 3
    single = _build(o3.Irreps("16x0e+16x1o"), n, seed=42)
    listed = _build([o3.Irreps("16x0e+16x1o")] * n, n, seed=42)

    s1, s2 = single.state_dict(), listed.state_dict()
    assert s1.keys() == s2.keys()
    for k in s1:
        assert torch.equal(s1[k], s2[k]), f"weight mismatch at {k}"

    batch = _batch(_positions)
    assert torch.allclose(
        single(batch)["energy"], listed(batch)["energy"], atol=1e-12
    )


def test_per_layer_builds_and_is_invariant():
    """Genuinely per-layer channels build, run, and keep energy rotation-invariant."""
    hidden = ["16x0e+16x1o", "8x0e+8x1o", "8x0e"]
    model = _build(hidden, num_interactions=3, seed=1)

    # layer-0 product output carries 16 scalar channels, layer-1 carries 8 -> per-layer
    assert model.products[0].linear.irreps_out.count(o3.Irrep(0, 1)) == 16
    assert model.products[1].linear.irreps_out.count(o3.Irrep(0, 1)) == 8

    e = model(_batch(_positions))["energy"]
    e_rot = model(_batch((_rot @ _positions.T).T))["energy"]
    assert torch.allclose(e, e_rot, atol=1e-8)


def test_extract_config_roundtrip_per_layer():
    """extract_config preserves per-layer irreps; rebuilt model matches exactly."""
    hidden = ["16x0e+16x1o", "8x0e+8x1o", "8x0e"]
    model = _build(hidden, num_interactions=3, seed=7)

    config = extract_config_mace_model(model)
    assert isinstance(config["hidden_irreps"], list)
    assert [str(h) for h in config["hidden_irreps"]] == [
        str(o3.Irreps(h)) for h in hidden
    ]

    rebuilt = modules.ScaleShiftMACE(**config)
    rebuilt.load_state_dict(model.state_dict())
    batch = _batch(_positions)
    assert torch.allclose(model(batch)["energy"], rebuilt(batch)["energy"], atol=1e-12)


def test_extract_config_uniform_stays_single_irreps():
    """A uniform model still reports a single Irreps (legacy-compatible config)."""
    model = _build(o3.Irreps("16x0e+16x1o"), num_interactions=3, seed=3)
    config = extract_config_mace_model(model)
    assert isinstance(config["hidden_irreps"], o3.Irreps)


def test_json_roundtrip_per_layer():
    """Per-layer hidden_irreps survive convert_to_json_format / convert_from_json_format."""
    hidden = ["16x0e+16x1o", "8x0e+8x1o", "8x0e"]
    model = _build(hidden, num_interactions=3, seed=5)
    config = extract_config_mace_model(model)

    as_json = convert_to_json_format(dict(config))
    assert as_json["hidden_irreps"] == "16x0e+16x1o | 8x0e+8x1o | 8x0e"
    back = convert_from_json_format(dict(as_json))
    assert back["hidden_irreps"] == [o3.Irreps(h) for h in hidden]
