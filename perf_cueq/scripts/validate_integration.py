"""cueq vs einsum for NonSOCSymmetricContraction: forward, both gradients, double backward.

The training loss is on FORCES, which are themselves autograd derivatives, so agreement at
first order is not enough -- the double backward has to be exact too.
"""

import os
import sys
import time

import torch
from e3nn import o3

sys.path.insert(0, "/home/jerry528/mace-ace-pr124/mace")
from mace.modules.symmetric_contraction_nonsoc import NonSOCSymmetricContraction

T0 = time.time()


def log(m):
    print(f"[{time.time()-T0:7.1f}s] {m}", flush=True)


torch.manual_seed(0)
DEV = "cuda"
DT = torch.float64
torch.set_default_dtype(DT)

MAX_ELL = int(os.environ.get("MAX_ELL", 2))
MAX_MELL = int(os.environ.get("MAX_MELL", 1))
CH = int(os.environ.get("CH", 16))
B = int(os.environ.get("BATCH", 12))
NU = int(os.environ.get("NU", 3))
NEL = int(os.environ.get("NEL", 2))
CENTER = os.environ.get("CENTER", "1") == "1"

irreps_in = o3.Irreps(
    "+".join(f"{CH}x{l}{'e' if l%2==0 else 'o'}" for l in range(MAX_ELL + 1))
)
magmom_irreps = o3.Irreps(
    "+".join(f"1x{l}{'e' if l%2==0 else 'o'}" for l in range(MAX_MELL + 1))
)
log(
    f"cueq check | max_ell={MAX_ELL} m_ell={MAX_MELL} nu={NU} ch={CH} B={B} center={CENTER}"
)
log(f"irreps_in={irreps_in} magmom={magmom_irreps}")


def build(use_cueq):
    torch.manual_seed(1234)
    return (
        NonSOCSymmetricContraction(
            irreps_in=irreps_in,
            irreps_out=o3.Irreps(f"{CH}x0e"),
            correlation=NU,
            num_elements=NEL,
            magmom_irreps=magmom_irreps,
            internal_weights=True,
            shared_weights=True,
            chunk_size=None,
            center_spin_coupling=True,
            use_cueq=use_cueq,
        )
        .to(DEV)
        .to(DT)
    )


ref = build(False)
log("built einsum reference")
cq = build(True)
log(
    f"built cueq backend (paths: "
    + ", ".join(
        f"nu{n}={cq.contractions[0].cueq_contractions[f'nu{n}'].num_paths}"
        for n in range(1, NU + 1)
    )
    + ")"
)
# Copy the trainable parameters explicitly. load_state_dict(strict=True) cannot be used:
# cueq >= 0.11 registers the STP coefficients as buffers on the cueq module, so the einsum
# reference's state_dict is missing keys the cueq module legitimately has.
with torch.no_grad():
    ref_params = dict(ref.named_parameters())
    n_copied = 0
    for name, p in cq.named_parameters():
        assert name in ref_params, f"cueq module has an unexpected parameter {name}"
        p.copy_(ref_params[name])
        n_copied += 1
    assert n_copied == len(
        ref_params
    ), f"copied {n_copied} of {len(ref_params)} parameters"
log(f"copied {n_copied} parameter tensors from the reference")

dr = irreps_in.num_irreps // CH if False else sum(2 * l + 1 for l in range(MAX_ELL + 1))
dm = sum(2 * l + 1 for l in range(MAX_MELL + 1))
x0 = torch.randn(B, CH, dr, dm, device=DEV, dtype=DT)
y = torch.nn.functional.one_hot(torch.randint(0, NEL, (B,), device=DEV), NEL).to(DT)
# centre_cols = cat([ones, m_center]) is indexed as 1 + s**2 + comp, so m_center is the
# FULL dm-dimensional flat spherical embedding of the centre moment (block s at s**2).
mc = torch.randn(B, dm, device=DEV, dtype=DT) if CENTER else None
log(
    f"inputs x={tuple(x0.shape)} y={tuple(y.shape)} m_center={None if mc is None else tuple(mc.shape)}"
)


def rel(a, b):
    d = (a - b).abs().max().item()
    s = b.abs().max().item()
    return d, d / s if s > 0 else d


FAIL = []


def check(name, a, b, tol=1e-11):
    d, r = rel(a, b)
    ok = r < tol
    log(f"  {name:<28} rel={r:.3e}  abs={d:.3e}  {'OK' if ok else 'MISMATCH'}")
    if not ok:
        FAIL.append(name)


# ---- forward -------------------------------------------------------------
with torch.no_grad():
    check("forward", cq(x0, y, mc), ref(x0, y, mc))


# ---- first backward: d/dx and d/dweights ---------------------------------
def run(mod, x, m):
    out = mod(x, y, m)
    return (out * torch.linspace(1, 2, out.shape[-1], device=DEV, dtype=DT)).sum()


grads = {}
for tag, mod in (("ref", ref), ("cueq", cq)):
    x = x0.clone().requires_grad_(True)
    m = None if mc is None else mc.clone().requires_grad_(True)
    loss = run(mod, x, m)
    ws = [p for p in mod.parameters()]
    gs = torch.autograd.grad(
        loss, [x] + ([] if m is None else [m]) + ws, create_graph=True
    )
    grads[tag] = (loss, gs, x, m)

check("d loss/dx", grads["cueq"][1][0], grads["ref"][1][0])
k = 1
if mc is not None:
    check("d loss/d m_center", grads["cueq"][1][1], grads["ref"][1][1])
    k = 2
for i in range(k, len(grads["ref"][1])):
    check(f"d loss/d w[{i-k}]", grads["cueq"][1][i], grads["ref"][1][i])

# ---- double backward: this is the one the force loss needs ---------------
for tag in ("ref", "cueq"):
    loss, gs, x, m = grads[tag]
    gx = gs[0]
    # a scalar built from the FIRST derivative, as a force loss is
    l2 = (
        gx * torch.arange(1, gx.numel() + 1, device=DEV, dtype=DT).view(gx.shape)
    ).sum()
    ws = [p for p in (ref if tag == "ref" else cq).parameters()]
    grads[tag] = grads[tag] + (torch.autograd.grad(l2, [x] + ws, retain_graph=True),)

check("d2 loss/dx2 (force-like)", grads["cueq"][-1][0], grads["ref"][-1][0])
for i in range(1, len(grads["ref"][-1])):
    check(f"d2 loss/dx dw[{i-1}]", grads["cueq"][-1][i], grads["ref"][-1][i])

log("")
if FAIL:
    log(f"FAILED: {len(FAIL)} check(s): {FAIL}")
    sys.exit(1)
log("ALL CHECKS PASSED (forward, first backward, double backward)")
