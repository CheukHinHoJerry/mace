"""Go/no-go: how many STP paths would the product-group encoding need?

cueq flattens angular components into unit segments and emits one path per nonzero
coefficient. Our coefficient is Us[ijf,k] * Um[lmg,Q], so the path count is
nnz(Us) x nnz(Um) unless the kernel can keep the factorised structure.
"""

import sys

import torch

torch.serialization.add_safe_globals([slice])
sys.path.insert(0, "/home/jerry528/mace-ace-pr124/mace")
import cuequivariance as cue
from e3nn import o3

from mace.tools.cg import U_matrix_real

torch.set_default_dtype(torch.float64)

print(
    f"{'case':<28} {'nu':>2} {'nnz(Us)':>9} {'nnz(Um)':>9} {'product = STP paths':>21}"
)
print("-" * 76)
for me, mme, label in (
    (2, 1, "max_ell=2,max_m_ell=1"),
    (3, 2, "max_ell=3,max_m_ell=2"),
):
    Vr = o3.Irreps.spherical_harmonics(me, p=-1)
    Vm = o3.Irreps.spherical_harmonics(mme, p=-1)
    for nu in (2, 3):
        Us = U_matrix_real(
            irreps_in=Vr,
            irreps_out=o3.Irreps("0e"),
            correlation=nu,
            dtype=torch.float64,
            use_cueq_cg=False,
        )[-1]
        Um = U_matrix_real(
            irreps_in=Vm,
            irreps_out=o3.Irreps("0e"),
            correlation=nu,
            dtype=torch.float64,
            use_cueq_cg=False,
        )[-1]
        nr, nm = int((Us.abs() > 1e-12).sum()), int((Um.abs() > 1e-12).sum())
        print(f"{label:<28} {nu:>2} {nr:>9,} {nm:>9,} {nr*nm:>21,}")
print()
print("for reference, cueq's own single-group symmetric_contraction:")
for deg in (2, 3):
    d = cue.descriptors.symmetric_contraction(
        cue.Irreps("O3", "0e+1o+2e+3o"), cue.Irreps("O3", "0e"), (deg,)
    )
    for opn, stp in d.polynomial.operations:
        print(f"   degree {deg}: num_paths={stp.num_paths}")
