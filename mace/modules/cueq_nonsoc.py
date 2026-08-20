###########################################################################################
# Product-group cuEquivariance backend for the non-SOC symmetric contraction
# Authors: Cheuk Hin Ho
# This program is distributed under the MIT License (see MIT.md)
###########################################################################################

"""cuEquivariance backend for the non-SOC contraction.

Every slot of the contraction carries a spatial index ``i`` and a spin index ``l``. Treating
the pair ``(i, l)`` as ONE irrep label of ``O(3)_space x SO(3)_spin`` turns the contraction
into a single ``SegmentedTensorProduct`` whose path coefficients are products of the spatial
and magnetic CG entries -- one path per (nonzero spatial entry, nonzero magnetic entry) pair.

Each slot gets its OWN buffer. cuEquivariance evaluates the right function but returns the
WRONG gradient when a single buffer is repeated across slots (measured rel 2.9e+01 at first
order and 8.7e+01 at second); letting PyTorch accumulate the per-slot gradients instead is
exact to machine precision at both orders, which the force loss requires.
"""

import inspect
import os
import time
from typing import List, Optional

import torch

try:
    import cuequivariance as cue
    import cuequivariance_torch as cuet

    CUEQ_NONSOC_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised only where cueq is absent
    CUEQ_NONSOC_AVAILABLE = False


def _nonzeros(u: torch.Tensor, nu: int, tol: float = 1e-12):
    """Index/column/value triples of a CG basis of shape ``(d,) * nu + (P,)``.

    Vectorised: the obvious nested-``itertools.product`` version costs one Python-level
    tensor index per ELEMENT (d**nu * P of them), which is minutes at production shapes.
    """
    hits = (u.abs() > tol).nonzero()
    if hits.numel() == 0:
        return []
    vals = u[tuple(hits.t())].tolist()
    rows = hits.tolist()
    return [(tuple(r[:nu]), r[nu], v) for r, v in zip(rows, vals)]


class PairedContractionCueq(torch.nn.Module):
    """One correlation order of the non-SOC contraction as a cueq SegmentedPolynomial.

    Reproduces ``einsum("ijfk,lmgQ,bkQa,bail,bajm,bafg->ba")`` (shown at nu=3), i.e. the
    production equation with the centre-moment factor ``v[b, Q]`` already folded into the
    weights -- folding is exact by linearity since ``v`` does not depend on the slots.
    """

    def __init__(
        self,
        u_spatial: torch.Tensor,
        u_magmom: torch.Tensor,
        num_features: int,
        nu: int,
        math_dtype: Optional[torch.dtype] = None,
        method: Optional[str] = None,
    ) -> None:
        super().__init__()
        if not CUEQ_NONSOC_AVAILABLE:
            raise ImportError(
                "PairedContractionCueq needs cuequivariance; pip install cuequivariance "
                "cuequivariance-torch cuequivariance-ops-torch-cu12"
            )
        self.nu = nu
        self.dr = int(u_spatial.shape[0])
        self.dm = int(u_magmom.shape[0])
        self.num_paths_spatial = int(u_spatial.shape[-1])
        self.num_paths_magmom = int(u_magmom.shape[-1])
        self.num_features = int(num_features)
        ch = self.num_features
        n_pair = self.dr * self.dm
        n_weight = self.num_paths_spatial * self.num_paths_magmom

        stp = cue.SegmentedTensorProduct.from_subscripts(",".join(["u"] * (nu + 2)))
        for _ in range(n_weight):
            stp.add_segment(0, (ch,))
        for slot in range(1, nu + 1):
            for _ in range(n_pair):
                stp.add_segment(slot, (ch,))
        stp.add_segment(nu + 1, (ch,))

        # One path per (spatial nonzero, magnetic nonzero); the paired component index is
        # i * dm + l and the coefficient is the product of the two CG entries.
        # Hoisted: recomputing the magnetic nonzeros inside the outer loop makes the build
        # O(n_spatial) times more expensive than it needs to be.
        _t0 = time.time()
        magnetic_nonzeros = _nonzeros(u_magmom, nu)
        for sidx, k, cs in _nonzeros(u_spatial, nu):
            for midx, q, cm in magnetic_nonzeros:
                stp.add_path(
                    k * self.num_paths_magmom + q,
                    *[sidx[t] * self.dm + midx[t] for t in range(nu)],
                    0,
                    c=cs * cm,
                )
        self.num_paths = stp.num_paths
        _t1 = time.time()

        poly = cue.SegmentedPolynomial(
            inputs=tuple(stp.operands[: nu + 1]),
            outputs=(stp.operands[nu + 1],),
            # Distinct buffer per slot -- see the module docstring.
            operations=[(cue.Operation(list(range(nu + 1)) + [nu + 1]), stp)],
        )
        # `method` selects the kernel (e.g. "naive" = pure PyTorch, "uniform_1d" = the JIT
        # CUDA kernel). It only exists from cuequivariance 0.11; on 0.5.1 the argument is
        # absent and the kernel is whatever that build picks, so pass it only if supported.
        kwargs = {"math_dtype": math_dtype or torch.get_default_dtype()}
        if method is None:
            method = os.environ.get("CUEQ_METHOD") or None
        if method:
            if (
                "method"
                not in inspect.signature(cuet.SegmentedPolynomial.__init__).parameters
            ):
                raise ValueError(
                    f"method={method!r} needs cuequivariance >= 0.11; this build's "
                    "SegmentedPolynomial takes no `method` argument."
                )
            kwargs["method"] = method
        self.method = method
        self.poly = cuet.SegmentedPolynomial(poly, **kwargs)
        if os.environ.get("CUEQ_NONSOC_VERBOSE"):
            print(
                f"[cueq_nonsoc] nu={nu} paths={self.num_paths} "
                f"stp_build={_t1 - _t0:.1f}s poly_build={time.time() - _t1:.1f}s "
                f"method={method or '<default>'}",
                flush=True,
            )

    def forward(self, weights: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """weights ``(B, P, Q, ch)`` with ``v`` folded in, x ``(B, ch, dr, dm)`` -> ``(B, ch)``."""
        n = x.shape[0]
        w_flat = weights.reshape(n, -1)
        # (b, a, i, l) -> (b, (i, l), a): the STP indexes the pair major, channel minor.
        x_flat = x.permute(0, 2, 3, 1).reshape(n, -1)
        out = self.poly([w_flat] + [x_flat] * self.nu)[0]
        return out.reshape(n, self.num_features)


def build_paired_contractions(
    u_spatial: List[torch.Tensor],
    u_magmom: List[torch.Tensor],
    num_features: int,
    correlation: int,
    math_dtype: Optional[torch.dtype] = None,
    method: Optional[str] = None,
) -> torch.nn.ModuleDict:
    """One :class:`PairedContractionCueq` per correlation order, keyed ``"nu{n}"``."""
    return torch.nn.ModuleDict(
        {
            f"nu{nu}": PairedContractionCueq(
                u_spatial[nu - 1],
                u_magmom[nu - 1],
                num_features,
                nu,
                math_dtype,
                method,
            )
            for nu in range(1, correlation + 1)
        }
    )
