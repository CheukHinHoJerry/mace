# Implementation of the non-symmetric contraction algorithm, the current implementation is slow and not optimized for speed.
# Authors: Cheuk Hin Ho
# This program is distributed under the MIT License (see MIT.md)

import logging
from typing import Dict, Optional, Union

import opt_einsum_fx
import torch
import torch.fx
from e3nn import o3
from e3nn.util.codegen import CodeGenMixin
from opt_einsum import contract

from mace.tools.cg import U_matrix_real

BATCH_EXAMPLE = 10
ALPHABET = ["w", "x", "v", "n", "z", "r", "t"]
ALPHABET_MAGMOM = ["y", "u", "o", "p", "s"]
NONSOC_CONTRACTION_EQUATIONS = {
    1: "ik,lq,ekqa,bail,be->ba",
    2: "ijk,lmq,ekqa,bail,bajm,be->ba",
    3: "ijfk,lmgq,ekqa,bail,bajm,bafg,be->ba",
}

LOGGER = logging.getLogger(__name__)


def _joint_symmetrised_probe(
    U_spatial: torch.Tensor, U_magmom: torch.Tensor, nu: int, nprobe: int = 0
) -> torch.Tensor:
    """Evaluate every (spatial, magmom) path pair on generic inputs, jointly symmetrised.

    Contracting against nu identical copies of a probe X[n, i, j] equals contracting against
    its symmetrisation under one simultaneous permutation of the paired slots (i_r, j_r).
    The probe must be GENERIC in the paired index -- a factorised x_i*y_j symmetrises the two
    factors independently and under-reports the rank.
    """
    d_r, n_r = U_spatial.shape[0], U_spatial.shape[-1]
    d_m, n_m = U_magmom.shape[0], U_magmom.shape[-1]
    if nprobe <= 0:
        nprobe = max(256, 4 * n_r * n_m)
    gen = torch.Generator().manual_seed(0)
    # generic (NOT factorised) in the paired slot index, so full rank in (i, j)
    probe = torch.randn(nprobe, d_r, d_m, dtype=U_spatial.dtype, generator=gen)
    sl, ml = "abcdef"[:nu], "ghijkl"[:nu]
    eq = f"{sl}p,{ml}q," + ",".join(f"n{sl[r]}{ml[r]}" for r in range(nu)) + "->npq"
    return torch.einsum(eq, U_spatial, U_magmom, *([probe] * nu))


def _rank(mat: torch.Tensor, tol: float = 1e-9) -> int:
    if mat.numel() == 0 or mat.shape[-1] == 0:
        return 0
    sv = torch.linalg.svdvals(mat.double())
    return int((sv > tol * max(1.0, float(sv[0]))).sum())


def _select_joint_paths(
    U_spatial: torch.Tensor, U_magmom: torch.Tensor, nu: int
) -> torch.Tensor:
    """Drop a magmom CG path only if the JOINT basis loses no rank without it.

    Not used when building the model -- the spin-axis parity labels already admit exactly the
    allowed couplings. Kept as the verification tool the tests use.
    """
    if U_magmom.numel() == 0 or U_magmom.dim() < 2 or nu < 2:
        return U_magmom
    n_m = U_magmom.shape[-1]
    values = _joint_symmetrised_probe(U_spatial, U_magmom, nu)  # [nprobe, n_r, n_m]
    full = _rank(values.reshape(values.shape[0], -1))
    keep = list(range(n_m))
    for beta in range(n_m):
        trial = [b for b in keep if b != beta]
        sub = (
            values[..., trial].reshape(values.shape[0], -1)
            if trial
            else values[:, :, :0]
        )
        if _rank(sub) == full:
            keep = trial
    if len(keep) == n_m:
        return U_magmom
    LOGGER.info(
        "[NonSOCContraction] correlation %d: joint rank %d; %d of %d magmom CG path(s) are "
        "zero or linearly dependent in the jointly symmetrised basis",
        nu,
        full,
        n_m - len(keep),
        n_m,
    )
    return U_magmom[..., keep].contiguous()


class NonSOCSymmetricContraction(CodeGenMixin, torch.nn.Module):
    def __init__(
        self,
        irreps_in: o3.Irreps,
        irreps_out: o3.Irreps,
        correlation: Union[int, Dict[str, int]],
        irrep_normalization: str = "component",
        path_normalization: str = "element",
        internal_weights: Optional[bool] = None,
        shared_weights: Optional[bool] = None,
        num_elements: Optional[int] = None,
        magmom_irreps: Optional[o3.Irreps] = None,
        chunk_size: Optional[int] = 250,
    ) -> None:
        super().__init__()
        self.chunk_size = chunk_size

        if irrep_normalization is None:
            irrep_normalization = "component"

        if path_normalization is None:
            path_normalization = "element"

        assert irrep_normalization in ["component", "norm", "none"]
        assert path_normalization in ["element", "path", "none"]

        self.irreps_in = o3.Irreps(irreps_in)
        self.irreps_out = o3.Irreps(irreps_out)
        self.magmom_irreps = (
            o3.Irreps(magmom_irreps)
            if magmom_irreps is not None
            else o3.Irreps("1x0e+1x1o")
        )

        del irreps_in, irreps_out

        if not isinstance(correlation, tuple):
            corr = correlation
            correlation = {}
            for irrep_out in self.irreps_out:
                correlation[irrep_out] = corr

        assert shared_weights or not internal_weights

        if internal_weights is None:
            internal_weights = True

        self.internal_weights = internal_weights
        self.shared_weights = shared_weights

        del internal_weights, shared_weights

        self.contractions = torch.nn.ModuleList()
        for irrep_out in self.irreps_out:
            self.contractions.append(
                NonSOCContraction(
                    irreps_in=self.irreps_in,
                    irrep_out=o3.Irreps(str(irrep_out.ir)),
                    correlation=correlation[irrep_out],
                    internal_weights=self.internal_weights,
                    num_elements=num_elements,
                    weights=self.shared_weights,
                    magmom_irreps=self.magmom_irreps,
                    chunk_size=self.chunk_size,
                )
            )

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        outs = [contraction(x, y) for contraction in self.contractions]
        return torch.cat(outs, dim=-1)


# @compile_mode("script")
class NonSOCContraction(torch.nn.Module):
    def __init__(
        self,
        irreps_in: o3.Irreps,
        irrep_out: o3.Irreps,
        correlation: int,
        internal_weights: bool = True,
        num_elements: Optional[int] = None,
        weights: Optional[torch.Tensor] = None,
        magmom_irreps: Optional[o3.Irreps] = None,
        chunk_size: Optional[int] = 250,
    ) -> None:
        super().__init__()
        self.chunk_size = chunk_size

        # In the non-SOC A-tensor path, einsum index 'a' is the channel multiplicity
        # (mul axis after reshape), not total irreps count.
        muls = [mul for mul, _ in irreps_in]
        if len(set(muls)) != 1:
            raise ValueError(
                f"NonSOCContraction expects uniform multiplicity in irreps_in for channel axis; got muls={muls}"
            )
        self.num_features = int(muls[0])
        self.coupling_irreps = o3.Irreps([irrep.ir for irrep in irreps_in])
        magmom_irreps_full = (
            o3.Irreps(magmom_irreps)
            if magmom_irreps is not None
            else o3.Irreps("1x0e+1x1o")
        )
        # Match standard MACE contraction behavior: construct CG basis on irreps types,
        # not multiplicities, to keep contraction basis dimensions consistent.
        self.coupling_irreps_magmom = o3.Irreps(
            [irrep.ir for irrep in magmom_irreps_full]
        )
        self.correlation = correlation
        self.irrep_out = irrep_out
        # Spatial output equivariance: for a scalar (0e) output this is 1
        self.out_lmax = int(irrep_out.lmax)
        self.num_equivariance = 2 * self.out_lmax + 1
        self._equiv_letter = "z" if self.out_lmax > 0 else ""
        dtype = torch.get_default_dtype()
        LOGGER.info(
            "[NonSOCContraction] Building CG basis (structure): irreps_in=%s irreps_out=%s max_correlation=%s dtype=%s",
            self.coupling_irreps,
            irrep_out,
            correlation,
            dtype,
        )
        for nu in range(1, correlation + 1):
            LOGGER.info(
                "[NonSOCContraction] U_matrix_%s with irreps_in=%s irreps_out=%s correlation=%s dtype=%s",
                nu,
                self.coupling_irreps,
                irrep_out,
                nu,
                dtype,
            )
            U_matrix = U_matrix_real(
                irreps_in=self.coupling_irreps,
                irreps_out=irrep_out,
                correlation=nu,
                dtype=dtype,
                use_cueq_cg=True,
            )[-1]
            self.register_buffer(f"U_matrix_{nu}", U_matrix)

        # Magmom coupling basis is configurable and should match the magmom-side
        # irreps used to build the non-SOC interaction tensor.
        LOGGER.info(
            "[NonSOCContraction] Building CG basis (magmom): irreps_in=%s irreps_out=%s max_correlation=%s dtype=%s",
            self.coupling_irreps_magmom,
            irrep_out,
            correlation,
            dtype,
        )
        # O(3)_space x O(3)_spin. The magmom parts always contract to a spin scalar.
        magmom_out = o3.Irreps("0e")
        for nu in range(1, correlation + 1):
            LOGGER.info(
                "[NonSOCContraction] U_matrix_magmom_%s with irreps_in=%s irreps_out=%s correlation=%s dtype=%s",
                nu,
                self.coupling_irreps_magmom,
                magmom_out,
                nu,
                dtype,
            )
            U_matrix = U_matrix_real(
                irreps_in=self.coupling_irreps_magmom,
                irreps_out=magmom_out,
                correlation=nu,
                dtype=dtype,
                use_cueq_cg=True,
            )[-1]
            self.register_buffer(f"U_matrix_magmom_{nu}", U_matrix)

        # Tensor contraction equations
        self.contractions_weighting = torch.nn.ModuleList()
        self.contractions_features = torch.nn.ModuleList()

        # Create weight for product basis
        self.weights = torch.nn.ParameterList([])
        lower_order_weights = []

        for i in range(correlation, 0, -1):
            # Shapes definying
            num_params = self.U_tensors(i).size()[-1]
            num_params_magmom = self.U_magmom_tensors(i).size()[-1]
            _num_equivariance = 2 * irrep_out.lmax + 1
            _num_ell = self.U_tensors(i).size()[-2]

            if i == correlation:
                # Parameters for the product basis
                w = torch.nn.Parameter(
                    torch.randn(
                        (num_elements, num_params, num_params_magmom, self.num_features)
                    )
                    / (num_params * num_params_magmom)
                )
                self.weights_max = w
            else:
                # Parameters for lower-order product basis terms.
                # Keep list ordered by nu ascending: weights[0] -> nu=1, weights[1] -> nu=2, ...
                w = torch.nn.Parameter(
                    torch.randn(
                        (num_elements, num_params, num_params_magmom, self.num_features)
                    )
                    / (num_params * num_params_magmom)
                )
                lower_order_weights.append(w)
        # Rebuild in ascending nu order: weights[0] -> nu=1, weights[1] -> nu=2, ...
        if len(lower_order_weights) > 0:
            self.weights = torch.nn.ParameterList(list(reversed(lower_order_weights)))

        if not internal_weights:
            self.weights = weights[:-1]
            self.weights_max = weights[-1]

        # Defensive: the magmom side must actually contribute. U_matrix_real returns an
        # all-zero U with num_params==1 when no CG path exists; if that ever happened for
        # the magmom coupling the model would silently train as if magmoms did not exist.
        for nu in range(1, correlation + 1):
            um = self.U_magmom_tensors(nu)
            if not bool(um.abs().sum() > 0):
                raise ValueError(
                    f"magmom CG basis U_magmom_{nu} is all-zero for irreps_in="
                    f"{self.coupling_irreps_magmom}->0e; magmom coupling would be dead."
                )

    def _build_optimized_contraction(self, nu: int) -> torch.nn.Module:
        dtype = self.weights_max.dtype
        device = self.weights_max.device
        u = self.U_tensors(nu)
        um = self.U_magmom_tensors(nu)
        weight = self.weights_max if nu == self.correlation else self.weights[nu - 1]
        num_ell_r = int(self.U_tensors(1).size(-2))
        num_ell_m = int(self.U_magmom_tensors(1).size(-2))
        x_example = torch.randn(
            (BATCH_EXAMPLE, self.num_features, num_ell_r, num_ell_m),
            dtype=dtype,
            device=device,
        )
        y_example = torch.randn(
            (BATCH_EXAMPLE, weight.shape[0]),
            dtype=dtype,
            device=device,
        )

        if nu == 1:
            graph_module = torch.fx.symbolic_trace(
                lambda u_t, um_t, w_t, x_t, y_t: torch.einsum(
                    NONSOC_CONTRACTION_EQUATIONS[1], u_t, um_t, w_t, x_t, y_t
                )
            )
            example_inputs = (
                torch.randn(tuple(u.shape), dtype=dtype, device=device),
                torch.randn(tuple(um.shape), dtype=dtype, device=device),
                torch.randn(tuple(weight.shape), dtype=dtype, device=device),
                x_example,
                y_example,
            )
        elif nu == 2:
            graph_module = torch.fx.symbolic_trace(
                lambda u_t, um_t, w_t, x0_t, x1_t, y_t: torch.einsum(
                    NONSOC_CONTRACTION_EQUATIONS[2], u_t, um_t, w_t, x0_t, x1_t, y_t
                )
            )
            example_inputs = (
                torch.randn(tuple(u.shape), dtype=dtype, device=device),
                torch.randn(tuple(um.shape), dtype=dtype, device=device),
                torch.randn(tuple(weight.shape), dtype=dtype, device=device),
                x_example,
                x_example,
                y_example,
            )
        elif nu == 3:
            graph_module = torch.fx.symbolic_trace(
                lambda u_t, um_t, w_t, x0_t, x1_t, x2_t, y_t: torch.einsum(
                    NONSOC_CONTRACTION_EQUATIONS[3],
                    u_t,
                    um_t,
                    w_t,
                    x0_t,
                    x1_t,
                    x2_t,
                    y_t,
                )
            )
            example_inputs = (
                torch.randn(tuple(u.shape), dtype=dtype, device=device),
                torch.randn(tuple(um.shape), dtype=dtype, device=device),
                torch.randn(tuple(weight.shape), dtype=dtype, device=device),
                x_example,
                x_example,
                x_example,
                y_example,
            )
        else:
            raise ValueError(f"Unsupported correlation order: {nu}")

        return opt_einsum_fx.optimize_einsums_full(
            model=graph_module,
            example_inputs=example_inputs,
        )

    def _ensure_optimized_contractions(self) -> None:
        # pylint: disable=attribute-defined-outside-init  # lazily built on first use
        if hasattr(self, "optimized_contractions"):
            return
        self.optimized_contractions = torch.nn.ModuleDict()
        for nu in range(1, self.correlation + 1):
            self.optimized_contractions[str(nu)] = self._build_optimized_contraction(nu)

    def forward_reference(self, x: torch.Tensor, y: torch.Tensor):
        # x is the non-SOC A-tensor of shape (node b, channel a, spatial-ell i, magmom-ell l).
        # y is node_attrs one-hot (B, e).
        #
        # The spatial output may be equivariant (irrep_out.lmax > 0, e.g. 1o). The spatial U
        # then carries a leading (2L+1) equivariance index; the magmom U is always a spin
        # scalar (0e), so only the spatial side and the output carry that index. `E` is a
        # single free einsum letter for it (empty when L == 0, which reproduces the original
        # scalar equations exactly). The magmom-ell indices (l,m,g) are still summed out.
        E = "d" if self.irrep_out.lmax > 0 else ""
        equations = {
            1: f"{E}ik,lq,ekqa,bail,be->ba{E}",
            2: f"{E}ijk,lmq,ekqa,bail,bajm,be->ba{E}",
            3: f"{E}ijfk,lmgq,ekqa,bail,bajm,bafg,be->ba{E}",
        }

        out = None
        for nu in range(1, self.correlation + 1):
            weight_nu = (
                self.weights_max if nu == self.correlation else self.weights[nu - 1]
            )
            if nu not in equations:
                raise ValueError(f"Unsupported correlation order: {nu}")
            # x is repeated nu times (bail, bajm, bafg); U/U_magmom/weight/y once.
            term = contract(
                equations[nu],
                self.U_tensors(nu),
                self.U_magmom_tensors(nu),
                weight_nu,
                *([x] * nu),
                y,
            )
            out = term if out is None else (out + term)

        # (node, channel[, 2L+1]) -> (node, channel * (2L+1)); mul-major, matching e3nn/MACE.
        return out.view(out.shape[0], -1)

    def forward_optimized(self, x: torch.Tensor, y: torch.Tensor):
        assert self.irrep_out.lmax == 0
        self._ensure_optimized_contractions()

        out = None
        for nu in range(1, self.correlation + 1):
            weight_nu = (
                self.weights_max if nu == self.correlation else self.weights[nu - 1]
            )
            optimized = self.optimized_contractions[str(nu)]

            if nu == 1:
                term = optimized(
                    self.U_tensors(nu),
                    self.U_magmom_tensors(nu),
                    weight_nu,
                    x,
                    y,
                )
            elif nu == 2:
                term = optimized(
                    self.U_tensors(nu),
                    self.U_magmom_tensors(nu),
                    weight_nu,
                    x,
                    x,
                    y,
                )
            elif nu == 3:
                term = optimized(
                    self.U_tensors(nu),
                    self.U_magmom_tensors(nu),
                    weight_nu,
                    x,
                    x,
                    x,
                    y,
                )
            else:
                raise ValueError(f"Unsupported correlation order: {nu}")

            out = term if out is None else (out + term)

        return out.view(out.shape[0], -1)

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        # The reference contraction materializes a (spatial)^nu x (magmom)^nu scratch
        # tensor whose size scales with the node count; at large batches it dominates
        # both memory (OOM) and runtime (memory-bandwidth bound). Splitting the nodes
        # into chunks bounds that scratch tensor -- numerically identical, and in
        # practice both lower-memory AND faster. chunk_size=None keeps the original
        # single-shot path. getattr() keeps models pickled before this attr existed working.
        chunk_size = getattr(self, "chunk_size", None)
        if chunk_size is None or x.shape[0] <= chunk_size:
            return self.forward_reference(x, y)
        outs = [
            self.forward_reference(x[i : i + chunk_size], y[i : i + chunk_size])
            for i in range(0, x.shape[0], chunk_size)
        ]
        return torch.cat(outs, dim=0)

    def U_tensors(self, nu: int):
        return dict(self.named_buffers())[f"U_matrix_{nu}"]

    def U_magmom_tensors(self, nu: int):
        return dict(self.named_buffers())[f"U_matrix_magmom_{nu}"]
