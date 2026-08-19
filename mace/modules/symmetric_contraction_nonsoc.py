# Implementation of the non-symmetric contraction algorithm, the current implementation is slow and not optimized for speed.
# Authors: Cheuk Hin Ho
# This program is distributed under the MIT License (see MIT.md)

import logging
from typing import Dict, List, Optional, Union

import opt_einsum_fx
import torch
import torch.fx
from e3nn import o3
from e3nn.util.codegen import CodeGenMixin
from opt_einsum import contract

from mace.tools.cg import U_matrix_real, _wigner_nj

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


def _assert_wigner_nj_available(irreps: o3.Irreps, nu: int, what: str) -> None:
    """Refuse to build if U_matrix_real would silently fall back to cuequivariance.

    `use_cieq_cg=False` asks for the full unsymmetrised coupling-tree basis from
    _wigner_nj, but U_matrix_real catches NotImplementedError from it and, when
    cuequivariance is installed, returns compute_U_cueq instead. That returns
    cue.reduced_symmetric_tensor_product_basis -- a basis of Sym^nu of ONE space -- and
    pairing two of those drops the mixed Young-symmetry sectors of Sym^nu(V_r (x) V_m),
    making the hypothesis class depend on whether cuequivariance happens to be installed.

    Probing _wigner_nj directly is exact: it is the only condition under which the
    fallback fires. Checking the returned basis for slot symmetry instead would
    false-positive, because at nu=2 the coupling to a scalar is legitimately symmetric in
    every column.
    """
    try:
        _wigner_nj([irreps] * nu, "component", None, torch.get_default_dtype())
    except NotImplementedError as exc:
        raise RuntimeError(
            f"{what}: _wigner_nj cannot build the correlation-{nu} basis for {irreps}, so "
            "U_matrix_real would fall back to the cuequivariance SYMMETRIC basis and "
            "silently drop the mixed Young-symmetry sectors. See "
            "mace/modules/docs/nonsoc_model_spec.md section 4.1."
        ) from exc


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
        center_spin_coupling: bool = True,
    ) -> None:
        super().__init__()
        self.chunk_size = chunk_size
        self.center_spin_coupling = center_spin_coupling

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
                    center_spin_coupling=center_spin_coupling,
                )
            )

    def forward(
        self, x: torch.Tensor, y: torch.Tensor, m_center: Optional[torch.Tensor] = None
    ):
        outs = [contraction(x, y, m_center) for contraction in self.contractions]
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
        center_spin_coupling: bool = True,
    ) -> None:
        super().__init__()
        if correlation > 3:
            raise ValueError(
                f"NonSOCContraction supports correlation <= 3, got {correlation}. "
                "The contraction equations and the joint-completeness verification both "
                "stop at nu = 3."
            )
        self.chunk_size = chunk_size
        self.center_spin_coupling = bool(center_spin_coupling)

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
        # use_cueq_cg=False is REQUIRED here; it is not a performance preference.
        #
        # The spatial and magmom bases are built separately and paired, and the contraction
        # against A^{(x)nu} projects the product onto the jointly symmetric subspace. That is
        # complete ONLY if each side is the FULL UNSYMMETRISED coupling-tree basis, which is
        # what _wigner_nj returns. cuequivariance's compute_U_cueq instead returns
        # cue.reduced_symmetric_tensor_product_basis, a basis of Sym^nu of ONE space; pairing
        # two of those spans only Sym^nu(V_r) (x) Sym^nu(V_m) and drops the mixed Young
        # sectors of Sym^nu(V_r (x) V_m) = (+)_lambda S^lambda(V_r) (x) S^lambda(V_m).
        # Measured loss at nu=3 (target 0e, spin T-even):
        #     max_ell=2,max_m_ell=1 : 13 -> 10       max_ell=3,max_m_ell=2 : 61 -> 40
        # Left at True, the hypothesis class would silently depend on whether cuequivariance
        # happens to be installed. See docs/nonsoc_model_spec.md section 4.1.
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
                use_cueq_cg=False,
            )[-1]
            _assert_wigner_nj_available(self.coupling_irreps, nu, "spatial CG basis")
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
                use_cueq_cg=False,
            )[-1]
            self.register_buffer(f"U_matrix_magmom_{nu}", U_matrix)

        # Centre-spin channels, one per spin order s = 1 .. max_m_ell.
        #
        # The s=0 basis above contracts the neighbour magmom slots to a spin SCALAR, after
        # which m_i can only re-enter through invariants of |m_i|. That makes the ordinary
        # exchange term J_ij(R) m_i . m_j unrepresentable. Requesting a spin-s output
        # instead leaves 2s+1 free spin indices,
        #     Gamma_{i,mu} = [U^s A^{(x)nu}]_mu ,
        # closed in forward_reference against the centre moment's own l=s block,
        #     [Gamma_i (x) M_i]_0 = sum_mu Gamma_{i,mu} M_{i,mu} .
        # s=1 at nu=1 is exactly sum_j J(r_ij) m_i . m_j; s=2 carries the biquadratic
        # (m_i . m_j)^2 order, and so on. s is capped by max_m_ell because the centre
        # attributes stop there.
        #
        # Time reversal is automatic and MUST NOT be imposed by hand: the e3nn parity slot
        # tracks T here, so irreps_out of parity (-1)^s admits only neighbour combinations
        # whose T-parity matches M_i's, making every product T-even. Hence 0e, 1o, 2e, ...
        #
        # Joint rank retained, max_ell=3 / max_m_ell=2 / nu=3 (verified two ways):
        #     s=0 -> 61,  s=1 -> 91,  s=2 -> 105,  total 257.
        # See docs/nonsoc_model_spec.md sections 3.4 and 4.1.
        self.center_spin_orders: List[int] = []
        if self.center_spin_coupling:
            self.center_spin_orders = list(
                range(1, int(self.coupling_irreps_magmom.lmax) + 1)
            )
            for s_order in self.center_spin_orders:
                parity = "e" if s_order % 2 == 0 else "o"
                for nu in range(1, correlation + 1):
                    U_matrix = U_matrix_real(
                        irreps_in=self.coupling_irreps_magmom,
                        irreps_out=o3.Irreps(f"{s_order}{parity}"),
                        correlation=nu,
                        dtype=dtype,
                        use_cueq_cg=False,
                    )[-1]
                    self.register_buffer(f"U_matrix_magmom_s{s_order}_{nu}", U_matrix)

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

        # Matching weights, one tensor per (centre-spin order, correlation order).
        if self.center_spin_coupling:
            center_weights = {}
            for s_order in self.center_spin_orders:
                for nu in range(1, correlation + 1):
                    num_params = self.U_tensors(nu).size()[-1]
                    num_params_s = self.U_magmom_center_tensors(s_order, nu).size()[-1]
                    center_weights[f"s{s_order}_nu{nu}"] = torch.nn.Parameter(
                        torch.randn(
                            (num_elements, num_params, num_params_s, self.num_features)
                        )
                        / (num_params * num_params_s)
                    )
            self.center_weights = torch.nn.ParameterDict(center_weights)

        # Merged magnetic basis: fold (s, spin component c, path) into ONE index Q.
        #
        # Every s-channel contracts the SAME expensive spatial object
        #     sum_{ijf} U^r_{ijf,k} A_{i.} A_{j.} A_{f.}
        # so running one contraction per s recomputes it each time. Stacking the magnetic
        # bases along a single Q axis lets the optimiser share that work; measured 2.40x
        # at nu=3, max_m_ell=2 with identical output (1.7e-12).
        #
        # center_index_{nu} says which component of [1, magmom_attrs] multiplies each Q:
        # index 0 is the constant 1, used by the s=0 columns which carry no centre factor
        # (their Y_0 is absorbed into the radial channel), and s>=1 columns select the
        # matching l=s component at 1 + s^2 + c.
        # weight_index_{nu} maps Q -> its weight column. The weight MUST be shared across
        # the 2s+1 spin components of a given (s, path): the components are summed against
        # the centre moment as [Gamma_s (x) M_s]_0 = sum_c Gamma_{s,c} M_{s,c}, so giving
        # each c its own coefficient would break SO(3)_spin invariance (it does -- the
        # symmetry tests catch it immediately).
        for nu in range(1, correlation + 1):
            cols: List[torch.Tensor] = []
            idx: List[int] = []
            widx: List[int] = []
            u_scalar = self.U_magmom_tensors(nu)
            for path in range(u_scalar.shape[-1]):
                cols.append(u_scalar[..., path])
                idx.append(0)
                widx.append(len(widx))
            next_w = len(widx)
            for s_order in self.center_spin_orders:
                u_s = self.U_magmom_center_tensors(s_order, nu)
                for comp in range(u_s.shape[0]):
                    for path in range(u_s.shape[-1]):
                        cols.append(u_s[comp][..., path])
                        idx.append(1 + s_order * s_order + comp)
                        widx.append(next_w + path)
                next_w += u_s.shape[-1]
            self.register_buffer(f"U_magmom_merged_{nu}", torch.stack(cols, dim=-1))
            self.register_buffer(
                f"center_index_{nu}", torch.tensor(idx, dtype=torch.long)
            )
            self.register_buffer(
                f"weight_index_{nu}", torch.tensor(widx, dtype=torch.long)
            )

        # One merged weight per correlation order, laid out to match the Q axis so the
        # per-s tensors above are never used in the forward pass.
        merged = {}
        for nu in range(1, correlation + 1):
            num_params = self.U_tensors(nu).size()[-1]
            widx = dict(self.named_buffers())[f"weight_index_{nu}"]
            num_w = int(widx.max()) + 1
            merged[f"nu{nu}"] = torch.nn.Parameter(
                torch.randn((num_elements, num_params, num_w, self.num_features))
                / (num_params * num_w)
            )
        self.merged_weights = torch.nn.ParameterDict(merged)

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

    def forward_reference(
        self, x: torch.Tensor, y: torch.Tensor, m_center: Optional[torch.Tensor] = None
    ):
        # x is the non-SOC A-tensor of shape (node b, channel a, spatial-ell i, magmom-ell l).
        # y is node_attrs one-hot (B, e).
        #
        # The spatial output may be equivariant (irrep_out.lmax > 0, e.g. 1o). The spatial U
        # then carries a leading (2L+1) equivariance index; the magmom U is always a spin
        # scalar (0e), so only the spatial side and the output carry that index. `E` is a
        # single free einsum letter for it (empty when L == 0, which reproduces the original
        # scalar equations exactly). The magmom-ell indices (l,m,g) are still summed out.
        E = "d" if self.irrep_out.lmax > 0 else ""
        # ONE contraction per nu over the merged magnetic index Q = (s, spin component,
        # path). 'Q' replaces the old separate 'q' (s=0) and 'c,q' (s>=1) axes, so the
        # spatial factor is contracted once instead of once per s.
        #
        # 'v' carries the centre factor per Q: v[b, Q] = [1, magmom_attrs][b, index[Q]],
        # so the s=0 columns are multiplied by 1 and each s>=1 column by its matching
        # l=s component of the centre moment. This is exactly
        #     [ Gamma_{i,s} (x) Y_s(m_i) ]_0 = sum_c Gamma_{i,s,c} M_{i,s,c}
        # written as a single sum over Q.
        equations = {
            1: f"{E}ik,lQ,bkQa,bail,bQ->ba{E}",
            2: f"{E}ijk,lmQ,bkQa,bail,bajm,bQ->ba{E}",
            3: f"{E}ijfk,lmgQ,bkQa,bail,bajm,bafg,bQ->ba{E}",
        }
        buffers = dict(self.named_buffers())
        # node_attrs is one-hot, so gather the element's weight rather than carrying the
        # element axis through the high-order contraction.
        species = torch.argmax(y, dim=-1)
        # Honour the disable flag as well as a missing centre moment: with the channel off
        # only the s=0 columns are kept, which is the original scalar contraction.
        use_center = self._use_center_spin(m_center)
        if not use_center:
            ones = torch.ones(
                y.shape[0], 1, dtype=self.weights_max.dtype, device=y.device
            )
            centre_cols = ones
        else:
            ones = torch.ones(
                m_center.shape[0], 1, dtype=m_center.dtype, device=m_center.device
            )
            centre_cols = torch.cat([ones, m_center], dim=1)

        out = None
        for nu in range(1, self.correlation + 1):
            if nu not in equations:
                raise ValueError(f"Unsupported correlation order: {nu}")
            u_magmom = buffers[f"U_magmom_merged_{nu}"]
            index = buffers[f"center_index_{nu}"]
            widx = buffers[f"weight_index_{nu}"]
            if not use_center:
                # No centre moment supplied (e.g. a checkpoint predating this channel):
                # keep only the s=0 columns, which is the original scalar contraction.
                keep = index == 0
                u_magmom = u_magmom[..., keep]
                v = torch.ones(
                    y.shape[0],
                    int(keep.sum()),
                    dtype=self.weights_max.dtype,
                    device=y.device,
                )
                widx = widx[keep]
            else:
                v = centre_cols[:, index]
            # gather along Q: columns sharing an (s, path) share one weight
            weight_nu = self.merged_weights[f"nu{nu}"][species][:, :, widx, :]
            term = contract(
                equations[nu],
                self.U_tensors(nu),
                u_magmom,
                weight_nu,
                *([x] * nu),
                v,
            )
            out = term if out is None else (out + term)

        # (node, channel[, 2L+1]) -> (node, channel * (2L+1)); mul-major, matching e3nn/MACE.
        return out.view(out.shape[0], -1)

    def _use_center_spin(self, m_center: Optional[torch.Tensor]) -> bool:
        # getattr keeps models pickled before this channel existed loadable: they have
        # neither the flag nor the buffers, and fall back to the spin-scalar path.
        return (
            getattr(self, "center_spin_coupling", False)
            and m_center is not None
            and bool(getattr(self, "center_spin_orders", []))
        )

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

    def forward(
        self, x: torch.Tensor, y: torch.Tensor, m_center: Optional[torch.Tensor] = None
    ):
        # The reference contraction materializes a (spatial)^nu x (magmom)^nu scratch
        # tensor whose size scales with the node count; at large batches it dominates
        # both memory (OOM) and runtime (memory-bandwidth bound). Splitting the nodes
        # into chunks bounds that scratch tensor -- numerically identical, and in
        # practice both lower-memory AND faster. chunk_size=None keeps the original
        # single-shot path. getattr() keeps models pickled before this attr existed working.
        chunk_size = getattr(self, "chunk_size", None)
        if chunk_size is None or x.shape[0] <= chunk_size:
            return self.forward_reference(x, y, m_center)
        outs = [
            self.forward_reference(
                x[i : i + chunk_size],
                y[i : i + chunk_size],
                None if m_center is None else m_center[i : i + chunk_size],
            )
            for i in range(0, x.shape[0], chunk_size)
        ]
        return torch.cat(outs, dim=0)

    def U_tensors(self, nu: int):
        return dict(self.named_buffers())[f"U_matrix_{nu}"]

    def U_magmom_tensors(self, nu: int):
        return dict(self.named_buffers())[f"U_matrix_magmom_{nu}"]

    def U_magmom_center_tensors(self, s_order: int, nu: int):
        return dict(self.named_buffers())[f"U_matrix_magmom_s{s_order}_{nu}"]
