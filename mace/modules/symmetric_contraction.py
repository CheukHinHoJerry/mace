###########################################################################################
# Implementation of the symmetric contraction algorithm presented in the MACE paper
# (Batatia et al, MACE: Higher Order Equivariant Message Passing Neural Networks for Fast and Accurate Force Fields , Eq.10 and 11)
# Authors: Ilyes Batatia
# This program is distributed under the MIT License (see MIT.md)
###########################################################################################

from typing import Dict, Optional, Union

import logging
import opt_einsum_fx
import torch
import torch.fx
from e3nn import o3
from e3nn.util.codegen import CodeGenMixin
from e3nn.util.jit import compile_mode
from opt_einsum import contract

from mace.tools.cg import U_matrix_real


try:
    import cuequivariance as cue

    CUET_AVAILABLE = True
except ImportError:
    CUET_AVAILABLE = False

BATCH_EXAMPLE = 10
ALPHABET = ["w", "x", "v", "n", "z", "r", "t"]
ALPHABET_MAGMOM = ["y", "u", "o", "p", "s"] # ["w", "x", "v", "n", "z", "r", "t", "y", "u", "o", "p", "s"]
NONSOC_CONTRACTION_EQUATIONS = {
    1: "ik,lq,ekqa,bail,be->ba",
    2: "ijk,lmq,ekqa,bail,bajm,be->ba",
    3: "ijfk,lmgq,ekqa,bail,bajm,bafg,be->ba",
}


LOGGER = logging.getLogger(__name__)


# @compile_mode("script")
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
    ) -> None:
        super().__init__()

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
            else o3.Irreps('1x0e+1x1o')
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
        # import pdb; pdb.set_trace();
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
                )
            )

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        outs = [contraction(x, y) for contraction in self.contractions]
        return torch.cat(outs, dim=-1)
    
# @compile_mode("script")
# class NonSOCContraction(torch.nn.Module):
#     def __init__(
#         self,
#         irreps_in: o3.Irreps,
#         irrep_out: o3.Irreps,
#         correlation: int,
#         internal_weights: bool = True,
#         num_elements: Optional[int] = None,
#         weights: Optional[torch.Tensor] = None,
#         magmom_irreps: Optional[o3.Irreps] = None,
#     ) -> None:
#         super().__init__()

#         if num_elements is None:
#             raise ValueError("num_elements must be provided for NonSOCContraction")
#         if correlation < 1 or correlation > 3:
#             raise ValueError(
#                 f"NonSOCContraction currently supports correlation in [1, 3], got {correlation}"
#             )

#         # In this non-SOC path, channel index 'a' is multiplicity axis.
#         muls = [mul for mul, _ in irreps_in]
#         if len(set(muls)) != 1:
#             raise ValueError(
#                 f"NonSOCContraction expects uniform multiplicity in irreps_in; got muls={muls}"
#             )
#         self.num_features = int(muls[0])

#         self.coupling_irreps = o3.Irreps([ir.ir for ir in irreps_in])
#         magmom_irreps_full = (
#             o3.Irreps(magmom_irreps)
#             if magmom_irreps is not None
#             else o3.Irreps("1x0e+1x1o")
#         )
#         # Match standard contraction behavior: CG basis on irrep types (not multiplicity).
#         self.coupling_irreps_magmom = o3.Irreps([ir.ir for ir in magmom_irreps_full])

#         self.correlation = correlation
#         self.irrep_out = irrep_out

#         assert CUET_AVAILABLE, "cuequivariance library is required but not available."
#         dtype = torch.get_default_dtype()

#         # Build CG basis buffers
#         for nu in range(1, correlation + 1):
#             U = U_matrix_real(
#                 irreps_in=self.coupling_irreps,
#                 irreps_out=irrep_out,
#                 correlation=nu,
#                 dtype=dtype,
#                 use_cueq_cg=True,
#             )[-1]
#             self.register_buffer(f"U_matrix_{nu}", U)

#         for nu in range(1, correlation + 1):
#             Um = U_matrix_real(
#                 irreps_in=self.coupling_irreps_magmom,
#                 irreps_out=irrep_out,
#                 correlation=nu,
#                 dtype=dtype,
#                 use_cueq_cg=True,
#             )[-1]
#             self.register_buffer(f"U_matrix_magmom_{nu}", Um)

#         # Recursive contraction modules, mirroring Contraction class design.
#         self.contractions_weighting = torch.nn.ModuleList()
#         self.contractions_features = torch.nn.ModuleList()
#         self.weights = torch.nn.ParameterList([])

#         # Build from high->low, as Contraction does.
#         for i in range(correlation, 0, -1):
#             num_params = int(self.U_tensors(i).size(-1))
#             num_params_magmom = int(self.U_magmom_tensors(i).size(-1))
#             num_equivariance = 2 * irrep_out.lmax + 1
#             num_ell_r = int(self.U_tensors(i).size(-2))
#             num_ell_m = int(self.U_magmom_tensors(i).size(-2))

#             if i == correlation:
#                 parse_subscript_main = (
#                     [ALPHABET[j] for j in range(i + min(irrep_out.lmax, 1) - 1)]
#                     + ["ik,"] +
#                     [ALPHABET_MAGMOM[j] for j in range(i + min(irrep_out.lmax, 1) - 1)]
#                     + ["iq,ekqc,bci,be -> bc"]
#                     + [ALPHABET[j] for j in range(i + min(irrep_out.lmax, 1) - 1)]
#                     + [ALPHABET_MAGMOM[j] for j in range(i + min(irrep_out.lmax, 1) - 1)]
#                 )
#                 graph_module_main = torch.fx.symbolic_trace(
#                     lambda x, y, p, w, z: torch.einsum(
#                         "".join(parse_subscript_main), x, y, p, w, z
#                     )
#                 )

#                 # Optimizing the contractions
#                 self.graph_opt_main = opt_einsum_fx.optimize_einsums_full(
#                     model=graph_module_main,
#                     example_inputs=(
#                         torch.randn(
#                             [num_equivariance] + [num_ell_r] * i + [num_equivariance] + [num_ell_m] * i+ [num_params]
#                         ).squeeze(0),
#                         torch.randn(
#                             [num_equivariance] + [num_ell_m] * i + [num_equivariance] + [num_ell_m] * i+ [num_params]
#                         ).squeeze(0),
#                         torch.randn((num_elements, num_params, self.num_features)),
#                         torch.randn((BATCH_EXAMPLE, self.num_features, num_ell_r, num_ell_m)),
#                         torch.randn((BATCH_EXAMPLE, num_elements)),
#                     ),
#                 )

#                 self.weights_max = torch.nn.Parameter(
#                     torch.randn((num_elements, num_params, num_params_magmom, self.num_features), dtype=dtype)
#                     / (num_params * num_params_magmom)
#                 )
#             else:
#                 # Lower-order recursive updates:
#                 # c = contract_weights(U_i, Um_i, w_i, y)
#                 # out = contract_features(c + out, x)
#                 if i == 1:
#                     eq_w = "ik,lq,ekqa,be->bail"
#                     eq_f = "bail,bail->ba"
#                     ex_u_shape = (num_equivariance, num_ell_r, num_params)
#                     ex_um_shape = (num_equivariance, num_ell_m, num_params_magmom)
#                     ex_c_shape = (BATCH_EXAMPLE, self.num_features, num_ell_r, num_ell_m)
#                     ex_x_shape = (BATCH_EXAMPLE, self.num_features, num_ell_r, num_ell_m)
#                 else:  # i == 2
#                     eq_w = "ijk,lmq,ekqa,be->bajm"
#                     eq_f = "bajm,bail->ba"
#                     ex_u_shape = (num_equivariance, num_ell_r, num_ell_r, num_params)
#                     ex_um_shape = (num_equivariance, num_ell_m, num_ell_m, num_params_magmom)
#                     ex_c_shape = (BATCH_EXAMPLE, self.num_features, num_ell_r, num_ell_m)
#                     ex_x_shape = (BATCH_EXAMPLE, self.num_features, num_ell_r, num_ell_m)

#                 # graph_module_weighting = torch.fx.symbolic_trace(
#                 #     lambda u, um, w, y, eq_w=eq_w: torch.einsum(eq_w, u, um, w, y)
#                 # )
#                 # graph_module_features = torch.fx.symbolic_trace(
#                 #     lambda c, x, eq_f=eq_f: torch.einsum(eq_f, c, x)
#                 # )
#                 if i == 1:
#                     graph_module_weighting = torch.fx.symbolic_trace(
#                         lambda u, um, w, y: torch.einsum("ik,lq,ekqa,be->bail", u, um, w, y)
#                     )
#                     graph_module_features = torch.fx.symbolic_trace(
#                         lambda c, x: torch.einsum("bail,bail->ba", c, x)
#                     )
#                 else:  # i == 2
#                     graph_module_weighting = torch.fx.symbolic_trace(
#                         lambda u, um, w, y: torch.einsum("ijk,lmq,ekqa,be->bajm", u, um, w, y)
#                     )
#                     graph_module_features = torch.fx.symbolic_trace(
#                         lambda c, x: torch.einsum("bajm,bail->ba", c, x)
#                     )


#                 graph_opt_weighting = opt_einsum_fx.optimize_einsums_full(
#                     model=graph_module_weighting,
#                     example_inputs=(
#                         torch.randn(ex_u_shape, dtype=dtype).squeeze(0),
#                         torch.randn(ex_um_shape, dtype=dtype).squeeze(0),
#                         torch.randn((num_elements, num_params, num_params_magmom, self.num_features), dtype=dtype),
#                         torch.randn((BATCH_EXAMPLE, num_elements), dtype=dtype),
#                     ),
#                 )
#                 graph_opt_features = opt_einsum_fx.optimize_einsums_full(
#                     model=graph_module_features,
#                     example_inputs=(
#                         torch.randn(ex_c_shape, dtype=dtype),
#                         torch.randn(ex_x_shape, dtype=dtype),
#                     ),
#                 )

#                 self.contractions_weighting.append(graph_opt_weighting)
#                 self.contractions_features.append(graph_opt_features)

#                 w = torch.nn.Parameter(
#                     torch.randn((num_elements, num_params, num_params_magmom, self.num_features), dtype=dtype)
#                     / (num_params * num_params_magmom)
#                 )
#                 self.weights.append(w)

#         if not internal_weights:
#             self.weights = weights[:-1]
#             self.weights_max = weights[-1]

#     def forward(self, x: torch.Tensor, y: torch.Tensor):
#         assert self.irrep_out.lmax == 0

#         # Main term with fixed signature (U, U_mag, W, x, y)
#         out = self.graph_opt_main(
#             self.U_tensors(self.correlation),
#             self.U_magmom_tensors(self.correlation),
#             self.weights_max,
#             x,
#             y,
#         )

#         # Recursive lower-order corrections, matching Contraction.forward pattern.
#         for i, (weight, contract_weights, contract_features) in enumerate(
#             zip(self.weights, self.contractions_weighting, self.contractions_features)
#         ):
#             nu = self.correlation - i - 1
#             c_tensor = contract_weights(
#                 self.U_tensors(nu),
#                 self.U_magmom_tensors(nu),
#                 weight,
#                 y,
#             )
#             c_tensor = c_tensor + out
#             out = contract_features(c_tensor, x)

#         return out.view(out.shape[0], -1)

#     def U_tensors(self, nu: int):
#         return dict(self.named_buffers())[f"U_matrix_{nu}"]

#     def U_magmom_tensors(self, nu: int):
#         return dict(self.named_buffers())[f"U_matrix_magmom_{nu}"]


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
    ) -> None:
        super().__init__()

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
            else o3.Irreps('1x0e+1x1o')
        )
        # Match standard MACE contraction behavior: construct CG basis on irreps types,
        # not multiplicities, to keep contraction basis dimensions consistent.
        self.coupling_irreps_magmom = o3.Irreps([irrep.ir for irrep in magmom_irreps_full])
        self.correlation = correlation
        self.irrep_out = irrep_out
        assert CUET_AVAILABLE, "cuequivariance library is required but not available."
        
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
                use_cueq_cg=True
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
        for nu in range(1, correlation + 1):
            LOGGER.info(
                "[NonSOCContraction] U_matrix_magmom_%s with irreps_in=%s irreps_out=%s correlation=%s dtype=%s",
                nu,
                self.coupling_irreps_magmom,
                irrep_out,
                nu,
                dtype,
            )
            U_matrix = U_matrix_real(
                irreps_in=self.coupling_irreps_magmom,
                irreps_out=irrep_out,
                correlation=nu,
                dtype=dtype,
                use_cueq_cg=True
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
            num_equivariance = 2 * irrep_out.lmax + 1
            num_ell = self.U_tensors(i).size()[-2]

            if i == correlation:
                # Parameters for the product basis
                w = torch.nn.Parameter(
                    torch.randn((num_elements, num_params, num_params_magmom, self.num_features))
                    / (num_params * num_params_magmom)
                )
                self.weights_max = w
            else:
                # Parameters for lower-order product basis terms.
                # Keep list ordered by nu ascending: weights[0] -> nu=1, weights[1] -> nu=2, ...
                w = torch.nn.Parameter(
                    torch.randn((num_elements, num_params, num_params_magmom, self.num_features))
                    / (num_params * num_params_magmom)
                )
                lower_order_weights.append(w)
        # Rebuild in ascending nu order: weights[0] -> nu=1, weights[1] -> nu=2, ...
        if len(lower_order_weights) > 0:
            self.weights = torch.nn.ParameterList(list(reversed(lower_order_weights)))

        if not internal_weights:
            self.weights = weights[:-1]
            self.weights_max = weights[-1]

        self._ensure_optimized_contractions()
        self._ensure_sparse_paths()

    # def sparse_contract_nu2(self, U_paths, U_mag_paths, weight, x, y):
    #     idx_U, val_U = U_paths
    #     idx_M, val_M = U_mag_paths

    #     B = x.shape[0]
    #     out = torch.zeros(B, weight.shape[-1], device=x.device)

    #     for i in range(len(val_U)):
    #         # indices from U
    #         a, b, k = idx_U[i]   # adapt if ordering differs
    #         w_u = val_U[i]

    #         for j in range(len(val_M)):
    #             l, m, q = idx_M[j]
    #             w_m = val_M[j]

    #             # combine weights
    #             w_total = w_u * w_m

    #             # contraction (YOU MUST VERIFY INDEX MATCHING)
    #             term = (
    #                 w_total
    #                 * x[:, a]       # first x
    #                 * x[:, b]       # second x
    #                 * y[:, q]       # magmom
    #             )

    #             # apply learnable weight
    #             term = term.unsqueeze(-1) * weight[:, k, q, :]

    #             out += term.sum(dim=1)

    #     return out

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
                    NONSOC_CONTRACTION_EQUATIONS[3], u_t, um_t, w_t, x0_t, x1_t, x2_t, y_t
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
        if hasattr(self, "optimized_contractions"):
            return
        self.optimized_contractions = torch.nn.ModuleDict()
        for nu in range(1, self.correlation + 1):
            self.optimized_contractions[str(nu)] = self._build_optimized_contraction(nu)

    def _build_sparse_paths(self, nu: int) -> dict[str, torch.Tensor]:
        u = self.U_tensors(nu)
        um = self.U_magmom_tensors(nu)
        u_idx = (u != 0).nonzero(as_tuple=False)
        um_idx = (um != 0).nonzero(as_tuple=False)
        u_val = u[tuple(u_idx[:, dim] for dim in range(u_idx.shape[1]))]
        um_val = um[tuple(um_idx[:, dim] for dim in range(um_idx.shape[1]))]

        n_u = u_idx.shape[0]
        n_um = um_idx.shape[0]
        path_count = n_u * n_um

        coeff = (u_val[:, None] * um_val[None, :]).reshape(path_count)
        path = {"coeff": coeff}

        for dim, label in enumerate(("i", "j", "f")[:nu]):
            path[label] = u_idx[:, dim][:, None].expand(n_u, n_um).reshape(path_count)
        path["k"] = u_idx[:, nu][:, None].expand(n_u, n_um).reshape(path_count)

        for dim, label in enumerate(("l", "m", "g")[:nu]):
            path[label] = um_idx[None, :, dim].expand(n_u, n_um).reshape(path_count)
        path["q"] = um_idx[None, :, nu].expand(n_u, n_um).reshape(path_count)
        return path

    def _ensure_sparse_paths(self) -> None:
        if hasattr(self, "sparse_path_buffers"):
            return
        self.sparse_path_buffers = {}
        for nu in range(1, self.correlation + 1):
            path = self._build_sparse_paths(nu)
            self.sparse_path_buffers[nu] = tuple(path.keys())
            for key, value in path.items():
                self.register_buffer(f"sparse_nu{nu}_{key}", value)

    def _get_sparse_path(self, nu: int) -> dict[str, torch.Tensor]:
        self._ensure_sparse_paths()
        keys = self.sparse_path_buffers[nu]
        return {key: getattr(self, f"sparse_nu{nu}_{key}") for key in keys}

    def forward_reference(self, x: torch.Tensor, y: torch.Tensor):
        # x is of shape A_{i, k, lm, l'm'}
        # y is node_attrs one-hot (B, e)
        assert self.irrep_out.lmax == 0

        out = None
        for nu in range(1, self.correlation + 1):
            weight_nu = self.weights_max if nu == self.correlation else self.weights[nu - 1]

            if nu == 1:
                term = contract(
                    NONSOC_CONTRACTION_EQUATIONS[1],
                    self.U_tensors(nu),
                    self.U_magmom_tensors(nu),
                    weight_nu,
                    x,
                    y,
                )
            elif nu == 2:
                term = contract(
                    NONSOC_CONTRACTION_EQUATIONS[2],
                    self.U_tensors(nu),
                    self.U_magmom_tensors(nu),
                    weight_nu,
                    x,
                    x,
                    y,
                )
            elif nu == 3:
                term = contract(
                    NONSOC_CONTRACTION_EQUATIONS[3],
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

    def forward_optimized(self, x: torch.Tensor, y: torch.Tensor):
        assert self.irrep_out.lmax == 0
        self._ensure_optimized_contractions()

        out = None
        for nu in range(1, self.correlation + 1):
            weight_nu = self.weights_max if nu == self.correlation else self.weights[nu - 1]
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

    def forward_sparse(self, x: torch.Tensor, y: torch.Tensor):
        assert self.irrep_out.lmax == 0
        self._ensure_sparse_paths()

        out = None
        for nu in range(1, self.correlation + 1):
            weight_nu = self.weights_max if nu == self.correlation else self.weights[nu - 1]
            path = self._get_sparse_path(nu)
            coeff = path["coeff"]
            w = weight_nu[:, path["k"], path["q"], :]

            if nu == 1:
                x0 = x[:, :, path["i"], path["l"]]
                term = torch.einsum("epa,bap,be,p->ba", w, x0, y, coeff)
            elif nu == 2:
                x0 = x[:, :, path["i"], path["l"]]
                x1 = x[:, :, path["j"], path["m"]]
                term = torch.einsum("epa,bap,bap,be,p->ba", w, x0, x1, y, coeff)
            elif nu == 3:
                x0 = x[:, :, path["i"], path["l"]]
                x1 = x[:, :, path["j"], path["m"]]
                x2 = x[:, :, path["f"], path["g"]]
                term = torch.einsum("epa,bap,bap,bap,be,p->ba", w, x0, x1, x2, y, coeff)
            else:
                raise ValueError(f"Unsupported correlation order: {nu}")

            out = term if out is None else (out + term)

        return out.view(out.shape[0], -1)

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        return self.forward_optimized(x, y)
    
    def U_tensors(self, nu: int):
        return dict(self.named_buffers())[f"U_matrix_{nu}"]

    def U_magmom_tensors(self, nu: int):
        return dict(self.named_buffers())[f"U_matrix_magmom_{nu}"]


    # def forward(self, x: torch.Tensor, y: torch.Tensor):
    #     # x is of shape A_{i, k, lm, l'm'}
    #     # 'cg_r, cg_m, A_{k, lm, l',m}

    #     assert self.irrep_out.lmax == 0
    #     bs = x.shape[0]
    #     outs_dict = dict()

    #     parse_instruction_1 = ([ALPHABET[j] for j in range(1 + min(self.irrep_out.lmax, 1) - 1)]+ ["ik,"] + [ALPHABET_MAGMOM[j] for j in range(1 + min(self.irrep_out.lmax, 1) - 1)]+ ["lq,"]+ ["ekqa,bail,be->ba"] + [ALPHABET[j] for j in range(1 + min(self.irrep_out.lmax, 1) - 1)] + [ALPHABET_MAGMOM[j] for j in range(1 + min(self.irrep_out.lmax, 1) - 1)])

    #     parse_instruction_2 = ([ALPHABET[j] for j in range(1 + min(self.irrep_out.lmax, 1) - 1)] + ["ijk,"] + [ALPHABET_MAGMOM[j] for j in range(1 + min(self.irrep_out.lmax, 1) - 1)] + ["lmq,"] + ["ekqa,bail,bajm,be->ba"] + [ALPHABET[j] for j in range(1 + min(self.irrep_out.lmax, 1) - 1)] + [ALPHABET_MAGMOM[j] for j in range(1 + min(self.irrep_out.lmax, 1) - 1)])
    #     parse_instruction_3 = ([ALPHABET[j] for j in range(1 + min(self.irrep_out.lmax, 1) - 1)] + ["ijfk,"] + [ALPHABET_MAGMOM[j] for j in range(1 + min(self.irrep_out.lmax, 1) - 1)] + ["lmgq,"] + ["ekqa,bail,bajm,bafg,be->ba"] + [ALPHABET[j] for j in range(1 + min(self.irrep_out.lmax, 1) - 1)] + [ALPHABET_MAGMOM[j] for j in range(1 + min(self.irrep_out.lmax, 1) - 1)])
    #     parse_instruction_list = ["".join(parse_instruction_1), "".join(parse_instruction_2), "".join(parse_instruction_3)]
        
    #     for nu in range(1, self.correlation + 1):
    #         # Prefer the top-order tensor for nu==correlation.
    #         # For lower orders, use weights[nu-1] when available.
    #         if nu == self.correlation:
    #             weight_nu = self.weights_max
    #         else:
    #             weight_nu = self.weights[nu - 1]

    #         LOGGER.info(
    #             "[NonSOCContraction] forward nu=%s U=%s U_mag=%s W=%s x=%s y=%s",
    #             nu,
    #             tuple(self.U_tensors(nu).shape),
    #             tuple(self.U_magmom_tensors(nu).shape),
    #             tuple(weight_nu.shape),
    #             tuple(x.shape),
    #             tuple(y.shape),
    #         )

    #         if nu == 1:
    #             outs_dict[nu] = contract(
    #                 parse_instruction_list[nu - 1],
    #                 self.U_tensors(nu),
    #                 self.U_magmom_tensors(nu),
    #                 weight_nu,
    #                 x,
    #                 y,
    #             )
    #         elif nu == 2:
    #             outs_dict[nu] = contract(
    #                 parse_instruction_list[nu - 1],
    #                 self.U_tensors(nu),
    #                 self.U_magmom_tensors(nu),
    #                 weight_nu,
    #                 x,
    #                 x,
    #                 y,
    #             )
    #         elif nu == 3:
    #             outs_dict[nu] = contract(
    #                 parse_instruction_list[nu - 1],
    #                 self.U_tensors(nu),
    #                 self.U_magmom_tensors(nu),
    #                 weight_nu,
    #                 x,
    #                 x,
    #                 x,
    #                 y,
    #             )
        
    #     out = outs_dict[self.correlation]
    #     for nu in range(self.correlation - 1, 0, -1):
    #         out += outs_dict[nu]
    #     return out.view(out.shape[0], -1)

    # def U_tensors(self, nu: int):
    #     return dict(self.named_buffers())[f"U_matrix_{nu}"]
    
    # def U_magmom_tensors(self, nu: int):
    #     return dict(self.named_buffers())[f"U_matrix_magmom_{nu}"]


@compile_mode("script")
class SymmetricContraction(CodeGenMixin, torch.nn.Module):
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
    ) -> None:
        super().__init__()

        if irrep_normalization is None:
            irrep_normalization = "component"

        if path_normalization is None:
            path_normalization = "element"

        assert irrep_normalization in ["component", "norm", "none"]
        assert path_normalization in ["element", "path", "none"]

        self.irreps_in = o3.Irreps(irreps_in)
        self.irreps_out = o3.Irreps(irreps_out)
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
                Contraction(
                    irreps_in=self.irreps_in,
                    irrep_out=o3.Irreps(str(irrep_out.ir)),
                    correlation=correlation[irrep_out],
                    internal_weights=self.internal_weights,
                    num_elements=num_elements,
                    weights=self.shared_weights,
                )
            )

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        outs = [contraction(x, y) for contraction in self.contractions]
        return torch.cat(outs, dim=-1)


@compile_mode("script")
class Contraction(torch.nn.Module):
    def __init__(
        self,
        irreps_in: o3.Irreps,
        irrep_out: o3.Irreps,
        correlation: int,
        internal_weights: bool = True,
        num_elements: Optional[int] = None,
        weights: Optional[torch.Tensor] = None,
    ) -> None:
        super().__init__()

        self.num_features = irreps_in.count((0, 1))
        self.coupling_irreps = o3.Irreps([irrep.ir for irrep in irreps_in])
        self.correlation = correlation
        dtype = torch.get_default_dtype()
        for nu in range(1, correlation + 1):
            U_matrix = U_matrix_real(
                irreps_in=self.coupling_irreps,
                irreps_out=irrep_out,
                correlation=nu,
                dtype=dtype,
            )[-1]
            self.register_buffer(f"U_matrix_{nu}", U_matrix)

        # Tensor contraction equations
        self.contractions_weighting = torch.nn.ModuleList()
        self.contractions_features = torch.nn.ModuleList()

        # Create weight for product basis
        self.weights = torch.nn.ParameterList([])

        for i in range(correlation, 0, -1):
            # Shapes definying
            num_params = self.U_tensors(i).size()[-1]
            num_equivariance = 2 * irrep_out.lmax + 1
            num_ell = self.U_tensors(i).size()[-2]

            if i == correlation:
                parse_subscript_main = (
                    [ALPHABET[j] for j in range(i + min(irrep_out.lmax, 1) - 1)]
                    + ["ik,ekc,bci,be -> bc"]
                    + [ALPHABET[j] for j in range(i + min(irrep_out.lmax, 1) - 1)]
                )
                graph_module_main = torch.fx.symbolic_trace(
                    lambda x, y, w, z: torch.einsum(
                        "".join(parse_subscript_main), x, y, w, z
                    )
                )

                # Optimizing the contractions
                self.graph_opt_main = opt_einsum_fx.optimize_einsums_full(
                    model=graph_module_main,
                    example_inputs=(
                        torch.randn(
                            [num_equivariance] + [num_ell] * i + [num_params]
                        ).squeeze(0),
                        torch.randn((num_elements, num_params, self.num_features)),
                        torch.randn((BATCH_EXAMPLE, self.num_features, num_ell)),
                        torch.randn((BATCH_EXAMPLE, num_elements)),
                    ),
                )
                # Parameters for the product basis
                w = torch.nn.Parameter(
                    torch.randn((num_elements, num_params, self.num_features))
                    / num_params
                )
                self.weights_max = w
            else:
                # Generate optimized contractions equations
                parse_subscript_weighting = (
                    [ALPHABET[j] for j in range(i + min(irrep_out.lmax, 1))]
                    + ["k,ekc,be->bc"]
                    + [ALPHABET[j] for j in range(i + min(irrep_out.lmax, 1))]
                )
                parse_subscript_features = (
                    ["bc"]
                    + [ALPHABET[j] for j in range(i - 1 + min(irrep_out.lmax, 1))]
                    + ["i,bci->bc"]
                    + [ALPHABET[j] for j in range(i - 1 + min(irrep_out.lmax, 1))]
                )

                # Symbolic tracing of contractions
                graph_module_weighting = torch.fx.symbolic_trace(
                    lambda x, y, z: torch.einsum(
                        "".join(parse_subscript_weighting), x, y, z
                    )
                )
                graph_module_features = torch.fx.symbolic_trace(
                    lambda x, y: torch.einsum("".join(parse_subscript_features), x, y)
                )

                # Optimizing the contractions
                graph_opt_weighting = opt_einsum_fx.optimize_einsums_full(
                    model=graph_module_weighting,
                    example_inputs=(
                        torch.randn(
                            [num_equivariance] + [num_ell] * i + [num_params]
                        ).squeeze(0),
                        torch.randn((num_elements, num_params, self.num_features)),
                        torch.randn((BATCH_EXAMPLE, num_elements)),
                    ),
                )
                graph_opt_features = opt_einsum_fx.optimize_einsums_full(
                    model=graph_module_features,
                    example_inputs=(
                        torch.randn(
                            [BATCH_EXAMPLE, self.num_features, num_equivariance]
                            + [num_ell] * i
                        ).squeeze(2),
                        torch.randn((BATCH_EXAMPLE, self.num_features, num_ell)),
                    ),
                )
                self.contractions_weighting.append(graph_opt_weighting)
                self.contractions_features.append(graph_opt_features)
                # Parameters for the product basis
                w = torch.nn.Parameter(
                    torch.randn((num_elements, num_params, self.num_features))
                    / num_params
                )
                self.weights.append(w)
        if not internal_weights:
            self.weights = weights[:-1]
            self.weights_max = weights[-1]

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        out = self.graph_opt_main(
            self.U_tensors(self.correlation),
            self.weights_max,
            x,
            y,
        )
        for i, (weight, contract_weights, contract_features) in enumerate(
            zip(self.weights, self.contractions_weighting, self.contractions_features)
        ):
            c_tensor = contract_weights(
                self.U_tensors(self.correlation - i - 1),
                weight,
                y,
            )
            c_tensor = c_tensor + out
            out = contract_features(c_tensor, x)

        return out.view(out.shape[0], -1)

    def U_tensors(self, nu: int):
        return dict(self.named_buffers())[f"U_matrix_{nu}"]
