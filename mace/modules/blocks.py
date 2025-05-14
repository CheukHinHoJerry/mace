###########################################################################################
# Elementary Block for Building O(3) Equivariant Higher Order Message Passing Neural Network
# Authors: Ilyes Batatia, Gregor Simm
# This program is distributed under the MIT License (see MIT.md)
###########################################################################################

from abc import abstractmethod
from typing import Callable, List, Optional, Tuple, Union

import numpy as np
import torch.nn.functional
from e3nn import nn, o3
from e3nn.util.jit import compile_mode

from mace.modules.wrapper_ops import (
    CuEquivarianceConfig,
    FullyConnectedTensorProduct,
    Linear,
    SymmetricContractionWrapper,
    TensorProduct,
)
from mace.tools.compile import simplify_if_compile
from mace.tools.scatter import scatter_sum

from .irreps_tools import (
    linear_out_irreps,
    mask_head,
    reshape_irreps,
    inverse_reshape_irreps,
    tp_out_irreps_with_instructions,
    tp_out_irreps_with_instructions_magmom,
)
from .radial import (
    AgnesiTransform,
    BesselBasis,
    ChebychevBasis,
    GaussianBasis,
    PolynomialCutoff,
    SoftTransform,
)


@compile_mode("script")
class LinearNodeEmbeddingBlock(torch.nn.Module):
    def __init__(
        self,
        irreps_in: o3.Irreps,
        irreps_out: o3.Irreps,
        cueq_config: Optional[CuEquivarianceConfig] = None,
    ):
        super().__init__()
        self.linear = Linear(
            irreps_in=irreps_in, irreps_out=irreps_out, cueq_config=cueq_config
        )

    def forward(
        self,
        node_attrs: torch.Tensor,
    ) -> torch.Tensor:  # [n_nodes, irreps]
        return self.linear(node_attrs)


@compile_mode("script")
class LinearReadoutBlock(torch.nn.Module):
    def __init__(
        self,
        irreps_in: o3.Irreps,
        irrep_out: o3.Irreps = o3.Irreps("0e"),
        cueq_config: Optional[CuEquivarianceConfig] = None,
    ):
        super().__init__()
        self.linear = Linear(
            irreps_in=irreps_in, irreps_out=irrep_out, cueq_config=cueq_config
        )

    def forward(
        self,
        x: torch.Tensor,
        heads: Optional[torch.Tensor] = None,  # pylint: disable=unused-argument
    ) -> torch.Tensor:  # [n_nodes, irreps]  # [..., ]
        return self.linear(x)  # [n_nodes, 1]


@simplify_if_compile
@compile_mode("script")
class NonLinearReadoutBlock(torch.nn.Module):
    def __init__(
        self,
        irreps_in: o3.Irreps,
        MLP_irreps: o3.Irreps,
        gate: Optional[Callable],
        irrep_out: o3.Irreps = o3.Irreps("0e"),
        num_heads: int = 1,
        cueq_config: Optional[CuEquivarianceConfig] = None,
    ):
        super().__init__()
        self.hidden_irreps = MLP_irreps
        self.num_heads = num_heads
        self.linear_1 = Linear(
            irreps_in=irreps_in, irreps_out=self.hidden_irreps, cueq_config=cueq_config
        )
        self.non_linearity = nn.Activation(irreps_in=self.hidden_irreps, acts=[gate])
        self.linear_2 = Linear(
            irreps_in=self.hidden_irreps, irreps_out=irrep_out, cueq_config=cueq_config
        )

    def forward(
        self, x: torch.Tensor, heads: Optional[torch.Tensor] = None
    ) -> torch.Tensor:  # [n_nodes, irreps]  # [..., ]
        x = self.non_linearity(self.linear_1(x))
        if hasattr(self, "num_heads"):
            if self.num_heads > 1 and heads is not None:
                x = mask_head(x, heads, self.num_heads)
        return self.linear_2(x)  # [n_nodes, len(heads)]


@simplify_if_compile
@compile_mode("script")
class LinearTPReadoutBlock(torch.nn.Module):
    def __init__(
        self,
        irreps_in1: o3.Irreps,
        irreps_in2: o3.Irreps,
        irrep_out: o3.Irreps = o3.Irreps("0e"),
        cueq_config: Optional[CuEquivarianceConfig] = None,
    ):
        super().__init__()
        self.mixing_tp = FullyConnectedTensorProduct(
            irreps_in1=irreps_in1, irreps_in2=irreps_in2, irreps_out=irrep_out, cueq_config=cueq_config
        )

    def forward(
        self, x: torch.Tensor, y: torch.Tensor, 
        heads: Optional[torch.Tensor] = None
    ) -> torch.Tensor:  # [n_nodes, irreps]  # [..., ]
        return self.mixing_tp(x, y)

@compile_mode("script")
class LinearDipoleReadoutBlock(torch.nn.Module):
    def __init__(
        self,
        irreps_in: o3.Irreps,
        dipole_only: bool = False,
        cueq_config: Optional[CuEquivarianceConfig] = None,
    ):
        super().__init__()
        if dipole_only:
            self.irreps_out = o3.Irreps("1x1o")
        else:
            self.irreps_out = o3.Irreps("1x0e + 1x1o")
        self.linear = Linear(
            irreps_in=irreps_in, irreps_out=self.irreps_out, cueq_config=cueq_config
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # [n_nodes, irreps]  # [..., ]
        return self.linear(x)  # [n_nodes, 1]


@compile_mode("script")
class NonLinearDipoleReadoutBlock(torch.nn.Module):
    def __init__(
        self,
        irreps_in: o3.Irreps,
        MLP_irreps: o3.Irreps,
        gate: Callable,
        dipole_only: bool = False,
        cueq_config: Optional[CuEquivarianceConfig] = None,
    ):
        super().__init__()
        self.hidden_irreps = MLP_irreps
        if dipole_only:
            self.irreps_out = o3.Irreps("1x1o")
        else:
            self.irreps_out = o3.Irreps("1x0e + 1x1o")
        irreps_scalars = o3.Irreps(
            [(mul, ir) for mul, ir in MLP_irreps if ir.l == 0 and ir in self.irreps_out]
        )
        irreps_gated = o3.Irreps(
            [(mul, ir) for mul, ir in MLP_irreps if ir.l > 0 and ir in self.irreps_out]
        )
        irreps_gates = o3.Irreps([mul, "0e"] for mul, _ in irreps_gated)
        self.equivariant_nonlin = nn.Gate(
            irreps_scalars=irreps_scalars,
            act_scalars=[gate for _, ir in irreps_scalars],
            irreps_gates=irreps_gates,
            act_gates=[gate] * len(irreps_gates),
            irreps_gated=irreps_gated,
        )
        self.irreps_nonlin = self.equivariant_nonlin.irreps_in.simplify()
        self.linear_1 = Linear(
            irreps_in=irreps_in, irreps_out=self.irreps_nonlin, cueq_config=cueq_config
        )
        self.linear_2 = Linear(
            irreps_in=self.hidden_irreps,
            irreps_out=self.irreps_out,
            cueq_config=cueq_config,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # [n_nodes, irreps]  # [..., ]
        x = self.equivariant_nonlin(self.linear_1(x))
        return self.linear_2(x)  # [n_nodes, 1]


@compile_mode("script")
class AtomicEnergiesBlock(torch.nn.Module):
    atomic_energies: torch.Tensor

    def __init__(self, atomic_energies: Union[np.ndarray, torch.Tensor]):
        super().__init__()
        # assert len(atomic_energies.shape) == 1

        self.register_buffer(
            "atomic_energies",
            torch.tensor(atomic_energies, dtype=torch.get_default_dtype()),
        )  # [n_elements, n_heads]

    def forward(
        self, x: torch.Tensor  # one-hot of elements [..., n_elements]
    ) -> torch.Tensor:  # [..., ]
        return torch.matmul(x, torch.atleast_2d(self.atomic_energies).T)

    def __repr__(self):
        formatted_energies = ", ".join(
            [
                "[" + ", ".join([f"{x:.4f}" for x in group]) + "]"
                for group in torch.atleast_2d(self.atomic_energies)
            ]
        )
        return f"{self.__class__.__name__}(energies=[{formatted_energies}])"


@compile_mode("script")
class RadialEmbeddingBlock(torch.nn.Module):
    def __init__(
        self,
        r_max: float,
        num_bessel: int,
        num_polynomial_cutoff: int,
        radial_type: str = "bessel",
        distance_transform: str = "None",
    ):
        super().__init__()
        if radial_type == "bessel":
            self.bessel_fn = BesselBasis(r_max=r_max, num_basis=num_bessel)
        elif radial_type == "gaussian":
            self.bessel_fn = GaussianBasis(r_max=r_max, num_basis=num_bessel)
        elif radial_type == "chebyshev":
            self.bessel_fn = ChebychevBasis(r_max=r_max, num_basis=num_bessel)
        if distance_transform == "Agnesi":
            self.distance_transform = AgnesiTransform()
        elif distance_transform == "Soft":
            self.distance_transform = SoftTransform()
        self.cutoff_fn = PolynomialCutoff(r_max=r_max, p=num_polynomial_cutoff)
        self.out_dim = num_bessel

    def forward(
        self,
        edge_lengths: torch.Tensor,  # [n_edges, 1]
        node_attrs: torch.Tensor,
        edge_index: torch.Tensor,
        atomic_numbers: torch.Tensor,
    ):
        cutoff = self.cutoff_fn(edge_lengths)  # [n_edges, 1]
        if hasattr(self, "distance_transform"):
            edge_lengths = self.distance_transform(
                edge_lengths, node_attrs, edge_index, atomic_numbers
            )
        radial = self.bessel_fn(edge_lengths)  # [n_edges, n_basis]
        if self.cutoff_fn.p != 0:
            radial = radial * cutoff  # [n_edges, n_basis]
        return radial


@compile_mode("script")
class EquivariantProductBasisBlock(torch.nn.Module):
    def __init__(
        self,
        node_feats_irreps: o3.Irreps,
        target_irreps: o3.Irreps,
        correlation: int,
        use_sc: bool = True,
        num_elements: Optional[int] = None,
        cueq_config: Optional[CuEquivarianceConfig] = None,
        contraction_cls: Optional[str] = "SymmetricContraction"
    ) -> None:
        super().__init__()

        self.use_sc = use_sc
        self.contraction_cls = contraction_cls
        if contraction_cls == "SymmetricContraction":
            self.symmetric_contractions = SymmetricContractionWrapper(
                irreps_in=node_feats_irreps,
                irreps_out=target_irreps,
                correlation=correlation,
                num_elements=num_elements,
                cueq_config=cueq_config,
            )
        else:
            raise ValueError("Contraction class not supported")
        # Update linear
        self.linear = Linear(
            target_irreps,
            target_irreps,
            internal_weights=True,
            shared_weights=True,
            cueq_config=cueq_config,
        )
        self.cueq_config = cueq_config

    def forward(
        self,
        node_feats: torch.Tensor,
        sc: Optional[torch.Tensor],
        node_attrs: torch.Tensor,
    ) -> torch.Tensor:
        use_cueq = False
        use_cueq_mul_ir = False
        if hasattr(self, "cueq_config"):
            if self.cueq_config is not None:
                if self.cueq_config.enabled and (
                    self.cueq_config.optimize_all or self.cueq_config.optimize_symmetric
                ):
                    use_cueq = True
                if self.cueq_config.layout_str == "mul_ir":
                    use_cueq_mul_ir = True
        if use_cueq:
            if use_cueq_mul_ir:
                node_feats = torch.transpose(node_feats, 1, 2)
            index_attrs = torch.nonzero(node_attrs)[:, 1].int()
            node_feats = self.symmetric_contractions(
                node_feats.flatten(1),
                index_attrs,
            )
        else:
            node_feats = self.symmetric_contractions(node_feats, node_attrs)
        if self.use_sc and sc is not None:
            return self.linear(node_feats) + sc
        return self.linear(node_feats)


@compile_mode("script")
class EquivariantProductBasisWithSelfMagmomBlock(torch.nn.Module):
    def __init__(
        self,
        node_feats_irreps: o3.Irreps,
        target_irreps: o3.Irreps,
        magmom_node_inv_feats_irreps: o3.Irreps,
        magmom_node_attrs_irreps: o3.Irreps,
        correlation: int,
        use_sc: bool = True,
        num_elements: Optional[int] = None,
        cueq_config: Optional[CuEquivarianceConfig] = None,
        contraction_cls: Optional[str] = "SymmetricContraction"
    ) -> None:
        super().__init__()

        self.use_sc = use_sc
        self.magmom_node_inv_feats_irreps = magmom_node_inv_feats_irreps
        self.magmom_node_attrs_irreps = magmom_node_attrs_irreps
        self.cueq_config = cueq_config
        self.contraction_cls = contraction_cls
        
        if contraction_cls == "SymmetricContraction":
            self.symmetric_contractions = SymmetricContractionWrapper(
                irreps_in=node_feats_irreps,
                irreps_out=target_irreps,
                correlation=correlation,
                num_elements=num_elements,
                cueq_config=cueq_config,
            )
        else:
            raise ValueError("Contraction class not supported")

        # interaction with self magnetic moment
        irreps_mid, instructions = tp_out_irreps_with_instructions(
            target_irreps,
            self.magmom_node_attrs_irreps,
            target_irreps,
        )
        self.conv_tp = TensorProduct(
            target_irreps,
            self.magmom_node_attrs_irreps,
            irreps_mid,
            instructions=instructions,
            shared_weights=False,
            internal_weights=False,
            cueq_config=self.cueq_config,
        )
        magmom_input_dim = self.magmom_node_inv_feats_irreps.num_irreps
        self.conv_tp_weights =  nn.FullyConnectedNet(
            [magmom_input_dim] + [64, 64, 64] + [self.conv_tp.weight_numel],
            torch.nn.functional.silu,
        )
        # self.conv_tp_weights =  nn.FullyConnectedNet(
        #     [magmom_input_dim] + [self.conv_tp.weight_numel],
        #     None,
        # )
        # Update linear
        self.linear = Linear(
            self.conv_tp.irreps_out,
            target_irreps,
            internal_weights=True,
            shared_weights=True,
            cueq_config=cueq_config,
        )
        self.linear_ori = Linear(
            target_irreps,
            target_irreps,
            internal_weights=True,
            shared_weights=True,
            cueq_config=cueq_config,
        )
    def forward(
        self,
        node_feats: torch.Tensor,
        sc: Optional[torch.Tensor],
        node_attrs: torch.Tensor,
        magmom_node_inv_feats: torch.Tensor,
        magmom_node_attrs: torch.Tensor,
    ) -> torch.Tensor:
        use_cueq = False
        use_cueq_mul_ir = False
        if hasattr(self, "cueq_config"):
            if self.cueq_config is not None:
                if self.cueq_config.enabled and (
                    self.cueq_config.optimize_all or self.cueq_config.optimize_symmetric
                ):
                    use_cueq = True
                if self.cueq_config.layout_str == "mul_ir":
                    use_cueq_mul_ir = True
        if use_cueq:
            if use_cueq_mul_ir:
                node_feats = torch.transpose(node_feats, 1, 2)
            index_attrs = torch.nonzero(node_attrs)[:, 1].int()
            node_feats = self.symmetric_contractions(
                node_feats.flatten(1),
                index_attrs,
            )
        else:
            node_feats = self.symmetric_contractions(node_feats, node_attrs)

        # interaction with magnectic moment
        tp_weights = self.conv_tp_weights(magmom_node_inv_feats)

        out = self.conv_tp(node_feats, magmom_node_attrs, tp_weights)
        
        # out = node_feats
        if self.use_sc and sc is not None:
            out_message = self.linear(out) + self.linear_ori(node_feats) + sc
        else:
            out_message = self.linear(out) + self.linear_ori(node_feats)
        return out_message

@compile_mode("script")
class EquivariantProductBasisWithOneBodySelfMagmomBlock(torch.nn.Module):
    def __init__(
        self,
        node_feats_irreps: o3.Irreps,
        target_irreps: o3.Irreps,
        magmom_node_inv_feats_irreps: o3.Irreps,
        magmom_node_attrs_irreps: o3.Irreps,
        correlation: int,
        use_sc: bool = True,
        num_elements: Optional[int] = None,
        cueq_config: Optional[CuEquivarianceConfig] = None,
        contraction_cls: Optional[str] = "SymmetricContraction"
    ) -> None:
        super().__init__()

        self.use_sc = use_sc
        self.magmom_node_inv_feats_irreps = magmom_node_inv_feats_irreps
        self.magmom_node_attrs_irreps = magmom_node_attrs_irreps
        self.cueq_config = cueq_config
        self.contraction_cls = contraction_cls
        
        if contraction_cls == "SymmetricContraction":
            self.symmetric_contractions = SymmetricContractionWrapper(
                irreps_in=node_feats_irreps,
                irreps_out=target_irreps,
                correlation=correlation,
                num_elements=num_elements,
                cueq_config=cueq_config,
            )
        else:
            raise ValueError("Contraction class not supported")

        # interaction with self magnetic moment
        irreps_mid, instructions = tp_out_irreps_with_instructions(
            target_irreps,
            self.magmom_node_attrs_irreps,
            target_irreps,
        )
        self.conv_tp = TensorProduct(
            target_irreps,
            self.magmom_node_attrs_irreps,
            irreps_mid,
            instructions=instructions,
            shared_weights=False,
            internal_weights=False,
            cueq_config=self.cueq_config,
        )
        magmom_input_dim = self.magmom_node_inv_feats_irreps.num_irreps
        self.conv_tp_weights =  nn.FullyConnectedNet(
            [magmom_input_dim] + [64, 64, 64] + [self.conv_tp.weight_numel],
            torch.nn.functional.silu,
        )

        # get invariant dimension
        self.onebody_magmombasis = nn.FullyConnectedNet(
            [magmom_input_dim] + [64, 64, 64] + [target_irreps[0].mul],
            torch.nn.functional.silu,
        )
        # TODO: add this for (1 - exp(-alpha * x))
        # self.species_dependent_transform = Linear(
            
        # )
        self.exp_scaling = torch.nn.Parameter(torch.tensor(5.0, requires_grad=True))

        # Update linear
        self.linear = Linear(
            self.conv_tp.irreps_out,
            target_irreps,
            internal_weights=True,
            shared_weights=True,
            cueq_config=cueq_config,
        )
        self.linear_ori = Linear(
            target_irreps,
            target_irreps,
            internal_weights=True,
            shared_weights=True,
            cueq_config=cueq_config,
        )

    def forward(
        self,
        node_feats: torch.Tensor,
        sc: Optional[torch.Tensor],
        node_attrs: torch.Tensor,
        magmom_node_inv_feats: torch.Tensor,
        magmom_node_attrs: torch.Tensor,
        magmom_lenghts: torch.Tensor,
    ) -> torch.Tensor:
        use_cueq = False
        use_cueq_mul_ir = False
        if hasattr(self, "cueq_config"):
            if self.cueq_config is not None:
                if self.cueq_config.enabled and (
                    self.cueq_config.optimize_all or self.cueq_config.optimize_symmetric
                ):
                    use_cueq = True
                if self.cueq_config.layout_str == "mul_ir":
                    use_cueq_mul_ir = True
        if use_cueq:
            if use_cueq_mul_ir:
                node_feats = torch.transpose(node_feats, 1, 2)
            index_attrs = torch.nonzero(node_attrs)[:, 1].int()
            node_feats = self.symmetric_contractions(
                node_feats.flatten(1),
                index_attrs,
            )
        else:
            node_feats = self.symmetric_contractions(node_feats, node_attrs)


        # interaction with magnectic moment
        tp_weights = self.conv_tp_weights(magmom_node_inv_feats)

        out = self.conv_tp(node_feats, magmom_node_attrs, tp_weights)

        # add self magmom one body contribution for large volume limit
        # invariant dimension

        if len(out.shape) == 2:
            out += (1 - torch.exp(-self.exp_scaling * magmom_lenghts)) * self.onebody_magmombasis(magmom_node_inv_feats)
        else:
            out[:, :, 0] += magmom_lenghts.unsqueeze(-1) * self.onebody_magmombasis(magmom_node_inv_feats)

        # out = node_feats
        if self.use_sc and sc is not None:
            out_message = self.linear(out) + self.linear_ori(node_feats) + sc
        else:
            out_message = self.linear(out) + self.linear_ori(node_feats)
        return out_message

@compile_mode("script")
class InteractionBlock(torch.nn.Module):
    def __init__(
        self,
        node_attrs_irreps: o3.Irreps,
        node_feats_irreps: o3.Irreps,
        edge_attrs_irreps: o3.Irreps,
        edge_feats_irreps: o3.Irreps,
        target_irreps: o3.Irreps,
        hidden_irreps: o3.Irreps,
        avg_num_neighbors: float,
        radial_MLP: Optional[List[int]] = None,
        cueq_config: Optional[CuEquivarianceConfig] = None,
    ) -> None:
        super().__init__()
        self.node_attrs_irreps = node_attrs_irreps
        self.node_feats_irreps = node_feats_irreps
        self.edge_attrs_irreps = edge_attrs_irreps
        self.edge_feats_irreps = edge_feats_irreps
        self.target_irreps = target_irreps
        self.hidden_irreps = hidden_irreps
        self.avg_num_neighbors = avg_num_neighbors
        if radial_MLP is None:
            radial_MLP = [64, 64, 64]
        self.radial_MLP = radial_MLP
        self.cueq_config = cueq_config

        self._setup()

    @abstractmethod
    def _setup(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def forward(
        self,
        node_attrs: torch.Tensor,
        node_feats: torch.Tensor,
        edge_attrs: torch.Tensor,
        edge_feats: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        raise NotImplementedError

@compile_mode("script")
class MagneticInteractionBlock(InteractionBlock):
    def __init__(
        self,
        magmom_node_inv_feats_irreps: Optional[o3.Irreps] = None,
        magmom_node_attrs_irreps: Optional[o3.Irreps] = None,
        **kwargs,
    ) -> None:
        self.magmom_node_inv_feats_irreps = magmom_node_inv_feats_irreps
        self.magmom_node_attrs_irreps = magmom_node_attrs_irreps
        super().__init__(**kwargs)

    @abstractmethod
    def _setup(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def forward(
        self,
        node_attrs: torch.Tensor,
        node_feats: torch.Tensor,
        edge_attrs: torch.Tensor,
        edge_feats: torch.Tensor,
        edge_index: torch.Tensor,
        magmom_node_inv_feats: Optional[torch.Tensor] = None,
        magmom_node_attrs: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        raise NotImplementedError

nonlinearities = {1: torch.nn.functional.silu, -1: torch.tanh}


@compile_mode("script")
class TensorProductWeightsBlock(torch.nn.Module):
    def __init__(self, num_elements: int, num_edge_feats: int, num_feats_out: int):
        super().__init__()

        weights = torch.empty(
            (num_elements, num_edge_feats, num_feats_out),
            dtype=torch.get_default_dtype(),
        )
        torch.nn.init.xavier_uniform_(weights)
        self.weights = torch.nn.Parameter(weights)

    def forward(
        self,
        sender_or_receiver_node_attrs: torch.Tensor,  # assumes that the node attributes are one-hot encoded
        edge_feats: torch.Tensor,
    ):
        return torch.einsum(
            "be, ba, aek -> bk", edge_feats, sender_or_receiver_node_attrs, self.weights
        )

    def __repr__(self):
        return (
            f'{self.__class__.__name__}(shape=({", ".join(str(s) for s in self.weights.shape)}), '
            f"weights={np.prod(self.weights.shape)})"
        )


@compile_mode("script")
class ResidualElementDependentInteractionBlock(InteractionBlock):
    def _setup(self) -> None:
        if not hasattr(self, "cueq_config"):
            self.cueq_config = None

        # First linear
        self.linear_up = Linear(
            self.node_feats_irreps,
            self.node_feats_irreps,
            internal_weights=True,
            shared_weights=True,
            cueq_config=self.cueq_config,
        )
        # TensorProduct
        irreps_mid, instructions = tp_out_irreps_with_instructions(
            self.node_feats_irreps, self.edge_attrs_irreps, self.target_irreps
        )
        self.conv_tp = TensorProduct(
            self.node_feats_irreps,
            self.edge_attrs_irreps,
            irreps_mid,
            instructions=instructions,
            shared_weights=False,
            internal_weights=False,
            cueq_config=self.cueq_config,
        )
        self.conv_tp_weights = TensorProductWeightsBlock(
            num_elements=self.node_attrs_irreps.num_irreps,
            num_edge_feats=self.edge_feats_irreps.num_irreps,
            num_feats_out=self.conv_tp.weight_numel,
        )

        # Linear
        irreps_mid = irreps_mid.simplify()
        self.irreps_out = linear_out_irreps(irreps_mid, self.target_irreps)
        self.irreps_out = self.irreps_out.simplify()
        self.linear = Linear(
            irreps_mid,
            self.irreps_out,
            internal_weights=True,
            shared_weights=True,
            cueq_config=self.cueq_config,
        )

        # Selector TensorProduct
        self.skip_tp = FullyConnectedTensorProduct(
            self.node_feats_irreps,
            self.node_attrs_irreps,
            self.irreps_out,
            cueq_config=self.cueq_config,
        )

    def forward(
        self,
        node_attrs: torch.Tensor,
        node_feats: torch.Tensor,
        edge_attrs: torch.Tensor,
        edge_feats: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        sender = edge_index[0]
        receiver = edge_index[1]
        num_nodes = node_feats.shape[0]
        sc = self.skip_tp(node_feats, node_attrs)
        node_feats = self.linear_up(node_feats)
        tp_weights = self.conv_tp_weights(node_attrs[sender], edge_feats)
        mji = self.conv_tp(
            node_feats[sender], edge_attrs, tp_weights
        )  # [n_edges, irreps]
        message = scatter_sum(
            src=mji, index=receiver, dim=0, dim_size=num_nodes
        )  # [n_nodes, irreps]
        message = self.linear(message) / self.avg_num_neighbors
        return message + sc  # [n_nodes, irreps]


@compile_mode("script")
class AgnosticNonlinearInteractionBlock(InteractionBlock):
    def _setup(self) -> None:
        if not hasattr(self, "cueq_config"):
            self.cueq_config = None
        self.linear_up = Linear(
            self.node_feats_irreps,
            self.node_feats_irreps,
            internal_weights=True,
            shared_weights=True,
            cueq_config=self.cueq_config,
        )
        # TensorProduct
        irreps_mid, instructions = tp_out_irreps_with_instructions(
            self.node_feats_irreps, self.edge_attrs_irreps, self.target_irreps
        )
        self.conv_tp = TensorProduct(
            self.node_feats_irreps,
            self.edge_attrs_irreps,
            irreps_mid,
            instructions=instructions,
            shared_weights=False,
            internal_weights=False,
            cueq_config=self.cueq_config,
        )

        # Convolution weights
        input_dim = self.edge_feats_irreps.num_irreps
        self.conv_tp_weights = nn.FullyConnectedNet(
            [input_dim] + self.radial_MLP + [self.conv_tp.weight_numel],
            torch.nn.functional.silu,
        )

        # Linear
        irreps_mid = irreps_mid.simplify()
        self.irreps_out = linear_out_irreps(irreps_mid, self.target_irreps)
        self.irreps_out = self.irreps_out.simplify()
        self.linear = Linear(
            irreps_mid,
            self.irreps_out,
            internal_weights=True,
            shared_weights=True,
            cueq_config=self.cueq_config,
        )

        # Selector TensorProduct
        self.skip_tp = FullyConnectedTensorProduct(
            self.irreps_out,
            self.node_attrs_irreps,
            self.irreps_out,
            cueq_config=self.cueq_config,
        )

    def forward(
        self,
        node_attrs: torch.Tensor,
        node_feats: torch.Tensor,
        edge_attrs: torch.Tensor,
        edge_feats: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        sender = edge_index[0]
        receiver = edge_index[1]
        num_nodes = node_feats.shape[0]
        tp_weights = self.conv_tp_weights(edge_feats)
        node_feats = self.linear_up(node_feats)
        mji = self.conv_tp(
            node_feats[sender], edge_attrs, tp_weights
        )  # [n_edges, irreps]
        message = scatter_sum(
            src=mji, index=receiver, dim=0, dim_size=num_nodes
        )  # [n_nodes, irreps]
        message = self.linear(message) / self.avg_num_neighbors
        message = self.skip_tp(message, node_attrs)
        return message  # [n_nodes, irreps]


@compile_mode("script")
class AgnosticResidualNonlinearInteractionBlock(InteractionBlock):
    def _setup(self) -> None:
        if not hasattr(self, "cueq_config"):
            self.cueq_config = None
        # First linear
        self.linear_up = Linear(
            self.node_feats_irreps,
            self.node_feats_irreps,
            internal_weights=True,
            shared_weights=True,
            cueq_config=self.cueq_config,
        )
        # TensorProduct
        irreps_mid, instructions = tp_out_irreps_with_instructions(
            self.node_feats_irreps, self.edge_attrs_irreps, self.target_irreps
        )
        self.conv_tp = TensorProduct(
            self.node_feats_irreps,
            self.edge_attrs_irreps,
            irreps_mid,
            instructions=instructions,
            shared_weights=False,
            internal_weights=False,
            cueq_config=self.cueq_config,
        )

        # Convolution weights
        input_dim = self.edge_feats_irreps.num_irreps
        self.conv_tp_weights = nn.FullyConnectedNet(
            [input_dim] + self.radial_MLP + [self.conv_tp.weight_numel],
            torch.nn.functional.silu,
        )

        # Linear
        irreps_mid = irreps_mid.simplify()
        self.irreps_out = linear_out_irreps(irreps_mid, self.target_irreps)
        self.irreps_out = self.irreps_out.simplify()
        self.linear = Linear(
            irreps_mid,
            self.irreps_out,
            internal_weights=True,
            shared_weights=True,
            cueq_config=self.cueq_config,
        )

        # Selector TensorProduct
        self.skip_tp = FullyConnectedTensorProduct(
            self.node_feats_irreps,
            self.node_attrs_irreps,
            self.irreps_out,
            cueq_config=self.cueq_config,
        )

    def forward(
        self,
        node_attrs: torch.Tensor,
        node_feats: torch.Tensor,
        edge_attrs: torch.Tensor,
        edge_feats: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        sender = edge_index[0]
        receiver = edge_index[1]
        num_nodes = node_feats.shape[0]
        sc = self.skip_tp(node_feats, node_attrs)
        node_feats = self.linear_up(node_feats)
        tp_weights = self.conv_tp_weights(edge_feats)
        mji = self.conv_tp(
            node_feats[sender], edge_attrs, tp_weights
        )  # [n_edges, irreps]
        message = scatter_sum(
            src=mji, index=receiver, dim=0, dim_size=num_nodes
        )  # [n_nodes, irreps]
        message = self.linear(message) / self.avg_num_neighbors
        message = message + sc
        return message  # [n_nodes, irreps]


@compile_mode("script")
class RealAgnosticInteractionBlock(InteractionBlock):
    def _setup(self) -> None:
        if not hasattr(self, "cueq_config"):
            self.cueq_config = None
        # First linear
        self.linear_up = Linear(
            self.node_feats_irreps,
            self.node_feats_irreps,
            internal_weights=True,
            shared_weights=True,
            cueq_config=self.cueq_config,
        )
        # TensorProduct
        irreps_mid, instructions = tp_out_irreps_with_instructions(
            self.node_feats_irreps,
            self.edge_attrs_irreps,
            self.target_irreps,
        )
        self.conv_tp = TensorProduct(
            self.node_feats_irreps,
            self.edge_attrs_irreps,
            irreps_mid,
            instructions=instructions,
            shared_weights=False,
            internal_weights=False,
            cueq_config=self.cueq_config,
        )

        # Convolution weights
        input_dim = self.edge_feats_irreps.num_irreps
        self.conv_tp_weights = nn.FullyConnectedNet(
            [input_dim] + self.radial_MLP + [self.conv_tp.weight_numel],
            torch.nn.functional.silu,
        )

        # Linear
        self.irreps_out = self.target_irreps
        self.linear = Linear(
            irreps_mid,
            self.irreps_out,
            internal_weights=True,
            shared_weights=True,
            cueq_config=self.cueq_config,
        )

        # Selector TensorProduct
        self.skip_tp = FullyConnectedTensorProduct(
            self.irreps_out,
            self.node_attrs_irreps,
            self.irreps_out,
            cueq_config=self.cueq_config,
        )
        self.reshape = reshape_irreps(self.irreps_out, cueq_config=self.cueq_config)

    def forward(
        self,
        node_attrs: torch.Tensor,
        node_feats: torch.Tensor,
        edge_attrs: torch.Tensor,
        edge_feats: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> Tuple[torch.Tensor, None]:
        sender = edge_index[0]
        receiver = edge_index[1]
        num_nodes = node_feats.shape[0]
        node_feats = self.linear_up(node_feats)
        tp_weights = self.conv_tp_weights(edge_feats)
        mji = self.conv_tp(
            node_feats[sender], edge_attrs, tp_weights
        )  # [n_edges, irreps]
        message = scatter_sum(
            src=mji, index=receiver, dim=0, dim_size=num_nodes
        )  # [n_nodes, irreps]
        message = self.linear(message) / self.avg_num_neighbors
        message = self.skip_tp(message, node_attrs)
        return (
            self.reshape(message),
            None,
        )  # [n_nodes, channels, (lmax + 1)**2]


@compile_mode("script")
class RealAgnosticResidualInteractionBlock(InteractionBlock):
    def _setup(self) -> None:
        if not hasattr(self, "cueq_config"):
            self.cueq_config = None
        # First linear
        self.linear_up = Linear(
            self.node_feats_irreps,
            self.node_feats_irreps,
            internal_weights=True,
            shared_weights=True,
            cueq_config=self.cueq_config,
        )
        # TensorProduct
        irreps_mid, instructions = tp_out_irreps_with_instructions(
            self.node_feats_irreps,
            self.edge_attrs_irreps,
            self.target_irreps,
        )
        self.conv_tp = TensorProduct(
            self.node_feats_irreps,
            self.edge_attrs_irreps,
            irreps_mid,
            instructions=instructions,
            shared_weights=False,
            internal_weights=False,
            cueq_config=self.cueq_config,
        )

        # Convolution weights
        input_dim = self.edge_feats_irreps.num_irreps
        self.conv_tp_weights = nn.FullyConnectedNet(
            [input_dim] + self.radial_MLP + [self.conv_tp.weight_numel],
            torch.nn.functional.silu,  # gate
        )

        # Linear
        self.irreps_out = self.target_irreps
        self.linear = Linear(
            irreps_mid,
            self.irreps_out,
            internal_weights=True,
            shared_weights=True,
            cueq_config=self.cueq_config,
        )

        # Selector TensorProduct
        self.skip_tp = FullyConnectedTensorProduct(
            self.node_feats_irreps,
            self.node_attrs_irreps,
            self.hidden_irreps,
            cueq_config=self.cueq_config,
        )
        self.reshape = reshape_irreps(self.irreps_out, cueq_config=self.cueq_config)

    def forward(
        self,
        node_attrs: torch.Tensor,
        node_feats: torch.Tensor,
        edge_attrs: torch.Tensor,
        edge_feats: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        sender = edge_index[0]
        receiver = edge_index[1]
        num_nodes = node_feats.shape[0]
        sc = self.skip_tp(node_feats, node_attrs)
        node_feats = self.linear_up(node_feats)
        tp_weights = self.conv_tp_weights(edge_feats)
        mji = self.conv_tp(
            node_feats[sender], edge_attrs, tp_weights
        )  # [n_edges, irreps]
        message = scatter_sum(
            src=mji, index=receiver, dim=0, dim_size=num_nodes
        )  # [n_nodes, irreps]
        message = self.linear(message) / self.avg_num_neighbors
        return (
            self.reshape(message),
            sc,
        )  # [n_nodes, channels, (lmax + 1)**2]


@compile_mode("script")
class RealAgnosticDensityInteractionBlock(InteractionBlock):
    def _setup(self) -> None:
        if not hasattr(self, "cueq_config"):
            self.cueq_config = None
        # First linear
        self.linear_up = Linear(
            self.node_feats_irreps,
            self.node_feats_irreps,
            internal_weights=True,
            shared_weights=True,
            cueq_config=self.cueq_config,
        )
        # TensorProduct
        irreps_mid, instructions = tp_out_irreps_with_instructions(
            self.node_feats_irreps,
            self.edge_attrs_irreps,
            self.target_irreps,
        )
        self.conv_tp = TensorProduct(
            self.node_feats_irreps,
            self.edge_attrs_irreps,
            irreps_mid,
            instructions=instructions,
            shared_weights=False,
            internal_weights=False,
            cueq_config=self.cueq_config,
        )

        # Convolution weights
        input_dim = self.edge_feats_irreps.num_irreps
        self.conv_tp_weights = nn.FullyConnectedNet(
            [input_dim] + self.radial_MLP + [self.conv_tp.weight_numel],
            torch.nn.functional.silu,
        )

        # Linear
        self.irreps_out = self.target_irreps
        self.linear = Linear(
            irreps_mid,
            self.irreps_out,
            internal_weights=True,
            shared_weights=True,
            cueq_config=self.cueq_config,
        )

        # Selector TensorProduct
        self.skip_tp = FullyConnectedTensorProduct(
            self.irreps_out,
            self.node_attrs_irreps,
            self.irreps_out,
            cueq_config=self.cueq_config,
        )

        # Density normalization
        self.density_fn = nn.FullyConnectedNet(
            [input_dim]
            + [
                1,
            ],
            torch.nn.functional.silu,
        )
        # Reshape
        self.reshape = reshape_irreps(self.irreps_out, cueq_config=self.cueq_config)

    def forward(
        self,
        node_attrs: torch.Tensor,
        node_feats: torch.Tensor,
        edge_attrs: torch.Tensor,
        edge_feats: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> Tuple[torch.Tensor, None]:
        sender = edge_index[0]
        receiver = edge_index[1]
        num_nodes = node_feats.shape[0]
        node_feats = self.linear_up(node_feats)
        tp_weights = self.conv_tp_weights(edge_feats)
        edge_density = torch.tanh(self.density_fn(edge_feats) ** 2)
        mji = self.conv_tp(
            node_feats[sender], edge_attrs, tp_weights
        )  # [n_edges, irreps]
        density = scatter_sum(
            src=edge_density, index=receiver, dim=0, dim_size=num_nodes
        )  # [n_nodes, 1]
        message = scatter_sum(
            src=mji, index=receiver, dim=0, dim_size=num_nodes
        )  # [n_nodes, irreps]
        message = self.linear(message) / (density + 1)
        message = self.skip_tp(message, node_attrs)
        return (
            self.reshape(message),
            None,
        )  # [n_nodes, channels, (lmax + 1)**2]


@compile_mode("script")
class MagneticRealAgnosticDensityInteractionBlock(MagneticInteractionBlock):
    def _setup(self) -> None:
        if not hasattr(self, "cueq_config"):
            self.cueq_config = None
        # First linear
        self.linear_up = Linear(
            self.node_feats_irreps,
            self.node_feats_irreps,
            internal_weights=True,
            shared_weights=True,
            cueq_config=self.cueq_config,
        )
        # TensorProduct for real space
        irreps_mid, instructions = tp_out_irreps_with_instructions(
            self.node_feats_irreps,
            self.edge_attrs_irreps,
            self.target_irreps,
        )
        self.conv_tp = TensorProduct(
            self.node_feats_irreps,
            self.edge_attrs_irreps,
            irreps_mid,
            instructions=instructions,
            shared_weights=False,
            internal_weights=False,
            cueq_config=self.cueq_config,
        )

        # TensorProduct in magnetic moment space
        magmom_irreps_mid, magmom_instructions = tp_out_irreps_with_instructions(
            self.node_feats_irreps,
            self.magmom_node_attrs_irreps,
            self.target_irreps,
        )
        self.magmom_conv_tp = TensorProduct(
            self.node_feats_irreps,
            self.magmom_node_attrs_irreps,
            magmom_irreps_mid,
            instructions=magmom_instructions,
            shared_weights=False,
            internal_weights=False,
            cueq_config=self.cueq_config,
        )
        
        # Convolution weights 
        # fix later
        input_dim = self.edge_feats_irreps.num_irreps
        magmom_input_dim = self.magmom_node_inv_feats_irreps.num_irreps
        self.conv_tp_weights = nn.FullyConnectedNet(
            [input_dim + magmom_input_dim] + self.radial_MLP + [self.conv_tp.weight_numel],
            torch.nn.functional.silu,
        )
        # transforming from radial l channels to magnetic l channels
        self.conv_tp_weights_magmom_trans = nn.FullyConnectedNet(
            [self.conv_tp.weight_numel, ] + [self.magmom_conv_tp.weight_numel, ]
        )

        # Linear
        self.irreps_out = self.target_irreps
        self.linear = Linear(
            irreps_mid,
            self.irreps_out,
            internal_weights=True,
            shared_weights=True,
            cueq_config=self.cueq_config,
        )

        self.magmom_linear = Linear(
            magmom_irreps_mid,
            self.irreps_out,
            internal_weights=True,
            shared_weights=True,
            cueq_config=self.cueq_config,
        )

        # Selector TensorProduct
        self.skip_tp = FullyConnectedTensorProduct(
            self.irreps_out,
            self.node_attrs_irreps,
            self.irreps_out,
            cueq_config=self.cueq_config,
        )

        self.magmom_skip_tp = FullyConnectedTensorProduct(
            self.irreps_out,
            self.node_attrs_irreps,
            self.irreps_out,
            cueq_config=self.cueq_config,
        )

        # Density normalization
        self.density_fn = nn.FullyConnectedNet(
            [input_dim]
            + [
                1,
            ],
            torch.nn.functional.silu,
        )
        # Reshape
        self.reshape = reshape_irreps(self.irreps_out, cueq_config=self.cueq_config)

    def forward(
        self,
        node_attrs: torch.Tensor,
        node_feats: torch.Tensor,
        edge_attrs: torch.Tensor,
        edge_feats: torch.Tensor, # (n_edges, n_basis)
        edge_index: torch.Tensor,
        magmom_node_inv_feats: torch.Tensor,
        magmom_node_attrs: torch.Tensor
    ) -> Tuple[torch.Tensor, None]:
        sender = edge_index[0]
        receiver = edge_index[1]
        #print("edge index shape: ", edge_index.shape)
        #print("magmom_node_attrs: ", magmom_node_attrs)
        num_nodes = node_feats.shape[0]
        node_feats = self.linear_up(node_feats)

        # boardcast node feats to number of nodes
        magmom_inv_feats_j = magmom_node_inv_feats[sender]
        
        edge_feats_with_magmom = torch.cat([edge_feats, magmom_inv_feats_j], dim=-1)

        # combined learnable radial
        tp_weights = self.conv_tp_weights(edge_feats_with_magmom)

        # density normalization
        edge_density = torch.tanh(self.density_fn(edge_feats) ** 2)

        mji = self.conv_tp(
            node_feats[sender], edge_attrs, tp_weights
        )  # [n_edges, irreps]

        if hasattr(self, "conv_tp_weights_magmom_trans"):
            tp_weights_magmom = self.conv_tp_weights_magmom_trans(tp_weights)
        else:
            tp_weights_magmom = tp_weights
        magmom_mji = self.magmom_conv_tp(
            node_feats[sender], magmom_node_attrs[sender], tp_weights_magmom
        )  # [n_edges, irreps]
        
        density = scatter_sum(
            src=edge_density, index=receiver, dim=0, dim_size=num_nodes
        )  # [n_nodes, 1]

        # highlighted message for central message

        message = scatter_sum(
            src=mji, index=receiver, dim=0, dim_size=num_nodes
        )  # [n_nodes, irreps]
        
        magmom_message = scatter_sum(
            src=magmom_mji, index=receiver, dim = 0, dim_size=num_nodes,
        )
        
        message = self.linear(message) / (density + 1)
        message = self.skip_tp(message, node_attrs)
        # not doing density normalization for now
        magmom_message = self.magmom_linear(magmom_message) / self.avg_num_neighbors
        magmom_message = self.magmom_skip_tp(magmom_message, node_attrs)

        return (
            self.reshape(message),
            self.reshape(magmom_message),
            None,
            None,
        )  # [n_nodes, channels, (lmax + 1)**2]

@compile_mode("script")
class MagneticRealAgnosticSpinOrbitCoupledDensityInteractionBlock(MagneticInteractionBlock):
    def _setup(self) -> None:
        if not hasattr(self, "cueq_config"):
            self.cueq_config = None

        print("into MagneticRealAgnosticSpinOrbitCoupledDensityInteractionBlock")
        # First linear
        self.linear_up = Linear(
            self.node_feats_irreps,
            self.node_feats_irreps,
            internal_weights=True,
            shared_weights=True,
            cueq_config=self.cueq_config,
        )
        print("===done init linear===")
        # TensorProduct for real space
        irreps_mid, instructions = tp_out_irreps_with_instructions(
            self.node_feats_irreps,
            self.edge_attrs_irreps,
            self.target_irreps,
        )
        self.conv_tp = TensorProduct(
            self.node_feats_irreps,
            self.edge_attrs_irreps,
            irreps_mid,
            instructions=instructions,
            shared_weights=False,
            internal_weights=False,
            cueq_config=self.cueq_config,
        )
        print("===done init conv_tp===")
        # TensorProduct in magnetic moment space
        magmom_irreps_mid, magmom_instructions = tp_out_irreps_with_instructions(
            #self.conv_tp.irreps_out,
            irreps_mid,
            self.magmom_node_attrs_irreps,
            self.target_irreps,
        )
        self.magmom_conv_tp = TensorProduct(
            self.conv_tp.irreps_out,
            self.magmom_node_attrs_irreps,
            magmom_irreps_mid,
            instructions=magmom_instructions,
            shared_weights=False,
            internal_weights=False,
            cueq_config=self.cueq_config,
        )
        print("===done init magmom conv_tp===")
        # Convolution weights 
        input_dim = self.edge_feats_irreps.num_irreps
        magmom_input_dim = self.magmom_node_inv_feats_irreps.num_irreps
        self.conv_tp_weights = nn.FullyConnectedNet(
            [input_dim + magmom_input_dim] + self.radial_MLP + [self.conv_tp.weight_numel],
            torch.nn.functional.silu,
        )
        # transforming from radial l channels to magnetic l channels
        self.conv_tp_weights_magmom = nn.FullyConnectedNet(
            [input_dim + magmom_input_dim, ] + [self.magmom_conv_tp.weight_numel, ]
        )

        # Linear
        self.irreps_out = self.target_irreps

        self.magmom_linear = Linear(
            self.magmom_conv_tp.irreps_out,
            self.irreps_out,
            internal_weights=True,
            shared_weights=True,
            cueq_config=self.cueq_config,
        )
        self.magmom_skip_tp = FullyConnectedTensorProduct(
            self.irreps_out,
            self.node_attrs_irreps,
            self.irreps_out,
            cueq_config=self.cueq_config,
        )

        # Density normalization
        self.density_fn = nn.FullyConnectedNet(
            [input_dim]
            + [
                1,
            ],
            torch.nn.functional.silu,
        )
        # Reshape
        self.reshape = reshape_irreps(self.irreps_out, cueq_config=self.cueq_config)

    def forward(
        self,
        node_attrs: torch.Tensor,
        node_feats: torch.Tensor,
        edge_attrs: torch.Tensor,
        edge_feats: torch.Tensor, # (n_edges, n_basis)
        edge_index: torch.Tensor,
        magmom_node_inv_feats: torch.Tensor,
        magmom_node_attrs: torch.Tensor
    ) -> Tuple[torch.Tensor, None]:
        sender = edge_index[0]
        receiver = edge_index[1]
        
        num_nodes = node_feats.shape[0]
        node_feats = self.linear_up(node_feats)
        
        # boardcast node feats to number of nodes
        magmom_inv_feats_j = magmom_node_inv_feats[sender]
        edge_feats_with_magmom = torch.cat([edge_feats, magmom_inv_feats_j], dim=-1)        
        
        # combined learnable radial
        tp_weights = self.conv_tp_weights(edge_feats_with_magmom)

        # density normalization
        edge_density = torch.tanh(self.density_fn(edge_feats) ** 2)

        mji = self.conv_tp(
            node_feats[sender], edge_attrs, tp_weights
        )  # [n_edges, irreps]
        
        tp_weights_magmom = self.conv_tp_weights_magmom(edge_feats_with_magmom)
        
        magmom_mji = self.magmom_conv_tp(
            mji, magmom_node_attrs[sender], tp_weights_magmom
        )  # [n_edges, irreps]
        
        density = scatter_sum(
            src=edge_density, index=receiver, dim=0, dim_size=num_nodes
        )  # [n_nodes, 1]
        
        magmom_message = scatter_sum(
            src=magmom_mji, index=receiver, dim = 0, dim_size=num_nodes,
        )

        # not doing density normalization for now
        magmom_message = self.magmom_linear(magmom_message) / (density + 1)
        magmom_message = self.magmom_skip_tp(magmom_message, node_attrs)
        return (
            self.reshape(magmom_message),
            None,
        )  # [n_nodes, channels, (lmax + 1)**2]


@compile_mode("script")
class MagneticRealAgnosticFlexibleSpinOrbitCoupledDensityInteractionBlock(MagneticInteractionBlock):
    def _setup(self) -> None:
        if not hasattr(self, "cueq_config"):
            self.cueq_config = None
        # First linear
        self.linear_up = Linear(
            self.node_feats_irreps,
            self.node_feats_irreps,
            internal_weights=True,
            shared_weights=True,
            cueq_config=self.cueq_config,
        )
        # TensorProduct for real space
        irreps_mid, instructions = tp_out_irreps_with_instructions(
            self.node_feats_irreps,
            self.edge_attrs_irreps,
            self.target_irreps,
        )
        self.conv_tp = TensorProduct(
            self.node_feats_irreps,
            self.edge_attrs_irreps,
            irreps_mid,
            instructions=instructions,
            shared_weights=False,
            internal_weights=False,
            cueq_config=self.cueq_config,
        )
        self.reshape_mji = reshape_irreps(self.conv_tp.irreps_out)
        self.inv_reshape_mji = inverse_reshape_irreps(self.conv_tp.irreps_out)

        # TensorProduct in magnetic moment space
        magmom_irreps_mid, magmom_instructions = tp_out_irreps_with_instructions(
            self.node_feats_irreps,
            self.magmom_node_attrs_irreps,
            self.target_irreps,
        )
        self.magmom_conv_tp = TensorProduct(
            self.node_feats_irreps,
            self.magmom_node_attrs_irreps,
            magmom_irreps_mid,
            instructions=magmom_instructions,
            shared_weights=False,
            internal_weights=False,
            cueq_config=self.cueq_config,
        )
        self.reshape_magmom_mji = reshape_irreps(self.magmom_conv_tp.irreps_out)
        self.inv_reshape_magmom_mji = inverse_reshape_irreps(self.magmom_conv_tp.irreps_out)

        # Convolution weights 
        # fix later
        input_dim = self.edge_feats_irreps.num_irreps
        magmom_input_dim = self.magmom_node_inv_feats_irreps.num_irreps
        self.conv_tp_weights = nn.FullyConnectedNet(
            [magmom_input_dim + input_dim] + self.radial_MLP + [self.conv_tp.weight_numel],
            torch.nn.functional.silu,
        )
        # transforming from radial l channels to magnetic l channels
        self.conv_tp_weights_magmom = nn.FullyConnectedNet(
            [magmom_input_dim + input_dim, ] + self.radial_MLP + [self.magmom_conv_tp.weight_numel, ]
        )

        # Linear
        self.irreps_out = self.target_irreps
        self.linear = Linear(
            irreps_mid,
            self.irreps_out,
            internal_weights=True,
            shared_weights=True,
            cueq_config=self.cueq_config,
        )

        self.magmom_linear = Linear(
            magmom_irreps_mid,
            self.irreps_out,
            internal_weights=True,
            shared_weights=True,
            cueq_config=self.cueq_config,
        )

        # Selector TensorProduct
        self.skip_tp = FullyConnectedTensorProduct(
            self.irreps_out,
            self.node_attrs_irreps,
            self.irreps_out,
            cueq_config=self.cueq_config,
        )

        self.magmom_skip_tp = FullyConnectedTensorProduct(
            self.irreps_out,
            self.node_attrs_irreps,
            self.irreps_out,
            cueq_config=self.cueq_config,
        )

        # Density normalization
        self.density_fn = nn.FullyConnectedNet(
            [input_dim]
            + [
                1,
            ],
            torch.nn.functional.silu,
        )
        # Reshape
        self.reshape = reshape_irreps(self.irreps_out, cueq_config=self.cueq_config)

    def forward(
        self,
        node_attrs: torch.Tensor,
        node_feats: torch.Tensor,
        edge_attrs: torch.Tensor,
        edge_feats: torch.Tensor, # (n_edges, n_basis)
        edge_index: torch.Tensor,
        magmom_node_inv_feats: torch.Tensor,
        magmom_node_attrs: torch.Tensor,
        couple_SO: Optional[bool] = False,
    ) -> Tuple[torch.Tensor, None]:
        sender = edge_index[0]
        receiver = edge_index[1]
        num_edges = len(sender)
        num_k = self.conv_tp.irreps_out[0].mul
        lmax_magmom = self.magmom_node_attrs_irreps.lmax
        #print("num_k:", num_k)

        num_nodes = node_feats.shape[0]
        node_feats = self.linear_up(node_feats)
        

        # boardcast node feats to number of nodes
        magmom_inv_feats_j = magmom_node_inv_feats[sender]
        

        edge_feats_with_magmom = torch.cat([edge_feats, magmom_inv_feats_j], dim=-1)        

        # combined learnable radial
        tp_weights = self.conv_tp_weights(edge_feats_with_magmom)

        # density normalization
        edge_density = torch.tanh(self.density_fn(edge_feats) ** 2)

        pre_mji = self.conv_tp(
            node_feats[sender], edge_attrs, tp_weights
        )  # [n_edges, irreps]

        
        tp_weights_magmom = self.conv_tp_weights_magmom(edge_feats_with_magmom)
        tp_weights_magmom = tp_weights_magmom.reshape(num_edges, num_k, lmax_magmom+1)
        tp_weights_magmom = tp_weights_magmom * pre_mji[:, :num_k].unsqueeze(-1)
        tp_weights_magmom = tp_weights_magmom.reshape(num_edges, -1)

        magmom_mji = self.magmom_conv_tp(
            node_feats[sender], magmom_node_attrs[sender], tp_weights_magmom
        )  # [n_edges, irreps]
        
        # 
        tp_weights = tp_weights.reshape(num_edges, num_k, -1)
        tp_weights = tp_weights * magmom_mji[:, :num_k].unsqueeze(-1)
        tp_weights = tp_weights.reshape(num_edges, -1)

        mji = self.conv_tp(
            node_feats[sender], edge_attrs, tp_weights
        )  # [n_edges, irreps]

        density = scatter_sum(
            src=edge_density, index=receiver, dim=0, dim_size=num_nodes
        )  # [n_nodes, 1]

        # reshape to [n_edges, k, lm]
        # magmom_mji = self.reshape_magmom_mji(magmom_mji)
        # mji = self.reshape_mji(mji)

        # noSO_magmom_mji = magmom_mji * mji[:, :, 0].unsqueeze(-1)
        # noSO_mji = magmom_mji[:, :, 0].unsqueeze(-1) * mji

        # noSO_magmom_mji = self.inv_reshape_magmom_mji(noSO_magmom_mji)
        # noSO_mji = self.inv_reshape_mji(noSO_mji)

        noSO_message = scatter_sum(
            src=mji, index=receiver, dim=0, dim_size=num_nodes
        )  # [n_nodes, irreps]
        
        noSO_magmom_message = scatter_sum(
            src=magmom_mji, index=receiver, dim = 0, dim_size=num_nodes,
        )

        if couple_SO:
            raise ValueError("Not implemented")
        else:
            SO_message = None

        noSO_message = self.linear(noSO_message) / (density + 1)
        noSO_message = self.skip_tp(noSO_message, node_attrs)
        # not doing density normalization for now
        noSO_magmom_message = self.magmom_linear(noSO_magmom_message) / self.avg_num_neighbors
        noSO_magmom_message = self.magmom_skip_tp(noSO_magmom_message, node_attrs)

        return (
            self.reshape(noSO_message),
            None,
            self.reshape(noSO_magmom_message),
            None,
            SO_message,
            None
        )  # [n_nodes, channels, (lmax + 1)**2]


@compile_mode("script")
class MagneticRealAgnosticSeparateRadialDensityInteractionBlock(MagneticInteractionBlock):
    def _setup(self) -> None:
        if not hasattr(self, "cueq_config"):
            self.cueq_config = None
        # First linear
        self.linear_up = Linear(
            self.node_feats_irreps,
            self.node_feats_irreps,
            internal_weights=True,
            shared_weights=True,
            cueq_config=self.cueq_config,
        )
        # TensorProduct for real space
        irreps_mid, instructions = tp_out_irreps_with_instructions(
            self.node_feats_irreps,
            self.edge_attrs_irreps,
            self.target_irreps,
        )
        self.conv_tp = TensorProduct(
            self.node_feats_irreps,
            self.edge_attrs_irreps,
            irreps_mid,
            instructions=instructions,
            shared_weights=False,
            internal_weights=False,
            cueq_config=self.cueq_config,
        )

        # TensorProduct in magnetic moment space
        magmom_irreps_mid, magmom_instructions = tp_out_irreps_with_instructions(
            self.node_feats_irreps,
            self.magmom_node_attrs_irreps,
            self.target_irreps,
        )
        self.magmom_conv_tp = TensorProduct(
            self.node_feats_irreps,
            self.magmom_node_attrs_irreps,
            magmom_irreps_mid,
            instructions=magmom_instructions,
            shared_weights=False,
            internal_weights=False,
            cueq_config=self.cueq_config,
        )
        
        # Convolution weights 
        # fix later
        input_dim = self.edge_feats_irreps.num_irreps
        magmom_input_dim = self.magmom_node_inv_feats_irreps.num_irreps
        self.conv_tp_weights = nn.FullyConnectedNet(
            [input_dim] + self.radial_MLP + [self.conv_tp.weight_numel],
            torch.nn.functional.silu,
        )
        # transforming from radial l channels to magnetic l channels
        self.conv_tp_weights_magmom = nn.FullyConnectedNet(
            [magmom_input_dim, ] + [self.magmom_conv_tp.weight_numel, ]
        )

        # Linear
        self.irreps_out = self.target_irreps
        self.linear = Linear(
            irreps_mid,
            self.irreps_out,
            internal_weights=True,
            shared_weights=True,
            cueq_config=self.cueq_config,
        )

        self.magmom_linear = Linear(
            magmom_irreps_mid,
            self.irreps_out,
            internal_weights=True,
            shared_weights=True,
            cueq_config=self.cueq_config,
        )

        # Selector TensorProduct
        self.skip_tp = FullyConnectedTensorProduct(
            self.irreps_out,
            self.node_attrs_irreps,
            self.irreps_out,
            cueq_config=self.cueq_config,
        )

        self.magmom_skip_tp = FullyConnectedTensorProduct(
            self.irreps_out,
            self.node_attrs_irreps,
            self.irreps_out,
            cueq_config=self.cueq_config,
        )

        # Density normalization
        self.density_fn = nn.FullyConnectedNet(
            [input_dim]
            + [
                1,
            ],
            torch.nn.functional.silu,
        )
        # Reshape
        self.reshape = reshape_irreps(self.irreps_out, cueq_config=self.cueq_config)

    def forward(
        self,
        node_attrs: torch.Tensor,
        node_feats: torch.Tensor,
        edge_attrs: torch.Tensor,
        edge_feats: torch.Tensor, # (n_edges, n_basis)
        edge_index: torch.Tensor,
        magmom_node_inv_feats: torch.Tensor,
        magmom_node_attrs: torch.Tensor
    ) -> Tuple[torch.Tensor, None]:
        sender = edge_index[0]
        receiver = edge_index[1]
        #print("edge index shape: ", edge_index.shape)
        #print("magmom_node_attrs: ", magmom_node_attrs)
        num_nodes = node_feats.shape[0]
        node_feats = self.linear_up(node_feats)

        # boardcast node feats to number of nodes
        magmom_inv_feats_j = magmom_node_inv_feats[sender]
        
        # combined learnable radial
        tp_weights = self.conv_tp_weights(edge_feats)

        # density normalization
        edge_density = torch.tanh(self.density_fn(edge_feats) ** 2)

        mji = self.conv_tp(
            node_feats[sender], edge_attrs, tp_weights
        )  # [n_edges, irreps]

        
        tp_weights_magmom = self.conv_tp_weights_magmom(magmom_inv_feats_j)
        
        magmom_mji = self.magmom_conv_tp(
            node_feats[sender], magmom_node_attrs[sender], tp_weights_magmom
        )  # [n_edges, irreps]
        
        density = scatter_sum(
            src=edge_density, index=receiver, dim=0, dim_size=num_nodes
        )  # [n_nodes, 1]

        # highlighted message for central message

        message = scatter_sum(
            src=mji, index=receiver, dim=0, dim_size=num_nodes
        )  # [n_nodes, irreps]
        
        magmom_message = scatter_sum(
            src=magmom_mji, index=receiver, dim = 0, dim_size=num_nodes,
        )
        
        message = self.linear(message) / (density + 1)
        message = self.skip_tp(message, node_attrs)
        # not doing density normalization for now
        magmom_message = self.magmom_linear(magmom_message) / self.avg_num_neighbors
        magmom_message = self.magmom_skip_tp(magmom_message, node_attrs)

        return (
            self.reshape(message),
            self.reshape(magmom_message),
            None,
            None,
        )  # [n_nodes, channels, (lmax + 1)**2]

@compile_mode("script")
class MagneticRealAgnosticSeparateRadialDensityTestingInteractionBlock(MagneticInteractionBlock):
    def _setup(self) -> None:
        if not hasattr(self, "cueq_config"):
            self.cueq_config = None
        # First linear
        self.linear_up = Linear(
            self.node_feats_irreps,
            self.node_feats_irreps,
            internal_weights=True,
            shared_weights=True,
            cueq_config=self.cueq_config,
        )
        # TensorProduct for real space
        irreps_mid, instructions = tp_out_irreps_with_instructions(
            self.node_feats_irreps,
            self.edge_attrs_irreps,
            self.target_irreps,
        )
        self.conv_tp = TensorProduct(
            self.node_feats_irreps,
            self.edge_attrs_irreps,
            irreps_mid,
            instructions=instructions,
            shared_weights=False,
            internal_weights=False,
            cueq_config=self.cueq_config,
        )

        # TensorProduct in magnetic moment space
        magmom_irreps_mid, magmom_instructions = tp_out_irreps_with_instructions(
            self.node_feats_irreps,
            self.magmom_node_attrs_irreps,
            self.target_irreps,
        )
        self.magmom_conv_tp = TensorProduct(
            self.node_feats_irreps,
            self.magmom_node_attrs_irreps,
            magmom_irreps_mid,
            instructions=magmom_instructions,
            shared_weights=False,
            internal_weights=False,
            cueq_config=self.cueq_config,
        )
        
        # Convolution weights 
        # fix later
        input_dim = self.edge_feats_irreps.num_irreps
        magmom_input_dim = self.magmom_node_inv_feats_irreps.num_irreps

        self.conv_tp_weights = nn.FullyConnectedNet(
            [input_dim + magmom_input_dim] + self.radial_MLP + [self.conv_tp.weight_numel],
            torch.nn.functional.silu,
        )
        # transforming from radial l channels to magnetic l channels
        self.conv_tp_weights_magmom = nn.FullyConnectedNet(
            [input_dim + magmom_input_dim, ] + self.radial_MLP + [self.magmom_conv_tp.weight_numel, ],
            torch.nn.functional.silu,
        )

        # Linear
        self.irreps_out = self.target_irreps
        self.linear = Linear(
            irreps_mid,
            self.irreps_out,
            internal_weights=True,
            shared_weights=True,
            cueq_config=self.cueq_config,
        )

        self.magmom_linear = Linear(
            magmom_irreps_mid,
            self.irreps_out,
            internal_weights=True,
            shared_weights=True,
            cueq_config=self.cueq_config,
        )

        # Selector TensorProduct
        self.skip_tp = FullyConnectedTensorProduct(
            self.irreps_out,
            self.node_attrs_irreps,
            self.irreps_out,
            cueq_config=self.cueq_config,
        )

        self.magmom_skip_tp = FullyConnectedTensorProduct(
            self.irreps_out,
            self.node_attrs_irreps,
            self.irreps_out,
            cueq_config=self.cueq_config,
        )

        # Density normalization
        self.density_fn = nn.FullyConnectedNet(
            [input_dim]
            + [
                1,
            ],
            torch.nn.functional.silu,
        )
        # Reshape
        self.reshape = reshape_irreps(self.irreps_out, cueq_config=self.cueq_config)

    def forward(
        self,
        node_attrs: torch.Tensor,
        node_feats: torch.Tensor,
        edge_attrs: torch.Tensor,
        edge_feats: torch.Tensor, # (n_edges, n_basis)
        edge_index: torch.Tensor,
        magmom_node_inv_feats: torch.Tensor,
        magmom_node_attrs: torch.Tensor
    ) -> Tuple[torch.Tensor, None]:
        sender = edge_index[0]
        receiver = edge_index[1]
        #print("edge index shape: ", edge_index.shape)
        #print("magmom_node_attrs: ", magmom_node_attrs)
        num_nodes = node_feats.shape[0]
        node_feats = self.linear_up(node_feats)

        # boardcast node feats to number of nodes
        magmom_inv_feats_j = magmom_node_inv_feats[sender]
        edge_feats_with_magmom = torch.cat([edge_feats, magmom_inv_feats_j], dim=-1)

        # learnable radial
        tp_weights = self.conv_tp_weights(edge_feats_with_magmom)
        tp_weights_magmom = self.conv_tp_weights_magmom(edge_feats_with_magmom)

        # and then form CP decomposition here

        # density normalization
        edge_density = torch.tanh(self.density_fn(edge_feats) ** 2)

        mji = self.conv_tp(
            node_feats[sender], edge_attrs, tp_weights
        )  # [n_edges, irreps]

                
        magmom_mji = self.magmom_conv_tp(
            node_feats[sender], magmom_node_attrs[sender], tp_weights_magmom
        )  # [n_edges, irreps]
        
        density = scatter_sum(
            src=edge_density, index=receiver, dim=0, dim_size=num_nodes
        )  # [n_nodes, 1]

        # highlighted message for central message

        message = scatter_sum(
            src=mji, index=receiver, dim=0, dim_size=num_nodes
        )  # [n_nodes, irreps]
        
        magmom_message = scatter_sum(
            src=magmom_mji, index=receiver, dim = 0, dim_size=num_nodes,
        )
        
        message = self.linear(message) / (density + 1)
        message = self.skip_tp(message, node_attrs)
        # not doing density normalization for now
        magmom_message = self.magmom_linear(magmom_message) / self.avg_num_neighbors
        magmom_message = self.magmom_skip_tp(magmom_message, node_attrs)

        return (
            self.reshape(message),
            self.reshape(magmom_message),
            None,
            None,
        )  # [n_nodes, channels, (lmax + 1)**2]

@compile_mode("script")
class MagneticRealAgnosticSeparateRadialCoupledDensityInteractionBlock(MagneticInteractionBlock):
    def _setup(self) -> None:
        if not hasattr(self, "cueq_config"):
            self.cueq_config = None
        # First linear
        self.linear_up = Linear(
            self.node_feats_irreps,
            self.node_feats_irreps,
            internal_weights=True,
            shared_weights=True,
            cueq_config=self.cueq_config,
        )
        # TensorProduct for real space
        irreps_mid, instructions = tp_out_irreps_with_instructions(
            self.node_feats_irreps,
            self.edge_attrs_irreps,
            self.target_irreps,
        )
        self.conv_tp = TensorProduct(
            self.node_feats_irreps,
            self.edge_attrs_irreps,
            irreps_mid,
            instructions=instructions,
            shared_weights=False,
            internal_weights=False,
            cueq_config=self.cueq_config,
        )

        # TensorProduct in magnetic moment space
        magmom_irreps_mid, magmom_instructions = tp_out_irreps_with_instructions(
            self.node_feats_irreps,
            self.magmom_node_attrs_irreps,
            self.target_irreps,
        )
        self.magmom_conv_tp = TensorProduct(
            self.node_feats_irreps,
            self.magmom_node_attrs_irreps,
            magmom_irreps_mid,
            instructions=magmom_instructions,
            shared_weights=False,
            internal_weights=False,
            cueq_config=self.cueq_config,
        )
        
        # Convolution weights 
        # fix later
        input_dim = self.edge_feats_irreps.num_irreps
        magmom_input_dim = self.magmom_node_inv_feats_irreps.num_irreps
        self.conv_tp_weights = nn.FullyConnectedNet(
            [input_dim + magmom_input_dim] + self.radial_MLP + [self.conv_tp.weight_numel],
            torch.nn.functional.silu,
        )
        # transforming from radial l channels to magnetic l channels
        self.conv_tp_weights_magmom = nn.FullyConnectedNet(
            [magmom_input_dim, ] + [self.magmom_conv_tp.weight_numel, ]
        )

        # Linear
        self.irreps_out = self.target_irreps
        self.linear = Linear(
            irreps_mid,
            self.irreps_out,
            internal_weights=True,
            shared_weights=True,
            cueq_config=self.cueq_config,
        )

        self.magmom_linear = Linear(
            magmom_irreps_mid,
            self.irreps_out,
            internal_weights=True,
            shared_weights=True,
            cueq_config=self.cueq_config,
        )

        # Selector TensorProduct
        self.skip_tp = FullyConnectedTensorProduct(
            self.irreps_out,
            self.node_attrs_irreps,
            self.irreps_out,
            cueq_config=self.cueq_config,
        )

        self.magmom_skip_tp = FullyConnectedTensorProduct(
            self.irreps_out,
            self.node_attrs_irreps,
            self.irreps_out,
            cueq_config=self.cueq_config,
        )

        # Density normalization
        self.density_fn = nn.FullyConnectedNet(
            [input_dim]
            + [
                1,
            ],
            torch.nn.functional.silu,
        )
        # Reshape
        self.reshape = reshape_irreps(self.irreps_out, cueq_config=self.cueq_config)

    def forward(
        self,
        node_attrs: torch.Tensor,
        node_feats: torch.Tensor,
        edge_attrs: torch.Tensor,
        edge_feats: torch.Tensor, # (n_edges, n_basis)
        edge_index: torch.Tensor,
        magmom_node_inv_feats: torch.Tensor,
        magmom_node_attrs: torch.Tensor
    ) -> Tuple[torch.Tensor, None]:
        # max l for position space : self.edge_attrs_irreps.lmax
        # max l for magmom space: self.magmom_node_attrs_irreps.lmax

        sender = edge_index[0]
        receiver = edge_index[1]
        #print("edge index shape: ", edge_index.shape)
        #print("magmom_node_attrs: ", magmom_node_attrs)
        num_nodes = node_feats.shape[0]
        node_feats = self.linear_up(node_feats)

        # boardcast node feats to number of nodes
        magmom_inv_feats_j = magmom_node_inv_feats[sender]

        #        
        edge_feats_with_magmom = torch.cat([edge_feats, magmom_inv_feats_j], dim=-1)

        # learnable radial for real space
        tp_weights = self.conv_tp_weights(edge_feats_with_magmom)
        
        # density normalization
        edge_density = torch.tanh(self.density_fn(edge_feats) ** 2)

        mji = self.conv_tp(
            node_feats[sender], edge_attrs, tp_weights
        )  # [n_edges, irreps]

        # learnable radial for magmom space
        tp_weights_magmom = self.conv_tp_weights_magmom(magmom_inv_feats_j)
        
        magmom_mji = self.magmom_conv_tp(
            node_feats[sender], magmom_node_attrs[sender], tp_weights_magmom
        )  # [n_edges, irreps]
        
        density = scatter_sum(
            src=edge_density, index=receiver, dim=0, dim_size=num_nodes
        )  # [n_nodes, 1]

        # highlighted message for central message

        message = scatter_sum(
            src=mji, index=receiver, dim=0, dim_size=num_nodes
        )  # [n_nodes, irreps]
        
        magmom_message = scatter_sum(
            src=magmom_mji, index=receiver, dim = 0, dim_size=num_nodes,
        )
        
        message = self.linear(message) / (density + 1)
        message = self.skip_tp(message, node_attrs)
        # not doing density normalization for now
        magmom_message = self.magmom_linear(magmom_message) / self.avg_num_neighbors
        magmom_message = self.magmom_skip_tp(magmom_message, node_attrs)

        return (
            self.reshape(message),
            self.reshape(magmom_message),
            None,
            None,
        )  # [n_nodes, channels, (lmax + 1)**2]

@compile_mode("script")
class MagneticRealAgnosticSeparateRadialCoupledPosToMagDensityInteractionBlock(MagneticInteractionBlock):
    def _setup(self) -> None:
        if not hasattr(self, "cueq_config"):
            self.cueq_config = None
        # First linear
        self.linear_up = Linear(
            self.node_feats_irreps,
            self.node_feats_irreps,
            internal_weights=True,
            shared_weights=True,
            cueq_config=self.cueq_config,
        )
        # TensorProduct for real space
        irreps_mid, instructions = tp_out_irreps_with_instructions(
            self.node_feats_irreps,
            self.edge_attrs_irreps,
            self.target_irreps,
        )
        self.conv_tp = TensorProduct(
            self.node_feats_irreps,
            self.edge_attrs_irreps,
            irreps_mid,
            instructions=instructions,
            shared_weights=False,
            internal_weights=False,
            cueq_config=self.cueq_config,
        )

        # TensorProduct in magnetic moment space
        magmom_irreps_mid, magmom_instructions = tp_out_irreps_with_instructions(
            self.node_feats_irreps,
            self.magmom_node_attrs_irreps,
            self.target_irreps,
        )
        self.magmom_conv_tp = TensorProduct(
            self.node_feats_irreps,
            self.magmom_node_attrs_irreps,
            magmom_irreps_mid,
            instructions=magmom_instructions,
            shared_weights=False,
            internal_weights=False,
            cueq_config=self.cueq_config,
        )
        
        # Convolution weights 
        # fix later
        input_dim = self.edge_feats_irreps.num_irreps
        magmom_input_dim = self.magmom_node_inv_feats_irreps.num_irreps
        self.conv_tp_weights = nn.FullyConnectedNet(
            [input_dim] + self.radial_MLP + [self.conv_tp.weight_numel],
            torch.nn.functional.silu,
        )
        # transforming from radial l channels to magnetic l channels
        self.conv_tp_weights_magmom = nn.FullyConnectedNet(
            [input_dim + magmom_input_dim, ] + [self.magmom_conv_tp.weight_numel, ]
        )

        # Linear
        self.irreps_out = self.target_irreps
        self.linear = Linear(
            irreps_mid,
            self.irreps_out,
            internal_weights=True,
            shared_weights=True,
            cueq_config=self.cueq_config,
        )

        self.magmom_linear = Linear(
            magmom_irreps_mid,
            self.irreps_out,
            internal_weights=True,
            shared_weights=True,
            cueq_config=self.cueq_config,
        )

        # Selector TensorProduct
        self.skip_tp = FullyConnectedTensorProduct(
            self.irreps_out,
            self.node_attrs_irreps,
            self.irreps_out,
            cueq_config=self.cueq_config,
        )

        self.magmom_skip_tp = FullyConnectedTensorProduct(
            self.irreps_out,
            self.node_attrs_irreps,
            self.irreps_out,
            cueq_config=self.cueq_config,
        )

        # Density normalization
        self.density_fn = nn.FullyConnectedNet(
            [input_dim]
            + [
                1,
            ],
            torch.nn.functional.silu,
        )
        # Reshape
        self.reshape = reshape_irreps(self.irreps_out, cueq_config=self.cueq_config)

    def forward(
        self,
        node_attrs: torch.Tensor,
        node_feats: torch.Tensor,
        edge_attrs: torch.Tensor,
        edge_feats: torch.Tensor, # (n_edges, n_basis)
        edge_index: torch.Tensor,
        magmom_node_inv_feats: torch.Tensor,
        magmom_node_attrs: torch.Tensor
    ) -> Tuple[torch.Tensor, None]:
        # max l for position space : self.edge_attrs_irreps.lmax
        # max l for magmom space: self.magmom_node_attrs_irreps.lmax

        sender = edge_index[0]
        receiver = edge_index[1]
        #print("edge index shape: ", edge_index.shape)
        #print("magmom_node_attrs: ", magmom_node_attrs)
        num_nodes = node_feats.shape[0]
        node_feats = self.linear_up(node_feats)

        # boardcast node feats to number of nodes
        magmom_inv_feats_j = magmom_node_inv_feats[sender]

        #        
        edge_feats_with_magmom = torch.cat([edge_feats, magmom_inv_feats_j], dim=-1)

        # learnable radial for real space
        tp_weights = self.conv_tp_weights(edge_feats)
        
        # density normalization
        edge_density = torch.tanh(self.density_fn(edge_feats) ** 2)

        mji = self.conv_tp(
            node_feats[sender], edge_attrs, tp_weights
        )  # [n_edges, irreps]

        # learnable radial for magmom space
        tp_weights_magmom = self.conv_tp_weights_magmom(edge_feats_with_magmom)
        
        magmom_mji = self.magmom_conv_tp(
            node_feats[sender], magmom_node_attrs[sender], tp_weights_magmom
        )  # [n_edges, irreps]
        
        density = scatter_sum(
            src=edge_density, index=receiver, dim=0, dim_size=num_nodes
        )  # [n_nodes, 1]

        # highlighted message for central message

        message = scatter_sum(
            src=mji, index=receiver, dim=0, dim_size=num_nodes
        )  # [n_nodes, irreps]
        
        magmom_message = scatter_sum(
            src=magmom_mji, index=receiver, dim = 0, dim_size=num_nodes,
        )
        
        message = self.linear(message) / (density + 1)
        message = self.skip_tp(message, node_attrs)
        # not doing density normalization for now
        magmom_message = self.magmom_linear(magmom_message) / self.avg_num_neighbors
        magmom_message = self.magmom_skip_tp(magmom_message, node_attrs)

        return (
            self.reshape(message),
            self.reshape(magmom_message),
            None,
            None,
        )  # [n_nodes, channels, (lmax + 1)**2]

@compile_mode("script")
class RealAgnosticDensityResidualInteractionBlock(InteractionBlock):
    def _setup(self) -> None:
        if not hasattr(self, "cueq_config"):
            self.cueq_config = None

        # First linear
        self.linear_up = Linear(
            self.node_feats_irreps,
            self.node_feats_irreps,
            internal_weights=True,
            shared_weights=True,
            cueq_config=self.cueq_config,
        )
        # TensorProduct
        irreps_mid, instructions = tp_out_irreps_with_instructions(
            self.node_feats_irreps,
            self.edge_attrs_irreps,
            self.target_irreps,
        )
        self.conv_tp = TensorProduct(
            self.node_feats_irreps,
            self.edge_attrs_irreps,
            irreps_mid,
            instructions=instructions,
            shared_weights=False,
            internal_weights=False,
            cueq_config=self.cueq_config,
        )

        # Convolution weights
        input_dim = self.edge_feats_irreps.num_irreps
        self.conv_tp_weights = nn.FullyConnectedNet(
            [input_dim] + self.radial_MLP + [self.conv_tp.weight_numel],
            torch.nn.functional.silu,  # gate
        )

        # Linear
        self.irreps_out = self.target_irreps
        self.linear = Linear(
            irreps_mid,
            self.irreps_out,
            internal_weights=True,
            shared_weights=True,
            cueq_config=self.cueq_config,
        )

        # Selector TensorProduct
        self.skip_tp = FullyConnectedTensorProduct(
            self.node_feats_irreps,
            self.node_attrs_irreps,
            self.hidden_irreps,
            cueq_config=self.cueq_config,
        )

        # Density normalization
        self.density_fn = nn.FullyConnectedNet(
            [input_dim]
            + [
                1,
            ],
            torch.nn.functional.silu,
        )

        # Reshape
        self.reshape = reshape_irreps(self.irreps_out, cueq_config=self.cueq_config)

    def forward(
        self,
        node_attrs: torch.Tensor,
        node_feats: torch.Tensor,
        edge_attrs: torch.Tensor,
        edge_feats: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        sender = edge_index[0]
        receiver = edge_index[1]
        num_nodes = node_feats.shape[0]
        sc = self.skip_tp(node_feats, node_attrs)
        node_feats = self.linear_up(node_feats)
        tp_weights = self.conv_tp_weights(edge_feats)
        edge_density = torch.tanh(self.density_fn(edge_feats) ** 2)
        mji = self.conv_tp(
            node_feats[sender], edge_attrs, tp_weights
        )  # [n_edges, irreps]
        density = scatter_sum(
            src=edge_density, index=receiver, dim=0, dim_size=num_nodes
        )  # [n_nodes, 1]
        message = scatter_sum(
            src=mji, index=receiver, dim=0, dim_size=num_nodes
        )  # [n_nodes, irreps]
        message = self.linear(message) / (density + 1)
        return (
            self.reshape(message),
            sc,
        )  # [n_nodes, channels, (lmax + 1)**2]


@compile_mode("script")
class RealAgnosticAttResidualInteractionBlock(InteractionBlock):
    def _setup(self) -> None:
        if not hasattr(self, "cueq_config"):
            self.cueq_config = None
        self.node_feats_down_irreps = o3.Irreps("64x0e")
        # First linear
        self.linear_up = Linear(
            self.node_feats_irreps,
            self.node_feats_irreps,
            internal_weights=True,
            shared_weights=True,
            cueq_config=self.cueq_config,
        )
        # TensorProduct
        irreps_mid, instructions = tp_out_irreps_with_instructions(
            self.node_feats_irreps,
            self.edge_attrs_irreps,
            self.target_irreps,
        )
        self.conv_tp = TensorProduct(
            self.node_feats_irreps,
            self.edge_attrs_irreps,
            irreps_mid,
            instructions=instructions,
            shared_weights=False,
            internal_weights=False,
            cueq_config=self.cueq_config,
        )

        # Convolution weights
        self.linear_down = Linear(
            self.node_feats_irreps,
            self.node_feats_down_irreps,
            internal_weights=True,
            shared_weights=True,
            cueq_config=self.cueq_config,
        )
        input_dim = (
            self.edge_feats_irreps.num_irreps
            + 2 * self.node_feats_down_irreps.num_irreps
        )
        self.conv_tp_weights = nn.FullyConnectedNet(
            [input_dim] + 3 * [256] + [self.conv_tp.weight_numel],
            torch.nn.functional.silu,
        )

        # Linear
        self.irreps_out = self.target_irreps
        self.linear = Linear(
            irreps_mid,
            self.irreps_out,
            internal_weights=True,
            shared_weights=True,
            cueq_config=self.cueq_config,
        )

        self.reshape = reshape_irreps(self.irreps_out, cueq_config=self.cueq_config)

        # Skip connection.
        self.skip_linear = Linear(
            self.node_feats_irreps, self.hidden_irreps, cueq_config=self.cueq_config
        )

    def forward(
        self,
        node_attrs: torch.Tensor,
        node_feats: torch.Tensor,
        edge_attrs: torch.Tensor,
        edge_feats: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> Tuple[torch.Tensor, None]:
        sender = edge_index[0]
        receiver = edge_index[1]
        num_nodes = node_feats.shape[0]
        sc = self.skip_linear(node_feats)
        node_feats_up = self.linear_up(node_feats)
        node_feats_down = self.linear_down(node_feats)
        augmented_edge_feats = torch.cat(
            [
                edge_feats,
                node_feats_down[sender],
                node_feats_down[receiver],
            ],
            dim=-1,
        )
        tp_weights = self.conv_tp_weights(augmented_edge_feats)
        mji = self.conv_tp(
            node_feats_up[sender], edge_attrs, tp_weights
        )  # [n_edges, irreps]
        message = scatter_sum(
            src=mji, index=receiver, dim=0, dim_size=num_nodes
        )  # [n_nodes, irreps]
        message = self.linear(message) / self.avg_num_neighbors
        return (
            self.reshape(message),
            sc,
        )  # [n_nodes, channels, (lmax + 1)**2]


@compile_mode("script")
class ScaleShiftBlock(torch.nn.Module):
    def __init__(self, scale: float, shift: float):
        super().__init__()
        self.register_buffer(
            "scale",
            torch.tensor(scale, dtype=torch.get_default_dtype()),
        )
        self.register_buffer(
            "shift",
            torch.tensor(shift, dtype=torch.get_default_dtype()),
        )

    def forward(self, x: torch.Tensor, head: torch.Tensor) -> torch.Tensor:
        return (
            torch.atleast_1d(self.scale)[head] * x + torch.atleast_1d(self.shift)[head]
        )

    def __repr__(self):
        formatted_scale = (
            ", ".join([f"{x:.4f}" for x in self.scale])
            if self.scale.numel() > 1
            else f"{self.scale.item():.4f}"
        )
        formatted_shift = (
            ", ".join([f"{x:.4f}" for x in self.shift])
            if self.shift.numel() > 1
            else f"{self.shift.item():.4f}"
        )
        return f"{self.__class__.__name__}(scale={formatted_scale}, shift={formatted_shift})"