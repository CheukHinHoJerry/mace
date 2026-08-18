import pytest
import torch
from e3nn import o3

from mace.tools.torch_tools import default_dtype

# ----------------------------------------------------------
# Node-chunking of the non-SOC symmetric contraction
# ----------------------------------------------------------
# Chunking only partitions the node axis and concatenates, so it must be
# numerically identical to the single-shot reference path, gradients included.


@pytest.mark.parametrize("chunk_size", [1, 7, 20, 1000])
def test_nonsoc_contraction_chunking_matches_reference(chunk_size):
    """Chunked and single-shot contractions agree in both forward and backward."""
    pytest.importorskip("cuequivariance")
    from mace.modules.symmetric_contraction_nonsoc import NonSOCContraction

    with default_dtype(torch.float64):
        torch.manual_seed(0)
        nf = 8
        c = NonSOCContraction(
            irreps_in=o3.Irreps(f"{nf}x0e+{nf}x1o+{nf}x2e"),
            irrep_out=o3.Irreps("0e"),
            correlation=3,
            num_elements=2,
            magmom_irreps=o3.Irreps("1x0e+1x1o"),
        )
        s = c.U_tensors(1).shape[0]
        m = c.U_magmom_tensors(1).shape[0]
        n_nodes = 20
        x = torch.randn(n_nodes, nf, s, m, requires_grad=True)
        y = torch.nn.functional.one_hot(torch.randint(0, 2, (n_nodes,)), 2).double()

        c.chunk_size = None
        ref = c(x, y)
        c.chunk_size = chunk_size
        chunked = c(x, y)

        assert torch.allclose(ref, chunked, atol=1e-10), (
            f"forward mismatch for chunk_size={chunk_size}: "
            f"{(ref - chunked).abs().max().item()}"
        )

        gref = torch.autograd.grad(ref.pow(2).sum(), x, retain_graph=True)[0]
        gchk = torch.autograd.grad(chunked.pow(2).sum(), x)[0]
        assert torch.allclose(gref, gchk, atol=1e-10)
