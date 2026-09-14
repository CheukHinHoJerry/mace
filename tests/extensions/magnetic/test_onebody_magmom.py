import pytest
import torch

from mace.modules.extensions import ChebyshevBasisGeneral, MagneticScaleShiftMACE
from mace.tools.torch_tools import default_dtype

# ----------------------------------------------------------
# Spectral damping of the one-body basis
# ----------------------------------------------------------
# Chebyshev degree k is attenuated by 1/(1+k)**p, so a linear head built on this
# basis prefers smooth, low-frequency curves. p=0 must be an exact no-op.


@pytest.mark.parametrize("p", [0.0, 1.0, 2.0])
def test_degree_scale_matches_the_definition(p):
    """The stored scale must be exactly (1 + k)^-p over the basis degrees."""
    with default_dtype(torch.float64):
        basis = ChebyshevBasisGeneral(
            r_max=1.0, num_basis=10, include_constant=True, degree_scale_power=p
        )
        expected = (1.0 + basis.n.to(torch.float64)) ** (-p)
        assert torch.allclose(basis.degree_scale, expected, atol=1e-14)


def test_zero_power_is_an_exact_no_op():
    """p=0 must reproduce the unscaled basis bit-for-bit, not merely closely."""
    with default_dtype(torch.float64):
        x = torch.linspace(-1.0, 1.0, 32, dtype=torch.float64).unsqueeze(-1)
        plain = ChebyshevBasisGeneral(r_max=1.0, num_basis=10, include_constant=True)
        scaled = ChebyshevBasisGeneral(
            r_max=1.0, num_basis=10, include_constant=True, degree_scale_power=0.0
        )
        assert torch.equal(plain(x), scaled(x))


@pytest.mark.parametrize("p", [1.0, 2.0])
def test_damping_suppresses_high_degrees_more_than_low(p):
    """Smoothing must be monotone in the degree, otherwise it is not smoothing."""
    with default_dtype(torch.float64):
        basis = ChebyshevBasisGeneral(
            r_max=1.0, num_basis=10, include_constant=True, degree_scale_power=p
        )
        scale = basis.degree_scale
        assert torch.all(scale[1:] < scale[:-1]), "scale must decrease with degree"
        assert scale[0] == pytest.approx(1.0), "degree 0 must be untouched"


def test_missing_buffer_behaves_as_unscaled():
    """Checkpoints predating this feature have no degree_scale and must still run."""
    with default_dtype(torch.float64):
        x = torch.linspace(-1.0, 1.0, 16, dtype=torch.float64).unsqueeze(-1)
        basis = ChebyshevBasisGeneral(r_max=1.0, num_basis=8, include_constant=True)
        reference = basis(x)
        del basis.degree_scale
        assert torch.equal(basis(x), reference)


# ----------------------------------------------------------
# One-body curve helpers
# ----------------------------------------------------------
class _Head(torch.nn.Module):  # pylint: disable=abstract-method
    """Minimal stand-in exposing just what the two one-body helpers need.

    The helpers are borrowed as class attributes rather than reimplemented, so these
    tests exercise the real methods.
    """

    onebody_curvature_penalty = MagneticScaleShiftMACE.onebody_curvature_penalty
    one_body_zero_offset = MagneticScaleShiftMACE.one_body_zero_offset

    def __init__(self, coeffs, degree_scale_power=0.0):
        super().__init__()
        self.onebody_magmombasis_coeffs = torch.nn.Parameter(coeffs)
        self.one_body_cheb_basis_with_const = ChebyshevBasisGeneral(
            r_max=1.0,
            num_basis=coeffs.shape[1],
            include_constant=True,
            degree_scale_power=degree_scale_power,
        )


def _make_head(coeffs, p=0.0):
    return _Head(coeffs, p)


def test_zero_offset_is_the_curve_value_at_zero_moment():
    """Subtracting the offset must make the one-body term vanish at |m| = 0.

    At |m| = 0 the transform is 1 - 2*0**2 = 1 and every Chebyshev satisfies T_b(1) = 1,
    so the offset is the plain coefficient sum.
    """
    with default_dtype(torch.float64):
        torch.manual_seed(0)
        coeffs = torch.randn(3, 6, 2, dtype=torch.float64)
        head = _make_head(coeffs)

        offset = head.one_body_zero_offset()  # (S, H)
        # evaluate the curve directly at |m| = 0
        radials = head.one_body_cheb_basis_with_const(
            torch.ones(1, 1, dtype=torch.float64)
        ).reshape(1, -1)
        curve_at_zero = torch.einsum("gb,sbh->sh", radials, coeffs)

        assert torch.allclose(offset, curve_at_zero, atol=1e-12)
        assert torch.allclose(
            curve_at_zero - offset, torch.zeros_like(offset), atol=1e-12
        )


def test_curvature_penalty_is_zero_for_a_flat_curve():
    """A constant curve has zero second derivative, so the penalty must vanish."""
    with default_dtype(torch.float64):
        coeffs = torch.zeros(2, 6, 1, dtype=torch.float64)
        coeffs[:, 0, :] = 1.7  # only the constant Chebyshev term
        assert _make_head(coeffs).onebody_curvature_penalty().item() == pytest.approx(
            0.0, abs=1e-20
        )


def test_curvature_penalty_grows_with_roughness():
    """A high-degree (rough) curve must be penalised more than a low-degree one."""
    with default_dtype(torch.float64):
        smooth = torch.zeros(1, 8, 1, dtype=torch.float64)
        smooth[0, 1, 0] = 1.0  # T_1
        rough = torch.zeros(1, 8, 1, dtype=torch.float64)
        rough[0, 7, 0] = 1.0  # T_7
        p_smooth = _make_head(smooth).onebody_curvature_penalty().item()
        p_rough = _make_head(rough).onebody_curvature_penalty().item()
        assert p_rough > p_smooth, f"rough {p_rough:.3e} !> smooth {p_smooth:.3e}"


def test_curvature_penalty_is_differentiable_wrt_coefficients():
    """It is added to the training loss, so it must carry gradient to the head."""
    with default_dtype(torch.float64):
        torch.manual_seed(0)
        coeffs = torch.randn(2, 6, 1, dtype=torch.float64)
        head = _make_head(coeffs)
        penalty = head.onebody_curvature_penalty()
        penalty.backward()
        grad = head.onebody_magmombasis_coeffs.grad
        assert grad is not None and bool(grad.abs().sum() > 0)


def test_spectral_damping_lowers_the_curvature_penalty():
    """The two mechanisms must act in the same direction: damping smooths the curve."""
    with default_dtype(torch.float64):
        rough = torch.zeros(1, 8, 1, dtype=torch.float64)
        rough[0, 7, 0] = 1.0
        undamped = _make_head(rough, p=0.0).onebody_curvature_penalty().item()
        damped = _make_head(rough, p=2.0).onebody_curvature_penalty().item()
        assert damped < undamped, f"damped {damped:.3e} !< undamped {undamped:.3e}"
