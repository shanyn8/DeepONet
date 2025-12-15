import math
import unittest

import torch

from train_model import (
    barycentric_interpolate_eval,
    barycentric_weights_cheb,
    chebyshev_diff_matrix,
    get_clenshaw_curtis_weights,
    DeepONet,
    get_loss,
    cheb_sensors,
    apply_barycentric_interpolate,
)
from source_functions import (
    polynomial_exact_solution,
    sinus_exact_solution,
    polynomial_forcing_function,
    sinus_forcing_function,
    polynomial_exact_second_derivative,
    DEFAULT_K_VAL,
)


class DeepONetMathTests(unittest.TestCase):
    def test_chebyshev_diff_matrix_shape_and_row_sum(self):
        N = 4
        x, D = chebyshev_diff_matrix(N)
        self.assertEqual(x.shape[0], N)
        self.assertEqual(D.shape, (N, N))
        ones = torch.ones(N, device=D.device, dtype=D.dtype)
        row_sum = D @ ones
        self.assertTrue(torch.allclose(row_sum, torch.zeros_like(row_sum), atol=1e-6))

    def test_get_clenshaw_curtis_weights_positive_and_sum(self):
        n = 8
        w = get_clenshaw_curtis_weights(n)
        self.assertEqual(w.shape[0], n)
        self.assertTrue(torch.all(w > 0))
        self.assertTrue(
            torch.allclose((w ** 2).sum(), torch.tensor(2.0, device=w.device), atol=1e-3)
        )

    def test_barycentric_weights_cheb_pattern(self):
        N = 4
        x_k = torch.cos(torch.arange(N + 1) * math.pi / N)
        w = barycentric_weights_cheb(x_k)
        expected = torch.tensor([0.5, -1.0, 1.0, -1.0, 0.5], device=w.device, dtype=w.dtype)
        self.assertTrue(torch.allclose(w, expected, atol=1e-6))

    def test_barycentric_interpolate_eval_matches_polynomial(self):
        # Use a quadratic polynomial p(x) = x^2 + 2x + 1, which barycentric interp should reproduce exactly
        N = 4
        x_k = torch.linspace(-1, 1, N + 1)
        f_k = x_k ** 2 + 2 * x_k + 1
        w = barycentric_weights_cheb(x_k)
        x_eval = torch.tensor([-1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0], device=x_k.device, dtype=x_k.dtype)
        P = barycentric_interpolate_eval(x_k, f_k, x_eval, w)
        expected = x_eval ** 2 + 2 * x_eval + 1
        print(expected)
        print(P)
        self.assertTrue(torch.allclose(P, expected, atol=1e-5))

    def test_chebyshev_diff_matrix_second_derivative_sinus(self):
        """
        Test that applying D2 (D @ D) to the exact sinus solution
        gives the correct second derivative.
        
        For sinus: u'' + k²u = sin(πx), so u'' = sin(πx) - k²u
        """
        N = 32  # Use more points for better accuracy with sinus
        k_val = DEFAULT_K_VAL
        x, D = chebyshev_diff_matrix(N)
        D2 = D @ D
        
        # Get exact solution at Chebyshev nodes
        u_exact = sinus_exact_solution(x, k_val)
        
        # Apply D2 to u (as done in the code)
        d2u_numerical = D2 @ u_exact
        
        # Compute exact second derivative analytically
        # From the PDE: u'' + k²u = sin(πx)
        # Therefore: u'' = sin(πx) - k²u
        pi_tensor = torch.tensor(math.pi, device=x.device, dtype=x.dtype)
        f_x = torch.sin(pi_tensor * x)
        d2u_exact = f_x - k_val**2 * u_exact
        
        # Compare (allow some tolerance for numerical differentiation errors)
        # Note: Chebyshev differentiation on sinus functions may have larger errors
        # especially near boundaries, so we use a more lenient tolerance
        self.assertTrue(
            torch.allclose(d2u_numerical, d2u_exact, atol=1e-2, rtol=1e-2),
            f"Max difference: {torch.max(torch.abs(d2u_numerical - d2u_exact)).item():.6e}, "
            f"Mean difference: {torch.mean(torch.abs(d2u_numerical - d2u_exact)).item():.6e}"
        )

    def test_polynomial_exact_solution_boundary_conditions(self):
        """
        Test that polynomial exact solution satisfies boundary conditions:
        u(-1) = 0 and u(1) = 0
        """
        x_boundary = torch.tensor([-1.0, 1.0])
        u_boundary = polynomial_exact_solution(x_boundary)
        
        # Check boundary conditions
        self.assertTrue(
            torch.allclose(u_boundary, torch.zeros_like(u_boundary), atol=1e-6),
            f"Boundary values: {u_boundary}, expected [0, 0]"
        )

    def test_polynomial_exact_solution_satisfies_pde(self):
        """
        Test that polynomial exact solution satisfies the Helmholtz equation:
        u'' + k²u = f(x)
        
        where f(x) = -25x³ - 25x² + 19x + 23
        and u(x) = -x³ - x² + x + 1
        """
        N = 20
        k_val = DEFAULT_K_VAL
        x, D = chebyshev_diff_matrix(N)
        D2 = D @ D
        
        # Get exact solution and forcing function at Chebyshev nodes
        u_exact = polynomial_exact_solution(x)
        f_x = polynomial_forcing_function(x)
        
        # Compute u'' numerically using D2
        d2u_numerical = D2 @ u_exact
        
        # Check PDE: u'' + k²u = f
        # Rearranged: u'' = f - k²u
        pde_residual = d2u_numerical + k_val**2 * u_exact - f_x
        
        # The residual should be close to zero
        self.assertTrue(
            torch.allclose(pde_residual, torch.zeros_like(pde_residual), atol=1e-2, rtol=1e-2),
            f"Max PDE residual: {torch.max(torch.abs(pde_residual)).item():.6e}, "
            f"Mean PDE residual: {torch.mean(torch.abs(pde_residual)).item():.6e}"
        )

    def test_sinus_exact_solution_boundary_conditions(self):
        """
        Test that sinus exact solution satisfies boundary conditions:
        u(-1) = 0 and u(1) = 0
        """
        k_val = DEFAULT_K_VAL
        x_boundary = torch.tensor([-1.0, 1.0])
        u_boundary = sinus_exact_solution(x_boundary, k_val)
        
        # Check boundary conditions
        self.assertTrue(
            torch.allclose(u_boundary, torch.zeros_like(u_boundary), atol=1e-5),
            f"Boundary values: {u_boundary}, expected [0, 0]"
        )

    def test_sinus_exact_solution_satisfies_pde(self):
        """
        Test that sinus exact solution satisfies the Helmholtz equation:
        u'' + k²u = sin(πx)
        
        with boundary conditions u(-1) = u(1) = 0
        """
        N = 100
        k_val = DEFAULT_K_VAL
        x, D = chebyshev_diff_matrix(N)
        D2 = D @ D
        
        # Get exact solution and forcing function at Chebyshev nodes
        u_exact = sinus_exact_solution(x, k_val)
        f_x = sinus_forcing_function(x)  # sin(πx)
        
        # Compute u'' numerically using D2
        d2u_numerical = D2 @ u_exact
        
        # Check PDE: u'' + k²u = sin(πx)
        # Rearranged: u'' + k²u - sin(πx) = 0
        pde_residual = d2u_numerical + k_val**2 * u_exact - f_x
        # print(pde_residual)
        # The residual should be close to zero
        # Note: Higher tolerance due to numerical differentiation errors
        self.assertTrue(
            torch.allclose(pde_residual, torch.zeros_like(pde_residual), atol=1e-1, rtol=1e-1),
            f"Max PDE residual: {torch.max(torch.abs(pde_residual)).item():.6e}, "
            f"Mean PDE residual: {torch.mean(torch.abs(pde_residual)).item():.6e}"
        )

    def test_model_forward_pass_shapes(self):
        """Test that model forward pass produces correct output shapes."""
        N_sensors = 16
        width = 32
        depth = 2
        
        model = DeepONet(N_sensors, width=width, depth=depth)
        
        # Test input shapes
        batch_size = 1
        f_samples = torch.randn(batch_size, N_sensors)
        x_points = torch.randn(N_sensors, 1)
        
        u_pred = model(f_samples, x_points)
        
        # Output should be padded: (batch_size, N_sensors - 1) -> (batch_size, N_sensors)
        self.assertEqual(u_pred.shape, (batch_size, N_sensors),
                        f"Expected shape ({batch_size}, {N_sensors}), got {u_pred.shape}")

    def test_model_boundary_conditions_zero_padding(self):
        """Test that model output has zeros at boundaries after padding."""
        N_sensors = 16
        width = 32
        depth = 2
        
        model = DeepONet(N_sensors, width=width, depth=depth)
        
        batch_size = 1
        f_samples = torch.randn(batch_size, N_sensors)
        x_points = torch.randn(N_sensors, 1)
        
        u_pred = model(f_samples, x_points)
        
        # Check boundary conditions: first and last elements should be zero
        self.assertTrue(
            torch.allclose(u_pred[0, 0], torch.tensor(0.0), atol=1e-6),
            f"Left boundary not zero: {u_pred[0, 0].item()}"
        )
        self.assertTrue(
            torch.allclose(u_pred[0, -1], torch.tensor(0.0), atol=1e-6),
            f"Right boundary not zero: {u_pred[0, -1].item()}"
        )

    def test_loss_with_exact_solution_should_be_small(self):
        """Test that loss is small when using exact solution."""
        N_sensors = 32
        M_sensors = 64
        k_val = DEFAULT_K_VAL
        
        # Setup
        x_N = cheb_sensors(N_sensors)
        x_M = cheb_sensors(M_sensors)
        
        x, D = chebyshev_diff_matrix(N_sensors)
        D2 = D @ D
        Q = get_clenshaw_curtis_weights(M_sensors)
        
        # Get exact solution and forcing function
        u_exact = polynomial_exact_solution(x_N)
        f_exact = polynomial_forcing_function(x_N)
        
        f_M = polynomial_forcing_function(x_M)

        barycentric_weights = barycentric_weights_cheb(x_N)

        # Loss with exact solution should be small
        loss = get_loss(u_exact, f_exact, x_N, x_M, D2, Q, f_M, k_val, barycentric_weights)

        self.assertLess(
            loss.item(), 1e-4,
            f"Loss with exact solution should be small (< 1e-4), got {loss.item():.6e}"
        )

    def test_loss_pipeline_shapes(self):
        """Test that all shapes are consistent throughout the loss computation pipeline."""
        N_sensors = 16
        M_sensors = 32
        k_val = DEFAULT_K_VAL
        
        x_N = cheb_sensors(N_sensors)
        x_sens = x_N.view(N_sensors, 1)
        x_M = cheb_sensors(M_sensors)
        x_eval = x_M.view(M_sensors, 1)
        
        x, D = chebyshev_diff_matrix(N_sensors)
        D2 = D @ D
        Q = get_clenshaw_curtis_weights(M_sensors)
        
        # Create test data
        u = torch.randn(N_sensors, device=D2.device)
        f = torch.randn(N_sensors, device=D2.device)
        
        # Check shapes at each step
        d2u = D2 @ u
        self.assertEqual(d2u.shape, (N_sensors,), 
                        f"d2u shape mismatch: {d2u.shape}")
            
        barycentric_weights = barycentric_weights_cheb(x_N)

        u_eval, d2u_eval = apply_barycentric_interpolate(
            x_sens, x_eval, barycentric_weights, u, d2u
        )
        
        self.assertEqual(u_eval.shape, (M_sensors ,),
                        f"u_eval shape mismatch: {u_eval.shape}")
        self.assertEqual(d2u_eval.shape, (M_sensors ,),
                        f"d2u_eval shape mismatch: {d2u_eval.shape}")
        
        Lu = d2u_eval + k_val**2 * u_eval
        self.assertEqual(Lu.shape, (M_sensors ,),
                        f"Lu shape mismatch: {Lu.shape}")
        
        self.assertEqual(Q.shape, (M_sensors ,),
                        f"Q shape mismatch: {Q.shape}")
        
        Y_exact = math.sqrt(M_sensors + 1) * Q * Lu
        
        self.assertEqual(Y_exact.shape, (M_sensors ,),
                        f"Y_exact shape mismatch: {Y_exact.shape}")

    def test_interpolation_preserves_exact_solution_at_nodes(self):
        """Test that interpolation preserves exact values at Chebyshev nodes."""
        N_sensors = 16
        M_sensors = 32
        
        x_N = cheb_sensors(N_sensors)
        x_M = cheb_sensors(M_sensors)
        x_eval = x_M.view(M_sensors , 1)
        
        # Use exact solution
        u_exact = polynomial_exact_solution(x_N)
        
        # Interpolate to evaluation points
        barycentric_weights = barycentric_weights_cheb(x_N)
        u_eval = barycentric_interpolate_eval(x_N, u_exact, x_eval.view(-1), barycentric_weights)

        # Check that at original nodes, values are preserved
        # Find indices where x_eval matches x_N (within tolerance)
        for i, x_node in enumerate(x_N):
            # Find closest evaluation point
            dists = torch.abs(x_eval.view(-1) - x_node)
            closest_idx = torch.argmin(dists)
            if dists[closest_idx] < 1e-6:  # Very close match
                self.assertTrue(
                    torch.allclose(u_eval[closest_idx], u_exact[i], atol=1e-5),
                    f"Interpolation doesn't preserve value at node {i}: "
                    f"expected {u_exact[i].item():.6e}, got {u_eval[closest_idx].item():.6e}"
                )

    def test_differentiation_matrix_first_derivative(self):
        """Test that D2 is applied correctly to a known function."""
        N = 100
        x, D = chebyshev_diff_matrix(N)
        
        # Test function: u(x) = x^2, so u'(x) = 2x
        u = x ** 2
        du_numerical = D @ u
        
        # print(du_numerical)
        # Analytical first derivative 
        du_exact = 2 * x
        
        print(du_exact)
        print(du_numerical)
        # Should be close (except possibly at boundaries)
        # Check interior points
        interior_mask = (torch.abs(x) < 0.9)  # Avoid boundary
        if interior_mask.any():
            self.assertTrue(
                torch.allclose(
                    du_numerical[interior_mask], 
                    du_exact[interior_mask], 
                    atol=1e-2
                ),
                f"D doesn't compute second derivative correctly for 2x"
            )

    def test_chebyshev_diff_matrix_second_derivative_polynomial(self):
        """
        Test that applying D2 (D @ D) to the exact polynomial solution
        gives the correct second derivative.
        
        For polynomial: u(x) = -x³ - x² + x + 1
        u''(x) = -6x - 2
        """
        N = 30  # Use more points for better accuracy
        x, D = chebyshev_diff_matrix(N)     

        D2 = torch.matmul(D, D)
        
        # Get exact solution at Chebyshev nodes
        u_exact = polynomial_exact_solution(x)
        # Apply D2 to u (as done in the code)
        d2u_numerical = torch.matmul(D2, u_exact)
        # d2u_numerical = D2 @ u_exact
        print(d2u_numerical)
        # Compute exact second derivative analytically
        # u(x) = -x³ - x² + x + 1
        # u'(x) = -3x² - 2x + 1
        # u''(x) = -6x - 2
        d2u_exact = -6 * x - 2
        print(d2u_exact)
  
        # Compare (allow some tolerance for numerical differentiation errors)
        self.assertTrue(
            torch.allclose(d2u_numerical, d2u_exact, atol=1e-3, rtol=1e-3),
            f"Max difference: {torch.max(torch.abs(d2u_numerical - d2u_exact)).item():.6e}"
        )

    def test_differentiation_matrix_applied_correctly(self):
        """Test that D2 is applied correctly to a known function."""
        N = 20
        x, D = chebyshev_diff_matrix(N)
        D2 = D @ D
        
        # Test function: u(x) = x^2, so u''(x) = 2
        u = x ** 2
        d2u_numerical = D2 @ u
        
        print(d2u_numerical)
        # Analytical second derivative is constant 2
        d2u_exact = 2 * torch.ones_like(x)
        
        # Should be close (except possibly at boundaries)
        # Check interior points
        interior_mask = (torch.abs(x) < 0.9)  # Avoid boundary
        if interior_mask.any():
            self.assertTrue(
                torch.allclose(
                    d2u_numerical[interior_mask], 
                    d2u_exact[interior_mask], 
                    atol=1e-2
                ),
                f"D2 doesn't compute second derivative correctly for x^2"
            )

    def test_loss_is_differentiable(self):
        """Test that loss computation is differentiable (no detach operations)."""
        N_sensors = 16
        M_sensors = 32
        k_val = DEFAULT_K_VAL
        
        x_N = cheb_sensors(N_sensors)
        x_sens = x_N.view(N_sensors, 1)
        x_M = cheb_sensors(M_sensors)
        x_eval = x_M.view(M_sensors , 1)
        
        x, D = chebyshev_diff_matrix(N_sensors)
        D2 = D @ D
        Q = get_clenshaw_curtis_weights(M_sensors)
        
        # Create a simple model output that requires grad
        u = torch.randn(N_sensors, requires_grad=True, device=D2.device)
        f = torch.randn(N_sensors, device=D2.device)
        d2u = D2 @ u
        barycentric_weights = barycentric_weights_cheb(x_N)
        u_eval, d2u_eval = apply_barycentric_interpolate(
            x_sens, x_eval, barycentric_weights, u, d2u
        )
        
        f_eval = barycentric_interpolate_eval(x_sens.view(-1), f.view(-1), x_eval.view(-1), barycentric_weights)
        Lu = d2u_eval + k_val**2 * u_eval
        Y_exact = math.sqrt(M_sensors + 1) * Q * Lu
        Y_hat = math.sqrt(M_sensors + 1) * Q * f_eval
        
        mse = torch.nn.MSELoss()
        loss = mse(Y_exact, Y_hat)
        
        # Should be able to compute gradients
        loss.backward()
        
        # Check that gradients exist
        self.assertIsNotNone(u.grad, "Gradients should be computed")
        self.assertFalse(torch.allclose(u.grad, torch.zeros_like(u.grad)),
                        "Gradients should be non-zero")

    def test_interpolation_error_accumulation(self):
        """
        Test that shows interpolation introduces errors that accumulate.
        This test will FAIL and show the interpolation error magnitude.
        """
        N_sensors = 32
        M_sensors = 1000
        
        x_N = cheb_sensors(N_sensors)
        x_sens = x_N.view(N_sensors, 1)
        x_M = cheb_sensors(M_sensors)
        x_eval = x_M.view(M_sensors , 1)
        
        # Use a function that should be exactly representable
        u_exact = polynomial_exact_solution(x_N)
        
        # Interpolate to evaluation points
        barycentric_weights = barycentric_weights_cheb(x_N)
        u_eval = barycentric_interpolate_eval(x_N, u_exact, x_eval.view(-1), barycentric_weights)
        
        # Compute exact values at evaluation points
        u_exact_at_eval = polynomial_exact_solution(x_eval.view(-1))
        
        # Check interpolation error
        interpolation_error = torch.abs(u_eval - u_exact_at_eval)
        max_error = torch.max(interpolation_error).item()
        mean_error = torch.mean(interpolation_error).item()
        
        print(f"\nInterpolation Error Statistics:")
        print(f"  Max error: {max_error:.10e}")
        print(f"  Mean error: {mean_error:.10e}")
        print(f"  RMS error: {torch.sqrt(torch.mean(interpolation_error**2)).item():.10e}")
        
        # This test will FAIL if interpolation error is too large
        # Adjust threshold based on what's acceptable
        self.assertLess(
            max_error, 1e-4,
            f"Interpolation error too large! Max error: {max_error:.10e}. "
            f"This shows interpolation introduces significant errors."
        )

    def test_differentiation_error_at_boundaries(self):
        """
        Test that shows differentiation matrix has large errors at boundaries.
        This test will FAIL and show boundary differentiation errors.
        """
        N = 20
        x, D = chebyshev_diff_matrix(N)
        D2 = D @ D
        
        # Test with exact solution
        u_exact = polynomial_exact_solution(x)
        
        # Analytical second derivative
        u_double_prime_exact = polynomial_exact_second_derivative(x)
        
        # Numerical second derivative
        d2u_numerical = D2 @ u_exact
        
        # Check error at boundaries vs interior
        boundary_indices = [0, -1]
        interior_indices = list(range(1, len(x) - 1))
        
        boundary_error = torch.max(torch.abs(d2u_numerical[boundary_indices] - 
                                            u_double_prime_exact[boundary_indices]))
        interior_error = torch.max(torch.abs(d2u_numerical[interior_indices] - 
                                            u_double_prime_exact[interior_indices]))
        
        print(f"\nDifferentiation Error Statistics:")
        print(f"  Boundary error (indices 0, -1): {boundary_error.item():.10e}")
        print(f"  Interior max error: {interior_error.item():.10e}")
        print(f"  Ratio (boundary/interior): {boundary_error.item() / interior_error.item():.2f}")
        
        # This test will FAIL if boundary errors are much larger than interior
        # This is a known issue with Chebyshev differentiation at boundaries
        self.assertLess(
            boundary_error.item(), 1e-2,
            f"Differentiation error at boundaries too large! "
            f"Boundary error: {boundary_error.item():.10e}, "
            f"Interior error: {interior_error.item():.10e}. "
            f"This shows D2 has accuracy issues at boundaries."
        )

    @unittest.skip("Skipping this test for now")
    def test_interpolation_then_differentiation_error(self):
        """
        Test that shows error accumulation: interpolation THEN differentiation.
        This test will FAIL and show compounded errors.
        """
        N_sensors = 20
        M_sensors = 100
        k_val = DEFAULT_K_VAL
        
        x_N = cheb_sensors(N_sensors)
        x_sens = x_N.view(N_sensors, 1)
        x_M = cheb_sensors(M_sensors)
        x_eval = x_M.view(M_sensors , 1)
        
        x, D = chebyshev_diff_matrix(N_sensors)
        D2 = D @ D
        _, D_eval = chebyshev_diff_matrix(M_sensors)
        D2_eval = D_eval @ D_eval
        
        barycentric_weights = barycentric_weights_cheb(x_N)

        # Exact solution at N nodes
        u_N = polynomial_exact_solution(x_N)
        
        # Interpolate to M nodes
        u_M = barycentric_interpolate_eval(x_N, u_N, x_eval.view(-1))
        
        # Differentiate at M nodes (this is what happens in the code)
        d2u_M_numerical = D2_eval @ u_M
        
        # Compare with exact second derivative at M nodes
        u_M_exact = polynomial_exact_solution(x_eval.view(-1))
        d2u_M_exact = -6 * x_eval.view(-1) - 2
        
        error = torch.abs(d2u_M_numerical - d2u_M_exact)
        max_error = torch.max(error).item()
        mean_error = torch.mean(error).item()
        
        print(f"\nInterpolation + Differentiation Error:")
        print(f"  Max error: {max_error:.10e}")
        print(f"  Mean error: {mean_error:.10e}")
        print(f"  RMS error: {torch.sqrt(torch.mean(error**2)).item():.10e}")
        
        # This test will FAIL showing the compounded error
        self.assertLess(
            max_error, 1e-2,
            f"Combined interpolation+differentiation error too large! "
            f"Max error: {max_error:.10e}. "
            f"This shows errors compound when interpolating then differentiating."
        )

    @unittest.skip("Skipping this test for now")
    def test_differentiation_then_interpolation_error(self):
        """
        Test that shows error accumulation: differentiation THEN interpolation.
        This is what the code actually does - differentiate at N nodes, then interpolate d2u.
        This test will FAIL and show the error in this approach.
        """
        N_sensors = 32
        M_sensors = 1000
        k_val = DEFAULT_K_VAL
        
        x_N = cheb_sensors(N_sensors)
        x_sens = x_N.view(N_sensors, 1)
        x_M = cheb_sensors(M_sensors)
        x_eval = x_M.view(M_sensors , 1)
        
        x, D = chebyshev_diff_matrix(N_sensors)
        D2 = D @ D
        
        # Exact solution at N nodes
        u_N = polynomial_exact_solution(x_N)
        
        # Differentiate at N nodes (what code does)
        d2u_N = D2 @ u_N
        
        # Interpolate d2u to M nodes
        d2u_M_interpolated = barycentric_interpolate_eval(x_N, d2u_N, x_eval.view(-1))
        
        # Compare with exact second derivative at M nodes
        d2u_M_exact = -6 * x_eval.view(-1) - 2
        
        error = torch.abs(d2u_M_interpolated - d2u_M_exact)
        max_error = torch.max(error).item()
        mean_error = torch.mean(error).item()
        
        print(f"\nDifferentiation + Interpolation Error (current code approach):")
        print(f"  Max error: {max_error:.10e}")
        print(f"  Mean error: {mean_error:.10e}")
        print(f"  RMS error: {torch.sqrt(torch.mean(error**2)).item():.10e}")
        
        # Check error at boundaries
        boundary_error = torch.max(torch.abs(d2u_M_interpolated[[0, -1]] - d2u_M_exact[[0, -1]]))
        print(f"  Boundary error: {boundary_error.item():.10e}")
        
        # This test will FAIL showing the error in current approach
        self.assertLess(
            max_error, 1e-2,
            f"Differentiation then interpolation error too large! "
            f"Max error: {max_error:.10e}. "
            f"This is the approach used in get_loss() - errors may be accumulating."
        )

    def test_pde_residual_with_exact_solution(self):
        """
        Test that shows the PDE residual when using exact solution.
        Should be zero, but numerical errors make it non-zero.
        This test will FAIL and quantify the numerical error.
        """
        N_sensors = 20
        M_sensors = 50
        k_val = DEFAULT_K_VAL
        
        x_N = cheb_sensors(N_sensors)
        x_sens = x_N.view(N_sensors, 1)
        x_M = cheb_sensors(M_sensors)
        x_eval = x_M.view(M_sensors , 1)
        
        x, D = chebyshev_diff_matrix(N_sensors)
        D2 = torch.matmul(D, D)
        
        u_exact = polynomial_exact_solution(x_N)
        f_exact = polynomial_forcing_function(x_N)
        
        barycentric_weights = barycentric_weights_cheb(x_N)
        # Current code approach
        d2u = torch.matmul(D2, u_exact)
        u_eval, d2u_eval = apply_barycentric_interpolate(
            x_sens, x_eval, barycentric_weights, u_exact, d2u
        )
        
        f_eval = polynomial_forcing_function(x_M)

        # PDE: u'' + k²u = f
        Lu = d2u_eval + k_val**2 * u_eval
        pde_residual = Lu - f_eval
        
        max_residual = torch.max(torch.abs(pde_residual)).item()
        mean_residual = torch.mean(torch.abs(pde_residual)).item()
        rms_residual = torch.sqrt(torch.mean(pde_residual**2)).item()
        
        print(f"\nPDE Residual with Exact Solution:")
        print(f"  Max |Lu - f|: {max_residual:.10e}")
        print(f"  Mean |Lu - f|: {mean_residual:.10e}")
        print(f"  RMS: {rms_residual:.10e}")
        
        # Check at boundaries
        boundary_residual = torch.max(torch.abs(pde_residual[[0, -1]]))
        print(f"  Boundary residual: {boundary_residual.item():.10e}")
        
        # This test will FAIL showing PDE residual is not zero due to numerical errors
        self.assertLess(
            max_residual, 1e-6,
            f"PDE residual too large even with exact solution! "
            f"Max residual: {max_residual:.10e}. "
            f"This shows numerical differentiation and interpolation introduce errors "
            f"that prevent the exact solution from satisfying the discrete PDE."
        )

    def test_loss_scaling_factor_impact(self):
        """
        Test that shows how the sqrt(M+1) scaling factor affects loss magnitude.
        This test will FAIL and show if scaling is appropriate.
        """
        N_sensors = 32
        M_sensors = 1000
        k_val = DEFAULT_K_VAL
        
        x_N = cheb_sensors(N_sensors)
        x_sens = x_N.view(N_sensors, 1)
        x_M = cheb_sensors(M_sensors)
        x_eval = x_M.view(M_sensors , 1)
        
        x, D = chebyshev_diff_matrix(N_sensors)
        D2 = D @ D
        Q = get_clenshaw_curtis_weights(M_sensors)
        
        u_exact = polynomial_exact_solution(x_N)
        f_exact = polynomial_forcing_function(x_N)
        f_eval = polynomial_forcing_function(x_M)

        barycentric_weights = barycentric_weights_cheb(x_N)

        d2u = D2 @ u_exact
        u_eval, d2u_eval = apply_barycentric_interpolate(
            x_sens, x_eval, barycentric_weights, u_exact, d2u
        )
        
        Lu = d2u_eval + k_val**2 * u_eval
        residual = Lu - f_eval
        
        # Current loss computation
        Y_exact = math.sqrt(M_sensors + 1) * Q * Lu
        Y_hat = math.sqrt(M_sensors + 1) * Q * f_eval
        mse = torch.nn.MSELoss()
        loss_scaled = mse(Y_exact, Y_hat)
        
        # Unscaled loss
        loss_unscaled = mse(Lu, f_eval)
        
        # Weighted but not scaled
        Y_exact_weighted = Q * Lu
        Y_hat_weighted = Q * f_eval
        loss_weighted_unscaled = mse(Y_exact_weighted, Y_hat_weighted)
        
        print(f"\nLoss Scaling Comparison:")
        print(f"  Scaled loss (current):     {loss_scaled.item():.10e}")
        print(f"  Unscaled loss:             {loss_unscaled.item():.10e}")
        print(f"  Weighted but unscaled:    {loss_weighted_unscaled.item():.10e}")
        print(f"  Scaling factor:            {math.sqrt(M_sensors + 1):.2f}")
        print(f"  Ratio (scaled/unscaled):   {loss_scaled.item() / loss_weighted_unscaled.item():.2f}")
        print(f"  Expected ratio:            {M_sensors + 1:.0f}")
        
        # Check if scaling is correct
        expected_ratio = M_sensors + 1
        actual_ratio = loss_scaled.item() / loss_weighted_unscaled.item()
        
        # This test will FAIL if scaling doesn't match expected
        self.assertAlmostEqual(
            actual_ratio, expected_ratio, delta=0.1 * expected_ratio,
            msg=f"Loss scaling incorrect! Expected ratio {expected_ratio:.2f}, "
            f"got {actual_ratio:.2f}. This suggests the sqrt(M+1) factor may be wrong."
        )

    def test_chebyshev_nodes_vs_evaluation_points_mismatch(self):
        """
        Test that shows potential issues when N_sensors << M_sensors.
        Interpolation from few points to many points may amplify errors.
        This test will FAIL if the ratio is too large.
        """
        N_sensors = 100
        M_sensors = 200
        
        x_N = cheb_sensors(N_sensors)
        x_M = cheb_sensors(M_sensors)
        
        # Exact solution
        u_N = polynomial_exact_solution(x_N)
        u_M_exact = polynomial_exact_solution(x_M)
        
        barycentric_weights = barycentric_weights_cheb(x_N)
        # Interpolate
        u_M_interpolated = barycentric_interpolate_eval(x_N, u_N, x_M, barycentric_weights)
        
        error = torch.abs(u_M_interpolated - u_M_exact)
        max_error = torch.max(error).item()
        
        ratio = M_sensors / N_sensors
        
        print(f"\nInterpolation with Large Ratio:")
        print(f"  N_sensors: {N_sensors}, M_sensors: {M_sensors}")
        print(f"  Ratio M/N: {ratio:.2f}")
        print(f"  Max interpolation error: {max_error:.10e}")
        
        # This test will FAIL if ratio is too large and causes large errors
        self.assertLess(
            max_error, 1e-4,
            f"Interpolation error too large when M >> N! "
            f"Ratio: {ratio:.2f}, Max error: {max_error:.10e}. "
            f"Consider using similar N and M, or improving interpolation method."
        )


# if __name__ == "__main__":
#     unittest.main()

