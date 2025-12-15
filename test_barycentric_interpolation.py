"""
Comprehensive tests for the updated barycentric_interpolate_vectorized function.
"""
import unittest
import torch
import math
import sys
sys.path.insert(0, '.')

from train_model import (
    barycentric_interpolate_eval,
    barycentric_weights_cheb,
    cheb_sensors,
)


class BarycentricInterpolationTests(unittest.TestCase):
    """Test suite for barycentric interpolation."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.eps = 1e-6
        self.rtol = 1e-5
        self.atol = 1e-6
    
    def test_exact_match_at_nodes(self):
        """Test that interpolation is exact at the original nodes."""
        print("\n=== Test: Exact match at nodes ===")
        N = 5
        # Use descending order nodes (as in actual code)
        x_k = cheb_sensors(N)
        f_k = x_k ** 2 + 2 * x_k + 1  # Polynomial
        
        w = barycentric_weights_cheb(x_k)
        
        # Evaluate at the nodes themselves
        P = barycentric_interpolate_eval(x_k, f_k, x_k, w, eps=self.eps)
        
        print(f"Nodes: {x_k.tolist()}")
        print(f"f_k: {f_k.tolist()}")
        print(f"Interpolated: {P.tolist()}")
        
        self.assertTrue(
            torch.allclose(P, f_k, rtol=self.rtol, atol=self.atol),
            f"Interpolation should be exact at nodes. Max error: {torch.max(torch.abs(P - f_k)).item():.2e}"
        )
        print("✓ PASSED: Exact match at nodes")
    
    def test_polynomial_interpolation_accuracy(self):
        """Test interpolation accuracy for polynomials."""
        print("\n=== Test: Polynomial interpolation accuracy ===")
        N = 10
        x_k = cheb_sensors(N)
        # Test with quadratic polynomial: p(x) = x^2 + 2x + 1
        f_k = x_k ** 2 + 2 * x_k + 1
        
        w = barycentric_weights_cheb(x_k)
        
        # Evaluate at points not in x_k
        x_eval = torch.linspace(-0.9, 0.9, 20, device=x_k.device)
        P = barycentric_interpolate_eval(x_k, f_k, x_eval, w, eps=self.eps)
        expected = x_eval ** 2 + 2 * x_eval + 1
        
        max_error = torch.max(torch.abs(P - expected)).item()
        mean_error = torch.mean(torch.abs(P - expected)).item()
        
        print(f"Max error: {max_error:.2e}")
        print(f"Mean error: {mean_error:.2e}")
        
        self.assertLess(
            max_error, 1e-4,
            f"Polynomial interpolation error too large: {max_error:.2e}"
        )
        print("✓ PASSED: Polynomial interpolation accurate")
    
    def test_descending_vs_ascending_order(self):
        """Test that interpolation works with both descending and ascending node order."""
        print("\n=== Test: Descending vs Ascending order ===")
        N = 5
        
        # Descending order (as used in code)
        x_k_desc = cheb_sensors(N)
        f_k = x_k_desc ** 2
        
        # Ascending order (reverse)
        x_k_asc = torch.flip(x_k_desc, dims=[0])
        f_k_asc = torch.flip(f_k, dims=[0])
        
        w_desc = barycentric_weights_cheb(x_k_desc)
        w_asc = barycentric_weights_cheb(x_k_asc)
        
        # Evaluate at same points
        x_eval = torch.linspace(-0.8, 0.8, 10, device=x_k_desc.device)
        
        P_desc = barycentric_interpolate_eval(x_k_desc, f_k, x_eval, w_desc, eps=self.eps)
        P_asc = barycentric_interpolate_eval(x_k_asc, f_k_asc, x_eval, w_asc, eps=self.eps)
        
        expected = x_eval ** 2
        
        error_desc = torch.max(torch.abs(P_desc - expected)).item()
        error_asc = torch.max(torch.abs(P_asc - expected)).item()
        
        print(f"Descending order max error: {error_desc:.2e}")
        print(f"Ascending order max error: {error_asc:.2e}")
        
        # Both should work, but results might differ slightly due to numerical precision
        self.assertLess(error_desc, 1e-3, "Descending order should work")
        self.assertLess(error_asc, 1e-3, "Ascending order should work")
        print("✓ PASSED: Both orders work")
    
    def test_near_zero_differences(self):
        """Test handling of evaluation points very close to nodes."""
        print("\n=== Test: Near-zero differences ===")
        N = 5
        x_k = cheb_sensors(N)
        f_k = x_k ** 2
        
        w = barycentric_weights_cheb(x_k)
        
        # Evaluate at points very close to nodes
        x_eval = x_k + torch.tensor([1e-7, -1e-7, 1e-8, -1e-8, 1e-9], device=x_k.device)
        
        P = barycentric_interpolate_eval(x_k, f_k, x_eval, w, eps=self.eps)
        expected = x_eval ** 2
        
        max_error = torch.max(torch.abs(P - expected)).item()
        print(f"Max error for near-zero differences: {max_error:.2e}")
        
        self.assertLess(max_error, 1e-3, "Should handle near-zero differences gracefully")
        print("✓ PASSED: Near-zero differences handled")
    
    def test_boundary_points(self):
        """Test interpolation at boundary points."""
        print("\n=== Test: Boundary points ===")
        N = 10
        x_k = cheb_sensors(N)
        f_k = torch.sin(math.pi * x_k)
        
        w = barycentric_weights_cheb(x_k)
        
        # Evaluate at boundaries
        x_eval = torch.tensor([-1.0, 1.0], device=x_k.device)
        P = barycentric_interpolate_eval(x_k, f_k, x_eval, w, eps=self.eps)
        expected = torch.sin(math.pi * x_eval)
        
        error = torch.max(torch.abs(P - expected)).item()
        print(f"Boundary interpolation error: {error:.2e}")
        
        self.assertLess(error, 1e-4, "Boundary interpolation should be accurate")
        print("✓ PASSED: Boundary points accurate")
    
    def test_large_ratio_interpolation(self):
        """Test interpolation from few nodes to many evaluation points."""
        print("\n=== Test: Large ratio interpolation (N << M) ===")
        N = 20  # Few nodes
        M = 100  # Many evaluation points
        
        x_k = cheb_sensors(N)
        f_k = torch.sin(math.pi * x_k)
        
        w = barycentric_weights_cheb(x_k)
        x_eval = torch.linspace(-0.95, 0.95, M, device=x_k.device)
        
        P = barycentric_interpolate_eval(x_k, f_k, x_eval, w, eps=self.eps)
        expected = torch.sin(math.pi * x_eval)
        
        max_error = torch.max(torch.abs(P - expected)).item()
        mean_error = torch.mean(torch.abs(P - expected)).item()
        
        print(f"N={N}, M={M}, Ratio={M/N:.1f}")
        print(f"Max error: {max_error:.2e}")
        print(f"Mean error: {mean_error:.2e}")
        
        # For large ratios, errors can accumulate, but should still be reasonable
        self.assertLess(max_error, 1e-2, f"Large ratio interpolation error too high: {max_error:.2e}")
        print("✓ PASSED: Large ratio interpolation acceptable")
    
    def test_exact_match_handling(self):
        """Test that exact matches (within eps) return the correct function value."""
        print("\n=== Test: Exact match handling ===")
        N = 5
        x_k = cheb_sensors(N)
        f_k = x_k ** 2 + 1
        
        w = barycentric_weights_cheb(x_k)
        
        # Evaluate at a node (exact match)
        x_eval = x_k[2:3]  # Middle node
        P = barycentric_interpolate_eval(x_k, f_k, x_eval, w, eps=self.eps)
        
        print(f"Node value: {x_k[2].item():.6f}")
        print(f"f_k[2]: {f_k[2].item():.6f}")
        print(f"Interpolated: {P[0].item():.6f}")
        
        self.assertTrue(
            torch.allclose(P, f_k[2:3], rtol=1e-5, atol=1e-5),
            f"Exact match should return node value. Got {P[0].item():.6f}, expected {f_k[2].item():.6f}"
        )
        print("✓ PASSED: Exact match returns correct value")
    
    def test_numerical_stability(self):
        """Test numerical stability with various functions."""
        print("\n=== Test: Numerical stability ===")
        N = 15
        x_k = cheb_sensors(N)
        
        test_functions = [
            ("x^2", lambda x: x ** 2),
            ("sin(πx)", lambda x: torch.sin(math.pi * x)),
            ("exp(x)", lambda x: torch.exp(x)),
            ("1/(1+x^2)", lambda x: 1.0 / (1.0 + x ** 2)),
        ]
        
        w = barycentric_weights_cheb(x_k)
        x_eval = torch.linspace(-0.9, 0.9, 30, device=x_k.device)
        
        for name, func in test_functions:
            f_k = func(x_k)
            P = barycentric_interpolate_eval(x_k, f_k, x_eval, w, eps=self.eps)
            expected = func(x_eval)
            
            max_error = torch.max(torch.abs(P - expected)).item()
            print(f"  {name}: max error = {max_error:.2e}")
            
            # Check for NaN or Inf
            self.assertFalse(
                torch.any(torch.isnan(P)) or torch.any(torch.isinf(P)),
                f"{name} produced NaN or Inf"
            )
        
        print("✓ PASSED: Numerical stability maintained")
    
    def test_barycentric_interpolate_eval_wrapper(self):
        """Test the convenience wrapper function."""
        print("\n=== Test: barycentric_interpolate_eval wrapper ===")
        N = 8
        x_k = cheb_sensors(N)
        f_k = x_k ** 2 + x_k + 1
        
        x_eval = torch.linspace(-0.8, 0.8, 15, device=x_k.device)
        w = barycentric_weights_cheb(x_k)
        P = barycentric_interpolate_eval(x_k, f_k, x_eval, w)
        expected = x_eval ** 2 + x_eval + 1
        
        max_error = torch.max(torch.abs(P - expected)).item()
        print(f"Wrapper function max error: {max_error:.2e}")
        
        self.assertLess(max_error, 1e-4, "Wrapper function should work correctly")
        print("✓ PASSED: Wrapper function works")
    
    def test_edge_case_single_eval_point(self):
        """Test with a single evaluation point."""
        print("\n=== Test: Single evaluation point ===")
        N = 5
        x_k = cheb_sensors(N)
        f_k = x_k ** 2
        
        w = barycentric_weights_cheb(x_k)
        x_eval = torch.tensor([0.5], device=x_k.device)
        
        P = barycentric_interpolate_eval(x_k, f_k, x_eval, w, eps=self.eps)
        expected = torch.tensor([0.25], device=x_k.device)
        
        error = torch.abs(P - expected).item()
        print(f"Single point error: {error:.2e}")
        
        self.assertLess(error, 1e-4, "Single point interpolation should work")
        print("✓ PASSED: Single evaluation point works")
    
    def test_eps_parameter_sensitivity(self):
        """Test sensitivity to eps parameter."""
        print("\n=== Test: eps parameter sensitivity ===")
        N = 5
        x_k = cheb_sensors(N)
        f_k = x_k ** 2
        
        w = barycentric_weights_cheb(x_k)
        x_eval = torch.linspace(-0.8, 0.8, 10, device=x_k.device)
        
        eps_values = [1e-8, 1e-6, 1e-4, 1e-3]
        results = []
        
        for eps in eps_values:
            P = barycentric_interpolate_eval(x_k, f_k, x_eval, w, eps=eps)
            expected = x_eval ** 2
            error = torch.max(torch.abs(P - expected)).item()
            results.append((eps, error))
            print(f"  eps={eps:.0e}: max error = {error:.2e}")
        
        # All should produce reasonable results
        for eps, error in results:
            self.assertLess(error, 1e-2, f"eps={eps} produced large error")
        
        print("✓ PASSED: eps parameter sensitivity acceptable")


if __name__ == '__main__':
    unittest.main(verbosity=2)

