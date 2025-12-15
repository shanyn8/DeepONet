"""
Test to verify loss computation correctness and identify scaling issues.
"""
import torch
import math
from DeepONet import (
    cheb_sensors,
    chebyshev_diff_matrix,
    get_clenshaw_curtis_weights,
    apply_barycentric_interpolate,
    device,
    dtype,
)
from source_functions import (
    polynomial_exact_solution,
    polynomial_forcing_function,
    DEFAULT_K_VAL,
)

def test_loss_scaling():
    """Check if loss scaling is correct."""
    N_sensors = 64
    M_sensors = 1000
    k_val = DEFAULT_K_VAL
    
    x_N = cheb_sensors(N_sensors).to(device)
    x_sens = x_N.view(N_sensors, 1).to(device)
    x_M = cheb_sensors(M_sensors).to(device)
    x_eval = x_M.view(M_sensors + 1, 1).to(device)
    
    _, D = chebyshev_diff_matrix(N_sensors)
    D2 = D @ D
    Q = get_clenshaw_curtis_weights(M_sensors)
    
    u_exact = polynomial_exact_solution(x_N)
    f_exact = polynomial_forcing_function(x_N)
    
    d2u = D2 @ u_exact
    u_eval, f_eval, d2u_eval = apply_barycentric_interpolate(
        x_sens, x_eval, u_exact, f_exact, d2u
    )
    
    Lu = d2u_eval + k_val**2 * u_eval
    
    # Current loss computation
    Y_exact = math.sqrt(M_sensors + 1) * Q * Lu
    Y_hat = math.sqrt(M_sensors + 1) * Q * f_eval
    
    mse = torch.nn.MSELoss()
    loss_current = mse(Y_exact, Y_hat)
    
    # Alternative 1: Unweighted loss
    loss_unweighted = mse(Lu, f_eval)
    
    # Alternative 2: Weighted loss without sqrt(M+1) scaling
    Y_exact_no_scale = Q * Lu
    Y_hat_no_scale = Q * f_eval
    loss_no_scale = mse(Y_exact_no_scale, Y_hat_no_scale)
    
    # Alternative 3: Weighted L2 norm (proper quadrature)
    # This should be: sum(weights * (Lu - f_eval)^2)
    weights = Q ** 2  # Q is sqrt(weights), so Q^2 = weights
    weighted_squared_diff = weights * (Lu - f_eval) ** 2
    loss_quadrature = torch.sum(weighted_squared_diff)
    
    # Alternative 4: Normalized weighted loss
    loss_normalized = loss_quadrature / (M_sensors + 1)
    
    print("=" * 70)
    print("LOSS COMPUTATION COMPARISON")
    print("=" * 70)
    print(f"\nCurrent loss (with sqrt(M+1) scaling): {loss_current.item():.10e}")
    print(f"Unweighted loss (Lu vs f_eval):         {loss_unweighted.item():.10e}")
    print(f"Weighted loss (no sqrt(M+1) scaling):    {loss_no_scale.item():.10e}")
    print(f"Quadrature loss (sum of weights*res^2): {loss_quadrature.item():.10e}")
    print(f"Normalized quadrature loss:              {loss_normalized.item():.10e}")
    
    print(f"\nPDE Residual statistics:")
    residual = Lu - f_eval
    print(f"  Max |Lu - f_eval|:  {torch.max(torch.abs(residual)).item():.10e}")
    print(f"  Mean |Lu - f_eval|: {torch.mean(torch.abs(residual)).item():.10e}")
    print(f"  RMS:                {torch.sqrt(torch.mean(residual**2)).item():.10e}")
    
    print(f"\nWeighted residual statistics:")
    weighted_residual = Q * residual
    print(f"  Max |Q*(Lu - f_eval)|:  {torch.max(torch.abs(weighted_residual)).item():.10e}")
    print(f"  Mean |Q*(Lu - f_eval)|: {torch.mean(torch.abs(weighted_residual)).item():.10e}")
    print(f"  RMS:                    {torch.sqrt(torch.mean(weighted_residual**2)).item():.10e}")
    
    # Check if the sqrt(M+1) factor is the issue
    print(f"\nScaling factor analysis:")
    print(f"  sqrt(M_sensors + 1) = {math.sqrt(M_sensors + 1):.6f}")
    print(f"  Current loss / Unweighted loss = {loss_current.item() / loss_unweighted.item():.6f}")
    print(f"  Expected ratio (if scaling is correct): {M_sensors + 1:.0f}")
    
    # The current loss should be: (M+1) * mean(Q^2 * (Lu - f_eval)^2)
    # Let's verify:
    expected_loss = (M_sensors + 1) * torch.mean(weights * residual ** 2)
    print(f"\nExpected loss (M+1) * mean(weights * res^2): {expected_loss.item():.10e}")
    print(f"Actual current loss:                           {loss_current.item():.10e}")
    print(f"Difference:                                     {abs(expected_loss.item() - loss_current.item()):.10e}")
    
    print("\n" + "=" * 70)
    
    # Check if the issue is numerical differentiation error
    print("\nChecking numerical differentiation error:")
    # Analytical second derivative for polynomial
    u_double_prime_analytical = -6 * x_eval.view(-1) - 2
    d2u_eval_analytical = u_double_prime_analytical
    
    # Compare numerical vs analytical
    diff_error = torch.abs(d2u_eval - d2u_eval_analytical)
    print(f"  Max |d2u_numerical - d2u_analytical|: {torch.max(diff_error).item():.10e}")
    print(f"  Mean |d2u_numerical - d2u_analytical|: {torch.mean(diff_error).item():.10e}")
    
    # If we use analytical d2u, what would the loss be?
    Lu_analytical = d2u_eval_analytical + k_val**2 * u_eval
    residual_analytical = Lu_analytical - f_eval
    loss_analytical = (M_sensors + 1) * torch.mean(weights * residual_analytical ** 2)
    print(f"\n  Loss with analytical d2u: {loss_analytical.item():.10e}")
    print(f"  Loss with numerical d2u:   {loss_current.item():.10e}")
    print(f"  Error from numerical diff: {abs(loss_analytical.item() - loss_current.item()):.10e}")

if __name__ == "__main__":
    test_loss_scaling()

