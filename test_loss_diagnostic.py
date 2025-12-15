"""
Diagnostic test to check why exact solution loss is not zero.
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
    sinus_exact_solution,
    sinus_forcing_function,
    DEFAULT_K_VAL,
)

def diagnostic_loss_check():
    """Detailed diagnostic of loss computation with exact solution."""
    N_sensors = 64
    M_sensors = 1000
    k_val = DEFAULT_K_VAL
    
    print("=" * 60)
    print("LOSS DIAGNOSTIC - Exact Solution")
    print("=" * 60)
    
    # Setup
    x_N = cheb_sensors(N_sensors).to(device)
    x_sens = x_N.view(N_sensors + 1, 1).to(device)
    x_M = cheb_sensors(M_sensors).to(device)
    x_eval = x_M.view(M_sensors + 1, 1).to(device)
    
    _, D = chebyshev_diff_matrix(N_sensors)
    D2 = D @ D
    Q = get_clenshaw_curtis_weights(M_sensors)
    
    # Test with polynomial exact solution
    print("\n--- Polynomial Exact Solution ---")
    u_exact = polynomial_exact_solution(x_N)
    f_exact = polynomial_forcing_function(x_N)
    
    print(f"u_exact shape: {u_exact.shape}")
    print(f"u_exact[0] (left boundary): {u_exact[0].item():.10e}")
    print(f"u_exact[-1] (right boundary): {u_exact[-1].item():.10e}")
    print(f"f_exact shape: {f_exact.shape}")
    
    # Check boundary conditions
    print(f"\nBoundary condition check:")
    print(f"  u(-1) should be 0: {u_exact[0].item():.10e}")
    print(f"  u(1) should be 0: {u_exact[-1].item():.10e}")
    
    # Compute second derivative
    d2u = D2 @ u_exact
    print(f"\nd2u shape: {d2u.shape}")
    print(f"d2u[0]: {d2u[0].item():.10e}")
    print(f"d2u[-1]: {d2u[-1].item():.10e}")
    
    # Interpolate
    u_eval, f_eval, d2u_eval = apply_barycentric_interpolate(
        x_sens, x_eval, u_exact, f_exact, d2u
    )
    
    print(f"\nAfter interpolation:")
    print(f"  u_eval shape: {u_eval.shape}")
    print(f"  f_eval shape: {f_eval.shape}")
    print(f"  d2u_eval shape: {d2u_eval.shape}")
    
    # Differential operator
    Lu = d2u_eval + k_val**2 * u_eval
    print(f"\nDifferential operator:")
    print(f"  Lu = d2u_eval + k²*u_eval")
    print(f"  Lu shape: {Lu.shape}")
    print(f"  Lu[0]: {Lu[0].item():.10e}")
    print(f"  Lu[-1]: {Lu[-1].item():.10e}")
    print(f"  f_eval[0]: {f_eval[0].item():.10e}")
    print(f"  f_eval[-1]: {f_eval[-1].item():.10e}")
    
    # Check PDE residual: Lu - f should be close to zero
    pde_residual = Lu - f_eval
    print(f"\nPDE Residual (Lu - f_eval):")
    print(f"  Max |Lu - f|: {torch.max(torch.abs(pde_residual)).item():.10e}")
    print(f"  Mean |Lu - f|: {torch.mean(torch.abs(pde_residual)).item():.10e}")
    print(f"  RMS: {torch.sqrt(torch.mean(pde_residual**2)).item():.10e}")
    
    # Weighted values
    Y_exact = math.sqrt(M_sensors + 1) * Q * Lu
    Y_hat = math.sqrt(M_sensors + 1) * Q * f_eval
    
    print(f"\nWeighted values:")
    print(f"  Y_exact shape: {Y_exact.shape}")
    print(f"  Y_hat shape: {Y_hat.shape}")
    print(f"  Max |Y_exact - Y_hat|: {torch.max(torch.abs(Y_exact - Y_hat)).item():.10e}")
    print(f"  Mean |Y_exact - Y_hat|: {torch.mean(torch.abs(Y_exact - Y_hat)).item():.10e}")
    
    # Final loss
    mse = torch.nn.MSELoss()
    loss = mse(Y_exact, Y_hat)
    print(f"\nFinal MSE Loss: {loss.item():.10e}")
    
    # Alternative: unweighted loss
    unweighted_loss = mse(Lu, f_eval)
    print(f"Unweighted Loss (Lu vs f_eval): {unweighted_loss.item():.10e}")
    
    # Check if the issue is with interpolation or differentiation
    print(f"\n--- Checking at original nodes (no interpolation) ---")
    Lu_original = d2u + k_val**2 * u_exact
    pde_residual_original = Lu_original - f_exact
    print(f"  Max |Lu - f| at original nodes: {torch.max(torch.abs(pde_residual_original)).item():.10e}")
    print(f"  Mean |Lu - f| at original nodes: {torch.mean(torch.abs(pde_residual_original)).item():.10e}")
    
    # Test with sinus
    print("\n\n--- Sinus Exact Solution ---")
    u_sinus = sinus_exact_solution(x_N, k_val)
    f_sinus = sinus_forcing_function(x_N)
    
    print(f"u_sinus[0] (left boundary): {u_sinus[0].item():.10e}")
    print(f"u_sinus[-1] (right boundary): {u_sinus[-1].item():.10e}")
    
    d2u_sinus = D2 @ u_sinus
    u_eval_s, f_eval_s, d2u_eval_s = apply_barycentric_interpolate(
        x_sens, x_eval, u_sinus, f_sinus, d2u_sinus
    )
    
    Lu_sinus = d2u_eval_s + k_val**2 * u_eval_s
    pde_residual_sinus = Lu_sinus - f_eval_s
    
    print(f"  Max |Lu - f|: {torch.max(torch.abs(pde_residual_sinus)).item():.10e}")
    print(f"  Mean |Lu - f|: {torch.mean(torch.abs(pde_residual_sinus)).item():.10e}")
    
    Y_exact_s = math.sqrt(M_sensors + 1) * Q * Lu_sinus
    Y_hat_s = math.sqrt(M_sensors + 1) * Q * f_eval_s
    loss_sinus = mse(Y_exact_s, Y_hat_s)
    print(f"  Final MSE Loss: {loss_sinus.item():.10e}")
    
    print("\n" + "=" * 60)

if __name__ == "__main__":
    diagnostic_loss_check()

