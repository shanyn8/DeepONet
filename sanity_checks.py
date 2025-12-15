"""
Sanity checks for DeepONet training.

This module provides sanity tests to verify that the training process
is working correctly.
"""

import torch
import torch.nn as nn
from source_functions import get_forcing_function, get_exact_solution, DEFAULT_K_VAL


def sanity_test_loss_decreases(
    model,
    cheb_sensors,
    chebyshev_diff_matrix,
    get_clenshaw_curtis_weights,
    apply_barycentric_interpolate,
    barycentric_weights_cheb,
    N_sensors,
    M_sensors,
    width,
    depth,
    lr,
    k_val,
    device,
    x_eval,
    forcing_function_type='sinus',
    num_test_steps=10
):
    """
    Sanity test: Run training flow with fixed sinus forcing function 
    and verify that loss decreases in each training step.
    
    Parameters:
        model: DeepONet model class
        cheb_sensors: Function to generate Chebyshev sensors
        chebyshev_diff_matrix: Function to generate differentiation matrix
        get_clenshaw_curtis_weights_mat: Function to get Clenshaw-Curtis weights
        apply_barycentric_interpolate: Function for barycentric interpolation
        N_sensors: Number of Chebyshev sensors
        M_sensors: Number of evaluation sensors
        width: Network width
        depth: Network depth
        lr: Learning rate
        k_val: Wave number for Helmholtz equation
        device: Torch device
        x_eval: Evaluation points tensor
        forcing_function_type: Type of forcing function ('sinus' or 'polynomial')
        num_test_steps: Number of training steps to run
        
    Returns:
        bool: True if loss decreases in each step, False otherwise
    """
    print("\n=== Running Sanity Test: Loss Decreases ===")
    print(f"Forcing function type: {forcing_function_type}")
    
    # Create test model and optimizer
    test_model = model(N_sensors, N_sensors - 2, width=width, depth=depth).to(device)
    test_opt = torch.optim.Adam(test_model.parameters(), lr=lr)
    
    # Generate fixed forcing function
    test_x_N = cheb_sensors(N_sensors).to(device)
    test_x_M = cheb_sensors(M_sensors).to(device)
    forcing_func = get_forcing_function(forcing_function_type)
    test_f_N = forcing_func(test_x_N).unsqueeze(0)  # (1, N_sensors)
    test_x_sens = test_x_N.view(N_sensors, 1).to(device)
    test_barycentric_weights = barycentric_weights_cheb(test_x_N)

    # Get test matrices
    _, test_D = chebyshev_diff_matrix(N_sensors)
    test_D2 = test_D @ test_D
    test_Q = get_clenshaw_curtis_weights(M_sensors)
    
    # Create a local get_loss function with test matrices
    def test_get_loss(u, f):
        # u is already padded with zeros at boundaries (shape: batch_size, N_sensors)
        # f shape: (batch_size, N_sensors)
        d2u = (test_D2 @ u.T).T  # (batch_size, N_sensors)
        
        # Barycentric Interpolation - N cheb Nodes -> M cheb Nodes
        u_flat = u.squeeze(0) if u.dim() > 1 and u.shape[0] == 1 else u.view(-1)
        d2u_flat = d2u.squeeze(0) if d2u.dim() > 1 and d2u.shape[0] == 1 else d2u.view(-1)
        f_eval = forcing_func(test_x_M)

        u_eval, d2u_eval = apply_barycentric_interpolate(
            test_x_sens, x_eval, test_barycentric_weights, u_flat, d2u_flat
        )
        
        # Differential operator
        Lu = d2u_eval + k_val**2 * u_eval
        Y_exact = (M_sensors + 1)**0.5 * test_Q * Lu
        Y_hat = (M_sensors + 1)**0.5 * test_Q * f_eval
        
        test_mse = nn.MSELoss()
        loss = test_mse(Y_exact, Y_hat)
        return loss
    
    # Run training steps
    test_losses = []
    test_model.train()
    
    for step in range(1, num_test_steps + 1):
        test_opt.zero_grad()
        u_pred = test_model(test_f_N, test_x_sens)
        loss = test_get_loss(u_pred, test_f_N)
        
        loss.backward()
        test_opt.step()
        test_losses.append(loss.item())
        print(f"  Step {step:3d}: loss = {loss.item():.6e}")
    
    # Verify loss decreases (allow for small fluctuations)
    losses_decreasing = all(
        test_losses[i] >= test_losses[i+1] - 1e-6 
        for i in range(len(test_losses)-1)
    )
    
    if losses_decreasing:
        print("✓ PASSED: Loss decreases in each training step")
    else:
        print("✗ FAILED: Loss does not decrease in all steps")
        print(f"  Loss sequence: {test_losses}")
    
    print("=== Sanity Test Complete ===\n")
    return losses_decreasing

