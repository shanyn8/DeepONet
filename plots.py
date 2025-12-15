"""
Plotting functions for DeepONet results.

This module provides plotting utilities that adapt based on the
source function generation strategy.
"""

import matplotlib.pyplot as plt
import torch


def get_plot_title(source_function_generation_strategy, forcing_function_type=None):
    """
    Get plot title based on source function generation strategy.
    
    Parameters:
        source_function_generation_strategy: str, 'forcing_function' or 'random_linear'
        forcing_function_type: str, 'sinus' or 'polynomial' (only for forcing_function strategy)
        
    Returns:
        str: Plot title
    """
    if source_function_generation_strategy == 'random_linear':
        return 'DeepONet vs Exact Solution - random linear source functions'
    
    if source_function_generation_strategy == 'forcing_function':
        if forcing_function_type == 'sinus':
            return 'DeepONet vs Exact Solution - sin(x) forcing source functions'
        if forcing_function_type == 'polynomial':
            return 'DeepONet vs Exact Solution - polynomial forcing source functions'
    
    return 'DeepONet vs Exact Solution'


def plot_training_loss(hist, title='Training loss'):
    """
    Plot training loss history.
    
    Parameters:
        hist: list of loss values
        title: str, plot title
    """
    plt.figure(figsize=(5, 3))
    plt.semilogy([h for h in hist], label='loss')
    plt.legend()
    plt.title(title)
    plt.tight_layout()
    plt.show()


def plot_forcing_function(
    model,
    x_N,
    f_N,
    exact_solution,
    source_function_generation_strategy,
    forcing_function_type=None,
    N_sensors=None
):
    """
    Plot predicted vs exact solution for forcing function strategy.
    """
    if source_function_generation_strategy != 'forcing_function':
        print(f"plot_forcing_function only supports 'forcing_function' strategy, "
              f"got '{source_function_generation_strategy}'")
        return
    
    if N_sensors is None:
        N_sensors = x_N.shape[0]
    
    model.eval()
    with torch.no_grad():
        x_plot = x_N.view(N_sensors, 1)
        f_plot = f_N.unsqueeze(0)  # Add batch dimension: (1, N_sensors)
        u_pred_plot = model(f_plot, x_plot)  # Output: (1, N_sensors) already padded
        u_pred_plot = u_pred_plot.squeeze(0)  # Remove batch dimension for plotting: (N_sensors,)
        u_exact_plot = exact_solution
    
    plt.figure(figsize=(8, 5))
    plt.plot(x_plot.cpu().numpy(), u_pred_plot.cpu().numpy(), label='Predicted u(x)', linewidth=2)
    plt.plot(x_plot.cpu().numpy(), u_exact_plot.cpu().numpy(), label='Exact u(x)', linestyle='--')
    plt.title(get_plot_title(source_function_generation_strategy, forcing_function_type))
    plt.xlabel('x')
    plt.ylabel('u(x)')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_random_linear_results(
    model,
    x_N,
    f_N,
    source_function_generation_strategy,
    N_sensors=None,
    num_samples=5
):
    """
    Plot results for random linear source functions.
    
    Parameters:
        model: DeepONet model
        x_N: torch.Tensor, Chebyshev sensor points
        f_N: torch.Tensor, source function values
        source_function_generation_strategy: str, source function strategy
        N_sensors: int, number of sensors (if Nexact_solutionone, inferred from x_N)
        num_samples: int, number of sample functions to plot
    """
    if source_function_generation_strategy != 'random_linear':
        print(f"plot_random_linear_results only supports 'random_linear' strategy, "
              f"got '{source_function_generation_strategy}'")
        return
    
    if N_sensors is None:
        N_sensors = x_N.shape[0] - 1
    
    model.eval()
    with torch.no_grad():
        x_plot = x_N.view(N_sensors, 1)
        
        # Plot multiple sample functions
        plt.figure(figsize=(10, 6))
        
        for i in range(min(num_samples, f_N.shape[0])):
            f_plot = f_N[i].unsqueeze(0)
            u_pred_plot = model(f_plot, x_plot)
            u_pred_plot = u_pred_plot.squeeze(0)
            
            plt.plot(x_plot.cpu().numpy(), u_pred_plot.cpu().numpy(), 
                    label=f'Predicted u(x) - Sample {i+1}', linewidth=2, alpha=0.7)
        
        plt.title(get_plot_title(source_function_generation_strategy))
        plt.xlabel('x')
        plt.ylabel('u(x)')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()


def plot_results(
    model,
    x_N,
    f_N,
    exact_solution,
    source_function_generation_strategy,
    forcing_function_type=None,
    N_sensors=None,
    hist=None
):
    """
    Main plotting function that dispatches to appropriate plot based on strategy.
    
    Parameters:
        model: DeepONet model
        x_N: torch.Tensor, Chebyshev sensor points
        f_N: torch.Tensor, source function values
        exact_solution: callable, exact solution function (for forcing_function strategy)
        source_function_generation_strategy: str, source function strategy
        forcing_function_type: str, forcing function type (if applicable)
        N_sensors: int, number of sensors
        hist: list, training loss history (optional)
    """
    # Plot training loss if provided
    # if hist is not None:
    #     plot_training_loss(hist)
    
    # Plot results based on strategy
    if source_function_generation_strategy == 'forcing_function':
        plot_forcing_function(
            model, x_N, f_N, exact_solution,
            source_function_generation_strategy,
            forcing_function_type,
            N_sensors
        )
    elif source_function_generation_strategy == 'random_linear':
        plot_random_linear_results(
            model, x_N, f_N,
            source_function_generation_strategy,
            N_sensors
        )
    else:
        print(f"Unknown source function generation strategy: {source_function_generation_strategy}")

