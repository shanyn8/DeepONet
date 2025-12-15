"""
Fixed source functions for DeepONet training.

This module provides predefined forcing functions and their exact solutions
for testing and validation of the DeepONet model.
"""

import torch
import math


# Configuration
FORCING_FUNCTION_TYPES = ['sinus', 'polynomial']
DEFAULT_K_VAL = 5.0  # Distribution for k (avoid resonance k≈1 and sin(k)≈0)


def sinus_forcing_function(x):
    """
    Sinusoidal forcing function: f(x) = sin(πx)
    """
    return torch.sin(torch.pi * x)


def polynomial_forcing_function(x):
    """
    Polynomial forcing function: f(x) = -25x³ - 25x² + 19x + 23
    """
    return -25*x**3 - 25*x**2 + 19*x + 23


def sinus_exact_solution(x, k_val=DEFAULT_K_VAL):
    """
    Exact solution for the Helmholtz equation with sinusoidal forcing.
    Solves: u'' + k²u = sin(πx) with boundary conditions u(-1) = u(1) = 0
    """
    pi_tensor = torch.tensor(math.pi)
    k_tensor = torch.tensor(k_val, dtype=torch.float32)

    denom = k_tensor**2 - pi_tensor**2
    rhs = torch.sin(pi_tensor) / denom

    # Compute sin and cos for k
    sin_k = torch.sin(k_tensor)
    cos_k = torch.cos(k_tensor)
    sin_neg_k = torch.sin(-k_tensor)
    cos_neg_k = torch.cos(-k_tensor)

    # Build system matrix and RHS
    M = torch.stack([
        torch.tensor([sin_neg_k.item(), cos_neg_k.item()]),
        torch.tensor([sin_k.item(), cos_k.item()])
    ])
    b = torch.tensor([-rhs.item(), -rhs.item()])

    # Solve for A and B
    AB = torch.linalg.solve(M, b)
    A, B = AB[0], AB[1]

    # Compute solution
    u_p = torch.sin(pi_tensor * x) / denom
    u_h = A * torch.sin(k_tensor * x) + B * torch.cos(k_tensor * x)
    return u_p + u_h


def polynomial_exact_solution(x):
    """
    Exact solution for the polynomial forcing function.        
    Returns:
        torch.Tensor: exact solution u(x) = -x³ - x² + x + 1
    """
    return -1*x**3 - 1*x**2 + 1*x + 1

def polynomial_exact_first_derivative(x):
    return -3*x**2 - 2*x + 1

def polynomial_exact_second_derivative(x):
    return -6*x - 2

def get_forcing_function(forcing_type):
    if forcing_type == 'sinus':
        return sinus_forcing_function
    elif forcing_type == 'polynomial':
        return polynomial_forcing_function
    else:
        raise ValueError(f"Invalid forcing_type: {forcing_type}. Expected one of {FORCING_FUNCTION_TYPES}")


def get_exact_solution(forcing_type, k_val=DEFAULT_K_VAL):
    if forcing_type == 'sinus':
        return lambda x: sinus_exact_solution(x, k_val)
    elif forcing_type == 'polynomial':
        return polynomial_exact_solution
    else:
        raise ValueError(f"Invalid forcing_type: {forcing_type}. Expected one of {FORCING_FUNCTION_TYPES}")

