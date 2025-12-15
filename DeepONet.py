# Environment & imports
import math, time, numpy as np
import torch, torch.nn as nn, torch.nn.functional as Functional
import matplotlib.pyplot as plt
import unittest
from source_functions import (
    get_forcing_function,
    get_exact_solution,
    DEFAULT_K_VAL
)
from plots import plot_results
from sanity_checks import sanity_test_loss_decreases
from train_model import train_model




# Need to fix f at first - random and then sample from these in batch minimization
# f_sens = torch.randn(1, m_sensors, device=device)

# def generate_source_functions_matrix(x_k):
#     print(f"Source function generation strategy: {source_function_generation_strategy}")
#     if source_function_generation_strategy == 'forcing_function':
#         print(f"Forcing function: {forcing_function_type}")
#         values = forcing_function(x_k)  # shape: (len(x),)
#         return values.unsqueeze(0).repeat(steps, 1)  # repeat along rows
#     elif source_function_generation_strategy == 'random_linear':
#         # Random parameters a_i and b_i for each function f_i
#         a = torch.rand(steps).to(device)  # shape: (n,)
#         b = torch.rand(steps).to(device)  # shape: (n,)
#         return b.unsqueeze(1) + a.unsqueeze(1) * x_k.unsqueeze(0)  # shape: (n, m)
#     else:
#         raise ValueError(f"Invalid value: {source_function_generation_strategy}. Expected 'forcing_function' or 'random_linear'.")



if __name__ == '__main__':
    device = ("cuda" if torch.cuda.is_available()
            else "mps" if getattr(torch.backends, 'mps', None) and torch.backends.mps.is_available()
            else "cpu")
    dtype = torch.float32 if device == "mps" else torch.float64
    torch.set_default_dtype(torch.float32)  #TODO float64

    # dtype = torch.float64
    # torch.set_default_dtype(torch.float64)
    torch.manual_seed(1234)

    print(f"dtype: {dtype}")
    print('Using device:', device)

    # TODO -
    ## polynomial.chebyshev.chebval ?
    # Batch minimization
        
    # === Config ===
    N_sensors   = 20      # number of Chebyshev sensors for branch input f
    M_sensors   = 50     # number of Chebyshev sensors as quadrature points for Clenshaw Curtis Algorithm
    width       = 100     # feature width (branch/trunk)
    depth       = 3       # hidden layers in branch/trunk
    steps       = 2000    # training steps
    lr          = 1e-3    # learning rate
    k_val       = 5.0

    # Random forcing generator: Chebyshev series with decaying coefficients
    Nf_terms = 12         # number of Chebyshev terms for random f
    decay    = 0.8        # geometric decay factor for coeff magnitudes (0<decay<1)


    # Branching of different strategies of the code
    source_function_generation_strategy = 'forcing_function'    # forcing_function|random_linear
    forcing_function_type = 'polynomial'                             # sinus|polynomial
    train_model(device, dtype, N_sensors, M_sensors, steps, lr, k_val,
                width, depth, source_function_generation_strategy, forcing_function_type)

   




