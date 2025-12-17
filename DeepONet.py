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


if __name__ == '__main__':
    device = ("cuda" if torch.cuda.is_available()
            else "mps" if getattr(torch.backends, 'mps', None) and torch.backends.mps.is_available()
            else "cpu")
    dtype = torch.float64
    torch.set_default_dtype(torch.float64)  #TODO float64

    # device = 'cpu'
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

    # Branching of different strategies of the code
    source_function_generation_strategy = 'forcing_function'    # forcing_function|random_linear
    forcing_function_type = 'sinus'                             # sinus|polynomial

    train_model(device, dtype, N_sensors, M_sensors, steps, lr, k_val,
                width, depth, source_function_generation_strategy, forcing_function_type)

   




