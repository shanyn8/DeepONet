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
# barycentric_weights_cheb should be global
## polynomial.chebyshev.chebval ?
# Batch minimization
    
# === Config ===
N_sensors   = 20      # number of Chebyshev sensors for branch input f
M_sensors   = 100     # number of Chebyshev sensors as quadrature points for Clenshaw Curtis Algorithm
width       = 128     # feature width (branch/trunk)
depth       = 3       # hidden layers in branch/trunk
B_batch     = 16      # number of functions per batch (operator learning)
N_bc        = 2       # number of BC points per boundary per function
steps       = 2000    # training steps
lr          = 1e-3    # learning rate
# Distribution for k (avoid resonance k≈1 and sin(k)≈0 for the test analytic comparison)
k_val = DEFAULT_K_VAL

# Random forcing generator: Chebyshev series with decaying coefficients
Nf_terms = 12         # number of Chebyshev terms for random f
decay    = 0.8        # geometric decay factor for coeff magnitudes (0<decay<1)


# Branching of different strategies of the code
source_function_generation_strategy = 'forcing_function'    # forcing_function|random_linear
forcing_function_type = 'polynomial'                             # sinus|polynomial

def print_diff(a, b):
    print(f"Max difference: {torch.max(torch.abs(a - b)).item():.6e}, ")
    print(f"Mean difference: {torch.mean(torch.abs(a - b)).item():.6e}")


def forcing_function(x):
    """Wrapper function that uses the selected forcing function type."""
    func = get_forcing_function(forcing_function_type)
    return func(x)


def exact_solution(x):
    """Wrapper function that uses the selected exact solution type."""
    func = get_exact_solution(forcing_function_type, k_val)
    return func(x)


def cheb_sensors(m, a=-1.0, b=1.0):
    """
    Generate m Chebyshev nodes in descending order (from 1 to -1).
    For m nodes including endpoints: x_i = cos(i*π/(m-1)) for i=0,...,m-1
    """
    if m < 2:
        raise ValueError(f"m must be >= 2, got {m}")
    print(f"Generate {m} chebyshev sensors on [{a}, {b}] (descending order)")
    j = torch.arange(m, dtype=dtype, device=device)
    xi = torch.cos(j * math.pi / (m - 1))  # nodes in [-1, 1], descending order
    
    # Scale from [-1, 1] to [a, b]
    scaled_xi = 0.5 * (b - a) * xi + 0.5 * (b + a)
    return scaled_xi.to(torch.float32)


def chebyshev_diff_matrix(N: int):
    # """
    # Compute Chebyshev differentiation matrix for N nodes.
    # Returns N nodes in descending order and (N, N) differentiation matrix.
    # """
    # if N < 2:
    #     raise ValueError(f"N must be >= 2, got {N}")
    # x = cheb_sensors(N)
    # c = torch.ones(N, device=device, dtype=dtype)
    # c[0], c[-1] = 2, 2
    # c = c * ((-1) ** torch.arange(N, device=device, dtype=dtype))
    # X = x.repeat(N, 1)
    # # dX = X - X.T
    # dX = X.T - X
    # D = torch.outer(c, 1 / c) / (dX + torch.eye(N, device=device, dtype=dtype))
    # # D = (c.unsqueeze(1) / c.unsqueeze(0)) / (dX + torch.eye(N, device=device, dtype=dtype))
    # D = D - torch.diag(torch.sum(D, dim=1))

    # for i in range(N):
    #     if i == 0:
    #         D[i, i] = (2 * N ** 2 + 1) / 6
    #     elif i == N:
    #         D[i, i] = -(2 * N ** 2 + 1) / 6
    #     else:
    #         D[i, i] = -x[i] / (2 * (1 - x[i] ** 2))

    # return x, D


    j = torch.arange(N, device=device, dtype=dtype)
    theta = (N - 1 - j) * math.pi / (N - 1)
    x_k = torch.cos(theta)  # nodes in [-1, 1], ascending order

    n = x_k.shape[0]

    # c_i: 2 for endpoints, 1 otherwise
    c = torch.ones(n, device=device, dtype=dtype)
    c[0] = 2.0
    c[-1] = 2.0

    # Initialize differentiation matrix
    D = torch.zeros((n, n), device=device, dtype=dtype)

    # Off-diagonal entries
    for i in range(n):
        for j in range(n):
            if i != j:
                sign = 1.0 if (i + j) % 2 == 0 else -1.0
                D[i, j] = (c[i] / c[j]) * sign / (x_k[j] - x_k[i])

    # Diagonal entries (interior points)
    for i in range(1, n - 1):
        D[i, i] = x_k[i] / (2.0 * (1.0 - x_k[i] ** 2))

    # Endpoints
    n_f = (n - 1)
    endpoint_val = (2.0 * n_f * n_f + 1.0) / 6.0
    D[0, 0] = endpoint_val
    D[-1, -1] = -endpoint_val

    D = D * (-1.0)
    return x_k, D






# Weights for applying clenshaw_curtis quadrature rule
def get_clenshaw_curtis_weights(n: int) -> torch.Tensor:
    """
    Returns the square root of Clenshaw-Curtis weights as a 1D tensor for n nodes.
    Nodes are in descending order.
    
    Args:
        n: Number of Chebyshev nodes
    
    Returns:
        torch.Tensor: Shape (n,) containing sqrt(weights) for element-wise multiplication
    """
    if n < 2:
        raise ValueError(f"n must be >= 2, got {n}")
    
    # Number of nodes is n
    N = n

    # Compute weights using the explicit formula with correction factor
    weights = torch.zeros(N, dtype=dtype)
    m = N // 2  # floor(N/2)
    for i in range(N):
        s = 0.0
        for j in range(1, m + 1):
            bj = 0.5 if j == m and N % 2 == 0 else 1.0  # correction factor for last term
            s += bj * (2 * math.cos(2 * j * i * math.pi / N)) / (4 * j**2 - 1)
        weights[i] = (2 / N) * (1 - s)

    # Reverse for descending Chebyshev nodes
    weights = torch.flip(weights, dims=[0])

    # Sqrt of Clenshew Curtis weights 
    weights = torch.sqrt(weights)
    return weights.to(device)


# Weights for cheb nodes interpolation
def barycentric_weights_cheb(x_k):
    """
    Compute barycentric weights for Chebyshev nodes.
    x_k: N nodes in descending order
    Returns: N weights
    """
    N = len(x_k)  # Number of nodes
    w = torch.ones(N, device=x_k.device, dtype=x_k.dtype)
    w[0], w[-1] = 0.5, 0.5  # Endpoints get weight 0.5
    w *= torch.pow(-1, torch.arange(N, device=x_k.device))
    return w


# Barycentric interpolation
def barycentric_interpolate_vectorized(x_k, f_k, w, x_eval, eps=1e-6):
    """
    Barycentric interpolation similar to Rust implementation.
    Assumes:
        - x_k: Chebyshev nodes in DESCENDING order (as generated by cheb_sensors)
        - f_k: function values aligned with x_k
        - w: barycentric weights aligned with x_k
        - x_eval: evaluation points
    """
    # Reshape for broadcasting
    eval_points = x_eval[:, None]  # Shape (M, 1)
    cheb_points = x_k[None, :]     # Shape (1, N)

    # Compute differences
    diff = eval_points - cheb_points  # Shape (M, N)

    # Mask for exact matches
    zero_mask = diff.abs() <= eps
    row_has_match = zero_mask.any(dim=1)  # Shape (M,)

    # Initialize result
    interp = torch.zeros(x_eval.shape[0], dtype=x_eval.dtype, device=x_eval.device)

    # Handle exact matches first: if x_eval == x_k, return f_k directly
    if row_has_match.any():
        # For each evaluation point with an exact match, find the matching node
        exact_eval_indices = torch.where(row_has_match)[0]
        for eval_idx in exact_eval_indices:
            # Find which node matches
            matching_node_idx = torch.where(zero_mask[eval_idx])[0]
            if len(matching_node_idx) > 0:
                interp[eval_idx] = f_k[matching_node_idx[0]]

    # Handle non-exact points
    non_exact_mask = ~row_has_match
    if non_exact_mask.any():
        # Only compute interpolation for non-exact points
        diff_non_exact = diff[non_exact_mask]  # Shape (M_non_exact, N)
        
        # Replace near-zero differences with a small value to avoid division issues
        # Use a value that preserves the sign but avoids division by zero
        safe_diff = diff_non_exact.clone()
        very_small = diff_non_exact.abs() < eps
        safe_diff[very_small] = eps * torch.sign(diff_non_exact[very_small])
        # Avoid zero sign
        safe_diff[safe_diff == 0] = eps

        # Compute reciprocal
        inv_diff = 1.0 / safe_diff

        # Reshape weights and function values
        bary_weights = w[None, :]      # Shape (1, N)
        func_values = f_k[None, :]      # Shape (1, N)

        # Compute numerator and denominator
        numerator = torch.sum(bary_weights * func_values * inv_diff, dim=1)
        denominator = torch.sum(bary_weights * inv_diff, dim=1)

        # Avoid division by zero
        denominator_safe = torch.where(
            denominator.abs() > eps,
            denominator,
            torch.ones_like(denominator) * eps
        )

        interp[non_exact_mask] = numerator / denominator_safe

    return interp


def barycentric_interpolate_eval(x_k, f_k, x_eval):
    w = barycentric_weights_cheb(x_k)
    P_eval = barycentric_interpolate_vectorized(x_k, f_k, w, x_eval)
    return P_eval


def apply_barycentric_interpolate(x_sens, x_eval, u_pred, f_sens, d2u):
    u_eval = barycentric_interpolate_eval(x_sens.view(-1), u_pred.view(-1), x_eval.view(-1))
    f_eval = barycentric_interpolate_eval(x_sens.view(-1), f_sens.view(-1), x_eval.view(-1)) 
    d2u_eval = barycentric_interpolate_eval(x_sens.view(-1), d2u.view(-1), x_eval.view(-1))
    return u_eval, f_eval, d2u_eval



# Branch network: encodes forcing function samples
class BranchNet(nn.Module):
    def __init__(self, m, width=128, depth=3, act=nn.Tanh):
        super().__init__()
        layers = [nn.Linear(m, width), act()]
        for _ in range(depth - 1):
            layers += [nn.Linear(width, width), act()]
        self.net = nn.Sequential(*layers)

    def forward(self, f_samples):
        # f_samples: shape (batch_size, m)
        return self.net(f_samples)  # (batch_size, width)
    

class DeepONet(nn.Module):
    def __init__(self, m, output_dim, width=128, depth=3):
        super().__init__()
        self.branch = BranchNet(m, width, depth)

        # Final layer maps branch features to output_dim
        self.fc_out = nn.Linear(width, output_dim)
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, f_samples, x_points):
        # f_samples: (batch_size, m) where m = N_sensors
        bfeat = self.branch(f_samples)  # (batch_size, width)
        u = self.fc_out(bfeat) + self.bias  # (batch_size, output_dim) where output_dim = N_sensors - 2
        
        # Apply zero-padding: prepend and append zeros for boundary conditions
        # u shape: (batch_size, N_sensors - 2)
        # padded_u shape: (batch_size, N_sensors)
        u_padded = Functional.pad(u, (1, 1), mode='constant', value=0.0)
        return u_padded


# Need to fix f at first - random and then sample from these in batch minimization
# f_sens = torch.randn(1, m_sensors, device=device)

def generate_source_functions_matrix(x_k):
    print(f"Source function generation strategy: {source_function_generation_strategy}")
    if source_function_generation_strategy == 'forcing_function':
        print(f"Forcing function: {forcing_function_type}")
        values = forcing_function(x_k)  # shape: (len(x),)
        return values.unsqueeze(0).repeat(steps, 1)  # repeat along rows
    elif source_function_generation_strategy == 'random_linear':
        # Random parameters a_i and b_i for each function f_i
        a = torch.rand(steps).to(device)  # shape: (n,)
        b = torch.rand(steps).to(device)  # shape: (n,)
        return b.unsqueeze(1) + a.unsqueeze(1) * x_k.unsqueeze(0)  # shape: (n, m)
    else:
        raise ValueError(f"Invalid value: {source_function_generation_strategy}. Expected 'forcing_function' or 'random_linear'.")


def get_loss(u, f):
    # u is already padded with zeros at boundaries (shape: batch_size, N_sensors)
    d2u = (D2 @ u.T).T  # (batch_size, N_sensors)
    
    # Barycentric Intepolation - N cheb Nodes -> M cheb Nodes
    # Squeeze batch dimension for interpolation (batch_size=1)
    u_flat = u.squeeze(0) if u.dim() > 1 and u.shape[0] == 1 else u.view(-1)
    f_flat = f.squeeze(0) if f.dim() > 1 and f.shape[0] == 1 else f.view(-1)
    d2u_flat = d2u.squeeze(0) if d2u.dim() > 1 and d2u.shape[0] == 1 else d2u.view(-1)
    
    u_eval, f_eval, d2u_eval = apply_barycentric_interpolate(x_sens, x_eval, u_flat, f_flat, d2u_flat)

    # Differential operator
    Lu = d2u_eval + k_val**2 * u_eval
    Y_exact = math.sqrt(M_sensors) * Q * Lu
    Y_hat = math.sqrt(M_sensors) * Q * f_eval 

    loss = mse(Y_exact, Y_hat) 
    return loss


def train_step(iteration):
    # Get source function to train
    f_sens = f_N[iteration - 1].unsqueeze(0)  # Add batch dimension: (1, N_sensors)

    # Model inference - u_pred is already padded inside model
    opt.zero_grad()
    u_pred = model(f_sens, x_sens)  # Output shape: (1, N_sensors) after padding
    loss = get_loss(u_pred, f_sens)

    loss.backward() # verify gradients for MSE 
    opt.step()
    return loss.item()


# Plotting functions moved to plots.py
def validate_loss_of_solution():
    if source_function_generation_strategy != 'forcing_function':
        return
    f = forcing_function(x_N)
    u = exact_solution(x_N)
    # u already has correct shape (N_sensors,) matching padded prediction
    loss = get_loss(u.unsqueeze(0), f.unsqueeze(0))
    print("Exact solution loss: {:.3e}".format(loss))


# Sanity check functions moved to sanity_checks.py

if __name__ == '__main__':

    x_N = cheb_sensors(N_sensors).to(device)        # (N_sensors,) Chebyshev sensors
    x_sens = x_N.view(N_sensors, 1).to(device)       # Reshape x_N (N_sensors, 1)
    x_M = cheb_sensors(M_sensors).to(device)         # (M_sensors,) Chebyshev sensors (Clenshew Curtis)
    x_eval = x_M.view(M_sensors, 1).to(device)       # Reshape x_M (M_sensors, 1)
    f_N = generate_source_functions_matrix(x_N)
    
    x, D = chebyshev_diff_matrix(N_sensors)   
    D2 = D @ D 
    Q = get_clenshaw_curtis_weights(M_sensors)

    model = DeepONet(N_sensors, N_sensors - 2, width=width, depth=depth).to(device)
    print(f"Num of model params: {sum(p.numel() for p in model.parameters())}")
    mse = nn.MSELoss()
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    validate_loss_of_solution()

    # Run sanity test
    # sanity_test_loss_decreases(
    #     DeepONet, cheb_sensors, chebyshev_diff_matrix,
    #     get_clenshaw_curtis_weights, apply_barycentric_interpolate,
    #     N_sensors, M_sensors, width, depth, lr, k_val, device, x_eval,
    #     forcing_function_type
    # )

    hist = []
    t0 = time.time()
    for it in range(1, steps+1):
        # print(f"iteration {it}")
        L = train_step(it)
        hist.append(L)
        if it % 200 == 0:
            dt = time.time()-t0
            print(f"iter {it:5d} | loss {L:.3e} | {dt:.1f}s")

    # Plot results based on source function strategy
    plot_results(
        model, x_N, f_N, exact_solution,
        source_function_generation_strategy,
        forcing_function_type,
        N_sensors,
        hist
    )


