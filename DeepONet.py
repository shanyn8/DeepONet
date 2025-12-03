# Environment & imports
import math, time, numpy as np
import torch, torch.nn as nn, torch.nn.functional as Functional
import matplotlib.pyplot as plt
import unittest

device = ("cuda" if torch.cuda.is_available()
          else "mps" if getattr(torch.backends, 'mps', None) and torch.backends.mps.is_available()
          else "cpu")
dtype = torch.float32 if device == "mps" else torch.float64

# print('Using device:', device)
torch.set_default_dtype(torch.float32)
_ = torch.manual_seed(1234)


    
# === Config ===
N_sensors   = 10      # number of Chebyshev sensors for branch input f
M_sensors   = 20     # number of Chebyshev sensors as quadrature points for Clenshaw Curtis Algorithm
width       = 128     # feature width (branch/trunk)
depth       = 3       # hidden layers in branch/trunk
B_batch     = 16      # number of functions per batch (operator learning)
N_bc        = 2       # number of BC points per boundary per function
steps       = 600    # training steps
lr          = 1e-3    # learning rate
# Distribution for k (avoid resonance k≈1 and sin(k)≈0 for the test analytic comparison)
k_val = 5.0

# Random forcing generator: Chebyshev series with decaying coefficients
Nf_terms = 12         # number of Chebyshev terms for random f
decay    = 0.8        # geometric decay factor for coeff magnitudes (0<decay<1)


# Branching of different strategies of the code
forcing_function = True
forced_boundary_conditions = True
include_trunk_net = False


def forcing_function(x):
    # return torch.sin(torch.pi * x)
    return -25*x**3 - 25*x**2 + 19*x + 23

# The exact solution is u(x) = -*x**3 - *x**2 + *x + 1 satisfy BC
def exact_solution(x):
    return -1*x**3 - 1*x**2 + 1*x + 1

def cheb_sensors(m):
    j = torch.arange(m+1, dtype=dtype, device=device)
    xi = torch.cos(j*math.pi/(m))
    return xi.to(torch.float32)


def chebyshev_diff_matrix(N: int):
    x = torch.cos(torch.pi * torch.arange(N + 1, device=device) / N)
    c = torch.ones(N + 1, device=device)
    c[0] = 2
    c[-1] = 2
    c = c * ((-1) ** torch.arange(N + 1, device=device))
    X = x.repeat(N + 1, 1)
    dX = X.T - X
    D = (c.unsqueeze(1) / c.unsqueeze(0)) / (dX + torch.eye(N + 1, device=device))
    D = D - torch.diag(torch.sum(D, dim=1))
    return x, D


# Weights for applying clenshaw_curtis quadrature rule
def get_clenshaw_curtis_weights_mat(n: int) -> torch.Tensor:
    """
    Returns a diagonal matrix of Clenshaw-Curtis weights for n intervals (n+1 nodes).
    Nodes are in descending order.
    """
    # Number of nodes
    N = n + 1

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

    # Return diagonal matrix
    # return torch.diag(weights).to(device)
    return weights.to(device)


# Weights for cheb nodes interpolation
def barycentric_weights_cheb(x_k):
    N = len(x_k) - 1
    w = torch.ones(N + 1, device=x_k.device)
    w[0], w[-1] = 0.5, 0.5
    w *= torch.pow(-1, torch.arange(N + 1, device=x_k.device))
    return w


# Barycentric interpolation
# Accelerate barycentric interpolation (using efficient Tretephen method from the book)
def barycentric_interpolate(x_k, f_k, w, x_eval):
    P = torch.zeros_like(x_eval)
    for i, x in enumerate(x_eval):
        diff = x - x_k
        if torch.any(diff == 0):
            P[i] = f_k[torch.where(diff == 0)[0][0]]
        else:
            terms = w / diff
            P[i] = torch.sum(terms * f_k) / torch.sum(terms)
    return P



def barycentric_interpolate_vectorized(x_k, f_k, w, x_eval, eps=1e-12):
    diff = x_eval[:, None] - x_k[None, :]
    mask = diff.abs() < eps  # near-zero check
    diff = diff + eps  # avoid division by zero
    terms = w / diff
    num = torch.sum(terms * f_k, dim=1)
    den = torch.sum(terms, dim=1)
    P = num / den
    if mask.any():
        idx_eval, idx_k = torch.where(mask)
        P[idx_eval] = f_k[idx_k]
    return P


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


class TrunkNet(nn.Module):
    def __init__(self, width=128, depth=3, act=nn.Tanh):
        super().__init__()
        layers = [nn.Linear(1, width), act()]
        for _ in range(depth - 1):
            layers += [nn.Linear(width, width), act()]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        # x: shape (N, 1)
        return self.net(x)
    

class DeepONet(nn.Module):
    def __init__(self, m, output_dim, width=128, depth=3):
        super().__init__()
        self.branch = BranchNet(m, width, depth)
        self.trunk = TrunkNet(width, depth)

        # Final layer maps branch features to output_dim
        self.fc_out = nn.Linear(width, output_dim)
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, f_samples, x_points):
        # f_samples: (batch_size, m)
        bfeat = self.branch(f_samples)  # (batch_size, width)
        if include_trunk_net:
            tfeat = self.trunk(x_points)          # (N, W)
            # Broadcast branch features to match trunk features
            u = (bfeat * tfeat).sum(dim=1, keepdim=True) + self.bias
        else:
            u = self.fc_out(bfeat) + self.bias  # (batch_size, output_dim)
        return u


# Need to fix f at first - random and then sample from these in batch minimization
# f_sens = torch.randn(1, m_sensors, device=device)

def generate_source_functions_matrix(x_k):
    if forcing_function:
        values = forcing_function(x_k)  # shape: (len(x),)
        forcing_function_mat_values = values.unsqueeze(0).repeat(steps, 1)  # repeat along rows
        return forcing_function_mat_values
    # Random parameters a_i and b_i for each function f_i
    a = torch.rand(steps).to(device)  # shape: (n,)
    b = torch.rand(steps).to(device)  # shape: (n,)

    # Compute f_i(x) = b_i + a_i * x for each xi
    f_matrix = b.unsqueeze(1) + a.unsqueeze(1) * x_k.unsqueeze(0)  # shape: (n, m)
    return f_matrix

## polynomial.chebyshev.chebval ?

# Verify chebyshev points - DESC
# Batch minimization

def train_step(iteration):
    ################################################################################
    ##                             Chebyshev Nodes                                ##
    ################################################################################

    f_sens = f_N[iteration - 1]
    x_sens = x_N.view(N_sensors + 1, 1).to(device)  # Reshape x_N (N_sensors, 1)
    

    # Model inference
    opt.zero_grad()
    u_pred = model(f_sens, x_sens)
    # print("u_pred range:", u_pred.min().item(), u_pred.max().item())
   

    
    ################################################################################
    ##      Evaluation Points - M Cheb Nodes with Clenshew Curtis Weights         ##
    ##      Convert Cheb Nodes to Eval points by Barycentric Interpolation        ##
    ################################################################################

    x_eval = x_M.view(M_sensors + 1, 1).to(device)  # Reshape x_M (M_sensors, 1)
    d2u = D2 @ u_pred
    # print("d2u range:", d2u.min().item(), d2u.max().item())

    # Barycentric Intepolation - N cheb Nodes -> M cheb Nodes

    u_eval, f_eval, d2u_eval = apply_barycentric_interpolate(x_sens, x_eval, u_pred, f_sens, d2u)
    # u_eval = barycentric_interpolate_eval(x_sens.view(-1), u_pred.view(-1), x_eval.view(-1))
    # f_eval = barycentric_interpolate_eval(x_sens.view(-1), f_sens.view(-1), x_eval.view(-1)) 
    # d2u_eval = barycentric_interpolate_eval(x_sens.view(-1), d2u.view(-1), x_eval.view(-1))

    if forced_boundary_conditions:
        u_eval[1] = 0
        u_eval[-1] = 0


    # # Differential operator
    Lu = d2u_eval + k_val**2 * u_eval
    Y_exact = math.sqrt(M_sensors + 1) * Q * Lu
    Y_hat = math.sqrt(M_sensors + 1) * Q * f_eval 

    if forced_boundary_conditions:
        loss = mse(Y_exact, Y_hat) 
    else:
        loss_pde = mse(Y_exact, Y_hat)
        loss_bc = mse(u_eval[0], torch.tensor([0.0], device=device)) + mse(u_eval[-1], torch.tensor([0.0], device=device))
        loss = loss_pde + loss_bc


    # residual = Lu - f_eval
    # loss_pde = mse(residual, torch.zeros_like(residual))
    # loss_bc = mse(u_eval[0], torch.tensor([0.0], device=device)) + mse(u_eval[-1], torch.tensor([0.0], device=device))
    # loss = loss_pde

    # d2u = D2 @ u_pred
    # print("d2u range:", d2u.min().item(), d2u.max().item())
    # residual = d2u + k_val**2 * u_pred - f_sens
    # loss_pde = mse(residual, torch.zeros_like(residual))
    # loss_bc = mse(u_pred[0], torch.tensor([0.0], device=device)) + mse(u_pred[-1], torch.tensor([0.0], device=device))
    # loss = loss_pde + loss_bc
 

    # print(f"loss: {loss}")
    # print(f"loss_pde: {loss_pde}")
    # opt.zero_grad()
    loss.backward() # verify gradients for MSE 
    opt.step()
    return loss.item()



if __name__ == '__main__':

    x_N = cheb_sensors(N_sensors).to(device)        # (N,) Chebyshev sensors
    x_M = cheb_sensors(M_sensors).to(device)        # (M,) Chebyshev sensors (Clenshew Curtis)
    f_N = generate_source_functions_matrix(x_N)
    
    model = DeepONet(N_sensors + 1, N_sensors + 1, width=width, depth=depth).to(device)
    print(f"Num of model params: {sum(p.numel() for p in model.parameters())}")

    mse = nn.MSELoss()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    _, D = chebyshev_diff_matrix(N_sensors)   
    D2 = D @ D 
    Q = get_clenshaw_curtis_weights_mat(M_sensors)
    # Q = torch.eye(M_sensors, device=device)  # identity matrix


    hist = []
    t0 = time.time()
    for it in range(1, steps+1):
        # print(f"iteration {it}")
        L = train_step(it)
        hist.append(L)
        if it % 200 == 0:
            dt = time.time()-t0
            print(f"iter {it:5d} | loss {L:.3e} | {dt:.1f}s")

    # print('hist')
    # print(hist)
    # plt.figure(figsize=(5,3))
    # plt.semilogy([h[0] for h in hist], label='total')
    # # plt.semilogy([h[1] for h in hist], label='pde')
    # # plt.semilogy([h[2] for h in hist], label='bc')
    # plt.legend(); plt.title('Training loss'); plt.tight_layout(); plt.show()

    model.eval()
    with torch.no_grad():
        x_plot = x_N.view(N_sensors + 1, 1)
        f_plot = f_N[0]
        u_pred_plot = model(f_plot, x_plot)
        u_exact_plot = exact_solution(x_plot)

    plt.figure(figsize=(8, 5))
    plt.plot(x_plot.cpu().numpy(), u_pred_plot.cpu().numpy(), label='Predicted u(x)', linewidth=2)
    plt.plot(x_plot.cpu().numpy(), u_exact_plot.cpu().numpy(), label='Exact u(x)', linestyle='--')
    plt.title('DeepONet vs Exact Solution')
    plt.xlabel('x')
    plt.ylabel('u(x)')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
    # add test with sampled u -> deduce f and verify zero loss
    # Add test that f are all linear combinations of 2 functions - linar y=x and const y=c
