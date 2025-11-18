# Environment & imports
import math, time, numpy as np
import torch, torch.nn as nn, torch.nn.functional as Functional
import matplotlib.pyplot as plt
import unittest

device = ("cuda" if torch.cuda.is_available()
          else "mps" if getattr(torch.backends, 'mps', None) and torch.backends.mps.is_available()
          else "cpu")
# print('Using device:', device)
torch.set_default_dtype(torch.float32)
_ = torch.manual_seed(1234)





    
# === Config ===
m_sensors   = 64      # number of Chebyshev sensors for branch input f
width       = 128     # feature width (branch/trunk)
depth       = 3       # hidden layers in branch/trunk
B_batch     = 16      # number of functions per batch (operator learning)
N_col       = 256     # collocation points per function
N_bc        = 2       # number of BC points per boundary per function
steps       = 20    # training steps
lr          = 1e-3    # learning rate

# Distribution for k (avoid resonance k≈1 and sin(k)≈0 for the test analytic comparison)
k_val = 18.0

# Random forcing generator: Chebyshev series with decaying coefficients
Nf_terms = 12         # number of Chebyshev terms for random f
decay    = 0.8        # geometric decay factor for coeff magnitudes (0<decay<1)



def cheb_sensors(m):
    dtype = torch.float32 if device == "mps" else torch.float64
    j = torch.arange(m, dtype=dtype, device=device)
    # j = torch.arange(m, dtype=torch.float64, device=device)
    
    # xi = torch.cos((2*j+1)*math.pi/(2*m))
    xi = torch.cos(j*math.pi/(m))
    return xi.to(torch.float32)


def chebyshev_diff_matrix(N: int):
    x = torch.cos(torch.pi * torch.arange(N, device=device) / N)
    c = torch.ones(N, device=device)
    c[0] = 2
    c[-1] = 2
    c = c * ((-1) ** torch.arange(N, device=device))
    X = x.repeat(N, 1)
    dX = X - X.T
    D = (c.unsqueeze(1) / c.unsqueeze(0)) / (dX + torch.eye(N, device=device))
    D = D - torch.diag(torch.sum(D, dim=1))
    return x, D


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


def barycentric_interpolate_vectorized(x_k, f_k, w, x_eval):
    diff = x_eval[:, None] - x_k[None, :]  # shape (M, N+1)
    mask = diff == 0
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


# Branch network: encodes forcing function samples
class BranchNet(nn.Module):
    def __init__(self, m, width=128, depth=3, act=nn.Tanh):
        super().__init__()
        layers = [nn.Linear(m, width), act()]
        for _ in range(depth - 1):
            layers += [nn.Linear(width, width), act()]
        self.net = nn.Sequential(*layers)

    def forward(self, f_samples):
        # f_samples: shape (m,) or (1, m)
        return self.net(f_samples)

# Trunk network: encodes spatial coordinate x
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

# DeepONet: combines branch and trunk features
class DeepONet(nn.Module):
    def __init__(self, m, width=128, depth=3):
        super().__init__()
        self.branch = BranchNet(m, width, depth)
        self.trunk = TrunkNet(width, depth)
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, f_samples, x_points):
        # f_samples: (1, m), x_points: (N, 1)
        bfeat = self.branch(f_samples)        # (1, W)
        tfeat = self.trunk(x_points)          # (N, W)
        # Broadcast branch features to match trunk features
        u = (bfeat * tfeat).sum(dim=1, keepdim=True) + self.bias
        return u  # shape (N, 1)

# Need to fix f at first - random and then sample from these in batch minimization
# f_sens = torch.randn(1, m_sensors, device=device)


def generate_source_functions_matrix(x_k):
    # Random parameters a_i and b_i for each function f_i
    a = torch.rand(steps)  # shape: (n,)
    b = torch.rand(steps)  # shape: (n,)

    # Compute f_i(x) = b_i + a_i * x for each xi
    f_matrix = b.unsqueeze(1) + a.unsqueeze(1) * x_k.unsqueeze(0)  # shape: (n, m)
    return f_matrix


# Verify chebyshev points ASC/DESC
# Batch minimization

def train_step(iteration):
    ################################################################################
    ##                             Chebyshev Nodes                                ##
    ################################################################################

    # Computes f(x1) ... f(xn) which are just random points as the values of f (what should be the range?)
    f_sens = generate_source_functions_matrix[iteration]
    x_sens = x_k.view(m_sensors, 1).to(device)  # Reshape x_k (m_sensors, 1)
    
    # train on (m_sensors - 2) internal nodes
    internal_cheb_sensors = x_sens[1:, -1]
    internal_f_sens = f_sens[1:, -1]
    # Model inference
    u_pred = model(internal_f_sens, internal_cheb_sensors)
    # Apply BC - zero paddings to first and last to get u(-1)=u(1)=0
    u_pred = Functional.pad(u_pred, (1, 1), mode='constant', value=0)

    
    ################################################################################
    ##                  Evaluation Points equispaced on [-1, 1]                   ##
    ##      Convert Cheb Nodes to Eval points by Barycentric Interpolation        ##
    ################################################################################

    # Clenshaw for quadrature rules

    # ----------
    x_eval  = -1 + 2*torch.rand(N_col, 1, device=device)
    x_eval = x_eval.view(-1)

    # I perform barycentric interpolation twice - once for u and once for f - makes sense?
    u_eval = barycentric_interpolate_eval(x_sens.view(-1), u_pred.view(-1), x_eval)
    f_eval = barycentric_interpolate_eval(x_sens.view(-1), f_sens.view(-1), x_eval)

    D2 = D @ D
    d2u = D2 @ u_pred
        
    d2u_eval = barycentric_interpolate_eval(x_sens.view(-1), d2u.view(-1), x_eval)

    # Add test that f are all linear combinations of 2 functions - linar y=x and const y=c
    # Minimize Loss by PDE Residual
    # ----------
    residual = d2u_eval + k_val**2 * u_eval - f_eval 
    loss = mse(residual, torch.zeros_like(residual))  
    print(f"loss: {loss}")
    opt.zero_grad()
    loss.backward()
    opt.step()
    return loss.item()



if __name__ == '__main__':

    x_k = cheb_sensors(m_sensors).to(device)        # (m,) Chebyshev sensors
    model = DeepONet(m_sensors, width=width, depth=depth).to(device)
    print(f"Num of model params: {sum(p.numel() for p in model.parameters())}")

    mse = nn.MSELoss()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    _, D = chebyshev_diff_matrix(m_sensors - 1)    

    hist = []
    t0 = time.time()
    for it in range(1, steps+1):
        print(f"iteration {it}")
        L = train_step(it)
        hist.append(L)
        if it % 200 == 0:
            dt = time.time()-t0
            print(f"iter {it:5d} | loss {L:.3e} | {dt:.1f}s")

    print('hist')
    print(hist)
    # plt.figure(figsize=(5,3))
    # plt.semilogy([h[0] for h in hist], label='total')
    # # plt.semilogy([h[1] for h in hist], label='pde')
    # # plt.semilogy([h[2] for h in hist], label='bc')
    # plt.legend(); plt.title('Training loss'); plt.tight_layout(); plt.show()
