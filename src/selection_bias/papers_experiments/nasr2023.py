"""
Nasr et al. (2023), "Tight Auditing of Differentially Private Machine Learning": white-box audit of DP-SGD on
CIFAR-10 with a Dirac gradient canary (their Algorithm 2), eps in {1, 4, 8, 16}.

Run: python -m selection_bias.papers_experiments.nasr2023 [eps ...]
Output: results/experiments/scores/nasr2023/whitebox_eps<eps>.parquet
"""
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from kymatio.scattering2d.frontend.torch_frontend import ScatteringTorch2D   # kymatio.torch does not import with scipy >= 1.17
from scipy import stats
from torch.func import functional_call, grad, vmap

from ..utils.datasets import load_cifar10
from ..utils.score_formats import audit_rows, save_scores, scores_path
from ..utils.utils import calibrate_sigma, cp_upper, epsilon_dpsgd

PAPER = "nasr2023"
EPS_LIST = [1.0, 4.0, 8.0, 16.0]
BATCH = 4096                   # expected batch size (Poisson sampling at rate BATCH / n)
T = 2500
LR, MOMENTUM, CLIP = 4.0, 0.9, 0.1
N_REPETITIONS = 16
DELTA = 1e-5
CONFIDENCE = 0.95
BLOCK = 1024                   # examples whose per-example gradients are computed together
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 0


# Data: ScatterNet features
def load_features():
    """The 50000 training images as scattering coefficients (n, 243, 8, 8) standardized per channel, and their labels."""
    images, y = load_cifar10()
    scattering = ScatteringTorch2D(J=2, shape=(32, 32)).to(DEVICE)       # fixed wavelet filters, no parameter
    with torch.no_grad():
        S = torch.cat([scattering(xb.to(DEVICE)).flatten(1, 2).cpu() for xb in images.split(500)])
    mean, std = S.mean((0, 2, 3), keepdim=True), S.std((0, 2, 3), keepdim=True)
    return S.sub_(mean).div_(std), y


# Model
def make_cnn():
    """The ScatterNet CNN of Tramèr & Boneh on the (243, 8, 8) features: 187,210 parameters."""
    return nn.Sequential(nn.Conv2d(243, 64, 3, padding=1), nn.Tanh(), nn.MaxPool2d(2),
                         nn.Conv2d(64, 64, 3, padding=1), nn.Tanh(), nn.Flatten(), nn.Linear(64 * 4 * 4, 10))

MODEL = make_cnn().to(DEVICE)          # template: every model is a dict of parameters applied through it

def initial_parameters():
    """theta_0, the same for every run."""
    torch.manual_seed(SEED)
    return {k: v.detach().to(DEVICE) for k, v in make_cnn().named_parameters()}

def loss_one(params, x, y):
    return F.cross_entropy(functional_call(MODEL, params, (x[None],)), y[None])

per_example_grads = vmap(grad(loss_one), in_dims=(None, 0, 0))    # dict of (batch, ...) gradients


# DP-SGD
def clipped_gradient_sum(params, X, y):
    """Sum over the examples of their gradients clipped to CLIP, BLOCK examples at a time."""
    total = {k: torch.zeros_like(v) for k, v in params.items()}
    for block in torch.arange(len(X), device=X.device).split(BLOCK):
        g = per_example_grads(params, X[block], y[block])
        norms = torch.sqrt(sum(v.flatten(1).square().sum(1) for v in g.values()))
        c = torch.clamp(CLIP / norms, max=1.0)
        for k in total:
            total[k] += (g[k] * c.view(-1, *[1] * (g[k].dim() - 1))).sum(0)
    return total

def poisson_batch(n, generator):
    return torch.nonzero(torch.rand(n, generator=generator, device=generator.device) < BATCH / n)[:, 0]

def sgd_step(params, velocity, noisy_sum):
    """SGD with momentum on the noisy mean gradient: v <- MOMENTUM v + noisy_sum / BATCH, theta <- theta - LR v."""
    for k in params:
        velocity[k] = MOMENTUM * velocity[k] + noisy_sum[k] / BATCH
        params[k] -= LR * velocity[k]


# The audited run
def coordinate_value(total, j):
    """Entry j of the flattened gradient sum."""
    return torch.cat([v.flatten() for v in total.values()])[j]

def add_at_coordinate(total, j, value):
    """total + value at flattened coordinate j (the canary gradient, or the observation's Dirac)."""
    for v in total.values():
        if j < v.numel():
            v.view(-1)[j] += value
            return
        j -= v.numel()

def audit_run(X, y, sigma, generator):
    """One training run; returns its T observations <canary gradient, noisy sum> / CLIP^2 without (O) and with (O')
    the canary in the sum."""
    params = initial_parameters()
    velocity = {k: torch.zeros_like(v) for k, v in params.items()}
    n_params = sum(v.numel() for v in params.values())
    obs_out, obs_in = np.empty(T), np.empty(T)
    for t in range(T):
        j = int(torch.randint(n_params, (1,), generator=generator, device=DEVICE))
        idx = poisson_batch(len(X), generator)
        sum_b = clipped_gradient_sum(params, X[idx], y[idx])
        idx_prime = poisson_batch(len(X), generator)
        sum_b_prime = clipped_gradient_sum(params, X[idx_prime], y[idx_prime])
        add_at_coordinate(sum_b_prime, j, CLIP)
        noise_b = {k: sigma * CLIP * torch.randn(v.shape, generator=generator, device=v.device) for k, v in sum_b.items()}
        noise_b_prime = {k: sigma * CLIP * torch.randn(v.shape, generator=generator, device=v.device) for k, v in sum_b.items()}
        obs_out[t] = float(coordinate_value(sum_b, j) + coordinate_value(noise_b, j)) / CLIP
        obs_in[t] = float(coordinate_value(sum_b_prime, j) + coordinate_value(noise_b_prime, j)) / CLIP
        sgd_step(params, velocity, {k: sum_b[k] + noise_b[k] for k in params})   # their Algorithm 2: the batch without the canary
    return obs_out, obs_in


# The paper's estimator
def compose_steps(mu_step, q):
    """eps at DELTA of T steps of DP-SGD at rate q whose noise multiplier is 1 / mu_step (0 when mu_step <= 0)."""
    return epsilon_dpsgd(1 / mu_step, q, T, DELTA) if mu_step > 0 else 0.0

def paper_bound(scores, membership, q):
    """Per-step mu at the best of every distinct threshold, composed over the T subsampled steps (their Section 5.5)."""
    thresholds = np.unique(scores)
    s_in, s_out = np.sort(scores[membership == 1]), np.sort(scores[membership == 0])
    fp = len(s_out) - np.searchsorted(s_out, thresholds)
    fn = np.searchsorted(s_in, thresholds)
    level = (1 - CONFIDENCE) / 2
    mu = stats.norm.ppf(1 - cp_upper(fp, len(s_out), level)) - stats.norm.ppf(cp_upper(fn, len(s_in), level))
    best = int(np.argmax(mu))
    return compose_steps(mu[best], q), float(thresholds[best])


def run(eps_values=EPS_LIST):
    X, y = load_features()
    X, y = X.to(DEVICE), y.to(DEVICE)
    q = BATCH / len(X)
    for eps in eps_values:
        setting = f"whitebox_eps{eps}"
        if scores_path(PAPER, setting).exists():
            continue
        sigma = calibrate_sigma(eps, q, T, DELTA)
        rows = []
        for rep in range(N_REPETITIONS):
            generator = torch.Generator(device=DEVICE).manual_seed(SEED + 100 * int(eps) + rep)
            obs_out, obs_in = audit_run(X, y, sigma, generator)
            scores, membership = np.concatenate([obs_out, obs_in]), np.repeat([0, 1], T)
            bound, threshold = paper_bound(scores, membership, q)
            rows.append(audit_rows(PAPER, setting, rep, membership, scores, eps_theory=eps, delta=DELTA,
                                   bound_paper=bound, threshold_paper=threshold, confidence_paper=CONFIDENCE,
                                   family_paper="eps_delta", compose_steps=T, sampling_rate=q,
                                   run_id=np.zeros(2 * T, dtype=int), unit_id=np.tile(np.arange(T), 2)))
            print(f"{setting} rep {rep}: sigma = {sigma:.3f}, eps_emp = {bound:.3f}", flush=True)
        save_scores(rows, PAPER, setting)


if __name__ == "__main__":
    run([float(a) for a in sys.argv[1:]] or EPS_LIST)
