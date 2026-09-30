"""
Steinke et al. (2023), "Privacy Auditing with One (1) Training Run": white-box audit of one DP-SGD run of a
WRN-16-4 on CIFAR-10 with 5000 Dirac canaries, eps in {1, 2, 4, 8}.

Run: python -m selection_bias.papers_experiments.steinke2023 [eps [seed]]
Outputs: results/experiments/scores/steinke2023/wrn16-4_eps<eps>_seed<seed>.parquet (scores of the run)
         results/experiments/lower_bounds/one_run_audits/steinke2023_wrn16-4_eps<eps>_seed<seed>.csv (its bounds)
"""
import math
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from dp_accounting.pld import privacy_loss_distribution as pld_lib
from scipy import stats
from torch.func import functional_call, grad, vmap

from ..utils.datasets import load_cifar10, load_cifar10_test
from ..utils.paths import BOUNDS_DIR
from ..utils.score_formats import audit_rows, save_scores, scores_path

PAPER = "steinke2023"
SETTING = "wrn16-4_eps{eps}"
EPS_LIST = [1.0, 2.0, 4.0, 8.0]
N_SEEDS = 16                                # runs per eps
M = 5000                                    # auditing examples (canaries), each in with probability 1/2
DELTA = 1e-5
P = 0.05                                    # 1 - confidence
K_GRID = np.unique(np.round(np.geomspace(10, M // 2, 15)).astype(int))   # the swept k (k+ = k- = k)
K_FOLDS = 5                                 # folds of the K-fold selection of k
# CIFAR-10 recipe of TAN (Sander et al. 2023, github.com/facebookresearch/tan)
BATCH, T, LR, CLIP = 4096, 2500, 4.0, 1.0   # expected batch size, steps, learning rate (SGD, no momentum), clipping norm
K = 16                                      # augmentation multiplicity
CROP_PADDING = 4                            # TAN: RandomCrop(32, padding=4, padding_mode="reflect"); Mahloujifar et al. write 20
GROUPS = 16                                 # GroupNorm groups
EMA_DECAY = 0.9999
BLOCK = 64                                  # examples (x K augmentations) whose per-example gradients are computed together
MEAN = torch.tensor([0.4914, 0.4822, 0.4465]).view(1, 3, 1, 1)   # CIFAR-10 per-channel mean and std
STD = torch.tensor([0.2470, 0.2435, 0.2616]).view(1, 3, 1, 1)
CSV_DIR = BOUNDS_DIR / "one_run_audits"                  # one file per run, so that parallel jobs never write the same file
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 0
torch.backends.cudnn.benchmark = True


# Data
def normalize(x):
    return (x - MEAN.to(x.device)) / STD.to(x.device)

def augment(x, generator):
    """K augmented copies of every image of x (b, 3, 32, 32): random 32 x 32 crop of the image reflect-padded by
    CROP_PADDING, random horizontal flip, normalized: (b, K, 3, 32, 32)."""
    b, n = len(x), len(x) * K
    padded = F.pad(x, (CROP_PADDING,) * 4, mode="reflect").repeat_interleave(K, dim=0)
    dy = torch.randint(0, 2 * CROP_PADDING + 1, (n,), generator=generator, device=x.device)
    dx = torch.randint(0, 2 * CROP_PADDING + 1, (n,), generator=generator, device=x.device)
    rows = (dy[:, None] + torch.arange(32, device=x.device))[:, :, None]
    cols = (dx[:, None] + torch.arange(32, device=x.device))[:, None, :]
    crops = padded[torch.arange(n, device=x.device)[:, None, None], :, rows, cols].permute(0, 3, 1, 2)
    flip = torch.rand(n, generator=generator, device=x.device) < 0.5
    crops = torch.where(flip[:, None, None, None], crops.flip(-1), crops)
    return normalize(crops).view(b, K, 3, 32, 32)


# Model: TAN's WideResNet(depth=16, num_classes=10, widen_factor=4, nb_groups=16, init=0, order1=0, order2=0)
class Block(nn.Module):
    """Their BasicBlock with order 0: relu, norm, conv, twice; when the shape changes the shortcut is norm and a 1 x 1
    conv of the relu'd input."""
    def __init__(self, c_in, c_out, stride):
        super().__init__()
        self.norm1, self.conv1 = nn.GroupNorm(GROUPS, c_in), nn.Conv2d(c_in, c_out, 3, stride, 1)
        self.norm2, self.conv2 = nn.GroupNorm(GROUPS, c_out), nn.Conv2d(c_out, c_out, 3, 1, 1)
        self.shortcut = None if c_in == c_out else nn.Sequential(nn.GroupNorm(GROUPS, c_in), nn.Conv2d(c_in, c_out, 1, stride, 0))

    def forward(self, x):
        h = F.relu(x)
        skip = x if self.shortcut is None else self.shortcut(h)
        out = self.conv1(self.norm1(h))
        out = self.conv2(self.norm2(F.relu(out)))
        return skip + out

class WideResNet(nn.Module):
    def __init__(self, depth=16, widen=4, num_classes=10):
        super().__init__()
        n, widths = (depth - 4) // 6, [16, 16 * widen, 32 * widen, 64 * widen]
        self.conv1 = nn.Conv2d(3, widths[0], 3, 1, 1)
        blocks = []
        for i, stride in enumerate([1, 2, 2]):
            blocks += [Block(widths[i] if j == 0 else widths[i + 1], widths[i + 1], stride if j == 0 else 1) for j in range(n)]
        self.blocks = nn.Sequential(*blocks)
        self.norm = nn.GroupNorm(GROUPS, widths[3])
        self.fc = nn.Linear(widths[3], num_classes)
        for m in self.modules():                                   # their init 0, "as in Deep Mind's paper"
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
                nn.init.trunc_normal_(m.weight, std=1 / math.sqrt(fan_in))
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.GroupNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        out = self.blocks(self.conv1(x))
        out = self.norm(F.relu(out))                               # their order2 = 0
        return self.fc(F.avg_pool2d(out, 8).flatten(1))

MODEL = WideResNet().to(DEVICE)          # template: every model is a dict of parameters applied through it

def initial_parameters(seed):
    """theta_0 of a run: a fresh init (TAN's init 0) drawn from seed."""
    torch.manual_seed(seed)
    return {k: v.detach().to(DEVICE) for k, v in WideResNet().named_parameters()}

def loss_augmented(params, xs, y):
    """Mean loss of one example over its K augmentations xs (K, 3, 32, 32)."""
    return F.cross_entropy(functional_call(MODEL, params, (xs,)), y.expand(K))

per_example_grads = vmap(grad(loss_augmented), in_dims=(None, 0, 0))     # dict of (b, ...) gradients

def accuracy(params, X, y):
    """Fraction of the images X (in [0, 1]) classified correctly by one model."""
    with torch.no_grad():
        predictions = torch.cat([functional_call(MODEL, params, (normalize(xb),)).argmax(1) for xb in X.split(1000)])
    return float((predictions == y).float().mean())


# DP-SGD
def clipped_gradient_sum(params, X, y, generator):
    """Sum over the examples of their augmentation-averaged gradients clipped to CLIP, BLOCK examples at a time."""
    total = {k: torch.zeros_like(v) for k, v in params.items()}
    for block in torch.arange(len(X), device=X.device).split(BLOCK):
        g = per_example_grads(params, augment(X[block], generator), y[block])
        norms = torch.sqrt(sum(v.flatten(1).square().sum(1) for v in g.values()))
        c = torch.clamp(CLIP / norms, max=1.0)
        for k in total:
            total[k] += (g[k] * c.view(-1, *[1] * (g[k].dim() - 1))).sum(0)
    return total

def poisson_batch(n, q, generator):
    return torch.nonzero(torch.rand(n, generator=generator, device=DEVICE) < q)[:, 0]

def sgd_step(params, noisy_sum):
    """theta <- theta - LR noisy_sum / BATCH (TAN: SGD without momentum on the noisy mean gradient)."""
    for k in params:
        params[k] -= LR * noisy_sum[k] / BATCH

def ema_update(ema, params, t):
    """TAN's update after step t: ema <- ema - (1 - decay) (ema - theta), decay = min(EMA_DECAY, (1 + t) / (10 + t))
    (the warm-up of tf.train.ExponentialMovingAverage)."""
    decay = min(EMA_DECAY, (1 + t) / (10 + t))
    for k in ema:
        ema[k] -= (1 - decay) * (ema[k] - params[k])

def epsilon_dpsgd(sigma, q):
    """eps at DELTA of T steps of the Poisson-subsampled Gaussian mechanism at rate q (PLD accountant)."""
    return pld_lib.from_gaussian_mechanism(standard_deviation=sigma, sampling_prob=q,
                                           value_discretization_interval=1e-3).self_compose(T).get_epsilon_for_delta(DELTA)

def calibrate_sigma(eps, q):
    """Noise multiplier such that T steps at rate q are (eps, DELTA)-DP."""
    low, high = 0.3, 100.0
    for _ in range(40):
        mid = (low + high) / 2
        low, high = (mid, high) if epsilon_dpsgd(mid, q) > eps else (low, mid)
    return high


# The training run with the canaries
def flat(params):
    """The parameters as one vector, in the order of the dict (the order of add_at_coordinates)."""
    return torch.cat([v.flatten() for v in params.values()])

def add_at_coordinates(total, coordinates, value):
    """total[j] += value for every flattened coordinate j in coordinates (distinct): the sampled canaries' gradients."""
    offset = 0
    for v in total.values():
        inside = (coordinates >= offset) & (coordinates < offset + v.numel())
        v.view(-1)[coordinates[inside] - offset] += value
        offset += v.numel()

def audit_run(X, y, X_test, y_test, sigma, q, seed):
    """One DP-SGD run of T steps with the M Dirac canaries; returns their membership bits (1 = in), their white-box
    scores, and the test accuracy of the EMA model and of the last iterate."""
    rng = np.random.default_rng(seed)
    generator = torch.Generator(device=DEVICE).manual_seed(seed)
    params = initial_parameters(seed)
    params0 = {k: v.clone() for k, v in params.items()}
    ema = {k: v.clone() for k, v in params.items()}
    coordinates = torch.tensor(rng.choice(sum(v.numel() for v in params.values()), M, replace=False), device=DEVICE)
    membership = rng.integers(0, 2, M)
    included = torch.tensor(membership == 1, device=DEVICE)
    start = time.time()
    for t in range(1, T + 1):
        idx = poisson_batch(len(X), q, generator)                                    # genuine examples of the batch
        total = clipped_gradient_sum(params, X[idx], y[idx], generator)
        sampled = included & (torch.rand(M, generator=generator, device=DEVICE) < q)  # canaries of the batch
        add_at_coordinates(total, coordinates[sampled], CLIP)                          # their clipped gradients
        noisy_sum = {k: v + sigma * CLIP * torch.randn(v.shape, generator=generator, device=DEVICE) for k, v in total.items()}
        sgd_step(params, noisy_sum)
        ema_update(ema, params, t)
        if t % 250 == 0:
            print(f"  step {t}: {time.time() - start:.0f} s, EMA test accuracy {accuracy(ema, X_test, y_test):.4f}", flush=True)
    scores = CLIP * (flat(params0)[coordinates] - flat(params)[coordinates])          # sum_t <w^{t-1} - w^t, g_i>
    return membership, scores.cpu().numpy().astype(float), accuracy(ema, X_test, y_test), accuracy(params, X_test, y_test)


# Steinke et al., Appendix D (verbatim port)
def p_value_dp_audit(m, r, v, eps, delta):
    """Probability of >= v correct guesses out of r under (eps, delta)-DP, m canaries."""
    q = 1 / (1 + math.exp(-eps))
    beta = stats.binom.sf(v - 1, r, q)
    alpha, total = 0.0, 0.0
    for i in range(1, v + 1):
        total += stats.binom.pmf(v - i, r, q)
        alpha = max(alpha, total / i)
    return min(beta + alpha * delta * 2 * m, 1.0)

def get_eps_audit(m, r, v, delta, p):
    """Largest eps whose p-value is below p (30 bisection steps)."""
    eps_min, eps_max = 0.0, 1.0
    while p_value_dp_audit(m, r, v, eps_max, delta) < p:
        eps_max += 1
    for _ in range(30):
        eps = (eps_min + eps_max) / 2
        if p_value_dp_audit(m, r, v, eps, delta) < p:
            eps_min = eps
        else:
            eps_max = eps
    return eps_min

def bound(scores, membership, k, level):
    """The paper's lower bound at level for k+ = k- = k guesses on these canaries (m = M canaries were randomized)."""
    return get_eps_audit(M, 2 * k, correct_guesses(scores, membership, k), DELTA, level)


# The sweep over k and its corrections
def correct_guesses(scores, membership, k):
    """v: correct guesses when guessing "in" on the k largest scores and "out" on the k smallest."""
    order = np.argsort(scores)
    return int(np.sum(membership[order[-k:]] == 1) + np.sum(membership[order[:k]] == 0))

def sweep(scores, membership, level):
    """Best bound over K_GRID (the k that fit in the scores) at level, and the k that gives it."""
    grid = K_GRID[K_GRID <= len(scores) // 2]
    values = [bound(scores, membership, k, level) for k in grid]
    best = int(np.argmax(values))
    return values[best], int(grid[best])

def k_fold(scores, membership, rng):
    """K_FOLDS-fold version of the hold-out, split reversed: the canaries are split at random into K_FOLDS folds; the k
    chosen by the sweep on one fold, scaled to the size of the other folds, is applied to those other folds at level
    P / K_FOLDS (m stays M); the best fold is reported (union bound over the folds). Evaluating on (K-1)/K of the
    canaries instead of 1/K keeps the union-bound cost on the large set, which is what makes it beat the hold-out."""
    fold = rng.permutation(len(scores)) % K_FOLDS
    best = 0.0
    for f in range(K_FOLDS):
        selection = fold == f
        _, k = sweep(scores[selection], membership[selection], P)
        k_eval = int(round(k * (~selection).sum() / selection.sum()))
        best = max(best, bound(scores[~selection], membership[~selection], k_eval, P / K_FOLDS))
    return best

def audit(scores, membership, rng):
    """The paper's bound (max over the sweep at level P) and its k, the union-corrected sweep (level P / |K_GRID|),
    the hold-out bound (k chosen by the sweep on a random half of the canaries, bound at level P on the other half)
    and the K-fold bound (the hold-out split is drawn from rng first, so it did not change when K-fold was added)."""
    best, k_best = sweep(scores, membership, P)
    union, _ = sweep(scores, membership, P / len(K_GRID))
    half = rng.permutation(len(scores)) < len(scores) // 2
    _, k_half = sweep(scores[half], membership[half], P)
    holdout = bound(scores[~half], membership[~half], k_half, P)
    return {"max": best, "k_max": k_best, "union": union, "holdout": holdout, "k_fold": k_fold(scores, membership, rng)}


def run(eps_values, seeds):
    X, y = load_cifar10()
    X_test, y_test = load_cifar10_test()
    X, y, X_test, y_test = X.to(DEVICE), y.to(DEVICE), X_test.to(DEVICE), y_test.to(DEVICE)
    q = BATCH / len(X)
    for eps in eps_values:
        setting = SETTING.format(eps=eps)
        todo = [s for s in seeds if not scores_path(PAPER, f"{setting}_seed{s}").exists()]
        if not todo:
            continue
        sigma = calibrate_sigma(eps, q)
        for s in todo:
            start = time.time()
            seed = SEED + 1000 * s + int(10 * eps)
            print(f"{setting} seed {s}: sigma = {sigma:.3f}, training", flush=True)
            membership, scores, acc_ema, acc_last = audit_run(X, y, X_test, y_test, sigma, q, seed)
            bounds = audit(scores, membership, np.random.default_rng(seed))
            row = {"paper": PAPER, "setting": setting, "eps": eps, "seed": s, "sigma": sigma, "q": q, "T": T,
                   "n_in": int(membership.sum()), "test_accuracy": acc_ema, "test_accuracy_last": acc_last, **bounds}
            CSV_DIR.mkdir(parents=True, exist_ok=True)
            pd.DataFrame([row]).to_csv(CSV_DIR / f"{PAPER}_{setting}_seed{s}.csv", index=False)
            save_scores([audit_rows(PAPER, setting, s, membership, scores, eps_theory=eps, delta=DELTA,
                                    bound_paper=bounds["max"], threshold_paper=np.nan, confidence_paper=1 - P,
                                    family_paper="one_run", run_id=np.zeros(M, dtype=int), unit_id=np.arange(M))],
                        PAPER, f"{setting}_seed{s}")   # written last: a run whose score file exists is skipped
            print(f"{setting} seed {s}: test accuracy {acc_ema:.4f} (EMA), {acc_last:.4f} (last), "
                  f"{PAPER} {bounds['max']:.3f} (k = {bounds['k_max']}), union {bounds['union']:.3f}, "
                  f"hold-out {bounds['holdout']:.3f}, K-fold {bounds['k_fold']:.3f}, {time.time() - start:.0f} s", flush=True)


if __name__ == "__main__":
    run([float(sys.argv[1])] if len(sys.argv) > 1 else EPS_LIST,
        [int(sys.argv[2])] if len(sys.argv) > 2 else list(range(N_SEEDS)))
