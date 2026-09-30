"""
Jagielski et al. (2020), "Auditing Differentially Private Machine Learning: How Private is Private SGD?":
ClipBKD poisoning on FMNIST (their Table 2), eps in {1, 2, 4, 8}, k in {1, 2, 4, 8} poison points.

Run: python -m selection_bias.papers_experiments.jagielski2020
Output: results/experiments/scores/jagielski2020/fmnist_eps<eps>_k<k>.parquet
"""
import numpy as np
import torch
from scipy import stats

from ..utils.datasets import clipbkd_file
from ..utils.score_formats import audit_rows, save_scores, scores_path

PAPER = "jagielski2020"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Configuration (their Table 1 for FMNIST; noise multipliers of their exp_run.py)
NOISE_MULTIPLIER = {1: 5.02, 2: 2.68, 4: 1.55, 8: 1.01}            # eps_th -> sigma_GD
EPS_LIST = list(NOISE_MULTIPLIER)
K_LIST = [1, 2, 4, 8]
N_REPETITIONS = 16
N_RUNS = 1000                                      # models per set (their trials); the first N_SEARCH pick the threshold
N_SEARCH = 500                                     # their search_ct
BATCH_SIZE, EPOCHS, LR = 250, 24, 0.15
CLIP_NORM = 1.0
HIDDEN = 32
CONFIDENCE = 0.99
CHUNK = 250                                        # runs trained together
SEED = 0


# Data
def load_dataset(k, poisoned):
    """(X, Y one-hot) of D1(k) if poisoned else of D0, and the poison point (x_p, y_p)."""
    (x0, y0), (x1, y1), (x_p, y_p), _ = np.load(clipbkd_file(k), allow_pickle=True)
    x, y = (x1, y1) if poisoned else (x0, y0)
    X = torch.tensor(x.reshape(len(x), -1), dtype=torch.float32, device=DEVICE)
    Y = torch.nn.functional.one_hot(torch.tensor(y, dtype=torch.long), 2).float().to(DEVICE)
    return X, Y, torch.tensor(np.asarray(x_p).reshape(-1), dtype=torch.float32, device=DEVICE), int(y_p)


# Model: parameters carry a leading "run" dimension so that many runs train at once
def glorot(fan_in, fan_out, generator):
    return torch.randn(fan_in, fan_out, generator=generator, device=generator.device) * np.sqrt(2 / (fan_in + fan_out))

def init_params(n_runs, d_in=784, n_classes=2):
    """The fixed theta_0 (Glorot-normal weights and zero biases, drawn once from SEED), copied for every run."""
    g = torch.Generator(device=DEVICE).manual_seed(SEED)
    params = [glorot(d_in, HIDDEN, g), torch.zeros(HIDDEN, device=DEVICE),
              glorot(HIDDEN, n_classes, g), torch.zeros(n_classes, device=DEVICE)]
    return [p.expand(n_runs, *p.shape).clone() for p in params]

def logits(params, x):
    """x: (runs, batch, d) -> (runs, batch, classes)."""
    W1, b1, W2, b2 = params
    return torch.relu(x @ W1 + b1[:, None, :]) @ W2 + b2[:, None, :]

def clipped_gradient_sum(params, x, y):
    """Sum over the batch of the per-example gradients clipped to CLIP_NORM, without forming them: the gradient of
    example i w.r.t. a weight matrix is the outer product a_i d_i^T (a_i = input of the layer, d_i = derivative of the
    loss w.r.t. its output), of norm |a_i| |d_i|; with the bias, the layer's block has norm sqrt(|a_i|^2 + 1) |d_i|.
    The clipped sum is then a^T (c d) with c_i = min(1, C / norm_i)."""
    W1, b1, W2, b2 = params
    pre = x @ W1 + b1[:, None, :]
    h = torch.relu(pre)
    d2 = torch.softmax(h @ W2 + b2[:, None, :], dim=2) - y                  # (runs, batch, classes)
    d1 = (d2 @ W2.transpose(1, 2)) * (pre > 0)                              # (runs, batch, HIDDEN)
    sq_norm = (h.square().sum(2) + 1) * d2.square().sum(2) + (x.square().sum(2) + 1) * d1.square().sum(2)
    c = torch.clamp(CLIP_NORM / sq_norm.sqrt(), max=1.0)[:, :, None]
    return [x.transpose(1, 2) @ (c * d1), (c * d1).sum(1), h.transpose(1, 2) @ (c * d2), (c * d2).sum(1)]

def train(X, Y, n_runs, sigma, generator):
    """DP-SGD of n_runs independent runs at once from the fixed theta_0: each run reshuffles the data at every epoch
    and takes its batches."""
    params = init_params(n_runs, X.shape[1], Y.shape[1])
    n = len(X)
    for epoch in range(EPOCHS):
        order = torch.argsort(torch.rand(n_runs, n, generator=generator, device=generator.device), dim=1)
        for j in range(n // BATCH_SIZE):
            idx = order[:, j * BATCH_SIZE:(j + 1) * BATCH_SIZE]
            grads = clipped_gradient_sum(params, X[idx], Y[idx])
            for p, g in zip(params, grads):
                noise = sigma * CLIP_NORM * torch.randn(g.shape, generator=generator, device=g.device)
                p -= LR * (g + noise) / BATCH_SIZE
    return params

def train_and_score(X, Y, x_p, y_p, sigma, seed):
    """N_RUNS models trained on (X, Y), CHUNK at a time; returns the score (logit of y_p at x_p) of each model."""
    generator = torch.Generator(device=DEVICE).manual_seed(seed)
    scores = []
    for start in range(0, N_RUNS, CHUNK):
        params = train(X, Y, min(CHUNK, N_RUNS - start), sigma, generator)
        scores.append(logits(params, x_p.expand(len(params[0]), 1, -1))[:, 0, y_p])
    return torch.cat(scores).cpu().numpy().astype(float)


# The paper's estimator (their bkd_parser.py)
def clopper_pearson(count, n, conf=1 - CONFIDENCE):
    """Two-sided interval, conf / 2 in each tail (their clopper_pearson)."""
    low = stats.beta.ppf(conf / 2, count, n - count + 1) if count > 0 else 0.0
    high = stats.beta.isf(conf / 2, count + 1, n - count) if count < n else 1.0
    return low, high

def eps_from_bounds(p1_lb, p0_ub, k):
    """log of the probability ratio of the event {score >= t} or of its complement, whichever is larger, over k."""
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(p0_ub + p1_lb > 1, (1 - p0_ub) / (1 - p1_lb), p1_lb / p0_ub)
        return np.log(ratio) / k

def find_threshold(nobkd, bkd):
    """Their bkd_find_thresh(use_dkw=True): the pooled score maximising the ratio computed with DKW-corrected rates."""
    thresholds = np.unique(np.concatenate([nobkd, bkd]))
    p0_ub = (nobkd[None, :] >= thresholds[:, None]).mean(1) + np.sqrt(np.log(2 / 0.05) / len(nobkd))
    p1_lb = (bkd[None, :] >= thresholds[:, None]).mean(1) - np.sqrt(np.log(2 / 0.05) / len(bkd))
    eps = np.nan_to_num(eps_from_bounds(p1_lb, p0_ub, 1), nan=-np.inf)
    return thresholds[np.argmax(eps)]

def paper_bound(nobkd, bkd, k):
    """Their bkd_parser.py: the threshold is chosen on the first N_SEARCH scores of each set and the 99% bound
    (bkd_get_eps(use_dkw=False)) is computed on the remaining ones."""
    threshold = find_threshold(nobkd[:N_SEARCH], bkd[:N_SEARCH])
    nobkd, bkd = nobkd[N_SEARCH:], bkd[N_SEARCH:]
    _, p0_ub = clopper_pearson(np.sum(nobkd >= threshold), len(nobkd))
    p1_lb, _ = clopper_pearson(np.sum(bkd >= threshold), len(bkd))
    return float(eps_from_bounds(p1_lb, p0_ub, k)), float(threshold)


# Experiment
def run():
    X0, Y0, x_p, y_p = load_dataset(1, poisoned=False)                     # D0: the k = 1 file's unpoisoned set
    poisoned = {k: load_dataset(k, poisoned=True) for k in K_LIST}
    for i, eps in enumerate(EPS_LIST):
        settings = {k: f"fmnist_eps{eps}_k{k}" for k in K_LIST}
        if all(scores_path(PAPER, s).exists() for s in settings.values()):
            continue
        sigma = NOISE_MULTIPLIER[eps]
        rows = {k: [] for k in K_LIST}
        for rep in range(N_REPETITIONS):
            seed = SEED + 1000 * i + 10 * rep                               # the unpoisoned models, then seed + 1.. per k
            nobkd = train_and_score(X0, Y0, x_p, y_p, sigma, seed)          # shared by the four k (their bkd_parser.py)
            bounds = {}
            for j, k in enumerate(K_LIST):
                X1, Y1, x_p1, y_p1 = poisoned[k]
                bkd = train_and_score(X1, Y1, x_p1, y_p1, sigma, seed + j + 1)
                bounds[k], threshold = paper_bound(nobkd, bkd, k)
                rows[k].append(audit_rows(PAPER, settings[k], rep, np.repeat([0, 1], N_RUNS), np.concatenate([nobkd, bkd]),
                                          eps_theory=eps, delta=0.0, bound_paper=bounds[k], threshold_paper=threshold,
                                          confidence_paper=CONFIDENCE, family_paper="eps_pure", group_size=k))
            best = max(bounds, key=bounds.get)
            print(f"eps {eps} rep {rep}: sigma = {sigma:.3f}, eps_LB = "
                  + ", ".join(f"{b:.3f} (k = {k})" for k, b in bounds.items()) + f"; best k = {best}", flush=True)
        for k in K_LIST:
            save_scores(rows[k], PAPER, settings[k])


if __name__ == "__main__":
    run()
