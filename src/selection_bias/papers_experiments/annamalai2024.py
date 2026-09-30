"""
Annamalai & De Cristofaro (2024), "Nearly Tight Black-Box Auditing of DP-SGD": last-layer fine-tuning on CIFAR-10
features (their scripts/model_init_finetune.sh), average and worst-case initialization, eps in {1, 2, 4, 10}.

Run: python -m selection_bias.papers_experiments.annamalai2024
Output: results/experiments/scores/annamalai2024_blackbox/finetune_<init>_eps<eps>.parquet
"""
import numpy as np
import torch
from scipy import optimize, stats

from ..utils.datasets import annamalai_files
from ..utils.score_formats import audit_rows, save_scores, scores_path
from ..utils.utils import cp_upper

PAPER = "annamalai2024_blackbox"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Configuration
EPS_LIST = [1.0, 2.0, 4.0, 10.0]
INITS = ["average", "worst"]
N_RUNS = 500                      # per world (their --n_reps 1000)
N_SEEDS = 16
EPOCHS, LEARNING_RATE, CLIP_NORM = 20, 4e-4, 1.0
DELTA = 1e-5
CONFIDENCE = 0.95                 # their --alpha 0.05
DIM, N_CLASSES = 640, 10
CHUNK = 100                       # runs trained together
SEED = 0


# Data and initialization
def load_data():
    folder = annamalai_files() / "data" / "cifar10_half_finetune_last"
    X = torch.tensor(np.load(folder / "X_train.npy"), device=DEVICE)
    y = torch.tensor(np.load(folder / "y_train.npy"), device=DEVICE)
    target_x = torch.tensor(np.load(annamalai_files() / "cifar10_half_finetune_last_clipbkd.npy"), device=DEVICE)[0]
    return X, y, target_x

def initial_parameters(init, generator):
    """(W (640, 10), b (10,)) shared by every run of the audit."""
    if init == "worst":
        state = torch.load(annamalai_files() / "cifar10_half_finetune_last.pt", map_location=DEVICE)
        return state["linear.weight"].T.clone(), state["linear.bias"].clone()
    W = torch.randn(DIM, N_CLASSES, generator=generator, device=DEVICE) * np.sqrt(2 / (DIM + N_CLASSES))   # Xavier normal
    return W, torch.full((N_CLASSES,), 0.01, device=DEVICE)


# Accounting: 20 Gaussian mechanisms with noise multiplier sigma are sqrt(20) / sigma-GDP
def delta_eps_mu(eps, mu):
    """delta(eps) of mu-GDP (the second term is computed in log space to avoid overflow)."""
    return stats.norm.cdf(-eps / mu + mu / 2) - np.exp(eps + stats.norm.logcdf(-eps / mu - mu / 2))

def eps_from_mu(mu):
    if mu <= 0 or delta_eps_mu(0, mu) <= DELTA:
        return 0.0
    # eps at delta grows like mu^2 / 2, so the bracket must grow with mu (500 only covers mu < 27.5)
    return optimize.brentq(lambda eps: delta_eps_mu(eps, mu) - DELTA, 0, max(500, mu * (mu / 2 + 10)))

def calibrate_sigma(eps):
    mu = optimize.brentq(lambda mu: eps_from_mu(mu) - eps, 1e-3, 20)
    return np.sqrt(EPOCHS) / mu


# Full-batch DP-SGD, vectorised over the runs
def train(W0, b0, X, Y, n_runs, sigma, generator):
    """The per-example gradient of example i is [x_i, 1] d_i^T with d_i = softmax(logits_i) - y_i, of norm
    sqrt(|x_i|^2 + 1) |d_i|, so the sum of the clipped gradients is X^T (c d) with c_i = min(1, C / norm_i)."""
    W = W0.expand(n_runs, DIM, N_CLASSES).clone()
    b = b0.expand(n_runs, N_CLASSES).clone()
    x_sq_norm = X.square().sum(1)
    for _ in range(EPOCHS):
        d = torch.softmax(X @ W + b[:, None, :], dim=2) - Y                  # (runs, n, 10)
        c = torch.clamp(CLIP_NORM / ((x_sq_norm + 1) * d.square().sum(2)).sqrt(), max=1.0)[:, :, None]
        W -= LEARNING_RATE * (X.T @ (c * d) + sigma * CLIP_NORM * torch.randn(W.shape, generator=generator, device=DEVICE))
        b -= LEARNING_RATE * ((c * d).sum(1) + sigma * CLIP_NORM * torch.randn(b.shape, generator=generator, device=DEVICE))
    return W, b

def train_world(W0, b0, X, Y, target_x, target_y, sigma, generator):
    """N_RUNS runs on (X, Y), CHUNK at a time; returns minus the loss of each final model on the target."""
    scores = []
    for start in range(0, N_RUNS, CHUNK):
        W, b = train(W0, b0, X, Y, min(CHUNK, N_RUNS - start), sigma, generator)
        logits = target_x @ W + b
        scores.append(logits[:, target_y] - torch.logsumexp(logits, dim=1))
    return torch.cat(scores).cpu().numpy().astype(float)


# The paper's estimator (their compute_eps_lower_from_mia with method 'GDP')
def paper_bound(scores, membership):
    """Every distinct score as threshold; the largest mu gives the largest eps (increasing conversion)."""
    thresholds = np.unique(scores)
    s_in, s_out = np.sort(scores[membership == 1]), np.sort(scores[membership == 0])
    fp = len(s_out) - np.searchsorted(s_out, thresholds)     # OUT runs with score >= t
    fn = np.searchsorted(s_in, thresholds)                    # IN runs with score < t
    level = (1 - CONFIDENCE) / 2
    mu = stats.norm.ppf(1 - cp_upper(fp, len(s_out), level)) - stats.norm.ppf(cp_upper(fn, len(s_in), level))
    best = int(np.argmax(mu))
    return eps_from_mu(mu[best]), float(thresholds[best])


# Experiment
def run():
    X, y, target_x = load_data()
    Y = torch.nn.functional.one_hot(y, N_CLASSES).float()
    for init in INITS:
        for eps in EPS_LIST:
            setting = f"finetune_{init}_eps{eps}"
            if scores_path(PAPER, setting).exists():
                continue
            sigma = calibrate_sigma(eps)
            rows = []
            for seed in range(N_SEEDS):
                generator = torch.Generator(device=DEVICE).manual_seed(SEED + seed)
                W0, b0 = initial_parameters(init, generator)
                target_y = int(torch.argmin(target_x @ W0 + b0))
                X_in = torch.cat([X, target_x[None]])
                Y_in = torch.cat([Y, torch.nn.functional.one_hot(torch.tensor([target_y], device=DEVICE), N_CLASSES).float()])
                s_out = train_world(W0, b0, X, Y, target_x, target_y, sigma, generator)
                s_in = train_world(W0, b0, X_in, Y_in, target_x, target_y, sigma, generator)
                scores, membership = np.concatenate([s_out, s_in]), np.repeat([0, 1], N_RUNS)
                bound, threshold = paper_bound(scores, membership)
                rows.append(audit_rows(PAPER, setting, seed, membership, scores, eps_theory=eps, delta=DELTA,
                                       bound_paper=bound, threshold_paper=threshold, confidence_paper=CONFIDENCE,
                                       family_paper="eps_delta"))
                print(f"{setting} seed {seed}: sigma = {sigma:.3f}, eps_emp = {bound:.3f}", flush=True)
            save_scores(rows, PAPER, setting)


if __name__ == "__main__":
    run()
