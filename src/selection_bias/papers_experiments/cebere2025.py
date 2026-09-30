"""
Cebere et al. (2025), "Tighter Privacy Auditing of DP-SGD in the Hidden State Threat Model": gradient-crafting
adversary on the Housing dataset (their Figure 3c), eps in {1, 4, 8, 12, 24}.

Run: python -m selection_bias.papers_experiments.cebere2025
Output: results/experiments/scores/cebere2025/<adversary>_k<k>_eps<eps>.parquet
"""
import numpy as np
import torch
from scipy import optimize, stats
from sklearn.datasets import fetch_california_housing

from ..utils.paths import DATA_DIR
from ..utils.score_formats import audit_rows, save_scores, scores_path
from ..utils.utils import cp_upper

PAPER = "cebere2025"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Configuration
ADVERSARIES = ["GC-R"]             # "GC-S" and "L" can be added
K_LIST = [1]                       # insertion periodicity
EPS_LIST = [1, 4, 8, 12, 24]       # theoretical eps at DELTA, one noise multiplier each
N_RUNS = 5000
N_REPETITIONS = 16
INSERTIONS = 250                   # T = INSERTIONS * k steps
BATCH_SIZE, LEARNING_RATE, CLIP_NORM = 400, 0.1, 1.0
DELTA = 1e-5
CONFIDENCE = 0.95
HIDDEN = 6
SHAPES = [(8, HIDDEN), (HIDDEN,), (HIDDEN, 2), (2,)]   # W1, b1, W2, b2
N_PARAMS = 68
SEED = 0


# Data and model
def load_housing():
    data = fetch_california_housing(data_home=DATA_DIR / "cebere2025")
    X = (data.data - data.data.mean(0)) / data.data.std(0)
    y = data.target > np.median(data.target)
    return torch.tensor(X, dtype=torch.float32, device=DEVICE), torch.tensor(y, dtype=torch.long, device=DEVICE)

def initial_parameters(generator):
    """theta_0, known to the adversary: PyTorch's default Linear initialization (uniform in +-1 / sqrt(fan_in))."""
    parts = []
    for fan_in, fan_out in [(8, HIDDEN), (HIDDEN, 2)]:
        bound = 1 / np.sqrt(fan_in)
        parts += [torch.rand(fan_in, fan_out, generator=generator, device=generator.device) * 2 * bound - bound,
                  torch.rand(fan_out, generator=generator, device=generator.device) * 2 * bound - bound]
    return torch.cat([p.flatten() for p in parts])

def unflatten(theta):
    """theta (runs, 68) -> W1 (runs, 8, 6), b1 (runs, 6), W2 (runs, 6, 2), b2 (runs, 2)."""
    parts, start = [], 0
    for shape in SHAPES:
        size = int(np.prod(shape))
        parts.append(theta[:, start:start + size].reshape(len(theta), *shape))
        start += size
    return parts

def forward(theta, x):
    """x (runs, batch, 8) -> pre-activations (runs, batch, 6), hidden units, logits (runs, batch, 2)."""
    W1, b1, W2, b2 = unflatten(theta)
    pre = x @ W1 + b1[:, None, :]
    h = torch.relu(pre)
    return pre, h, h @ W2 + b2[:, None, :]

def losses(theta, x, y):
    """Cross-entropy of every example: (runs, 68), (runs, batch, 8), (runs, batch) -> (runs, batch)."""
    logits = forward(theta, x)[2]
    return torch.logsumexp(logits, dim=2) - logits.gather(2, y[:, :, None])[:, :, 0]

def per_example_grads(theta, x, y):
    """Gradient of the cross-entropy of every example, (runs, batch, 68), in the order of SHAPES. Two layers, so by
    hand: d2 = softmax - onehot is the derivative w.r.t. the logits, d1 = (d2 W2^T) 1[pre > 0] w.r.t. the
    pre-activations, and the gradient of a layer's weights is the outer product of its input with these."""
    pre, h, logits = forward(theta, x)
    W2 = unflatten(theta)[2]
    d2 = torch.softmax(logits, dim=2) - torch.nn.functional.one_hot(y, 2)
    d1 = (d2 @ W2.transpose(1, 2)) * (pre > 0)
    return torch.cat([(x[:, :, :, None] * d1[:, :, None, :]).flatten(2), d1,
                      (h[:, :, :, None] * d2[:, :, None, :]).flatten(2), d2], dim=2)


# DP-SGD, vectorised over the runs
def clip(g, C):
    return g * torch.clamp(C / g.norm(dim=-1, keepdim=True), max=1.0)

def batches(n_runs, n, n_steps, generator):
    """Mini-batch indices (n_runs, B) of every step: each run reshuffles the data at every epoch, then takes slices."""
    per_epoch = n // BATCH_SIZE
    for t in range(n_steps):
        if t % per_epoch == 0:
            order = torch.argsort(torch.rand(n_runs, n, generator=generator, device=generator.device), dim=1)
        j = t % per_epoch
        yield order[:, j * BATCH_SIZE:(j + 1) * BATCH_SIZE]

def train(theta0, X, y, bits, k, sigma, canary, generator):
    """DP-SGD with noise multiplier sigma from theta0 for INSERTIONS * k steps in every run at once. canary(theta)
    gives the (R, 68) vector added to the gradient sum of the runs with bit 1 at the steps t = k, 2k, ... Returns
    theta_T of each run."""
    R = len(bits)
    theta = theta0.expand(R, N_PARAMS).clone()
    for t, idx in enumerate(batches(R, len(X), INSERTIONS * k, generator), start=1):
        total = clip(per_example_grads(theta, X[idx], y[idx]), CLIP_NORM).sum(1)
        total += sigma * CLIP_NORM * torch.randn(R, N_PARAMS, generator=generator, device=generator.device)
        if t % k == 0:
            total += bits[:, None] * canary(theta)
        theta -= LEARNING_RATE / BATCH_SIZE * total
    return theta

def least_updated_coordinate(theta0, X, y, k, generator):
    """GC-S (Appendix A, noiseless simulation, per-step ranking): one run of plain SGD (no noise, no clipping) from
    theta0 with the same schedule; d = argmin of the accumulated squared per-step updates."""
    theta, updates = theta0.clone()[None], torch.zeros(N_PARAMS, device=theta0.device)
    for idx in batches(1, len(X), INSERTIONS * k, generator):
        step = LEARNING_RATE / BATCH_SIZE * per_example_grads(theta, X[idx], y[idx]).sum(1)
        updates += step[0].square()
        theta -= step
    return int(torch.argmin(updates))


# The paper's estimator (Algorithm 4)
def eps_from_mu(mu):
    """eps such that mu-GDP implies (eps, DELTA)-DP (Corollary 1 of the paper); 0 when mu gives no evidence.
    The second term of delta(eps) is computed in log space to avoid overflow."""
    def delta_gap(eps):
        return stats.norm.cdf(-eps / mu + mu / 2) - np.exp(eps + stats.norm.logcdf(-eps / mu - mu / 2)) - DELTA
    if mu <= 0 or delta_gap(0) <= 0:
        return 0.0
    return optimize.brentq(delta_gap, 0, max(500, mu * (mu / 2 + 10)))   # 500 only covers mu < 27.5

def sigma_for_eps(eps):
    """Noise multiplier for which the INSERTIONS Gaussian mechanisms are (eps, DELTA)-DP: mu = sqrt(INSERTIONS) / sigma."""
    mu = optimize.brentq(lambda mu: eps_from_mu(mu) - eps, 1e-3, 20)
    return np.sqrt(INSERTIONS) / mu

def paper_bound(scores, membership):
    """Midpoint thresholds, 95% CP upper bounds on FPR and FNR, mu at each threshold; the largest mu gives the
    largest eps (the conversion is increasing), which is reported with its threshold."""
    s = np.sort(scores)
    thresholds = (s[:-1] + s[1:]) / 2
    s_in, s_out = np.sort(scores[membership == 1]), np.sort(scores[membership == 0])
    fp = len(s_out) - np.searchsorted(s_out, thresholds)     # OUT runs with score >= t
    fn = np.searchsorted(s_in, thresholds)                    # IN runs with score < t
    level = (1 - CONFIDENCE) / 2
    mu = stats.norm.ppf(1 - cp_upper(fp, len(s_out), level)) - stats.norm.ppf(cp_upper(fn, len(s_in), level))
    best = int(np.argmax(mu))
    return eps_from_mu(mu[best]), float(thresholds[best])


# Experiment
def run():
    X, y = load_housing()
    for adversary in ADVERSARIES:
        for k in K_LIST:
            for eps in EPS_LIST:
                setting = f"{adversary}_k{k}_eps{eps}"
                if scores_path(PAPER, setting).exists():
                    continue
                sigma = sigma_for_eps(eps)
                rows = []
                for rep in range(N_REPETITIONS):
                    generator = torch.Generator(device=DEVICE).manual_seed(SEED + 1000 * rep + round(10 * eps))
                    theta0 = initial_parameters(generator)
                    bits = torch.randint(0, 2, (N_RUNS,), generator=generator, device=DEVICE)    # b ~ Bernoulli(1/2)
                    if adversary == "L":
                        i = int(torch.randint(len(X), (1,), generator=generator, device=DEVICE))
                        x_star, y_star = X[i], 1 - y[i]                              # a training point, label flipped
                        x_one, y_one = x_star.expand(N_RUNS, 1, 8), y_star.expand(N_RUNS, 1)
                        canary = lambda theta: clip(per_example_grads(theta, x_one, y_one)[:, 0], CLIP_NORM)
                    else:
                        d = (int(torch.randint(N_PARAMS, (1,), generator=generator, device=DEVICE)) if adversary == "GC-R"
                             else least_updated_coordinate(theta0, X, y, k, generator))
                        crafted = torch.zeros(N_PARAMS, device=DEVICE)
                        crafted[d] = CLIP_NORM
                        canary = lambda theta: crafted.expand(len(theta), N_PARAMS)
                    theta_T = train(theta0, X, y, bits, k, sigma, canary, generator)
                    if adversary == "L":
                        scores = -losses(theta_T, x_one, y_one)[:, 0].cpu().numpy()
                    else:
                        scores = (theta0[d] - theta_T[:, d]).cpu().numpy()
                    membership = bits.cpu().numpy()
                    bound, threshold = paper_bound(scores, membership)
                    rows.append(audit_rows(PAPER, setting, rep, membership, scores, eps_theory=eps, delta=DELTA,
                                           bound_paper=bound, threshold_paper=threshold, confidence_paper=CONFIDENCE,
                                           family_paper="eps_delta"))
                    print(f"{setting} rep {rep}: sigma = {sigma:.3f}, eps_emp = {bound:.3f}", flush=True)
                save_scores(rows, PAPER, setting)


if __name__ == "__main__":
    run()
