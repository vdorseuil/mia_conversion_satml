"""
Empirical coverage of privacy lower bounds under threshold selection, on a Gaussian mechanism whose true mu is known.

Run: python -m selection_bias.theoretical_simulation
Outputs (delete to recompute)
    results/theoretical/Theoretical_coverage_bounds.csv   every lower bound of every audit (to plot their distribution)
    results/theoretical/Theoretical_coverage.csv          coverage, mean and std of the bound per (n, mu, gamma, family, method)
"""


import sys
from functools import partial
from multiprocessing import Pool

import numpy as np
import pandas as pd

from .utils.paths import THEORY_DIR
from .utils.utils import estimate_cp_fixed, estimate_cp_max, estimate_cp_union, estimate_dkw, estimate_holdout, estimate_k_fold, convert_mu_epsilon, mu_gdp_lower, epsilon_dp_lower

# Configuration
N_LIST = [100, 1000, 10000] # samples per hypothesis (n_in = n_out = n)
MU_LIST = [0.5, 1, 2, 4]
GAMMAS = [0.05, 0.01] # CI = 1-gamma  
DELTA = 1e-5
N_TRIALS = 10000
K_FOLDS = 5
SEED = 0

CSV_PATH = THEORY_DIR / "Theoretical_coverage.csv"
BOUNDS_PATH = THEORY_DIR / "Theoretical_coverage_distribution.csv"

METHODS = ["cp_max", "cp_fixed", "cp_union", "dkw", "holdout", "k_fold"]  
LOWER_FUNCTIONS = {"mu_gdp": mu_gdp_lower, "eps_delta": partial(epsilon_dp_lower, delta=DELTA)}


def generate_distrib(n_in, n_out, mu_in, mu_out, rng):
    x_in = rng.standard_normal(n_in) + mu_in
    x_out = rng.standard_normal(n_out) + mu_out
    return x_in, x_out


def score(x):
    return -x


def true_value(family, mu):
    return mu if family == "mu_gdp" else convert_mu_epsilon(mu, DELTA)


# Simulation
def run_cell(cell):
    """Every method on every audit for one (n, mu).
    Returns one row per (trial, gamma, family) with the lower bound of each method in its column."""
    n, mu, seed = cell
    rng = np.random.default_rng(seed)
    methods = {
        "cp_fixed": partial(estimate_cp_fixed, threshold=-mu / 2),   # midpoint between the two score means
        "cp_max": estimate_cp_max,
        "cp_union": estimate_cp_union,
        "dkw": estimate_dkw,
        "holdout": estimate_holdout,
        "k_fold": partial(estimate_k_fold, k_folds=K_FOLDS),
    }

    rows = []
    for trial in range(N_TRIALS):
        x_in, x_out = generate_distrib(n, n, 0, mu, rng)
        s_in, s_out = score(x_in), score(x_out)
        for gamma in GAMMAS:
            for family, lower_function in LOWER_FUNCTIONS.items():
                row = {"n": n, "mu": mu, "gamma": gamma, "family": family, "trial": trial}
                for method in METHODS:
                    row[method] = methods[method](s_in, s_out, gamma, lower_function)
                rows.append(row)
    print(f"done n={n} mu={mu}", flush=True)
    return rows

def save_bounds(bounds):
    """Every estimated mu / epsilon of every audit, to plot their distribution.
    One row per (n, mu, gamma, family, trial), one column per method; the methods of a row saw the same data."""
    BOUNDS_PATH.parent.mkdir(parents=True, exist_ok=True)
    bounds.to_csv(BOUNDS_PATH, index=False)

def summarize(bounds):
    """Coverage, mean and std of the bound over the audits, one row per (n, mu, gamma, family, method)."""
    rows = []
    for (n, mu, gamma, family), cell in bounds.groupby(["n", "mu", "gamma", "family"]):
        true = true_value(family, mu)
        for method in METHODS:
            values = cell[method].to_numpy()
            rows.append({
                "n": n, "mu": mu, "gamma": gamma, "family": family, "method": method,
                "true_value": true,
                "coverage": np.mean(values <= true),
                "mean_bound": np.mean(values),
                "std_bound": np.std(values),
                "n_trials": len(values),
            })
    return pd.DataFrame(rows).sort_values(["gamma", "family", "mu", "n"])

def run_experiment():
    cells = [(n, mu) for n in N_LIST for mu in MU_LIST]
    cells = [(n, mu, SEED + i) for i, (n, mu) in enumerate(cells)] 
    cells = sorted(cells, reverse=True)
    with Pool() as pool:
        results = pool.map(run_cell, cells, chunksize=1)
    bounds = pd.DataFrame([row for rows in results for row in rows]).sort_values(["gamma", "family", "mu", "n", "trial"])
    save_bounds(bounds)
    summarize(bounds).to_csv(CSV_PATH, index=False)


if __name__ == "__main__":
    if CSV_PATH.exists():
        sys.exit(f"{CSV_PATH} already exists, delete it to recompute")
    run_experiment()
