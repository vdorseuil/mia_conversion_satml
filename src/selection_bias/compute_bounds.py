"""
Lower bounds of every audit saved in results/experiments/scores/, with every estimator of utils/utils.py (cp_max:
best threshold on the same scores, no correction; cp_union, dkw, holdout, k_fold: corrected), at the confidence
level of the paper and for both families (mu_gdp, eps_delta).

Bounds are divided by group_size (group privacy). Per-step scores of one run (compose_steps = T > 1, Nasr et al.)
give a per-step mu, read as a noise multiplier 1 / mu and composed over the T subsampled steps with the PLD accountant.
One-run audits (family_paper = "one_run") are skipped: their scripts report their own corrected bounds.

Run: python -m selection_bias.compute_bounds
Output (delete to recompute): results/experiments/lower_bounds/bounds.csv, one row per (paper, setting, repetition,
family) with the paper's own bound and one column per estimator.
"""
import sys
from functools import partial

import numpy as np
import pandas as pd

from .utils.paths import BOUNDS_DIR
from .utils.score_formats import load_scores
from .utils.utils import (epsilon_dp_lower, epsilon_dpsgd, estimate_cp_max, estimate_cp_union, estimate_dkw,
                          estimate_holdout, estimate_k_fold, mu_gdp_lower, mu_theory)

GAMMA = 0.05                       # when the paper gives a point estimate
CSV_PATH = BOUNDS_DIR / "bounds.csv"
METHODS = {"cp_max": estimate_cp_max, "cp_union": estimate_cp_union, "dkw": estimate_dkw,
           "holdout": estimate_holdout, "k_fold": estimate_k_fold}


def audit_bounds(audit):
    """Every method for both families on one audit. Returns one row per family."""
    info = audit.iloc[0]
    s_in = audit.loc[audit.membership == 1, "score"].to_numpy()
    s_out = audit.loc[audit.membership == 0, "score"].to_numpy()
    gamma = GAMMA if np.isnan(info.confidence_paper) else 1 - info.confidence_paper
    T = int(info.compose_steps)
    bounds = {"mu_gdp": {name: method(s_in, s_out, gamma, mu_gdp_lower) / info.group_size
                         for name, method in METHODS.items()}}
    if T > 1:
        bounds["eps_delta"] = {name: epsilon_dpsgd(1 / mu, info.sampling_rate, T, info.delta) if mu > 0 else 0.0
                               for name, mu in bounds["mu_gdp"].items()}
        bounds["mu_gdp"] = {name: mu_theory(eps, info.delta) for name, eps in bounds["eps_delta"].items()}
    else:
        lower_function = partial(epsilon_dp_lower, delta=info.delta)
        bounds["eps_delta"] = {name: method(s_in, s_out, gamma, lower_function) / info.group_size
                               for name, method in METHODS.items()}
    true_values = {"eps_delta": info.eps_theory, "mu_gdp": mu_theory(info.eps_theory, info.delta)}
    rows = []
    for family in ["eps_delta", "mu_gdp"]:
        row = {"paper": info.paper, "setting": info.setting, "repetition": info.repetition, "family": family,
               "n_in": len(s_in), "n_out": len(s_out), "gamma": gamma, "delta": info.delta,
               "group_size": info.group_size, "compose_steps": T, "sampling_rate": info.sampling_rate,
               "true_value": true_values[family],
               "bound_paper": info.bound_paper, "family_paper": info.family_paper}
        row.update(bounds[family])
        rows.append(row)
    return rows

def run():
    scores = load_scores()
    rows = []
    for (paper, setting, repetition), audit in scores.groupby(["paper", "setting", "repetition"], sort=True):
        if str(audit.family_paper.iloc[0]).startswith("one_run"):
            continue
        rows += audit_bounds(audit)
        print(f"done {paper} {setting} rep {repetition}", flush=True)
    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(CSV_PATH, index=False)


if __name__ == "__main__":
    if CSV_PATH.exists():
        sys.exit(f"{CSV_PATH} already exists, delete it to recompute")
    run()
