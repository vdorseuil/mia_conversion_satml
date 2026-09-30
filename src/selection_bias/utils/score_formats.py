"""
Format of the score files written by the paper scripts and read by compute_bounds.py.

One parquet file per (paper, setting) in results/experiments/scores/<paper>/<setting>.parquet, one row per observation:
    paper, setting, repetition   the audit the observation belongs to (repetition: independent repeat of the whole audit)
    run_id, unit_id              the training run it comes from, and its index inside that run (0 when one score per run)
    membership                   1 if the canary / target was present in that run, 0 otherwise
    score                        the attack score, oriented so that a LARGER score means "present"
    eps_theory, delta            the guarantee the paper compares to (delta = 0 for pure DP)
    group_size                   copies of the canary in the dataset (Jagielski et al.); bounds are divided by it
    compose_steps, sampling_rate 1 and 1, or the number of steps T and the Poisson sampling rate q when the observations
                                 are per-step scores of one run (Nasr et al.): the per-step bound is then composed
    bound_paper, threshold_paper, confidence_paper, family_paper
                                 the lower bound the paper's own method gives on these scores, with its threshold
                                 (NaN if none), confidence level (NaN for a point estimate) and family
                                 ("eps_delta", "eps_pure", "mu_gdp", "one_run")
"""
import numpy as np
import pandas as pd

from .paths import SCORES_DIR

COLUMNS = ["paper", "setting", "repetition", "run_id", "unit_id", "membership", "score",
           "eps_theory", "delta", "group_size", "compose_steps", "sampling_rate",
           "bound_paper", "threshold_paper", "confidence_paper", "family_paper"]


def scores_path(paper, setting):
    return SCORES_DIR / paper / f"{setting}.parquet"


def audit_rows(paper, setting, repetition, membership, score, eps_theory, delta, bound_paper, threshold_paper,
               confidence_paper, family_paper, group_size=1, compose_steps=1, sampling_rate=1.0, run_id=None,
               unit_id=None):
    """The rows of one audit, one per score. run_id defaults to the position of the score (one score per run)."""
    n = len(score)
    return pd.DataFrame({
        "paper": paper, "setting": setting, "repetition": repetition,
        "run_id": np.arange(n) if run_id is None else np.asarray(run_id),
        "unit_id": np.zeros(n, dtype=int) if unit_id is None else np.asarray(unit_id),
        "membership": np.asarray(membership, dtype=int),
        "score": np.asarray(score, dtype=float),
        "eps_theory": float(eps_theory), "delta": float(delta), "group_size": int(group_size),
        "compose_steps": int(compose_steps), "sampling_rate": float(sampling_rate),
        "bound_paper": float(bound_paper), "threshold_paper": float(threshold_paper),
        "confidence_paper": float(confidence_paper), "family_paper": family_paper,
    })


def save_scores(rows, paper, setting):
    """Write the rows of one setting (a list of DataFrames from audit_rows, one per repetition)."""
    path = scores_path(paper, setting)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.concat(rows, ignore_index=True)[COLUMNS].to_parquet(path, index=False)


def load_scores():
    """Every score file, concatenated."""
    files = sorted(SCORES_DIR.glob("*/*.parquet"))
    if not files:
        raise FileNotFoundError(f"no score file in {SCORES_DIR}: run the scripts of papers_experiments first")
    frames = [pd.read_parquet(f) for f in files]
    for frame in frames:
        for column, default in [("compose_steps", 1), ("sampling_rate", 1.0)]:   # files written before the column existed
            if column not in frame:
                frame[column] = default
    return pd.concat(frames, ignore_index=True)
