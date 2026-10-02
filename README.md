# Reliable Estimation of Differential Privacy under Selection Bias

Code for the paper, anonymized for review.

The repository has three parts: a simulation that measures the coverage of several estimators on a mechanism whose true parameter is known, the re-run of six published audits, which saves their attack scores, and the re-estimation of their lower ounds on these scores with corrected estimators.

## Setup
```bash
uv sync  # or: pip install -e .
```
Tested with Python 3.13 on Linux. The datasets (about 500 MB) are downloaded to `data/` the first time a script
needs them: CIFAR-10, the Fashion-MNIST files released with the code of Jagielski et al., the CIFAR-10 features and
worst-case initialization released with the code of Annamalai & De Cristofaro, and California Housing through
scikit-learn.

Run the commands below from the root of the repository, with `uv run` or after activating `.venv`. Every script
writes under `results/`. A setting whose output file exists is skipped; delete the file to recompute it. Privacy
levels, numbers of runs and repetitions are the constants at the top of each script.

## 1. Coverage of the estimators on a Gaussian mechanism
```bash
    python -m selection_bias.theoretical_simulation  # ->  results/theoretical/
```
This script simulates the Gaussian mechanism and computes the coverage of different estimators. In the paper we reported the miscoverage (1-coverage) of each estimator. The estimators are implemented in
`src/selection_bias/utils/utils.py` and differ in how the threshold is chosen:

| Name | Threshold | Valid |
|---|---|---|
| `cp_fixed` | fixed before seeing the scores (midpoint of the two means; simulation only) | yes |
| `cp_max` | best threshold on the same scores, no correction | no |
| `cp_union` | best threshold on the same scores, union bound over the candidate thresholds | yes |
| `dkw` | best threshold on the same scores, DKW uniform band on the error rates | yes |
| `holdout` | chosen on one half of the scores, bound computed on the other half | yes |
| `k_fold` | chosen on one fold, bound on the K − 1 others, best fold kept at level γ / K | yes |

Two CSV files are written. `Theoretical_coverage_distribution.csv` has one row per ($n, \mu, \gamma$, family, trial)$ and one colum per estimator with its bound, to plot their distributions. `Theoretical_coverage.csv` aggregates them into the overage (fraction of the bounds below the true value), mean and standard deviation of each estimator in each cell. The script also computes the coverage and statistics of the same Gaussian distribution but with $(\varepsilon, \delta)$ family.
The simulation runs on CPU, the cells in parallel.
To be fast to run the script is implemented with `N_TRIALS = 100`, while the results were reported for `N_TRIALS = 10000` in the paper.

## 2. Re-run of the published audits
```bash 
    python -m selection_bias.papers_experiments.jagielski2020

    python -m selection_bias.papers_experiments.nasr2023         [eps]
    python -m selection_bias.papers_experiments.steinke2023      [eps [seed]]
    python -m selection_bias.papers_experiments.mahloujifar2024  [eps [seed]]
    python -m selection_bias.papers_experiments.annamalai2024
    python -m selection_bias.papers_experiments.cebere2025
    # ->  results/experiments/scores/<paper>/<setting>.parquet
```
Each script re-implements the protocol of one paper, as described in the appendix of ours: it trains the models,
computes the attack scores and the lower bound with the paper's own estimator, and saves the scores so that the
other estimators can be run on them afterwards. `jagielski2020`, `nasr2023` and `steinke2023` need a GPU, the
others run on CPU.


### Score files
One parquet file is generated per setting (a privacy level and, where relevant, a number of copies, an initialization or a run),
with one row per observation. The columns described in `src/selection_bias/utils/score_formats.py`.

The one-run scripts also write one CSV file per run in `results/experiments/lower_bounds/one_run_audits/`.

### Splitting the long scripts over several jobs
The three scripts that take arguments are the long ones: the arguments split their work over several jobs. 
- `nasr2023` takes the privacy levels to run, e.g. `nasr2023 8 16`, and writes one file per $\varepsilon$ with all its repetitions, so the natural split is one job per $\varepsilon$, about an hour each on a GPU. 
- `steinke2023` takes a privacy level and the seed
of one run, e.g. `steinke2023 8 3`, and writes one file per run, so the natural split is one job per run, since a run is a training of several GPU hours. 
- `mahloujifar2024` takes the same arguments because it audits the same runs; it
trains a run itself only when `steinke2023` did not save it. Without arguments, a script runs every setting of its
constants one after the other. The other scripts finish within the hour, so they take no arguments and run everything
in one go.

Jobs on different settings can therefore run in parallel, and a stopped job is resumed by launching it again with the
same arguments. A file is written only when its setting is complete, so a setting interrupted midway is recomputed
from scratch.

## 3. Lower bounds on the saved scores, and the table
```bash
    python -m selection_bias.compute_bounds              # ->  results/experiments/lower_bounds/bounds.csv
```
`compute_bounds` runs every estimator of the table above on every saved audit, at the confidence level of the paper
and for both families ($\varepsilon$ at $\delta$, and $\mu$). `bounds.csv` has one row per (paper, setting, repetition, family) with the
true value, the paper's own bound and one column per estimator. The one-run audits are skipped, their bounds being in
`one_run_audits/`.

