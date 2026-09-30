# Reliable Estimation of Differential Privacy under Selection Bias

Code for the paper, anonymized for review.

## Setup
```bash
    uv sync        # or: pip install -e .
```
Tested with Python 3.13 on Linux. The datasets (about 500 MB) are downloaded to `data/` the first time a script
needs them.

## Reproducing the results

Run the commands from the root of the repository, with `uv run` or after activating `.venv`.
```bash
    # coverage of the estimators on a Gaussian mechanism  ->  results/theoretical/
    python -m selection_bias.theoretical_simulation

    # re-run of the audits, attack scores  ->  results/experiments/scores/<paper>/
    python -m selection_bias.papers_experiments.jagielski2020
    python -m selection_bias.papers_experiments.nasr2023         [eps ...]
    python -m selection_bias.papers_experiments.steinke2023      [eps [seed]]
    python -m selection_bias.papers_experiments.mahloujifar2024  [eps [seed]]
    python -m selection_bias.papers_experiments.annamalai2024
    python -m selection_bias.papers_experiments.cebere2025

    # lower bounds on the saved scores, one column per estimator  ->  results/experiments/lower_bounds/bounds.csv
    python -m selection_bias.compute_bounds

    # LaTeX table  ->  results/tables/
    python -m selection_bias.utils.table_results
```
A setting whose output file exists is skipped; delete the file to recompute it. Privacy levels, numbers of runs and
repetitions are the constants at the top of each script.

The three scripts that take arguments are the long ones: the arguments split their work over several jobs.
`nasr2023` takes the privacy levels to run, e.g. `nasr2023 8 16`, and writes one file per ε with all its repetitions,
so the natural split is one job per ε, about an hour each on a GPU. `steinke2023` takes a privacy level and the seed
of one run, e.g. `steinke2023 8 3`, and writes one file per run, so the natural split is one job per run, since a run
is a training of several GPU hours. `mahloujifar2024` takes the same arguments because it audits the same runs; it
trains a run itself only when `steinke2023` did not save it. Without arguments, a script runs every setting of its
constants one after the other. The other scripts finish within the hour, so they take no arguments and run everything
in one go.

Jobs on different settings can therefore run in parallel, and a stopped job is resumed by launching it again with the
same arguments. A file is written only when its setting is complete, so a setting interrupted midway is recomputed
from scratch.

`steinke2023` and `mahloujifar2024` audit the same training runs: run `steinke2023` first, `mahloujifar2024` then
reuses its saved runs instead of training again. These two scripts compute their bounds themselves, one CSV file per
run in `results/experiments/lower_bounds/one_run_audits/`, because the scores of a single run are not independent.
`compute_bounds` handles the four other audits.

The estimators are in `src/selection_bias/utils/utils.py` and the format of the score files is described in
`src/selection_bias/utils/score_formats.py`.

