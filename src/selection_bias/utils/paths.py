"""Folders of the repository (the package is installed in editable mode, so they are found from this file)."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = ROOT / "data"                                   # datasets, downloaded on first use
RESULTS_DIR = ROOT / "results"
THEORY_DIR = RESULTS_DIR / "theoretical"                   # simulation
SCORES_DIR = RESULTS_DIR / "experiments" / "scores"        # attack scores of the re-run audits
BOUNDS_DIR = RESULTS_DIR / "experiments" / "lower_bounds"  # lower bounds computed on these scores
TABLES_DIR = RESULTS_DIR / "tables"                        # LaTeX tables
