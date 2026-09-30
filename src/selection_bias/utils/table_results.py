"""
LaTeX table of the re-audits: one block per paper, one row per theoretical eps, columns: the paper's own bound, then
Union, DKW, Hold-out and K-fold on the same scores (mean over the repetitions, one std as a subscript). The rows of
the one-run audits (Steinke, Mahloujifar) come from one_run_audits/, the others from bounds.csv.

Run: python -m selection_bias.utils.table_results [--family eps_delta|mu_gdp] [--csv path]
Input: results/experiments/lower_bounds/bounds.csv (compute_bounds.py) and one_run_audits/ (the one-run scripts)
Output: results/tables/prior_audits_results.tex (prior_audits_results_mu.tex for the mu_gdp family); it needs
\\usepackage{tabularx}, \\usepackage{booktabs}, \\usepackage{multirow} and \\usepackage[table]{xcolor}.
"""
import argparse
import datetime
import re
from pathlib import Path

import numpy as np
import pandas as pd

from .paths import BOUNDS_DIR, ROOT, TABLES_DIR
from .utils import eps_of_mu, mu_theory   # the GDP conversions of compute_bounds.py, for the GDP-based estimators

BOUNDS_PATH = BOUNDS_DIR / "bounds.csv"
ONE_RUN_DIR = BOUNDS_DIR / "one_run_audits"             # one CSV file per run
ONE_RUN_GAMMA = 0.05                                 # P of steinke2023.py / mahloujifar2024.py (not in their CSV)

METHODS = {"cp_max": "Max", "cp_union": "Union", "dkw": "DKW", "holdout": "Hold-out", "k_fold": "K-fold"}
MAIN_METHODS = ["cp_union", "dkw", "holdout", "k_fold"]   # columns of the main table, after the paper's own bound
MAIN_TABLE = {   # the papers of the main table: regex of the settings shown, theoretical eps of the protocol (one row
                 # per eps in this order, "--" when not run yet), confidence 1 - gamma of the paper, and whether the
                 # paper's own bound is valid (False: threshold or number of guesses chosen on the same scores without
                 # correction; the bound is then marked with INVALID_MARK and never in bold)
    "jagielski2020": dict(settings=r"fmnist_eps[\d.]+_k4", eps=[1, 2, 4, 8], confidence=0.99,
                          valid=True),        # ClipBKD canary inserted k = 4 times, threshold chosen on other runs
    "nasr2023": dict(settings=r"whitebox_eps[\d.]+", eps=[1, 4, 8, 16], confidence=0.95,
                     valid=False),            # white-box Dirac canaries, best threshold on the same runs
    "steinke2023": dict(settings=r"wrn16-4_eps[\d.]+", eps=[1, 2, 4, 8], confidence=0.95,
                        valid=False),         # one run, WRN-16-4, best number of guesses on the same run
    "mahloujifar2024": dict(settings=r"wrn16-4_eps[\d.]+", eps=[1, 2, 4, 8], confidence=0.95,
                            valid=False),     # same runs, their estimator, best number of guesses on the same run
    "annamalai2024_blackbox": dict(settings=r"finetune_worst_eps[\d.]+", eps=[1, 2, 4, 10], confidence=0.95,
                                   valid=False, via_mu=True),   # last-layer fine-tuning, worst-case init, ClipBKD
                                   # target, their GDP estimator with the best threshold on the same runs; via_mu: the
                                   # corrected bounds are the mu-family ones converted to eps with the same GDP formula
                                   # (in the (eps, delta) family the gap would mostly be the conversion, not the selection)
    "cebere2025": dict(settings=r"GC-R_k1_eps[\d.]+", eps=[1, 4, 8, 12, 24], confidence=0.95,
                       valid=False, via_mu=True),   # hidden state, Housing FCNN, adversary A_GC-R (canary gradient at a
                       # random coordinate), C = 1, canary inserted at every step (k = 1), their GDP estimator with the
                       # best threshold on the same runs
}
MIN_REPETITIONS = 16               # every row of the main table is a mean over at least this many audits
INVALID_MARK = r"\cellcolor{lightgray}"
# FOOTNOTE = (r"$^{*}$ Threshold or number of guesses chosen on the same scores without correction: "
#             r"not a valid lower bound at confidence $1-\gamma$.")
FOOTNOTE = (r"\textcolor{lightgray}{\rule[0ex]{3ex}{1.5ex}} Threshold or number of guesses chosen on the same scores without correction: not a valid lower bound at confidence $1-\gamma$.")
ORDER = ["jagielski2020", "nasr2023", "steinke2023", "mahloujifar2024", "annamalai2024_blackbox", "cebere2025"]
PAPER_NAMES = {
    "jagielski2020": r"Jagielski et al.~\cite{jagielskiAuditingDifferentiallyPrivate2020a}",
    "nasr2023": r"Nasr et al.~\cite{nasrTightAuditingDifferentially2023}",
    "steinke2023": r"Steinke et al.~\cite{steinkePrivacyAuditingOne2023}",
    "mahloujifar2024": r"Mahloujifar et al.~\cite{mahloujifarAuditing$f$DifferentialPrivacy2024}",
    "annamalai2024_blackbox": r"Annamalai \& De Cristofaro~\cite{annamalaiNearlyTightBlackBox2024}",
    "cebere2025": r"Cebere et al.~\cite{cebereTighterPrivacyAuditing2025}",
}


# Formatting
def setting_label(setting):
    """'finetune_average_eps1.0' -> 'average, $\\varepsilon{=}1$'; 'GC-R_k1' -> 'GC-R, $k{=}1$'."""
    symbols = {"eps": r"\varepsilon", "sigma": r"\sigma", "k": "k", "K": "K", "T": "T", "q": "q"}
    parts = []
    for token in setting.split("_"):
        m = re.fullmatch(r"(eps|sigma|k|K|T|q)(\d+(?:\.\d+)?|inf)", token)
        if m:
            value = r"\infty" if m[2] == "inf" else m[2].rstrip("0").rstrip(".") if "." in m[2] else m[2]
            parts.append(f"${symbols[m[1]]}{{=}}{value}$")
        elif token != "finetune":
            parts.append(token.replace("_", r"\_"))
    return ", ".join(parts)

def value(mean, std=0.0, bold=False, mark=""):
    """'$1.23$', '$1.23_{\\pm0.05}$' when std > 0, the mean in bold when bold, mark ('^{*}') appended; '--' when
    undefined."""
    if mean is None or not np.isfinite(mean):
        return "--"
    text = rf"\mathbf{{{mean:.2f}}}" if bold else f"{mean:.2f}"
    if np.isfinite(std) and std > 0:
        text += rf"_{{\pm{std:.2f}}}"
    return f"${text}{mark}$"

def number(x):
    """A theoretical value or a confidence level: '$4$', '$0.95$', '$0.27$'; '--' when undefined."""
    if x is None or not np.isfinite(x):
        return "--"
    return f"${round(x, 2):g}$" if float(round(x, 2)).is_integer() else f"${x:.2f}$"

def change(ours, theirs):
    """Relative change of ours against theirs in %, signed: '$+28\\%$'; '--' when theirs is not a positive bound."""
    if not (np.isfinite(ours) and np.isfinite(theirs)) or theirs <= 0:
        return "--"
    return rf"${100 * (ours - theirs) / theirs:+.0f}\%$"

def paper_name(paper):
    return PAPER_NAMES.get(paper, paper.replace("_", r"\_"))

def header(csv_path):
    return (f"% Generated by table_results.py from {csv_path.relative_to(ROOT) if csv_path.is_relative_to(ROOT) else csv_path}"
            f" on {datetime.date.today()}; do not edit by hand.\n"
            "% Requires \\usepackage{tabularx}, \\usepackage{booktabs}, \\usepackage{multirow} and "
            "\\usepackage[table]{xcolor}.\n")

def summarize(df, family):
    """One row per (paper, setting) of the family: theoretical eps, claimed value of the family (eps or mu), gamma,
    the paper's bound (NaN if not of the family) and the mean and std over repetitions of every method.
    For a paper of MAIN_TABLE with via_mu (an estimator that goes through mu-GDP), the eps rows hold the mu-family
    bounds of every method converted to eps with the GDP formula the paper uses (per repetition), and the mu rows
    hold the paper's eps bound converted to mu, so that both families compare like with like."""
    rows = []
    for (paper, setting), g in df[df.family == family].groupby(["paper", "setting"], sort=True):
        via_mu = MAIN_TABLE.get(paper, {}).get("via_mu", False)
        if via_mu and family == "eps_delta":
            mu_rows = df[(df.paper == paper) & (df.setting == setting) & (df.family == "mu_gdp")].set_index("repetition").loc[g.repetition]
            g = g.copy()
            for m in METHODS:
                g[m] = [eps_of_mu(mu, d) if mu > 0 else 0.0 for mu, d in zip(mu_rows[m], mu_rows.delta)]
        if (family == "mu_gdp") == (g.family_paper.iloc[0] == "mu_gdp"):
            theirs = g.bound_paper
        elif via_mu and family == "mu_gdp":
            theirs = pd.Series([mu_theory(b, d) for b, d in zip(g.bound_paper, g.delta)])
        else:
            theirs = pd.Series([np.nan])
        eps = df[(df.paper == paper) & (df.setting == setting) & (df.family == "eps_delta")].true_value.iloc[0]
        row = {"paper": paper, "setting": setting, "n_reps": len(g), "eps": eps, "claimed": g.true_value.iloc[0],
               "gamma": g.gamma.iloc[0], "theirs": theirs.mean(), "theirs_std": theirs.std() if len(g) > 1 else 0.0}
        for m in METHODS:
            row[m], row[m + "_std"] = g[m].mean(), g[m].std() if len(g) > 1 else 0.0
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["paper", "claimed", "setting"], na_position="last")

def one_run_summary(folder):
    """Rows of the main table for the one-run papers, from the files of one_run_audits/ (one per run): their best
    sweep (max, "theirs"), the union-corrected sweep (cp_union), the hold-out sweep (holdout) and the K-fold sweep
    (k_fold), mean and std over the runs of a setting; DKW does not apply. Rows audited before a column existed
    (NaN, or the column missing) are left out of that column's mean."""
    files = sorted(folder.glob("*.csv"))
    if not files:
        return pd.DataFrame(columns=["paper", "setting", "n_reps", "eps", "claimed", "gamma", "theirs", "theirs_std"])
    rows = []
    for (paper, setting), g in pd.concat([pd.read_csv(f) for f in files]).groupby(["paper", "setting"]):
        row = {"paper": paper, "setting": setting, "n_reps": len(g), "eps": g.eps.iloc[0], "claimed": g.eps.iloc[0],
               "gamma": ONE_RUN_GAMMA}
        for column, name in [("max", "theirs"), ("union", "cp_union"), ("holdout", "holdout"), ("k_fold", "k_fold")]:
            values = g[column].dropna() if column in g else pd.Series(dtype=float)
            row[name], row[name + "_std"] = values.mean(), values.std() if len(values) > 1 else 0.0
        rows.append(row)
    return pd.DataFrame(rows)

def main_rows(rows, paper, family):
    """The rows of a paper in the main table: its settings of MAIN_TABLE, one per theoretical eps, completed with
    the eps of the protocol that have no data (eps family only, where the claimed value is that eps)."""
    found = rows[(rows.paper == paper) & rows.setting.str.fullmatch(MAIN_TABLE[paper]["settings"])]
    planned = MAIN_TABLE[paper]["eps"]
    if family == "eps_delta":
        missing = [eps for eps in planned if not np.isclose(found.eps, eps, atol=0.01).any()]
        found = pd.concat([found, pd.DataFrame({"paper": paper, "eps": missing, "claimed": missing})], ignore_index=True)
    return found.sort_values("eps")


def bound_cells(r, theirs_valid):
    """The paper's bound, the methods of MAIN_METHODS and Delta for one row. The largest valid bound of the row is in
    bold (ties on the displayed value all in bold, nothing when it is 0); the paper's bound counts as valid only when
    theirs_valid, otherwise it is marked with INVALID_MARK. Delta is the relative change of that largest valid bound
    against the paper's bound, on the means."""
    columns = ["theirs"] + MAIN_METHODS
    means = {c: float(r.get(c, np.nan)) for c in columns}
    valid = [c for c in columns if c != "theirs" or theirs_valid]
    best = max([means[c] for c in valid if np.isfinite(means[c])], default=np.nan)
    cells = [value(means[c], r.get(c + "_std", 0.0), bold=c in valid and best > 0 and round(means[c], 2) == round(best, 2),
                   mark=INVALID_MARK if c == "theirs" and not theirs_valid else "") for c in columns]
    return cells + [change(best, means["theirs"])]


def main_table(summary, one_run, family):
    symbol = r"\varepsilon" if family == "eps_delta" else r"\mu"
    n_columns = 5 + len(MAIN_METHODS)                # paper, confidence, theoretical, theirs, the methods, Delta
    lines = [r"\begin{table*}[t]", r"\centering",
             rf"\caption{{Prior audits re-run and re-estimated on the same scores. ${symbol}_\text{{th}}$: the guarantee "
             rf"of the audited mechanism. $\underline{symbol}$: lower bound at the paper's confidence level $1-\gamma$, "
             rf"with the paper's own estimator (their method) and with the corrected estimators of "
             rf"Section~\ref{{sec:correction}}; mean over the repetitions of the audit, one std as a subscript. "
             rf"Bold: the largest valid bound of the row. $^{{*}}$: the paper's bound is not a valid lower bound "
             rf"(selection on the same scores). $\Delta$: relative change of the bold bound against the paper's, on "
             rf"the means. ``--'': not applicable or not run. For an estimator that goes through $\mu$-GDP (Annamalai \& "
             rf"De Cristofaro, Cebere et al.), the corrected bounds are $\mu$ bounds converted to $\varepsilon$ with the paper's formula. "
             rf"Protocols in Appendix~\ref{{app:exp_details}}.}}",
             r"\label{tab:prior_audits_results}", r"\footnotesize", r"\setlength{\tabcolsep}{4pt}",
             r"\renewcommand{\arraystretch}{1.1}",
             r"\begin{tabularx}{\textwidth}{@{} >{\raggedright\arraybackslash}X c c c c c c c c @{}}", r"\toprule",
             r" & Confidence & Theoretical & Paper's method & Union & DKW & Held-out & K-fold & Best vs.\ paper \\",
             rf"\cmidrule(lr){{2-2}} \cmidrule(lr){{3-3}} \cmidrule(lr){{4-4}} \cmidrule(lr){{5-{n_columns - 1}}} "
             rf"\cmidrule(lr){{{n_columns}-{n_columns}}}",
             rf"Paper & $1-\gamma$ & ${symbol}_\text{{th}}$ & $\underline{symbol}$ & $\underline{symbol}$ & "
             rf"$\underline{symbol}$ & $\underline{symbol}$ & $\underline{symbol}$ & $\Delta$ \\",
             r"\midrule"]
    rows = pd.concat([summary, one_run], ignore_index=True)
    for paper in [p for p in ORDER if p in MAIN_TABLE]:
        block = main_rows(rows, paper, family)
        if block.empty:
            continue
        info = MAIN_TABLE[paper]
        if not np.allclose(1 - block.gamma.dropna(), info["confidence"]):
            print(f"warning: {paper} audited at 1 - gamma = {set(1 - block.gamma.dropna().round(4))}, "
                  f"MAIN_TABLE says {info['confidence']}")
        few = block[block.n_reps < MIN_REPETITIONS]
        if len(few):
            print(f"warning: {paper} has fewer than {MIN_REPETITIONS} repetitions at eps = "
                  f"{dict(zip(few.eps.round(2), few.n_reps.astype(int)))}")
        for i, (_, r) in enumerate(block.iterrows()):
            first = [rf"\multirow{{{len(block)}}}{{=}}{{{paper_name(paper)}}}",
                     rf"\multirow{{{len(block)}}}{{*}}{{{number(info['confidence'])}}}"]
            cells = (first if i == 0 else ["", ""]) + [number(r.claimed)] + bound_cells(r, info["valid"])
            lines.append(" & ".join(cells) + r" \\")
        lines.append(r"\addlinespace[4pt]")
    lines += [r"\bottomrule", r"\addlinespace[2pt]", rf"\multicolumn{{{n_columns}}}{{@{{}}l@{{}}}}{{\scriptsize {FOOTNOTE}}} \\",
              r"\end{tabularx}", r"\end{table*}"]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--family", default="eps_delta", choices=["eps_delta", "mu_gdp"])
    parser.add_argument("--csv", type=Path, default=BOUNDS_PATH)
    args = parser.parse_args()
    summary = summarize(pd.read_csv(args.csv), args.family)
    one_run = one_run_summary(ONE_RUN_DIR if args.family == "eps_delta" else Path("/nonexistent"))   # eps only
    suffix = "" if args.family == "eps_delta" else "_mu"
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    path = TABLES_DIR / f"prior_audits_results{suffix}.tex"
    path.write_text(header(args.csv) + main_table(summary, one_run, args.family))
    print(f"wrote {path}")
