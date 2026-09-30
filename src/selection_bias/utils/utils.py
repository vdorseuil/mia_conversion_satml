"""Privacy conversions, accounting and lower-bound estimators shared by the simulation and the re-audits."""
from functools import lru_cache

import numpy as np
from dp_accounting.pld import privacy_loss_distribution as pld_lib
from scipy import optimize, stats

DELTA = 1e-5
K_FOLDS = 5


# Conversion functions
def delta_eps_mu(eps, mu):
    return stats.norm.cdf(-eps/mu + mu/2) - np.exp(eps) * stats.norm.cdf(-eps/mu - mu/2)

def convert_mu_epsilon(mu, delta):   # [0, 500] covers up to mu = 27
    def objective(eps):
        return delta_eps_mu(eps, mu) - delta
    return optimize.root_scalar(objective, bracket=[0, 500], method="brentq").root

def eps_of_mu(mu, delta):
    """eps at delta of mu-GDP; 0 when delta alone covers the mechanism (delta(eps = 0) <= delta, e.g. tiny mu)."""
    return 0.0 if delta_eps_mu(0.0, mu) <= delta else convert_mu_epsilon(mu, delta)

def mu_theory(eps, delta):
    """mu of the mu-GDP mechanism with the same (eps, delta) point; NaN for pure DP or eps = inf, 0 below the bracket."""
    if not np.isfinite(eps) or delta <= 0:
        return np.nan
    if eps <= eps_of_mu(1e-4, delta):
        return 0.0
    return optimize.brentq(lambda mu: eps_of_mu(mu, delta) - eps, 1e-4, 20)   # mu = 20 is eps ~ 285


# Accounting of DP-SGD
def epsilon_dpsgd(sigma, q, T, delta=DELTA):
    """eps at delta of T steps of the Poisson-subsampled Gaussian mechanism at rate q (PLD accountant)."""
    return pld_lib.from_gaussian_mechanism(standard_deviation=sigma, sampling_prob=q,
                                           value_discretization_interval=1e-3).self_compose(T).get_epsilon_for_delta(delta)

def calibrate_sigma(eps, q, T, delta=DELTA):
    """Noise multiplier such that T steps of Poisson-subsampled Gaussian mechanism at rate q are (eps, delta)-DP."""
    low, high = 0.3, 100.0
    for _ in range(40):
        mid = (low + high) / 2
        low, high = (mid, high) if epsilon_dpsgd(mid, q, T, delta) > eps else (low, mid)
    return high


# Lower bounds (work on arrays: one value per threshold)
def mu_gdp_lower(alpha_upper, beta_upper):
    return stats.norm.ppf(1 - alpha_upper) - stats.norm.ppf(beta_upper)

def epsilon_dp_lower(alpha_upper, beta_upper, delta=DELTA):
    with np.errstate(divide="ignore"):
        res_1 = np.log(np.maximum(1 - alpha_upper - delta, 0) / beta_upper)
        res_2 = np.log(np.maximum(1 - beta_upper - delta, 0) / alpha_upper)
    return np.maximum(res_1, res_2)


# Clopper-Pearson: it only depends on the count k, so we compute it once for k = 0..n
@lru_cache(maxsize=None)
def cp_table(n, gamma):
    k = np.arange(n)
    return np.append(stats.beta.ppf(1 - gamma, k + 1, n - k), 1.0)

def cp_upper_bound(k, n, gamma):   # many bounds with the same n (the estimators below)
    return cp_table(n, gamma)[k]

def cp_upper(k, n, gamma):         # without the table (the papers' own estimators)
    """One-sided Clopper-Pearson upper bound at confidence 1 - gamma (arrays allowed)."""
    return np.where(k < n, stats.beta.ppf(1 - gamma, k + 1, np.maximum(n - k, 1)), 1.0)


# Attack "guess IN if score >= threshold", evaluated at many thresholds at once
def candidate_thresholds(s_in, s_out):
    return np.sort(np.concatenate([s_in, s_out]))   # every observed score (sorted only for speed)

def error_counts(s_in, s_out, thresholds):
    fp = len(s_out) - np.searchsorted(np.sort(s_out), thresholds)   # OUT scores >= threshold
    fn = np.searchsorted(np.sort(s_in), thresholds)                 # IN scores < threshold
    return fp, fn

def cp_lower_bounds(s_in, s_out, thresholds, gamma, lower_function):
    """Lower bound at each threshold, CP at level gamma/2 on the FPR and on the FNR."""
    fp, fn = error_counts(s_in, s_out, thresholds)
    alpha_upper = cp_upper_bound(fp, len(s_out), gamma / 2)
    beta_upper = cp_upper_bound(fn, len(s_in), gamma / 2)
    return lower_function(alpha_upper, beta_upper)


# Estimators. All return a lower bound that should hold with confidence 1 - gamma.
def estimate_cp_fixed(s_in, s_out, gamma, lower_function, threshold):
    """Threshold chosen before seeing the data."""
    return max(0, cp_lower_bounds(s_in, s_out, np.array([threshold]), gamma, lower_function)[0])

def estimate_cp_max(s_in, s_out, gamma, lower_function):
    """Naive: best threshold picked on the same data, no correction."""
    thresholds = candidate_thresholds(s_in, s_out)
    return max(0, np.max(cp_lower_bounds(s_in, s_out, thresholds, gamma, lower_function)))

def estimate_cp_union(s_in, s_out, gamma, lower_function):
    """Best threshold picked on the same data, union bound over the T candidate thresholds."""
    thresholds = candidate_thresholds(s_in, s_out)
    T = len(thresholds)
    return max(0, np.max(cp_lower_bounds(s_in, s_out, thresholds, gamma / T, lower_function)))

def estimate_dkw(s_in, s_out, gamma, lower_function):
    """DKW uniform band at level gamma/2 on each error, then best threshold."""
    thresholds = candidate_thresholds(s_in, s_out)
    fp, fn = error_counts(s_in, s_out, thresholds)
    alpha_upper = np.minimum(1, fp / len(s_out) + np.sqrt(np.log(2 / gamma) / (2 * len(s_out))))
    beta_upper = np.minimum(1, fn / len(s_in) + np.sqrt(np.log(2 / gamma) / (2 * len(s_in))))
    return max(0, np.max(lower_function(alpha_upper, beta_upper)))

def select_threshold(s_in, s_out, gamma, lower_function):
    thresholds = candidate_thresholds(s_in, s_out)
    return thresholds[np.argmax(cp_lower_bounds(s_in, s_out, thresholds, gamma, lower_function))]

def estimate_holdout(s_in, s_out, gamma, lower_function):
    """First half of the data selects the threshold, second half evaluates it."""
    h_in, h_out = len(s_in) // 2, len(s_out) // 2
    threshold = select_threshold(s_in[:h_in], s_out[:h_out], gamma, lower_function)
    return estimate_cp_fixed(s_in[h_in:], s_out[h_out:], gamma, lower_function, threshold)

def estimate_k_fold(s_in, s_out, gamma, lower_function, k_folds=K_FOLDS):
    """Each fold selects a threshold, which is evaluated on the K-1 other folds (the hold-out split reversed: the
    bound uses (K-1)/K of the data). We keep the best fold, so each fold is tested at level gamma / K (union bound)."""
    folds_in = np.array_split(s_in, k_folds)
    folds_out = np.array_split(s_out, k_folds)
    best = 0
    for i in range(k_folds):
        threshold = select_threshold(folds_in[i], folds_out[i], gamma, lower_function)
        eval_in = np.concatenate(folds_in[:i] + folds_in[i+1:])
        eval_out = np.concatenate(folds_out[:i] + folds_out[i+1:])
        best = max(best, estimate_cp_fixed(eval_in, eval_out, gamma / k_folds, lower_function, threshold))
    return best
