"""
Non-Parametric Statistical Testing Module:
1. Two-sided Wilcoxon rank-sum tests (for independent stochastic runs)
2. One-sample Wilcoxon signed-rank test (for comparisons against deterministic WS-KSP)
3. Holm-Bonferroni step-down correction at alpha = 0.05
4. Vargha-Delaney A12 effect size calculation
5. Non-parametric Friedman ranking test across stochastic algorithms
(Section V-C in Tawfeeq 2026)
"""

import numpy as np
from scipy import stats
from typing import List, Dict, Tuple

def compute_vargha_delaney_a12(sample_a: np.ndarray, sample_b: np.ndarray) -> float:
    """
    Computes Vargha-Delaney A12 effect size (probability that a randomly selected
    value from sample_a is greater than a randomly selected value from sample_b):
    A12 = (R_1 / m - (m + 1) / 2) / n
    where R_1 is rank sum of sample_a, m = len(sample_a), n = len(sample_b).
    A12 > 0.71 denotes large positive effect.
    A12 < 0.29 denotes large negative effect.
    """
    m = len(sample_a)
    n = len(sample_b)
    if m == 0 or n == 0:
        return 0.5
        
    combined = np.concatenate([sample_a, sample_b])
    ranks = stats.rankdata(combined)
    r1 = np.sum(ranks[:m])
    
    a12 = (r1 / m - (m + 1.0) / 2.0) / n
    return float(a12)


def holm_bonferroni_correction(p_values: List[float], alpha: float = 0.05) -> Tuple[List[float], List[bool]]:
    """
    Applies Holm-Bonferroni step-down procedure:
    Given ordered p-values p_(1) <= p_(2) <= ... <= p_(k),
    adjusted p-value: p_adj_(i) = min(1.0, max_{j <= i} (k - j + 1) * p_(j)).
    Returns:
        adjusted_p_values: List[float]
        is_significant: List[bool]
    """
    k = len(p_values)
    if k == 0:
        return [], []
        
    indexed_p = sorted(enumerate(p_values), key=lambda x: x[1])
    adj_p = [0.0] * k
    running_max = 0.0
    
    for rank_idx, (orig_idx, p_val) in enumerate(indexed_p):
        step_factor = k - rank_idx
        val = p_val * step_factor
        if val > running_max:
            running_max = val
        adj_p[orig_idx] = min(1.0, max(0.0, running_max))
        
    significant = [adj <= alpha for adj in adj_p]
    return adj_p, significant


def compare_algorithms_wilcoxon(control_runs: np.ndarray, baseline_runs: np.ndarray) -> Tuple[float, float]:
    """
    Compares HGC-DMPA (control) against a baseline optimizer over independent runs:
    Returns (raw_p_value, a12_effect_size).
    """
    control_runs = np.array(control_runs, dtype=float)
    baseline_runs = np.array(baseline_runs, dtype=float)
    
    # Check if identical arrays (e.g. all zeros)
    if np.array_equal(control_runs, baseline_runs):
        return 1.0, 0.5
        
    # Two-sided Mann-Whitney U test (Wilcoxon rank-sum for independent samples)
    try:
        stat, p_val = stats.mannwhitneyu(control_runs, baseline_runs, alternative='two-sided')
    except Exception:
        p_val = 1.0
        
    a12 = compute_vargha_delaney_a12(control_runs, baseline_runs)
    return float(p_val), float(a12)


def compare_against_deterministic(control_runs: np.ndarray, deterministic_val: float) -> Tuple[float, float]:
    """
    Compares HGC-DMPA against deterministic WS-KSP using Wilcoxon signed-rank test.
    """
    control_runs = np.array(control_runs, dtype=float)
    diffs = control_runs - deterministic_val
    if np.all(diffs == 0.0):
        return 1.0, 0.5
    try:
        stat, p_val = stats.wilcoxon(diffs, alternative='two-sided')
    except Exception:
        # Fallback to sign test or binomial
        pos = np.sum(diffs > 0)
        neg = np.sum(diffs < 0)
        p_val = stats.binomtest(pos, pos + neg, 0.5).pvalue if (pos + neg) > 0 else 1.0
        
    a12 = 1.0 if np.mean(control_runs) > deterministic_val else (0.0 if np.mean(control_runs) < deterministic_val else 0.5)
    return float(p_val), float(a12)


def compute_friedman_test(algorithm_means_matrix: np.ndarray) -> Tuple[float, float, np.ndarray]:
    """
    Computes non-parametric Friedman test across algorithms across benchmark instances.
    algorithm_means_matrix: shape (n_instances, n_algorithms), higher is better.
    Returns:
        friedman_stat: float
        p_value: float
        mean_ranks: np.ndarray (lower rank is better, Rank 1.0 is highest)
    """
    n_instances, n_algs = algorithm_means_matrix.shape
    
    # Rank algorithms per instance (Rank 1 for best / highest metric)
    ranks = np.zeros_like(algorithm_means_matrix)
    for i in range(n_instances):
        # We invert so highest value gets rank 1
        ranks[i] = stats.rankdata(-algorithm_means_matrix[i])
        
    mean_ranks = np.mean(ranks, axis=0)
    
    # SciPy friedmanchisquare takes one array per algorithm across instances
    try:
        args = [algorithm_means_matrix[:, j] for j in range(n_algs)]
        stat, p_val = stats.friedmanchisquare(*args)
    except Exception:
        stat, p_val = 0.0, 1.0
        
    return float(stat), float(p_val), mean_ranks
