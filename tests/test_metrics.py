"""
Automated unit tests for performance metrics and statistical validation.
"""

import numpy as np
from src.evaluation.metrics import compute_hypervolume, compute_igd
from src.evaluation.statistical_tests import (
    compute_vargha_delaney_a12,
    holm_bonferroni_correction,
    compare_algorithms_wilcoxon,
    compute_friedman_test
)

def test_metrics_and_statistics():
    # 1. Hypervolume
    front = np.array([
        [0.2, 0.5, 0.4, 0.3],
        [0.4, 0.2, 0.6, 0.5]
    ])
    hv = compute_hypervolume(front)
    assert hv > 0.0
    assert compute_hypervolume(np.empty((0, 4))) == 0.0

    # 2. IGD
    ref_front = np.array([
        [0.1, 0.2, 0.3, 0.2],
        [0.2, 0.3, 0.2, 0.1]
    ])
    igd = compute_igd(front, ref_front)
    assert igd > 0.0

    # 3. Vargha-Delaney A12
    a = np.array([10, 11, 12, 13, 14])
    b = np.array([1, 2, 3, 4, 5])
    a12 = compute_vargha_delaney_a12(a, b)
    assert a12 == 1.0 # a strictly dominates b

    # 4. Holm Bonferroni
    raw_p = [0.01, 0.04, 0.03]
    adj_p, sig = holm_bonferroni_correction(raw_p, alpha=0.05)
    assert len(adj_p) == 3
    # Ordered: 0.01 (factor 3 -> 0.03), 0.03 (factor 2 -> 0.06), 0.04 (factor 1 -> 0.04, max running -> 0.06)
    assert adj_p[0] <= adj_p[2]

    # 5. Friedman test
    # Matrix of shape (n_instances=5, n_algs=3)
    # Alg 0 strictly best
    matrix = np.array([
        [0.9, 0.7, 0.5],
        [0.85, 0.65, 0.45],
        [0.92, 0.72, 0.52],
        [0.88, 0.68, 0.48],
        [0.91, 0.71, 0.51]
    ])
    stat, p_val, ranks = compute_friedman_test(matrix)
    assert ranks[0] == 1.0 # Alg 0 is ranked 1.0 (best)
    assert p_val < 0.01

if __name__ == "__main__":
    test_metrics_and_statistics()
    print("test_metrics.py PASSED!")
