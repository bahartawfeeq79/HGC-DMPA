"""
Automated unit tests for Beta-reputation and social trust calculations (Eq. 1-4).
"""

import math
import networkx as nx
import pytest
from src.trust.beta_reputation import (
    compute_social_score,
    compute_direct_trust,
    compute_indirect_trust,
    compute_composite_trust
)
from src.trust.social_affinity import compute_affinity_weight

def test_social_score():
    # S_uv = 0.4*CL + 0.4*CW + 0.2*OL
    assert compute_social_score(1, 1, 1) == 1.0
    assert compute_social_score(0, 0, 0) == 0.0
    assert abs(compute_social_score(1, 0, 1) - 0.6) < 1.0e-9
    assert abs(compute_social_score(0, 1, 0) - 0.4) < 1.0e-9

def test_direct_trust():
    # T_dir = (s + 1) / (s + f + 2)
    # Neutral prior
    assert compute_direct_trust(0, 0) == 0.5
    # High trust: 9 successful, 0 failed -> 10 / 11
    assert abs(compute_direct_trust(9, 0) - (10.0 / 11.0)) < 1.0e-9
    # Untrusted: 0 successful, 8 failed -> 1 / 10 = 0.1
    assert abs(compute_direct_trust(0, 8) - 0.1) < 1.0e-9

def test_composite_trust():
    # T = (alpha * T_dir + (1-alpha)*T_ind + beta * S) / (1 + beta)
    # alpha = 0.7, beta = 0.3
    t_dir = 0.8
    t_ind = 0.6
    s_uv = 0.5
    # num = 0.7 * 0.8 + 0.3 * 0.6 + 0.3 * 0.5 = 0.56 + 0.18 + 0.15 = 0.89
    # den = 1.3
    expected = 0.89 / 1.3
    assert abs(compute_composite_trust(t_dir, t_ind, s_uv, alpha=0.7, beta=0.3) - expected) < 1.0e-9

def test_affinity_weight():
    # w_uv = T(u, v) * (1 + gamma * S_uv), gamma = 0.5
    t = 0.8
    s = 1.0
    expected = 0.8 * (1.0 + 0.5 * 1.0) # 1.2
    assert abs(compute_affinity_weight(t, s, gamma=0.5) - expected) < 1.0e-9

if __name__ == "__main__":
    test_social_score()
    test_direct_trust()
    test_composite_trust()
    test_affinity_weight()
    print("test_trust.py PASSED!")
