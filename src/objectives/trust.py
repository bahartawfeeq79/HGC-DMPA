"""
Objective 1: Path Trust (Geometric Mean Link Trust)
f_Trust(P) = (\prod_{j=1}^{m-1} T(v_j, v_{j+1}))^{1 / (m - 1)}
(Equation 10 in Tawfeeq 2026)
"""

import math
from typing import List

def calculate_path_trust(path: List[int], graph) -> float:
    """
    Computes geometric mean link trust along the simple path P = (v_1, ..., v_m).
    """
    if len(path) <= 1:
        return 1.0
        
    m = len(path)
    log_sum = 0.0
    for j in range(m - 1):
        u, v = path[j], path[j + 1]
        edge_data = graph.get_edge_data(u, v, default={})
        t = edge_data.get('trust', 0.5)
        # Avoid log(0) with lower bound 1e-6
        t_clamped = max(1.0e-6, min(1.0, float(t)))
        log_sum += math.log(t_clamped)
        
    geom_mean = math.exp(log_sum / (m - 1))
    return float(geom_mean)
