"""
Hard Operational Constraints and Normalized Aggregate Constraint Violation CV(P):
f_Lat(P) <= t_dl
min_{v \in P^\circ} Bat(v) >= Bat_min (0.10)
min_{v \in P^\circ} T_base(v) >= T_min (0.35)
Simple loop-free path constraint
Micro-hop budget H = h + max(3, floor(0.3*h)) + 3
(Equations 11-12 in Tawfeeq 2026)
"""

from typing import List, Tuple

def check_simple_path(path: List[int]) -> bool:
    """Returns True if path contains no loops and has >= 2 nodes."""
    return len(path) >= 2 and len(path) == len(set(path))


def calculate_constraint_violation(path: List[int], graph, latency: float, t_dl: float,
                                  bat_min: float = 0.10, t_min: float = 0.35,
                                  hop_budget: int = 100) -> Tuple[float, dict]:
    """
    Computes normalized aggregate constraint violation CV(P) and violation breakdown:
    CV(P) = max(0, f_Lat - t_dl) / t_dl
          + max(0, Bat_min - min_{v \in P^\circ} Bat(v))
          + max(0, T_min - min_{v \in P^\circ} T_base(v))
          + structural penalties (if not simple or exceeding hop budget)
    """
    violations = {
        "is_simple": True,
        "hop_count": len(path) - 1,
        "hop_violation": 0.0,
        "lat_violation": 0.0,
        "bat_violation": 0.0,
        "trust_violation": 0.0,
        "is_feasible": False,
        "cv_total": 0.0
    }
    
    # Check simple path
    if not check_simple_path(path):
        violations["is_simple"] = False
        violations["cv_total"] = 10.0 # Heavy structural violation penalty
        return 10.0, violations
        
    hops = len(path) - 1
    if hops > hop_budget:
        violations["hop_violation"] = float((hops - hop_budget) / hop_budget)
        
    # Latency violation
    if latency > t_dl:
        violations["lat_violation"] = float((latency - t_dl) / max(1.0e-6, t_dl))
        
    # Intermediate relays P^\circ = {v_2, ..., v_{m-1}}
    relays = path[1:-1]
    min_bat = float("inf")
    min_t_base = float("inf")
    
    if relays:
        for r in relays:
            node_data = graph.nodes.get(r, {})
            b = node_data.get('bat', 1.0)
            t = node_data.get('t_base', 1.0)
            if b < min_bat: min_bat = b
            if t < min_t_base: min_t_base = t
            
        if min_bat < bat_min:
            violations["bat_violation"] = float(bat_min - min_bat)
        if min_t_base < t_min:
            violations["trust_violation"] = float(t_min - min_t_base)
            
    cv = (
        violations["lat_violation"] +
        violations["bat_violation"] +
        violations["trust_violation"] +
        violations["hop_violation"]
    )
    violations["cv_total"] = float(cv)
    violations["is_feasible"] = bool(cv < 1.0e-9)
    
    return float(cv), violations
