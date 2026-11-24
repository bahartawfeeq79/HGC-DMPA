"""
Performance Indicators and Route Quality Evaluation:
1. Hypervolume (HV) in 4-objective normalized space against reference point (1.1, 1.1, 1.1, 1.1)
2. Inverted Generational Distance (IGD) against empirical reference Pareto front P^*
3. Constraint Feasibility Rate (%)
4. Comprehensive route quality metrics
(Section V-C in Tawfeeq 2026)
"""

import numpy as np
from typing import List, Dict
try:
    from pymoo.indicators.hv import HV
    from pymoo.indicators.igd import IGD
    PYMOO_AVAILABLE = True
except Exception:
    PYMOO_AVAILABLE = False

REF_POINT = np.array([1.1, 1.1, 1.1, 1.1])

def compute_hypervolume(front: np.ndarray, ref_point: np.ndarray = REF_POINT) -> float:
    """
    Computes Hypervolume indicator of normalized objective front F.
    All objectives are in canonical minimization form, normalized to [0, 1].
    Reference point: (1.1, 1.1, 1.1, 1.1).
    Returns 0.0 if front is empty or contains no non-dominated solutions.
    """
    if front is None or len(front) == 0:
        return 0.0
        
    front = np.array(front, dtype=float)
    # Filter points that strictly dominate the reference point
    valid_points = front[np.all(front < ref_point, axis=1)]
    if len(valid_points) == 0:
        return 0.0
        
    if PYMOO_AVAILABLE:
        try:
            ind = HV(ref_point=ref_point)
            return float(ind(valid_points))
        except Exception:
            pass

    # Monte Carlo estimation fallback for 4D Hypervolume if exact calculation fails
    n_samples = 50000
    rng = np.random.RandomState(42)
    # Samples in [0, 1.1]^4
    samples = rng.uniform(0.0, 1.1, size=(n_samples, 4))
    
    # Check which samples are dominated by at least one point in valid_points
    # Sample s is dominated by point p if p <= s for all 4 coordinates
    dominated = np.zeros(n_samples, dtype=bool)
    for p in valid_points:
        dom_p = np.all(p <= samples, axis=1)
        dominated = dominated | dom_p
        if np.all(dominated):
            break
            
    total_volume = (1.1 ** 4)
    hv_est = total_volume * (np.sum(dominated) / n_samples)
    return float(hv_est)


def compute_igd(front: np.ndarray, ref_front: np.ndarray) -> float:
    """
    Computes Inverted Generational Distance (IGD):
    Mean Euclidean distance from each point in ref_front to nearest point in front.
    Returns float('nan') or None if front is empty.
    """
    if front is None or len(front) == 0 or ref_front is None or len(ref_front) == 0:
        return float("nan")
        
    front = np.array(front, dtype=float)
    ref_front = np.array(ref_front, dtype=float)
    
    if PYMOO_AVAILABLE:
        try:
            ind = IGD(ref_front)
            return float(ind(front))
        except Exception:
            pass

    # Direct Euclidean distance computation:
    # For each r in ref_front, min_p ||r - p||_2
    dists = np.min(np.linalg.norm(ref_front[:, None, :] - front[None, :, :], axis=2), axis=1)
    return float(np.mean(dists))


def calculate_route_statistics(path: List[int], graph, graph_model,
                               t_dl: float) -> Dict[str, float]:
    """
    Extracts all route quality metrics from an actual physical path.
    """
    from src.objectives.trust import calculate_path_trust
    from src.objectives.latency import calculate_path_latency
    from src.objectives.energy import calculate_path_energy
    from src.objectives.qos import calculate_path_qos
    
    hops = len(path) - 1
    trust = calculate_path_trust(path, graph)
    lat = calculate_path_latency(path, graph_model)
    ene = calculate_path_energy(path, graph_model)
    qos = calculate_path_qos(path, graph)
    
    relays = path[1:-1]
    min_bat = min([graph.nodes[r].get('bat', 1.0) for r in relays]) if relays else 1.0
    min_t_base = min([graph.nodes[r].get('t_base', 1.0) for r in relays]) if relays else 1.0
    compromised_count = sum([1 for r in relays if graph.nodes[r].get('compromised', False)])
    
    return {
        "hops": hops,
        "trust": trust,
        "latency": lat,
        "energy": ene,
        "qos": qos,
        "min_relay_battery": min_bat,
        "min_relay_trust": min_t_base,
        "compromised_relays": compromised_count,
        "latency_margin": (t_dl - lat) / max(1e-6, t_dl)
    }
