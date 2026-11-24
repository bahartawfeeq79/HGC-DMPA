"""
Deterministic Baseline: WS-KSP (Maristany et al. 2021)
Weighted Scalarized Yen's K-Shortest Paths.
Evaluates k = 150 candidate paths across 20 uniformly distributed weight vectors
over normalized (-ln T, l_adv, e) edge costs, totaling 20 * 150 = 3,000 path evaluations.
(Section V-B in Tawfeeq 2026)
"""

import time
import math
import itertools
import numpy as np
import networkx as nx
from typing import Dict, List, Tuple

from src.graph.graph_model import SIoTGraphModel
from src.objectives.trust import calculate_path_trust
from src.objectives.latency import calculate_path_latency
from src.objectives.energy import calculate_path_energy
from src.objectives.qos import calculate_path_qos
from src.objectives.constraints import calculate_constraint_violation
from src.evaluation.pareto import ParetoArchive

def generate_3d_simplex_weights(n_weights: int = 20) -> np.ndarray:
    """Generates 20 uniformly spaced weight vectors on the 3D unit simplex (w1 + w2 + w3 = 1)."""
    weights = []
    # Grid for p = 5: (5+1)*(5+2)/2 = 21 vectors
    p = 5
    for i in range(p + 1):
        for j in range(p + 1 - i):
            k = p - i - j
            weights.append([i / p, j / p, k / p])
    weights = np.array(weights[:n_weights], dtype=float)
    weights = weights / np.sum(weights, axis=1, keepdims=True)
    return weights


class WSKSPOptimizer:
    def __init__(self, graph_model: SIoTGraphModel, src_node: int, dst_node: int,
                 t_dl: float, hop_budget: int = 100,
                 n_arch: int = 40, k_paths: int = 150, n_weights: int = 20):
        self.graph_model = graph_model
        self.raw_graph = graph_model.raw_graph
        self.src_node = src_node
        self.dst_node = dst_node
        self.t_dl = t_dl
        self.hop_budget = hop_budget
        
        self.n_arch = n_arch
        self.k_paths = k_paths
        self.n_weights = n_weights
        
        self.weights = generate_3d_simplex_weights(n_weights=self.n_weights)
        self.archive = ParetoArchive(capacity=self.n_arch)
        self.evals = 0

    def evaluate_path(self, path: List[int]) -> Tuple[np.ndarray, float]:
        trust = calculate_path_trust(path, self.raw_graph)
        lat = calculate_path_latency(path, self.graph_model)
        ene = calculate_path_energy(path, self.graph_model)
        qos = calculate_path_qos(path, self.raw_graph)
        
        F = np.array([1.0 - trust, lat, ene, 1.0 - qos], dtype=float)
        cv, _ = calculate_constraint_violation(
            path, self.raw_graph, latency=lat, t_dl=self.t_dl,
            bat_min=self.graph_model.bat_min, t_min=self.graph_model.t_min,
            hop_budget=self.hop_budget
        )
        return F, cv

    def run(self) -> Dict:
        t0 = time.time()
        
        # Precompute mean edge latency and energy for normalization
        lat_list, ene_list = [], []
        for u, v in self.raw_graph.edges():
            lat_list.append(self.graph_model.get_hop_latency(u, v))
            ene_list.append(self.graph_model.get_hop_energy(u, v))
        mean_l = float(np.mean(lat_list)) if lat_list else 0.01
        mean_e = float(np.mean(ene_list)) if ene_list else 0.01

        # Precompute edge cost terms once
        for u, v, data in self.raw_graph.edges(data=True):
            t = max(1.0e-6, min(1.0, data.get('trust', 0.5)))
            d_p_adv = data.get('adv_prop_delay', data.get('prop_delay', 0.001))
            dist = data.get('distance', 50.0)
            data['_c1'] = -math.log(t)
            data['_c2'] = (d_p_adv + 0.005) / mean_l
            data['_c3'] = (dist ** 2) / (100.0 ** 2)

        evaluated_paths_set = set()
        
        for w in self.weights:
            w1, w2, w3 = w[0], w[1], w[2]
            
            # Fast vectorized edge weight assignment
            for u, v, data in self.raw_graph.edges(data=True):
                data['_cost'] = max(0.0001, w1 * data['_c1'] + w2 * data['_c2'] + w3 * data['_c3'])

            # Generate top-k shortest simple paths via Yen's algorithm using fast string key
            try:
                paths_generator = nx.shortest_simple_paths(
                    self.raw_graph, source=self.src_node, target=self.dst_node, weight='_cost'
                )
                k_subset = list(itertools.islice(paths_generator, self.k_paths))
            except Exception:
                k_subset = []

            for p in k_subset:
                self.evals += 1
                p_tuple = tuple(p)
                if p_tuple not in evaluated_paths_set:
                    evaluated_paths_set.add(p_tuple)
                    F, cv = self.evaluate_path(p)
                    self.archive.add(p, F, cv)
                    
            if self.evals >= self.n_weights * self.k_paths:
                break

        total_time = time.time() - t0
        return {
            "algorithm": "WS-KSP",
            "evals": self.evals,
            "runtime": total_time,
            "feasible_solutions": len(self.archive.solutions),
            "is_feasible": bool(len(self.archive.solutions) > 0),
            "front": self.archive.get_front(),
            "archive": self.archive.solutions
        }
