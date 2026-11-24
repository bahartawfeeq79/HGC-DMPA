"""
Baseline: Flat MOEA/D (Zhang and Li 2007)
Multiobjective Evolutionary Algorithm Based on Decomposition.
Operates on continuous N-dimensional decision vectors using augmented Tchebycheff
decomposition with Das-Dennis weights and neighborhood replacement.
Capacity N_arch = 40, Budget = 3,000 evaluations.
"""

import time
import numpy as np
from typing import Dict, List, Tuple

from src.graph.graph_model import SIoTGraphModel
from src.algorithms.flat_decoder import decode_flat_random_key_path
from src.objectives.trust import calculate_path_trust
from src.objectives.latency import calculate_path_latency
from src.objectives.energy import calculate_path_energy
from src.objectives.qos import calculate_path_qos
from src.objectives.constraints import calculate_constraint_violation
from src.evaluation.pareto import (
    ParetoArchive,
    generate_das_dennis_4d_weights,
    compute_tchebycheff_scalar
)

class FlatMOEADOptimizer:
    def __init__(self, graph_model: SIoTGraphModel, src_node: int, dst_node: int,
                 t_dl: float, hop_budget: int = 100,
                 n_pop: int = 40, n_arch: int = 40, t_nbr: int = 8, n_r: int = 2,
                 eval_budget: int = 3000, lb: float = -5.0, ub: float = 5.0, seed: int = 42):
        self.graph_model = graph_model
        self.raw_graph = graph_model.raw_graph
        self.src_node = src_node
        self.dst_node = dst_node
        self.t_dl = t_dl
        self.hop_budget = hop_budget
        
        self.n_pop = n_pop
        self.n_arch = n_arch
        self.t_nbr = t_nbr
        self.n_r = n_r
        self.eval_budget = eval_budget
        self.lb = lb
        self.ub = ub
        self.seed = seed
        
        np.random.seed(seed)
        self.dim = self.raw_graph.number_of_nodes()
        self.weights = generate_das_dennis_4d_weights(n_weights=self.n_pop, seed=seed)
        
        # Neighborhoods
        self.neighborhoods = []
        for i in range(self.n_pop):
            dists = np.linalg.norm(self.weights - self.weights[i], axis=1)
            self.neighborhoods.append(np.argsort(dists)[:self.t_nbr])
            
        self.population = np.random.uniform(self.lb, self.ub, size=(self.n_pop, self.dim))
        self.objectives = np.zeros((self.n_pop, 4))
        self.cvs = np.zeros(self.n_pop)
        
        self.z_ideal = np.ones(4) * float("inf")
        self.z_nadir = np.zeros(4)
        
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
        
        for i in range(self.n_pop):
            path = decode_flat_random_key_path(self.raw_graph, self.population[i], self.src_node, self.dst_node, hop_budget=self.hop_budget)
            F, cv = self.evaluate_path(path)
            self.evals += 1
            
            self.objectives[i] = F
            self.cvs[i] = cv
            self.z_ideal = np.minimum(self.z_ideal, F)
            self.z_nadir = np.maximum(self.z_nadir, F)
            self.archive.add(path, F, cv, key=self.population[i])

        while self.evals < self.eval_budget:
            for i in range(self.n_pop):
                if self.evals >= self.eval_budget:
                    break
                    
                # Differential evolution operator over neighborhood
                nbrs = self.neighborhoods[i]
                r1, r2 = np.random.choice(nbrs, size=2, replace=False)
                
                # DE mutation: v = x_i + F * (x_r1 - x_r2)
                F_de = 0.5
                v = self.population[i] + F_de * (self.population[r1] - self.population[r2])
                
                # Crossover CR = 0.8
                cr = 0.8
                mask = np.random.rand(self.dim) < cr
                mask[np.random.randint(self.dim)] = True # Ensure at least one coordinate
                
                child_x = np.where(mask, v, self.population[i])
                child_x = np.clip(child_x, self.lb, self.ub)
                
                child_path = decode_flat_random_key_path(self.raw_graph, child_x, self.src_node, self.dst_node, hop_budget=self.hop_budget)
                child_F, child_cv = self.evaluate_path(child_path)
                self.evals += 1
                
                self.z_ideal = np.minimum(self.z_ideal, child_F)
                self.z_nadir = np.maximum(self.z_nadir, child_F)
                self.archive.add(child_path, child_F, child_cv, key=child_x)
                
                # Neighborhood update
                denom = np.maximum(1.0e-6, self.z_nadir - self.z_ideal)
                f_norm = (child_F - self.z_ideal) / denom
                
                updated = 0
                for j in np.random.permutation(nbrs):
                    if updated >= self.n_r:
                        break
                    old_f_norm = (self.objectives[j] - self.z_ideal) / denom
                    tch_child = compute_tchebycheff_scalar(f_norm, self.weights[j], np.zeros(4))
                    tch_old = compute_tchebycheff_scalar(old_f_norm, self.weights[j], np.zeros(4))
                    
                    replace = False
                    if child_cv < self.cvs[j] - 1.0e-9:
                        replace = True
                    elif abs(child_cv - self.cvs[j]) < 1.0e-9 and tch_child < tch_old:
                        replace = True
                        
                    if replace:
                        self.population[j] = child_x
                        self.objectives[j] = child_F
                        self.cvs[j] = child_cv
                        updated += 1

        total_time = time.time() - t0
        return {
            "algorithm": "MOEA/D",
            "evals": self.evals,
            "runtime": total_time,
            "feasible_solutions": len(self.archive.solutions),
            "is_feasible": bool(len(self.archive.solutions) > 0),
            "front": self.archive.get_front(),
            "archive": self.archive.solutions
        }
