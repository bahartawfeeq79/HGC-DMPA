"""
Baseline: Flat NSGA-III (Deb and Jain 2014)
Reference-point based many-objective evolutionary algorithm operating on flat keys.
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
from src.evaluation.pareto import ParetoArchive, generate_das_dennis_4d_weights
from src.algorithms.nsga2 import sbx_crossover, polynomial_mutation

class FlatNSGA3Optimizer:
    def __init__(self, graph_model: SIoTGraphModel, src_node: int, dst_node: int,
                 t_dl: float, hop_budget: int = 100,
                 n_pop: int = 40, n_arch: int = 40, eval_budget: int = 3000,
                 lb: float = -5.0, ub: float = 5.0, seed: int = 42):
        self.graph_model = graph_model
        self.raw_graph = graph_model.raw_graph
        self.src_node = src_node
        self.dst_node = dst_node
        self.t_dl = t_dl
        self.hop_budget = hop_budget
        
        self.n_pop = n_pop
        self.n_arch = n_arch
        self.eval_budget = eval_budget
        self.lb = lb
        self.ub = ub
        self.seed = seed
        
        np.random.seed(seed)
        self.dim = self.raw_graph.number_of_nodes()
        self.ref_points = generate_das_dennis_4d_weights(n_weights=self.n_pop, seed=seed)
        self.population = np.random.uniform(self.lb, self.ub, size=(self.n_pop, self.dim))
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
        pop_F = np.zeros((self.n_pop, 4))
        pop_cv = np.zeros(self.n_pop)
        
        for i in range(self.n_pop):
            path = decode_flat_random_key_path(self.raw_graph, self.population[i], self.src_node, self.dst_node, hop_budget=self.hop_budget)
            F, cv = self.evaluate_path(path)
            self.evals += 1
            pop_F[i] = F
            pop_cv[i] = cv
            self.archive.add(path, F, cv, key=self.population[i])

        while self.evals < self.eval_budget:
            offspring = []
            for _ in range(self.n_pop // 2):
                if self.evals >= self.eval_budget:
                    break
                p1_idx = np.random.randint(self.n_pop)
                p2_idx = np.random.randint(self.n_pop)
                c1, c2 = sbx_crossover(self.population[p1_idx], self.population[p2_idx], lb=self.lb, ub=self.ub)
                c1 = polynomial_mutation(c1, lb=self.lb, ub=self.ub)
                c2 = polynomial_mutation(c2, lb=self.lb, ub=self.ub)
                offspring.extend([c1, c2])

            if not offspring:
                break
                
            off_F = []
            off_cv = []
            for ind in offspring:
                if self.evals >= self.eval_budget:
                    break
                path = decode_flat_random_key_path(self.raw_graph, ind, self.src_node, self.dst_node, hop_budget=self.hop_budget)
                F, cv = self.evaluate_path(path)
                self.evals += 1
                off_F.append(F)
                off_cv.append(cv)
                self.archive.add(path, F, cv, key=ind)

            if len(off_F) == 0:
                break
            pool_X = np.vstack([self.population, np.array(offspring[:len(off_F)])])
            pool_F = np.vstack([pop_F, np.array(off_F)])
            pool_cv = np.concatenate([pop_cv, np.array(off_cv)])
            
            # Reference-point niching selection:
            # Normalize objectives
            ideal = np.min(pool_F, axis=0)
            nadir = np.max(pool_F, axis=0)
            denom = np.maximum(1.0e-6, nadir - ideal)
            norm_F = (pool_F - ideal) / denom
            
            # Associate each solution with closest reference direction
            # Using perpendicular distance to reference lines
            ref_norms = np.linalg.norm(self.ref_points, axis=1, keepdims=True)
            normalized_refs = self.ref_points / np.maximum(1.0e-6, ref_norms)
            
            # Projection of norm_F onto normalized_refs
            projections = np.dot(norm_F, normalized_refs.T)
            # Distance: ||F - (F . ref) * ref||
            dists = np.zeros((len(pool_F), self.n_pop))
            for r in range(self.n_pop):
                proj_vec = projections[:, r:r+1] * normalized_refs[r:r+1, :]
                dists[:, r] = np.linalg.norm(norm_F - proj_vec, axis=1)
                
            closest_ref = np.argmin(dists, axis=1)
            
            # Feasibility-first selection prioritizing diverse reference associations
            feas_indices = np.where(pool_cv <= 1.0e-9)[0]
            infeas_indices = np.where(pool_cv > 1.0e-9)[0]
            infeas_sorted = infeas_indices[np.argsort(pool_cv[infeas_indices])]
            
            selected = []
            if len(feas_indices) > 0:
                # One candidate per reference point if possible
                ref_to_cands = {r: [] for r in range(self.n_pop)}
                for idx in feas_indices:
                    ref_to_cands[closest_ref[idx]].append(idx)
                    
                for r in range(self.n_pop):
                    if ref_to_cands[r]:
                        # Pick candidate with minimum distance to reference line
                        best_cand = min(ref_to_cands[r], key=lambda i: dists[i, r])
                        selected.append(best_cand)
                        if len(selected) == self.n_pop:
                            break
                            
                # Fill remaining feasible
                remaining_feas = [idx for idx in feas_indices if idx not in selected]
                needed = self.n_pop - len(selected)
                selected.extend(remaining_feas[:needed])

            # Fill remaining with infeasibles sorted by CV
            if len(selected) < self.n_pop:
                needed = self.n_pop - len(selected)
                selected.extend(infeas_sorted[:needed])

            self.population = pool_X[selected]
            pop_F = pool_F[selected]
            pop_cv = pool_cv[selected]

        total_time = time.time() - t0
        return {
            "algorithm": "NSGA-III",
            "evals": self.evals,
            "runtime": total_time,
            "feasible_solutions": len(self.archive.solutions),
            "is_feasible": bool(len(self.archive.solutions) > 0),
            "front": self.archive.get_front(),
            "archive": self.archive.solutions
        }
