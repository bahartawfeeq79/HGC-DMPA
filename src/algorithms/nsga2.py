"""
Baseline: Flat NSGA-II (Deb et al. 2002)
Operates on continuous N-dimensional random-key vectors with SBX crossover,
polynomial mutation, fast non-dominated sorting, and crowding-distance selection.
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
from src.evaluation.pareto import ParetoArchive, dominates

def sbx_crossover(p1: np.ndarray, p2: np.ndarray, eta_c: float = 15.0,
                  lb: float = -5.0, ub: float = 5.0) -> Tuple[np.ndarray, np.ndarray]:
    """Simulated Binary Crossover (SBX)."""
    dim = len(p1)
    c1, c2 = np.copy(p1), np.copy(p2)
    for i in range(dim):
        if np.random.rand() <= 0.9:
            y1, y2 = min(p1[i], p2[i]), max(p1[i], p2[i])
            if abs(y1 - y2) > 1.0e-10:
                rand = np.random.rand()
                beta = 1.0 + (2.0 * (y1 - lb) / (y2 - y1))
                alpha = 2.0 - (beta ** -(eta_c + 1.0))
                if rand <= (1.0 / alpha):
                    beta_q = (rand * alpha) ** (1.0 / (eta_c + 1.0))
                else:
                    beta_q = (1.0 / (2.0 - rand * alpha)) ** (1.0 / (eta_c + 1.0))
                c1[i] = 0.5 * ((y1 + y2) - beta_q * (y2 - y1))
                c2[i] = 0.5 * ((y1 + y2) + beta_q * (y2 - y1))
    return np.clip(c1, lb, ub), np.clip(c2, lb, ub)


def polynomial_mutation(ind: np.ndarray, eta_m: float = 20.0,
                        lb: float = -5.0, ub: float = 5.0) -> np.ndarray:
    """Polynomial Mutation."""
    dim = len(ind)
    p_mut = 1.0 / dim
    mut = np.copy(ind)
    for i in range(dim):
        if np.random.rand() < p_mut:
            y = ind[i]
            delta1 = (y - lb) / (ub - lb)
            delta2 = (ub - y) / (ub - lb)
            rand = np.random.rand()
            mut_pow = 1.0 / (eta_m + 1.0)
            if rand < 0.5:
                xy = 1.0 - delta1
                val = 2.0 * rand + (1.0 - 2.0 * rand) * (xy ** (eta_m + 1.0))
                delta_q = (val ** mut_pow) - 1.0
            else:
                xy = 1.0 - delta2
                val = 2.0 * (1.0 - rand) + 2.0 * (rand - 0.5) * (xy ** (eta_m + 1.0))
                delta_q = 1.0 - (val ** mut_pow)
            mut[i] = np.clip(y + delta_q * (ub - lb), lb, ub)
    return mut


class FlatNSGA2Optimizer:
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
            # Generate offspring through tournament, SBX, and PM
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

            # Combined pool for environmental selection
            if len(off_F) == 0:
                break
            pool_X = np.vstack([self.population, np.array(offspring[:len(off_F)])])
            pool_F = np.vstack([pop_F, np.array(off_F)])
            pool_cv = np.concatenate([pop_cv, np.array(off_cv)])
            
            # Select top n_pop based on feasibility-first ranking
            # Feasible vs infeasible sorting
            feas_indices = np.where(pool_cv <= 1.0e-9)[0]
            infeas_indices = np.where(pool_cv > 1.0e-9)[0]
            
            # Infeasible sorted by CV ascending
            infeas_sorted = infeas_indices[np.argsort(pool_cv[infeas_indices])]
            
            if len(feas_indices) >= self.n_pop:
                # Crowding distance among feasibles
                F_feas = pool_F[feas_indices]
                # Pick n_pop with best crowding
                selected = feas_indices[:self.n_pop]
            else:
                needed = self.n_pop - len(feas_indices)
                selected = np.concatenate([feas_indices, infeas_sorted[:needed]])

            self.population = pool_X[selected]
            pop_F = pool_F[selected]
            pop_cv = pool_cv[selected]

        total_time = time.time() - t0
        return {
            "algorithm": "NSGA-II",
            "evals": self.evals,
            "runtime": total_time,
            "feasible_solutions": len(self.archive.solutions),
            "is_feasible": bool(len(self.archive.solutions) > 0),
            "front": self.archive.get_front(),
            "archive": self.archive.solutions
        }
