"""
Baseline: Flat Zebra Optimization Algorithm (ZOA) (Trojovska et al. 2022)
Nature-inspired metaheuristic simulating zebra foraging and defense behaviors,
adapted for constrained multi-objective SIoT service discovery.
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

class FlatZOAOptimizer:
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
            # Phase 1: Foraging - pioneer zebra guidance
            for i in range(self.n_pop):
                if self.evals >= self.eval_budget:
                    break
                    
                if len(self.archive.solutions) > 0:
                    best_sol = self.archive.solutions[np.random.randint(len(self.archive.solutions))]['key']
                    pioneer = np.array(best_sol) if best_sol is not None and len(best_sol) == self.dim else self.population[0]
                else:
                    pioneer = self.population[np.argmin(pop_cv)]
                    
                r = np.random.rand(self.dim)
                I = np.random.choice([1, 2])
                new_x = self.population[i] + r * (pioneer - I * self.population[i])
                new_x = np.clip(new_x, self.lb, self.ub)
                
                path = decode_flat_random_key_path(self.raw_graph, new_x, self.src_node, self.dst_node, hop_budget=self.hop_budget)
                F, cv = self.evaluate_path(path)
                self.evals += 1
                self.archive.add(path, F, cv, key=new_x)
                
                if dominates(F, cv, pop_F[i], pop_cv[i]):
                    self.population[i] = new_x
                    pop_F[i] = F
                    pop_cv[i] = cv

            # Phase 2: Defense against predator (attack or defense)
            for i in range(self.n_pop):
                if self.evals >= self.eval_budget:
                    break
                    
                Ps = 0.5
                if np.random.rand() < Ps:
                    # Escape / zig-zag movement
                    r = np.random.rand(self.dim)
                    new_x = self.population[i] + (2.0 * r - 1.0) * (1.0 - self.evals / self.eval_budget) * self.population[i]
                else:
                    # Defense
                    other_idx = np.random.randint(self.n_pop)
                    new_x = self.population[i] + np.random.rand(self.dim) * (self.population[other_idx] - self.population[i])
                    
                new_x = np.clip(new_x, self.lb, self.ub)
                path = decode_flat_random_key_path(self.raw_graph, new_x, self.src_node, self.dst_node, hop_budget=self.hop_budget)
                F, cv = self.evaluate_path(path)
                self.evals += 1
                self.archive.add(path, F, cv, key=new_x)
                
                if dominates(F, cv, pop_F[i], pop_cv[i]):
                    self.population[i] = new_x
                    pop_F[i] = F
                    pop_cv[i] = cv

        total_time = time.time() - t0
        return {
            "algorithm": "ZOA",
            "evals": self.evals,
            "runtime": total_time,
            "feasible_solutions": len(self.archive.solutions),
            "is_feasible": bool(len(self.archive.solutions) > 0),
            "front": self.archive.get_front(),
            "archive": self.archive.solutions
        }
