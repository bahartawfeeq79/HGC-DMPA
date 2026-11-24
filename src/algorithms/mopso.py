"""
Baseline: Flat MOPSO (Coello Coello et al. 2004)
Multi-Objective Particle Swarm Optimization.
Particle velocities, personal best, and global archive leaders.
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

class FlatMOPSOOptimizer:
    def __init__(self, graph_model: SIoTGraphModel, src_node: int, dst_node: int,
                 t_dl: float, hop_budget: int = 100,
                 n_pop: int = 40, n_arch: int = 40, eval_budget: int = 3000,
                 lb: float = -5.0, ub: float = 5.0, w: float = 0.5, c1: float = 1.5, c2: float = 1.5,
                 seed: int = 42):
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
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.seed = seed
        
        np.random.seed(seed)
        self.dim = self.raw_graph.number_of_nodes()
        
        self.positions = np.random.uniform(self.lb, self.ub, size=(self.n_pop, self.dim))
        v_max = 0.2 * (self.ub - self.lb)
        self.velocities = np.random.uniform(-v_max, v_max, size=(self.n_pop, self.dim))
        
        self.pbest_pos = np.copy(self.positions)
        self.pbest_F = np.zeros((self.n_pop, 4))
        self.pbest_cv = np.zeros(self.n_pop)
        
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
            path = decode_flat_random_key_path(self.raw_graph, self.positions[i], self.src_node, self.dst_node, hop_budget=self.hop_budget)
            F, cv = self.evaluate_path(path)
            self.evals += 1
            
            self.pbest_F[i] = F
            self.pbest_cv[i] = cv
            self.archive.add(path, F, cv, key=self.positions[i])

        v_max = 0.2 * (self.ub - self.lb)
        
        while self.evals < self.eval_budget:
            for i in range(self.n_pop):
                if self.evals >= self.eval_budget:
                    break
                    
                # Pick leader from external archive
                if len(self.archive.solutions) > 0:
                    leader = np.array(self.archive.solutions[np.random.randint(len(self.archive.solutions))]['key'])
                    if leader is None or len(leader) != self.dim:
                        leader = self.pbest_pos[np.random.randint(self.n_pop)]
                else:
                    leader = self.pbest_pos[np.random.randint(self.n_pop)]
                    
                r1 = np.random.rand(self.dim)
                r2 = np.random.rand(self.dim)
                
                # Update velocity and position
                self.velocities[i] = (
                    self.w * self.velocities[i] +
                    self.c1 * r1 * (self.pbest_pos[i] - self.positions[i]) +
                    self.c2 * r2 * (leader - self.positions[i])
                )
                self.velocities[i] = np.clip(self.velocities[i], -v_max, v_max)
                self.positions[i] = np.clip(self.positions[i] + self.velocities[i], self.lb, self.ub)
                
                path = decode_flat_random_key_path(self.raw_graph, self.positions[i], self.src_node, self.dst_node, hop_budget=self.hop_budget)
                F, cv = self.evaluate_path(path)
                self.evals += 1
                
                self.archive.add(path, F, cv, key=self.positions[i])
                
                # Update personal best
                if dominates(F, cv, self.pbest_F[i], self.pbest_cv[i]):
                    self.pbest_pos[i] = np.copy(self.positions[i])
                    self.pbest_F[i] = F
                    self.pbest_cv[i] = cv

        total_time = time.time() - t0
        return {
            "algorithm": "MOPSO",
            "evals": self.evals,
            "runtime": total_time,
            "feasible_solutions": len(self.archive.solutions),
            "is_feasible": bool(len(self.archive.solutions) > 0),
            "front": self.archive.get_front(),
            "archive": self.archive.solutions
        }
