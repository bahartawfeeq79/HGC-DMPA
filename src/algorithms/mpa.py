"""
Baseline: Flat Standard Marine Predators Algorithm (MPA)
Operates on continuous N-dimensional decision vectors over the unpartitioned graph
with identical external archive capacity N_arch = 40 and eval_budget = 3,000.
(Faramarzi et al. 2020)
"""

import time
import math
import numpy as np
from typing import Dict, List, Tuple

try:
    import torch
    HAS_CUDA = torch.cuda.is_available()
except ImportError:
    HAS_CUDA = False

from src.graph.graph_model import SIoTGraphModel
from src.algorithms.flat_decoder import decode_flat_random_key_path
from src.objectives.trust import calculate_path_trust
from src.objectives.latency import calculate_path_latency
from src.objectives.energy import calculate_path_energy
from src.objectives.qos import calculate_path_qos
from src.objectives.constraints import calculate_constraint_violation
from src.evaluation.pareto import ParetoArchive, dominates
from src.algorithms.hgc_dmpa import levy_flight

class FlatMPAOptimizer:
    def __init__(self, graph_model: SIoTGraphModel, src_node: int, dst_node: int,
                 t_dl: float, hop_budget: int = 100,
                 n_pop: int = 40, n_arch: int = 40, eval_budget: int = 3000,
                 lb: float = -5.0, ub: float = 5.0, p_step: float = 0.5, beta_l: float = 1.5,
                 seed: int = 42, use_gpu: bool = True):
        self.use_gpu = use_gpu and HAS_CUDA
        self.device = torch.device('cuda') if self.use_gpu else None
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
        self.p_step = p_step
        self.beta_l = beta_l
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
        
        # Evaluate initial population
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
            tau = float(self.evals / self.eval_budget)
            cf = (1.0 - tau) ** (2.0 * tau)
            
            # Elite matrix
            elites = np.zeros_like(self.population)
            for i in range(self.n_pop):
                if len(self.archive.solutions) > 0 and np.random.rand() < 0.5:
                    sol = self.archive.solutions[np.random.randint(len(self.archive.solutions))]
                    if sol['key'] is not None and len(sol['key']) == self.dim:
                        elites[i] = np.array(sol['key'])
                    else:
                        elites[i] = self.population[np.random.randint(self.n_pop)]
                else:
                    elites[i] = self.population[np.random.randint(self.n_pop)]

            R_L = np.zeros((self.n_pop, self.dim))
            for i in range(self.n_pop):
                R_L[i] = 0.05 * levy_flight(beta=self.beta_l, dim=self.dim)

            if self.use_gpu:
                # GPU Tensor acceleration on NVIDIA GeForce RTX 2080
                X_t = torch.tensor(self.population, device=self.device, dtype=torch.float32)
                E_t = torch.tensor(elites, device=self.device, dtype=torch.float32)
                R_t = torch.rand((self.n_pop, self.dim), device=self.device, dtype=torch.float32)
                RB_t = torch.randn((self.n_pop, self.dim), device=self.device, dtype=torch.float32)
                RL_t = torch.tensor(R_L, device=self.device, dtype=torch.float32)

                if tau < 1.0 / 3.0:
                    step = self.p_step * R_t * RB_t * (E_t - RB_t * X_t)
                    new_pop_t = X_t + step
                elif tau < 2.0 / 3.0:
                    mid = self.n_pop // 2 + 1
                    step_prey = self.p_step * R_t[:mid] * RL_t[:mid] * (E_t[:mid] - RL_t[:mid] * X_t[:mid])
                    step_pred = self.p_step * cf * RB_t[mid:] * (RB_t[mid:] * E_t[mid:] - X_t[mid:])
                    new_pop_t = torch.cat([X_t[:mid] + step_prey, E_t[mid:] + step_pred], dim=0)
                else:
                    step = self.p_step * cf * RL_t * (RL_t * E_t - X_t)
                    new_pop_t = E_t + step
                new_pop_t = torch.clamp(new_pop_t, self.lb, self.ub)
                new_pop = new_pop_t.cpu().numpy()
            else:
                R = np.random.uniform(0.0, 1.0, size=(self.n_pop, self.dim))
                R_B = np.random.normal(0.0, 1.0, size=(self.n_pop, self.dim))
                new_pop = np.zeros_like(self.population)
                for i in range(self.n_pop):
                    X_i = self.population[i]
                    E_i = elites[i]
                    if tau < 1.0 / 3.0:
                        step = self.p_step * R[i] * R_B[i] * (E_i - R_B[i] * X_i)
                        new_pop[i] = X_i + step
                    elif tau < 2.0 / 3.0:
                        if i <= self.n_pop // 2:
                            step = self.p_step * R[i] * R_L[i] * (E_i - R_L[i] * X_i)
                            new_pop[i] = X_i + step
                        else:
                            step = self.p_step * cf * R_B[i] * (R_B[i] * E_i - X_i)
                            new_pop[i] = E_i + step
                    else:
                        step = self.p_step * cf * R_L[i] * (R_L[i] * E_i - X_i)
                        new_pop[i] = E_i + step
                new_pop = np.clip(new_pop, self.lb, self.ub)
            
            for i in range(self.n_pop):
                if self.evals >= self.eval_budget:
                    break
                path = decode_flat_random_key_path(self.raw_graph, new_pop[i], self.src_node, self.dst_node, hop_budget=self.hop_budget)
                F, cv = self.evaluate_path(path)
                self.evals += 1
                self.archive.add(path, F, cv, key=new_pop[i])
                
                # Feasibility-first replacement
                if dominates(F, cv, pop_F[i], pop_cv[i]):
                    self.population[i] = new_pop[i]
                    pop_F[i] = F
                    pop_cv[i] = cv
                    
        total_time = time.time() - t0
        return {
            "algorithm": "MPA",
            "evals": self.evals,
            "runtime": total_time,
            "feasible_solutions": len(self.archive.solutions),
            "is_feasible": bool(len(self.archive.solutions) > 0),
            "front": self.archive.get_front(),
            "archive": self.archive.solutions
        }
