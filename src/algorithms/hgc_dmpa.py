"""
Proposed Algorithm: HGC-DMPA
Hierarchical Graph-Clustered Decomposition–Marine Predators Algorithm
for Scalable Service Discovery in Ultra-Large Social IoT Networks.
(Algorithm 1 and Section IV in Tawfeeq 2026)
"""

import time
import math
import numpy as np
import networkx as nx
from typing import Dict, List, Tuple

try:
    import torch
    HAS_CUDA = torch.cuda.is_available()
except ImportError:
    HAS_CUDA = False

from src.graph.graph_model import SIoTGraphModel
from src.graph.coarsening import CoarsenedMacroGraph
from src.decoder.macro_decoder import decode_macro_corridor
from src.decoder.micro_decoder import decode_micro_path
from src.decoder.path_repair import apply_boundary_aware_tpr
from src.objectives.trust import calculate_path_trust
from src.objectives.latency import calculate_path_latency
from src.objectives.energy import calculate_path_energy
from src.objectives.qos import calculate_path_qos
from src.objectives.constraints import calculate_constraint_violation
from src.evaluation.pareto import (
    generate_das_dennis_4d_weights,
    compute_tchebycheff_scalar,
    ParetoArchive
)

def levy_flight(beta: float = 1.5, dim: int = 1) -> np.ndarray:
    """Generates Levy flight step vector via Mantegna's algorithm."""
    sigma_u = (
        math.gamma(1.0 + beta) * math.sin(math.pi * beta / 2.0) /
        (math.gamma((1.0 + beta) / 2.0) * beta * (2.0 ** ((beta - 1.0) / 2.0)))
    ) ** (1.0 / beta)
    u = np.random.normal(0.0, sigma_u, size=dim)
    v = np.random.normal(0.0, 1.0, size=dim)
    step = u / (np.abs(v) ** (1.0 / beta))
    return step


class HGCDMPAOptimizer:
    def __init__(self, graph_model: SIoTGraphModel, src_node: int, dst_node: int,
                 t_dl: float, hop_budget: int = 100,
                 n_pop: int = 40, n_arch: int = 40, t_nbr: int = 8, n_r: int = 2,
                 eval_budget: int = 3000, lb: float = -5.0, ub: float = 5.0,
                 p_step: float = 0.5, beta_l: float = 1.5, t_rep: float = 0.70,
                 p_fads_min: float = 0.1, p_fads_max: float = 0.4,
                 seed: int = 42, coarsening_weight: str = 'affinity_weight',
                 use_gpu: bool = True):
        self.use_gpu = use_gpu and HAS_CUDA
        self.device = torch.device('cuda') if self.use_gpu else None
        self.graph_model = graph_model
        self.raw_graph = graph_model.raw_graph
        self.adm_graph = graph_model.adm_graph
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
        self.p_step = p_step
        self.beta_l = beta_l
        self.t_rep = t_rep
        self.p_fads_min = p_fads_min
        self.p_fads_max = p_fads_max
        self.seed = seed
        self.coarsening_weight = coarsening_weight
        
        np.random.seed(seed)
        
        # 1. Tier 1 Offline Coarsening
        t0 = time.time()
        self.coarsened_mg = CoarsenedMacroGraph(
            self.adm_graph,
            physical_nodes_total=self.raw_graph.number_of_nodes(),
            weight_attr=self.coarsening_weight,
            random_state=seed,
            model_ref=self.graph_model
        )
        self.coarsening_time = time.time() - t0
        self.d_eff = self.coarsened_mg.d_eff
        
        # Check source and destination communities
        self.c_src = self.coarsened_mg.partition.get(src_node, 0)
        self.c_dst = self.coarsened_mg.partition.get(dst_node, 0)
        
        # 2. Reference weight vectors and neighborhoods
        self.weights = generate_das_dennis_4d_weights(n_weights=self.n_pop, seed=seed)
        # Compute weight neighborhoods B(i)
        self.neighborhoods = []
        for i in range(self.n_pop):
            dists = np.linalg.norm(self.weights - self.weights[i], axis=1)
            nbr_indices = np.argsort(dists)[:self.t_nbr]
            self.neighborhoods.append(nbr_indices)

        # 3. Evolution state
        self.population = np.random.uniform(self.lb, self.ub, size=(self.n_pop, self.d_eff))
        self.paths = [None] * self.n_pop
        self.objectives = np.zeros((self.n_pop, 4))
        self.cvs = np.zeros(self.n_pop)
        
        # Ideal point z* for augmented Tchebycheff
        self.z_ideal = np.ones(4) * float("inf")
        self.z_nadir = np.zeros(4)
        
        # External Pareto archive
        self.archive = ParetoArchive(capacity=self.n_arch)
        self.evals = 0
        
        # Repair statistics
        self.repairs_attempted = 0
        self.repairs_accepted = 0

    def evaluate_solution(self, path: List[int]) -> Tuple[np.ndarray, float, dict]:
        """Evaluates canonical objective vector F(P) and constraint violation CV(P)."""
        trust = calculate_path_trust(path, self.raw_graph)
        lat = calculate_path_latency(path, self.graph_model)
        ene = calculate_path_energy(path, self.graph_model)
        qos = calculate_path_qos(path, self.raw_graph)
        
        # Canonical minimization objectives: [1 - Trust, Lat, Ene, 1 - QoS]
        F = np.array([1.0 - trust, lat, ene, 1.0 - qos], dtype=float)
        
        cv, v_dict = calculate_constraint_violation(
            path, self.raw_graph, latency=lat, t_dl=self.t_dl,
            bat_min=self.graph_model.bat_min, t_min=self.graph_model.t_min,
            hop_budget=self.hop_budget
        )
        return F, cv, v_dict

    def decode_key(self, key_vector: np.ndarray) -> List[int]:
        """Executes two-tier biased decoding from key vector to simple micro-path."""
        K = self.coarsened_mg.K
        x_comm = key_vector[:K]
        x_gw = key_vector[K:]
        
        corridor = decode_macro_corridor(
            self.coarsened_mg.macro_graph, x_comm, self.c_src, self.c_dst,
            h_m_budget=40, mu=1.5, nu=1.0, mean_macro_lat=self.coarsened_mg.mean_macro_latency
        )
        path = decode_micro_path(
            self.coarsened_mg, x_gw, corridor, self.src_node, self.dst_node,
            mu=1.5, nu=1.0, mean_lat_ref=self.coarsened_mg.mean_macro_latency
        )
        return path

    def run(self) -> Dict:
        """Executes complete HGC-DMPA optimization loop under eval_budget."""
        t_start = time.time()
        
        # Initial Population Evaluation
        for i in range(self.n_pop):
            path = self.decode_key(self.population[i])
            F, cv, _ = self.evaluate_solution(path)
            self.evals += 1
            
            self.paths[i] = path
            self.objectives[i] = F
            self.cvs[i] = cv
            
            # Update ideal and nadir bounds
            self.z_ideal = np.minimum(self.z_ideal, F)
            self.z_nadir = np.maximum(self.z_nadir, F)
            
            # Archive update
            self.archive.add(path, F, cv, key=self.population[i])
            
        # Main Evolutionary Loop
        while self.evals < self.eval_budget:
            # Progress fraction tau and smooth convergence factor CF
            tau = float(self.evals / self.eval_budget)
            cf = (1.0 - tau) ** (2.0 * tau)
            
            # Compute normalized key diversity D_t across population
            pop_std = np.std(self.population, axis=0)
            d_t = float(np.mean(pop_std) / ((self.ub - self.lb) / 4.0))
            p_fads = self.p_fads_min + (self.p_fads_max - self.p_fads_min) * (1.0 - np.clip(d_t, 0.0, 1.0))

            # Select predator guide E_i for each agent
            elites = np.zeros_like(self.population)
            for i in range(self.n_pop):
                if np.random.rand() < 0.9 or len(self.archive.solutions) == 0:
                    # Draw from weight neighborhood
                    nbr_idx = np.random.choice(self.neighborhoods[i])
                    elites[i] = self.population[nbr_idx]
                else:
                    # Draw from external archive
                    arch_sol = self.archive.solutions[np.random.randint(len(self.archive.solutions))]
                    if arch_sol['key'] is not None and len(arch_sol['key']) == self.d_eff:
                        elites[i] = np.array(arch_sol['key'])
                    else:
                        elites[i] = self.population[i]

            # Generate candidate vectors X' via three-phase MPA equations (Equation 17)
            R = np.random.uniform(0.0, 1.0, size=(self.n_pop, self.d_eff))
            R_B = np.random.normal(0.0, 1.0, size=(self.n_pop, self.d_eff))
            
            # Levy vectors
            R_L = np.zeros((self.n_pop, self.d_eff))
            for i in range(self.n_pop):
                R_L[i] = 0.05 * levy_flight(beta=self.beta_l, dim=self.d_eff)

            if self.use_gpu:
                # GPU Tensor acceleration on NVIDIA GeForce RTX 2080
                X_t = torch.tensor(self.population, device=self.device, dtype=torch.float32)
                E_t = torch.tensor(elites, device=self.device, dtype=torch.float32)
                R_t = torch.rand((self.n_pop, self.d_eff), device=self.device, dtype=torch.float32)
                RB_t = torch.randn((self.n_pop, self.d_eff), device=self.device, dtype=torch.float32)
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
                new_pop = np.zeros_like(self.population)
                for i in range(self.n_pop):
                    X_i = self.population[i]
                    E_i = elites[i]
                    
                    if tau < 1.0 / 3.0:
                        # Phase 1: High velocity ratio - Brownian exploration
                        step = self.p_step * R[i] * R_B[i] * (E_i - R_B[i] * X_i)
                        new_pop[i] = X_i + step
                    elif tau < 2.0 / 3.0:
                        # Phase 2: Unit velocity ratio
                        if i <= self.n_pop // 2:
                            # Prey moves in Levy flight
                            step = self.p_step * R[i] * R_L[i] * (E_i - R_L[i] * X_i)
                            new_pop[i] = X_i + step
                        else:
                            # Predator moves in Brownian motion
                            step = self.p_step * cf * R_B[i] * (R_B[i] * E_i - X_i)
                            new_pop[i] = E_i + step
                    else:
                        # Phase 3: Low velocity ratio - Levy exploitation
                        step = self.p_step * cf * R_L[i] * (R_L[i] * E_i - X_i)
                        new_pop[i] = E_i + step
                        
                # Clip to [lb, ub]
                new_pop = np.clip(new_pop, self.lb, self.ub)

            # Evaluate each candidate and apply BA-TPR
            for i in range(self.n_pop):
                if self.evals >= self.eval_budget:
                    break
                    
                cand_key = new_pop[i]
                cand_path = self.decode_key(cand_key)
                cand_F, cand_cv, _ = self.evaluate_solution(cand_path)
                self.evals += 1
                
                self.z_ideal = np.minimum(self.z_ideal, cand_F)
                self.z_nadir = np.maximum(self.z_nadir, cand_F)
                self.archive.add(cand_path, cand_F, cand_cv, key=cand_key)

                # Lamarckian BA-TPR Repair (Section IV-B)
                rep_path, rep_key, applied, _ = apply_boundary_aware_tpr(
                    cand_path, cand_key, self.coarsened_mg, self.graph_model,
                    self.src_node, self.dst_node, t_rep=self.t_rep, lb=self.lb, ub=self.ub
                )
                
                if applied and self.evals < self.eval_budget:
                    self.repairs_attempted += 1
                    rep_F, rep_cv, _ = self.evaluate_solution(rep_path)
                    self.evals += 1
                    
                    self.z_ideal = np.minimum(self.z_ideal, rep_F)
                    self.z_nadir = np.maximum(self.z_nadir, rep_F)
                    self.archive.add(rep_path, rep_F, rep_cv, key=rep_key)
                    
                    # Acceptance: if repair is strictly more feasible or higher trust
                    if rep_cv < cand_cv or (rep_cv <= cand_cv and rep_F[0] < cand_F[0]):
                        cand_path = rep_path
                        cand_key = rep_key
                        cand_F = rep_F
                        cand_cv = rep_cv
                        self.repairs_accepted += 1
                        
                # Update up to n_r = 2 neighboring subproblems using augmented Tchebycheff
                norm_denom = np.maximum(1.0e-6, self.z_nadir - self.z_ideal)
                f_norm = (cand_F - self.z_ideal) / norm_denom
                
                updated_count = 0
                shuffled_nbrs = np.random.permutation(self.neighborhoods[i])
                for j in shuffled_nbrs:
                    if updated_count >= self.n_r:
                        break
                        
                    old_cv = self.cvs[j]
                    old_f_norm = (self.objectives[j] - self.z_ideal) / norm_denom
                    
                    tch_cand = compute_tchebycheff_scalar(f_norm, self.weights[j], np.zeros(4))
                    tch_old = compute_tchebycheff_scalar(old_f_norm, self.weights[j], np.zeros(4))
                    
                    replace = False
                    if cand_cv < old_cv - 1.0e-9:
                        replace = True
                    elif abs(cand_cv - old_cv) < 1.0e-9 and tch_cand < tch_old:
                        replace = True
                        
                    if replace:
                        self.population[j] = cand_key
                        self.paths[j] = cand_path
                        self.objectives[j] = cand_F
                        self.cvs[j] = cand_cv
                        updated_count += 1

            # Diversity-adaptive FADs perturbation
            if np.random.rand() < p_fads:
                # Perturb a random fraction of agents
                for i in range(self.n_pop):
                    if np.random.rand() < 0.2:
                        self.population[i] = np.clip(
                            self.population[i] + (self.ub - self.lb) * 0.1 * np.random.normal(0, 1, size=self.d_eff),
                            self.lb, self.ub
                        )

        total_runtime = time.time() - t_start
        feasible_count = len(self.archive.solutions)
        
        return {
            "algorithm": "HGC-DMPA",
            "evals": self.evals,
            "runtime": total_runtime,
            "coarsening_time": self.coarsening_time,
            "feasible_solutions": feasible_count,
            "is_feasible": bool(feasible_count > 0),
            "front": self.archive.get_front(),
            "archive": self.archive.solutions,
            "d_eff": self.d_eff,
            "compression_pct": self.coarsened_mg.compression_pct,
            "repairs_attempted": self.repairs_attempted,
            "repairs_accepted": self.repairs_accepted
        }
