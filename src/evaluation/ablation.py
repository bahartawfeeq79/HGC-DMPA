"""
Component Ablation Study Engine:
Evaluates the five architectural variants from Tawfeeq 2026 Table VI & Section VI-B:
1. Full HGC-DMPA (Complete framework)
2. w/o Decomposition (HGC-MPA: crowding-distance archive)
3. Unweighted Louvain Coarsening (gamma = 0, purely topological modularity)
4. w/o BA-TPR (No Lamarckian Boundary-Aware Path Repair)
5. w/o Coarsening (Flat Decomp.-MPA over unpartitioned N-dimensional keys)
Evaluated across RGG-1000, RGG-2000, and RGG-5000 over 30 independent runs.
"""

import time
import numpy as np
from typing import Dict, List

from src.graph.graph_model import SIoTGraphModel
from src.algorithms.hgc_dmpa import HGCDMPAOptimizer, levy_flight
from src.algorithms.flat_decoder import decode_flat_random_key_path
from src.decoder.path_repair import apply_boundary_aware_tpr
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

class AblationVariantRunner:
    def __init__(self, variant_name: str, graph_model: SIoTGraphModel,
                 src_node: int, dst_node: int, t_dl: float, hop_budget: int = 100,
                 eval_budget: int = 3000, seed: int = 42):
        self.variant_name = variant_name
        self.graph_model = graph_model
        self.src_node = src_node
        self.dst_node = dst_node
        self.t_dl = t_dl
        self.hop_budget = hop_budget
        self.eval_budget = eval_budget
        self.seed = seed

    def run(self) -> Dict:
        if self.variant_name == "Full HGC-DMPA":
            opt = HGCDMPAOptimizer(
                self.graph_model, self.src_node, self.dst_node,
                t_dl=self.t_dl, hop_budget=self.hop_budget,
                eval_budget=self.eval_budget, seed=self.seed,
                coarsening_weight='affinity_weight'
            )
            res = opt.run()
            res["variant"] = self.variant_name
            return res

        elif self.variant_name == "Unweighted Louvain Coarsening":
            # Coarsening with unweighted topological modularity
            opt = HGCDMPAOptimizer(
                self.graph_model, self.src_node, self.dst_node,
                t_dl=self.t_dl, hop_budget=self.hop_budget,
                eval_budget=self.eval_budget, seed=self.seed,
                coarsening_weight='unweighted'
            )
            res = opt.run()
            res["variant"] = self.variant_name
            return res

        elif self.variant_name == "w/o BA-TPR":
            # HGC-DMPA without path repair
            return self._run_hgc_dmpa_no_repair()

        elif self.variant_name == "w/o Decomposition (HGC-MPA)":
            # HGC-MPA without Tchebycheff decomposition (crowding-distance archive)
            return self._run_hgc_mpa_no_decomp()

        elif self.variant_name == "w/o Coarsening (Flat Decomp.-MPA)":
            # Flat decomposition MPA without graph coarsening
            return self._run_flat_decomp_mpa()

        else:
            raise ValueError(f"Unknown ablation variant: {self.variant_name}")

    def _run_hgc_dmpa_no_repair(self) -> Dict:
        """HGC-DMPA with BA-TPR disabled."""
        opt = HGCDMPAOptimizer(
            self.graph_model, self.src_node, self.dst_node,
            t_dl=self.t_dl, hop_budget=self.hop_budget,
            eval_budget=self.eval_budget, seed=self.seed
        )
        t_start = time.time()
        for i in range(opt.n_pop):
            path = opt.decode_key(opt.population[i])
            F, cv, _ = opt.evaluate_solution(path)
            opt.evals += 1
            opt.paths[i] = path
            opt.objectives[i] = F
            opt.cvs[i] = cv
            opt.z_ideal = np.minimum(opt.z_ideal, F)
            opt.z_nadir = np.maximum(opt.z_nadir, F)
            opt.archive.add(path, F, cv, key=opt.population[i])

        while opt.evals < opt.eval_budget:
            tau = float(opt.evals / opt.eval_budget)
            cf = (1.0 - tau) ** (2.0 * tau)
            
            elites = np.zeros_like(opt.population)
            for i in range(opt.n_pop):
                nbr_idx = np.random.choice(opt.neighborhoods[i])
                elites[i] = opt.population[nbr_idx]
                
            R = np.random.uniform(0.0, 1.0, size=(opt.n_pop, opt.d_eff))
            R_B = np.random.normal(0.0, 1.0, size=(opt.n_pop, opt.d_eff))
            R_L = np.zeros((opt.n_pop, opt.d_eff))
            for i in range(opt.n_pop):
                R_L[i] = 0.05 * levy_flight(beta=opt.beta_l, dim=opt.d_eff)

            new_pop = np.zeros_like(opt.population)
            for i in range(opt.n_pop):
                if tau < 1.0 / 3.0:
                    new_pop[i] = opt.population[i] + opt.p_step * R[i] * R_B[i] * (elites[i] - R_B[i] * opt.population[i])
                elif tau < 2.0 / 3.0:
                    if i <= opt.n_pop // 2:
                        new_pop[i] = opt.population[i] + opt.p_step * R[i] * R_L[i] * (elites[i] - R_L[i] * opt.population[i])
                    else:
                        new_pop[i] = elites[i] + opt.p_step * cf * R_B[i] * (R_B[i] * elites[i] - opt.population[i])
                else:
                    new_pop[i] = elites[i] + opt.p_step * cf * R_L[i] * (R_L[i] * elites[i] - opt.population[i])
            new_pop = np.clip(new_pop, opt.lb, opt.ub)

            for i in range(opt.n_pop):
                if opt.evals >= opt.eval_budget:
                    break
                cand_path = opt.decode_key(new_pop[i])
                cand_F, cand_cv, _ = opt.evaluate_solution(cand_path)
                opt.evals += 1
                opt.z_ideal = np.minimum(opt.z_ideal, cand_F)
                opt.z_nadir = np.maximum(opt.z_nadir, cand_F)
                opt.archive.add(cand_path, cand_F, cand_cv, key=new_pop[i])

                # Update neighbors without BA-TPR
                norm_denom = np.maximum(1.0e-6, opt.z_nadir - opt.z_ideal)
                f_norm = (cand_F - opt.z_ideal) / norm_denom
                updated = 0
                for j in np.random.permutation(opt.neighborhoods[i]):
                    if updated >= opt.n_r: break
                    old_f_norm = (opt.objectives[j] - opt.z_ideal) / norm_denom
                    tch_cand = compute_tchebycheff_scalar(f_norm, opt.weights[j], np.zeros(4))
                    tch_old = compute_tchebycheff_scalar(old_f_norm, opt.weights[j], np.zeros(4))
                    if cand_cv < opt.cvs[j] - 1.0e-9 or (abs(cand_cv - opt.cvs[j]) < 1.0e-9 and tch_cand < tch_old):
                        opt.population[j] = new_pop[i]
                        opt.paths[j] = cand_path
                        opt.objectives[j] = cand_F
                        opt.cvs[j] = cand_cv
                        updated += 1

        return {
            "variant": self.variant_name,
            "evals": opt.evals,
            "runtime": time.time() - t_start,
            "feasible_solutions": len(opt.archive.solutions),
            "is_feasible": bool(len(opt.archive.solutions) > 0),
            "front": opt.archive.get_front(),
            "archive": opt.archive.solutions
        }

    def _run_hgc_mpa_no_decomp(self) -> Dict:
        """HGC-MPA without Tchebycheff decomposition (crowding distance archive)."""
        opt = HGCDMPAOptimizer(
            self.graph_model, self.src_node, self.dst_node,
            t_dl=self.t_dl, hop_budget=self.hop_budget,
            eval_budget=self.eval_budget, seed=self.seed
        )
        t_start = time.time()
        for i in range(opt.n_pop):
            path = opt.decode_key(opt.population[i])
            F, cv, _ = opt.evaluate_solution(path)
            opt.evals += 1
            opt.paths[i] = path
            opt.objectives[i] = F
            opt.cvs[i] = cv
            opt.archive.add(path, F, cv, key=opt.population[i])

        while opt.evals < opt.eval_budget:
            tau = float(opt.evals / opt.eval_budget)
            cf = (1.0 - tau) ** (2.0 * tau)
            
            # Guides drawn purely from archive or population without weight vector guidance
            elites = np.zeros_like(opt.population)
            for i in range(opt.n_pop):
                if len(opt.archive.solutions) > 0:
                    sol = opt.archive.solutions[np.random.randint(len(opt.archive.solutions))]
                    elites[i] = np.array(sol['key']) if sol['key'] is not None else opt.population[i]
                else:
                    elites[i] = opt.population[np.random.randint(opt.n_pop)]

            R = np.random.uniform(0.0, 1.0, size=(opt.n_pop, opt.d_eff))
            R_B = np.random.normal(0.0, 1.0, size=(opt.n_pop, opt.d_eff))
            R_L = np.zeros((opt.n_pop, opt.d_eff))
            for i in range(opt.n_pop):
                R_L[i] = 0.05 * levy_flight(beta=opt.beta_l, dim=opt.d_eff)

            new_pop = np.zeros_like(opt.population)
            for i in range(opt.n_pop):
                if tau < 1.0 / 3.0:
                    new_pop[i] = opt.population[i] + opt.p_step * R[i] * R_B[i] * (elites[i] - R_B[i] * opt.population[i])
                elif tau < 2.0 / 3.0:
                    if i <= opt.n_pop // 2:
                        new_pop[i] = opt.population[i] + opt.p_step * R[i] * R_L[i] * (elites[i] - R_L[i] * opt.population[i])
                    else:
                        new_pop[i] = elites[i] + opt.p_step * cf * R_B[i] * (R_B[i] * elites[i] - opt.population[i])
                else:
                    new_pop[i] = elites[i] + opt.p_step * cf * R_L[i] * (R_L[i] * elites[i] - opt.population[i])
            new_pop = np.clip(new_pop, opt.lb, opt.ub)

            for i in range(opt.n_pop):
                if opt.evals >= opt.eval_budget:
                    break
                cand_path = opt.decode_key(new_pop[i])
                cand_F, cand_cv, _ = opt.evaluate_solution(cand_path)
                opt.evals += 1
                opt.archive.add(cand_path, cand_F, cand_cv, key=new_pop[i])

                # Lamarckian repair
                rep_path, rep_key, applied, _ = apply_boundary_aware_tpr(
                    cand_path, new_pop[i], opt.coarsened_mg, opt.graph_model,
                    opt.src_node, opt.dst_node, t_rep=opt.t_rep, lb=opt.lb, ub=opt.ub
                )
                if applied and opt.evals < opt.eval_budget:
                    rep_F, rep_cv, _ = opt.evaluate_solution(rep_path)
                    opt.evals += 1
                    opt.archive.add(rep_path, rep_F, rep_cv, key=rep_key)
                    if rep_cv < cand_cv or (rep_cv <= cand_cv and rep_F[0] < cand_F[0]):
                        cand_path = rep_path
                        cand_F = rep_F
                        cand_cv = rep_cv
                        new_pop[i] = rep_key

                # Replacement purely by single-objective dominance
                if cand_cv < opt.cvs[i] or (cand_cv == opt.cvs[i] and np.sum(cand_F) < np.sum(opt.objectives[i])):
                    opt.population[i] = new_pop[i]
                    opt.objectives[i] = cand_F
                    opt.cvs[i] = cand_cv

        return {
            "variant": self.variant_name,
            "evals": opt.evals,
            "runtime": time.time() - t_start,
            "feasible_solutions": len(opt.archive.solutions),
            "is_feasible": bool(len(opt.archive.solutions) > 0),
            "front": opt.archive.get_front(),
            "archive": opt.archive.solutions
        }

    def _run_flat_decomp_mpa(self) -> Dict:
        """Flat decomposition MPA without coarsening (N dimensions)."""
        dim = self.graph_model.raw_graph.number_of_nodes()
        weights = generate_das_dennis_4d_weights(n_weights=40, seed=self.seed)
        neighborhoods = [np.argsort(np.linalg.norm(weights - weights[i], axis=1))[:8] for i in range(40)]
        
        pop = np.random.uniform(-5.0, 5.0, size=(40, dim))
        pop_F = np.zeros((40, 4))
        pop_cv = np.zeros(40)
        
        z_ideal = np.ones(4) * float("inf")
        z_nadir = np.zeros(4)
        archive = ParetoArchive(capacity=40)
        evals = 0
        t_start = time.time()
        
        for i in range(40):
            path = decode_flat_random_key_path(self.graph_model.raw_graph, pop[i], self.src_node, self.dst_node, hop_budget=self.hop_budget)
            F = np.array([
                1.0 - calculate_path_trust(path, self.graph_model.raw_graph),
                calculate_path_latency(path, self.graph_model),
                calculate_path_energy(path, self.graph_model),
                1.0 - calculate_path_qos(path, self.graph_model.raw_graph)
            ])
            cv, _ = calculate_constraint_violation(
                path, self.graph_model.raw_graph, latency=F[1], t_dl=self.t_dl,
                bat_min=self.graph_model.bat_min, t_min=self.graph_model.t_min,
                hop_budget=self.hop_budget
            )
            evals += 1
            pop_F[i] = F
            pop_cv[i] = cv
            z_ideal = np.minimum(z_ideal, F)
            z_nadir = np.maximum(z_nadir, F)
            archive.add(path, F, cv, key=pop[i])

        while evals < self.eval_budget:
            tau = float(evals / self.eval_budget)
            cf = (1.0 - tau) ** (2.0 * tau)
            
            elites = np.zeros_like(pop)
            for i in range(40):
                nbr_idx = np.random.choice(neighborhoods[i])
                elites[i] = pop[nbr_idx]
                
            R = np.random.uniform(0.0, 1.0, size=(40, dim))
            R_B = np.random.normal(0.0, 1.0, size=(40, dim))
            R_L = np.zeros((40, dim))
            for i in range(40):
                R_L[i] = 0.05 * levy_flight(beta=1.5, dim=dim)

            new_pop = np.zeros_like(pop)
            for i in range(40):
                if tau < 1.0 / 3.0:
                    new_pop[i] = pop[i] + 0.5 * R[i] * R_B[i] * (elites[i] - R_B[i] * pop[i])
                elif tau < 2.0 / 3.0:
                    if i <= 20:
                        new_pop[i] = pop[i] + 0.5 * R[i] * R_L[i] * (elites[i] - R_L[i] * pop[i])
                    else:
                        new_pop[i] = elites[i] + 0.5 * cf * R_B[i] * (R_B[i] * elites[i] - pop[i])
                else:
                    new_pop[i] = elites[i] + 0.5 * cf * R_L[i] * (R_L[i] * elites[i] - pop[i])
            new_pop = np.clip(new_pop, -5.0, 5.0)

            for i in range(40):
                if evals >= self.eval_budget:
                    break
                path = decode_flat_random_key_path(self.graph_model.raw_graph, new_pop[i], self.src_node, self.dst_node, hop_budget=self.hop_budget)
                F = np.array([
                    1.0 - calculate_path_trust(path, self.graph_model.raw_graph),
                    calculate_path_latency(path, self.graph_model),
                    calculate_path_energy(path, self.graph_model),
                    1.0 - calculate_path_qos(path, self.graph_model.raw_graph)
                ])
                cv, _ = calculate_constraint_violation(
                    path, self.graph_model.raw_graph, latency=F[1], t_dl=self.t_dl,
                    bat_min=self.graph_model.bat_min, t_min=self.graph_model.t_min,
                    hop_budget=self.hop_budget
                )
                evals += 1
                z_ideal = np.minimum(z_ideal, F)
                z_nadir = np.maximum(z_nadir, F)
                archive.add(path, F, cv, key=new_pop[i])

                norm_denom = np.maximum(1.0e-6, z_nadir - z_ideal)
                f_norm = (F - z_ideal) / norm_denom
                updated = 0
                for j in np.random.permutation(neighborhoods[i]):
                    if updated >= 2: break
                    old_f_norm = (pop_F[j] - z_ideal) / norm_denom
                    tch_cand = compute_tchebycheff_scalar(f_norm, weights[j], np.zeros(4))
                    tch_old = compute_tchebycheff_scalar(old_f_norm, weights[j], np.zeros(4))
                    if cv < pop_cv[j] - 1.0e-9 or (abs(cv - pop_cv[j]) < 1.0e-9 and tch_cand < tch_old):
                        pop[j] = new_pop[i]
                        pop_F[j] = F
                        pop_cv[j] = cv
                        updated += 1

        return {
            "variant": self.variant_name,
            "evals": evals,
            "runtime": time.time() - t_start,
            "feasible_solutions": len(archive.solutions),
            "is_feasible": bool(len(archive.solutions) > 0),
            "front": archive.get_front(),
            "archive": archive.solutions
        }
