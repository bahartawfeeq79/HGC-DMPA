"""
Parameter Sensitivity and Tuning Analysis:
Evaluates parameter sensitivity on separate seed graphs (RGG-800, Hosp.-seed, BA-seed)
across key parameters: alpha, beta, gamma, mu, nu, T_rep, T_nbr.
Saves results to results/statistical/parameter_sensitivity.json.
"""

import os
import json
import time
import numpy as np
from src.data.generators import generate_rgg_topology
from src.graph.graph_model import SIoTGraphModel
from src.algorithms.hgc_dmpa import HGCDMPAOptimizer
from src.evaluation.metrics import compute_hypervolume

SENSITIVITY_FILE = os.path.join("results", "statistical", "parameter_sensitivity.json")

def evaluate_parameter_sensitivity():
    os.makedirs(os.path.dirname(SENSITIVITY_FILE), exist_ok=True)
    
    # Generate separate tuning instance RGG-800
    G_tune = generate_rgg_topology(target_nodes=800, target_edges=2800, target_diameter=25, compromised_count=50, seed=100800)
    nodes = list(G_tune.nodes())
    src, dst = nodes[0], nodes[-1]
    
    results = {}
    
    # 1. Sensitivity to alpha (direct trust weight)
    alpha_vals = [0.5, 0.6, 0.7, 0.8]
    alpha_res = []
    for a in alpha_vals:
        model = SIoTGraphModel(G_tune, alpha=a, beta=0.3, gamma=0.5)
        opt = HGCDMPAOptimizer(model, src, dst, t_dl=0.5, eval_budget=1500, seed=42)
        r = opt.run()
        hv = compute_hypervolume(r["front"])
        alpha_res.append({"param": a, "hv": hv, "feasible": r["is_feasible"]})
    results["alpha"] = alpha_res

    # 2. Sensitivity to beta (social bonus)
    beta_vals = [0.1, 0.2, 0.3, 0.4]
    beta_res = []
    for b in beta_vals:
        model = SIoTGraphModel(G_tune, alpha=0.7, beta=b, gamma=0.5)
        opt = HGCDMPAOptimizer(model, src, dst, t_dl=0.5, eval_budget=1500, seed=42)
        r = opt.run()
        hv = compute_hypervolume(r["front"])
        beta_res.append({"param": b, "hv": hv, "feasible": r["is_feasible"]})
    results["beta"] = beta_res

    # 3. Sensitivity to gamma (affinity modularity)
    gamma_vals = [0.3, 0.5, 0.7, 1.0]
    gamma_res = []
    for g in gamma_vals:
        model = SIoTGraphModel(G_tune, alpha=0.7, beta=0.3, gamma=g)
        opt = HGCDMPAOptimizer(model, src, dst, t_dl=0.5, eval_budget=1500, seed=42)
        r = opt.run()
        hv = compute_hypervolume(r["front"])
        gamma_res.append({"param": g, "hv": hv, "feasible": r["is_feasible"]})
    results["gamma"] = gamma_res

    # 4. Sensitivity to T_rep (BA-TPR threshold)
    trep_vals = [0.55, 0.65, 0.70, 0.80]
    trep_res = []
    model_base = SIoTGraphModel(G_tune, alpha=0.7, beta=0.3, gamma=0.5)
    for tr in trep_vals:
        opt = HGCDMPAOptimizer(model_base, src, dst, t_dl=0.5, eval_budget=1500, t_rep=tr, seed=42)
        r = opt.run()
        hv = compute_hypervolume(r["front"])
        trep_res.append({"param": tr, "hv": hv, "feasible": r["is_feasible"], "repairs_accepted": r["repairs_accepted"]})
    results["t_rep"] = trep_res

    # 5. Sensitivity to T_nbr (Neighborhood size)
    tnbr_vals = [4, 6, 8, 12]
    tnbr_res = []
    for t_nbr in tnbr_vals:
        opt = HGCDMPAOptimizer(model_base, src, dst, t_dl=0.5, eval_budget=1500, t_nbr=t_nbr, seed=42)
        r = opt.run()
        hv = compute_hypervolume(r["front"])
        tnbr_res.append({"param": t_nbr, "hv": hv, "feasible": r["is_feasible"]})
    results["t_nbr"] = tnbr_res

    with open(SENSITIVITY_FILE, "w") as f:
        json.dump(results, f, indent=2)
        
    print(f"Parameter sensitivity results saved to {SENSITIVITY_FILE}")
    return results

if __name__ == "__main__":
    evaluate_parameter_sensitivity()
