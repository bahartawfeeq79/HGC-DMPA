"""
Result Aggregator and Statistical Analysis Engine:
1. Gathers all raw run records from results/raw/ and results/raw/ablation/
2. Builds empirical reference Pareto front P^* per benchmark instance
3. Calculates Hypervolume (HV), IGD, Feasibility Rate, and Route Quality metrics
4. Performs Holm-corrected Wilcoxon tests, Vargha-Delaney A12, and Friedman rankings
5. Outputs structured JSON results for automated publication table generation.
"""

import os
import glob
import json
import logging
import numpy as np
import pandas as pd

from src.evaluation.metrics import compute_hypervolume, compute_igd
from src.evaluation.statistical_tests import (
    compare_algorithms_wilcoxon,
    compare_against_deterministic,
    holm_bonferroni_correction,
    compute_friedman_test
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

RAW_DIR = os.path.join("results", "raw")
ABLATION_RAW_DIR = os.path.join("results", "raw", "ablation")
PROCESSED_DIR = os.path.join("results", "processed")
STATISTICAL_DIR = os.path.join("results", "statistical")

ALGORITHM_ORDER = [
    "HGC-DMPA", "MPA", "NSGA-II", "NSGA-III", "MOEA/D", "MOPSO", "ZOA", "CPO", "WS-KSP"
]

BENCHMARK_ORDER = [
    "RGG-500", "RGG-1000", "RGG-2000", "RGG-3500", "RGG-5000",
    "Hosp.-1000", "Brightkite", "BA-500", "BA-1500"
]

ABLATION_VARIANTS = [
    "Full HGC-DMPA",
    "w/o Decomposition (HGC-MPA)",
    "Unweighted Louvain Coarsening",
    "w/o BA-TPR",
    "w/o Coarsening (Flat Decomp.-MPA)"
]

def load_all_raw_runs():
    """Loads all run JSON files from results/raw/."""
    files = glob.glob(os.path.join(RAW_DIR, "*.json"))
    runs_by_instance = {inst: {alg: [] for alg in ALGORITHM_ORDER} for inst in BENCHMARK_ORDER}
    
    for fpath in files:
        fname = os.path.basename(fpath)
        try:
            with open(fpath, "r") as f:
                data = json.load(f)
            inst = data.get("instance")
            alg = data.get("algorithm")
            if inst in runs_by_instance and alg in runs_by_instance[inst]:
                runs_by_instance[inst][alg].append(data)
        except Exception as e:
            logging.warning(f"Error loading {fpath}: {e}")
            
    return runs_by_instance


def construct_reference_front(instance_runs: dict) -> np.ndarray:
    """
    Constructs the reference Pareto front P^* from the union of all feasible
    solutions across all algorithms and all runs on this instance.
    """
    all_points = []
    for alg, runs in instance_runs.items():
        for r in runs:
            if r.get("is_feasible") and r.get("front"):
                front = np.array(r["front"], dtype=float)
                if len(front) > 0:
                    all_points.append(front)
                    
    if not all_points:
        return np.empty((0, 4))
        
    combined = np.vstack(all_points)
    
    # Filter non-dominated points
    # Minimization form
    n_pts = len(combined)
    is_dominated = np.zeros(n_pts, dtype=bool)
    for i in range(n_pts):
        if is_dominated[i]:
            continue
        # Points that strictly dominate combined[i]
        dom = np.all(combined <= combined[i] + 1.0e-9, axis=1) & np.any(combined < combined[i] - 1.0e-9, axis=1)
        if np.any(dom):
            is_dominated[i] = True
            
    non_dom = combined[~is_dominated]
    # Remove duplicates
    non_dom = np.unique(np.round(non_dom, 6), axis=0)
    return non_dom


def aggregate_benchmarks():
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    os.makedirs(STATISTICAL_DIR, exist_ok=True)
    
    runs_by_instance = load_all_raw_runs()
    
    summary_results = {}
    statistical_results = {}
    
    # Reference fronts per instance
    ref_fronts = {}
    for inst in BENCHMARK_ORDER:
        ref_front = construct_reference_front(runs_by_instance[inst])
        ref_fronts[inst] = ref_front
        logging.info(f"Instance {inst}: Reference front contains {len(ref_front)} non-dominated points.")

    for inst in BENCHMARK_ORDER:
        summary_results[inst] = {}
        ref_front = ref_fronts[inst]
        
        # Min-max bounds for normalization
        if len(ref_front) > 0:
            ideal = np.min(ref_front, axis=0)
            nadir = np.max(ref_front, axis=0)
            denom = np.maximum(1.0e-6, nadir - ideal)
        else:
            ideal = np.zeros(4)
            denom = np.ones(4)

        norm_ref_front = (ref_front - ideal) / denom if len(ref_front) > 0 else np.empty((0, 4))
        
        control_hv_runs = []
        stochastic_hv_map = {}

        for alg in ALGORITHM_ORDER:
            runs = runs_by_instance[inst][alg]
            n_total = len(runs)
            
            hv_list = []
            igd_list = []
            feas_count = 0
            runtime_list = []
            
            for r in runs:
                runtime_list.append(r.get("runtime", 0.0))
                if r.get("is_feasible") and r.get("front") and len(r["front"]) > 0:
                    feas_count += 1
                    raw_front = np.array(r["front"], dtype=float)
                    norm_front = (raw_front - ideal) / denom
                    
                    hv = compute_hypervolume(norm_front)
                    igd = compute_igd(norm_front, norm_ref_front) if len(norm_ref_front) > 0 else float("nan")
                    
                    hv_list.append(hv)
                    if not np.isnan(igd):
                        igd_list.append(igd)
                else:
                    hv_list.append(0.0)
                    
            feas_pct = (feas_count / max(1, n_total)) * 100.0
            
            mean_hv = float(np.mean(hv_list)) if hv_list else 0.0
            std_hv = float(np.std(hv_list)) if hv_list else 0.0
            mean_igd = float(np.mean(igd_list)) if igd_list else float("nan")
            std_igd = float(np.std(igd_list)) if igd_list else float("nan")
            mean_rt = float(np.mean(runtime_list)) if runtime_list else 0.0
            
            summary_results[inst][alg] = {
                "runs_completed": n_total,
                "feas_count": feas_count,
                "feas_pct": feas_pct,
                "hv_mean": mean_hv,
                "hv_std": std_hv,
                "igd_mean": mean_igd,
                "igd_std": std_igd,
                "runtime_mean": mean_rt,
                "hv_runs": hv_list
            }
            
            if alg == "HGC-DMPA":
                control_hv_runs = hv_list
            elif alg != "WS-KSP":
                stochastic_hv_map[alg] = hv_list

        # Statistical tests against baselines
        statistical_results[inst] = {}
        raw_p_values = []
        algs_to_compare = []
        a12_map = {}
        
        for alg in ALGORITHM_ORDER[1:]:
            algs_to_compare.append(alg)
            if alg == "WS-KSP":
                ws_val = summary_results[inst]["WS-KSP"]["hv_mean"]
                p_val, a12 = compare_against_deterministic(np.array(control_hv_runs), ws_val)
            else:
                base_runs = stochastic_hv_map.get(alg, [0.0] * len(control_hv_runs))
                p_val, a12 = compare_algorithms_wilcoxon(np.array(control_hv_runs), np.array(base_runs))
                
            raw_p_values.append(p_val)
            a12_map[alg] = a12

        # Apply Holm correction
        adj_p_values, is_sig = holm_bonferroni_correction(raw_p_values, alpha=0.05)
        for i, alg in enumerate(algs_to_compare):
            statistical_results[inst][alg] = {
                "raw_p": raw_p_values[i],
                "holm_p": adj_p_values[i],
                "a12": a12_map[alg],
                "significant": is_sig[i]
            }

    # Save summary
    with open(os.path.join(PROCESSED_DIR, "summary_results.json"), "w") as f:
        # Convert runs lists to strings or exclude to keep summary clean
        clean_summary = {}
        for inst, d in summary_results.items():
            clean_summary[inst] = {}
            for alg, metrics in d.items():
                m_copy = {k: v for k, v in metrics.items() if k != "hv_runs"}
                clean_summary[inst][alg] = m_copy
        json.dump(clean_summary, f, indent=2)

    with open(os.path.join(STATISTICAL_DIR, "statistical_summary.json"), "w") as f:
        json.dump(statistical_results, f, indent=2)

    logging.info("Benchmarking aggregation complete.")
    return summary_results, statistical_results


def aggregate_ablation():
    """Aggregates ablation raw runs for Table VI and Table VII."""
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    os.makedirs(STATISTICAL_DIR, exist_ok=True)
    files = glob.glob(os.path.join(ABLATION_RAW_DIR, "*.json"))
    ablation_by_inst = {inst: {v: [] for v in ABLATION_VARIANTS} for inst in ["RGG-1000", "RGG-2000", "RGG-5000"]}
    
    for fpath in files:
        try:
            with open(fpath, "r") as f:
                d = json.load(f)
            inst = d.get("instance")
            var = d.get("variant")
            if inst in ablation_by_inst and var in ablation_by_inst[inst]:
                ablation_by_inst[inst][var].append(d)
        except Exception:
            pass
            
    ablation_summary = {}
    
    for inst in ["RGG-1000", "RGG-2000", "RGG-5000"]:
        ablation_summary[inst] = {}
        
        # Build reference front for this instance
        all_pts = []
        for var, runs in ablation_by_inst[inst].items():
            for r in runs:
                if r.get("is_feasible") and r.get("front"):
                    front = np.array(r["front"], dtype=float)
                    if len(front) > 0:
                        all_pts.append(front)
        ref_front = np.vstack(all_pts) if all_pts else np.empty((0, 4))
        
        ideal = np.min(ref_front, axis=0) if len(ref_front) > 0 else np.zeros(4)
        nadir = np.max(ref_front, axis=0) if len(ref_front) > 0 else np.ones(4)
        denom = np.maximum(1.0e-6, nadir - ideal)
        norm_ref = (ref_front - ideal) / denom if len(ref_front) > 0 else np.empty((0, 4))

        full_hv_mean = None
        for var in ABLATION_VARIANTS:
            runs = ablation_by_inst[inst][var]
            hv_list = []
            igd_list = []
            feas = 0
            for r in runs:
                if r.get("is_feasible") and r.get("front") and len(r["front"]) > 0:
                    feas += 1
                    raw_f = np.array(r["front"], dtype=float)
                    norm_f = (raw_f - ideal) / denom
                    hv = compute_hypervolume(norm_f)
                    igd = compute_igd(norm_f, norm_ref)
                    hv_list.append(hv)
                    if not np.isnan(igd): igd_list.append(igd)
                else:
                    hv_list.append(0.0)
                    
            m_hv = float(np.mean(hv_list)) if hv_list else 0.0
            s_hv = float(np.std(hv_list)) if hv_list else 0.0
            m_igd = float(np.mean(igd_list)) if igd_list else float("nan")
            feas_pct = (feas / max(1, len(runs))) * 100.0
            
            if var == "Full HGC-DMPA":
                full_hv_mean = m_hv
                delta_hv = 0.0
            else:
                delta_hv = ((full_hv_mean - m_hv) / max(1.0e-6, full_hv_mean)) * 100.0 if full_hv_mean else 0.0
                
            ablation_summary[inst][var] = {
                "hv_mean": m_hv,
                "hv_std": s_hv,
                "delta_hv_pct": delta_hv,
                "igd_mean": m_igd,
                "feas_pct": feas_pct,
                "hv_runs": hv_list
            }

    with open(os.path.join(PROCESSED_DIR, "ablation_summary.json"), "w") as f:
        clean = {}
        for inst, d in ablation_summary.items():
            clean[inst] = {}
            for var, m in d.items():
                clean[inst][var] = {k: v for k, v in m.items() if k != "hv_runs"}
        json.dump(clean, f, indent=2)
        
    logging.info("Ablation aggregation complete.")
    return ablation_summary

if __name__ == "__main__":
    aggregate_benchmarks()
    aggregate_ablation()
