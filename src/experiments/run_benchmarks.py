"""
Parallel Benchmark Experiment Execution Runner:
Executes runs across all 9 SIoT benchmark instances using multi-processing pool (12 workers).
Algorithms: HGC-DMPA, MPA, NSGA-II, NSGA-III, MOEA/D, MOPSO, ZOA, CPO, WS-KSP.
Saves individual raw run checkpoints to results/raw/.
"""

import os
import sys
import json
import time
import math
import logging
import yaml
import multiprocessing as mp
import networkx as nx

from src.data.preprocess import deserialize_graph
from src.graph.graph_model import SIoTGraphModel
from src.objectives.latency import calculate_path_latency
from src.algorithms.hgc_dmpa import HGCDMPAOptimizer
from src.algorithms.mpa import FlatMPAOptimizer
from src.algorithms.nsga2 import FlatNSGA2Optimizer
from src.algorithms.nsga3 import FlatNSGA3Optimizer
from src.algorithms.moead import FlatMOEADOptimizer
from src.algorithms.mopso import FlatMOPSOOptimizer
from src.algorithms.zoa import FlatZOAOptimizer
from src.algorithms.cpo import FlatCPOOptimizer
from src.algorithms.ws_ksp import WSKSPOptimizer
from src.evaluation.metrics import calculate_route_statistics

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

RAW_RESULTS_DIR = os.path.join("results", "raw")
CHECKPOINTS_DIR = os.path.join("experiments", "checkpoints")

def get_instance_filepath(instance_name: str) -> str:
    mapping = {
        "Brightkite": os.path.join("datasets", "processed", "brightkite", "brightkite_1000.json"),
        "RGG-500": os.path.join("datasets", "processed", "rgg", "rgg-500.json"),
        "RGG-1000": os.path.join("datasets", "processed", "rgg", "rgg-1000.json"),
        "RGG-2000": os.path.join("datasets", "processed", "rgg", "rgg-2000.json"),
        "RGG-3500": os.path.join("datasets", "processed", "rgg", "rgg-3500.json"),
        "RGG-5000": os.path.join("datasets", "processed", "rgg", "rgg-5000.json"),
        "Hosp.-1000": os.path.join("datasets", "processed", "hospital", "hosp_1000.json"),
        "BA-500": os.path.join("datasets", "processed", "barabasi_albert", "ba_500.json"),
        "BA-1500": os.path.join("datasets", "processed", "barabasi_albert", "ba_1500.json"),
    }
    return mapping.get(instance_name, "")


def find_consumer_provider_endpoints(adm_graph, raw_graph, model):
    if nx.number_connected_components(adm_graph) == 0:
        nodes = list(raw_graph.nodes())
        return nodes[0], nodes[-1], 10, 0.5
        
    largest_cc = max(nx.connected_components(adm_graph), key=len)
    subg = adm_graph.subgraph(largest_cc)
    
    nodes_list = list(subg.nodes())
    start_node = nodes_list[0]
    lengths = nx.single_source_shortest_path_length(subg, start_node)
    farthest_1 = max(lengths, key=lengths.get)
    
    lengths_2 = nx.single_source_shortest_path_length(subg, farthest_1)
    farthest_2 = max(lengths_2, key=lengths_2.get)
    shortest_hop_diam = lengths_2[farthest_2]
    
    def hop_lat_weight(u, v, d):
        return model.get_hop_latency(u, v)
        
    try:
        fastest_path = nx.shortest_path(subg, source=farthest_1, target=farthest_2, weight=hop_lat_weight)
        fastest_lat = calculate_path_latency(fastest_path, model)
    except Exception:
        fastest_path = nx.shortest_path(subg, source=farthest_1, target=farthest_2)
        fastest_lat = calculate_path_latency(fastest_path, model)

    return farthest_1, farthest_2, shortest_hop_diam, fastest_lat


def _worker_run_task(task_args):
    """Worker function executed by multiprocessing pool."""
    inst_name, alg_name, seed, run_idx, fpath, deadline_mult = task_args
    alg_slug = alg_name.replace("/", "_")
    ckpt_file = os.path.join(RAW_RESULTS_DIR, f"{inst_name}_{alg_slug}_run{run_idx}_seed{seed}.json")
    if os.path.exists(ckpt_file):
        return f"[SKIP] {inst_name} - {alg_name} run {run_idx}"

    try:
        G = deserialize_graph(fpath)
        model = SIoTGraphModel(G)
        src, dst, diam, fastest_lat = find_consumer_provider_endpoints(model.adm_graph, G, model)
        t_dl = fastest_lat * deadline_mult
        hop_budget = diam + max(3, int(math.floor(0.3 * diam))) + 3
        
        t0 = time.time()
        if alg_name == "HGC-DMPA":
            opt = HGCDMPAOptimizer(model, src, dst, t_dl=t_dl, hop_budget=hop_budget, eval_budget=3000, seed=seed)
            res = opt.run()
        elif alg_name == "MPA":
            opt = FlatMPAOptimizer(model, src, dst, t_dl=t_dl, hop_budget=hop_budget, eval_budget=3000, seed=seed)
            res = opt.run()
        elif alg_name == "NSGA-II":
            opt = FlatNSGA2Optimizer(model, src, dst, t_dl=t_dl, hop_budget=hop_budget, eval_budget=3000, seed=seed)
            res = opt.run()
        elif alg_name == "NSGA-III":
            opt = FlatNSGA3Optimizer(model, src, dst, t_dl=t_dl, hop_budget=hop_budget, eval_budget=3000, seed=seed)
            res = opt.run()
        elif alg_name == "MOEA/D":
            opt = FlatMOEADOptimizer(model, src, dst, t_dl=t_dl, hop_budget=hop_budget, eval_budget=3000, seed=seed)
            res = opt.run()
        elif alg_name == "MOPSO":
            opt = FlatMOPSOOptimizer(model, src, dst, t_dl=t_dl, hop_budget=hop_budget, eval_budget=3000, seed=seed)
            res = opt.run()
        elif alg_name == "ZOA":
            opt = FlatZOAOptimizer(model, src, dst, t_dl=t_dl, hop_budget=hop_budget, eval_budget=3000, seed=seed)
            res = opt.run()
        elif alg_name == "CPO":
            opt = FlatCPOOptimizer(model, src, dst, t_dl=t_dl, hop_budget=hop_budget, eval_budget=3000, seed=seed)
            res = opt.run()
        elif alg_name == "WS-KSP":
            opt = WSKSPOptimizer(model, src, dst, t_dl=t_dl, hop_budget=hop_budget, k_paths=150, n_weights=20)
            res = opt.run()
        else:
            return f"[ERROR] Unknown alg {alg_name}"

        runtime = time.time() - t0
        route_stats = []
        if res.get("archive"):
            for sol in res["archive"][:5]: # Store top 5 route stats
                stats = calculate_route_statistics(sol["path"], model.raw_graph, model, t_dl)
                route_stats.append(stats)

        record = {
            "instance": inst_name,
            "algorithm": alg_name,
            "seed": seed,
            "run_idx": run_idx,
            "evals": res["evals"],
            "runtime": runtime,
            "is_feasible": res["is_feasible"],
            "feasible_count": len(res["archive"]),
            "front": res["front"].tolist() if hasattr(res["front"], "tolist") else res["front"],
            "route_stats": route_stats
        }
        with open(ckpt_file, "w") as f:
            json.dump(record, f, indent=2)
        return f"[DONE] {inst_name} - {alg_name} (run {run_idx}, feas: {res['is_feasible']}, {runtime:.2f}s)"
    except Exception as e:
        return f"[FAIL] {inst_name} - {alg_name} run {run_idx}: {e}"


def run_benchmark_campaign(instances_subset=None, algorithms_subset=None, n_runs=10, max_workers=10):
    os.makedirs(RAW_RESULTS_DIR, exist_ok=True)
    os.makedirs(CHECKPOINTS_DIR, exist_ok=True)
    
    with open("config/benchmark_config.yaml", "r") as f:
        bench_cfg = yaml.safe_load(f)["benchmarks"]
    with open("config/random_seeds.json", "r") as f:
        seeds = json.load(f)["run_seeds"]

    instances = list(bench_cfg.keys()) if instances_subset is None else instances_subset
    all_algs = ["HGC-DMPA", "MPA", "NSGA-II", "NSGA-III", "MOEA/D", "MOPSO", "ZOA", "CPO", "WS-KSP"]
    algs = all_algs if algorithms_subset is None else algorithms_subset

    tasks = []
    for inst in instances:
        fpath = get_instance_filepath(inst)
        if not os.path.exists(fpath):
            continue
        deadline_mult = bench_cfg[inst].get("deadline_factor", 1.5)
        for alg in algs:
            runs_to_exec = 1 if alg == "WS-KSP" else n_runs
            for r_idx in range(runs_to_exec):
                seed = seeds[r_idx]
                tasks.append((inst, alg, seed, r_idx, fpath, deadline_mult))

    logging.info(f"Queued {len(tasks)} benchmark tasks across {max_workers} parallel workers...")
    
    with mp.Pool(processes=max_workers) as pool:
        for i, status in enumerate(pool.imap_unordered(_worker_run_task, tasks), 1):
            if i % 10 == 0 or i == len(tasks):
                logging.info(f"[{i}/{len(tasks)}] {status}")

    logging.info("All benchmark tasks completed successfully.")

if __name__ == "__main__":
    run_benchmark_campaign(n_runs=10, max_workers=10)
