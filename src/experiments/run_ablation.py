"""
Parallel Ablation Experiment Execution Runner:
Executes runs across the 5 architectural variants on:
RGG-1000, RGG-2000, and RGG-5000 using multiprocessing pool.
Saves results to results/raw/ablation/.
"""

import os
import json
import time
import math
import logging
import multiprocessing as mp

from src.data.preprocess import deserialize_graph
from src.graph.graph_model import SIoTGraphModel
from src.evaluation.ablation import AblationVariantRunner
from src.experiments.run_benchmarks import get_instance_filepath, find_consumer_provider_endpoints

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

ABLATION_RAW_DIR = os.path.join("results", "raw", "ablation")

def _worker_ablation_task(task_args):
    inst, var, seed, r_idx, fpath = task_args
    var_slug = var.replace(" ", "_").replace("/", "").replace("(", "").replace(")", "").replace(".", "")
    ckpt_file = os.path.join(ABLATION_RAW_DIR, f"{inst}_{var_slug}_run{r_idx}_seed{seed}.json")
    if os.path.exists(ckpt_file):
        return f"[SKIP] {inst} - {var} run {r_idx}"

    try:
        G = deserialize_graph(fpath)
        model = SIoTGraphModel(G)
        src, dst, diam, fastest_lat = find_consumer_provider_endpoints(model.adm_graph, G, model)
        t_dl = fastest_lat * 1.5
        hop_budget = diam + max(3, int(math.floor(0.3 * diam))) + 3
        
        runner = AblationVariantRunner(
            variant_name=var, graph_model=model, src_node=src, dst_node=dst,
            t_dl=t_dl, hop_budget=hop_budget, eval_budget=3000, seed=seed
        )
        res = runner.run()
        
        record = {
            "instance": inst,
            "variant": var,
            "seed": seed,
            "run_idx": r_idx,
            "evals": res["evals"],
            "runtime": res["runtime"],
            "is_feasible": res["is_feasible"],
            "feasible_count": len(res["archive"]),
            "front": res["front"].tolist() if hasattr(res["front"], "tolist") else res["front"]
        }
        with open(ckpt_file, "w") as f:
            json.dump(record, f, indent=2)
        return f"[DONE] {inst} - {var} (run {r_idx}, feas: {res['is_feasible']}, {res['runtime']:.2f}s)"
    except Exception as e:
        return f"[FAIL] {inst} - {var} run {r_idx}: {e}"


def run_ablation_campaign(instances_subset=None, n_runs=10, max_workers=10):
    os.makedirs(ABLATION_RAW_DIR, exist_ok=True)
    with open("config/random_seeds.json", "r") as f:
        seeds = json.load(f)["run_seeds"]

    target_instances = ["RGG-1000", "RGG-2000", "RGG-5000"] if instances_subset is None else instances_subset
    variants = [
        "Full HGC-DMPA",
        "w/o Decomposition (HGC-MPA)",
        "Unweighted Louvain Coarsening",
        "w/o BA-TPR",
        "w/o Coarsening (Flat Decomp.-MPA)"
    ]

    tasks = []
    for inst in target_instances:
        fpath = get_instance_filepath(inst)
        if not os.path.exists(fpath):
            continue
        for var in variants:
            for r_idx in range(n_runs):
                seed = seeds[r_idx]
                tasks.append((inst, var, seed, r_idx, fpath))

    logging.info(f"Queued {len(tasks)} ablation tasks across {max_workers} parallel workers...")
    with mp.Pool(processes=max_workers) as pool:
        for i, status in enumerate(pool.imap_unordered(_worker_ablation_task, tasks), 1):
            if i % 10 == 0 or i == len(tasks):
                logging.info(f"[{i}/{len(tasks)}] {status}")

    logging.info("Ablation campaign completed successfully.")

if __name__ == "__main__":
    run_ablation_campaign(n_runs=10, max_workers=10)
