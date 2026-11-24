"""
End-to-end integration tests for HGC-DMPA and baseline optimizers.
"""

import networkx as nx
from src.data.generators import generate_rgg_topology
from src.graph.graph_model import SIoTGraphModel
from src.algorithms.hgc_dmpa import HGCDMPAOptimizer
from src.algorithms.mpa import FlatMPAOptimizer
from src.algorithms.nsga2 import FlatNSGA2Optimizer
from src.algorithms.moead import FlatMOEADOptimizer
from src.algorithms.ws_ksp import WSKSPOptimizer
from src.evaluation.ablation import AblationVariantRunner

def test_algorithm_smoke_runs():
    G = generate_rgg_topology(target_nodes=80, target_edges=250, target_diameter=6, compromised_count=8, seed=42)
    model = SIoTGraphModel(G)
    nodes = list(G.nodes())
    src, dst = nodes[0], nodes[-1]
    sp_len = nx.shortest_path_length(G, src, dst)
    t_dl = 0.5 # Generous deadline for test
    
    # 1. Test HGC-DMPA
    opt_hgc = HGCDMPAOptimizer(
        model, src, dst, t_dl=t_dl, eval_budget=80, n_pop=10, seed=42
    )
    res_hgc = opt_hgc.run()
    assert res_hgc["evals"] >= 80
    assert "front" in res_hgc

    # 2. Test Flat MPA
    opt_mpa = FlatMPAOptimizer(
        model, src, dst, t_dl=t_dl, eval_budget=50, n_pop=10, seed=42
    )
    res_mpa = opt_mpa.run()
    assert res_mpa["evals"] >= 50

    # 3. Test NSGA-II
    opt_nsga2 = FlatNSGA2Optimizer(
        model, src, dst, t_dl=t_dl, eval_budget=50, n_pop=10, seed=42
    )
    res_nsga2 = opt_nsga2.run()
    assert res_nsga2["evals"] >= 50

    # 4. Test MOEA/D
    opt_moead = FlatMOEADOptimizer(
        model, src, dst, t_dl=t_dl, eval_budget=50, n_pop=10, seed=42
    )
    res_moead = opt_moead.run()
    assert res_moead["evals"] >= 50

    # 5. Test WS-KSP
    opt_wsksp = WSKSPOptimizer(
        model, src, dst, t_dl=t_dl, k_paths=5, n_weights=4
    )
    res_wsksp = opt_wsksp.run()
    assert res_wsksp["evals"] >= 20

    # 6. Test Ablation variant (w/o BA-TPR)
    runner = AblationVariantRunner(
        "w/o BA-TPR", model, src, dst, t_dl=t_dl, eval_budget=50, seed=42
    )
    res_abl = runner.run()
    assert res_abl["evals"] >= 50

if __name__ == "__main__":
    test_algorithm_smoke_runs()
    print("test_algorithms.py PASSED!")
