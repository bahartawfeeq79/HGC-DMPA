"""
Automated unit tests for graph coarsening, Louvain partitioning, and gateway detection.
"""

import networkx as nx
from src.graph.graph_model import SIoTGraphModel
from src.graph.coarsening import CoarsenedMacroGraph
from src.data.generators import generate_rgg_topology

def test_graph_filtering_and_coarsening():
    # Build a small RGG
    G = generate_rgg_topology(target_nodes=100, target_edges=300, target_diameter=8, compromised_count=10, seed=123)
    model = SIoTGraphModel(G, t_min=0.35, bat_min=0.10)
    
    # Admissible graph should exclude all compromised nodes with t_base < 0.35
    for n, data in model.adm_graph.nodes(data=True):
        assert data['t_base'] >= 0.35
        assert data['bat'] >= 0.10
        
    mg = CoarsenedMacroGraph(model.adm_graph, physical_nodes_total=G.number_of_nodes(), model_ref=model)
    
    assert mg.K > 1
    assert mg.d_eff == mg.K + mg.num_gateways
    assert mg.d_eff < G.number_of_nodes()
    assert mg.compression_pct > 0.0
    assert mg.macro_graph.number_of_nodes() == mg.K
    assert mg.macro_graph.number_of_edges() > 0

if __name__ == "__main__":
    test_graph_filtering_and_coarsening()
    print("test_graph.py PASSED!")
