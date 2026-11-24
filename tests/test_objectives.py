"""
Automated unit tests for 4 objectives and hard constraints.
"""

import numpy as np
from src.graph.graph_model import SIoTGraphModel
from src.data.generators import generate_rgg_topology
from src.objectives.trust import calculate_path_trust
from src.objectives.latency import calculate_path_latency
from src.objectives.energy import calculate_path_energy
from src.objectives.qos import calculate_path_qos
from src.objectives.constraints import calculate_constraint_violation, check_simple_path

def test_objectives_and_constraints():
    G = generate_rgg_topology(target_nodes=100, target_edges=350, target_diameter=8, compromised_count=10, seed=42)
    model = SIoTGraphModel(G)
    
    # Simple path of 4 nodes: 0 - 1 - 2 - 3
    # Find a connected path in G
    nodes = list(G.nodes())
    import networkx as nx
    path = nx.shortest_path(G, nodes[0], nodes[-1])
    
    trust = calculate_path_trust(path, G)
    lat = calculate_path_latency(path, model)
    ene = calculate_path_energy(path, model)
    qos = calculate_path_qos(path, G)
    
    assert 0.0 <= trust <= 1.0
    assert lat > 0.0
    assert ene > 0.0
    assert 0.0 <= qos <= 1.0
    
    # Test simple path
    assert check_simple_path(path) == True
    # Test loop detection
    loop_path = path + [path[0]]
    assert check_simple_path(loop_path) == False
    
    # Test constraint violation
    cv_valid, v_dict = calculate_constraint_violation(path, G, latency=lat, t_dl=lat * 1.5)
    assert cv_valid >= 0.0

if __name__ == "__main__":
    test_objectives_and_constraints()
    print("test_objectives.py PASSED!")
