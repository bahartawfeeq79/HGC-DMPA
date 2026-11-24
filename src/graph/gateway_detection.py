"""
Boundary gateway bridge identification and active gateway set extraction:
B_{ab} = {(u, v) \in E_adm | u \in C_a, v \in C_b}
V_gw = \bigcup_{(C_a, C_b) \in E_M} {u, v | (u, v) \in B_{ab}}
(Equation 7 in Tawfeeq 2026)
"""

from collections import defaultdict
import networkx as nx

def identify_boundary_bridges(graph: nx.Graph, partition: dict):
    """
    Identifies all inter-community boundary bridge edges and groups them by community pair.
    Returns:
        boundary_bridges: dict mapping (comm_a, comm_b) with comm_a < comm_b -> list of (u, v)
        gateway_nodes: set of node IDs participating in at least one boundary bridge
        node_to_comm: dict mapping node_id -> comm_id
    """
    boundary_bridges = defaultdict(list)
    gateway_nodes = set()
    
    for u, v in graph.edges():
        c_u = partition.get(u)
        c_v = partition.get(v)
        if c_u is not None and c_v is not None and c_u != c_v:
            pair = (min(c_u, c_v), max(c_u, c_v))
            boundary_bridges[pair].append((u, v))
            gateway_nodes.add(u)
            gateway_nodes.add(v)
            
    return dict(boundary_bridges), sorted(list(gateway_nodes))
