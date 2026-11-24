"""
Trust-weighted Louvain community detection maximizing Q_T (Equation 6):
Q_T = \frac{1}{2W} \sum_{u, v} [w_{uv} - \frac{k_u^w k_v^w}{2W}] \delta(c_u, c_v)
Supports trust-weighted affinity weights or unweighted modularity for ablation.
"""

import community as community_louvain
import networkx as nx

def partition_graph_louvain(graph: nx.Graph, weight_attr: str = 'affinity_weight',
                            random_state: int = 42) -> dict:
    """
    Executes Louvain community detection on the admissible subgraph.
    Returns:
        partition: dict mapping node_id -> community_id (0..K-1)
    """
    # If the graph is empty or has 0 nodes
    if graph.number_of_nodes() == 0:
        return {}

    # If disconnected, apply Louvain per connected component to guarantee valid community indices
    partition = {}
    next_comm_id = 0
    
    # Assign unit weights if unweighted or missing
    has_weights = (weight_attr != 'unweighted') and all(weight_attr in data for _, _, data in graph.edges(data=True)) if graph.number_of_edges() > 0 else False
    if has_weights:
        effective_weight = weight_attr
    else:
        for u, v in graph.edges():
            graph[u][v]['_unit_weight'] = 1.0
        effective_weight = '_unit_weight'

    components = list(nx.connected_components(graph))
    # Sort components by size descending for reproducibility
    components.sort(key=len, reverse=True)
    
    for comp in components:
        subg = graph.subgraph(comp)
        if len(comp) == 1:
            node = list(comp)[0]
            partition[node] = next_comm_id
            next_comm_id += 1
        else:
            sub_part = community_louvain.best_partition(
                subg,
                weight=effective_weight,
                random_state=random_state
            )
            # Offset community IDs to avoid collisions across components
            unique_sub_ids = {old_id: next_comm_id + i for i, old_id in enumerate(sorted(set(sub_part.values())))}
            for node, old_id in sub_part.items():
                partition[node] = unique_sub_ids[old_id]
            next_comm_id += len(unique_sub_ids)

    return partition


def compute_weighted_modularity(graph: nx.Graph, partition: dict,
                                weight_attr: str = 'affinity_weight') -> float:
    """
    Computes modularity Q_T of the partition under weight_attr.
    """
    if not partition or graph.number_of_edges() == 0:
        return 0.0
    has_weights = (weight_attr != 'unweighted') and all(weight_attr in data for _, _, data in graph.edges(data=True))
    if has_weights:
        effective_weight = weight_attr
    else:
        for u, v in graph.edges():
            graph[u][v]['_unit_weight'] = 1.0
        effective_weight = '_unit_weight'
    try:
        mod = community_louvain.modularity(partition, graph, weight=effective_weight)
        return float(mod)
    except Exception:
        return 0.0

