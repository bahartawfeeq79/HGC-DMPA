r"""
Flat random-key greedy path decoder for baseline optimizers:
Decodes an N-dimensional continuous random-key vector X \in [LB, UB]^N
into a simple loop-free physical path from src to dst.
"""

import math
from typing import List
import networkx as nx
from src.decoder.micro_decoder import erase_loops

def decode_flat_random_key_path(graph, x_keys: List[float],
                                src_node: int, dst_node: int,
                                hop_budget: int = 100, mu: float = 1.5, nu: float = 1.0,
                                mean_lat_ref: float = 0.01) -> List[int]:
    r"""
    Decodes continuous flat keys X_v into a physical path:
    v^* = argmax_{v \in N(u) \setminus visited} [ X_v + mu * ln T(u, v) - nu * ln(l(u, v) / l_bar) ]
    """
    if src_node == dst_node:
        return [src_node]
        
    path = [src_node]
    visited = {src_node}
    curr = src_node
    mean_lat_ref = max(1.0e-6, mean_lat_ref)
    
    # Node to index mapping if node IDs are arbitrary
    node_to_idx = getattr(graph, '_node_to_idx', None)
    if node_to_idx is None:
        node_to_idx = {n: i for i, n in enumerate(graph.nodes())}
        try:
            graph._node_to_idx = node_to_idx
        except Exception:
            pass
    
    step = 0
    while curr != dst_node and step < hop_budget:
        step += 1
        nbrs = list(graph.neighbors(curr))
        if dst_node in nbrs:
            path.append(dst_node)
            break
            
        unvisited_nbrs = [nb for nb in nbrs if nb not in visited]
        if not unvisited_nbrs:
            break
            
        best_score = -float("inf")
        best_next = unvisited_nbrs[0]
        
        for nb in unvisited_nbrs:
            idx = node_to_idx.get(nb, 0)
            key_val = x_keys[idx] if idx < len(x_keys) else 0.0
            
            edge_data = graph.get_edge_data(curr, nb, default={})
            t_uv = max(1.0e-6, min(1.0, edge_data.get('trust', 0.5)))
            
            # Note: flat baselines on the physical graph may use advertised latency if adversarial
            l_uv = edge_data.get('adv_prop_delay', edge_data.get('prop_delay', 0.001)) + 0.005
            l_uv = max(1.0e-6, l_uv)
            
            score = key_val + mu * math.log(t_uv) - nu * math.log(l_uv / mean_lat_ref)
            if score > best_score:
                best_score = score
                best_next = nb
                
        path.append(best_next)
        visited.add(best_next)
        curr = best_next

    clean_path = erase_loops(path)
    if clean_path[-1] != dst_node:
        try:
            sp = nx.shortest_path(graph, clean_path[-1], dst_node)
            clean_path.extend(sp[1:])
            clean_path = erase_loops(clean_path)
        except Exception:
            pass

    return clean_path
