"""
Tier 1: Macro-Community Corridor Selection
Selects cycle-free community sequence P_M = (C_(1), ..., C_(L)) on macro-graph G_M
according to Equation (15) in Tawfeeq 2026.
"""

import math
from typing import List
import networkx as nx

def decode_macro_corridor(macro_graph: nx.Graph, x_comm: List[float],
                          c_src: int, c_dst: int, h_m_budget: int = 50,
                          mu: float = 1.5, nu: float = 1.0,
                          mean_macro_lat: float = 0.01) -> List[int]:
    """
    Decodes continuous community random keys into a cycle-free macro-community corridor.
    C^* = argmax_{C_b} [ X_{i, C_b}^{comm} + mu * ln(T_bar(C_a, C_b)) - nu * ln(l_bar(C_a, C_b) / l_bar_M) ]
    """
    if c_src == c_dst:
        return [c_src]
        
    # Precompute shortest-path distances to c_dst on macro-graph
    try:
        dist_to_dst = nx.shortest_path_length(macro_graph, target=c_dst)
    except Exception:
        dist_to_dst = {}

    corridor = [c_src]
    visited = {c_src}
    curr = c_src
    
    mean_macro_lat = max(1.0e-6, mean_macro_lat)
    
    step = 0
    max_steps = min(h_m_budget, macro_graph.number_of_nodes() + 5)
    
    while curr != c_dst and step < max_steps:
        step += 1
        nbrs = list(macro_graph.neighbors(curr))
        
        # If destination is directly adjacent, greedily step to it
        if c_dst in nbrs:
            corridor.append(c_dst)
            break
            
        unvisited_nbrs = [nb for nb in nbrs if nb not in visited]
        if not unvisited_nbrs:
            # Backtrack or break if dead-end
            break
            
        # Filter by macro-hop budget if possible
        admissible_nbrs = []
        for nb in unvisited_nbrs:
            h_rem = dist_to_dst.get(nb, 999)
            if step + 1 + h_rem <= h_m_budget:
                admissible_nbrs.append(nb)
        if not admissible_nbrs:
            admissible_nbrs = unvisited_nbrs
            
        # Score each candidate neighbor
        best_score = -float("inf")
        best_next = admissible_nbrs[0]
        
        for nb in admissible_nbrs:
            edge_data = macro_graph.get_edge_data(curr, nb, default={})
            t_bar = max(1.0e-6, min(1.0, edge_data.get('mean_trust', 0.5)))
            l_bar = max(1.0e-6, edge_data.get('mean_latency', 0.01))
            
            key_val = x_comm[nb] if nb < len(x_comm) else 0.0
            score = key_val + mu * math.log(t_bar) - nu * math.log(l_bar / mean_macro_lat)
            
            if score > best_score:
                best_score = score
                best_next = nb
                
        corridor.append(best_next)
        visited.add(best_next)
        curr = best_next

    # Loop erasure on corridor
    clean_corridor = []
    seen = {}
    for i, c in enumerate(corridor):
        if c in seen:
            prev_idx = seen[c]
            # Erase loop
            for rem in clean_corridor[prev_idx:]:
                if rem in seen:
                    del seen[rem]
            clean_corridor = clean_corridor[:prev_idx]
        clean_corridor.append(c)
        seen[c] = len(clean_corridor) - 1

    # If corridor didn't reach destination, append shortest path from last reached community
    if clean_corridor[-1] != c_dst:
        try:
            sp = nx.shortest_path(macro_graph, clean_corridor[-1], c_dst)
            clean_corridor.extend(sp[1:])
        except Exception:
            pass

    return clean_corridor
