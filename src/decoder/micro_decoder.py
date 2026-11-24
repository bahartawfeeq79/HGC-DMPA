"""
Tier 2: Boundary Gateway Selection and Intra-Community Stitching
Constructs physical end-to-end micro-path P = (v_1, ..., v_m) from macro-corridor
according to Equation (16) in Tawfeeq 2026.
"""

import math
from typing import List, Dict, Tuple
import networkx as nx

def erase_loops(path: List[int]) -> List[int]:
    """Applies loop erasure to produce a simple cycle-free path."""
    clean = []
    pos = {}
    for node in path:
        if node in pos:
            idx = pos[node]
            # Erase loop
            for rem in clean[idx:]:
                if rem in pos:
                    del pos[rem]
            clean = clean[:idx]
        clean.append(node)
        pos[node] = len(clean) - 1
    return clean


def decode_micro_path(coarsened_mg, x_gw: List[float],
                      macro_corridor: List[int], src_node: int, dst_node: int,
                      mu: float = 1.5, nu: float = 1.0,
                      mean_lat_ref: float = 0.01) -> List[int]:
    """
    Decodes boundary gateways and intra-community paths along macro-corridor.
    (u_r^*, v_{r+1}^*) = argmax_{(u, v)} [ (X_{i, u}^{gw} + X_{i, v}^{gw}) / 2 + mu * ln T(u, v) - nu * ln(l(u, v) / l_bar) ]
    """
    adm_graph = coarsened_mg.adm_graph
    gw_to_idx = coarsened_mg.gw_to_idx
    mean_lat_ref = max(1.0e-6, mean_lat_ref)
    
    if len(macro_corridor) == 1:
        # Source and destination are in the same community
        comm_nodes = coarsened_mg.comm_to_nodes[macro_corridor[0]]
        subg = adm_graph.subgraph(comm_nodes)
        try:
            # Weighted shortest path using intra-community cost
            path = _intra_community_shortest_path(subg, src_node, dst_node, mu=mu, nu=nu, mean_lat_ref=mean_lat_ref)
            return erase_loops(path)
        except Exception:
            return [src_node, dst_node]

    # 1. Select boundary bridges between consecutive communities in corridor
    selected_bridges = [] # list of ((u_r, v_{r+1}), comm_r, comm_{r+1})
    for r in range(len(macro_corridor) - 1):
        c_curr = macro_corridor[r]
        c_next = macro_corridor[r + 1]
        pair = (min(c_curr, c_next), max(c_curr, c_next))
        bridges = coarsened_mg.boundary_bridges.get(pair, [])
        
        if not bridges:
            # Fallback if no direct bridge
            continue
            
        best_score = -float("inf")
        best_bridge = bridges[0]
        
        for u, v in bridges:
            # Orient bridge from c_curr to c_next
            u_orig, v_orig = (u, v) if coarsened_mg.partition[u] == c_curr else (v, u)
            
            idx_u = gw_to_idx.get(u_orig, 0)
            idx_v = gw_to_idx.get(v_orig, 0)
            k_u = x_gw[idx_u] if idx_u < len(x_gw) else 0.0
            k_v = x_gw[idx_v] if idx_v < len(x_gw) else 0.0
            
            edge_data = adm_graph.get_edge_data(u_orig, v_orig, default={})
            t_uv = max(1.0e-6, min(1.0, edge_data.get('trust', 0.5)))
            # Compute latency
            if coarsened_mg.model_ref is not None:
                l_uv = coarsened_mg.model_ref.get_hop_latency(u_orig, v_orig)
            else:
                l_uv = edge_data.get('prop_delay', 0.001) + 0.005
            l_uv = max(1.0e-6, l_uv)
            
            score = ((k_u + k_v) / 2.0) + mu * math.log(t_uv) - nu * math.log(l_uv / mean_lat_ref)
            if score > best_score:
                best_score = score
                best_bridge = (u_orig, v_orig)
                
        selected_bridges.append((best_bridge, c_curr, c_next))

    # 2. Stitch intra-community subpaths
    full_path = []
    
    # Community 0: from src_node to exit gateway of community 0
    if selected_bridges:
        first_bridge, c_0, _ = selected_bridges[0]
        exit_gw_0 = first_bridge[0]
        comm_nodes_0 = coarsened_mg.comm_to_nodes[c_0]
        subg_0 = adm_graph.subgraph(comm_nodes_0)
        seg_0 = _intra_community_shortest_path(subg_0, src_node, exit_gw_0, mu=mu, nu=nu, mean_lat_ref=mean_lat_ref)
        full_path.extend(seg_0)
        
        # Intermediate communities
        for idx in range(len(selected_bridges) - 1):
            curr_bridge, c_curr, _ = selected_bridges[idx]
            next_bridge, _, _ = selected_bridges[idx + 1]
            
            entry_gw = curr_bridge[1]
            exit_gw = next_bridge[0]
            comm_nodes = coarsened_mg.comm_to_nodes[c_curr] # Host of exit gateway
            # If entry_gw is in c_next, host of entry_gw is c_next
            host_comm = coarsened_mg.partition[entry_gw]
            subg_mid = adm_graph.subgraph(coarsened_mg.comm_to_nodes[host_comm])
            
            full_path.append(entry_gw)
            if entry_gw != exit_gw:
                seg_mid = _intra_community_shortest_path(subg_mid, entry_gw, exit_gw, mu=mu, nu=nu, mean_lat_ref=mean_lat_ref)
                full_path.extend(seg_mid[1:])
                
        # Last community: from entry gateway of last bridge to dst_node
        last_bridge, _, c_last = selected_bridges[-1]
        entry_gw_last = last_bridge[1]
        comm_nodes_last = coarsened_mg.comm_to_nodes[coarsened_mg.partition[dst_node]]
        subg_last = adm_graph.subgraph(comm_nodes_last)
        
        full_path.append(entry_gw_last)
        if entry_gw_last != dst_node:
            seg_last = _intra_community_shortest_path(subg_last, entry_gw_last, dst_node, mu=mu, nu=nu, mean_lat_ref=mean_lat_ref)
            full_path.extend(seg_last[1:])
    else:
        # Fallback shortest path across entire admissible graph
        try:
            full_path = nx.shortest_path(adm_graph, src_node, dst_node)
        except Exception:
            full_path = [src_node, dst_node]

    return erase_loops(full_path)


def _intra_community_shortest_path(subgraph: nx.Graph, source: int, target: int,
                                   mu: float, nu: float, mean_lat_ref: float) -> List[int]:
    """
    Computes shortest intra-community path under cost:
    c(u, v) = -mu * ln T(u, v) + nu * ln(l(u, v) / l_bar) + offset (to keep cost positive)
    """
    if source == target:
        return [source]
        
    def edge_weight(u, v, data):
        t = max(1.0e-6, min(1.0, data.get('trust', 0.5)))
        l = max(1.0e-6, data.get('prop_delay', 0.001) + 0.005)
        # Shift cost to ensure non-negative weights for Dijkstra
        raw_cost = -mu * math.log(t) + nu * math.log(l / mean_lat_ref)
        return max(0.001, raw_cost + 10.0)

    try:
        return nx.shortest_path(subgraph, source=source, target=target, weight=edge_weight)
    except Exception:
        # Fallback to unweighted shortest path
        try:
            return nx.shortest_path(subgraph, source=source, target=target)
        except Exception:
            return [source, target]
