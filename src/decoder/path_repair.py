"""
Lamarckian Boundary-Aware Trust-Guided Path Repair (BA-TPR):
1. Weak-Point Identification: checks internal relays for T_base < T_rep (0.70) or max latency hop
2. Tier-Aware Sub-Path Rerouting:
   - Case A (Intra-community relay v_j not in V_gw): local intra-community detour
   - Case B (Boundary gateway v_j in V_gw): suffix re-decoding from prefix corridor
3. Lamarckian Key Write-Back:
   - Suppresses banned keys: X_{i, ban} <- L_B (-5.0)
   - Reinforces detour elements: X_{i, w} <- max(X_{i, w}, 0.8 * U_B = 4.0)
(Section IV-B in Tawfeeq 2026)
"""

from typing import List, Tuple, Set
import copy
import networkx as nx
from src.decoder.micro_decoder import erase_loops, _intra_community_shortest_path
from src.decoder.macro_decoder import decode_macro_corridor

def identify_weak_relay(path: List[int], graph, graph_model,
                        t_rep: float = 0.70) -> Tuple[int, str]:
    """
    Identifies the weak relay node to repair:
    - If any internal relay has T_base < T_rep, pick argmin T_base
    - Else pick the node preceding the maximum latency hop
    """
    relays = path[1:-1]
    if not relays:
        return path[0], "none"
        
    lowest_t_node = None
    lowest_t_val = 1.0
    for r in relays:
        t_base = graph.nodes[r].get('t_base', 1.0)
        if t_base < lowest_t_val:
            lowest_t_val = t_base
            lowest_t_node = r
            
    if lowest_t_val < t_rep and lowest_t_node is not None:
        return lowest_t_node, "low_trust"
        
    # Otherwise find maximum latency hop
    max_lat = -1.0
    worst_node = relays[0]
    for j in range(len(path) - 1):
        u, v = path[j], path[j + 1]
        lat = graph_model.get_hop_latency(u, v)
        if lat > max_lat:
            max_lat = lat
            worst_node = u
            
    return worst_node, "high_latency"


def apply_boundary_aware_tpr(path: List[int], x_key: List[float],
                             coarsened_mg, graph_model,
                             src_node: int, dst_node: int,
                             t_rep: float = 0.70, lb: float = -5.0, ub: float = 5.0):
    """
    Executes BA-TPR with Lamarckian key write-back.
    Returns:
        repaired_path: List[int]
        repaired_key: List[float]
        repair_applied: bool
        repair_case: str ("case_a", "case_b", "none")
    """
    if len(path) <= 2:
        return path, x_key, False, "none"
        
    v_weak, reason = identify_weak_relay(path, coarsened_mg.adm_graph, graph_model, t_rep=t_rep)
    if v_weak in (src_node, dst_node):
        return path, x_key, False, "none"

    adm_graph = coarsened_mg.adm_graph
    gateway_set = set(coarsened_mg.gateway_nodes)
    partition = coarsened_mg.partition
    
    repaired_key = list(x_key)
    K = coarsened_mg.K
    
    # Check if weak relay is boundary gateway or intra-community node
    if v_weak not in gateway_set:
        # Case A: Intra-community relay
        host_comm = partition.get(v_weak)
        if host_comm is None:
            return path, x_key, False, "none"
            
        comm_nodes = [n for n in coarsened_mg.comm_to_nodes[host_comm] if n != v_weak]
        if len(comm_nodes) < 2:
            return path, x_key, False, "none"
            
        # Find index range of path in host_comm
        idx_weak = path.index(v_weak)
        # Find preceding entry node and succeeding exit node in this community
        start_idx = idx_weak
        while start_idx > 0 and partition.get(path[start_idx - 1]) == host_comm:
            start_idx -= 1
        end_idx = idx_weak
        while end_idx < len(path) - 1 and partition.get(path[end_idx + 1]) == host_comm:
            end_idx += 1
            
        entry_node = path[start_idx]
        exit_node = path[end_idx]
        
        subg = adm_graph.subgraph(comm_nodes)
        try:
            detour_seg = _intra_community_shortest_path(subg, entry_node, exit_node, mu=1.5, nu=1.0, mean_lat_ref=0.01)
            repaired_path = path[:start_idx] + detour_seg + path[end_idx + 1:]
            repaired_path = erase_loops(repaired_path)
            
            # Lamarckian write-back: reinforce host community key slightly
            c_idx = coarsened_mg.comm_to_idx.get(host_comm, 0)
            if c_idx < K:
                repaired_key[c_idx] = min(ub, repaired_key[c_idx] + 0.5)
                
            return repaired_path, repaired_key, True, "case_a"
        except Exception:
            # Subgraph disconnected without v_weak
            pass

    # Case B: Boundary Gateway (v_weak in gateway_set or Case A failed)
    host_comm = partition.get(v_weak)
    weak_gw_idx = coarsened_mg.gw_to_idx.get(v_weak, None)
    
    # Lamarckian write-back: suppress banned gateway key
    if weak_gw_idx is not None and K + weak_gw_idx < len(repaired_key):
        repaired_key[K + weak_gw_idx] = lb
        
    # Re-decode suffix with perturbed keys
    idx_weak = path.index(v_weak)
    prefix_nodes = path[:max(1, idx_weak)]
    last_prefix_node = prefix_nodes[-1]
    
    c_prefix = partition.get(last_prefix_node, coarsened_mg.partition[src_node])
    c_dst = partition.get(dst_node)
    
    # Re-decode macro corridor from c_prefix to c_dst
    x_comm = repaired_key[:K]
    x_gw = repaired_key[K:]
    
    try:
        suffix_corridor = decode_macro_corridor(
            coarsened_mg.macro_graph, x_comm, c_prefix, c_dst,
            h_m_budget=30, mu=1.5, nu=1.0, mean_macro_lat=coarsened_mg.mean_macro_latency
        )
        # Import micro decode
        from src.decoder.micro_decoder import decode_micro_path
        suffix_path = decode_micro_path(
            coarsened_mg, x_gw, suffix_corridor, last_prefix_node, dst_node,
            mu=1.5, nu=1.0, mean_lat_ref=coarsened_mg.mean_macro_latency
        )
        repaired_path = erase_loops(prefix_nodes[:-1] + suffix_path)
        
        # Reinforce adopted detour gateways in key vector
        for node in suffix_path:
            gw_idx = coarsened_mg.gw_to_idx.get(node)
            if gw_idx is not None and K + gw_idx < len(repaired_key):
                repaired_key[K + gw_idx] = max(repaired_key[K + gw_idx], 0.8 * ub)
                
        return repaired_path, repaired_key, True, "case_b"
    except Exception:
        return path, x_key, False, "none"
