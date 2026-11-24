"""
Hierarchical Graph Coarsening:
Constructs coarse macro-graph G_M = (C, E_M) from admissible partition and boundary bridges.
Computes macro-edge prior metrics:
- Mean trust: \bar{T}(C_a, C_b)
- Mean bridge latency: \bar{l}(C_a, C_b)
- Macro-graph mean latency \bar{l}_M
Computes effective dimensionality D_eff = K + |V_gw| and compression percentage.
"""

import networkx as nx
import numpy as np
from src.graph.louvain import partition_graph_louvain, compute_weighted_modularity
from src.graph.gateway_detection import identify_boundary_bridges

class CoarsenedMacroGraph:
    def __init__(self, adm_graph: nx.Graph, physical_nodes_total: int,
                 weight_attr: str = 'affinity_weight', random_state: int = 42,
                 model_ref = None):
        self.adm_graph = adm_graph
        self.total_N = physical_nodes_total
        self.weight_attr = weight_attr
        self.model_ref = model_ref
        
        # 1. Partition admissible graph via Louvain
        self.partition = partition_graph_louvain(adm_graph, weight_attr=weight_attr, random_state=random_state)
        self.modularity = compute_weighted_modularity(adm_graph, self.partition, weight_attr=weight_attr)
        
        # Number of communities K
        self.communities = sorted(list(set(self.partition.values())))
        self.K = len(self.communities)
        
        # Invert partition: comm_id -> list of node IDs
        self.comm_to_nodes = {c: [] for c in self.communities}
        for n, c in self.partition.items():
            self.comm_to_nodes[c].append(n)
            
        # 2. Extract boundary bridges and active gateways
        self.boundary_bridges, self.gateway_nodes = identify_boundary_bridges(adm_graph, self.partition)
        self.num_gateways = len(self.gateway_nodes)
        
        # Map gateway nodes to 0-indexed gateway positions for random key decoding
        self.gw_to_idx = {gw: idx for idx, gw in enumerate(self.gateway_nodes)}
        self.comm_to_idx = {c: idx for idx, c in enumerate(self.communities)}
        
        # 3. Dimensionality metrics
        self.d_eff = self.K + self.num_gateways
        self.compression_pct = float(100.0 * (1.0 - (self.d_eff / max(1, self.total_N))))
        
        # 4. Construct macro-graph G_M
        self.macro_graph = nx.Graph()
        for c in self.communities:
            self.macro_graph.add_node(c, size=len(self.comm_to_nodes[c]))
            
        all_bridge_latencies = []
        for (c_a, c_b), bridges in self.boundary_bridges.items():
            trusts = []
            latencies = []
            for u, v in bridges:
                edge_data = adm_graph.get_edge_data(u, v, default={})
                t = edge_data.get('trust', 0.5)
                trusts.append(t)
                if model_ref is not None:
                    lat = model_ref.get_hop_latency(u, v)
                else:
                    lat = edge_data.get('prop_delay', 0.001) + 0.005
                latencies.append(lat)
                all_bridge_latencies.append(lat)
                
            mean_t = float(np.mean(trusts)) if trusts else 0.5
            mean_l = float(np.mean(latencies)) if latencies else 0.01
            
            self.macro_graph.add_edge(
                c_a, c_b,
                mean_trust=mean_t,
                mean_latency=mean_l,
                num_bridges=len(bridges),
                bridges=bridges
            )
            
        self.mean_macro_latency = float(np.mean(all_bridge_latencies)) if all_bridge_latencies else 0.01

    def get_summary_dict(self):
        return {
            "total_nodes": self.total_N,
            "adm_nodes": self.adm_graph.number_of_nodes(),
            "adm_edges": self.adm_graph.number_of_edges(),
            "communities_K": self.K,
            "gateways_count": self.num_gateways,
            "d_eff": self.d_eff,
            "compression_pct": self.compression_pct,
            "modularity_Q": self.modularity,
            "macro_edges": self.macro_graph.number_of_edges()
        }
