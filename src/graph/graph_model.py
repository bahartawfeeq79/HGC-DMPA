"""
SIoT Graph Model:
Encapsulates physical graph, admissible graph filtering, node/edge lookups,
and per-hop latency/energy calculations.
"""

import math
import networkx as nx
from src.trust.beta_reputation import compute_all_edge_trusts
from src.trust.social_affinity import annotate_affinity_weights

class SIoTGraphModel:
    def __init__(self, physical_graph: nx.Graph,
                 t_min: float = 0.35, bat_min: float = 0.10,
                 alpha: float = 0.7, beta: float = 0.3, gamma: float = 0.5):
        self.raw_graph = physical_graph.copy()
        self.t_min = t_min
        self.bat_min = bat_min
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        
        # Annotate composite trust and affinity weights on physical graph
        compute_all_edge_trusts(self.raw_graph, alpha=self.alpha, beta=self.beta)
        annotate_affinity_weights(self.raw_graph, gamma=self.gamma)
        
        # Build admissible subgraph
        self.adm_graph = self._build_admissible_subgraph()
        
    def _build_admissible_subgraph(self) -> nx.Graph:
        """
        Filters physical graph into G_adm by removing zero-trust and low-battery nodes:
        T_base(v) >= T_min and Bat(v) >= Bat_min
        """
        adm_nodes = [
            n for n, data in self.raw_graph.nodes(data=True)
            if data.get('t_base', 1.0) >= self.t_min and data.get('bat', 1.0) >= self.bat_min
        ]
        adm_subg = self.raw_graph.subgraph(adm_nodes).copy()
        return adm_subg

    def get_hop_latency(self, u, v, payload_bits: float = 1.0e5, cpu_cycles: float = 1.0e7) -> float:
        """
        Per-hop latency l(u, v) = D_p(u, v) + B / BW(u, v) + D_q(v) + C / CPU(v)
        (Equation 8)
        """
        edge_data = self.raw_graph.get_edge_data(u, v, default={})
        node_data = self.raw_graph.nodes.get(v, {})
        
        d_p = edge_data.get('prop_delay', 0.0)
        bw_mbps = edge_data.get('bw', 20.0) # Mbps
        bw_bps = max(1.0e5, bw_mbps * 1.0e6)
        
        d_q = node_data.get('queue_delay', 0.002)
        cpu_ghz = node_data.get('cpu', 1.5) # GHz
        cpu_hz = max(1.0e8, cpu_ghz * 1.0e9)
        
        trans_delay = payload_bits / bw_bps
        proc_delay = cpu_cycles / cpu_hz
        
        return float(d_p + trans_delay + d_q + proc_delay)

    def get_hop_energy(self, u, v, payload_bits: float = 1.0e5, cpu_cycles: float = 1.0e7,
                       e_el: float = 50.0e-9, eps_amp: float = 100.0e-12, kappa: float = 1.0e-27) -> float:
        """
        Per-hop energy e(u, v) = 2 * E_el * B + eps_amp * B * ||p_u - p_v||^2 + kappa * CPU(v)^2 * C
        (Equation 9)
        """
        edge_data = self.raw_graph.get_edge_data(u, v, default={})
        node_data = self.raw_graph.nodes.get(v, {})
        
        dist = edge_data.get('distance', 50.0) # meters
        cpu_ghz = node_data.get('cpu', 1.5)
        cpu_hz = cpu_ghz * 1.0e9
        
        radio_energy = 2.0 * e_el * payload_bits
        amp_energy = eps_amp * payload_bits * (dist ** 2)
        cpu_energy = kappa * (cpu_hz ** 2) * cpu_cycles
        
        return float(radio_energy + amp_energy + cpu_energy)
