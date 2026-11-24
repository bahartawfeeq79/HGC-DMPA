"""
Objective 4: Bottleneck QoS Matching
f_QoS(P) = (1 / 3) * (\rho_cpu + \rho_ram + \rho_bw)
where:
\rho_cpu = min(1.0, min_{v \in P} CPU(v) / CPU_ref)
\rho_ram = min(1.0, min_{v \in P} RAM(v) / RAM_ref)
\rho_bw  = min(1.0, min_{(u, v) \in P} BW(u, v) / BW_ref)
(Equation 10 in Tawfeeq 2026)
"""

from typing import List

def calculate_path_qos(path: List[int], graph,
                       cpu_ref: float = 1.5, ram_ref: float = 2.0, bw_ref: float = 20.0) -> float:
    """
    Computes bottleneck QoS matching ratio f_QoS(P) in [0, 1].
    """
    if len(path) == 0:
        return 0.0
        
    min_cpu = float("inf")
    min_ram = float("inf")
    for v in path:
        node_data = graph.nodes.get(v, {})
        c = node_data.get('cpu', 1.5)
        r = node_data.get('ram', 2.0)
        if c < min_cpu: min_cpu = c
        if r < min_ram: min_ram = r
        
    min_bw = float("inf")
    if len(path) > 1:
        for j in range(len(path) - 1):
            u, v = path[j], path[j + 1]
            edge_data = graph.get_edge_data(u, v, default={})
            b = edge_data.get('bw', 20.0)
            if b < min_bw: min_bw = b
    else:
        min_bw = bw_ref
        
    rho_cpu = min(1.0, max(0.0, min_cpu / cpu_ref))
    rho_ram = min(1.0, max(0.0, min_ram / ram_ref))
    rho_bw  = min(1.0, max(0.0, min_bw / bw_ref))
    
    qos = (rho_cpu + rho_ram + rho_bw) / 3.0
    return float(qos)
