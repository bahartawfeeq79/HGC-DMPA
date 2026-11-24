"""
Objective 2: End-to-End Latency
f_Lat(P) = \sum_{j=1}^{m-1} l(v_j, v_{j+1})
where l(u, v) = D_p(u, v) + B / BW(u, v) + D_q(v) + C / CPU(v)
(Equations 8 and 10 in Tawfeeq 2026)
"""

from typing import List

def calculate_path_latency(path: List[int], graph_model,
                           payload_bits: float = 1.0e5, cpu_cycles: float = 1.0e7) -> float:
    """
    Computes total end-to-end path latency across all hops.
    """
    if len(path) <= 1:
        return 0.0
        
    total_lat = 0.0
    for j in range(len(path) - 1):
        u, v = path[j], path[j + 1]
        hop_lat = graph_model.get_hop_latency(u, v, payload_bits=payload_bits, cpu_cycles=cpu_cycles)
        total_lat += hop_lat
        
    return float(total_lat)
