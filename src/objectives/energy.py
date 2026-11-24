"""
Objective 3: Total Path Energy Consumption
f_Ene(P) = \sum_{j=1}^{m-1} e(v_j, v_{j+1})
where e(u, v) = 2 * E_el * B + eps_amp * B * ||p_u - p_v||^2 + kappa * CPU(v)^2 * C
(Equations 9 and 10 in Tawfeeq 2026)
"""

from typing import List

def calculate_path_energy(path: List[int], graph_model,
                          payload_bits: float = 1.0e5, cpu_cycles: float = 1.0e7,
                          e_el: float = 50.0e-9, eps_amp: float = 100.0e-12, kappa: float = 1.0e-27) -> float:
    """
    Computes total energy consumption across all hops.
    """
    if len(path) <= 1:
        return 0.0
        
    total_ene = 0.0
    for j in range(len(path) - 1):
        u, v = path[j], path[j + 1]
        hop_ene = graph_model.get_hop_energy(
            u, v, payload_bits=payload_bits, cpu_cycles=cpu_cycles,
            e_el=e_el, eps_amp=eps_amp, kappa=kappa
        )
        total_ene += hop_ene
        
    return float(total_ene)
