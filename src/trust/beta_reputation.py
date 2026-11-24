"""
Beta-reputation and social trust calculations according to Equations (1)-(4):
- Direct trust: Bayesian Beta-reputation with neutral prior 0.5
- Indirect recommendation trust from common neighbors
- Normalized social relationship score
- Composite link trust T(u, v)
"""

import numpy as np

def compute_social_score(cl: float, cw: float, ol: float,
                         w_cl: float = 0.4, w_cw: float = 0.4, w_ol: float = 0.2) -> float:
    """
    Computes normalized social relationship score S_uv:
    S_uv = w_cl * CL_uv + w_cw * CW_uv + w_ol * OL_uv
    (Equation 1 in Tawfeeq 2026)
    """
    return float(w_cl * cl + w_cw * cw + w_ol * ol)


def compute_direct_trust(s_uv: int, f_uv: int) -> float:
    """
    Computes Bayesian Beta-reputation direct trust:
    T_dir(u, v) = (s_uv + 1) / (s_uv + f_uv + 2)
    (Equation 2 in Tawfeeq 2026)
    """
    return float((s_uv + 1.0) / (s_uv + f_uv + 2.0))


def compute_indirect_trust(u, v, graph, eps: float = 1.0e-9) -> float:
    """
    Computes indirect recommendation trust from common neighbors K = N(u) \cap N(v):
    T_ind(u, v) = \sum_{k \in K} T_dir(u, k) * T_dir(k, v) / (\sum_{k \in K} T_dir(u, k) + eps)
    If K is empty, defaults to T_dir(u, v).
    (Equation 3 in Tawfeeq 2026)
    """
    nbrs_u = set(graph.neighbors(u))
    nbrs_v = set(graph.neighbors(v))
    common = nbrs_u.intersection(nbrs_v)
    
    # If no common neighbors, fallback to direct trust
    if not common:
        edge_data = graph.get_edge_data(u, v, default={})
        s = edge_data.get('s_uv', 1)
        f = edge_data.get('f_uv', 1)
        return compute_direct_trust(s, f)
        
    num = 0.0
    den = 0.0
    for k in common:
        data_uk = graph.get_edge_data(u, k, default={})
        data_kv = graph.get_edge_data(k, v, default={})
        t_uk = compute_direct_trust(data_uk.get('s_uv', 1), data_uk.get('f_uv', 1))
        t_kv = compute_direct_trust(data_kv.get('s_uv', 1), data_kv.get('f_uv', 1))
        num += t_uk * t_kv
        den += t_uk
        
    return float(num / (den + eps))


def compute_composite_trust(t_dir: float, t_ind: float, s_uv: float,
                            alpha: float = 0.7, beta: float = 0.3) -> float:
    """
    Computes composite link trust:
    T(u, v) = [alpha * T_dir + (1 - alpha) * T_ind + beta * S_uv] / (1 + beta)
    (Equation 4 in Tawfeeq 2026)
    """
    return float((alpha * t_dir + (1.0 - alpha) * t_ind + beta * s_uv) / (1.0 + beta))


def compute_all_edge_trusts(graph, alpha: float = 0.7, beta: float = 0.3):
    """
    Calculates composite trust for every edge in the graph and annotates edge attributes.
    """
    for u, v, data in graph.edges(data=True):
        s_uv = data.get('s_uv', 1)
        f_uv = data.get('f_uv', 1)
        cl = data.get('cl', 0)
        cw = data.get('cw', 0)
        ol = data.get('ol', 0)
        
        t_dir = compute_direct_trust(s_uv, f_uv)
        t_ind = compute_indirect_trust(u, v, graph)
        s_score = compute_social_score(cl, cw, ol)
        t_comp = compute_composite_trust(t_dir, t_ind, s_score, alpha=alpha, beta=beta)
        
        data['t_dir'] = t_dir
        data['t_ind'] = t_ind
        data['s_score'] = s_score
        data['trust'] = t_comp
        
    return graph
