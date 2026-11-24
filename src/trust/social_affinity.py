"""
Social affinity weight calculation for trust-weighted Louvain partitioning:
w_{uv} = T(u, v) * (1 + \gamma * S_{uv})
(Equation 5 in Tawfeeq 2026)
"""

def compute_affinity_weight(trust: float, social_score: float, gamma: float = 0.5) -> float:
    """
    Computes edge affinity weight w_uv for community detection:
    w_uv = T(u, v) * (1 + gamma * S_uv)
    """
    return float(trust * (1.0 + gamma * social_score))


def annotate_affinity_weights(graph, gamma: float = 0.5):
    """
    Annotates each edge with its affinity weight w_uv.
    """
    for u, v, data in graph.edges(data=True):
        trust = data.get('trust', 0.5)
        s_score = data.get('s_score', 0.0)
        w_uv = compute_affinity_weight(trust, s_score, gamma=gamma)
        data['affinity_weight'] = w_uv
    return graph
