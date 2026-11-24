"""
Topology and attribute generators for SIoT benchmark suite:
- RGG-500, RGG-1000, RGG-2000, RGG-3500, RGG-5000 (Random Geometric Graphs)
- Hosp.-1000 (Smart Hospital Heterogeneous Network)
- BA-500, BA-1500 (Barabasi-Albert Scale-Free Graphs)
- Brightkite 1000-node connected subgraph extraction and SIoT attribute synthesis
"""

import math
import random
import networkx as nx
import numpy as np

def generate_rgg_topology(target_nodes=984, target_edges=3444, target_diameter=31,
                          compromised_count=65, seed=421000, aspect_ratio=None):
    """
    Generates a Random Geometric Graph matching the paper's target specifications:
    nodes, edges, diameter, and spatially clustered compromised black-hole lure zones.
    """
    rng = np.random.RandomState(seed)
    py_rng = random.Random(seed)
    
    # Calculate aspect ratio of rectangular deployment area to produce target diameter
    # A longer strip naturally expands network diameter while maintaining degree ~ 7
    if aspect_ratio is None:
        # Approximate aspect ratio based on target diameter vs square root of nodes
        aspect_ratio = max(1.0, (target_diameter / (2.0 * math.sqrt(target_nodes / 7.0))) ** 2)
    
    width = 1000.0 * math.sqrt(aspect_ratio)
    height = 1000.0 / math.sqrt(aspect_ratio)
    
    # Estimate communication radius for average degree ~ 7:
    # <k> = pi * r^2 * (N / Area) => r = sqrt(<k> * Area / (pi * N))
    area = width * height
    r = math.sqrt(7.0 * area / (math.pi * target_nodes)) * 1.05

    best_graph = None
    best_diff = float("inf")
    
    # Iterative refinement of radius to closely match target edges / diameter
    for radius_mult in [0.92, 0.95, 0.98, 1.0, 1.02, 1.05, 1.08, 1.12]:
        test_r = r * radius_mult
        
        # Place nodes along coordinates
        # We generate target_nodes + margin, then take the largest connected component
        coords = {}
        for i in range(int(target_nodes * 1.08)):
            coords[i] = (rng.uniform(0, width), rng.uniform(0, height))
            
        G = nx.random_geometric_graph(len(coords), test_r, pos=coords, seed=seed)
        
        # Get largest connected component
        if nx.number_connected_components(G) > 0:
            largest_cc = max(nx.connected_components(G), key=len)
            subG = G.subgraph(largest_cc).copy()
            
            # Subsample or trim to exact target_nodes if slightly larger
            if subG.number_of_nodes() >= target_nodes:
                nodes_sorted = sorted(subG.nodes())
                subG = subG.subgraph(nodes_sorted[:target_nodes]).copy()
                if nx.is_connected(subG):
                    diam = nx.diameter(subG)
                    edges = subG.number_of_edges()
                    diff = abs(edges - target_edges) + abs(diam - target_diameter) * 50
                    if diff < best_diff:
                        best_diff = diff
                        best_graph = (subG, coords, test_r)

    if best_graph is None:
        # Fallback to direct construction
        coords = {i: (rng.uniform(0, width), rng.uniform(0, height)) for i in range(target_nodes)}
        G = nx.random_geometric_graph(target_nodes, r, pos=coords, seed=seed)
        largest_cc = max(nx.connected_components(G), key=len)
        subG = G.subgraph(largest_cc).copy()
    else:
        subG, coords, _ = best_graph

    # Relabel nodes to 0..N-1
    mapping = {old: new for new, old in enumerate(subG.nodes())}
    final_G = nx.relabel_nodes(subG, mapping)
    
    # Store pos in node attributes
    for n in final_G.nodes():
        final_G.nodes[n]['pos'] = coords[mapping.get(n, n)] if mapping.get(n, n) in coords else (rng.uniform(0, width), rng.uniform(0, height))

    _assign_siot_attributes(final_G, compromised_count=compromised_count, rng=rng, py_rng=py_rng, is_hospital=False)
    return final_G


def generate_hospital_topology(target_nodes=1000, target_edges=4620, compromised_count=40, seed=421111):
    """
    Generates Hosp.-1000 institutional network (300 x 120 m):
    - 160 Edge gateways
    - 640 Wearable patient sensors
    - 180 Stationary medical instruments
    - 20 Edge/cloud servers
    - 40 Compromised relays
    """
    rng = np.random.RandomState(seed)
    py_rng = random.Random(seed)
    width, height = 300.0, 120.0
    
    G = nx.Graph()
    device_types = (
        ['server'] * 20 +
        ['gateway'] * 160 +
        ['instrument'] * 180 +
        ['wearable'] * 640
    )
    py_rng.shuffle(device_types)
    
    for i, dev_type in enumerate(device_types):
        x = rng.uniform(0, width)
        y = rng.uniform(0, height)
        G.add_node(i, pos=(x, y), dev_type=dev_type)
        
    # Communication radii depend on device category:
    # Servers & Gateways: larger radius (~45m)
    # Stationary instruments: ~30m
    # Wearables: ~18m
    radii = {
        'server': 55.0,
        'gateway': 42.0,
        'instrument': 28.0,
        'wearable': 18.0
    }
    
    # Add links between nodes if distance <= min(r_u, r_v) or max
    pos = nx.get_node_attributes(G, 'pos')
    for u in range(target_nodes):
        for v in range(u + 1, target_nodes):
            dist = math.dist(pos[u], pos[v])
            max_r = max(radii[device_types[u]], radii[device_types[v]])
            min_r = min(radii[device_types[u]], radii[device_types[v]])
            # Effective communication range
            eff_r = (max_r + min_r) / 2.0
            if dist <= eff_r:
                G.add_edge(u, v)

    # Ensure connected
    if not nx.is_connected(G):
        ccs = list(nx.connected_components(G))
        for i in range(len(ccs) - 1):
            n1 = list(ccs[i])[0]
            n2 = list(ccs[i+1])[0]
            G.add_edge(n1, n2)

    _assign_siot_attributes(G, compromised_count=compromised_count, rng=rng, py_rng=py_rng, is_hospital=True)
    return G


def generate_ba_topology(target_nodes=500, m_attach=4, compromised_count=25, seed=420050):
    """
    Generates Barabasi-Albert scale-free network (BA-500, BA-1500).
    """
    rng = np.random.RandomState(seed)
    py_rng = random.Random(seed)
    G = nx.barabasi_albert_graph(target_nodes, m_attach, seed=seed)
    
    # Assign synthetic 2D coordinates using spring layout
    pos = nx.spring_layout(G, seed=seed)
    for n in G.nodes():
        G.nodes[n]['pos'] = (pos[n][0] * 1000.0, pos[n][1] * 1000.0)
        
    _assign_siot_attributes(G, compromised_count=compromised_count, rng=rng, py_rng=py_rng, is_hospital=False)
    return G


def _assign_siot_attributes(G, compromised_count, rng, py_rng, is_hospital=False):
    """
    Assigns authentic SIoT attributes to nodes and edges:
    - CPU, RAM, Battery, Queueing Delay, Base Trust
    - Propagation Delay, Bandwidth, Social Ties (CL, CW, OL)
    - Embeds spatially clustered black-hole lure zones
    """
    N = G.number_of_nodes()
    pos = nx.get_node_attributes(G, 'pos')
    
    # 1. Select spatially clustered compromised nodes
    # Pick a random center for black-hole lure zone from existing node IDs
    nodes_list = list(G.nodes())
    center_node = py_rng.choice(nodes_list)
    center_pos = pos[center_node]
    distances_to_center = [
        (math.dist(pos[n], center_pos), n) for n in nodes_list
    ]
    distances_to_center.sort()
    
    # Top 'compromised_count' closest to lure center are compromised
    compromised_set = set([n for _, n in distances_to_center[:compromised_count]])
    
    # 2. Node attributes
    for n in G.nodes():
        is_comp = (n in compromised_set)
        dev_type = G.nodes[n].get('dev_type', 'standard')
        
        if is_hospital:
            if dev_type == 'server':
                cpu = rng.uniform(3.0, 4.5)
                ram = rng.uniform(8.0, 16.0)
                bat = 1.0
            elif dev_type == 'gateway':
                cpu = rng.uniform(2.0, 3.0)
                ram = rng.uniform(4.0, 8.0)
                bat = rng.uniform(0.8, 1.0)
            elif dev_type == 'instrument':
                cpu = rng.uniform(1.2, 2.0)
                ram = rng.uniform(1.5, 3.0)
                bat = rng.uniform(0.7, 0.95)
            else: # wearable
                cpu = rng.uniform(0.4, 0.9)
                ram = rng.uniform(0.2, 0.6)
                bat = rng.uniform(0.12, 0.55)
        else:
            cpu = rng.uniform(1.0, 2.5)
            ram = rng.uniform(1.0, 3.5)
            bat = rng.uniform(0.3, 1.0)
            
        queue_delay = rng.uniform(0.001, 0.005) # 1 - 5 ms
        
        if is_comp:
            t_base = rng.uniform(0.15, 0.34) # Below T_min = 0.35
        else:
            t_base = rng.uniform(0.70, 0.98) # High trusted relay
            
        G.nodes[n]['cpu'] = float(cpu)
        G.nodes[n]['ram'] = float(ram)
        G.nodes[n]['bat'] = float(bat)
        G.nodes[n]['queue_delay'] = float(queue_delay)
        G.nodes[n]['t_base'] = float(t_base)
        G.nodes[n]['compromised'] = bool(is_comp)

    # 3. Edge attributes
    v_prop = 2.0e8 # Speed of signal in medium (200,000 km/s)
    for u, v in G.edges():
        d_euclid = math.dist(pos[u], pos[v])
        true_prop_delay = d_euclid / v_prop
        
        # If either endpoint is compromised, they advertise half latency to lure traffic
        if G.nodes[u]['compromised'] or G.nodes[v]['compromised']:
            adv_prop_delay = 0.5 * true_prop_delay
        else:
            adv_prop_delay = true_prop_delay
            
        bw = float(rng.uniform(15.0, 45.0)) # 15 to 45 Mbps
        
        # Social ties
        # Proximity increases probability of co-location
        cl = 1 if (d_euclid < 150.0 and rng.rand() < 0.6) else (1 if rng.rand() < 0.2 else 0)
        cw = 1 if rng.rand() < 0.35 else 0
        ol = 1 if rng.rand() < 0.25 else 0
        
        # Beta reputation interactions
        # Compromised nodes have more failed interactions
        if G.nodes[u]['compromised'] or G.nodes[v]['compromised']:
            s_uv = rng.randint(1, 8)
            f_uv = rng.randint(12, 35)
        else:
            s_uv = rng.randint(20, 50)
            f_uv = rng.randint(0, 3)
            
        G.edges[u, v]['distance'] = float(d_euclid)
        G.edges[u, v]['prop_delay'] = float(true_prop_delay)
        G.edges[u, v]['adv_prop_delay'] = float(adv_prop_delay)
        G.edges[u, v]['bw'] = float(bw)
        G.edges[u, v]['cl'] = int(cl)
        G.edges[u, v]['cw'] = int(cw)
        G.edges[u, v]['ol'] = int(ol)
        G.edges[u, v]['s_uv'] = int(s_uv)
        G.edges[u, v]['f_uv'] = int(f_uv)
