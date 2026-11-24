"""
Preprocessing module for authentic and synthetic datasets:
1. Brightkite connected subgraph extraction (target 1,000 nodes, 7,381 edges).
2. Complete dataset suite preprocessing and serialization into JSON / GraphML.
3. Generation of dataset manifest and integrity reports.
"""

import os
import json
import math
import random
import gzip
import logging
import networkx as nx
import numpy as np
import pandas as pd

from src.data.generators import (
    generate_rgg_topology,
    generate_hospital_topology,
    generate_ba_topology,
    _assign_siot_attributes
)
from src.data.download_datasets import compute_sha256

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

RAW_BRIGHTKITE_EDGES = os.path.join("datasets", "raw", "brightkite", "loc-brightkite_edges.txt")
PROCESSED_DIR = os.path.join("datasets", "processed")
MANIFEST_FILE = os.path.join("datasets", "manifests", "dataset_manifest.csv")
REPORTS_DIR = os.path.join("datasets", "reports")

def extract_brightkite_subgraph(target_nodes=1000, target_edges=7381, seed=427381):
    """
    Extracts a 1,000-node connected subgraph from Brightkite using BFS expansion
    from a high-degree core node, targeting ~7,381 edges.
    Preserves original node IDs and records degree distribution and component stats.
    """
    logging.info("Loading full Brightkite graph...")
    full_G = nx.Graph()
    with open(RAW_BRIGHTKITE_EDGES, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                u, v = int(parts[0]), int(parts[1])
                full_G.add_edge(u, v)
                
    total_nodes = full_G.number_of_nodes()
    total_edges = full_G.number_of_edges()
    logging.info(f"Full Brightkite loaded: {total_nodes} nodes, {total_edges} edges.")
    
    # We want a connected subgraph of exactly 1,000 nodes with high density ~7.38 avg degree
    # Deterministic search over top core seeds to find the best matching subgraph
    rng = random.Random(seed)
    np_rng = np.random.RandomState(seed)
    
    # Sort nodes by degree
    degrees = dict(full_G.degree())
    # Find candidates where degree is between 15 and 60 to avoid single-hub collapse
    candidate_seeds = [n for n, d in degrees.items() if 15 <= d <= 80]
    rng.shuffle(candidate_seeds)
    
    best_subG = None
    best_diff = float("inf")
    
    # Test BFS / snowball from candidates
    for seed_node in candidate_seeds[:40]:
        visited = []
        queue = [seed_node]
        seen = {seed_node}
        
        while queue and len(visited) < target_nodes:
            curr = queue.pop(0)
            visited.append(curr)
            nbrs = list(full_G.neighbors(curr))
            # Shuffle neighbors with fixed seed to achieve balanced expansion
            rng.shuffle(nbrs)
            for nbr in nbrs:
                if nbr not in seen:
                    seen.add(nbr)
                    queue.append(nbr)
                    if len(seen) >= target_nodes:
                        break
                        
        candidate_nodes = visited[:target_nodes]
        subG = full_G.subgraph(candidate_nodes).copy()
        
        if nx.is_connected(subG) and subG.number_of_nodes() == target_nodes:
            e_count = subG.number_of_edges()
            diam = nx.diameter(subG)
            # We want diameter close to 9 and edges close to 7381
            diff = abs(e_count - target_edges) + abs(diam - 9) * 200
            if diff < best_diff:
                best_diff = diff
                best_subG = subG
                if abs(diam - 9) <= 2 and abs(e_count - target_edges) < 500:
                    break

    if best_subG is None:
        # Fallback to largest CC BFS
        seed_node = sorted_nodes[0]
        nodes = list(nx.bfs_tree(full_G, seed_node))[:target_nodes]
        best_subG = full_G.subgraph(nodes).copy()

    actual_nodes = best_subG.number_of_nodes()
    actual_edges = best_subG.number_of_edges()
    actual_diam = nx.diameter(best_subG) if nx.is_connected(best_subG) else "N/A"
    logging.info(f"Brightkite subgraph extracted: {actual_nodes} nodes, {actual_edges} edges, diameter: {actual_diam}")
    
    # Generate 2D spring coordinates for visualization and physical simulation
    pos = nx.spring_layout(best_subG, seed=seed)
    for n in best_subG.nodes():
        best_subG.nodes[n]['pos'] = (float(pos[n][0] * 1000.0), float(pos[n][1] * 1000.0))
        best_subG.nodes[n]['orig_id'] = int(n)

    # Assign synthetic SIoT simulation attributes (clearly labeled)
    _assign_siot_attributes(best_subG, compromised_count=50, rng=np_rng, py_rng=rng, is_hospital=False)
    
    return best_subG, {
        "full_nodes": total_nodes,
        "full_edges": total_edges,
        "sub_nodes": actual_nodes,
        "sub_edges": actual_edges,
        "sub_diameter": actual_diam,
        "target_nodes": target_nodes,
        "target_edges": target_edges,
        "edge_difference": actual_edges - target_edges
    }


def serialize_graph(G, filepath_json):
    """Saves a networkx graph with all attributes into a clean JSON structure."""
    os.makedirs(os.path.dirname(filepath_json), exist_ok=True)
    data = {
        "nodes": [],
        "edges": []
    }
    for n, attrs in G.nodes(data=True):
        node_dict = {"id": n}
        for k, v in attrs.items():
            if isinstance(v, (np.floating, float)):
                node_dict[k] = float(v)
            elif isinstance(v, (np.integer, int)):
                node_dict[k] = int(v)
            elif isinstance(v, tuple):
                node_dict[k] = list(v)
            else:
                node_dict[k] = v
        data["nodes"].append(node_dict)
        
    for u, v, attrs in G.edges(data=True):
        edge_dict = {"u": u, "v": v}
        for k, v in attrs.items():
            if isinstance(v, (np.floating, float)):
                edge_dict[k] = float(v)
            elif isinstance(v, (np.integer, int)):
                edge_dict[k] = int(v)
            else:
                edge_dict[k] = v
        data["edges"].append(edge_dict)
        
    with open(filepath_json, "w") as f:
        json.dump(data, f, indent=2)
    logging.info(f"Serialized graph to {filepath_json}")


def deserialize_graph(filepath_json):
    """Loads a networkx graph from JSON."""
    with open(filepath_json, "r") as f:
        data = json.load(f)
    G = nx.Graph()
    for n in data["nodes"]:
        nid = n["id"]
        attrs = {k: v for k, v in n.items() if k != "id"}
        if "pos" in attrs and isinstance(attrs["pos"], list):
            attrs["pos"] = tuple(attrs["pos"])
        G.add_node(nid, **attrs)
    for e in data["edges"]:
        u, v = e["u"], e["v"]
        attrs = {k: v for k, v in e.items() if k not in ("u", "v")}
        G.add_edge(u, v, **attrs)
    return G


def process_all_datasets():
    """Builds and serializes all nine benchmark datasets, generating dataset_manifest.csv."""
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    os.makedirs(REPORTS_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(MANIFEST_FILE), exist_ok=True)
    
    with open("config/random_seeds.json", "r") as f:
        seeds = json.load(f)["topology_seeds"]

    manifest_records = []
    
    # 1. Brightkite
    logging.info("Processing Brightkite...")
    bk_graph, bk_meta = extract_brightkite_subgraph(seed=seeds["Brightkite"])
    bk_path = os.path.join(PROCESSED_DIR, "brightkite", "brightkite_1000.json")
    serialize_graph(bk_graph, bk_path)
    
    manifest_records.append({
        "Dataset Name": "Brightkite",
        "Original Source": "SNAP (Stanford Network Analysis Project)",
        "Download URL": "https://snap.stanford.edu/data/loc-Brightkite.html",
        "Download Date": "2026-11-23",
        "Original Filename": "loc-brightkite_edges.txt.gz",
        "File Size (Bytes)": os.path.getsize(RAW_BRIGHTKITE_EDGES),
        "SHA-256 Checksum": compute_sha256(RAW_BRIGHTKITE_EDGES),
        "Number of Nodes": bk_graph.number_of_nodes(),
        "Number of Edges": bk_graph.number_of_edges(),
        "Number of Connected Components": nx.number_connected_components(bk_graph),
        "Network Diameter": nx.diameter(bk_graph),
        "Preprocessing Procedure": "BFS expansion from dense core node",
        "Attribute Generation Method": "Simulated SIoT attributes with black-hole lures",
        "Dataset Classification": "Authentic topology + simulated SIoT attributes"
    })
    
    # 2. RGG instances
    rgg_specs = [
        ("RGG-500", 491, 1718, 19, 32),
        ("RGG-1000", 984, 3444, 31, 65),
        ("RGG-2000", 1968, 6888, 53, 130),
        ("RGG-3500", 3441, 12043, 69, 225),
        ("RGG-5000", 4912, 17192, 84, 320)
    ]
    for name, n_nodes, n_edges, diam, comp in rgg_specs:
        logging.info(f"Generating {name}...")
        G = generate_rgg_topology(
            target_nodes=n_nodes,
            target_edges=n_edges,
            target_diameter=diam,
            compromised_count=comp,
            seed=seeds[name]
        )
        path = os.path.join(PROCESSED_DIR, "rgg", f"{name.lower()}.json")
        serialize_graph(G, path)
        
        manifest_records.append({
            "Dataset Name": name,
            "Original Source": "Synthetic (Paper Specification Table I)",
            "Download URL": "N/A (Algorithmic Generation)",
            "Download Date": "2026-11-23",
            "Original Filename": f"{name.lower()}.json",
            "File Size (Bytes)": os.path.getsize(path),
            "SHA-256 Checksum": compute_sha256(path),
            "Number of Nodes": G.number_of_nodes(),
            "Number of Edges": G.number_of_edges(),
            "Number of Connected Components": nx.number_connected_components(G),
            "Network Diameter": nx.diameter(G) if nx.is_connected(G) else "N/A",
            "Preprocessing Procedure": "Random Geometric Graph with target aspect ratio and degree ~ 7",
            "Attribute Generation Method": "Synthetic SIoT device models and clustered black-hole lure zones",
            "Dataset Classification": "Synthetic"
        })

    # 3. Smart Hospital
    logging.info("Generating Hosp.-1000...")
    hosp_G = generate_hospital_topology(
        target_nodes=1000,
        target_edges=4620,
        compromised_count=40,
        seed=seeds["Hosp.-1000"]
    )
    hosp_path = os.path.join(PROCESSED_DIR, "hospital", "hosp_1000.json")
    serialize_graph(hosp_G, hosp_path)
    manifest_records.append({
        "Dataset Name": "Hosp.-1000",
        "Original Source": "Synthetic Institutional Model (Paper Section V-A)",
        "Download URL": "N/A (Algorithmic Generation)",
        "Download Date": "2026-11-23",
        "Original Filename": "hosp_1000.json",
        "File Size (Bytes)": os.path.getsize(hosp_path),
        "SHA-256 Checksum": compute_sha256(hosp_path),
        "Number of Nodes": hosp_G.number_of_nodes(),
        "Number of Edges": hosp_G.number_of_edges(),
        "Number of Connected Components": nx.number_connected_components(hosp_G),
        "Network Diameter": nx.diameter(hosp_G) if nx.is_connected(hosp_G) else "N/A",
        "Preprocessing Procedure": "Multi-tier clinical IoT placement (gateways, wearables, instruments, servers)",
        "Attribute Generation Method": "Device-category specific capacities and 40 compromised relays",
        "Dataset Classification": "Synthetic"
    })

    # 4. Barabasi-Albert instances
    ba_specs = [
        ("BA-500", 500, 4, 25),
        ("BA-1500", 1500, 4, 75)
    ]
    for name, n_nodes, m_att, comp in ba_specs:
        logging.info(f"Generating {name}...")
        ba_G = generate_ba_topology(
            target_nodes=n_nodes,
            m_attach=m_att,
            compromised_count=comp,
            seed=seeds[name]
        )
        ba_path = os.path.join(PROCESSED_DIR, "barabasi_albert", f"{name.lower().replace('-', '_')}.json")
        serialize_graph(ba_G, ba_path)
        manifest_records.append({
            "Dataset Name": name,
            "Original Source": "Synthetic Scale-Free (Barabasi-Albert 1999)",
            "Download URL": "N/A (Algorithmic Generation)",
            "Download Date": "2026-11-23",
            "Original Filename": f"{name.lower().replace('-', '_')}.json",
            "File Size (Bytes)": os.path.getsize(ba_path),
            "SHA-256 Checksum": compute_sha256(ba_path),
            "Number of Nodes": ba_G.number_of_nodes(),
            "Number of Edges": ba_G.number_of_edges(),
            "Number of Connected Components": nx.number_connected_components(ba_G),
            "Network Diameter": nx.diameter(ba_G) if nx.is_connected(ba_G) else "N/A",
            "Preprocessing Procedure": "Scale-free preferential attachment (m=4)",
            "Attribute Generation Method": "Synthetic SIoT device models and clustered black-hole lure zones",
            "Dataset Classification": "Synthetic"
        })

    # Save manifest
    df_manifest = pd.DataFrame(manifest_records)
    df_manifest.to_csv(MANIFEST_FILE, index=False)
    logging.info(f"Dataset manifest successfully saved to {MANIFEST_FILE}")

    # Generate dataset report
    report_path = os.path.join(REPORTS_DIR, "dataset_integrity_report.md")
    with open(report_path, "w") as f:
        f.write("# Dataset Integrity and Benchmark Specification Report\n\n")
        f.write("Generated automatically from preprocessed benchmark suite.\n\n")
        f.write(df_manifest.to_markdown(index=False))
        f.write("\n\n### Brightkite Extraction Notes\n")
        f.write(f"- Full SNAP Brightkite nodes: {bk_meta['full_nodes']}, edges: {bk_meta['full_edges']}\n")
        f.write(f"- Extracted connected subgraph nodes: {bk_meta['sub_nodes']}, edges: {bk_meta['sub_edges']}\n")
        f.write(f"- Target nodes: {bk_meta['target_nodes']}, target edges: {bk_meta['target_edges']}\n")
        f.write(f"- Difference from paper: {bk_meta['edge_difference']} edges\n")
    logging.info(f"Dataset integrity report saved to {report_path}")

if __name__ == "__main__":
    process_all_datasets()
