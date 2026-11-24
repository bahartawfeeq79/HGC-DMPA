# Dataset Provenance and Benchmark Specification Documentation

**Document:** `documentation/dataset_documentation.md`  
**Project:** HGC-DMPA Research Replication & Experimental Campaign  
**Author:** Research Scientist & Experimental Research Auditor  
**Date:** November 2026

---

## 1. Overview of the Benchmark Suite

The experimental evaluation of the proposed **HGC-DMPA** framework employs nine diverse Social Internet of Things (SIoT) benchmark instances spanning three distinct topological categories:
1. **Planar Wireless Sensor Meshes with Adversarial Lure Corridors:** Random Geometric Graphs (`RGG-500`, `RGG-1000`, `RGG-2000`, `RGG-3500`, `RGG-5000`).
2. **Empirical Location-Based Social Network Graph:** SNAP `Brightkite` ($1,000$ devices, $6,847$ edges, organic community clustering).
3. **Heterogeneous Institutional Clinical Deployment & Scale-Free Hub Networks:** Smart Hospital Mesh (`Hosp.-1000`, $1.3\times$ latency deadline) and Barabási–Albert Scale-Free Graphs (`BA-500`, `BA-1500`).

---

## 2. Authentic SNAP Brightkite Dataset

### 2.1 Primary Source and Download Verification
- **Official Source:** Stanford Network Analysis Project (SNAP), Stanford University
- **Canonical URL:** `https://snap.stanford.edu/data/loc-Brightkite.html`
- **Downloaded Files:**
  - `loc-brightkite_edges.txt.gz` (1,364,607 bytes, SHA-256: `0714a1ff54225f49fae8bd228d807e6a2a7ee27b8ff3c60989cfd9d20089a229`)
  - `loc-brightkite_totalCheckins.txt.gz` (59,986,698 bytes, SHA-256: `7e35e86b74b5bc80043b7506984b054f16d2553a18e10a5f3ae121cdb0bd8361`)
- **Official Network Dimensions:** 58,228 nodes, 214,078 undirected friendship edges, 4,491,143 check-in interactions.

### 2.2 Connected Subgraph Extraction Procedure
Following the article's target of a 1,000-node connected subgraph:
1. Load the complete 58,228-node graph from `loc-brightkite_edges.txt`.
2. Compute node degrees across the full topology.
3. Apply breadth-first search (BFS) expansion initialized from seed nodes with degrees in $[15, 80]$ to capture balanced community density without collapsing into an artificial 2-hop ego-network.
4. Extract the induced subgraph on the first 1,000 unique discovered nodes.
5. Verify connected-component status and record network diameter ($h=5$) and edge count ($|E| = 6,847$).
6. Generate 2D spring layout coordinates for spatial simulation while preserving authentic node IDs.

### 2.3 Clarification on Simulated vs. Authentic Attributes
- **Authentic Data:** Social network friendship connectivity and user interaction topology from SNAP.
- **Simulated Attributes:** Physical wireless SIoT device attributes (CPU frequency, RAM, residual battery, queueing delay, base reputation, propagation delay, bandwidth, and co-location/co-work/ownership indicators). These attributes are synthesized using reproducible Gaussian/uniform distributions with fixed random seeds (`config/random_seeds.json`).

---

## 3. Synthetic Benchmark Instance Generators

### 3.1 Random Geometric Graphs (RGG)
- **Mathematical Model:** Devices are distributed across a 2D rectangular planar strip with aspect ratio adjusted to expand shortest-hop diameter from 19 up to 99 hops while maintaining average degree $\langle k \rangle \approx 7.0$.
- **Adversarial Modeling:** Spatially clustered compromised black-hole lure zones are embedded. Compromised nodes have base reputation $T_{\mathrm{base}} < 0.35$ and advertise falsified propagation delay $D_p^{\mathrm{adv}} = 0.5 \cdot D_p$ to attract routing traffic.

### 3.2 Smart Hospital Institutional Mesh (`Hosp.-1000`)
- **Deployment Dimensions:** $300\text{ m} \times 120\text{ m}$ clinical environment.
- **Device Breakdown:**
  - 160 Edge gateways (CPU 2.0–3.0 GHz, RAM 4–8 GB, Battery 0.8–1.0)
  - 640 Wearable patient sensors (battery-constrained: 0.12–0.55, CPU 0.4–0.9 GHz)
  - 180 Stationary medical instruments (CPU 1.2–2.0 GHz, RAM 1.5–3.0 GB)
  - 20 Edge/cloud servers (CPU 3.0–4.5 GHz, RAM 8–16 GB)
  - 40 Compromised relays ($T_{\mathrm{base}} < 0.35$)
- **Tightened Deadline:** $t_{\mathrm{dl}} = 1.3 \times \text{fastest admissible path}$.

### 3.3 Barabási–Albert Scale-Free Networks (`BA-500`, `BA-1500`)
- **Mathematical Model:** Preferential attachment with $m=4$ initial attachments per node.
- **Characteristics:** Scale-free power-law degree distribution, short network diameter ($3$--$5$ hops), and dense hub connectivity.

---

## 4. Dataset Manifest and Integrity Auditing
All preprocessed graph datasets are serialized into standard JSON structures in `datasets/processed/` and cataloged with SHA-256 checksums in `datasets/manifests/dataset_manifest.csv`.
