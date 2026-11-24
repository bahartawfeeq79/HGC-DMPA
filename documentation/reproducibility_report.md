# Reproducibility Report and Audit Verification

**Document:** `documentation/reproducibility_report.md`  
**Project:** HGC-DMPA Research Replication & Experimental Campaign  
**Author:** Experimental Research Auditor  
**Date:** November 2026

---

## 1. Environment and Software Manifest

| Component | Specification |
|---|---|
| Operating System | Windows 11 Enterprise (x64) |
| Python Environment | Python 3.12.10 (MSC v.1943 64-bit AMD64) |
| Core Numerical | `numpy` 2.4.4, `scipy` 1.15.1, `pandas` 2.2.3 |
| Graph Modeling | `networkx` 3.6.1, `python-louvain` 0.16 |
| Multi-Objective | `pymoo` 0.6.1.3 |
| Visualization | `matplotlib` 3.10.0 |
| Tabular Export | `openpyxl` 3.1.5, `tabulate` 0.10.0 |

---

## 2. Directory Structure and Reproducibility Architecture

The project adheres to the specified structure:
```
HGC-DMPA-Research/
├── config/                  # Hyperparameters, benchmark configs, random seeds
├── datasets/
│   ├── raw/                 # Downloaded SNAP Brightkite with SHA-256
│   ├── processed/           # Serialized JSON graphs with SIoT attributes
│   └── manifests/           # dataset_manifest.csv
├── src/
│   ├── data/                # Download, preprocessing, generators
│   ├── trust/               # Beta-reputation and social affinity
│   ├── graph/               # Graph model, Louvain, coarsening, gateways
│   ├── objectives/          # Trust, latency, energy, QoS, constraints
│   ├── algorithms/          # HGC-DMPA and all 8 baseline implementations
│   ├── decoder/             # Two-tier biased decoder and BA-TPR
│   └── evaluation/          # Metrics, Pareto archive, statistical tests
├── results/
│   ├── raw/                 # Individual run JSON files
│   ├── processed/           # Aggregated benchmark and ablation summaries
│   ├── statistical/         # Statistical validation, sensitivity, scalability
│   ├── tables/              # LaTeX, CSV, and Excel tables
│   └── figures/             # 300 DPI publication figures (PNG, PDF, SVG)
├── manuscript/
│   ├── practical_section.md # Complete written experimental section
│   ├── practical_section.tex# Publication-ready LaTeX source
│   └── tables/              # Insertable LaTeX table snippets
└── scripts/
    ├── run_all.py           # End-to-end execution pipeline
    ├── regenerate_tables.py # Direct experimental table generator
    ├── regenerate_figures.py# Direct scientific figure generator
    └── verify_results.py    # Result integrity and audit verifier
```

---

## 3. Step-by-Step Reproduction Instructions

1. **Verify Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
2. **Execute Automated Unit & Integration Tests:**
   ```bash
   python -m pytest tests/
   ```
3. **Download Authentic Datasets & Preprocess Benchmarks:**
   ```bash
   python -m src.data.preprocess
   ```
4. **Execute Full Experimental Campaign:**
   ```bash
   python scripts/run_all.py
   ```
5. **Audit Results and Regenerate Tables and Figures:**
   ```bash
   python scripts/verify_results.py
   python scripts/regenerate_tables.py
   python scripts/regenerate_figures.py
   ```
