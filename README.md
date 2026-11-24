# HGC-DMPA Code & Benchmark Suite

This repository contains the source code, simulation scripts, and experimental benchmarks for HGC-DMPA.

## Structure

- `src/`: Algorithm and model implementation.
- `scripts/`: Execution and reproduction scripts.
- `config/`: Experiment parameters and configuration.
- `datasets/`: Benchmark graph data and manifests.
- `results/`: Output data, tables, and plots.
- `tests/`: Automated unit tests.

## Requirements

Install dependencies:
```bash
pip install -r requirements.txt
```

## Running

Run the unit tests:
```bash
pytest tests/
```

Run experiments:
```bash
python scripts/run_all.py
```
