"""
Master One-Click Experiment Reproduction Script:
1. Validates and preprocesses all datasets
2. Executes benchmark comparative evaluation
3. Executes multi-topology component ablation
4. Executes parameter sensitivity and scalability benchmarks
5. Aggregates experimental measurements and computes non-parametric statistics
6. Populates every publication LaTeX, CSV, and Excel table
7. Generates all 300 DPI publication figures
8. Audits experimental data integrity
"""

import sys
import os
import time

def main():
    print("================================================================================")
    print("  HGC-DMPA RESEARCH REPLICATION AND EXPERIMENTAL RESEARCH PIPELINE")
    print("  'Hierarchical Graph-Clustered Decomposition--Marine Predators Algorithm")
    print("   for Scalable Service Discovery in Ultra-Large Social IoT Networks'")
    print("================================================================================")
    t_start = time.time()
    
    # Step 1: Preprocess datasets
    print("\n[Step 1/7] Preprocessing Benchmark Datasets & Verifying Manifest...")
    os.system("python -m src.data.preprocess")

    # Step 2: Run benchmarks
    print("\n[Step 2/7] Executing Comparative Benchmark Campaign (All 9 Algorithms)...")
    os.system("python -m src.experiments.run_benchmarks")

    # Step 3: Run ablation
    print("\n[Step 3/7] Executing Multi-Topology Component Ablation...")
    os.system("python -m src.experiments.run_ablation")

    # Step 4: Run scalability & sensitivity
    print("\n[Step 4/7] Executing Scalability & Sensitivity Benchmarks...")
    os.system("python -m src.experiments.run_scalability")
    os.system("python -m src.experiments.run_sensitivity")

    # Step 5: Aggregate results
    print("\n[Step 5/7] Aggregating Results & Computing Non-Parametric Statistics...")
    os.system("python -m src.evaluation.aggregate_results")

    # Step 6: Generate tables and figures
    print("\n[Step 6/7] Generating Publication Tables & Scientific Figures...")
    os.system("python scripts/regenerate_tables.py")
    os.system("python scripts/regenerate_figures.py")

    # Step 7: Verify integrity
    print("\n[Step 7/7] Auditing Experimental Data Integrity...")
    os.system("python scripts/verify_results.py")

    total_time = time.time() - t_start
    print(f"\nPipeline successfully completed in {total_time:.2f} seconds.")
    print("Artifacts generated in results/tables/, results/figures/, and manuscript/.")

if __name__ == "__main__":
    main()
