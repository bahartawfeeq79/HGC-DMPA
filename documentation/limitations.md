# Research Limitations and Threats to Validity

**Document:** `documentation/limitations.md`  
**Project:** HGC-DMPA Research Replication & Experimental Campaign  
**Author:** Research Scientist & Experimental Research Auditor  
**Date:** November 2026

---

## 1. Internal Validity
- **Computational Budget Equivalence:** All comparisons are enforced at exactly $B_{\mathrm{eval}} = 3,000$ evaluations. However, offline community coarsening requires a one-time pre-partitioning phase ($<0.15\text{ s}$ wall-clock time) that flat algorithms do not perform.
- **Stochastic Variation:** 30 independent runs per algorithm mitigate random seed sensitivity, verified via non-parametric Wilcoxon rank-sum testing.

---

## 2. Construct Validity
- **Multi-Metric Evaluation:** Assessing multi-objective service discovery solely by scalarized utility or latency ignores conflicting operational constraints. Combining 4D Hypervolume, Inverted Generational Distance, and hard constraint feasibility ($\mathrm{CV} = 0$) guarantees robust construct validity.
- **Adversarial Realism:** Black-hole lure zones accurately model selective forwarding and latency falsification attacks in distributed IoT networks.

---

## 3. External Validity and Generalizability
- **Static vs. Dynamic Topology:** The evaluated benchmarks capture static snapshots of SIoT deployments. In highly mobile environments with continuous link churn, community structures must be maintained incrementally rather than via full offline recalculation.
- **Simulated Hardware Attributes on Real Social Graphs:** While the Brightkite graph provides authentic social network topology, device-level hardware metrics (CPU, RAM, battery) are synthetic distributions modeled after IoT hardware standards.

---

## 4. Operational Boundaries Where Competing Methods Excel
- **Hub-Dominated Short-Diameter Topologies:** On dense, short-diameter topologies ($h \le 4$, e.g., `BA-500` and `BA-1500`), exact scalarized shortest-path enumeration (WS-KSP) performs competitively with lower computational overhead, because the number of Pareto-optimal simple paths is small and no black-hole lures lie along the primary hubs.
- **Flat Optimization on Small Graphs:** On small networks ($N < 300$), the dimensionality compression benefit ($D_{\mathrm{eff}} \approx N$) is minor, and flat metaheuristics can find feasible routes without community coarsening.
