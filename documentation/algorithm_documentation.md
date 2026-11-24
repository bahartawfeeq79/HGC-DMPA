# Algorithmic Documentation: HGC-DMPA and Baselines

**Document:** `documentation/algorithm_documentation.md`  
**Project:** HGC-DMPA Research Replication & Experimental Campaign  
**Author:** Research Scientist & Experimental Research Auditor  
**Date:** November 2026

---

## 1. HGC-DMPA Algorithmic Architecture

The Hierarchical Graph-Clustered Decomposition–Marine Predators Algorithm (**HGC-DMPA**) couples two synergistic optimization tiers:

### Tier 1: Trust-Weighted Social Community Coarsening (Offline Preprocessing)
1. **Admissible Graph Filtering:** Devices violating the hard zero-trust floor ($T_{\mathrm{base}} < T_{\min} = 0.35$) or minimum residual battery floor ($\mathrm{Bat} < \mathrm{Bat}_{\min} = 0.10$) are filtered out prior to search.
2. **Social-Trust Affinity Modularity Maximization:** Partitions the admissible subgraph $G_{\mathrm{adm}}$ into $K$ communities $\mathcal{C} = \{C_1, \dots, C_K\}$ using affinity weights $w_{uv} = T(u, v)(1 + \gamma S_{uv})$ with $\gamma = 0.5$.
3. **Macro-Graph $\mathcal{G}_M$ Construction:** Boundary bridge edges $B_{ab}$ connect adjacent communities, defining the active boundary gateway set $V_{\mathrm{gw}} = \bigcup B_{ab}$.
4. **Dimensionality Compression:** Reduces continuous decision space from $N$ physical devices down to $D_{\mathrm{eff}} = K + |V_{\mathrm{gw}}| \ll N$, achieving $>81\%$ search-space compression on ultra-large graphs.

### Tier 2: Hybrid Decomposition–MPA Search Engine
1. **Augmented Tchebycheff Decomposition:** Maintains $N_{\mathrm{pop}} = 40$ uniformly spaced reference weight vectors $\{\boldsymbol{\lambda}^1, \dots, \boldsymbol{\lambda}^{40}\}$ on the 4D unit simplex, each associated with its $T_{\mathrm{nbr}} = 8$ nearest weight neighbors.
2. **Three-Phase Brownian–Lévy Foraging Dynamics:**
   - $\tau < 1/3$: Brownian exploration (high velocity ratio).
   - $1/3 \le \tau < 2/3$: Hybrid exploration/exploitation (prey Lévy flight, predator Brownian motion).
   - $\tau \ge 2/3$: Lévy exploitation (low velocity ratio) modulated by convergence factor $\mathrm{CF} = (1 - \tau)^{2\tau}$.
3. **Two-Tier Biased Random-Key Decoding:**
   - Tier 1: Selects cycle-free macro-community corridor $\mathcal{P}_M$ guided by community keys and $(\mu, \nu)$ trust-latency priors.
   - Tier 2: Selects boundary gateway bridges via gateway keys and stitches intra-community segments via Dijkstra shortest paths under cost $c(u, v) = -\mu \ln T(u, v) + \nu \ln (l(u, v) / \overline{l})$.
4. **Lamarckian Boundary-Aware Trust-Guided Path Repair (BA-TPR):**
   - Identifies vulnerable relays ($T_{\mathrm{base}} < T_{\mathrm{rep}} = 0.70$) or latency bottlenecks.
   - Case A (Intra-community relay): local intra-cluster detour.
   - Case B (Boundary gateway): suffix corridor re-decoding.
   - Lamarckian key write-back: suppresses banned keys ($X_{i, \mathrm{ban}} \leftarrow -5.0$) and reinforces adopted detour gateways ($X_{i, w} \leftarrow 4.0$).
5. **Diversity-Adaptive FADs Perturbation:** Modulates perturbation probability $P_{\mathrm{FADs}} = 0.1 + 0.3(1 - \mathrm{clip}(D_t, 0, 1)) \in [0.1, 0.4]$ based on population key diversity $D_t$.

---

## 2. Baseline Algorithms and Implementation Notes

1. **Flat MPA (Faramarzi et al. 2020):** Evolves flat $N$-dimensional continuous keys with greedy random-key decoding and external archive ($N_{\mathrm{arch}} = 40$).
2. **Flat NSGA-II (Deb et al. 2002):** Simulated Binary Crossover ($\eta_c = 15, p_c = 0.9$) and Polynomial Mutation ($\eta_m = 20, p_m = 1/N$), fast non-dominated sorting, crowding distance.
3. **Flat NSGA-III (Deb and Jain 2014):** Reference-point niching on 4D Das-Dennis reference directions over flat keys.
4. **Flat MOEA/D (Zhang and Li 2007):** Differential evolution and polynomial mutation with augmented Tchebycheff decomposition over flat keys.
5. **Flat MOPSO (Coello Coello et al. 2004):** Particle velocities guided by personal bests and external archive leaders.
6. **Flat ZOA (Trojovska et al. 2022):** Zebra Optimization Algorithm with foraging and predator defense phases.
7. **Flat CPO (Abdel-Basset et al. 2024):** Crested Porcupine Optimizer with defensive strategies.
8. **Deterministic WS-KSP (Maristany et al. 2021):** Evaluates $k=150$ simple paths across 20 weight vectors ($20 \times 150 = 3,000$ evaluations) under normalized $(-\ln T, l, e)$ edge costs.
