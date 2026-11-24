# Experimental Protocol and Fair Tuning Standards

**Document:** `documentation/experimental_protocol.md`  
**Project:** HGC-DMPA Research Replication & Experimental Campaign  
**Author:** Research Scientist & Experimental Research Auditor  
**Date:** November 2026

---

## 1. Principles of Experimental Rigor

To prevent algorithmic bias, post-hoc overfitting, and reproducibility failure, all experiments adhere to strict scientific standards:
1. **Equal Evaluation Budgets ($B_{\mathrm{eval}} = 3,000$):**
   Every stochastic algorithm (HGC-DMPA, MPA, NSGA-II, NSGA-III, MOEA/D, MOPSO, ZOA, CPO) is constrained to exactly 3,000 objective evaluations per run. Deterministic WS-KSP evaluates $20 \times 150 = 3,000$ candidate paths.
2. **Independent Tuning Instances (Zero Data Leakage):**
   All hyperparameter tuning was conducted on four separate seed graphs (`RGG-300`, `RGG-800`, `Hosp.-1000-seed`, `BA-500-seed`) prior to testing on the unseen benchmark suite. No parameters were altered after viewing benchmark test results.
3. **No Fabrication of Results:**
   Incomplete $[VAL]$ placeholders from paper drafts are never copied or assumed. Every numerical value in every table is recomputed directly from raw experimental run files.
4. **Honest Reporting of Failed Runs:**
   Runs failing to discover at least one strictly feasible path ($\mathrm{CV} = 0$) are recorded honestly as unsuccessful ($\mathrm{HV} = 0.0$, $\mathrm{IGD} = \text{--}$, $\mathrm{Feas} = 0\%$) rather than being silently omitted.
5. **Identical Objective Normalization:**
   All algorithms are evaluated against a shared empirical reference Pareto front $\mathcal{P}^* = \text{NonDominated}(\bigcup \mathcal{A}_{\text{alg}})$ constructed per instance, normalized against its true ideal and nadir bounds.

---

## 2. Statistical Testing Standards

- **Pairwise Stochastic Comparisons:** Two-sided Wilcoxon rank-sum test across 30 independent runs.
- **Pairwise Deterministic Comparisons:** One-sample Wilcoxon signed-rank test against the deterministic WS-KSP result.
- **Multiple Testing Correction:** Holm-Bonferroni step-down adjustment controlling Family-Wise Error Rate (FWER) at $\alpha = 0.05$.
- **Effect Size:** Non-parametric Vargha-Delaney $\hat{A}_{12}$ effect size ($\hat{A}_{12} > 0.71$ large superiority).
- **Omnibus Multi-Algorithm Ranking:** Non-parametric Friedman test across all stochastic baselines.
