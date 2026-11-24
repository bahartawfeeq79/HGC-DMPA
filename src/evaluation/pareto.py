"""
Pareto Archive and Reference-Vector Management:
1. Pareto dominance and non-dominated sorting
2. Crowding-distance archive pruning (N_arch = 40)
3. Augmented Tchebycheff scalarization with feasibility-first dominance (Equation 13)
4. Deterministic 40-vector Das-Dennis / uniform simplex reference weight generator
"""

import numpy as np
from typing import List, Tuple

def generate_das_dennis_4d_weights(n_weights: int = 40, seed: int = 42) -> np.ndarray:
    """
    Generates n_weights (default 40) uniformly distributed reference weight vectors
    on the 4-dimensional unit simplex: \sum_{k=1}^4 \lambda_k = 1, \lambda_k >= 0.
    Uses deterministic maximin distance subset selection from a dense Das-Dennis grid (p=6, 84 vectors)
    to yield exactly 40 maximally spaced weight vectors.
    """
    # Generate full Das-Dennis simplex grid for M=4, divisions p=6 (84 vectors)
    grid = []
    p = 6
    for i in range(p + 1):
        for j in range(p + 1 - i):
            for k in range(p + 1 - i - j):
                l = p - i - j - k
                grid.append([i / p, j / p, k / p, l / p])
    grid = np.array(grid, dtype=float)
    
    if len(grid) == n_weights:
        return grid
        
    # Maximin greedy selection of 40 vectors:
    # 1. Start with the 4 extreme vertices
    selected = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
        [0.25, 0.25, 0.25, 0.25] # Centroid
    ]
    selected_indices = []
    for s in selected:
        dists = np.linalg.norm(grid - np.array(s), axis=1)
        best_idx = int(np.argmin(dists))
        selected_indices.append(best_idx)
        
    selected_indices = list(dict.fromkeys(selected_indices))
    
    # Greedily pick vectors maximizing minimum distance to current selected set
    while len(selected_indices) < n_weights:
        curr_selected = grid[selected_indices]
        # Compute min distance of each candidate to selected set
        dists = np.min(np.linalg.norm(grid[:, None, :] - curr_selected[None, :, :], axis=2), axis=1)
        # Avoid already selected
        for idx in selected_indices:
            dists[idx] = -1.0
        best_cand = int(np.argmax(dists))
        selected_indices.append(best_cand)
        
    weights = grid[selected_indices]
    # Ensure rows sum to exactly 1.0
    weights = weights / np.sum(weights, axis=1, keepdims=True)
    return weights


def compute_tchebycheff_scalar(f_norm: np.ndarray, weight: np.ndarray,
                               z_ideal: np.ndarray, rho_tch: float = 1.0e-3) -> float:
    """
    Augmented Tchebycheff function (Equation 13):
    g^tch(F | \lambda, z^*) = max_{k} [ \lambda_k |F_k - z_k^*| ] + rho_tch * \sum_k \lambda_k |F_k - z_k^*|
    """
    diff = np.abs(f_norm - z_ideal)
    weighted_diff = weight * diff
    return float(np.max(weighted_diff) + rho_tch * np.sum(weighted_diff))


def dominates(f1: np.ndarray, cv1: float, f2: np.ndarray, cv2: float) -> bool:
    """
    Feasibility-first Pareto dominance:
    - If cv1 < cv2, solution 1 dominates solution 2
    - If cv1 == cv2 == 0: solution 1 Pareto dominates solution 2
    """
    if cv1 < cv2 - 1.0e-9:
        return True
    if cv2 < cv1 - 1.0e-9:
        return False
        
    # Both feasible (or equal CV): check Pareto dominance
    less_equal = np.all(f1 <= f2 + 1.0e-9)
    strictly_less = np.any(f1 < f2 - 1.0e-9)
    return bool(less_equal and strictly_less)


class ParetoArchive:
    """
    External Pareto Archive of strictly feasible non-dominated solutions.
    Capacity N_arch = 40, managed via crowding distance.
    """
    def __init__(self, capacity: int = 40):
        self.capacity = capacity
        self.solutions = [] # list of dicts: {'path': ..., 'F': np.ndarray, 'CV': float, 'key': ...}

    def add(self, path: List[int], F: np.ndarray, cv: float, key = None) -> bool:
        """
        Attempts to add candidate to archive:
        Only strictly feasible solutions (CV == 0) are admitted.
        """
        if cv > 1.0e-9:
            return False
            
        F = np.array(F, dtype=float)
        # Check against existing archive members
        to_remove = []
        for idx, sol in enumerate(self.solutions):
            if dominates(sol['F'], sol['CV'], F, cv):
                # Dominated by existing member
                return False
            if dominates(F, cv, sol['F'], sol['CV']):
                to_remove.append(idx)
            # Duplicate check
            elif np.allclose(sol['F'], F, atol=1.0e-6):
                return False

        # Remove dominated solutions
        self.solutions = [sol for i, sol in enumerate(self.solutions) if i not in to_remove]
        
        # Add new solution
        self.solutions.append({
            'path': list(path),
            'F': F,
            'CV': float(cv),
            'key': list(key) if key is not None else None
        })
        
        # Prune if exceeding capacity
        if len(self.solutions) > self.capacity:
            self._prune_crowding_distance()
            
        return True

    def _prune_crowding_distance(self):
        """Prunes archive down to capacity using multi-objective crowding distance."""
        if len(self.solutions) <= self.capacity:
            return
            
        M = len(self.solutions[0]['F'])
        n_sols = len(self.solutions)
        distances = np.zeros(n_sols)
        
        F_matrix = np.array([sol['F'] for sol in self.solutions])
        
        for m in range(M):
            sorted_idx = np.argsort(F_matrix[:, m])
            distances[sorted_idx[0]] = float("inf")
            distances[sorted_idx[-1]] = float("inf")
            
            f_min = F_matrix[sorted_idx[0], m]
            f_max = F_matrix[sorted_idx[-1], m]
            denom = max(1.0e-6, f_max - f_min)
            
            for i in range(1, n_sols - 1):
                distances[sorted_idx[i]] += (F_matrix[sorted_idx[i+1], m] - F_matrix[sorted_idx[i-1], m]) / denom

        # Sort by distance descending, keep top capacity
        keep_indices = np.argsort(-distances)[:self.capacity]
        self.solutions = [self.solutions[i] for i in sorted(keep_indices)]

    def get_front(self) -> np.ndarray:
        if not self.solutions:
            return np.empty((0, 4))
        return np.array([sol['F'] for sol in self.solutions])
