#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
RLE-EMO: Standalone Ablation Study
==================================

This script runs ONLY the six-variant ablation study of the RLE-EMO
algorithm, using the LHS-guided initialization and all other components
exactly as defined in v5.3 of the benchmark suite.

It is self-contained: it does not run NSGA-II / MOEA/D / RVEA, it does
not produce Pareto-front or convergence plots, and it does not write
the benchmark summary tables. It produces only:

    ablation_summary.txt    problem x variant table with HV / IGD / Card
    ablation_heatmap.png    per-problem normalized mean-HV heatmap
    ablation_heatmap.pdf    vector version of the same
    ablation_results.json   raw per-run numbers (JSON)

Variants
--------
    full                 all five components enabled (reference)
    no_archive_return    do not include the diversity archive in the
                         return set; return the current population only
    no_region_select     replace region-based selection with crowding
                         distance during environmental selection
    no_lhs_control       replace the LHS subsample with uniform random
    no_local_search      disable the periodic coordinate-descent search
    no_reset             disable both reset triggers

Reference front policy
----------------------
For each instance, the reference front is built as in v5.3:
    - analytic Pareto front, if reliable;
    - otherwise the non-dominated union of the best-observed fronts of
      all six variants.
The same reference front is used for every variant on a given instance,
so the comparison is fair.

Usage
-----
    python rle_emo_ablation.py                 # 30 runs, 11 problems
    python rle_emo_ablation.py --runs 5
    python rle_emo_ablation.py --problems zdt1,dtlz6
    python rle_emo_ablation.py --output "D:\path\to\out"
"""

# ============================================================================
# DEPENDENCY CHECK
# ============================================================================
import importlib
import sys

_REQUIRED = ["numpy", "scipy", "matplotlib", "pymoo"]
_MISSING = []
for _pkg in _REQUIRED:
    try:
        importlib.import_module(_pkg)
    except ImportError:
        _MISSING.append(_pkg)

if _MISSING:
    print("=" * 70)
    print("MISSING REQUIRED PACKAGES")
    print("=" * 70)
    print(f"  pip install {' '.join(_MISSING)}")
    sys.exit(1)


# ============================================================================
# IMPORTS
# ============================================================================
import argparse
import json
import os
import random
import time
import warnings
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.projections
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
try:
    matplotlib.projections.get_projection_class("3d")
except (KeyError, ValueError):
    matplotlib.projections.register_projection(Axes3D)

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from scipy.stats import qmc

from pymoo.indicators.hv import HV
from pymoo.indicators.igd import IGD
from pymoo.indicators.spacing import SpacingIndicator
from pymoo.problems import get_problem
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

warnings.filterwarnings("ignore")


# ============================================================================
# GLOBAL CONSTANTS (kept identical to v5.3)
# ============================================================================
UNRELIABLE_ANALYTIC_FRONT = {"dtlz6", "dtlz7", "wfg1", "wfg9"}

BOUNDING_BOX_UPPER_FACTOR = 1.2
BOUNDING_BOX_LOWER_MARGIN = 0.5
BOUNDING_BOX_USE_REF_POINT_UPPER = True

REFERENCE_FRONT_POLICY = "auto"

DEGENERATE_THRESHOLD = 1e8

DEFAULT_OUTPUT_DIR = (
    r"D:\Container Transportation Routing Problems"
    r"\Manuscript3-A Low-carbon based Robust Optimization under uncertainty"
    r"\rle_emo_benchmark_results"
)

PALETTE = [
    "#1f77b4", "#2ca02c", "#9467bd", "#e377c2", "#bcbd22",
    "#17becf", "#d62728", "#ff7f0e", "#7f7f7f", "#8c564b",
]


# ============================================================================
# MATPLOTLIB STYLE
# ============================================================================
def style_elsevier():
    matplotlib.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif", "serif"],
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linewidth": 0.5,
        "axes.linewidth": 0.8,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.major.size": 3.5,
        "ytick.major.size": 3.5,
        "lines.linewidth": 1.6,
        "lines.markersize": 5,
        "legend.frameon": True,
        "legend.framealpha": 0.9,
        "legend.edgecolor": "0.7",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


style_elsevier()


# ============================================================================
# CONFIGURATION
# ============================================================================
@dataclass
class RLEEMOConfig:
    population_size_base: int = 40
    population_size_divisor: int = 5

    lhs_proportion: float = 0.30
    heuristic_proportion: float = 0.30
    random_proportion: float = 0.30
    opposite_proportion: float = 0.10

    p_bar_c: float = 0.85
    p_bar_m: float = 0.15

    K_regions: int = 6
    underpop_frac: float = 0.5

    tau_diversity: float = 0.3
    delta_rate: float = 0.15
    reset_min_interval: int = 10
    archive_size_ratio: float = 1.0
    diversity_reset_proportion: float = 0.2

    local_search_freq_small: int = 5
    local_search_freq_medium: int = 10
    local_search_freq_large: int = 15
    local_search_top_proportion: float = 0.10
    local_search_steps: int = 5
    local_search_step_frac: float = 0.05

    def get_population_size(self, n: int) -> int:
        return self.population_size_base + n // self.population_size_divisor

    def get_generations(self, suite: str) -> int:
        return {"DTLZ": 100, "WFG": 100, "DASCMOP": 80}.get(suite, 80)

    def get_local_search_freq(self, n: int) -> int:
        if n < 50:
            return self.local_search_freq_small
        if n < 100:
            return self.local_search_freq_medium
        return self.local_search_freq_large


@dataclass
class ExperimentConfig:
    num_runs: int = 30
    base_seed: int = 42
    rle_emo: RLEEMOConfig = None  # type: ignore

    def __post_init__(self):
        if self.rle_emo is None:
            self.rle_emo = RLEEMOConfig()

    def get_seed(self, run_id: int) -> int:
        return self.base_seed + run_id

    def get_generations_for_suite(self, suite: str) -> int:
        return self.rle_emo.get_generations(suite)

    def get_population_size(self, n_var: int) -> int:
        return self.rle_emo.get_population_size(n_var)


# ============================================================================
# NUMERICAL SAFETY UTILITIES
# ============================================================================
def summary_statistics(data) -> Dict[str, float]:
    empty = {"mean": float("nan"), "std": float("nan"),
             "min": float("nan"), "max": float("nan"),
             "median": float("nan"), "q1": float("nan"),
             "q3": float("nan"), "cv": float("nan"),
             "n": 0, "n_inf": 0,
             "ci95_low": float("nan"), "ci95_high": float("nan"),
             "valid": False}
    if data is None or len(data) == 0:
        return empty
    n_total = len(data)
    valid = [float(v) for v in data if np.isfinite(v)]
    n_inf = n_total - len(valid)
    if not valid:
        return {**empty, "n": n_total, "n_inf": n_inf}
    mean = float(np.mean(valid))
    std = float(np.std(valid, ddof=1)) if len(valid) > 1 else 0.0
    se = std / np.sqrt(len(valid)) if len(valid) > 1 else 0.0
    return {
        "mean": mean, "std": std,
        "min": float(np.min(valid)), "max": float(np.max(valid)),
        "median": float(np.median(valid)),
        "q1": float(np.percentile(valid, 25)),
        "q3": float(np.percentile(valid, 75)),
        "cv": float(std / mean) if mean != 0 else 0.0,
        "n": n_total, "n_inf": n_inf,
        "ci95_low": mean - 1.96 * se,
        "ci95_high": mean + 1.96 * se,
        "valid": True,
    }


def compute_reference_point(ref_front, n_obj, fallback: float = 1.1):
    if ref_front is not None and len(ref_front) > 0:
        rf = np.asarray(ref_front, dtype=float)
        rf = rf[np.all(np.isfinite(rf), axis=1)]
        if len(rf) > 0:
            rp = rf.max(axis=0) * 1.1
            return np.where(rp > 1e-9, rp, fallback)
    return np.full(n_obj, fallback, dtype=float)


def _build_bounding_box(reference_front, ref_point=None):
    if reference_front is None or len(reference_front) == 0:
        return None, None, False
    rf = np.asarray(reference_front, dtype=float)
    if rf.ndim == 1:
        rf = rf.reshape(1, -1)
    rf = rf[np.all(np.isfinite(rf)
                   & (np.abs(rf) < DEGENERATE_THRESHOLD), axis=1)]
    if len(rf) == 0:
        return None, None, False
    ref_min = rf.min(axis=0)
    ref_max = rf.max(axis=0)
    lower = ref_min - BOUNDING_BOX_LOWER_MARGIN
    upper = BOUNDING_BOX_UPPER_FACTOR * ref_max
    if BOUNDING_BOX_USE_REF_POINT_UPPER and ref_point is not None:
        rp = np.asarray(ref_point, dtype=float).flatten()
        if rp.shape[0] == ref_max.shape[0]:
            upper = np.minimum(upper, rp)
    span = ref_max - ref_min
    safe = span > 1e-12
    lower = np.where(safe, lower, ref_min - 1e-6)
    upper = np.where(safe, upper, ref_max + 1e-6)
    return lower, upper, True


def filter_solutions(F, reference_front, ref_point=None):
    if F is None or len(F) == 0:
        return np.zeros((0, 0))
    F = np.asarray(F, dtype=float)
    if F.ndim == 1:
        F = F.reshape(1, -1)
    finite_mask = np.all(np.isfinite(F)
                         & (np.abs(F) < DEGENERATE_THRESHOLD), axis=1)
    F = F[finite_mask]
    if len(F) == 0:
        return F
    lower, upper, ok = _build_bounding_box(reference_front, ref_point)
    if not ok or lower.shape[0] != F.shape[1]:
        return F
    inside = np.all((F >= lower[None, :]) & (F <= upper[None, :]), axis=1)
    return F[inside]


def save_figure(fig, save_path: str):
    base, _ = os.path.splitext(save_path)
    try:
        fig.savefig(base + ".png", dpi=300, bbox_inches="tight")
    except Exception as e:
        print(f"    [warn] PNG save failed for {base}: {e}")
    try:
        fig.savefig(base + ".pdf", bbox_inches="tight")
    except Exception as e:
        print(f"    [warn] PDF save failed for {base}: {e}")
    plt.close(fig)


# ============================================================================
# PROBLEM WRAPPER
# ============================================================================
class PymooProblemWrapper:
    def __init__(self, pymoo_problem, name: str, suite: str):
        self._problem = pymoo_problem
        self.name = name
        self.suite = suite
        self.n_obj = int(pymoo_problem.n_obj)
        self.n_var = int(pymoo_problem.n_var)
        self.n_constr = int(getattr(pymoo_problem, "n_constr", 0) or 0)
        self.xl = np.asarray(pymoo_problem.xl, dtype=float).flatten()
        self.xu = np.asarray(pymoo_problem.xu, dtype=float).flatten()

    def evaluate(self, x):
        try:
            arr = np.asarray(x, dtype=float).reshape(1, -1)
            F = self._problem.evaluate(arr, return_values_of=["F"])
            f = np.asarray(F, dtype=float).flatten()
            return [float(v) if np.isfinite(v) else 1e10 for v in f]
        except Exception:
            return [1e10] * self.n_obj

    def analytic_pareto_front(self, n_points: int = 1000):
        try:
            pf = self._problem.pareto_front(n_pareto_points=n_points)
            if pf is None or len(pf) == 0:
                return None
            return np.asarray(pf, dtype=float)
        except Exception:
            return None

    @property
    def pymoo(self):
        return self._problem


def make_problem(problem_name: str, **kwargs) -> PymooProblemWrapper:
    name_lower = problem_name.lower()
    if name_lower.startswith("dascmop"):
        idx = int("".join(c for c in name_lower if c.isdigit()) or "1")
        try:
            from pymoo.problems.multi import dascmop as dm
            cls = getattr(dm, f"DASCMOP{idx}", None)
            if cls is None:
                raise ValueError(f"DASCMOP{idx} not found in pymoo")
            p = None
            for attempt in (lambda: cls(),
                            lambda: cls(difficulty=1),
                            lambda: cls(difficulty_factors=(1, 1, 1))):
                try:
                    p = attempt()
                    break
                except TypeError:
                    continue
            if p is None:
                raise ValueError(f"Cannot instantiate DASCMOP{idx}")
        except Exception as e:
            raise ValueError(f"Cannot create DASCMOP{idx}: {e}")
    else:
        try:
            p = get_problem(name_lower, **kwargs)
        except Exception as e:
            raise ValueError(f"Cannot create '{problem_name}': {e}")
    suite = ("ZDT" if name_lower.startswith("zdt")
             else "DTLZ" if name_lower.startswith("dtlz")
             else "WFG" if name_lower.startswith("wfg")
             else "DASCMOP" if name_lower.startswith("dascmop")
             else "Unknown")
    return PymooProblemWrapper(p, name=problem_name, suite=suite)


# ============================================================================
# RLE-EMO ALGORITHM (identical to v5.3)
# ============================================================================
class RLEEMO:
    def __init__(self, problem: PymooProblemWrapper,
                 config: RLEEMOConfig, suite: str = "ZDT",
                 max_generations: Optional[int] = None,
                 seed: Optional[int] = None,
                 use_archive_return: bool = True,
                 use_region_select: bool = True,
                 use_lhs_control: bool = True,
                 use_local_search: bool = True,
                 use_reset: bool = True):

        self.problem = problem
        self.config = config
        self.suite = suite
        self.n_var = problem.n_var
        self.n_obj = problem.n_obj

        self.population_size = config.get_population_size(self.n_var)
        self.max_generations = (max_generations
                                if max_generations is not None
                                else config.get_generations(suite))

        self.local_search_freq = config.get_local_search_freq(self.n_var)
        self.seed = seed if seed is not None else 42
        self.rng = np.random.default_rng(self.seed)

        self.use_archive_return = use_archive_return
        self.use_region_select = use_region_select
        self.use_lhs_control = use_lhs_control
        self.use_local_search = use_local_search
        self.use_reset = use_reset

        self.region_weights = self._make_region_weights()

        self.population: List[List[float]] = []
        self.fitness: List[List[float]] = []
        self.diversity_archive: List[List[float]] = []
        self.history = {"hypervolume": []}
        self._last_reset_gen = -10**9

    def _make_region_weights(self) -> np.ndarray:
        K = self.config.K_regions
        m = self.n_obj
        if m == 2:
            w = np.column_stack([np.linspace(0, 1, K),
                                 np.linspace(1, 0, K)])
        elif m == 3:
            from itertools import combinations_with_replacement
            dirs, p = [], 3
            while len(dirs) < K:
                dirs = [[c / p for c in combo]
                        for combo in combinations_with_replacement(
                            range(p + 1), m)
                        if sum(combo) == p]
                p += 1
            w = np.array(dirs[:K])
        else:
            w = self.rng.dirichlet(np.ones(m), K)
        return w / np.maximum(np.linalg.norm(w, axis=1, keepdims=True),
                              1e-12)

    def _compute_diversity_state(self, pop):
        if not pop:
            return 0.0
        F = np.array([self.evaluate(s) for s in pop], dtype=float)
        F = np.nan_to_num(F, nan=1e10, posinf=1e10, neginf=-1e10)
        if np.any(np.abs(F) >= DEGENERATE_THRESHOLD):
            return 0.0
        lo, hi = F.min(axis=0), F.max(axis=0)
        rng = np.where(hi - lo > 1e-12, hi - lo, 1.0)
        F_n = (F - lo) / rng
        F_norm = F_n / np.maximum(
            np.linalg.norm(F_n, axis=1, keepdims=True), 1e-12)
        assoc = np.argmax(F_norm @ self.region_weights.T, axis=1)
        counts = np.bincount(assoc, minlength=self.config.K_regions)
        expected = len(pop) / self.config.K_regions
        threshold = max(2, int(self.config.underpop_frac * expected))
        return float(np.sum(counts < threshold)) / self.config.K_regions

    def _compute_convergence_state(self, pop):
        if not pop:
            return 0.0
        F = np.array([self.evaluate(s) for s in pop], dtype=float)
        F = np.nan_to_num(F, nan=1e10, posinf=1e10, neginf=-1e10)
        if np.any(np.abs(F) >= DEGENERATE_THRESHOLD):
            return 0.0
        lo, hi = F.min(axis=0), F.max(axis=0)
        rng = np.where(hi - lo > 1e-12, hi - lo, 1.0)
        return float(np.mean((F - lo) / rng))

    def _adaptive_operators(self, gen):
        xi_conv = self._compute_convergence_state(self.population)
        xi_div = self._compute_diversity_state(self.population)
        G = max(1, self.max_generations)
        feedback = 1.0 + xi_div - xi_conv
        feedback = max(0.5, min(1.5, feedback))
        p_c = self.config.p_bar_c * (1 - np.exp(-gen / G)) * feedback
        p_m = self.config.p_bar_m * np.exp(-gen / G) / feedback
        return max(0.5, min(0.95, p_c)), max(0.02, min(0.3, p_m))

    def _generate_lhs_solutions(self, n_samples: int) -> List[List[float]]:
        if n_samples <= 0:
            return []
        sampler = qmc.LatinHypercube(d=self.n_var, seed=self.rng)
        unit = sampler.random(n=n_samples)
        if hasattr(self.problem, "xl") and hasattr(self.problem, "xu"):
            scaled = qmc.scale(unit, self.problem.xl, self.problem.xu)
        else:
            scaled = unit
        return [row.tolist() for row in scaled]

    def _generate_heuristic_solution(self) -> List[float]:
        return [float((self.problem.xl[i] + self.problem.xu[i]) / 2)
                for i in range(self.n_var)]

    def _generate_random_solution(self) -> List[float]:
        return [float(self.rng.uniform(self.problem.xl[i],
                                       self.problem.xu[i]))
                for i in range(self.n_var)]

    def _generate_opposite_bias_solution(self) -> List[float]:
        return [float(self.problem.xu[i]
                      - self.rng.random()
                      * (self.problem.xu[i] - self.problem.xl[i]))
                for i in range(self.n_var)]

    def initialize_population(self):
        N = self.population_size
        n_lhs = max(1, int(N * self.config.lhs_proportion))
        n_heur = max(1, int(N * self.config.heuristic_proportion))
        n_rand = max(1, int(N * self.config.random_proportion))

        pop = []
        if self.use_lhs_control:
            pop += self._generate_lhs_solutions(n_lhs)
        else:
            pop += [self._generate_random_solution() for _ in range(n_lhs)]
        pop += [self._generate_heuristic_solution() for _ in range(n_heur)]
        pop += [self._generate_random_solution() for _ in range(n_rand)]
        while len(pop) < N:
            pop.append(self._generate_opposite_bias_solution())
        return pop[:N]

    def evaluate(self, x):
        return self.problem.evaluate(x)

    def _dominates(self, a, b):
        one = False
        for ai, bi in zip(a, b):
            if ai > bi:
                return False
            if ai < bi:
                one = True
        return one

    def _fast_non_dominated_sort(self, pop):
        n = len(pop)
        if n == 0:
            return []
        dominated = [set() for _ in range(n)]
        dc = [0] * n
        fit = [self.evaluate(pop[i]) for i in range(n)]
        for i in range(n):
            for j in range(i + 1, n):
                if self._dominates(fit[i], fit[j]):
                    dominated[i].add(j)
                    dc[j] += 1
                elif self._dominates(fit[j], fit[i]):
                    dominated[j].add(i)
                    dc[i] += 1
        front = [i for i in range(n) if dc[i] == 0]
        fronts = [[pop[i] for i in front]]
        while front:
            nf = []
            for i in front:
                for j in dominated[i]:
                    dc[j] -= 1
                    if dc[j] == 0:
                        nf.append(j)
            front = nf
            if front:
                fronts.append([pop[i] for i in front])
        return fronts

    def _crowding_distance(self, front):
        n = len(front)
        if n <= 2:
            return [float("inf")] * n
        dist = [0.0] * n
        obj = [self.evaluate(s) for s in front]
        m = len(obj[0])
        for k in range(m):
            order = sorted(range(n), key=lambda i: obj[i][k])
            dist[order[0]] = float("inf")
            dist[order[-1]] = float("inf")
            lo, hi = obj[order[0]][k], obj[order[-1]][k]
            rng = hi - lo if hi > lo else 1.0
            for i in range(1, n - 1):
                idx = order[i]
                dist[idx] += (obj[order[i + 1]][k]
                              - obj[order[i - 1]][k]) / rng
        return dist

    def _region_select(self, front, n_take):
        if len(front) <= n_take:
            return front[:n_take]
        F = np.array([self.evaluate(s) for s in front], dtype=float)
        F = np.nan_to_num(F, nan=1e10, posinf=1e10, neginf=-1e10)
        lo, hi = F.min(axis=0), F.max(axis=0)
        rng = np.where(hi - lo > 1e-12, hi - lo, 1.0)
        F_n = (F - lo) / rng
        F_norm = F_n / np.maximum(
            np.linalg.norm(F_n, axis=1, keepdims=True), 1e-12)
        sim = F_norm @ self.region_weights.T
        assoc = np.argmax(sim, axis=1)
        dist_to_w = 1.0 - sim[np.arange(len(front)), assoc]
        counts = np.bincount(assoc, minlength=self.config.K_regions)
        order = [(counts[assoc[i]], dist_to_w[i], i)
                 for i in range(len(front))]
        order.sort()
        return [front[idx] for _, _, idx in order[:n_take]]

    def _update_diversity_archive(self):
        max_size = max(1, int(self.config.archive_size_ratio
                              * self.population_size))
        fronts = self._fast_non_dominated_sort(self.population)
        if not fronts:
            return
        min_dist = 1e-3 * np.sqrt(max(1, self.n_var))
        for sol in fronts[0]:
            arr = np.asarray(sol, dtype=float)
            too_close = False
            if self.diversity_archive:
                A = np.asarray(self.diversity_archive, dtype=float)
                if A.ndim == 2 and A.shape[1] == arr.shape[0]:
                    d = np.linalg.norm(A - arr, axis=1)
                    if len(d) > 0 and d.min() < min_dist:
                        too_close = True
            if too_close:
                continue
            if len(self.diversity_archive) < max_size:
                self.diversity_archive.append(sol)
            else:
                dist = self._crowding_distance(self.diversity_archive)
                finite = [d for d in dist if np.isfinite(d)]
                if finite:
                    idx = int(np.argmin(dist))
                    if dist[idx] < float("inf"):
                        self.diversity_archive[idx] = sol

    def _reset_population_from_archive(self, gen):
        if not self.diversity_archive:
            return
        if gen - self._last_reset_gen < self.config.reset_min_interval:
            return
        n_rep = min(int(self.config.diversity_reset_proportion
                        * self.population_size),
                    len(self.diversity_archive))
        if n_rep == 0:
            return
        fit_sums = [sum(f) for f in self.fitness]
        worst = sorted(range(len(self.population)),
                       key=lambda i: fit_sums[i], reverse=True)
        arch_idx = self.rng.choice(len(self.diversity_archive),
                                   n_rep, replace=False)
        for i, a in enumerate(arch_idx):
            if i < len(worst):
                self.population[worst[i]] = \
                    self.diversity_archive[a].copy()
                self.fitness[worst[i]] = self.evaluate(
                    self.diversity_archive[a])
        self._last_reset_gen = gen

    def _sbx(self, p1, p2, rate):
        if self.rng.random() > rate:
            return p1.copy()
        return [p1[i] if self.rng.random() < 0.5 else p2[i]
                for i in range(len(p1))]

    def _poly_mut(self, sol, rate):
        m = sol.copy()
        for i in range(len(m)):
            if self.rng.random() < rate:
                d = self.rng.uniform(-0.1, 0.1) \
                    * (self.problem.xu[i] - self.problem.xl[i])
                m[i] = float(np.clip(m[i] + d,
                                     self.problem.xl[i],
                                     self.problem.xu[i]))
        return m

    def _local_search(self, sol):
        best = list(sol)
        best_fit = self.evaluate(best)
        step_frac = self.config.local_search_step_frac
        for _ in range(self.config.local_search_steps):
            improved_any = False
            for i in range(len(best)):
                lo, hi = self.problem.xl[i], self.problem.xu[i]
                step = step_frac * (hi - lo)
                for direction in (+1, -1):
                    cand = list(best)
                    cand[i] = float(np.clip(best[i] + direction * step,
                                            lo, hi))
                    cf = self.evaluate(cand)
                    if self._dominates(cf, best_fit):
                        best, best_fit = cand, cf
                        improved_any = True
                        break
            if not improved_any:
                step_frac *= 0.5
                if step_frac < 1e-4:
                    break
        return best

    def _tournament(self, k):
        selected = []
        for _ in range(k):
            idx = self.rng.choice(len(self.population), size=2,
                                  replace=False)
            f0, f1 = self.fitness[idx[0]], self.fitness[idx[1]]
            if self._dominates(f0, f1):
                selected.append(self.population[idx[0]])
            elif self._dominates(f1, f0):
                selected.append(self.population[idx[1]])
            else:
                selected.append(self.population[idx[0]])
        return selected

    def run(self):
        self.population = self.initialize_population()
        self.fitness = [self.evaluate(s) for s in self.population]

        fronts = self._fast_non_dominated_sort(self.population)
        if fronts:
            self.diversity_archive = fronts[0][
                :max(1, int(self.config.archive_size_ratio
                            * self.population_size))]

        for gen in range(self.max_generations):
            xi_div = self._compute_diversity_state(self.population)
            xi_conv = self._compute_convergence_state(self.population)
            p_c, p_m = self._adaptive_operators(gen)

            offspring = []
            while len(offspring) < self.population_size:
                parents = self._tournament(2)
                child = self._sbx(parents[0], parents[1], p_c)
                child = self._poly_mut(child, p_m)
                offspring.append(child)

            off_fit = [self.evaluate(s) for s in offspring]
            combined = self.population + offspring
            combined_fit = self.fitness + off_fit

            fronts = self._fast_non_dominated_sort(combined)
            new_pop, new_fit = [], []
            for front in fronts:
                if len(new_pop) + len(front) <= self.population_size:
                    new_pop.extend(front)
                    new_fit.extend([self.evaluate(s) for s in front])
                else:
                    remaining = self.population_size - len(new_pop)
                    if self.use_region_select:
                        chosen = self._region_select(front, remaining)
                    else:
                        dist = self._crowding_distance(front)
                        ordered = sorted(zip(front, dist),
                                         key=lambda x: x[1],
                                         reverse=True)
                        chosen = [s for s, _ in ordered[:remaining]]
                    for s in chosen:
                        new_pop.append(s)
                        new_fit.append(self.evaluate(s))
                    break
            self.population, self.fitness = new_pop, new_fit

            self._update_diversity_archive()

            if self.use_reset:
                if xi_div > self.config.tau_diversity:
                    self._reset_population_from_archive(gen)
                feedback = 1.0 + xi_div - xi_conv
                if abs(feedback - 1.0) > self.config.delta_rate:
                    self._reset_population_from_archive(gen)

            if self.use_local_search and \
                    gen % self.local_search_freq == 0:
                n_top = max(1, int(self.population_size
                                   * self.config.local_search_top_proportion))
                top = sorted(range(len(self.population)),
                             key=lambda i: sum(self.fitness[i]))[:n_top]
                for idx in top:
                    improved = self._local_search(self.population[idx])
                    self.population[idx] = improved
                    self.fitness[idx] = self.evaluate(improved)

        if self.use_archive_return:
            combined = self.population + self.diversity_archive
        else:
            combined = self.population
        fit_arr = np.array([self.evaluate(s) for s in combined],
                           dtype=float)
        if len(fit_arr) == 0:
            return {"population": [], "fitness": []}
        fit_arr = np.nan_to_num(fit_arr, nan=1e10,
                                posinf=1e10, neginf=-1e10)
        try:
            nd_idx = NonDominatedSorting().do(
                fit_arr, only_non_dominated_front=True)
            pareto_pop = [combined[i] for i in nd_idx]
            pareto_fit = [fit_arr[i].tolist() for i in nd_idx]
        except Exception:
            pareto_pop = list(combined)
            pareto_fit = fit_arr.tolist()

        seen = set()
        uniq_pop, uniq_fit = [], []
        for s, f in zip(pareto_pop, pareto_fit):
            key = tuple(np.round(f, 8))
            if key not in seen:
                seen.add(key)
                uniq_pop.append(s)
                uniq_fit.append(f)

        return {"population": uniq_pop, "fitness": uniq_fit}


# ============================================================================
# METRICS (HV, IGD, spacing, cardinality)
# ============================================================================
def compute_metrics(fitness_array, reference_front, n_obj,
                    ref_point=None, use_filter=True) -> Dict[str, float]:
    if fitness_array is None or len(fitness_array) == 0:
        return {"hypervolume": 0.0, "igd": float("inf"),
                "spacing": 0.0, "cardinality": 0}
    F_raw = np.asarray(fitness_array, dtype=float)
    if F_raw.ndim == 1:
        F_raw = F_raw.reshape(1, -1)
    if ref_point is None:
        ref_point = compute_reference_point(reference_front, n_obj)
    if use_filter and reference_front is not None:
        F = filter_solutions(F_raw, reference_front, ref_point=ref_point)
    else:
        F = filter_solutions(F_raw, None)
    if len(F) == 0:
        return {"hypervolume": 0.0, "igd": float("inf"),
                "spacing": 0.0, "cardinality": 0}
    try:
        hv = float(HV(ref_point=ref_point)(F))
        if not np.isfinite(hv):
            hv = 0.0
    except Exception:
        hv = 0.0
    if reference_front is not None and len(reference_front) > 0:
        try:
            ref_pf = np.asarray(reference_front, dtype=float)
            ref_pf = ref_pf[np.all(np.isfinite(ref_pf), axis=1)]
            igd = (float(IGD(ref_pf)(F)) if len(ref_pf) > 0
                   else float("inf"))
            if not np.isfinite(igd):
                igd = float("inf")
        except Exception:
            igd = float("inf")
    else:
        igd = float("inf")
    try:
        sp = float(SpacingIndicator()(F))
        if not np.isfinite(sp):
            sp = 0.0
    except Exception:
        sp = 0.0
    return {"hypervolume": hv, "igd": igd, "spacing": sp,
            "cardinality": int(len(F))}


# ============================================================================
# REFERENCE FRONT CONSTRUCTION
# ============================================================================
def build_reference_fronts(problem, raw_fronts_by_variant, policy):
    n_obj = problem.n_obj
    analytic = None
    analytic_ok = False
    try:
        analytic = problem.analytic_pareto_front(1000)
        if analytic is not None and len(analytic) > 0:
            analytic = analytic[np.all(np.isfinite(analytic), axis=1)]
            if len(analytic) > 0:
                analytic_ok = True
    except Exception:
        analytic = None
    name = problem.name.lower()
    if policy == "none":
        return None, np.full(n_obj, 1.1), False
    if policy == "analytic":
        if analytic_ok:
            return analytic, compute_reference_point(analytic, n_obj), True
    if policy == "best-observed":
        best = _best_observed_front(raw_fronts_by_variant)
        if best is not None and len(best) > 0:
            return best, compute_reference_point(best, n_obj), True
        return None, np.full(n_obj, 1.1), False
    if name in UNRELIABLE_ANALYTIC_FRONT:
        best = _best_observed_front(raw_fronts_by_variant)
        if best is not None and len(best) > 0:
            return best, compute_reference_point(best, n_obj), True
        return (analytic, compute_reference_point(analytic, n_obj),
                analytic_ok)
    if analytic_ok:
        return analytic, compute_reference_point(analytic, n_obj), True
    best = _best_observed_front(raw_fronts_by_variant)
    if best is not None and len(best) > 0:
        return best, compute_reference_point(best, n_obj), True
    return None, np.full(n_obj, 1.1), False


def _best_observed_front(raw_fronts_by_variant):
    all_fronts = []
    for variant, fronts in raw_fronts_by_variant.items():
        if not fronts:
            continue
        best = _select_best_front(fronts)
        if best is not None and len(best) > 0:
            all_fronts.append(best)
    if not all_fronts:
        return None
    union = np.vstack(all_fronts)
    union = union[np.all(np.isfinite(union), axis=1)]
    union = union[np.all(np.abs(union) < DEGENERATE_THRESHOLD, axis=1)]
    if len(union) == 0:
        return None
    try:
        nd_idx = NonDominatedSorting().do(
            union, only_non_dominated_front=True)
        return union[nd_idx]
    except Exception:
        return union


def _select_best_front(fronts):
    best_hv = -np.inf
    best_front = None
    for f in fronts:
        if f is None or len(f) == 0:
            continue
        f = np.asarray(f, dtype=float)
        f = f[np.all(np.isfinite(f)
                     & (np.abs(f) < DEGENERATE_THRESHOLD), axis=1)]
        if len(f) == 0:
            continue
        try:
            nd = NonDominatedSorting().do(
                f, only_non_dominated_front=True)
            f_nd = f[nd]
            ref = f_nd.max(axis=0) * 1.1
            ref = np.where(ref > 1e-9, ref, 1.0)
            hv = float(HV(ref_point=ref)(f_nd))
            if np.isfinite(hv) and hv > best_hv:
                best_hv = hv
                best_front = f_nd
        except Exception:
            continue
    return best_front


# ============================================================================
# ABLATION RUNNER
# ============================================================================
VARIANTS = {
    "full": {},
    "no_archive_return": {"use_archive_return": False},
    "no_region_select": {"use_region_select": False},
    "no_lhs_control": {"use_lhs_control": False},
    "no_local_search": {"use_local_search": False},
    "no_reset": {"use_reset": False},
}


def run_ablation(problems, config: ExperimentConfig,
                 reference_policy: str = "auto"):
    results: Dict[str, Dict[str, Dict[str, Any]]] = {}

    for cfg in problems:
        name = cfg["name"]
        kwargs = cfg.get("kwargs", {})
        print(f"\n{'=' * 60}\nABLATION on {name}\n{'=' * 60}")
        try:
            problem = make_problem(name, **kwargs)
        except Exception as e:
            print(f"  ERROR creating {name}: {e}")
            continue

        n_obj = problem.n_obj
        suite = problem.suite
        max_gen = config.get_generations_for_suite(suite)
        n_var = problem.n_var
        pop_size = config.get_population_size(n_var)
        print(f"    budget: N_pop={pop_size}, G_max={max_gen}, "
              f"n_var={n_var}, n_obj={n_obj}")

        # Pass 1: run all variants, collect raw fronts.
        variant_raw: Dict[str, List[Dict[str, Any]]] = {}
        for variant, switches in VARIANTS.items():
            print(f"  Variant: {variant}")
            runs = []
            for run in range(config.num_runs):
                seed = config.get_seed(run)
                np.random.seed(seed)
                random.seed(seed)
                try:
                    algo = RLEEMO(problem, config.rle_emo,
                                  suite=suite,
                                  max_generations=max_gen,
                                  seed=seed, **switches)
                    res = algo.run()
                except Exception as e:
                    print(f"    [warn] {variant} run {run} failed: {e}")
                    res = {"population": [], "fitness": []}
                runs.append(res)
            variant_raw[variant] = runs

        # Pass 2: build the reference front from all variants.
        raw_fronts_by_variant: Dict[str, List[np.ndarray]] = {}
        for variant, runs in variant_raw.items():
            fronts = []
            for r in runs:
                f = r.get("fitness", [])
                if f:
                    arr = np.asarray(f, dtype=float)
                    if arr.ndim == 1:
                        arr = arr.reshape(1, -1)
                    fronts.append(arr)
                else:
                    fronts.append(np.zeros((0, n_obj)))
            raw_fronts_by_variant[variant] = fronts

        ref_front, ref_point, filter_enabled = build_reference_fronts(
            problem, raw_fronts_by_variant, reference_policy)

        lower, upper, box_ok = _build_bounding_box(ref_front, ref_point)
        if box_ok:
            print(f"    bounding box lower: {np.round(lower, 4)}")
            print(f"    bounding box upper: {np.round(upper, 4)}")
        if ref_front is not None and len(ref_front) > 0:
            print(f"    reference: shape={ref_front.shape}, "
                  f"filter={'ON' if filter_enabled else 'OFF'}")
        else:
            print(f"    reference: none, "
                  f"filter={'ON' if filter_enabled else 'OFF'}")

        results[name] = {}
        for variant, runs in variant_raw.items():
            hv_list, igd_list, card_list = [], [], []
            for r in runs:
                f = r.get("fitness", [])
                arr = (np.asarray(f, dtype=float) if f
                       else np.zeros((0, n_obj)))
                m = compute_metrics(arr, ref_front, n_obj, ref_point,
                                    use_filter=filter_enabled)
                hv_list.append(m["hypervolume"])
                igd_list.append(m["igd"])
                card_list.append(m["cardinality"])
            results[name][variant] = {
                "hv": summary_statistics(hv_list),
                "igd": summary_statistics(igd_list),
                "cardinality": summary_statistics(card_list)}
            hv_mean = results[name][variant]["hv"]["mean"]
            hv_std = results[name][variant]["hv"]["std"]
            card_mean = (results[name][variant]["cardinality"]["mean"])
            if np.isfinite(hv_mean):
                print(f"    {variant}: HV {hv_mean:.4f} "
                      f"± {hv_std:.4f}  Card {card_mean:.1f}")
            else:
                print(f"    {variant}: HV invalid  Card {card_mean}")

    return results


# ============================================================================
# OUTPUT WRITERS
# ============================================================================
def write_ablation_table(output_dir: str,
                         ablation_results: Dict[str, Any]):
    lines = ["\n" + "=" * 130,
             "ABLATION STUDY (RLE-EMO variants)",
             "=" * 130,
             f"{'Problem':<12} {'Variant':<22} {'HV Mean':<14} "
             f"{'HV Std':<14} {'IGD Mean':<14} {'Card':<8}",
             "-" * 130]
    for prob, variants in sorted(ablation_results.items()):
        for variant, vals in variants.items():
            hv = vals["hv"]
            igd = vals["igd"]
            card = vals["cardinality"]
            hv_mean = (f"{hv['mean']:.4f}" if hv["valid"]
                       else "invalid")
            hv_std = f"{hv['std']:.4f}" if hv["valid"] else "-"
            igd_mean = (f"{igd['mean']:.4f}" if igd["valid"]
                        else "invalid")
            card_mean = (f"{card['mean']:.1f}" if card["valid"]
                         else "invalid")
            lines.append(
                f"{prob:<12} {variant:<22} {hv_mean:<14} "
                f"{hv_std:<14} {igd_mean:<14} {card_mean:<8}")
        lines.append("-" * 130)
    txt = "\n".join(lines)
    print(txt)
    with open(os.path.join(output_dir, "ablation_summary.txt"), "w") as f:
        f.write(txt)


def plot_ablation_heatmap(ablation_results: Dict[str, Any],
                          output_dir: str):
    try:
        probs = list(ablation_results.keys())
        if not probs:
            return
        variants = list(ablation_results[probs[0]].keys())
        M = np.zeros((len(probs), len(variants)))
        for i, p in enumerate(probs):
            for j, v in enumerate(variants):
                m = ablation_results[p][v]["hv"]["mean"]
                M[i, j] = m if np.isfinite(m) else 0.0
        rmax = M.max(axis=1, keepdims=True)
        rmax[rmax == 0] = 1e-10
        Mn = M / rmax
        fig, ax = plt.subplots(figsize=(7.6, 6.8))
        im = ax.imshow(Mn, cmap="RdYlGn", aspect="auto",
                       vmin=0, vmax=1)
        ax.set_xticks(range(len(variants)))
        ax.set_yticks(range(len(probs)))
        ax.set_xticklabels(variants, rotation=35, ha="right")
        ax.set_yticklabels(probs)
        cbar = plt.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
        cbar.set_label("Normalized HV (vs. best variant)", fontsize=9)
        for i in range(len(probs)):
            for j in range(len(variants)):
                ax.text(j, i, f"{M[i, j]:.2f}", ha="center",
                        va="center", fontsize=7,
                        color=("black" if Mn[i, j] < 0.7
                               else "white"))
        for spine in ax.spines.values():
            spine.set_edgecolor("black")
            spine.set_linewidth(0.6)
        ax.set_title("Ablation: HV by variant "
                     "(per-instance normalized)",
                     fontweight="bold")
        save_figure(fig, os.path.join(output_dir, "ablation_heatmap.png"))
    except Exception as e:
        print(f"    [warn] ablation heatmap failed: {e}")
        plt.close("all")


def save_ablation_json(ablation_results: Dict[str, Any],
                       output_dir: str):
    def conv(o):
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o) if np.isfinite(o) else None
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, dict):
            return {k: conv(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [conv(v) for v in o]
        if isinstance(o, float):
            return o if np.isfinite(o) else None
        if isinstance(o, (int, str, bool)) or o is None:
            return o
        return str(o)

    path = os.path.join(output_dir, "ablation_results.json")
    with open(path, "w") as f:
        json.dump(conv(ablation_results), f, indent=2)
    print(f"\nRaw ablation results saved to {path}")


# ============================================================================
# MAIN
# ============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="RLE-EMO standalone ablation study.")
    parser.add_argument("--runs", type=int, default=30,
                        help="Number of independent runs per "
                             "(problem, variant) pair.")
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT_DIR,
                        help="Output directory. Default: manuscript folder.")
    parser.add_argument("--problems", type=str, default=None,
                        help="Comma-separated subset of problem names.")
    parser.add_argument("--reference-front", type=str, default="auto",
                        choices=["auto", "analytic", "best-observed",
                                 "none"])

    args, _unknown = parser.parse_known_args()

    print("=" * 90)
    print("RLE-EMO: Standalone Ablation Study")
    print("=" * 90)
    print(f"Output directory: {args.output}")
    print("=" * 90)

    config = ExperimentConfig()
    config.num_runs = args.runs

    all_problems = [
        {"name": "zdt1", "kwargs": {"n_var": 30}},
        {"name": "zdt2", "kwargs": {"n_var": 30}},
        {"name": "zdt3", "kwargs": {"n_var": 30}},
        {"name": "dtlz2", "kwargs": {"n_obj": 3, "n_var": 12}},
        {"name": "dtlz6", "kwargs": {"n_obj": 3, "n_var": 12}},
        {"name": "dtlz7", "kwargs": {"n_obj": 3, "n_var": 22}},
        {"name": "wfg1", "kwargs": {"n_obj": 3, "n_var": 10}},
        {"name": "wfg4", "kwargs": {"n_obj": 3, "n_var": 10}},
        {"name": "wfg9", "kwargs": {"n_obj": 3, "n_var": 10}},
        {"name": "dascmop1", "kwargs": {}},
        {"name": "dascmop7", "kwargs": {}},
    ]
    if args.problems:
        wanted = set(s.strip().lower()
                     for s in args.problems.split(","))
        problems = [p for p in all_problems
                    if p["name"].lower() in wanted]
    else:
        problems = all_problems

    os.makedirs(args.output, exist_ok=True)

    print(f"\nProblems: {len(problems)} | Variants: {len(VARIANTS)} "
          f"| Runs each: {config.num_runs}")
    print(f"Total runs: {len(problems) * len(VARIANTS) * config.num_runs}")
    print("=" * 60)

    t0 = time.time()
    ablation_results = run_ablation(problems, config,
                                    reference_policy=args.reference_front)
    elapsed = time.time() - t0

    write_ablation_table(args.output, ablation_results)
    plot_ablation_heatmap(ablation_results, args.output)
    save_ablation_json(ablation_results, args.output)

    print("\n" + "=" * 90)
    print("ABLATION COMPLETED SUCCESSFULLY")
    print("=" * 90)
    print(f"Elapsed time: {elapsed / 60:.1f} min")
    print(f"\nOutput directory: {args.output}")
    print("  ablation_summary.txt    ablation table "
          "(problem x variant, HV mean +/- std, IGD, cardinality)")
    print("  ablation_heatmap.png    per-problem normalized HV heatmap")
    print("  ablation_heatmap.pdf    vector version of the heatmap")
    print("  ablation_results.json   raw ablation results (JSON)")


if __name__ == "__main__":
    main()
