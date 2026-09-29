#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
RLE-EMO: Comprehensive Benchmark Suite (v5.3)
=============================================

Change log v5.2 -> v5.3
-----------------------
1. **Initialization is now LHS-guided, not PPO-guided.** The previous
   version (v5.2) still used `PPOAgent` in the initialization step, which
   contradicted the manuscript's Section 3.3, Table 3.3, Table 4.3, and
   Algorithm 1. The `PPOAgent` and `PPOTrainer` classes have been removed
   entirely. Initialization now draws an LHS subsample via
   `scipy.stats.qmc.LatinHypercube`, exactly as the manuscript describes.

2. **No offline cost, no warm-start pool.** With the PPO trainer removed,
   the initialization is fully deterministic given the run seed and there
   is no amortized offline cost. This brings the code into agreement with
   the manuscript's Section 3.5 and Section 4.4.

3. All other components (region-based selection, dual reset triggers,
   diversity archive, coordinate-descent local search, adaptive operator
   rates, bounding-box filter, two-pass reference construction,
   pymoo-native baselines, statistics) are unchanged from v5.2.

Manuscript alignment (verified):
    - Initialization:    LHS 30%, heuristic 30%, random 30%, opposite 10%
    - Budget rule:       N_pop = 40 + floor(n_var/5)
                         G_max = 100 for DTLZ, WFG; 80 for ZDT, DASCMOP
    - Bounding box:      1.2 * max(ref) upper, min(ref) - 0.5 lower
    - Reference point:   1.1 * max(ref) per objective
    - Degenerate guard:  |f_i| > 1e8
    - Reference front:   analytic, best-observed for DTLZ6/7, WFG1/9
    - Algorithm 1:       Phase I (adaptive), Phase II (LHS ensemble),
                         Phase III (adaptive NSGA-II + archive + local search)

Outputs go to the manuscript folder by default; override with --output.
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

_OPTIONAL = ["scikit_posthocs", "pandas"]
_OPTIONAL_MISSING = []
for _pkg in _OPTIONAL:
    try:
        importlib.import_module(_pkg)
    except ImportError:
        _OPTIONAL_MISSING.append(_pkg)
if _OPTIONAL_MISSING:
    print("[info] Optional packages not installed: "
          f"{', '.join(_OPTIONAL_MISSING)}")
    print("       Nemenyi post-hoc and CSV export will be skipped.")


# ============================================================================
# IMPORTS
# ============================================================================
import argparse
import json
import os
import random
import time
import warnings
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.projections
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
try:
    matplotlib.projections.get_projection_class("3d")
except (KeyError, ValueError):
    matplotlib.projections.register_projection(Axes3D)


def _safe_set_rcparam(key, value):
    try:
        if key in matplotlib.rcParams:
            matplotlib.rcParams[key] = value
    except Exception:
        pass


_safe_set_rcparam("axes3d.depthshade_minalpha", 0.3)
_safe_set_rcparam("axes3d.depthshade", True)

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from scipy.spatial import distance
from scipy.stats import qmc  # LHS sampler

from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.moo.moead import MOEAD
from pymoo.algorithms.moo.rvea import RVEA
from pymoo.indicators.hv import HV
from pymoo.indicators.igd import IGD
from pymoo.indicators.spacing import SpacingIndicator
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.problems import get_problem
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from pymoo.util.ref_dirs import get_reference_directions

warnings.filterwarnings("ignore")


# ============================================================================
# GLOBAL SWITCHES AND POLICIES
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
# MATPLOTLIB STYLE (Elsevier friendly)
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
_safe_set_rcparam("axes3d.depthshade_minalpha", 0.3)


# ============================================================================
# CONFIGURATION
# ============================================================================
@dataclass
class RLEEMOConfig:
    population_size_base: int = 40
    population_size_divisor: int = 5

    # LHS-guided ensemble proportions
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
    num_test_points: int = 1000
    base_seed: int = 42
    rle_emo: RLEEMOConfig = field(default_factory=RLEEMOConfig)

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
    rf = rf[np.all(np.isfinite(rf) & (np.abs(rf) < DEGENERATE_THRESHOLD),
                   axis=1)]
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


def filter_solutions(F, reference_front, ref_point=None, rel_tol=None):
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


def filter_solutions_with_report(F, reference_front, ref_point=None):
    if F is None or len(F) == 0:
        return np.zeros((0, 0)), None, None, 0
    F = np.asarray(F, dtype=float)
    if F.ndim == 1:
        F = F.reshape(1, -1)
    finite_mask = np.all(np.isfinite(F)
                         & (np.abs(F) < DEGENERATE_THRESHOLD), axis=1)
    F_finite = F[finite_mask]
    n_nonfinite = int(np.sum(~finite_mask))
    lower, upper, ok = _build_bounding_box(reference_front, ref_point)
    if not ok or lower.shape[0] != F_finite.shape[1]:
        return F_finite, lower, upper, n_nonfinite
    inside = np.all((F_finite >= lower[None, :])
                    & (F_finite <= upper[None, :]), axis=1)
    n_outside = int(np.sum(~inside))
    return F_finite[inside], lower, upper, n_nonfinite + n_outside


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
# RLE-EMO ALGORITHM (LHS-guided, no PPO)
# ============================================================================
class RLEEMO:
    """
    RLE-EMO with LHS-guided ensemble initialization.

    All attributes and behaviors match the manuscript:
        - Phase I:   size-adaptive configuration
        - Phase II:  LHS-guided ensemble initialization (30/30/30/10)
        - Phase III: adaptive NSGA-II with region-based selection,
                     dual reset triggers, diversity archive, and
                     periodic coordinate-descent local search.

    The `use_*` flags below allow the ablation study to switch each
    component on or off; the default `full` variant activates all.
    """

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
        self.history = {"hypervolume": [], "diversity": [], "convergence": []}
        self._last_reset_gen = -10**9

    # ------------------------------------------------------------------
    # Region weights
    # ------------------------------------------------------------------
    def _make_region_weights(self) -> np.ndarray:
        K = self.config.K_regions
        m = self.n_obj
        if m == 2:
            w = np.column_stack([np.linspace(0, 1, K), np.linspace(1, 0, K)])
        elif m == 3:
            from itertools import combinations_with_replacement
            dirs, p = [], 3
            while len(dirs) < K:
                dirs = [[c / p for c in combo]
                        for combo in combinations_with_replacement(range(p + 1), m)
                        if sum(combo) == p]
                p += 1
            w = np.array(dirs[:K])
        else:
            w = self.rng.dirichlet(np.ones(m), K)
        return w / np.maximum(np.linalg.norm(w, axis=1, keepdims=True), 1e-12)

    # ------------------------------------------------------------------
    # State computations
    # ------------------------------------------------------------------
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
        F_norm = F_n / np.maximum(np.linalg.norm(F_n, axis=1, keepdims=True), 1e-12)
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

    # ------------------------------------------------------------------
    # Initialization: LHS-guided ensemble (matches manuscript Sec. 3.3)
    # ------------------------------------------------------------------
    def _generate_lhs_solutions(self, n_samples: int) -> List[List[float]]:
        """
        Draw `n_samples` Latin hypercube points and scale them to the
        problem's bounding box. Deterministic given self.rng's seed.
        """
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
        """Midpoint of the bounding box (nearest-neighbor heuristic proxy)."""
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
            # Ablation: replace the LHS subsample with uniform random.
            pop += [self._generate_random_solution() for _ in range(n_lhs)]
        pop += [self._generate_heuristic_solution() for _ in range(n_heur)]
        pop += [self._generate_random_solution() for _ in range(n_rand)]
        while len(pop) < N:
            pop.append(self._generate_opposite_bias_solution())
        return pop[:N]

    # ------------------------------------------------------------------
    # Core evolutionary machinery
    # ------------------------------------------------------------------
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
        F_norm = F_n / np.maximum(np.linalg.norm(F_n, axis=1, keepdims=True), 1e-12)
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
                self.population[worst[i]] = self.diversity_archive[a].copy()
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

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------
    def run(self):
        self.population = self.initialize_population()
        self.fitness = [self.evaluate(s) for s in self.population]

        fronts = self._fast_non_dominated_sort(self.population)
        if fronts:
            self.diversity_archive = fronts[0][
                :max(1, int(self.config.archive_size_ratio
                            * self.population_size))]

        best_so_far = 0.0

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
                                         key=lambda x: x[1], reverse=True)
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

            if self.use_local_search and gen % self.local_search_freq == 0:
                n_top = max(1, int(self.population_size
                                   * self.config.local_search_top_proportion))
                top = sorted(range(len(self.population)),
                             key=lambda i: sum(self.fitness[i]))[:n_top]
                for idx in top:
                    improved = self._local_search(self.population[idx])
                    self.population[idx] = improved
                    self.fitness[idx] = self.evaluate(improved)

            hv_val = _compute_internal_hv(np.array(self.fitness), self.n_obj)
            self.history["hypervolume"].append(hv_val)
            best_so_far = max(best_so_far, hv_val)
            self.history.setdefault("hypervolume_best", []).append(best_so_far)
            self.history["diversity"].append(xi_div)
            self.history["convergence"].append(xi_conv)

        if self.use_archive_return:
            combined = self.population + self.diversity_archive
        else:
            combined = self.population
        fit_arr = np.array([self.evaluate(s) for s in combined], dtype=float)
        if len(fit_arr) == 0:
            return {"population": [], "fitness": [],
                    "history": self.history, "hypervolume": 0.0}
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

        return {"population": uniq_pop, "fitness": uniq_fit,
                "history": self.history, "hypervolume": 0.0}


def _compute_internal_hv(fitness_array, n_obj):
    if fitness_array is None or len(fitness_array) == 0:
        return 0.0
    try:
        F = np.asarray(fitness_array, dtype=float)
        F = F[np.all(np.isfinite(F)
                     & (np.abs(F) < DEGENERATE_THRESHOLD), axis=1)]
        if len(F) == 0:
            return 0.0
        nds = NonDominatedSorting().do(F, only_non_dominated_front=True)
        nd_fit = F[nds]
        ref = nd_fit.max(axis=0) * 1.1
        ref = np.where(ref > 1e-9, ref, 1.0)
        hv = HV(ref_point=ref)(nd_fit)
        return float(hv) if np.isfinite(hv) else 0.0
    except Exception:
        return 0.0


# ============================================================================
# STANDARD BASELINES (pymoo native)
# ============================================================================
class PymooBaseline:
    def __init__(self, problem_wrapper, config, name,
                 max_generations=None):
        self.problem = problem_wrapper
        self.pymoo_problem = problem_wrapper.pymoo
        self.config = config
        self.name = name
        self.n_var = problem_wrapper.n_var
        self.n_obj = problem_wrapper.n_obj
        self.n_constr = problem_wrapper.n_constr
        self.population_size = config.get_population_size(self.n_var)
        self.max_generations = (max_generations
                                if max_generations is not None
                                else config.get_generations_for_suite(
                                    problem_wrapper.suite))
        self.algorithm = self._build_algorithm()

    def _build_algorithm(self):
        N = self.population_size
        sampling = FloatRandomSampling()
        crossover = SBX(prob=0.9, eta=15)
        mutation = PM(eta=20)
        if self.name == "nsga2":
            return NSGA2(pop_size=N, sampling=sampling,
                         crossover=crossover, mutation=mutation,
                         eliminate_duplicates=True)
        if self.name == "moead":
            if self.n_constr > 0:
                raise ValueError(
                    f"MOEA/D does not support constrained problems "
                    f"(n_constr={self.n_constr}); skipping.")
            n_partitions = max(2, int(round(
                N ** (1.0 / max(1, self.n_obj - 1)))))
            ref_dirs = get_reference_directions(
                "das-dennis", self.n_obj, n_partitions=n_partitions)
            return MOEAD(ref_dirs=ref_dirs,
                         n_neighbors=min(15, len(ref_dirs)),
                         prob_neighbor_mating=0.9,
                         sampling=sampling, crossover=crossover,
                         mutation=mutation)
        if self.name == "rvea":
            n_partitions = max(2, int(round(
                N ** (1.0 / max(1, self.n_obj - 1)))))
            ref_dirs = get_reference_directions(
                "das-dennis", self.n_obj, n_partitions=n_partitions)
            return RVEA(ref_dirs=ref_dirs, sampling=sampling,
                        crossover=crossover, mutation=mutation)
        raise ValueError(f"Unknown baseline: {self.name}")

    def run(self):
        from pymoo.optimize import minimize
        res = minimize(self.pymoo_problem, self.algorithm,
                       ("n_gen", self.max_generations),
                       seed=None, verbose=False, save_history=False)
        F = (np.asarray(res.F, dtype=float)
             if res.F is not None else np.zeros((0, self.n_obj)))
        X = (np.asarray(res.X, dtype=float)
             if res.X is not None else np.zeros((0, self.n_var)))
        return {"population": X.tolist(), "fitness": F.tolist(),
                "history": {"hypervolume": [0.0] * self.max_generations},
                "hypervolume": 0.0}


# ============================================================================
# METRICS
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
def build_reference_fronts(problem, raw_fronts_by_alg, policy):
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
        best = _best_observed_front(raw_fronts_by_alg)
        if best is not None and len(best) > 0:
            return best, compute_reference_point(best, n_obj), True
        return None, np.full(n_obj, 1.1), False
    if name in UNRELIABLE_ANALYTIC_FRONT:
        best = _best_observed_front(raw_fronts_by_alg)
        if best is not None and len(best) > 0:
            return best, compute_reference_point(best, n_obj), True
        return analytic, compute_reference_point(analytic, n_obj), analytic_ok
    if analytic_ok:
        return analytic, compute_reference_point(analytic, n_obj), True
    best = _best_observed_front(raw_fronts_by_alg)
    if best is not None and len(best) > 0:
        return best, compute_reference_point(best, n_obj), True
    return None, np.full(n_obj, 1.1), False


def _best_observed_front(raw_fronts_by_alg):
    all_fronts = []
    for alg_name, fronts in raw_fronts_by_alg.items():
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
# STATISTICS
# ============================================================================
def holm_bonferroni(p_values):
    m = len(p_values)
    if m == 0:
        return []
    order = np.argsort(p_values)
    adjusted = np.empty(m, dtype=float)
    running = 0.0
    for rank, idx in enumerate(order):
        val = (m - rank) * float(p_values[idx])
        running = max(running, val)
        adjusted[idx] = min(running, 1.0)
    return adjusted.tolist()


def cohens_d(x, y):
    x = [v for v in x if np.isfinite(v)]
    y = [v for v in y if np.isfinite(v)]
    if len(x) < 2 or len(y) < 2:
        return 0.0
    nx, ny = len(x), len(y)
    vx, vy = np.var(x, ddof=1), np.var(y, ddof=1)
    sp = np.sqrt(((nx - 1) * vx + (ny - 1) * vy) / (nx + ny - 2))
    return float((np.mean(x) - np.mean(y)) / sp) if sp > 0 else 0.0


def interpret_cohens_d(d):
    ad = abs(d)
    if ad < 0.2:
        return "negligible"
    if ad < 0.5:
        return "small"
    if ad < 0.8:
        return "medium"
    return "large"


# ============================================================================
# EXPERIMENT RUNNER
# ============================================================================
class ExperimentRunner:
    def __init__(self, config: Optional[ExperimentConfig] = None):
        self.config = config or ExperimentConfig()
        self.results: Dict[str, Any] = {}
        self.all_fronts: Dict[str, Any] = {}
        self.all_hv_history: Dict[str, Any] = {}
        self.all_metrics: Dict[str, Any] = {}
        self.reference_info: Dict[str, Any] = {}
        self._problem_kwargs: Dict[str, Dict[str, Any]] = {}
        np.random.seed(self.config.base_seed)
        random.seed(self.config.base_seed)

    def get_problem(self, problem_name: str, **kwargs):
        return make_problem(problem_name, **kwargs)

    def get_algorithm(self, name, problem, n, suite="ZDT",
                      max_generations=None, seed=None, ablation=None):
        name_lower = name.lower()
        if name_lower == "rle_emo":
            kwargs = dict(ablation or {})
            return RLEEMO(problem, self.config.rle_emo, suite=suite,
                          max_generations=max_generations,
                          seed=seed, **kwargs)
        if name_lower in ("nsga2", "moead", "rvea"):
            return PymooBaseline(problem, self.config, name_lower,
                                 max_generations=max_generations)
        raise ValueError(f"Unknown algorithm: {name}")

    def run_experiment(self, problem_name, algorithms, num_runs=None,
                       **kwargs):
        num_runs = num_runs or self.config.num_runs
        problem = self.get_problem(problem_name, **kwargs)
        self._problem_kwargs[problem_name] = kwargs
        n_var = problem.n_var
        n_obj = problem.n_obj
        n_constr = problem.n_constr
        suite = problem.suite
        max_generations = self.config.get_generations_for_suite(suite)
        pop_size = self.config.get_population_size(n_var)

        print(f"    budget: N_pop={pop_size}, G_max={max_generations}, "
              f"n_constr={n_constr}")

        raw_results: Dict[str, List[Dict[str, Any]]] = {}
        raw_times: Dict[str, List[float]] = {}
        not_applicable: Dict[str, bool] = {}

        for alg_name in algorithms:
            if alg_name == "moead" and n_constr > 0:
                print(f"  Skipping {alg_name} on {problem_name}: "
                      f"constrained problems not supported by pymoo MOEAD.")
                raw_results[alg_name] = []
                raw_times[alg_name] = []
                not_applicable[alg_name] = True
                continue
            print(f"  Running {alg_name} on {problem_name}...")
            alg_results, alg_times = [], []
            for run in range(num_runs):
                seed = self.config.get_seed(run)
                np.random.seed(seed)
                random.seed(seed)
                t0 = time.time()
                try:
                    algo = self.get_algorithm(alg_name, problem, n_var, suite,
                                              max_generations=max_generations,
                                              seed=seed)
                    result = algo.run()
                except Exception as e:
                    print(f"    [warn] {alg_name} run {run} failed: {e}")
                    result = {"population": [], "fitness": [],
                              "history": {"hypervolume": []},
                              "hypervolume": 0.0}
                elapsed = time.time() - t0
                alg_results.append(result)
                alg_times.append(elapsed)
            raw_results[alg_name] = alg_results
            raw_times[alg_name] = alg_times
            not_applicable[alg_name] = False

        raw_fronts_by_alg: Dict[str, List[np.ndarray]] = {}
        for alg_name, runs in raw_results.items():
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
            raw_fronts_by_alg[alg_name] = fronts

        ref_front, ref_point, filter_enabled = build_reference_fronts(
            problem, raw_fronts_by_alg, REFERENCE_FRONT_POLICY)

        policy_used = REFERENCE_FRONT_POLICY
        if policy_used == "auto":
            policy_used = ("best-observed"
                           if problem.name.lower()
                           in UNRELIABLE_ANALYTIC_FRONT
                           else "analytic")
        lower, upper, box_ok = _build_bounding_box(ref_front, ref_point)
        print(f"    reference front: policy={policy_used}, "
              f"shape={None if ref_front is None else ref_front.shape}, "
              f"filter={'ON' if filter_enabled else 'OFF'}")
        if box_ok:
            print(f"      bounding box lower: {np.round(lower, 4)}")
            print(f"      bounding box upper: {np.round(upper, 4)}")
        if ref_front is not None and len(ref_front) > 0:
            print(f"      ref range: "
                  f"{np.round(ref_front.min(axis=0), 4)} .. "
                  f"{np.round(ref_front.max(axis=0), 4)}")

        self.reference_info[problem_name] = {
            "policy": policy_used,
            "shape": None if ref_front is None else list(ref_front.shape),
            "filter_enabled": filter_enabled,
            "ref_point": ref_point.tolist(),
            "bounding_box_lower": (None if lower is None
                                   else lower.tolist()),
            "bounding_box_upper": (None if upper is None
                                   else upper.tolist()),
        }

        results: Dict[str, Any] = {}
        problem_fronts: Dict[str, Any] = {}
        problem_history: Dict[str, Any] = {}
        metrics_by_alg: Dict[str, Any] = {}

        for alg_name in algorithms:
            if not_applicable.get(alg_name, False):
                empty = summary_statistics([])
                results[alg_name] = {
                    "results": [], "metrics": [], "times": [],
                    "summary": {"hypervolume": dict(empty),
                                "igd": dict(empty),
                                "spacing": dict(empty),
                                "cardinality": dict(empty),
                                "time": dict(empty)},
                    "not_applicable": True}
                metrics_by_alg[alg_name] = []
                problem_fronts[alg_name] = np.zeros((0, n_obj))
                problem_history[alg_name] = []
                continue

            alg_results = raw_results[alg_name]
            alg_times = raw_times[alg_name]
            metrics = []
            n_dropped_total = 0
            n_kept_total = 0
            for r in alg_results:
                f = r.get("fitness", [])
                arr = (np.asarray(f, dtype=float) if f
                       else np.zeros((0, n_obj)))
                if len(arr) > 0:
                    _, _, _, n_dropped = filter_solutions_with_report(
                        arr, ref_front, ref_point=ref_point)
                    n_dropped_total += n_dropped
                    n_kept_total += len(filter_solutions(
                        arr, ref_front, ref_point=ref_point))
                metrics.append(compute_metrics(
                    arr, ref_front, n_obj, ref_point,
                    use_filter=filter_enabled))
            metrics_by_alg[alg_name] = metrics
            print(f"    {alg_name}: kept {n_kept_total} solutions, "
                  f"dropped {n_dropped_total} across {num_runs} runs")

            hv_values = []
            for r in alg_results:
                f = r.get("fitness", [])
                if f:
                    arr = np.asarray(f, dtype=float)
                    if arr.ndim == 1:
                        arr = arr.reshape(1, -1)
                    if filter_enabled and ref_front is not None:
                        arr_f = filter_solutions(arr, ref_front,
                                                 ref_point=ref_point)
                    else:
                        arr_f = filter_solutions(arr, None)
                    if len(arr_f) > 0:
                        try:
                            hv = float(HV(ref_point=ref_point)(arr_f))
                            hv_values.append(hv if np.isfinite(hv)
                                             else 0.0)
                        except Exception:
                            hv_values.append(0.0)
                    else:
                        hv_values.append(0.0)
                else:
                    hv_values.append(0.0)

            best_idx = int(np.argmax(hv_values)) if hv_values else 0
            best_fit = alg_results[best_idx].get("fitness", [])
            problem_fronts[alg_name] = (np.asarray(best_fit, dtype=float)
                                        if best_fit
                                        else np.zeros((0, n_obj)))
            problem_history[alg_name] = alg_results[best_idx].get(
                "history", {}).get("hypervolume", [])

            results[alg_name] = {
                "results": alg_results,
                "metrics": metrics,
                "times": alg_times,
                "summary": {
                    "hypervolume": summary_statistics(
                        [m["hypervolume"] for m in metrics]),
                    "igd": summary_statistics([m["igd"] for m in metrics]),
                    "spacing": summary_statistics(
                        [m["spacing"] for m in metrics]),
                    "cardinality": summary_statistics(
                        [m["cardinality"] for m in metrics]),
                    "time": summary_statistics(alg_times)}}

            hv_sum = results[alg_name]["summary"]["hypervolume"]
            card_sum = results[alg_name]["summary"]["cardinality"]
            hv_str = (f"{hv_sum['mean']:.4f} ± {hv_sum['std']:.4f}"
                      if hv_sum["valid"] else
                      f"invalid (n_inf={hv_sum['n_inf']})")
            card_str = (f"{card_sum['mean']:.1f}"
                        if card_sum["valid"] else "invalid")
            print(f"    {alg_name}: HV: {hv_str}  Card: {card_str}")

            max_hv = float(np.prod(np.maximum(ref_point, 1e-12)))
            if hv_sum["valid"] and hv_sum["mean"] > 1.5 * max_hv:
                print(f"    [WARNING] {alg_name} mean HV "
                      f"({hv_sum['mean']:.4f}) exceeds the physical "
                      f"maximum of the reference box ({max_hv:.4f}).")

        self.results[problem_name] = results
        self.all_fronts[problem_name] = problem_fronts
        self.all_hv_history[problem_name] = problem_history
        self.all_metrics[problem_name] = metrics_by_alg
        return results

    def run_all(self, problems, algorithms):
        for cfg in problems:
            name = cfg["name"]
            kwargs = cfg.get("kwargs", {})
            print(f"\n{'=' * 60}\nRunning on {name}\n{'=' * 60}")
            try:
                self.run_experiment(name, algorithms, **kwargs)
            except Exception as e:
                print(f"  ERROR running {name}: {e}")
                import traceback
                traceback.print_exc()
                continue
        return self.results

    # ------------------------------------------------------------------
    # Ablation
    # ------------------------------------------------------------------
    def run_ablation(self, problems, num_runs=None):
        num_runs = num_runs or self.config.num_runs
        variants = {
            "full": {},
            "no_archive_return": {"use_archive_return": False},
            "no_region_select": {"use_region_select": False},
            "no_lhs_control": {"use_lhs_control": False},
            "no_local_search": {"use_local_search": False},
            "no_reset": {"use_reset": False},
        }
        ablation_results: Dict[str, Dict[str, Dict[str, Any]]] = {}

        for cfg in problems:
            name = cfg["name"]
            kwargs = cfg.get("kwargs", {})
            print(f"\n{'=' * 60}\nABLATION on {name}\n{'=' * 60}")
            try:
                problem = self.get_problem(name, **kwargs)
                self._problem_kwargs[name] = kwargs
            except Exception as e:
                print(f"  ERROR creating {name}: {e}")
                continue

            n_obj = problem.n_obj
            suite = problem.suite
            max_gen = self.config.get_generations_for_suite(suite)

            variant_raw: Dict[str, List[Dict[str, Any]]] = {}
            for variant, switches in variants.items():
                print(f"  Variant: {variant}")
                runs = []
                for run in range(num_runs):
                    seed = self.config.get_seed(run)
                    np.random.seed(seed)
                    random.seed(seed)
                    try:
                        algo = RLEEMO(problem, self.config.rle_emo,
                                      suite=suite,
                                      max_generations=max_gen,
                                      seed=seed, **switches)
                        res = algo.run()
                    except Exception as e:
                        print(f"    [warn] {variant} run {run} failed: {e}")
                        res = {"population": [], "fitness": [],
                               "history": {"hypervolume": []}}
                    runs.append(res)
                variant_raw[variant] = runs

            raw_fronts_by_alg = {}
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
                raw_fronts_by_alg[variant] = fronts

            ref_front, ref_point, filter_enabled = build_reference_fronts(
                problem, raw_fronts_by_alg, REFERENCE_FRONT_POLICY)

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

            ablation_results[name] = {}

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
                ablation_results[name][variant] = {
                    "hv": summary_statistics(hv_list),
                    "igd": summary_statistics(igd_list),
                    "cardinality": summary_statistics(card_list)}
                hv_mean = ablation_results[name][variant]["hv"]["mean"]
                hv_std = ablation_results[name][variant]["hv"]["std"]
                card_mean = (ablation_results[name][variant]["cardinality"]
                             ["mean"])
                if np.isfinite(hv_mean):
                    print(f"    {variant}: HV {hv_mean:.4f} "
                          f"± {hv_std:.4f}  Card {card_mean:.1f}")
                else:
                    print(f"    {variant}: HV invalid  Card {card_mean}")

        return ablation_results

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------
    def statistical_analysis(self, reference_alg="rle_emo"):
        report = {}
        for inst_name, metrics_by_alg in self.all_metrics.items():
            if reference_alg not in metrics_by_alg:
                continue
            if not metrics_by_alg[reference_alg]:
                continue
            entry: Dict[str, Any] = {}
            samples = [[m["hypervolume"] for m in v]
                       for v in metrics_by_alg.values() if v]
            samples = [[v for v in s if np.isfinite(v)] for s in samples]
            samples = [s for s in samples if len(s) > 0]
            try:
                kw_stat, kw_p = stats.kruskal(*samples)
            except Exception:
                kw_stat, kw_p = 0.0, 1.0
            entry["kruskal_wallis"] = {
                "statistic": float(kw_stat),
                "p_value": float(kw_p),
                "significant": bool(kw_p < 0.05)}
            ref_hv = [m["hypervolume"]
                      for m in metrics_by_alg[reference_alg]]
            other_algs = [a for a in metrics_by_alg
                          if a != reference_alg and metrics_by_alg[a]]
            raw_p = []
            keys = []
            for other in other_algs:
                other_hv = [m["hypervolume"]
                            for m in metrics_by_alg[other]]
                try:
                    w_stat, w_p = stats.ranksums(ref_hv, other_hv)
                except Exception:
                    w_stat, w_p = 0.0, 1.0
                d = cohens_d(ref_hv, other_hv)
                all_hv = np.array([ref_hv] + [
                    [m["hypervolume"] for m in metrics_by_alg[a]]
                    for a in other_algs])
                best = np.nanmax(all_hv, axis=0)
                ref_wins = np.sum(np.isclose(ref_hv, best, rtol=1e-9))
                success = (float(ref_wins / len(ref_hv))
                           if len(ref_hv) else 0.0)
                raw_p.append(float(w_p))
                keys.append({"other": other, "w_stat": float(w_stat),
                             "d": float(d), "success": success})
            adjusted = holm_bonferroni(raw_p)
            for idx, key in enumerate(keys):
                entry[f"{reference_alg}_vs_{key['other']}"] = {
                    "wilcoxon_stat": key["w_stat"],
                    "p_value_raw": raw_p[idx],
                    "p_value_holm": adjusted[idx],
                    "significant_holm": bool(adjusted[idx] < 0.05),
                    "cohens_d": key["d"],
                    "effect_size": interpret_cohens_d(key["d"]),
                    "success_rate": key["success"],
                    "mean_ref": (float(np.mean(ref_hv)) if ref_hv
                                 else float("nan"))}
            report[inst_name] = entry
        report["_global"] = self._global_friedman(reference_alg)
        return report

    def _global_friedman(self, reference_alg):
        try:
            import scikit_posthocs as sp
            have_sp = True
        except ImportError:
            have_sp = False
        instances = list(self.all_metrics.keys())
        algs = None
        for inst in instances:
            if inst in self.all_metrics and self.all_metrics[inst]:
                algs = sorted(self.all_metrics[inst].keys())
                break
        if algs is None or len(algs) < 3:
            return {"friedman": None, "nemenyi": None}
        mat = np.full((len(instances), len(algs)), np.nan)
        for i, inst in enumerate(instances):
            for j, alg in enumerate(algs):
                if alg in self.all_metrics[inst]:
                    hvs = [m["hypervolume"]
                           for m in self.all_metrics[inst][alg]]
                    hvs = [v for v in hvs if np.isfinite(v)]
                    if hvs:
                        mat[i, j] = float(np.mean(hvs))
        valid_rows = np.all(np.isfinite(mat), axis=1)
        mat = mat[valid_rows]
        if mat.shape[0] < 3:
            return {"friedman": None, "nemenyi": None,
                    "note": f"fewer than 3 complete instances "
                            f"(found {mat.shape[0]})"}
        try:
            fr_stat, fr_p = stats.friedmanchisquare(*mat.T)
        except Exception:
            fr_stat, fr_p = 0.0, 1.0
        result = {"friedman": {
            "statistic": float(fr_stat), "p_value": float(fr_p),
            "significant": bool(fr_p < 0.05),
            "n_instances": int(mat.shape[0]),
            "n_algorithms": int(mat.shape[1]),
            "algorithms": algs},
            "nemenyi": None}
        if have_sp and fr_p < 0.05:
            try:
                nemenyi = sp.posthoc_nemenyi_friedman(mat)
                result["nemenyi"] = {
                    "p_matrix": nemenyi.values.tolist(),
                    "algorithms": algs}
            except Exception:
                pass
        return result

    # ------------------------------------------------------------------
    # Saving
    # ------------------------------------------------------------------
    def save_results(self, path):
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
        payload = {}
        for prob, pr in self.results.items():
            payload[prob] = {
                "reference_info": conv(self.reference_info.get(prob, {})),
                "algorithms": {}}
            for alg, r in pr.items():
                payload[prob]["algorithms"][alg] = {
                    "summary": conv(r["summary"]),
                    "times": conv(r["times"]),
                    "metrics": conv(r["metrics"]),
                    "not_applicable": r.get("not_applicable", False)}
        with open(path, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"\nResults saved to {path}")

    def save_results_csv(self, path):
        try:
            import pandas as pd
        except ImportError:
            print("[info] pandas not installed, CSV export skipped.")
            return
        rows = []
        for prob, pr in self.results.items():
            for alg, r in pr.items():
                if r.get("not_applicable", False):
                    rows.append({"instance": prob, "algorithm": alg,
                                 "run": -1, "hypervolume": np.nan,
                                 "igd": np.nan, "spacing": np.nan,
                                 "cardinality": 0, "time": np.nan,
                                 "status": "not_applicable"})
                    continue
                for run, m in enumerate(r["metrics"]):
                    rows.append({
                        "instance": prob, "algorithm": alg, "run": run,
                        "hypervolume": m["hypervolume"],
                        "igd": (m["igd"] if np.isfinite(m["igd"])
                                else np.nan),
                        "spacing": m["spacing"],
                        "cardinality": m["cardinality"],
                        "time": (r["times"][run]
                                 if run < len(r["times"]) else np.nan),
                        "status": "ok"})
        df = pd.DataFrame(rows)
        df.to_csv(path, index=False)
        print(f"Results saved to {path}")

    # ------------------------------------------------------------------
    # Report
    # ------------------------------------------------------------------
    def generate_report(self, output_dir, stats_report,
                        ablation_results=None):
        os.makedirs(output_dir, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.save_results(os.path.join(output_dir, f"results_{ts}.json"))
        self.save_results_csv(os.path.join(output_dir,
                                           f"results_{ts}.csv"))
        pareto_dir = os.path.join(output_dir, "pareto_fronts")
        conv_dir = os.path.join(output_dir, "convergence")
        stats_dir = os.path.join(output_dir, "statistics")
        for d in (pareto_dir, conv_dir, stats_dir):
            os.makedirs(d, exist_ok=True)

        for prob, pr in self.results.items():
            print(self._table(pr, prob))
            try:
                problem_obj = self.get_problem(
                    prob, **self._problem_kwargs.get(prob, {}))
                n_obj = problem_obj.n_obj
                fronts = self.all_fronts.get(prob, {})
                history = self.all_hv_history.get(prob, {})
                try:
                    raw_fronts_by_alg = {}
                    for alg_name in pr:
                        if pr[alg_name].get("not_applicable", False):
                            continue
                        fronts_alg = []
                        for r in pr[alg_name]["results"]:
                            f = r.get("fitness", [])
                            if f:
                                arr = np.asarray(f, dtype=float)
                                if arr.ndim == 1:
                                    arr = arr.reshape(1, -1)
                                fronts_alg.append(arr)
                            else:
                                fronts_alg.append(np.zeros((0, n_obj)))
                        raw_fronts_by_alg[alg_name] = fronts_alg
                    ref_front_used, _, _ = build_reference_fronts(
                        problem_obj, raw_fronts_by_alg,
                        REFERENCE_FRONT_POLICY)
                except Exception:
                    ref_front_used = problem_obj.analytic_pareto_front(
                        self.config.num_test_points)
                if n_obj == 2:
                    self._plot_2d(
                        fronts, ref_front_used,
                        title=f"{prob.upper()} \u2014 Pareto front",
                        save=os.path.join(pareto_dir,
                                          f"{prob}_pareto_2d.png"))
                else:
                    self._plot_3d(
                        fronts, ref_front_used,
                        title=f"{prob.upper()} \u2014 3D Pareto front",
                        save=os.path.join(pareto_dir,
                                          f"{prob}_pareto_3d.png"))
                self._plot_conv(
                    history,
                    title=f"{prob.upper()} \u2014 Convergence",
                    save=os.path.join(conv_dir,
                                      f"{prob}_convergence.png"))
                hv_data = {alg: [m["hypervolume"] for m in r["metrics"]]
                           for alg, r in pr.items()
                           if not r.get("not_applicable", False)}
                if hv_data:
                    self._plot_box(
                        hv_data,
                        title=f"{prob.upper()} \u2014 HV distribution",
                        save=os.path.join(stats_dir,
                                          f"{prob}_boxplot.png"))
            except Exception as e:
                print(f"  [warn] plotting {prob}: {e}")

        try:
            heat = {}
            for prob, pr in self.results.items():
                row = {}
                for alg, r in pr.items():
                    if r.get("not_applicable", False):
                        row[alg] = float("nan")
                    else:
                        hv = r["summary"]["hypervolume"]
                        row[alg] = (hv["mean"] if hv["valid"]
                                    else float("nan"))
                heat[prob] = row
            self._plot_heat(heat,
                            save=os.path.join(output_dir, "heatmap.png"))
        except Exception as e:
            print(f"  [warn] heatmap: {e}")

        self._write_summary_table(output_dir)
        self._write_stats_table(output_dir, stats_report)
        self._plot_success_rate(stats_report, stats_dir)
        self._plot_effect_size(stats_report, stats_dir)
        self._plot_cd_diagram(stats_report, stats_dir)

        if ablation_results:
            self._write_ablation_table(output_dir, ablation_results)
            try:
                self._plot_ablation(ablation_results, output_dir)
            except Exception as e:
                print(f"  [warn] ablation plot: {e}")

        print(f"\nReport generated in {output_dir}")

    # ------------------------------------------------------------------
    # Plots
    # ------------------------------------------------------------------
    @staticmethod
    def _plot_2d(fronts, ref_front, title, save):
        try:
            fig, ax = plt.subplots(figsize=(7.0, 5.6))
            names = list(fronts.keys())
            for i, name in enumerate(names):
                f = fronts[name]
                if len(f) > 0:
                    f = np.asarray(f, dtype=float)
                    f = f[np.all(np.isfinite(f)
                                 & (np.abs(f) < DEGENERATE_THRESHOLD),
                                 axis=1)]
                    if len(f) > 0:
                        ax.scatter(f[:, 0], f[:, 1], alpha=0.75,
                                   label=name,
                                   color=PALETTE[i % len(PALETTE)],
                                   s=32, edgecolors="black",
                                   linewidth=0.4, zorder=3)
            if ref_front is not None and len(ref_front) > 0:
                rf = np.asarray(ref_front, dtype=float)
                if rf.ndim == 2 and rf.shape[1] >= 2:
                    order = np.argsort(rf[:, 0])
                    ax.plot(rf[order, 0], rf[order, 1], "k--",
                            label="Reference PF", linewidth=1.6,
                            alpha=0.9, zorder=4)
            ax.set_xlabel("Objective 1")
            ax.set_ylabel("Objective 2")
            ax.set_title(title, fontweight="bold")
            ax.legend(loc="best", ncol=2, frameon=True,
                      columnspacing=0.8, handletextpad=0.4)
            ax.grid(True, alpha=0.3)
            save_figure(fig, save)
        except Exception as e:
            print(f"    [warn] 2D plot failed for {title}: {e}")
            plt.close("all")

    @staticmethod
    def _plot_3d(fronts, ref_front, title, save):
        try:
            fig = plt.figure(figsize=(7.4, 6.2))
            ax = fig.add_subplot(111, projection="3d")
            names = list(fronts.keys())
            for i, name in enumerate(names):
                f = fronts[name]
                if len(f) > 0 and f.ndim == 2 and f.shape[1] == 3:
                    f = np.asarray(f, dtype=float)
                    f = f[np.all(np.isfinite(f)
                                 & (np.abs(f) < DEGENERATE_THRESHOLD),
                                 axis=1)]
                    if len(f) > 0:
                        ax.scatter(f[:, 0], f[:, 1], f[:, 2],
                                   alpha=0.75, label=name,
                                   color=PALETTE[i % len(PALETTE)],
                                   s=28, edgecolors="black",
                                   linewidth=0.3)
            if (ref_front is not None and ref_front.ndim == 2
                    and ref_front.shape[1] == 3):
                n = min(200, len(ref_front))
                if n > 0:
                    idx = np.random.choice(len(ref_front), n,
                                           replace=False)
                    ax.scatter(ref_front[idx, 0], ref_front[idx, 1],
                               ref_front[idx, 2], c="black",
                               marker="o", alpha=0.25, s=12,
                               label="Reference PF")
            ax.set_xlabel("Objective 1")
            ax.set_ylabel("Objective 2")
            ax.set_zlabel("Objective 3")
            ax.set_title(title, fontweight="bold")
            ax.legend(loc="upper right", ncol=2, frameon=True,
                      fontsize=8)
            ax.view_init(elev=22, azim=-58)
            try:
                ax.set_box_aspect((1, 1, 0.8))
            except Exception:
                pass
            save_figure(fig, save)
        except Exception as e:
            print(f"    [warn] 3D plot failed for {title}: {e}")
            plt.close("all")

    @staticmethod
    def _plot_conv(history, title, save):
        try:
            fig, ax = plt.subplots(figsize=(7.4, 4.6))
            names = list(history.keys())
            for i, name in enumerate(names):
                h = history[name]
                if not h:
                    continue
                c = PALETTE[i % len(PALETTE)]
                h_clean = [v if np.isfinite(v) else 0.0 for v in h]
                ax.plot(range(len(h_clean)), h_clean, linestyle=":",
                        linewidth=1.0, alpha=0.45, color=c,
                        label=f"{name} (instantaneous)")
                envelope = np.maximum.accumulate(h_clean)
                ax.plot(range(len(envelope)), envelope, linestyle="-",
                        linewidth=1.8, color=c,
                        label=f"{name} (best-so-far)")
            ax.set_xlabel("Generation")
            ax.set_ylabel("Internal HV tracker")
            ax.set_title(title, fontweight="bold")
            ax.legend(loc="best", ncol=2, frameon=True, fontsize=7.5)
            ax.grid(True, alpha=0.3)
            save_figure(fig, save)
        except Exception as e:
            print(f"    [warn] conv plot failed for {title}: {e}")
            plt.close("all")

    @staticmethod
    def _plot_box(data, title, save):
        try:
            fig, ax = plt.subplots(figsize=(7.4, 4.6))
            names, vals = [], []
            for n, v in data.items():
                v = [x for x in v if np.isfinite(x)]
                if v:
                    names.append(n)
                    vals.append(v)
            if not vals:
                plt.close(fig)
                return
            bp = ax.boxplot(vals, patch_artist=True, showmeans=True,
                            meanprops=dict(marker="D",
                                           markerfacecolor="white",
                                           markeredgecolor="black",
                                           markersize=4),
                            medianprops=dict(color="black",
                                             linewidth=1.0),
                            flierprops=dict(marker="o", markersize=3,
                                            markerfacecolor="none",
                                            markeredgecolor="gray"))
            ax.set_xticks(range(1, len(names) + 1))
            ax.set_xticklabels(names, rotation=30, ha="right")
            for p, c in zip(bp["boxes"],
                            [PALETTE[i % len(PALETTE)]
                             for i in range(len(names))]):
                p.set_facecolor(c)
                p.set_alpha(0.65)
                p.set_edgecolor("black")
            ax.set_ylabel("Hypervolume")
            ax.set_title(title, fontweight="bold")
            ax.grid(True, alpha=0.3, axis="y")
            save_figure(fig, save)
        except Exception as e:
            print(f"    [warn] box plot failed for {title}: {e}")
            plt.close("all")

    @staticmethod
    def _plot_heat(data, save):
        try:
            probs = list(data.keys())
            algs = list(data[probs[0]].keys())
            M = np.full((len(probs), len(algs)), np.nan)
            for i, p in enumerate(probs):
                for j, a in enumerate(algs):
                    v = data[p][a]
                    if np.isfinite(v):
                        M[i, j] = v
            rmax = np.nanmax(M, axis=1, keepdims=True)
            rmax = np.where(np.isfinite(rmax) & (rmax > 0),
                            rmax, 1e-10)
            Mn = np.where(np.isfinite(M), M / rmax, np.nan)
            fig, ax = plt.subplots(figsize=(7.6, 6.8))
            im = ax.imshow(Mn, cmap="RdYlGn", aspect="auto",
                           vmin=0, vmax=1)
            ax.set_xticks(range(len(algs)))
            ax.set_yticks(range(len(probs)))
            ax.set_xticklabels(algs, rotation=35, ha="right")
            ax.set_yticklabels(probs)
            cbar = plt.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
            cbar.set_label("Normalized HV", fontsize=9)
            for i in range(len(probs)):
                for j in range(len(algs)):
                    if np.isfinite(Mn[i, j]):
                        ax.text(j, i, f"{M[i, j]:.2f}", ha="center",
                                va="center", fontsize=7,
                                color=("black" if Mn[i, j] < 0.7
                                       else "white"))
                    else:
                        ax.text(j, i, "n/a", ha="center", va="center",
                                fontsize=7, color="black")
            for spine in ax.spines.values():
                spine.set_edgecolor("black")
                spine.set_linewidth(0.6)
            ax.set_title("Overall HV heatmap "
                         "(per-instance normalized)",
                         fontweight="bold")
            save_figure(fig, save)
        except Exception as e:
            print(f"    [warn] heatmap failed: {e}")
            plt.close("all")

    def _plot_ablation(self, ablation_results, output_dir):
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
            save_figure(fig, os.path.join(output_dir,
                                          "ablation_heatmap.png"))
        except Exception as e:
            print(f"    [warn] ablation heatmap failed: {e}")
            plt.close("all")

    def _plot_cd_diagram(self, stats_report, stats_dir):
        global_rep = stats_report.get("_global", {})
        nemenyi = global_rep.get("nemenyi") if global_rep else None
        if not nemenyi:
            return
        try:
            import scikit_posthocs as sp
            import pandas as pd
            algs = nemenyi["algorithms"]
            pmat = pd.DataFrame(nemenyi["p_matrix"],
                                index=algs, columns=algs)
            fig, ax = plt.subplots(figsize=(7.4, 4.4))
            sp.critical_difference_diagram(
                ranks={a: i for i, a in enumerate(algs)},
                sig_matrix=pmat, ax=ax)
            ax.set_title("Nemenyi critical-difference diagram",
                         fontweight="bold")
            save_figure(fig, os.path.join(stats_dir, "cd_diagram.png"))
        except Exception as e:
            print(f"  [warn] CD diagram: {e}")
            plt.close("all")

    # ------------------------------------------------------------------
    # Tables
    # ------------------------------------------------------------------
    def _table(self, pr, prob_name):
        lines = [f"\n{'=' * 120}", f"Results for {prob_name}",
                 f"{'=' * 120}",
                 f"{'Algorithm':<14} {'HV Mean':<14} {'HV Std':<12} "
                 f"{'IGD Mean':<14} {'Spacing':<12} {'Card':<8} "
                 f"{'Valid':<6} {'Status':<16}",
                 f"{'-' * 120}"]
        for alg, r in sorted(
                pr.items(),
                key=lambda x: (x[1]["summary"]["hypervolume"]["mean"]
                               if x[1]["summary"]["hypervolume"]["valid"]
                               and not x[1].get("not_applicable", False)
                               else -np.inf),
                reverse=True):
            s = r["summary"]
            hv = s["hypervolume"]
            igd = s["igd"]
            sp = s["spacing"]
            card = s["cardinality"]
            na = r.get("not_applicable", False)
            if na:
                lines.append(
                    f"{alg:<14} {'n/a':<14} {'n/a':<12} {'n/a':<14} "
                    f"{'n/a':<12} {'n/a':<8} {'no':<6} "
                    f"{'not applicable':<16}")
                continue
            hv_mean = f"{hv['mean']:.4f}" if hv["valid"] else "invalid"
            hv_std = f"{hv['std']:.4f}" if hv["valid"] else "-"
            igd_mean = f"{igd['mean']:.4f}" if igd["valid"] else "invalid"
            sp_mean = f"{sp['mean']:.4f}" if sp["valid"] else "invalid"
            card_mean = (f"{card['mean']:.1f}" if card["valid"]
                         else "invalid")
            valid = ("yes" if (card["valid"] and card["mean"] > 0)
                     else "no")
            status = "ok" if valid == "yes" else "empty front"
            lines.append(
                f"{alg:<14} {hv_mean:<14} {hv_std:<12} "
                f"{igd_mean:<14} {sp_mean:<12} {card_mean:<8} "
                f"{valid:<6} {status:<16}")
        return "\n".join(lines)

    def _write_summary_table(self, output_dir):
        lines = ["\n" + "=" * 140, "COMPREHENSIVE SUMMARY TABLE",
                 "=" * 140,
                 f"{'Suite':<10} {'Problem':<12} {'Algorithm':<10} "
                 f"{'HV Mean':<14} {'HV Std':<12} {'IGD Mean':<14} "
                 f"{'Card':<8} {'Valid':<6} {'Status':<16}",
                 "-" * 140]
        for prob, pr in sorted(self.results.items()):
            try:
                suite = self.get_problem(
                    prob, **self._problem_kwargs.get(prob, {})).suite
            except Exception:
                suite = "Unknown"
            for alg, r in sorted(
                    pr.items(),
                    key=lambda x: (x[1]["summary"]["hypervolume"]["mean"]
                                   if x[1]["summary"]["hypervolume"]["valid"]
                                   and not x[1].get("not_applicable",
                                                    False)
                                   else -np.inf),
                    reverse=True):
                s = r["summary"]
                hv = s["hypervolume"]
                igd = s["igd"]
                card = s["cardinality"]
                na = r.get("not_applicable", False)
                if na:
                    lines.append(
                        f"{suite:<10} {prob:<12} {alg:<10} "
                        f"{'n/a':<14} {'n/a':<12} {'n/a':<14} "
                        f"{'n/a':<8} {'no':<6} {'not applicable':<16}")
                    continue
                hv_mean = (f"{hv['mean']:.4f}" if hv["valid"]
                           else "invalid")
                hv_std = f"{hv['std']:.4f}" if hv["valid"] else "-"
                igd_mean = (f"{igd['mean']:.4f}" if igd["valid"]
                            else "invalid")
                card_mean = (f"{card['mean']:.1f}" if card["valid"]
                             else "invalid")
                valid = ("yes" if (card["valid"] and card["mean"] > 0)
                         else "no")
                status = "ok" if valid == "yes" else "empty front"
                lines.append(
                    f"{suite:<10} {prob:<12} {alg:<10} "
                    f"{hv_mean:<14} {hv_std:<12} {igd_mean:<14} "
                    f"{card_mean:<8} {valid:<6} {status:<16}")
            lines.append("-" * 140)
        txt = "\n".join(lines)
        print(txt)
        with open(os.path.join(output_dir, "summary_table.txt"), "w") as f:
            f.write(txt)

    def _write_stats_table(self, output_dir, stats_report):
        lines = ["\n" + "=" * 150,
                 "STATISTICAL VALIDATION (RLE-EMO vs each competitor)",
                 "Pairwise Wilcoxon with Holm-Bonferroni correction.",
                 "=" * 150,
                 f"{'Instance':<12} {'Comparison':<26} {'p_raw':<12} "
                 f"{'p_holm':<12} {'Signif.':<8} {'Cohen d':<10} "
                 f"{'Effect':<10} {'Success':<10}",
                 "-" * 150]
        for inst, rep in stats_report.items():
            if inst == "_global":
                continue
            for key, val in rep.items():
                if key == "kruskal_wallis":
                    lines.append(
                        f"{inst:<12} {'Kruskal-Wallis':<26} "
                        f"{val['p_value']:<12.4g} {'-':<12} "
                        f"{str(val['significant']):<8} "
                        f"{'-':<10} {'-':<10} {'-':<10}")
                    continue
                lines.append(
                    f"{inst:<12} {key:<26} "
                    f"{val['p_value_raw']:<12.4g} "
                    f"{val['p_value_holm']:<12.4g} "
                    f"{str(val['significant_holm']):<8} "
                    f"{val['cohens_d']:<10.4f} "
                    f"{val['effect_size']:<10} "
                    f"{val['success_rate']:<10.3f}")
            lines.append("-" * 150)
        global_rep = stats_report.get("_global", {})
        if global_rep and global_rep.get("friedman"):
            fr = global_rep["friedman"]
            lines.append("")
            lines.append("GLOBAL FRIEDMAN TEST")
            lines.append("-" * 150)
            lines.append(
                f"  statistic = {fr['statistic']:.4f}, "
                f"p = {fr['p_value']:.4g}, "
                f"n_instances = {fr['n_instances']}, "
                f"n_algorithms = {fr['n_algorithms']}")
            lines.append(f"  algorithms = {fr['algorithms']}")
            lines.append(f"  significant = {fr['significant']}")
        elif global_rep and global_rep.get("note"):
            lines.append("")
            lines.append("GLOBAL FRIEDMAN TEST")
            lines.append("-" * 150)
            lines.append(f"  skipped: {global_rep['note']}")
        txt = "\n".join(lines)
        print(txt)
        with open(os.path.join(output_dir, "statistics_table.txt"),
                  "w") as f:
            f.write(txt)

    def _write_ablation_table(self, output_dir, ablation_results):
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
        with open(os.path.join(output_dir, "ablation_summary.txt"),
                  "w") as f:
            f.write(txt)

    def _plot_success_rate(self, stats_report, stats_dir):
        comps: Dict[str, List[float]] = {}
        for inst, rep in stats_report.items():
            if inst == "_global":
                continue
            for key, val in rep.items():
                if key == "kruskal_wallis":
                    continue
                comps.setdefault(key, []).append(val["success_rate"])
        if not comps:
            return
        try:
            fig, ax = plt.subplots(figsize=(7.4, 4.4))
            names = list(comps.keys())
            means = [np.mean(comps[k]) for k in names]
            colors = [PALETTE[i % len(PALETTE)]
                      for i in range(len(names))]
            ax.bar(range(len(names)), means, color=colors,
                   edgecolor="black", linewidth=0.5)
            ax.axhline(0.5, color="red", linestyle="--",
                       linewidth=1.0, label="50% baseline")
            ax.set_xticks(range(len(names)))
            ax.set_xticklabels(names, rotation=30, ha="right")
            ax.set_ylabel("Success rate (mean across instances)")
            ax.set_ylim(0, 1.05)
            ax.set_title("RLE-EMO success rate vs. competitors",
                         fontweight="bold")
            ax.legend(loc="best", ncol=1)
            ax.grid(True, alpha=0.3, axis="y")
            save_figure(fig, os.path.join(stats_dir, "success_rate.png"))
        except Exception as e:
            print(f"  [warn] success-rate plot: {e}")
            plt.close("all")

    def _plot_effect_size(self, stats_report, stats_dir):
        comps: Dict[str, List[float]] = {}
        for inst, rep in stats_report.items():
            if inst == "_global":
                continue
            for key, val in rep.items():
                if key == "kruskal_wallis":
                    continue
                comps.setdefault(key, []).append(val["cohens_d"])
        if not comps:
            return
        try:
            fig, ax = plt.subplots(figsize=(7.4, 4.4))
            names = list(comps.keys())
            means = [np.mean(comps[k]) for k in names]
            colors = [PALETTE[i % len(PALETTE)]
                      for i in range(len(names))]
            ax.bar(range(len(names)), means, color=colors,
                   edgecolor="black", linewidth=0.5)
            ax.axhline(0.2, color="gray", linestyle="--",
                       label="small (0.2)", linewidth=1.0)
            ax.axhline(0.5, color="gray", linestyle="-.",
                       label="medium (0.5)", linewidth=1.0)
            ax.axhline(0.8, color="gray", linestyle=":",
                       label="large (0.8)", linewidth=1.0)
            ax.axhline(0.0, color="black", linewidth=0.6)
            ax.set_xticks(range(len(names)))
            ax.set_xticklabels(names, rotation=30, ha="right")
            ax.set_ylabel("Cohen's $d$ (mean across instances)")
            ax.set_title("Effect size: RLE-EMO vs. competitors",
                         fontweight="bold")
            ax.legend(loc="best", ncol=2, fontsize=8)
            ax.grid(True, alpha=0.3, axis="y")
            save_figure(fig, os.path.join(stats_dir, "effect_size.png"))
        except Exception as e:
            print(f"  [warn] effect-size plot: {e}")
            plt.close("all")


# ============================================================================
# MAIN
# ============================================================================
def main():
    global REFERENCE_FRONT_POLICY

    parser = argparse.ArgumentParser(
        description="RLE-EMO benchmark suite and ablation study (v5.3).")
    parser.add_argument("--ablation", action="store_true",
                        help="Run only the ablation study.")
    parser.add_argument("--all", action="store_true",
                        help="Run benchmark + ablation in one call.")
    parser.add_argument("--runs", type=int, default=30)
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT_DIR,
                        help="Output directory. Default: manuscript folder.")
    parser.add_argument("--problems", type=str, default=None)
    parser.add_argument("--algorithms", type=str, default=None)
    parser.add_argument("--reference-front", type=str, default="auto",
                        choices=["auto", "analytic", "best-observed",
                                 "none"])

    args, _unknown = parser.parse_known_args()
    REFERENCE_FRONT_POLICY = args.reference_front

    print("=" * 90)
    print("RLE-EMO: Comprehensive Benchmark Suite (v5.3, LHS-guided)")
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
        wanted = set(s.strip().lower() for s in args.problems.split(","))
        problems = [p for p in all_problems if p["name"].lower() in wanted]
    else:
        problems = all_problems

    algorithms = (["rle_emo", "nsga2", "moead", "rvea"]
                  if not args.algorithms
                  else [s.strip().lower()
                        for s in args.algorithms.split(",")])

    os.makedirs(args.output, exist_ok=True)
    runner = ExperimentRunner(config)

    if args.ablation:
        print("\n" + "=" * 60)
        print("ABLATION MODE")
        print("=" * 60)
        ablation_results = runner.run_ablation(problems,
                                               num_runs=config.num_runs)
        runner._write_ablation_table(args.output, ablation_results)
        try:
            runner._plot_ablation(ablation_results, args.output)
        except Exception as e:
            print(f"  [warn] ablation plot: {e}")
        print("\nABLATION COMPLETED SUCCESSFULLY")
        return

    print(f"\nProblems: {len(problems)} | Algorithms: {len(algorithms)} "
          f"| Runs each: {config.num_runs}")
    runner.run_all(problems, algorithms)

    print("\n" + "=" * 60)
    print("Statistical analysis (RLE-EMO vs competitors)")
    print("=" * 60)
    stats_report = runner.statistical_analysis(reference_alg="rle_emo")

    ablation_results = None
    if args.all:
        print("\n" + "=" * 60)
        print("ABLATION (integrated, same script)")
        print("=" * 60)
        ablation_results = runner.run_ablation(problems,
                                               num_runs=config.num_runs)

    runner.generate_report(args.output, stats_report, ablation_results)

    print("\n" + "=" * 90)
    print("EXPERIMENTS COMPLETED SUCCESSFULLY")
    print("=" * 90)
    print(f"\nOutput directory: {args.output}")
    print("  pareto_fronts/       2D/3D Pareto front plots (PNG + PDF)")
    print("  convergence/         convergence curves")
    print("  statistics/          boxplots, success rate, effect size")
    print("  heatmap.png/pdf      overall HV heatmap")
    print("  summary_table.txt    comprehensive results table")
    print("  statistics_table.txt Wilcoxon/Holm, Cohen's d, Friedman")
    if ablation_results:
        print("  ablation_summary.txt ablation table")
        print("  ablation_heatmap.png/pdf")
    print("  results_*.json       raw results (JSON)")
    print("  results_*.csv        raw results (flat CSV)")


if __name__ == "__main__":
    main()
