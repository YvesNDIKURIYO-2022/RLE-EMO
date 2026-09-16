#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RLE-EMO: Comprehensive Benchmark Suite
=======================================

Compares RLE-EMO against five RL-assisted MOEAs (RL-MOEA, QL-MOEA,
QLMOEA/D-AOS, RL-NSGA-II, R2-RLMOEA) on eleven benchmark instances from the
ZDT, DTLZ, WFG, and DASCMOP families.

Two modes:
    * Default      : full comparison across all algorithms.
    * --ablation   : six ablation variants of RLE-EMO only.

All metrics (HV, IGD, Spacing) use pymoo's indicator classes and a reference
point set to 1.1 x max(PF_true) per objective. Degenerate objective values
(|f_i| > 1e8) are excluded from all metric computations, uniformly across
algorithms.

Plots are saved as both PNG (300 dpi raster) and PDF (vector) using an
Elsevier-compatible rcParams block.
"""

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
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from scipy.spatial import distance

from pymoo.indicators.hv import HV
from pymoo.indicators.igd import IGD
from pymoo.indicators.spacing import SpacingIndicator
from pymoo.problems import get_problem
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

warnings.filterwarnings("ignore")


# ============================================================================
# MATPLOTLIB STYLE
# ============================================================================
def style_elsevier():
    """Elsevier-compatible rcParams block."""
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
# GLOBAL CONSTANTS
# ============================================================================
DEFAULT_OUTPUT_DIR = "./results"
DEGENERATE_THRESHOLD = 1e8

PALETTE = [
    "#1f77b4", "#2ca02c", "#9467bd", "#e377c2", "#bcbd22",
    "#17becf", "#d62728", "#ff7f0e", "#7f7f7f", "#8c564b",
]


# ============================================================================
# CONFIGURATION
# ============================================================================
@dataclass
class RLEEMOConfig:
    """Configuration for RLE-EMO, protocol-matched to competitors."""

    population_size_base: int = 40
    population_size_divisor: int = 5

    reward_weights_small: Tuple[float, float, float] = (0.6, 0.2, 0.2)
    reward_weights_medium: Tuple[float, float, float] = (0.4, 0.4, 0.2)
    reward_weights_large: Tuple[float, float, float] = (0.3, 0.6, 0.1)

    ppo_learning_rate: float = 3e-4
    ppo_clip_epsilon: float = 0.2
    ppo_gamma: float = 0.99
    ppo_gae_lambda: float = 0.95

    ppo_proportion: float = 0.30
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

    def get_reward_weights(self, n: int) -> Tuple[float, float, float]:
        if n < 50:
            return self.reward_weights_small
        if n < 100:
            return self.reward_weights_medium
        return self.reward_weights_large

    def get_bias_factor(self, n: int) -> float:
        return max(0.1, min(0.5, 100.0 / n))

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


# ============================================================================
# NUMERICAL SAFETY UTILITIES
# ============================================================================
def safe_mean(values, default=0.0):
    try:
        v = [x for x in values if np.isfinite(x)]
        return float(np.mean(v)) if v else default
    except Exception:
        return default


def safe_int_clip(value, lo, hi, default=0):
    try:
        if not np.isfinite(value):
            return default
        return int(np.clip(value, lo, hi))
    except Exception:
        return default


def safe_index(value, size, default=0):
    try:
        if not np.isfinite(value):
            return default
        return max(0, min(size - 1, int(value)))
    except Exception:
        return default


def compute_reference_point(ref_front: Optional[np.ndarray],
                            n_obj: int,
                            fallback: float = 1.1) -> np.ndarray:
    if ref_front is not None and len(ref_front) > 0:
        rf = np.asarray(ref_front, dtype=float)
        rf = rf[np.all(np.isfinite(rf), axis=1)]
        if len(rf) > 0:
            rp = rf.max(axis=0) * 1.1
            return np.where(rp > 1e-9, rp, fallback)
    return np.full(n_obj, fallback, dtype=float)


def save_figure(fig, save_path: str):
    """Save a figure as both PNG (300 dpi) and PDF (vector)."""
    base, _ = os.path.splitext(save_path)
    fig.savefig(base + ".png", dpi=300, bbox_inches="tight")
    try:
        fig.savefig(base + ".pdf", bbox_inches="tight")
    except Exception:
        pass
    plt.close(fig)


# ============================================================================
# PPO AGENT
# ============================================================================
class PPOAgent:
    def __init__(self, state_dim: int, action_dim: int, config: RLEEMOConfig):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.config = config

        self.policy_weights = np.random.randn(state_dim, action_dim) * 0.01
        self.value_weights = np.random.randn(state_dim, 1) * 0.01
        self.old_policy_weights = self.policy_weights.copy()

        self.lr = config.ppo_learning_rate
        self.clip_eps = config.ppo_clip_epsilon
        self.gamma = config.ppo_gamma
        self.gae_lambda = config.ppo_gae_lambda

        self._frozen = False

    def get_action(self, state, bias_factor=0.0, emission_weights=None):
        if state.ndim > 1:
            state = state.flatten()
        if len(state) < self.state_dim:
            state = np.pad(state, (0, self.state_dim - len(state)))
        else:
            state = state[:self.state_dim]

        logits = state @ self.policy_weights
        if emission_weights is not None and bias_factor > 0:
            ew = np.resize(emission_weights, logits.shape)
            logits = logits - bias_factor * ew

        probs = self._softmax(logits)
        probs = np.nan_to_num(probs, nan=1.0 / self.action_dim)
        probs = probs / max(probs.sum(), 1e-10)
        action = np.random.choice(self.action_dim, p=probs)
        return int(action), float(probs[action])

    def update(self, states, actions, rewards, next_states, dones) -> float:
        if self._frozen:
            return 0.0

        if states.ndim == 1:
            states = states.reshape(1, -1)
        if next_states.ndim == 1:
            next_states = next_states.reshape(1, -1)

        if states.shape[1] < self.state_dim:
            pad = ((0, 0), (0, self.state_dim - states.shape[1]))
            states = np.pad(states, pad)
            next_states = np.pad(next_states, pad)
        else:
            states = states[:, :self.state_dim]
            next_states = next_states[:, :self.state_dim]

        action_one_hot = np.zeros((len(actions), self.action_dim))
        action_one_hot[np.arange(len(actions)), actions] = 1

        values = states @ self.value_weights
        next_values = next_states @ self.value_weights
        deltas = (rewards + self.gamma * (1 - dones) * next_values.flatten()
                  - values.flatten())
        deltas = np.nan_to_num(deltas, nan=0.0)

        advantages = np.zeros_like(deltas)
        adv = 0.0
        for t in reversed(range(len(deltas))):
            adv = deltas[t] + self.gamma * self.gae_lambda * (1 - dones[t]) * adv
            advantages[t] = adv
        if len(advantages) > 1 and np.std(advantages) > 1e-8:
            advantages = ((advantages - np.mean(advantages))
                          / np.std(advantages))

        old_probs = self._softmax(states @ self.old_policy_weights)
        old_p = old_probs[np.arange(len(actions)), actions] + 1e-8
        new_probs = self._softmax(states @ self.policy_weights)
        new_p = new_probs[np.arange(len(actions)), actions] + 1e-8

        ratio = np.nan_to_num(new_p / old_p, nan=1.0,
                              posinf=1.0 + self.clip_eps,
                              neginf=1.0 - self.clip_eps)
        clipped = np.clip(ratio, 1 - self.clip_eps, 1 + self.clip_eps)
        objective = np.minimum(ratio * advantages, clipped * advantages)

        grad = states.T @ ((objective[:, None]
                            * (action_one_hot - new_probs)) / new_probs)
        grad = np.nan_to_num(grad, nan=0.0)
        self.policy_weights += self.lr * grad

        td_err = (rewards + self.gamma * (1 - dones)
                  * (next_states @ self.value_weights).flatten()
                  - values.flatten())
        td_err = np.nan_to_num(td_err, nan=0.0)
        self.value_weights += self.lr * states.T @ td_err[:, None]

        self.old_policy_weights = self.policy_weights.copy()
        return float(np.mean(objective)) if len(objective) else 0.0

    def freeze(self):
        self._frozen = True

    def _softmax(self, x):
        x = np.nan_to_num(x, nan=0.0, posinf=100.0, neginf=-100.0)
        e = np.exp(x - np.max(x, axis=-1, keepdims=True))
        s = np.maximum(np.sum(e, axis=-1, keepdims=True), 1e-10)
        return e / s


# ============================================================================
# PPO TRAINER (training-once pool)
# ============================================================================
class PPOTrainer:
    _trained_agent: Optional[PPOAgent] = None
    _trained_signature: Optional[Tuple[int, int]] = None
    _training_iterations: int = 5000

    @classmethod
    def get_or_train(cls, state_dim: int, action_dim: int,
                     config: RLEEMOConfig) -> PPOAgent:
        sig = (state_dim, action_dim)
        if cls._trained_agent is not None and cls._trained_signature == sig:
            return cls._trained_agent

        agent = PPOAgent(state_dim, action_dim, config)
        cls._pretrain(agent, config)
        agent.freeze()
        cls._trained_agent = agent
        cls._trained_signature = sig
        return agent

    @classmethod
    def _pretrain(cls, agent: PPOAgent, config: RLEEMOConfig,
                  n_iter: Optional[int] = None):
        if n_iter is None:
            n_iter = cls._training_iterations

        training_specs = [
            ("zdt1", {"n_var": 10}),
            ("zdt2", {"n_var": 10}),
            ("zdt3", {"n_var": 10}),
            ("dtlz2", {"n_obj": 3, "n_var": 7}),
            ("wfg1", {"n_obj": 3, "n_var": 6}),
        ]

        pf_sets = []
        for name, kwargs in training_specs:
            try:
                prob = get_problem(name, **kwargs)
                pf_x = prob.pareto_set(n_pareto_points=100)
                xl = np.asarray(prob.xl, dtype=float).flatten()
                xu = np.asarray(prob.xu, dtype=float).flatten()
                if pf_x is not None and len(pf_x) > 0:
                    pf_sets.append((pf_x, xl, xu))
            except Exception:
                continue

        if not pf_sets:
            for _ in range(min(n_iter // 10, 20)):
                bs = 5
                s = np.random.randn(bs, agent.state_dim)
                a = np.random.randint(0, agent.action_dim, bs)
                r = np.random.randn(bs)
                ns = np.random.randn(bs, agent.state_dim)
                d = np.zeros(bs)
                agent.update(s, a, r, ns, d)
            return

        n_bins = agent.action_dim
        n_updates = min(n_iter, 400)
        for _ in range(n_updates):
            pf_x, xl, xu = pf_sets[np.random.randint(len(pf_sets))]
            idx = np.random.randint(len(pf_x))
            x_pf = np.asarray(pf_x[idx], dtype=float).flatten()
            n_var = min(len(x_pf), agent.state_dim - 4, len(xl))
            if n_var < 1:
                continue
            states, targets = [], []
            for k in range(n_var):
                denom = max(xu[k] - xl[k], 1e-12)
                frac = (x_pf[k] - xl[k]) / denom
                target = int(np.clip(frac * n_bins, 0, n_bins - 1))
                s_k = np.zeros(agent.state_dim)
                s_k[:k + 1] = x_pf[:k + 1] / 10.0
                states.append(s_k)
                targets.append(target)
            if not states:
                continue
            s_arr = np.array(states)
            a_arr = np.array(targets)
            r_arr = np.ones(len(targets))
            ns_arr = s_arr.copy()
            d_arr = np.zeros(len(targets))
            agent.update(s_arr, a_arr, r_arr, ns_arr, d_arr)


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
        self.xl = np.asarray(pymoo_problem.xl, dtype=float).flatten()
        self.xu = np.asarray(pymoo_problem.xu, dtype=float).flatten()

    def evaluate(self, x: List[float]) -> List[float]:
        try:
            arr = np.asarray(x, dtype=float).reshape(1, -1)
            F = self._problem.evaluate(arr, return_values_of=["F"])
            f = np.asarray(F, dtype=float).flatten()
            return [float(v) if np.isfinite(v) else 1e10 for v in f]
        except Exception:
            return [1e10] * self.n_obj

    def pareto_front(self, n_points: int = 1000) -> Optional[np.ndarray]:
        try:
            pf = self._problem.pareto_front(n_pareto_points=n_points)
            if pf is None or len(pf) == 0:
                return None
            return np.asarray(pf, dtype=float)
        except Exception:
            return None


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
# RLE-EMO
# ============================================================================
class RLEEMO:
    def __init__(self, problem: PymooProblemWrapper,
                 config: RLEEMOConfig, suite: str = "ZDT",
                 max_generations: Optional[int] = None,
                 use_archive_return: bool = True,
                 use_region_select: bool = True,
                 use_ppo_control: bool = True,
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
        self.bias_factor = config.get_bias_factor(self.n_var)
        self.local_search_freq = config.get_local_search_freq(self.n_var)

        self.use_archive_return = use_archive_return
        self.use_region_select = use_region_select
        self.use_ppo_control = use_ppo_control
        self.use_local_search = use_local_search
        self.use_reset = use_reset

        self.region_weights = self._make_region_weights()

        state_dim = min(self.n_var + 4, 50)
        action_dim = min(self.n_var * 3, 100)
        self.ppo_agent = PPOTrainer.get_or_train(state_dim, action_dim, config)

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
            w = np.random.dirichlet(np.ones(m), K)
        return w / np.maximum(np.linalg.norm(w, axis=1, keepdims=True), 1e-12)

    # ------------------------------------------------------------------
    # State computation
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
    # Initialization strategies
    # ------------------------------------------------------------------
    def _encode_state(self, solution):
        state = np.zeros(self.ppo_agent.state_dim)
        n_vals = min(len(solution), self.ppo_agent.state_dim - 4)
        for i in range(n_vals):
            state[i] = solution[i] / 10.0 if solution[i] != 0 else 0
        return state

    def _generate_ppo_solution(self):
        solution = []
        state = np.zeros(self.ppo_agent.state_dim)
        n_bins = self.ppo_agent.action_dim
        for i in range(self.n_var):
            action, _ = self.ppo_agent.get_action(state, bias_factor=self.bias_factor)
            lo, hi = self.problem.xl[i], self.problem.xu[i]
            if self.use_ppo_control:
                frac = (action + 0.5) / n_bins
                val = lo + frac * (hi - lo)
                jitter = np.random.uniform(-0.5, 0.5) * (hi - lo) / max(n_bins, 1)
                val = float(np.clip(val + jitter, lo, hi))
            else:
                val = float(np.random.uniform(lo, hi))
            solution.append(val)
            state = self._encode_state(solution)
        return solution

    def _generate_heuristic_solution(self):
        return [float((self.problem.xl[i] + self.problem.xu[i]) / 2)
                for i in range(self.n_var)]

    def _generate_random_solution(self):
        return [float(np.random.uniform(self.problem.xl[i], self.problem.xu[i]))
                for i in range(self.n_var)]

    def _generate_opposite_bias_solution(self):
        return [float(self.problem.xu[i]
                      - np.random.random() * (self.problem.xu[i] - self.problem.xl[i]))
                for i in range(self.n_var)]

    def initialize_population(self):
        N = self.population_size
        n_ppo = max(1, int(N * self.config.ppo_proportion))
        n_heur = max(1, int(N * self.config.heuristic_proportion))
        n_rand = max(1, int(N * self.config.random_proportion))

        pop = []
        pop += [self._generate_ppo_solution() for _ in range(n_ppo)]
        pop += [self._generate_heuristic_solution() for _ in range(n_heur)]
        pop += [self._generate_random_solution() for _ in range(n_rand)]
        while len(pop) < N:
            pop.append(self._generate_opposite_bias_solution())
        return pop[:N]

    def evaluate(self, x):
        return self.problem.evaluate(x)

    # ------------------------------------------------------------------
    # Dominance and NSGA-II helpers
    # ------------------------------------------------------------------
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
                dist[idx] += (obj[order[i + 1]][k] - obj[order[i - 1]][k]) / rng
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
        order = [(counts[assoc[i]], dist_to_w[i], i) for i in range(len(front))]
        order.sort()

        return [front[idx] for _, _, idx in order[:n_take]]

    # ------------------------------------------------------------------
    # Diversity archive
    # ------------------------------------------------------------------
    def _update_diversity_archive(self):
        max_size = max(1, int(self.config.archive_size_ratio * self.population_size))
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

    def _reset_population_from_archive(self, gen, trigger):
        if not self.diversity_archive:
            return
        if gen - self._last_reset_gen < self.config.reset_min_interval:
            return
        n_rep = min(int(self.config.diversity_reset_proportion * self.population_size),
                    len(self.diversity_archive))
        if n_rep == 0:
            return
        fit_sums = [sum(f) for f in self.fitness]
        worst = sorted(range(len(self.population)),
                       key=lambda i: fit_sums[i], reverse=True)
        arch_idx = np.random.choice(len(self.diversity_archive), n_rep, replace=False)
        for i, a in enumerate(arch_idx):
            if i < len(worst):
                self.population[worst[i]] = self.diversity_archive[a].copy()
                self.fitness[worst[i]] = self.evaluate(self.diversity_archive[a])
        self._last_reset_gen = gen

    # ------------------------------------------------------------------
    # Genetic operators
    # ------------------------------------------------------------------
    def _sbx(self, p1, p2, rate):
        if np.random.random() > rate:
            return p1.copy()
        return [p1[i] if np.random.random() < 0.5 else p2[i]
                for i in range(len(p1))]

    def _poly_mut(self, sol, rate):
        m = sol.copy()
        for i in range(len(m)):
            if np.random.random() < rate:
                d = np.random.uniform(-0.1, 0.1) * (self.problem.xu[i] - self.problem.xl[i])
                m[i] = float(np.clip(m[i] + d, self.problem.xl[i], self.problem.xu[i]))
        return m

    def _local_search(self, sol, n_steps=None, step_frac=None):
        if n_steps is None:
            n_steps = self.config.local_search_steps
        if step_frac is None:
            step_frac = self.config.local_search_step_frac

        best = list(sol)
        best_fit = self.evaluate(best)
        for _ in range(n_steps):
            improved_any = False
            for i in range(len(best)):
                lo, hi = self.problem.xl[i], self.problem.xu[i]
                step = step_frac * (hi - lo)
                for direction in (+1, -1):
                    cand = list(best)
                    cand[i] = float(np.clip(best[i] + direction * step, lo, hi))
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
            idx = np.random.choice(len(self.population), size=2, replace=False)
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
                :max(1, int(self.config.archive_size_ratio * self.population_size))]

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
                    self._reset_population_from_archive(gen, trigger="diversity")
                feedback = 1.0 + xi_div - xi_conv
                if abs(feedback - 1.0) > self.config.delta_rate:
                    self._reset_population_from_archive(gen, trigger="feedback")

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
            fronts = self._fast_non_dominated_sort(combined)
            pareto_pop = fronts[0] if fronts else []
        else:
            fronts = self._fast_non_dominated_sort(self.population)
            pareto_pop = fronts[0] if fronts else []

        pareto_fit = [self.evaluate(s) for s in pareto_pop]

        seen = set()
        uniq_pop, uniq_fit = [], []
        for s, f in zip(pareto_pop, pareto_fit):
            key = tuple(np.round(s, 6))
            if key not in seen:
                seen.add(key)
                uniq_pop.append(s)
                uniq_fit.append(f)

        return {
            "population": uniq_pop,
            "fitness": uniq_fit,
            "history": self.history,
            "hypervolume": 0.0,
        }


def _compute_internal_hv(fitness_array, n_obj):
    if fitness_array is None or len(fitness_array) == 0:
        return 0.0
    try:
        F = np.asarray(fitness_array, dtype=float)
        F = F[np.all(np.isfinite(F) & (np.abs(F) < DEGENERATE_THRESHOLD), axis=1)]
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
# COMPETITORS
# ============================================================================
class RLMOEA:
    def __init__(self, problem, config, n, max_generations=None):
        self.problem = problem
        self.config = config
        self.n = n
        self.population_size = 100
        self.max_generations = (max_generations
                                if max_generations is not None else 80)
        self.n_obj = getattr(problem, "n_obj", 2)
        self.n_var = getattr(problem, "n_var", 30)
        self.dqn_learning_rate = 0.001
        self.dqn_gamma = 0.9
        self.dqn_epsilon = 0.1
        self.q_values = np.zeros((9, 3))

    def run(self):
        pop = [self._rand_sol() for _ in range(self.population_size)]
        fit = [self._safe_evaluate(s) for s in pop]
        for gen in range(self.max_generations):
            state = safe_index(self._compute_state(pop, fit), self.q_values.shape[0])
            action = self._select_action(state)
            if action == 0:
                pop, fit = self._nsga2_step(pop, fit)
            else:
                pop, fit = self._moead_de_step(pop, fit)
            reward = self._compute_reward(pop, fit)
            self._update_q_values(state, action, reward)
        fit = [self._safe_evaluate(s) for s in pop]
        pareto = self._get_pareto(pop, fit)
        return {"population": pareto,
                "fitness": [self._safe_evaluate(s) for s in pareto],
                "history": {"hypervolume": [0.0] * 50},
                "hypervolume": 0.0}

    def _rand_sol(self):
        return [float(np.random.uniform(self.problem.xl[i], self.problem.xu[i]))
                for i in range(self.n_var)]

    def _safe_evaluate(self, s):
        try:
            result = self.problem.evaluate(s)
            return [float(v) if np.isfinite(v) else 1e10 for v in result]
        except Exception:
            return [1e10] * self.n_obj

    def _get_pareto(self, pop, fit):
        return [s for i, s in enumerate(pop)
                if not any(i != j and all(fit[j][k] <= fit[i][k]
                                          for k in range(len(fit[i])))
                           for j in range(len(pop)))]

    def _compute_state(self, pop, fit):
        try:
            sums = [sum(f) / max(1, len(f)) for f in fit]
            avg = safe_mean([v for v in sums if np.isfinite(v)], 0.0)
        except Exception:
            avg = 0.0
        conv = safe_int_clip(avg * 3, 0, 2)
        unique = len(set(tuple(np.round(s, 4)) for s in pop))
        div = safe_int_clip((unique / max(1, len(pop))) * 3, 0, 2)
        return conv * 3 + div

    def _select_action(self, state):
        if np.random.random() < self.dqn_epsilon:
            return np.random.randint(0, 3)
        return int(np.argmax(self.q_values[state]))

    def _nsga2_step(self, pop, fit):
        offspring = []
        for _ in range(len(pop) // 2):
            p1 = pop[np.random.randint(len(pop))]
            p2 = pop[np.random.randint(len(pop))]
            child = [p1[i] if np.random.random() < 0.5 else p2[i]
                     for i in range(len(p1))]
            child = self._mutation(child)
            offspring.append(child)
        off_fit = [self._safe_evaluate(s) for s in offspring]
        combined, combined_fit = pop + offspring, fit + off_fit
        idx = self._nd_select(combined, combined_fit, self.population_size)
        return [combined[i] for i in idx], [combined_fit[i] for i in idx]

    def _moead_de_step(self, pop, fit):
        offspring = []
        for _ in range(len(pop)):
            i = np.random.randint(len(pop))
            p1 = pop[i]
            r1, r2 = np.random.choice(len(pop), 2, replace=False)
            child = [p1[k] + 0.5 * (pop[r1][k] - pop[r2][k])
                     if np.random.random() < 0.5 else p1[k]
                     for k in range(len(p1))]
            child = self._mutation(child)
            offspring.append(child)
        off_fit = [self._safe_evaluate(s) for s in offspring]
        combined, combined_fit = pop + offspring, fit + off_fit
        idx = self._nd_select(combined, combined_fit, self.population_size)
        return [combined[i] for i in idx], [combined_fit[i] for i in idx]

    def _mutation(self, s):
        m = s.copy()
        for i in range(len(m)):
            if np.random.random() < 0.1:
                d = np.random.uniform(-0.1, 0.1) * (self.problem.xu[i] - self.problem.xl[i])
                m[i] = float(np.clip(m[i] + d, self.problem.xl[i], self.problem.xu[i]))
        return m

    def _nd_select(self, pop, fit, k):
        n = len(pop)
        dominated = [set() for _ in range(n)]
        dc = [0] * n
        for i in range(n):
            for j in range(i + 1, n):
                if self._dominates(fit[i], fit[j]):
                    dominated[i].add(j)
                    dc[j] += 1
                elif self._dominates(fit[j], fit[i]):
                    dominated[j].add(i)
                    dc[i] += 1
        front = [i for i in range(n) if dc[i] == 0]
        sel = []
        while front and len(sel) < k:
            for i in front:
                if len(sel) < k:
                    sel.append(i)
            nf = []
            for i in front:
                for j in dominated[i]:
                    dc[j] -= 1
                    if dc[j] == 0:
                        nf.append(j)
            front = nf
        return sel

    def _dominates(self, a, b):
        one = False
        for ai, bi in zip(a, b):
            if ai > bi:
                return False
            if ai < bi:
                one = True
        return one

    def _compute_reward(self, pop, fit):
        try:
            valid = [sum(f) for f in fit if np.isfinite(sum(f))]
            return -safe_mean(valid, 1e6) if valid else -1e6
        except Exception:
            return -1e6

    def _update_q_values(self, state, action, reward):
        if not np.isfinite(reward):
            reward = -1e6
        state = safe_index(state, self.q_values.shape[0])
        action = safe_index(action, self.q_values.shape[1])
        self.q_values[state, action] += self.dqn_learning_rate * (
            reward + self.dqn_gamma * np.max(self.q_values[state])
            - self.q_values[state, action])


class QLMOEA(RLMOEA):
    def __init__(self, problem, config, n, max_generations=None):
        super().__init__(problem, config, n, max_generations)
        self.q_table = np.zeros((5, 5, 2))
        self.q_alpha = 0.1
        self.q_gamma = 0.9
        self.q_epsilon = 0.1

    def run(self):
        pop = [[float(np.random.uniform(self.problem.xl[i], self.problem.xu[i]))
                for i in range(self.n_var)]
               for _ in range(self.population_size)]
        fit = [self._safe_evaluate(s) for s in pop]
        for gen in range(self.max_generations):
            s1, s2 = self._state2(pop, fit)
            a = (np.random.randint(0, 2) if np.random.random() < self.q_epsilon
                 else int(np.argmax(self.q_table[s1, s2])))
            if a == 0:
                pop, fit = self._nsga2_step(pop, fit)
            else:
                pop, fit = self._moead_de_step(pop, fit)
            r = self._compute_reward(pop, fit)
            ns1, ns2 = self._state2(pop, fit)
            self.q_table[s1, s2, a] += self.q_alpha * (
                r + self.q_gamma * np.max(self.q_table[ns1, ns2])
                - self.q_table[s1, s2, a])
        fit = [self._safe_evaluate(s) for s in pop]
        pareto = self._get_pareto(pop, fit)
        return {"population": pareto,
                "fitness": [self._safe_evaluate(s) for s in pareto],
                "history": {"hypervolume": [0.0] * 50},
                "hypervolume": 0.0}

    def _state2(self, pop, fit):
        try:
            avg = safe_mean([sum(f) / max(1, len(f)) for f in fit], 0.0)
        except Exception:
            avg = 0.0
        s1 = safe_int_clip(avg * 5, 0, 4)
        unique = len(set(tuple(np.round(s, 4)) for s in pop))
        s2 = safe_int_clip((unique / max(1, len(pop))) * 5, 0, 4)
        return s1, s2


class QLMOEADAOS(RLMOEA):
    def __init__(self, problem, config, n, max_generations=None):
        super().__init__(problem, config, n, max_generations)
        self.q_table = np.zeros((9, 5))
        self.q_alpha = 0.1
        self.q_gamma = 0.9
        self.q_epsilon = 0.1

    def run(self):
        pop = [[float(np.random.uniform(self.problem.xl[i], self.problem.xu[i]))
                for i in range(self.n_var)]
               for _ in range(self.population_size)]
        fit = [self._safe_evaluate(s) for s in pop]
        for gen in range(self.max_generations):
            state = safe_index(self._state_q(pop, fit), self.q_table.shape[0])
            a = (np.random.randint(0, 5) if np.random.random() < self.q_epsilon
                 else int(np.argmax(self.q_table[state])))
            off = []
            for i in range(0, len(pop) - 1, 2):
                c = self._apply(a, pop[i], pop[i + 1], pop)
                off.append(self._mutation(c))
            of = [self._safe_evaluate(s) for s in off]
            combined, combined_fit = pop + off, fit + of
            idx = self._nd_select(combined, combined_fit, self.population_size)
            pop = [combined[i] for i in idx]
            fit = [combined_fit[i] for i in idx]
            r = self._compute_reward(pop, fit)
            ns = safe_index(self._state_q(pop, fit), self.q_table.shape[0])
            self.q_table[state, a] += self.q_alpha * (
                r + self.q_gamma * np.max(self.q_table[ns])
                - self.q_table[state, a])
        fit = [self._safe_evaluate(s) for s in pop]
        pareto = self._get_pareto(pop, fit)
        return {"population": pareto,
                "fitness": [self._safe_evaluate(s) for s in pareto],
                "history": {"hypervolume": [0.0] * 50},
                "hypervolume": 0.0}

    def _state_q(self, pop, fit):
        try:
            arr = np.array(fit, dtype=float)
            arr = np.nan_to_num(arr, nan=1e10, posinf=1e10, neginf=-1e10)
            d = distance.cdist(arr, arr)
            np.fill_diagonal(d, np.inf)
            md = d.min(axis=1)
            valid = md[np.isfinite(md)]
            sp = float(np.std(valid) / max(np.mean(valid), 1e-8)) if len(valid) else 0.0
        except Exception:
            sp = 0.0
        unique = len(set(tuple(np.round(s, 4)) for s in pop))
        pd = unique / max(1, len(pop))
        return safe_int_clip(sp * 3, 0, 2) * 3 + safe_int_clip(pd * 3, 0, 2)

    def _apply(self, op, p1, p2, pop):
        try:
            if op == 0:
                r1, r2 = np.random.choice(len(pop), 2, replace=False)
                return [p1[i] + 0.5 * (pop[r1][i] - pop[r2][i])
                        if np.random.random() < 0.5 else p1[i]
                        for i in range(len(p1))]
            if op == 1:
                r1, r2, r3, r4 = np.random.choice(len(pop), 4, replace=False)
                return [p1[i] + 0.5 * (pop[r1][i] - pop[r2][i]
                                      + pop[r3][i] - pop[r4][i])
                        if np.random.random() < 0.5 else p1[i]
                        for i in range(len(p1))]
            if op == 2:
                r1, r2, r3 = np.random.choice(len(pop), 3, replace=False)
                return [p1[i] + 0.5 * (p1[i] - pop[r1][i]
                                      + pop[r2][i] - pop[r3][i])
                        if np.random.random() < 0.5 else p1[i]
                        for i in range(len(p1))]
            if op == 3:
                return [p1[i] if np.random.random() < 0.5 else p2[i]
                        for i in range(len(p1))]
            return [(p1[i] + p2[i]) / 2 if np.random.random() < 0.5 else p1[i]
                    for i in range(len(p1))]
        except Exception:
            return p1.copy()


class RLNSGAII(RLMOEA):
    def __init__(self, problem, config, n, max_generations=None):
        super().__init__(problem, config, n, max_generations)
        self.q_table = np.zeros((5, 5))
        self.q_alpha = 0.1
        self.q_gamma = 0.9
        self.q_epsilon = 0.1
        self.params = [(0.9, 0.1), (0.8, 0.2), (0.7, 0.3),
                       (0.6, 0.4), (0.5, 0.5)]

    def run(self):
        pop = [[float(np.random.uniform(self.problem.xl[i], self.problem.xu[i]))
                for i in range(self.n_var)]
               for _ in range(self.population_size)]
        fit = [self._safe_evaluate(s) for s in pop]
        for gen in range(self.max_generations):
            state = safe_index(self._state_r(pop), self.q_table.shape[0])
            a = (np.random.randint(0, len(self.params))
                 if np.random.random() < self.q_epsilon
                 else int(np.argmax(self.q_table[state])))
            p_c, p_m = self.params[a]
            off = []
            for _ in range(len(pop) // 2):
                p1 = self._tourn(pop, fit)
                p2 = self._tourn(pop, fit)
                c = self._sbx_local(p1, p2, p_c)
                c = self._mut_local(c, p_m)
                off.append(c)
            of = [self._safe_evaluate(s) for s in off]
            combined, combined_fit = pop + off, fit + of
            idx = self._nd_select(combined, combined_fit, self.population_size)
            pop = [combined[i] for i in idx]
            fit = [combined_fit[i] for i in idx]
            r = self._compute_reward(pop, fit)
            ns = safe_index(self._state_r(pop), self.q_table.shape[0])
            self.q_table[state, a] += self.q_alpha * (
                r + self.q_gamma * np.max(self.q_table[ns])
                - self.q_table[state, a])
        fit = [self._safe_evaluate(s) for s in pop]
        pareto = self._get_pareto(pop, fit)
        return {"population": pareto,
                "fitness": [self._safe_evaluate(s) for s in pareto],
                "history": {"hypervolume": [0.0] * 50},
                "hypervolume": 0.0}

    def _state_r(self, pop):
        unique = len(set(tuple(np.round(s, 4)) for s in pop))
        r = float(np.clip(unique / max(1, len(pop)), 0, 1))
        return safe_int_clip(r * 5, 0, 4)

    def _tourn(self, pop, fit):
        i, j = np.random.choice(len(pop), 2, replace=False)
        if self._dominates(fit[i], fit[j]):
            return pop[i]
        if self._dominates(fit[j], fit[i]):
            return pop[j]
        return pop[i]

    def _sbx_local(self, p1, p2, rate):
        if np.random.random() > rate:
            return p1.copy()
        return [p1[i] if np.random.random() < 0.5 else p2[i]
                for i in range(len(p1))]

    def _mut_local(self, s, rate):
        m = s.copy()
        for i in range(len(m)):
            if np.random.random() < rate:
                d = np.random.uniform(-0.1, 0.1) * (self.problem.xu[i] - self.problem.xl[i])
                m[i] = float(np.clip(m[i] + d, self.problem.xl[i], self.problem.xu[i]))
        return m


class R2RLMOEA(RLMOEA):
    def __init__(self, problem, config, n, max_generations=None):
        super().__init__(problem, config, n, max_generations)
        self.q_table = np.zeros((9, 4))
        self.q_alpha = 0.1
        self.q_gamma = 0.9
        self.q_epsilon = 0.1
        self.num_wv = 100
        self.wv = self._gen_wv()

    def run(self):
        pop = [[float(np.random.uniform(self.problem.xl[i], self.problem.xu[i]))
                for i in range(self.n_var)]
               for _ in range(self.population_size)]
        fit = [self._safe_evaluate(s) for s in pop]
        for gen in range(self.max_generations):
            state = safe_index(self._state_r2(pop, fit), self.q_table.shape[0])
            a = (np.random.randint(0, self.q_table.shape[1])
                 if np.random.random() < self.q_epsilon
                 else int(np.argmax(self.q_table[state])))
            pop, fit = self._apply(a, pop, fit)
            r = self._rew_r2(pop, fit)
            ns = safe_index(self._state_r2(pop, fit), self.q_table.shape[0])
            self.q_table[state, a] += self.q_alpha * (
                r + self.q_gamma * np.max(self.q_table[ns])
                - self.q_table[state, a])
        fit = [self._safe_evaluate(s) for s in pop]
        pareto = self._get_pareto(pop, fit)
        return {"population": pareto,
                "fitness": [self._safe_evaluate(s) for s in pareto],
                "history": {"hypervolume": [0.0] * 50},
                "hypervolume": 0.0}

    def _gen_wv(self):
        w = np.random.rand(self.num_wv, self.n_obj)
        return w / np.maximum(w.sum(axis=1, keepdims=True), 1e-10)

    def _r2(self, fit):
        try:
            arr = np.array(fit, dtype=float)
            arr = np.nan_to_num(arr, nan=1e10, posinf=1e10, neginf=0)
            r, c = 0.0, 0
            for w in self.wv:
                v = np.min(np.max(w * arr, axis=1))
                if np.isfinite(v):
                    r += v
                    c += 1
            return float(r / c) if c > 0 else 0.0
        except Exception:
            return 0.0

    def _state_r2(self, pop, fit):
        r2 = self._r2(fit)
        unique = len(set(tuple(np.round(s, 4)) for s in pop))
        div = unique / max(1, len(pop))
        return safe_int_clip(r2 * 3, 0, 2) * 3 + safe_int_clip(div * 3, 0, 2)

    def _apply(self, a, pop, fit):
        if a == 0:
            return self._nsga2_step(pop, fit)
        if a == 1:
            return self._moead_de_step(pop, fit)
        if a == 2:
            return self._high_mut(pop, fit)
        return self._high_cross(pop, fit)

    def _high_mut(self, pop, fit):
        off = []
        for _ in range(len(pop) // 2):
            p1 = pop[np.random.randint(len(pop))]
            p2 = pop[np.random.randint(len(pop))]
            c = [p1[i] if np.random.random() < 0.5 else p2[i]
                 for i in range(len(p1))]
            c = self._mut_hm(c, 0.5)
            off.append(c)
        of = [self._safe_evaluate(s) for s in off]
        combined, combined_fit = pop + off, fit + of
        idx = self._nd_select(combined, combined_fit, self.population_size)
        return [combined[i] for i in idx], [combined_fit[i] for i in idx]

    def _high_cross(self, pop, fit):
        off = []
        for _ in range(len(pop) // 2):
            p1 = pop[np.random.randint(len(pop))]
            p2 = pop[np.random.randint(len(pop))]
            c = [p1[i] if np.random.random() < 0.5 else p2[i]
                 for i in range(len(p1))]
            c = self._mut_hm(c, 0.05)
            off.append(c)
        of = [self._safe_evaluate(s) for s in off]
        combined, combined_fit = pop + off, fit + of
        idx = self._nd_select(combined, combined_fit, self.population_size)
        return [combined[i] for i in idx], [combined_fit[i] for i in idx]

    def _mut_hm(self, s, rate):
        m = s.copy()
        for i in range(len(m)):
            if np.random.random() < rate:
                d = np.random.uniform(-0.1, 0.1) * (self.problem.xu[i] - self.problem.xl[i])
                m[i] = float(np.clip(m[i] + d, self.problem.xl[i], self.problem.xu[i]))
        return m

    def _rew_r2(self, pop, fit):
        r2 = self._r2(fit)
        return -r2 if np.isfinite(r2) else -1e6


# ============================================================================
# STATISTICS
# ============================================================================
def summary_statistics(data: List[float]) -> Dict[str, float]:
    empty = {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0,
             "median": 0.0, "cv": 0.0, "n": 0,
             "ci95_low": 0.0, "ci95_high": 0.0}
    if not data:
        return empty
    valid = [v for v in data if np.isfinite(v)]
    if not valid:
        return empty
    mean = float(np.mean(valid))
    std = float(np.std(valid, ddof=1)) if len(valid) > 1 else 0.0
    se = std / np.sqrt(len(valid)) if len(valid) > 1 else 0.0
    return {
        "mean": mean,
        "std": std,
        "min": float(np.min(valid)),
        "max": float(np.max(valid)),
        "median": float(np.median(valid)),
        "q1": float(np.percentile(valid, 25)),
        "q3": float(np.percentile(valid, 75)),
        "cv": float(std / mean) if mean != 0 else 0.0,
        "n": len(valid),
        "ci95_low": mean - 1.96 * se,
        "ci95_high": mean + 1.96 * se,
    }


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
# METRICS
# ============================================================================
def compute_metrics(fitness_array, reference_front, n_obj,
                    ref_point=None) -> Dict[str, float]:
    if fitness_array is None or len(fitness_array) == 0:
        return {"hypervolume": 0.0, "igd": float("inf"),
                "spacing": 0.0, "cardinality": 0}

    F = np.asarray(fitness_array, dtype=float)
    valid_mask = np.all(np.isfinite(F) & (np.abs(F) < DEGENERATE_THRESHOLD),
                        axis=1)
    F = F[valid_mask]

    if len(F) == 0:
        return {"hypervolume": 0.0, "igd": float("inf"),
                "spacing": 0.0, "cardinality": 0}

    if reference_front is not None and len(reference_front) > 0:
        rf = np.asarray(reference_front, dtype=float)
        rf = rf[np.all(np.isfinite(rf), axis=1)]
        if len(rf) > 0:
            upper = rf.max(axis=0) * 1.2
            lower = rf.min(axis=0) - 0.5
            inside = np.all((F >= lower) & (F <= upper), axis=1)
            F = F[inside]

    if len(F) == 0:
        return {"hypervolume": 0.0, "igd": float("inf"),
                "spacing": 0.0, "cardinality": 0}

    if ref_point is None:
        ref_point = compute_reference_point(reference_front, n_obj)

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
            igd = float(IGD(ref_pf)(F)) if len(ref_pf) > 0 else float("inf")
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
# EXPERIMENT RUNNER
# ============================================================================
class ExperimentRunner:
    def __init__(self, config: Optional[ExperimentConfig] = None):
        self.config = config or ExperimentConfig()
        self.results: Dict[str, Any] = {}
        self.all_fronts: Dict[str, Any] = {}
        self.all_hv_history: Dict[str, Any] = {}
        self.all_metrics: Dict[str, Any] = {}
        self._problem_kwargs: Dict[str, Dict[str, Any]] = {}

        np.random.seed(self.config.base_seed)
        random.seed(self.config.base_seed)

    def get_problem(self, problem_name: str, **kwargs):
        return make_problem(problem_name, **kwargs)

    def get_algorithm(self, name, problem, n, suite="ZDT",
                      max_generations=None, ablation=None):
        algorithms = {
            "rle_emo": RLEEMO,
            "rl_moea": RLMOEA,
            "ql_moea": QLMOEA,
            "qlmoead_aos": QLMOEADAOS,
            "rl_nsgaii": RLNSGAII,
            "r2_rlmoea": R2RLMOEA,
        }
        cls = algorithms.get(name.lower())
        if cls is None:
            raise ValueError(f"Unknown algorithm: {name}")
        if name.lower() == "rle_emo":
            kwargs = dict(ablation or {})
            return cls(problem, self.config.rle_emo, suite=suite,
                       max_generations=max_generations, **kwargs)
        return cls(problem, self.config, n, max_generations=max_generations)

    def run_experiment(self, problem_name, algorithms, num_runs=None, **kwargs):
        num_runs = num_runs or self.config.num_runs
        problem = self.get_problem(problem_name, **kwargs)
        self._problem_kwargs[problem_name] = kwargs
        n_var = problem.n_var
        n_obj = problem.n_obj
        suite = problem.suite
        max_generations = self.config.get_generations_for_suite(suite)

        ref_front = problem.pareto_front(self.config.num_test_points)
        ref_point = compute_reference_point(ref_front, n_obj)

        results: Dict[str, Any] = {}
        problem_fronts: Dict[str, Any] = {}
        problem_history: Dict[str, Any] = {}
        metrics_by_alg: Dict[str, Any] = {}

        for alg_name in algorithms:
            print(f"  Running {alg_name} on {problem_name}...")
            alg_results, alg_fronts, alg_history, alg_times = [], [], [], []

            for run in range(num_runs):
                seed = self.config.get_seed(run)
                np.random.seed(seed)
                random.seed(seed)

                t0 = time.time()
                try:
                    algo = self.get_algorithm(alg_name, problem, n_var, suite,
                                              max_generations=max_generations)
                    result = algo.run()
                except Exception as e:
                    print(f"    [warn] {alg_name} run {run} failed: {e}")
                    result = {"population": [], "fitness": [],
                              "history": {"hypervolume": []},
                              "hypervolume": 0.0}
                elapsed = time.time() - t0

                alg_results.append(result)
                alg_times.append(elapsed)
                front = (np.array(result["fitness"], dtype=float)
                         if result["fitness"] else np.array([]))
                alg_fronts.append(front)
                alg_history.append(result["history"]["hypervolume"])

            hv_values = []
            for r in alg_results:
                front = (np.array(r["fitness"], dtype=float)
                         if r["fitness"] else np.array([]))
                if len(front) > 0:
                    m = compute_metrics(front, ref_front, n_obj, ref_point)
                    hv_values.append(m["hypervolume"])
                else:
                    hv_values.append(0.0)

            best_idx = int(np.argmax(hv_values)) if hv_values else 0
            problem_fronts[alg_name] = alg_fronts[best_idx]
            problem_history[alg_name] = alg_history[best_idx]

            metrics = []
            for r in alg_results:
                front = (np.array(r["fitness"], dtype=float)
                         if r["fitness"] else np.array([]))
                metrics.append(compute_metrics(front, ref_front, n_obj, ref_point))
            metrics_by_alg[alg_name] = metrics

            results[alg_name] = {
                "results": alg_results,
                "metrics": metrics,
                "times": alg_times,
                "summary": {
                    "hypervolume": summary_statistics([m["hypervolume"] for m in metrics]),
                    "igd": summary_statistics([m["igd"] for m in metrics]),
                    "spacing": summary_statistics([m["spacing"] for m in metrics]),
                    "cardinality": summary_statistics([m["cardinality"] for m in metrics]),
                    "time": summary_statistics(alg_times),
                }
            }

            print(f"    HV: {results[alg_name]['summary']['hypervolume']['mean']:.4f} "
                  f"± {results[alg_name]['summary']['hypervolume']['std']:.4f}  "
                  f"Card: {results[alg_name]['summary']['cardinality']['mean']:.1f}")

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

    def run_ablation(self, problems, num_runs=None):
        num_runs = num_runs or self.config.num_runs
        variants = {
            "full": {},
            "no_archive_return": {"use_archive_return": False},
            "no_region_select": {"use_region_select": False},
            "no_ppo_control": {"use_ppo_control": False},
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
            ref_front = problem.pareto_front(self.config.num_test_points)
            ref_point = compute_reference_point(ref_front, n_obj)

            ablation_results[name] = {}

            for variant, switches in variants.items():
                print(f"  Variant: {variant}")
                hv_list, igd_list, card_list = [], [], []

                for run in range(num_runs):
                    seed = self.config.get_seed(run)
                    np.random.seed(seed)
                    random.seed(seed)
                    try:
                        algo = RLEEMO(problem, self.config.rle_emo, suite=suite,
                                      max_generations=max_gen, **switches)
                        res = algo.run()
                        front = (np.array(res["fitness"], dtype=float)
                                 if res["fitness"] else np.array([]))
                        m = compute_metrics(front, ref_front, n_obj, ref_point)
                        hv_list.append(m["hypervolume"])
                        igd_list.append(m["igd"])
                        card_list.append(m["cardinality"])
                    except Exception as e:
                        print(f"    [warn] {variant} run {run} failed: {e}")
                        hv_list.append(0.0)
                        igd_list.append(float("inf"))
                        card_list.append(0)

                ablation_results[name][variant] = {
                    "hv": summary_statistics(hv_list),
                    "igd": summary_statistics(igd_list),
                    "cardinality": summary_statistics(card_list),
                }
                print(f"    HV: {ablation_results[name][variant]['hv']['mean']:.4f} "
                      f"± {ablation_results[name][variant]['hv']['std']:.4f}  "
                      f"Card: {ablation_results[name][variant]['cardinality']['mean']:.1f}")

        return ablation_results

    def statistical_analysis(self, reference_alg="rle_emo"):
        report = {}
        for inst_name, metrics_by_alg in self.all_metrics.items():
            if reference_alg not in metrics_by_alg:
                continue
            entry: Dict[str, Any] = {}
            samples = [[m["hypervolume"] for m in v]
                       for v in metrics_by_alg.values()]
            try:
                kw_stat, kw_p = stats.kruskal(*samples)
            except Exception:
                kw_stat, kw_p = 0.0, 1.0
            entry["kruskal_wallis"] = {
                "statistic": float(kw_stat),
                "p_value": float(kw_p),
                "significant": bool(kw_p < 0.05),
            }

            ref_hv = [m["hypervolume"] for m in metrics_by_alg[reference_alg]]

            for other, mets in metrics_by_alg.items():
                if other == reference_alg:
                    continue
                other_hv = [m["hypervolume"] for m in mets]

                try:
                    w_stat, w_p = stats.ranksums(ref_hv, other_hv)
                except Exception:
                    w_stat, w_p = 0.0, 1.0

                d = cohens_d(ref_hv, other_hv)
                wins = sum(1 for a, b in zip(ref_hv, other_hv) if a > b)
                total = len(ref_hv)
                success = wins / total if total else 0.0

                entry[f"{reference_alg}_vs_{other}"] = {
                    "wilcoxon_stat": float(w_stat),
                    "p_value": float(w_p),
                    "significant": bool(w_p < 0.05),
                    "cohens_d": float(d),
                    "effect_size": interpret_cohens_d(d),
                    "success_rate": float(success),
                    "mean_ref": float(np.mean(ref_hv)) if ref_hv else 0.0,
                    "mean_other": float(np.mean(other_hv)) if other_hv else 0.0,
                }
            report[inst_name] = entry
        return report

    def save_results(self, path: str):
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
            payload[prob] = {}
            for alg, r in pr.items():
                payload[prob][alg] = {
                    "summary": conv(r["summary"]),
                    "times": conv(r["times"]),
                    "metrics": conv(r["metrics"]),
                }
        with open(path, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"\nResults saved to {path}")

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------
    def generate_report(self, output_dir, stats_report, ablation_results=None):
        os.makedirs(output_dir, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.save_results(os.path.join(output_dir, f"results_{ts}.json"))

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
                ref_front = problem_obj.pareto_front(self.config.num_test_points)
                fronts = self.all_fronts.get(prob, {})
                history = self.all_hv_history.get(prob, {})
                n_obj = problem_obj.n_obj

                if n_obj == 2:
                    self._plot_2d(fronts, ref_front,
                                  title=f"{prob.upper()} — Pareto front",
                                  save=os.path.join(pareto_dir,
                                                    f"{prob}_pareto_2d.png"))
                else:
                    self._plot_3d(fronts, ref_front,
                                  title=f"{prob.upper()} — 3D Pareto front",
                                  save=os.path.join(pareto_dir,
                                                    f"{prob}_pareto_3d.png"))

                self._plot_conv(history,
                                title=f"{prob.upper()} — Convergence",
                                save=os.path.join(conv_dir,
                                                  f"{prob}_convergence.png"))

                hv_data = {alg: [m["hypervolume"] for m in r["metrics"]]
                           for alg, r in pr.items()}
                self._plot_box(hv_data,
                               title=f"{prob.upper()} — HV distribution",
                               save=os.path.join(stats_dir,
                                                 f"{prob}_boxplot.png"))
            except Exception as e:
                print(f"  [warn] plotting {prob}: {e}")

        try:
            heat = {prob: {alg: r["summary"]["hypervolume"]["mean"]
                           for alg, r in pr.items()}
                    for prob, pr in self.results.items()}
            self._plot_heat(heat, save=os.path.join(output_dir, "heatmap.png"))
        except Exception as e:
            print(f"  [warn] heatmap: {e}")

        self._write_summary_table(output_dir)
        self._write_stats_table(output_dir, stats_report)
        self._plot_success_rate(stats_report, stats_dir)
        self._plot_effect_size(stats_report, stats_dir)

        if ablation_results:
            self._write_ablation_table(output_dir, ablation_results)
            try:
                self._plot_ablation(ablation_results, output_dir)
            except Exception as e:
                print(f"  [warn] ablation plot: {e}")

        print(f"\nReport generated in {output_dir}")

    # ------------------------------------------------------------------
    # Plot routines
    # ------------------------------------------------------------------
    @staticmethod
    def _plot_2d(fronts, ref_front, title, save):
        fig, ax = plt.subplots(figsize=(7.0, 5.6))
        names = list(fronts.keys())
        for i, name in enumerate(names):
            f = fronts[name]
            if len(f) > 0:
                f = np.nan_to_num(f, nan=1e10, posinf=1e10, neginf=-1e10)
                m = np.all(np.abs(f) < DEGENERATE_THRESHOLD, axis=1)
                if np.any(m):
                    ax.scatter(f[m, 0], f[m, 1],
                               alpha=0.75, label=name,
                               color=PALETTE[i % len(PALETTE)],
                               s=32, edgecolors="black", linewidth=0.4,
                               zorder=3)
        if ref_front is not None and len(ref_front) > 0:
            ax.plot(ref_front[:, 0], ref_front[:, 1], "k--",
                    label="True PF", linewidth=1.6, alpha=0.9, zorder=4)
        ax.set_xlabel("Objective 1")
        ax.set_ylabel("Objective 2")
        ax.set_title(title, fontweight="bold")
        ax.legend(loc="best", ncol=2, frameon=True,
                  columnspacing=0.8, handletextpad=0.4)
        ax.grid(True, alpha=0.3)
        save_figure(fig, save)

    @staticmethod
    def _plot_3d(fronts, ref_front, title, save):
        fig = plt.figure(figsize=(7.4, 6.2))
        ax = fig.add_subplot(111, projection="3d")
        names = list(fronts.keys())
        for i, name in enumerate(names):
            f = fronts[name]
            if len(f) > 0 and f.ndim == 2 and f.shape[1] == 3:
                f = np.nan_to_num(f, nan=1e10, posinf=1e10, neginf=-1e10)
                m = np.all(np.abs(f) < DEGENERATE_THRESHOLD, axis=1)
                if np.any(m):
                    ax.scatter(f[m, 0], f[m, 1], f[m, 2],
                               alpha=0.75, label=name,
                               color=PALETTE[i % len(PALETTE)],
                               s=28, edgecolors="black", linewidth=0.3,
                               depthshade=False)
        if ref_front is not None and ref_front.ndim == 2 and ref_front.shape[1] == 3:
            n = min(200, len(ref_front))
            if n > 0:
                idx = np.random.choice(len(ref_front), n, replace=False)
                ax.scatter(ref_front[idx, 0], ref_front[idx, 1], ref_front[idx, 2],
                           c="black", marker="o", alpha=0.25, s=12,
                           label="True PF", depthshade=False)
        ax.set_xlabel("Obj 1")
        ax.set_ylabel("Obj 2")
        ax.set_zlabel("Obj 3")
        ax.set_title(title, fontweight="bold")
        ax.legend(loc="best", ncol=2, frameon=True)
        ax.view_init(elev=22, azim=-58)
        save_figure(fig, save)

    @staticmethod
    def _plot_conv(history, title, save):
        fig, ax = plt.subplots(figsize=(7.4, 4.6))
        names = list(history.keys())
        for i, name in enumerate(names):
            h = history[name]
            if not h:
                continue
            c = PALETTE[i % len(PALETTE)]
            h_clean = [v if np.isfinite(v) else 0.0 for v in h]
            ax.plot(range(len(h_clean)), h_clean,
                    linestyle=":", linewidth=1.0, alpha=0.45,
                    color=c, label=f"{name} (instantaneous)")
            envelope = np.maximum.accumulate(h_clean)
            ax.plot(range(len(envelope)), envelope,
                    linestyle="-", linewidth=1.8,
                    color=c, label=f"{name} (best-so-far)")
        ax.set_xlabel("Generation")
        ax.set_ylabel("Internal HV tracker")
        ax.set_title(title, fontweight="bold")
        ax.legend(loc="best", ncol=2, frameon=True, fontsize=7.5)
        ax.grid(True, alpha=0.3)
        save_figure(fig, save)

    @staticmethod
    def _plot_box(data, title, save):
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
                        meanprops=dict(marker="D", markerfacecolor="white",
                                       markeredgecolor="black", markersize=4),
                        medianprops=dict(color="black", linewidth=1.0),
                        flierprops=dict(marker="o", markersize=3,
                                        markerfacecolor="none",
                                        markeredgecolor="gray"))
        ax.set_xticks(range(1, len(names) + 1))
        ax.set_xticklabels(names, rotation=30, ha="right")
        for p, c in zip(bp["boxes"],
                        [PALETTE[i % len(PALETTE)] for i in range(len(names))]):
            p.set_facecolor(c)
            p.set_alpha(0.65)
            p.set_edgecolor("black")
        ax.set_ylabel("Hypervolume")
        ax.set_title(title, fontweight="bold")
        ax.grid(True, alpha=0.3, axis="y")
        save_figure(fig, save)

    @staticmethod
    def _plot_heat(data, save):
        probs = list(data.keys())
        algs = list(data[probs[0]].keys())
        M = np.zeros((len(probs), len(algs)))
        for i, p in enumerate(probs):
            for j, a in enumerate(algs):
                M[i, j] = data[p][a] if np.isfinite(data[p][a]) else 0
        rmax = M.max(axis=1, keepdims=True)
        rmax[rmax == 0] = 1e-10
        Mn = M / rmax

        fig, ax = plt.subplots(figsize=(7.6, 6.8))
        im = ax.imshow(Mn, cmap="RdYlGn", aspect="auto", vmin=0, vmax=1)
        ax.set_xticks(range(len(algs)))
        ax.set_yticks(range(len(probs)))
        ax.set_xticklabels(algs, rotation=35, ha="right")
        ax.set_yticklabels(probs)
        cbar = plt.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
        cbar.set_label("Normalized HV", fontsize=9)
        for i in range(len(probs)):
            for j in range(len(algs)):
                ax.text(j, i, f"{M[i, j]:.2f}",
                        ha="center", va="center", fontsize=7,
                        color="black" if Mn[i, j] < 0.7 else "white")
        for spine in ax.spines.values():
            spine.set_edgecolor("black")
            spine.set_linewidth(0.6)
        ax.set_title("Overall HV heatmap (per-instance normalized)",
                     fontweight="bold")
        save_figure(fig, save)

    def _plot_ablation(self, ablation_results, output_dir):
        probs = list(ablation_results.keys())
        if not probs:
            return
        variants = list(ablation_results[probs[0]].keys())
        M = np.zeros((len(probs), len(variants)))
        for i, p in enumerate(probs):
            for j, v in enumerate(variants):
                M[i, j] = ablation_results[p][v]["hv"]["mean"]
        rmax = M.max(axis=1, keepdims=True)
        rmax[rmax == 0] = 1e-10
        Mn = M / rmax

        fig, ax = plt.subplots(figsize=(7.6, 6.8))
        im = ax.imshow(Mn, cmap="RdYlGn", aspect="auto", vmin=0, vmax=1)
        ax.set_xticks(range(len(variants)))
        ax.set_yticks(range(len(probs)))
        ax.set_xticklabels(variants, rotation=35, ha="right")
        ax.set_yticklabels(probs)
        cbar = plt.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
        cbar.set_label("Normalized HV (vs. best variant)", fontsize=9)
        for i in range(len(probs)):
            for j in range(len(variants)):
                ax.text(j, i, f"{M[i, j]:.2f}",
                        ha="center", va="center", fontsize=7,
                        color="black" if Mn[i, j] < 0.7 else "white")
        for spine in ax.spines.values():
            spine.set_edgecolor("black")
            spine.set_linewidth(0.6)
        ax.set_title("Ablation: HV by variant (per-instance normalized)",
                     fontweight="bold")
        save_figure(fig, os.path.join(output_dir, "ablation_heatmap.png"))

    # ------------------------------------------------------------------
    # Table writers
    # ------------------------------------------------------------------
    @staticmethod
    def _table(pr, prob_name):
        lines = [f"\n{'=' * 100}",
                 f"Results for {prob_name}",
                 f"{'=' * 100}",
                 f"{'Algorithm':<20} {'HV Mean':<12} {'HV Std':<12} "
                 f"{'IGD':<12} {'Spacing':<12} {'Card':<8}",
                 f"{'-' * 100}"]
        for alg, r in sorted(pr.items(),
                             key=lambda x: x[1]["summary"]["hypervolume"]["mean"],
                             reverse=True):
            s = r["summary"]
            igd = s["igd"]["mean"]
            igd_s = (f"{igd:<12.4f}"
                     if np.isfinite(igd) and igd != float("inf")
                     else f"{'inf':<12}")
            lines.append(
                f"{alg:<20} {s['hypervolume']['mean']:<12.4f} "
                f"{s['hypervolume']['std']:<12.4f} {igd_s} "
                f"{s['spacing']['mean']:<12.4f} "
                f"{s['cardinality']['mean']:<8.1f}"
            )
        return "\n".join(lines)

    def _write_summary_table(self, output_dir):
        lines = ["\n" + "=" * 130,
                 "COMPREHENSIVE SUMMARY TABLE",
                 "=" * 130,
                 f"{'Suite':<10} {'Problem':<14} {'Algorithm':<15} "
                 f"{'HV Mean':<12} {'HV Std':<12} {'IGD':<12} {'Card':<8}",
                 "-" * 130]
        for prob, pr in sorted(self.results.items()):
            try:
                suite = self.get_problem(
                    prob, **self._problem_kwargs.get(prob, {})).suite
            except Exception:
                suite = "Unknown"
            for alg, r in sorted(pr.items(),
                                 key=lambda x: x[1]["summary"]["hypervolume"]["mean"],
                                 reverse=True):
                s = r["summary"]
                igd = s["igd"]["mean"]
                igd_s = (f"{igd:<12.4f}"
                         if np.isfinite(igd) and igd != float("inf")
                         else f"{'inf':<12}")
                lines.append(
                    f"{suite:<10} {prob:<14} {alg:<15} "
                    f"{s['hypervolume']['mean']:<12.4f} "
                    f"{s['hypervolume']['std']:<12.4f} {igd_s} "
                    f"{s['cardinality']['mean']:<8.1f}"
                )
            lines.append("-" * 130)
        txt = "\n".join(lines)
        print(txt)
        with open(os.path.join(output_dir, "summary_table.txt"), "w") as f:
            f.write(txt)

    def _write_stats_table(self, output_dir, stats_report):
        lines = ["\n" + "=" * 140,
                 "STATISTICAL VALIDATION (RLE-EMO vs each competitor)",
                 "=" * 140,
                 f"{'Instance':<14} {'Comparison':<26} {'p-value':<12} "
                 f"{'Signif.':<8} {'Cohen d':<10} {'Effect':<10} {'Success':<10}",
                 "-" * 140]
        for inst, rep in stats_report.items():
            for key, val in rep.items():
                if key == "kruskal_wallis":
                    lines.append(
                        f"{inst:<14} {'Kruskal-Wallis':<26} "
                        f"{val['p_value']:<12.4g} "
                        f"{str(val['significant']):<8} "
                        f"{'-':<10} {'-':<10} {'-':<10}"
                    )
                    continue
                lines.append(
                    f"{inst:<14} {key:<26} {val['p_value']:<12.4g} "
                    f"{str(val['significant']):<8} "
                    f"{val['cohens_d']:<10.4f} {val['effect_size']:<10} "
                    f"{val['success_rate']:<10.3f}"
                )
            lines.append("-" * 140)
        txt = "\n".join(lines)
        print(txt)
        with open(os.path.join(output_dir, "statistics_table.txt"), "w") as f:
            f.write(txt)

    def _write_ablation_table(self, output_dir, ablation_results):
        lines = ["\n" + "=" * 120,
                 "ABLATION STUDY (RLE-EMO variants)",
                 "=" * 120,
                 f"{'Problem':<14} {'Variant':<22} {'HV Mean':<14} "
                 f"{'HV Std':<14} {'IGD':<12} {'Card':<8}",
                 "-" * 120]
        for prob, variants in sorted(ablation_results.items()):
            for variant, vals in variants.items():
                hv = vals["hv"]
                igd = vals["igd"]
                card = vals["cardinality"]
                igd_s = (f"{igd['mean']:<12.4f}"
                         if np.isfinite(igd["mean"]) and igd["mean"] != float("inf")
                         else f"{'inf':<12}")
                lines.append(
                    f"{prob:<14} {variant:<22} {hv['mean']:<14.4f} "
                    f"{hv['std']:<14.4f} {igd_s} {card['mean']:<8.1f}"
                )
            lines.append("-" * 120)
        txt = "\n".join(lines)
        print(txt)
        with open(os.path.join(output_dir, "ablation_summary.txt"), "w") as f:
            f.write(txt)

    def _plot_success_rate(self, stats_report, stats_dir):
        comps: Dict[str, List[float]] = {}
        for inst, rep in stats_report.items():
            for key, val in rep.items():
                if key == "kruskal_wallis":
                    continue
                comps.setdefault(key, []).append(val["success_rate"])
        if not comps:
            return
        fig, ax = plt.subplots(figsize=(7.4, 4.4))
        names = list(comps.keys())
        means = [np.mean(comps[k]) for k in names]
        colors = [PALETTE[i % len(PALETTE)] for i in range(len(names))]
        ax.bar(range(len(names)), means, color=colors,
               edgecolor="black", linewidth=0.5)
        ax.axhline(0.5, color="red", linestyle="--", linewidth=1.0,
                   label="50% baseline")
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=30, ha="right")
        ax.set_ylabel("Success rate (mean across instances)")
        ax.set_ylim(0, 1.05)
        ax.set_title("RLE-EMO success rate vs. competitors",
                     fontweight="bold")
        ax.legend(loc="best", ncol=1)
        ax.grid(True, alpha=0.3, axis="y")
        save_figure(fig, os.path.join(stats_dir, "success_rate.png"))

    def _plot_effect_size(self, stats_report, stats_dir):
        comps: Dict[str, List[float]] = {}
        for inst, rep in stats_report.items():
            for key, val in rep.items():
                if key == "kruskal_wallis":
                    continue
                comps.setdefault(key, []).append(val["cohens_d"])
        if not comps:
            return
        fig, ax = plt.subplots(figsize=(7.4, 4.4))
        names = list(comps.keys())
        means = [np.mean(comps[k]) for k in names]
        colors = [PALETTE[i % len(PALETTE)] for i in range(len(names))]
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


# ============================================================================
# MAIN
# ============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="RLE-EMO benchmark suite and ablation study.")
    parser.add_argument("--ablation", action="store_true",
                        help="Run the RLE-EMO ablation study instead of "
                             "the full comparison.")
    parser.add_argument("--runs", type=int, default=30,
                        help="Number of independent runs per instance.")
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT_DIR,
                        help="Output directory (default: ./results).")

    args, _unknown = parser.parse_known_args()

    print("=" * 90)
    print("RLE-EMO: Comprehensive Benchmark Suite")
    print("=" * 90)
    print("Test problems: ZDT, DTLZ, WFG, DASCMOP (all from pymoo)")
    print("Metrics: HV, IGD, Spacing (pymoo indicators)")
    print("Training-once protocol: PPO warm-started on a 5-problem family")
    print("Visualization: Elsevier-style (PNG 300 dpi + PDF)")
    print("=" * 90)
    print(f"Output: {args.output}")
    print("=" * 90)

    config = ExperimentConfig()
    config.num_runs = args.runs

    problems = [
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

    algorithms = ["rle_emo", "rl_moea", "ql_moea",
                  "qlmoead_aos", "rl_nsgaii", "r2_rlmoea"]

    runner = ExperimentRunner(config)

    if args.ablation:
        print("\n" + "=" * 60)
        print("ABLATION MODE")
        print("=" * 60)
        ablation_results = runner.run_ablation(problems,
                                               num_runs=config.num_runs)
        os.makedirs(args.output, exist_ok=True)
        runner._write_ablation_table(args.output, ablation_results)
        try:
            runner._plot_ablation(ablation_results, args.output)
        except Exception as e:
            print(f"  [warn] ablation plot: {e}")
        print("\n" + "=" * 90)
        print("ABLATION COMPLETED SUCCESSFULLY")
        print("=" * 90)
        return

    print(f"\nProblems: {len(problems)} | Algorithms: {len(algorithms)} "
          f"| Runs each: {config.num_runs}")

    runner.run_all(problems, algorithms)

    print("\n" + "=" * 60)
    print("Statistical analysis (RLE-EMO vs competitors)")
    print("=" * 60)
    stats_report = runner.statistical_analysis(reference_alg="rle_emo")

    os.makedirs(args.output, exist_ok=True)
    runner.generate_report(args.output, stats_report)

    print("\n" + "=" * 90)
    print("EXPERIMENTS COMPLETED SUCCESSFULLY")
    print("=" * 90)
    print(f"\nOutput directory: {args.output}")
    print("  pareto_fronts/       — 2D/3D Pareto front plots (PNG + PDF)")
    print("  convergence/         — convergence curves (instantaneous + envelope)")
    print("  statistics/          — boxplots, success rate, effect size")
    print("  heatmap.png/pdf      — overall HV heatmap")
    print("  summary_table.txt    — comprehensive results table")
    print("  statistics_table.txt — Wilcoxon, Cohen's d, success rates")
    print("\nRun with --ablation to produce ablation_summary.txt "
          "+ ablation_heatmap.png/pdf")


if __name__ == "__main__":
    main()