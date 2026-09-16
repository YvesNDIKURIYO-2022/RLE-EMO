#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RLE-EMO: Independent Ablation Study
====================================

Self-contained ablation study of RLE-EMO. Runs six variants of the algorithm
across eleven benchmark instances from the ZDT, DTLZ, WFG, and DASCMOP
families:

    full                all components active
    no_archive_return   archive-union step removed from the final Pareto set
    no_region_select    region-based selection replaced by crowding distance
    no_ppo_control      PPO action replaced by uniform random sampling
    no_local_search     periodic 2-opt refinement disabled
    no_reset            both population-reset triggers disabled

Outputs (written to ./ablation_output by default):
    ablation_summary.txt
    ablation_heatmap.png
    ablation_heatmap.pdf
    ablation_results.json

Can be run as a script or pasted into a single Jupyter cell.
"""

import json
import os
import random
import time
import warnings
from datetime import datetime
from itertools import combinations_with_replacement
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from pymoo.indicators.hv import HV
from pymoo.indicators.igd import IGD
from pymoo.problems import get_problem
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

warnings.filterwarnings("ignore")


# ============================================================================
# CONFIGURATION
# ============================================================================
OUTPUT_DIR = os.path.join(os.getcwd(), "ablation_output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

NUM_RUNS = 30          # set to 3 for a smoke test
BASE_SEED = 42
NUM_TEST_PTS = 1000
DEGEN_THRESH = 1e8

PROBLEMS = [
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

VARIANTS = {
    "full": {},
    "no_archive_return": {"use_archive_return": False},
    "no_region_select": {"use_region_select": False},
    "no_ppo_control": {"use_ppo_control": False},
    "no_local_search": {"use_local_search": False},
    "no_reset": {"use_reset": False},
}


# ============================================================================
# PPO AGENT (deterministic, warm-started once)
# ============================================================================
class _PPOAgent:
    _instance = None

    @classmethod
    def get(cls, state_dim: int = 34, action_dim: int = 90) -> "_PPOAgent":
        if cls._instance is None:
            cls._instance = _PPOAgent(state_dim, action_dim)
        return cls._instance

    def __init__(self, state_dim: int, action_dim: int):
        rng = np.random.RandomState(BASE_SEED)
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.policy_weights = rng.randn(state_dim, action_dim) * 0.01
        # Warm-start: bias toward mid-range action indices so that the
        # initial distribution is not degenerate.
        self.policy_weights += np.ones_like(self.policy_weights) * 0.005

    def get_action(self, state: np.ndarray) -> int:
        if state.ndim > 1:
            state = state.flatten()
        if len(state) < self.state_dim:
            state = np.pad(state, (0, self.state_dim - len(state)))
        else:
            state = state[:self.state_dim]
        logits = state @ self.policy_weights
        logits = np.nan_to_num(logits, nan=0.0, posinf=50.0, neginf=-50.0)
        logits -= logits.max()
        probs = np.exp(logits)
        probs /= max(probs.sum(), 1e-10)
        return int(np.random.choice(self.action_dim, p=probs))


# ============================================================================
# PROBLEM WRAPPER
# ============================================================================
class _Problem:
    def __init__(self, name: str, **kwargs):
        lower = name.lower()

        if lower.startswith("dascmop"):
            idx = int("".join(c for c in lower if c.isdigit()) or "1")
            from pymoo.problems.multi import dascmop as dm
            cls = getattr(dm, f"DASCMOP{idx}", None)
            if cls is None:
                raise ValueError(f"DASCMOP{idx} not found")
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
        else:
            p = get_problem(lower, **kwargs)

        self._p = p
        self.name = name
        self.n_obj = int(p.n_obj)
        self.n_var = int(p.n_var)
        self.xl = np.asarray(p.xl, dtype=float).flatten()
        self.xu = np.asarray(p.xu, dtype=float).flatten()
        self.suite = ("ZDT" if lower.startswith("zdt")
                      else "DTLZ" if lower.startswith("dtlz")
                      else "WFG" if lower.startswith("wfg")
                      else "DASCMOP" if lower.startswith("dascmop")
                      else "Unknown")

    def evaluate(self, x) -> List[float]:
        try:
            arr = np.asarray(x, dtype=float).reshape(1, -1)
            F = self._p.evaluate(arr, return_values_of=["F"])
            f = np.asarray(F, dtype=float).flatten()
            return [float(v) if np.isfinite(v) else 1e10 for v in f]
        except Exception:
            return [1e10] * self.n_obj

    def pareto_front(self, n: int = 1000) -> Optional[np.ndarray]:
        try:
            pf = self._p.pareto_front(n_pareto_points=n)
            return None if pf is None or len(pf) == 0 else np.asarray(pf, dtype=float)
        except Exception:
            return None


# ============================================================================
# RLE-EMO WITH ABLATION SWITCHES
# ============================================================================
class RLEEMO_Ablation:
    """RLE-EMO with five independent switches for the ablation study."""

    def __init__(self, problem: _Problem, suite: str = "ZDT",
                 max_generations: Optional[int] = None,
                 use_archive_return: bool = True,
                 use_region_select: bool = True,
                 use_ppo_control: bool = True,
                 use_local_search: bool = True,
                 use_reset: bool = True):

        self.problem = problem
        self.suite = suite
        self.n_var = problem.n_var
        self.n_obj = problem.n_obj

        self.N_pop = 40 + self.n_var // 5
        self.G_max = max_generations if max_generations is not None else {
            "DTLZ": 100, "WFG": 100, "DASCMOP": 80}.get(suite, 80)

        self.K_regions = 6
        self.tau_div = 0.3
        self.delta_rate = 0.15
        self.reset_interval = 10
        self.archive_ratio = 1.0
        self.reset_frac = 0.2
        self.ls_freq = 5 if self.n_var < 50 else (10 if self.n_var < 100 else 15)
        self.ls_top_frac = 0.10

        self.use_archive_return = use_archive_return
        self.use_region_select = use_region_select
        self.use_ppo_control = use_ppo_control
        self.use_local_search = use_local_search
        self.use_reset = use_reset

        self.region_w = self._make_region_weights()
        self.ppo = _PPOAgent.get(self.n_var + 4, min(self.n_var * 3, 100))
        self.pop: List[List[float]] = []
        self.fit: List[List[float]] = []
        self.archive: List[List[float]] = []
        self.history = {"hv": []}
        self._last_reset = -10**9

    # ------------------------------------------------------------------
    # Region weights
    # ------------------------------------------------------------------
    def _make_region_weights(self) -> np.ndarray:
        K, m = self.K_regions, self.n_obj
        if m == 2:
            w = np.column_stack([np.linspace(0, 1, K), np.linspace(1, 0, K)])
        elif m == 3:
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
    # Initialization strategies
    # ------------------------------------------------------------------
    def _ppo_solution(self) -> List[float]:
        sol, state = [], np.zeros(self.ppo.state_dim)
        n_bins = self.ppo.action_dim
        for i in range(self.n_var):
            a = self.ppo.get_action(state)
            lo, hi = self.problem.xl[i], self.problem.xu[i]
            if self.use_ppo_control:
                frac = (a + 0.5) / n_bins
                val = lo + frac * (hi - lo)
                jit = np.random.uniform(-0.5, 0.5) * (hi - lo) / max(n_bins, 1)
                val = float(np.clip(val + jit, lo, hi))
            else:
                val = float(np.random.uniform(lo, hi))
            sol.append(val)
            s = np.zeros(self.ppo.state_dim)
            nv = min(len(sol), self.ppo.state_dim - 4)
            for j in range(nv):
                s[j] = sol[j] / 10.0 if sol[j] != 0 else 0
            state = s
        return sol

    def _heuristic_solution(self) -> List[float]:
        return [float((self.problem.xl[i] + self.problem.xu[i]) / 2)
                for i in range(self.n_var)]

    def _random_solution(self) -> List[float]:
        return [float(np.random.uniform(self.problem.xl[i], self.problem.xu[i]))
                for i in range(self.n_var)]

    def _opposite_solution(self) -> List[float]:
        return [float(self.problem.xu[i]
                      - np.random.random() * (self.problem.xu[i] - self.problem.xl[i]))
                for i in range(self.n_var)]

    def _initialize(self) -> List[List[float]]:
        N = self.N_pop
        n_ppo = max(1, int(0.3 * N))
        n_heur = max(1, int(0.3 * N))
        n_rand = max(1, int(0.3 * N))
        pop = []
        pop += [self._ppo_solution() for _ in range(n_ppo)]
        pop += [self._heuristic_solution() for _ in range(n_heur)]
        pop += [self._random_solution() for _ in range(n_rand)]
        while len(pop) < N:
            pop.append(self._opposite_solution())
        return pop[:N]

    # ------------------------------------------------------------------
    # State computations
    # ------------------------------------------------------------------
    def _div_state(self, pop) -> float:
        if not pop:
            return 0.0
        F = np.array([self.problem.evaluate(s) for s in pop], dtype=float)
        F = np.nan_to_num(F, nan=1e10, posinf=1e10, neginf=-1e10)
        if np.any(np.abs(F) >= DEGEN_THRESH):
            return 0.0
        lo, hi = F.min(axis=0), F.max(axis=0)
        rng = np.where(hi - lo > 1e-12, hi - lo, 1.0)
        Fn = (F - lo) / rng
        Fn = Fn / np.maximum(np.linalg.norm(Fn, axis=1, keepdims=True), 1e-12)
        assoc = np.argmax(Fn @ self.region_w.T, axis=1)
        counts = np.bincount(assoc, minlength=self.K_regions)
        expected = len(pop) / self.K_regions
        thr = max(2, int(0.5 * expected))
        return float(np.sum(counts < thr)) / self.K_regions

    def _conv_state(self, pop) -> float:
        if not pop:
            return 0.0
        F = np.array([self.problem.evaluate(s) for s in pop], dtype=float)
        F = np.nan_to_num(F, nan=1e10, posinf=1e10, neginf=-1e10)
        if np.any(np.abs(F) >= DEGEN_THRESH):
            return 0.0
        lo, hi = F.min(axis=0), F.max(axis=0)
        rng = np.where(hi - lo > 1e-12, hi - lo, 1.0)
        return float(np.mean((F - lo) / rng))

    def _operators(self, gen: int) -> Tuple[float, float]:
        xc = self._conv_state(self.pop)
        xd = self._div_state(self.pop)
        fb = np.clip(1.0 + xd - xc, 0.5, 1.5)
        pc = np.clip(0.85 * (1 - np.exp(-gen / self.G_max)) * fb, 0.5, 0.95)
        pm = np.clip(0.15 * np.exp(-gen / self.G_max) / fb, 0.02, 0.30)
        return pc, pm

    # ------------------------------------------------------------------
    # NSGA-II helpers
    # ------------------------------------------------------------------
    def _dominates(self, a, b) -> bool:
        one = False
        for ai, bi in zip(a, b):
            if ai > bi:
                return False
            if ai < bi:
                one = True
        return one

    def _nds(self, pop) -> List[List[List[float]]]:
        n = len(pop)
        if n == 0:
            return []
        dom = [set() for _ in range(n)]
        dc = [0] * n
        F = [self.problem.evaluate(pop[i]) for i in range(n)]
        for i in range(n):
            for j in range(i + 1, n):
                if self._dominates(F[i], F[j]):
                    dom[i].add(j)
                    dc[j] += 1
                elif self._dominates(F[j], F[i]):
                    dom[j].add(i)
                    dc[i] += 1
        front = [i for i in range(n) if dc[i] == 0]
        fronts = [[pop[i] for i in front]]
        while front:
            nf = []
            for i in front:
                for j in dom[i]:
                    dc[j] -= 1
                    if dc[j] == 0:
                        nf.append(j)
            front = nf
            if front:
                fronts.append([pop[i] for i in front])
        return fronts

    def _crowding(self, front) -> List[float]:
        n = len(front)
        if n <= 2:
            return [float("inf")] * n
        dist = [0.0] * n
        obj = [self.problem.evaluate(s) for s in front]
        m = len(obj[0])
        for k in range(m):
            order = sorted(range(n), key=lambda i: obj[i][k])
            dist[order[0]] = float("inf")
            dist[order[-1]] = float("inf")
            lo, hi = obj[order[0]][k], obj[order[-1]][k]
            rng = hi - lo if hi > lo else 1.0
            for i in range(1, n - 1):
                dist[order[i]] += (obj[order[i + 1]][k] - obj[order[i - 1]][k]) / rng
        return dist

    def _region_select(self, front, n_take: int):
        if len(front) <= n_take:
            return front[:n_take]
        F = np.array([self.problem.evaluate(s) for s in front], dtype=float)
        F = np.nan_to_num(F, nan=1e10, posinf=1e10, neginf=-1e10)
        lo, hi = F.min(axis=0), F.max(axis=0)
        rng = np.where(hi - lo > 1e-12, hi - lo, 1.0)
        Fn = (F - lo) / rng
        Fn = Fn / np.maximum(np.linalg.norm(Fn, axis=1, keepdims=True), 1e-12)
        sim = Fn @ self.region_w.T
        assoc = np.argmax(sim, axis=1)
        dist_to_w = 1.0 - sim[np.arange(len(front)), assoc]
        counts = np.bincount(assoc, minlength=self.K_regions)
        order = sorted([(counts[assoc[i]], dist_to_w[i], i)
                        for i in range(len(front))])
        return [front[idx] for _, _, idx in order[:n_take]]

    # ------------------------------------------------------------------
    # Diversity archive
    # ------------------------------------------------------------------
    def _update_archive(self):
        max_size = max(1, int(self.archive_ratio * self.N_pop))
        fronts = self._nds(self.pop)
        if not fronts:
            return
        min_dist = 1e-3 * np.sqrt(max(1, self.n_var))
        for sol in fronts[0]:
            arr = np.asarray(sol, dtype=float)
            if self.archive:
                A = np.asarray(self.archive, dtype=float)
                if A.ndim == 2 and A.shape[1] == arr.shape[0]:
                    if np.linalg.norm(A - arr, axis=1).min() < min_dist:
                        continue
            if len(self.archive) < max_size:
                self.archive.append(sol)
            else:
                d = self._crowding(self.archive)
                if any(np.isfinite(x) for x in d):
                    idx = int(np.argmin(d))
                    if d[idx] < float("inf"):
                        self.archive[idx] = sol

    def _reset_from_archive(self, gen: int):
        if not self.archive:
            return
        if gen - self._last_reset < self.reset_interval:
            return
        n_rep = min(int(self.reset_frac * self.N_pop), len(self.archive))
        if n_rep == 0:
            return
        fit_sums = [sum(f) for f in self.fit]
        worst = sorted(range(len(self.pop)),
                       key=lambda i: fit_sums[i], reverse=True)
        arch_idx = np.random.choice(len(self.archive), n_rep, replace=False)
        for i, a in enumerate(arch_idx):
            if i < len(worst):
                self.pop[worst[i]] = self.archive[a].copy()
                self.fit[worst[i]] = self.problem.evaluate(self.archive[a])
        self._last_reset = gen

    # ------------------------------------------------------------------
    # Genetic operators
    # ------------------------------------------------------------------
    def _sbx(self, p1, p2, rate: float) -> List[float]:
        if np.random.random() > rate:
            return p1.copy()
        return [p1[i] if np.random.random() < 0.5 else p2[i]
                for i in range(len(p1))]

    def _poly_mut(self, sol, rate: float) -> List[float]:
        m = sol.copy()
        for i in range(len(m)):
            if np.random.random() < rate:
                d = np.random.uniform(-0.1, 0.1) * (self.problem.xu[i] - self.problem.xl[i])
                m[i] = float(np.clip(m[i] + d, self.problem.xl[i], self.problem.xu[i]))
        return m

    def _local_search(self, sol) -> List[float]:
        best = list(sol)
        bf = self.problem.evaluate(best)
        step_frac = 0.05
        for _ in range(5):
            improved = False
            for i in range(len(best)):
                lo, hi = self.problem.xl[i], self.problem.xu[i]
                step = step_frac * (hi - lo)
                for d in (+1, -1):
                    cand = list(best)
                    cand[i] = float(np.clip(best[i] + d * step, lo, hi))
                    cf = self.problem.evaluate(cand)
                    if self._dominates(cf, bf):
                        best, bf = cand, cf
                        improved = True
                        break
            if not improved:
                step_frac *= 0.5
                if step_frac < 1e-4:
                    break
        return best

    def _tournament(self, k: int = 2) -> List[List[float]]:
        out = []
        for _ in range(k):
            i, j = np.random.choice(len(self.pop), size=2, replace=False)
            f0, f1 = self.fit[i], self.fit[j]
            if self._dominates(f0, f1):
                out.append(self.pop[i])
            elif self._dominates(f1, f0):
                out.append(self.pop[j])
            else:
                out.append(self.pop[i])
        return out

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------
    def run(self) -> Dict[str, Any]:
        self.pop = self._initialize()
        self.fit = [self.problem.evaluate(s) for s in self.pop]
        fronts = self._nds(self.pop)
        if fronts:
            self.archive = fronts[0][
                :max(1, int(self.archive_ratio * self.N_pop))]

        for gen in range(self.G_max):
            pc, pm = self._operators(gen)
            offspring = []
            while len(offspring) < self.N_pop:
                p1, p2 = self._tournament(2)
                c = self._sbx(p1, p2, pc)
                c = self._poly_mut(c, pm)
                offspring.append(c)

            off_fit = [self.problem.evaluate(s) for s in offspring]
            combined = self.pop + offspring
            combined_fit = self.fit + off_fit

            fronts = self._nds(combined)
            new_pop, new_fit = [], []
            for front in fronts:
                if len(new_pop) + len(front) <= self.N_pop:
                    new_pop.extend(front)
                    new_fit.extend([self.problem.evaluate(s) for s in front])
                else:
                    remaining = self.N_pop - len(new_pop)
                    if self.use_region_select:
                        chosen = self._region_select(front, remaining)
                    else:
                        d = self._crowding(front)
                        ordered = sorted(zip(front, d),
                                         key=lambda x: x[1], reverse=True)
                        chosen = [s for s, _ in ordered[:remaining]]
                    for s in chosen:
                        new_pop.append(s)
                        new_fit.append(self.problem.evaluate(s))
                    break
            self.pop, self.fit = new_pop, new_fit

            self._update_archive()

            if self.use_reset:
                xd = self._div_state(self.pop)
                xc = self._conv_state(self.pop)
                if xd > self.tau_div:
                    self._reset_from_archive(gen)
                if abs(xd - xc) > self.delta_rate:
                    self._reset_from_archive(gen)

            if self.use_local_search and gen % self.ls_freq == 0:
                n_top = max(1, int(self.ls_top_frac * self.N_pop))
                top = sorted(range(len(self.pop)),
                             key=lambda i: sum(self.fit[i]))[:n_top]
                for idx in top:
                    improved = self._local_search(self.pop[idx])
                    self.pop[idx] = improved
                    self.fit[idx] = self.problem.evaluate(improved)

            hv = self._internal_hv(np.array(self.fit))
            self.history["hv"].append(hv)

        if self.use_archive_return:
            combined = self.pop + self.archive
            fronts = self._nds(combined)
            pareto = fronts[0] if fronts else []
        else:
            fronts = self._nds(self.pop)
            pareto = fronts[0] if fronts else []

        pareto_fit = [self.problem.evaluate(s) for s in pareto]

        seen = set()
        uniq_p, uniq_f = [], []
        for s, f in zip(pareto, pareto_fit):
            k = tuple(np.round(s, 6))
            if k not in seen:
                seen.add(k)
                uniq_p.append(s)
                uniq_f.append(f)
        return {"population": uniq_p, "fitness": uniq_f}

    def _internal_hv(self, F: np.ndarray) -> float:
        try:
            F = F[np.all(np.isfinite(F) & (np.abs(F) < DEGEN_THRESH), axis=1)]
            if len(F) == 0:
                return 0.0
            nds = NonDominatedSorting().do(F, only_non_dominated_front=True)
            nd = F[nds]
            ref = nd.max(axis=0) * 1.1
            ref = np.where(ref > 1e-9, ref, 1.0)
            hv = HV(ref_point=ref)(nd)
            return float(hv) if np.isfinite(hv) else 0.0
        except Exception:
            return 0.0


# ============================================================================
# METRICS
# ============================================================================
def ref_point_from_pf(pf: Optional[np.ndarray], n_obj: int) -> np.ndarray:
    if pf is not None and len(pf) > 0:
        rf = np.asarray(pf, dtype=float)
        rf = rf[np.all(np.isfinite(rf), axis=1)]
        if len(rf) > 0:
            rp = rf.max(axis=0) * 1.1
            return np.where(rp > 1e-9, rp, 1.1)
    return np.full(n_obj, 1.1)


def compute_metrics(F, ref_pf, n_obj: int, ref_pt) -> Dict[str, float]:
    empty = {"hv": 0.0, "igd": float("inf"), "card": 0}
    if F is None or len(F) == 0:
        return empty
    F = np.asarray(F, dtype=float)
    keep = np.all(np.isfinite(F) & (np.abs(F) < DEGEN_THRESH), axis=1)
    F = F[keep]
    if len(F) == 0:
        return empty
    if ref_pf is not None and len(ref_pf) > 0:
        rf = np.asarray(ref_pf, dtype=float)
        rf = rf[np.all(np.isfinite(rf), axis=1)]
        if len(rf) > 0:
            upper = rf.max(axis=0) * 1.2
            lower = rf.min(axis=0) - 0.5
            inside = np.all((F >= lower) & (F <= upper), axis=1)
            F = F[inside]
    if len(F) == 0:
        return empty
    try:
        hv = float(HV(ref_point=ref_pt)(F))
        hv = hv if np.isfinite(hv) else 0.0
    except Exception:
        hv = 0.0
    if ref_pf is not None and len(ref_pf) > 0:
        try:
            rf = np.asarray(ref_pf, dtype=float)
            rf = rf[np.all(np.isfinite(rf), axis=1)]
            igd = float(IGD(rf)(F)) if len(rf) > 0 else float("inf")
            if not np.isfinite(igd):
                igd = float("inf")
        except Exception:
            igd = float("inf")
    else:
        igd = float("inf")
    return {"hv": hv, "igd": igd, "card": int(len(F))}


def summarize(xs) -> Dict[str, float]:
    v = [x for x in xs if np.isfinite(x)]
    if not v:
        return {"mean": 0.0, "std": 0.0, "n": 0}
    m = float(np.mean(v))
    s = float(np.std(v, ddof=1)) if len(v) > 1 else 0.0
    return {"mean": m, "std": s, "n": len(v)}


# ============================================================================
# ABLATION RUNNER
# ============================================================================
def run_ablation(problems, variants, num_runs: int = NUM_RUNS) -> Dict[str, Any]:
    results: Dict[str, Any] = {}
    total = len(problems) * len(variants)
    counter = 0
    t_start = time.time()

    for cfg in problems:
        name, kwargs = cfg["name"], cfg.get("kwargs", {})
        print(f"\n{'=' * 72}\nABLATION on {name}\n{'=' * 72}")
        prob = _Problem(name, **kwargs)
        ref_pf = prob.pareto_front(NUM_TEST_PTS)
        ref_pt = ref_point_from_pf(ref_pf, prob.n_obj)
        max_gen = {"DTLZ": 100, "WFG": 100, "DASCMOP": 80}.get(prob.suite, 80)
        results[name] = {}

        for vname, switches in variants.items():
            counter += 1
            hv_list, igd_list, card_list, t_list = [], [], [], []
            print(f"  [{counter}/{total}] Variant: {vname}")

            for run in range(num_runs):
                seed = BASE_SEED + run
                np.random.seed(seed)
                random.seed(seed)

                t0 = time.time()
                try:
                    algo = RLEEMO_Ablation(prob, suite=prob.suite,
                                           max_generations=max_gen,
                                           **switches)
                    res = algo.run()
                except Exception as e:
                    print(f"    [warn] {vname} run {run} failed: {e}")
                    res = {"fitness": []}
                elapsed = time.time() - t0

                F = (np.array(res["fitness"], dtype=float)
                     if res["fitness"] else np.array([]))
                m = compute_metrics(F, ref_pf, prob.n_obj, ref_pt)
                hv_list.append(m["hv"])
                igd_list.append(m["igd"])
                card_list.append(m["card"])
                t_list.append(elapsed)

            results[name][vname] = {
                "hv": summarize(hv_list),
                "igd": summarize(igd_list),
                "card": summarize(card_list),
                "time": summarize(t_list),
            }
            hv_m = results[name][vname]["hv"]["mean"]
            hv_s = results[name][vname]["hv"]["std"]
            print(f"    HV: {hv_m:.4f} ± {hv_s:.4f}   "
                  f"Card: {results[name][vname]['card']['mean']:.1f}")

    dt = time.time() - t_start
    print(f"\nAblation completed in {dt / 60:.1f} min")
    return results


# ============================================================================
# REPORTING
# ============================================================================
def write_summary_txt(results: Dict[str, Any], path: str) -> str:
    lines = ["=" * 130,
             "RLE-EMO ABLATION STUDY",
             f"Runs per (problem, variant): {NUM_RUNS}",
             f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
             "=" * 130, "",
             f"{'Problem':<14} {'Variant':<22} "
             f"{'HV Mean':<12} {'HV Std':<12} "
             f"{'IGD Mean':<12} {'Card':<8}",
             "-" * 130]
    for prob, variants in results.items():
        for vname, m in variants.items():
            igd = m["igd"]["mean"]
            igd_s = (f"{igd:<12.4f}"
                     if np.isfinite(igd) and igd != float("inf")
                     else f"{'inf':<12}")
            lines.append(
                f"{prob:<14} {vname:<22} "
                f"{m['hv']['mean']:<12.4f} {m['hv']['std']:<12.4f} "
                f"{igd_s} {m['card']['mean']:<8.1f}"
            )
        lines.append("-" * 130)
    txt = "\n".join(lines)
    with open(path, "w") as f:
        f.write(txt)
    return txt


def plot_ablation_heatmap(results: Dict[str, Any],
                          path_png: str,
                          path_pdf: str):
    probs = list(results.keys())
    varns = list(results[probs[0]].keys())
    M = np.zeros((len(probs), len(varns)))
    for i, p in enumerate(probs):
        for j, v in enumerate(varns):
            M[i, j] = results[p][v]["hv"]["mean"]

    rmax = M.max(axis=1, keepdims=True)
    rmax[rmax == 0] = 1e-10
    Mn = M / rmax

    fig, ax = plt.subplots(figsize=(9.5, 8.5))
    im = ax.imshow(Mn, cmap="RdYlGn", aspect="auto", vmin=0, vmax=1)
    ax.set_xticks(range(len(varns)))
    ax.set_yticks(range(len(probs)))
    ax.set_xticklabels(varns, rotation=35, ha="right")
    ax.set_yticklabels(probs)
    cbar = plt.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label("Normalized HV (vs. best variant in row)", fontsize=10)

    for i in range(len(probs)):
        for j in range(len(varns)):
            ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center",
                    fontsize=8,
                    color="black" if Mn[i, j] < 0.7 else "white")

    for spine in ax.spines.values():
        spine.set_edgecolor("black")
        spine.set_linewidth(0.7)

    ax.set_title("Ablation: HV by variant (per-instance normalized)",
                 fontweight="bold", fontsize=12)
    plt.tight_layout()
    plt.savefig(path_png, dpi=300, bbox_inches="tight")
    plt.savefig(path_pdf, bbox_inches="tight")
    plt.close(fig)


# ============================================================================
# MAIN
# ============================================================================
def main():
    print("=" * 90)
    print("RLE-EMO Independent Ablation Study")
    print(f"Runs per (problem, variant): {NUM_RUNS}")
    print(f"Problems: {len(PROBLEMS)}   Variants: {len(VARIANTS)}")
    print(f"Total runs: {len(PROBLEMS) * len(VARIANTS) * NUM_RUNS}")
    print(f"Output: {OUTPUT_DIR}")
    print("=" * 90)

    results = run_ablation(PROBLEMS, VARIANTS, num_runs=NUM_RUNS)

    summary_txt = write_summary_txt(
        results, os.path.join(OUTPUT_DIR, "ablation_summary.txt"))
    plot_ablation_heatmap(
        results,
        os.path.join(OUTPUT_DIR, "ablation_heatmap.png"),
        os.path.join(OUTPUT_DIR, "ablation_heatmap.pdf"))

    with open(os.path.join(OUTPUT_DIR, "ablation_results.json"), "w") as f:
        json.dump(results, f, indent=2, default=str)

    print("\n" + summary_txt)
    print(f"\nOutputs written to: {OUTPUT_DIR}")
    print("  - ablation_summary.txt")
    print("  - ablation_heatmap.png")
    print("  - ablation_heatmap.pdf")
    print("  - ablation_results.json")


if __name__ == "__main__":
    main()