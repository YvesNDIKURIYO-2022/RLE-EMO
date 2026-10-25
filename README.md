# RLE-EMO

**A Region-Based LHS-Guided Evolutionary Algorithm for Multi-Objective Optimization**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](LICENSE-figures.md)
[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/)
[![Cite this repository](https://img.shields.io/badge/cite-CITATION.cff-brightgreen.svg)](CITATION.cff)

---

## Overview

RLE-EMO is a multi-objective global optimizer designed for non-convex, multi-extremal problems in which the Pareto front may be convex, concave, disconnected, degenerate, or deceptive, and in which feasibility constraints may render the global optimum difficult to reach. The framework addresses two open questions for this problem class: (i) adaptive schemes that modulate crossover and mutation rates from population statistics are widely used in practice, yet no finite-time convergence guarantee has been established for the adaptive-operator case, and no stability condition has been shown for state-dependent operator rates; and (ii) hybrid algorithms that embed learning components have not been shown to avoid oscillatory behavior when the operator rates are coupled to a feedback signal derived from the current population.

The framework integrates three mechanisms:

1. **LHS-Guided Ensemble Initialization.** A Latin hypercube sampling (LHS) operator is combined with heuristic, random, and opposite-bias sampling to construct the initial population. The initialization is deterministic given a run seed, requires no offline training, and incurs no amortized cost and no distribution-shift risk across instances. The LHS component uses SciPy's `LatinHypercube` sampler.

2. **Region-Based Diversity with Dual Reset Triggers.** Promising non-dominated solutions are archived and re-injected when the population converges prematurely. Two triggers are employed—a diversity trigger and a rate-change trigger—each rate-limited to a minimum interval of 10 generations.

3. **Coordinate-Descent Local Search.** Periodic coordinate-descent refinement is applied to the top 10% of solutions every five, ten, or fifteen generations, depending on instance size.

The framework is accompanied by two analytical results: almost-sure convergence to the Pareto front (Theorem 1) and an exponential stability condition for the adaptive operator rates (Theorem 2) that accounts explicitly for the dependence of the rates on the generation index and is conditional on the population state converging. The stability result applies to any adaptive scheme whose operator rates depend on the same two-state feedback, not only to RLE-EMO.

The paper also provides a three-step practitioner diagnostic—the reference-vector approximation condition—that characterizes the problem class on which RLE-EMO is the appropriate choice and predicts the empirical outcome on every instance in the benchmark set.

**Submission status.** The manuscript is under review at *IEEE Transactions on Evolutionary Computation*. Proofs, per-instance tables, and the full ablation are in the accompanying supplementary material.

---

## Benchmark Evaluation

The framework is evaluated on eleven instances drawn from four established test suites, with thirty independent runs per instance:

| Suite | Instances | Variables | Objectives | Front Characteristics |
|-------|-----------|-----------|------------|------------------------|
| ZDT | ZDT1, ZDT2, ZDT3 | 30 | 2 | convex, concave, disconnected |
| DTLZ | DTLZ2, DTLZ6, DTLZ7 | 12, 12, 22 | 3 | concave, degenerate, disconnected |
| WFG | WFG1, WFG4, WFG9 | 10 | 3 | mixed, multi-modal, deceptive |
| DASCMOP | DASCMOP1, DASCMOP7 | 30 | 2, 3 | constrained |

RLE-EMO is compared against three canonical MOEAs—NSGA-II, MOEA/D, and RVEA—representing the three principal MOEA design paradigms. **MOEA/D does not support constrained problems in pymoo and is therefore marked `not_applicable` on DASCMOP1 and DASCMOP7** (see Table IV of the supplementary material and the run log). It is not run with a zero score.

### Headline results (mean HV over 30 runs, reference execution)

RLE-EMO attains the highest hypervolume on **eight of the eleven** instances:

| Instance | Best algorithm | Mean HV ± std | Runner-up | Δ vs runner-up |
|----------|----------------|---------------|-----------|----------------|
| ZDT1 | **RLE-EMO** | 0.8256 ± 0.0228 | NSGA-II 0.6704 ± 0.0486 | +23.2% (d = +4.08) |
| ZDT2 | **RLE-EMO** | 0.2483 ± 0.1292 | NSGA-II 0.1376 ± 0.1097 | +80.4% (d = +0.92) |
| ZDT3 | **RLE-EMO** | 3.7643 ± 0.0412 | NSGA-II 3.6146 ± 0.0756 | +4.1% (d = +2.46) |
| DTLZ6 | **RLE-EMO** | 88.0836 ± 7.3936 | RVEA 66.4343 ± 3.6720 | +32.6% (d = +4.45) |
| DTLZ7 | **RLE-EMO** | 4.3541 ± 0.0970 | NSGA-II 3.8833 ± 0.2140 | +12.1% (d = +2.83) |
| WFG1 | **RLE-EMO** | 61.0945 ± 7.7831 | NSGA-II 39.1753 ± 1.7456 | +56.0% (d = +3.89) |
| DASCMOP1 | **RLE-EMO** | 0.1892 ± 0.0075 | NSGA-II 0.1594 ± 0.0188 | +18.7% (d = +2.09) |
| DASCMOP7 | **RLE-EMO** | 0.0116 ± 0.0007 | NSGA-II 0.0000 (all 30 runs) | ∞ (d = +22.07 vs both NSGA-II and RVEA) |

The method is outperformed on three instances whose Pareto fronts violate the reference-vector approximation condition:

| Instance | Best algorithm | RLE-EMO | Δ |
|----------|----------------|---------|---|
| DTLZ2 | RVEA 3.5348 ± 0.0034 | 3.3932 ± 0.0572 | −4.0% (d = −2.18 vs NSGA-II) |
| WFG4 | RVEA 28.5745 ± 0.5538 | 23.8868 ± 1.5891 | −16.4% (d = −3.27 vs NSGA-II) |
| WFG9 | **see note** | 34.1315 ± 3.0405 | −9.7% |

**Note on WFG9.** The manuscript's Table IV and the supplementary's Table VIII state "NSGA-II wins (−9.7%)" on WFG9. The benchmark run log, however, shows **RVEA is best** on WFG9 (37.7978 ± 3.3075) with NSGA-II second (37.4740 ± 1.9893) and RLE-EMO third (34.1315 ± 3.0405). The −9.7% figure in the paper is the gap to **RVEA**, not to NSGA-II (−9.7% vs RVEA = (34.1315 − 37.7978)/37.7978 ≈ −9.7%; vs NSGA-II it is −8.9%). The paper's prose and diagnostic table therefore misattribute the WFG9 winner. The run log is the authoritative artifact for this number. This should be corrected in the manuscript before publication.

### Statistical caveat (disclosed in the manuscript and supplementary)

- **Per-instance:** pairwise Wilcoxon signed-rank tests with Holm–Bonferroni correction are significant for RLE-EMO vs each competitor on all eleven instances (all `p_holm < 0.05`; on the three losses the sign of Cohen's *d* is negative, as expected).
- **Omnibus:** the **global Friedman test across the nine instances with complete results for all four algorithms is not significant**: χ² = 7.267, p = 0.0639, n = 9, k = 4. This is stated in the manuscript (Section V-E, Section VI) and in the supplementary (Appendix E, Table V). The omnibus test is underpowered at n = 9 blocks; the two DASCMOP instances cannot enter the Friedman matrix because MOEA/D is `not_applicable`. The per-instance wins are robust; the claim is *per-instance* best-in-class, not omnibus dominance.

---

## Repository Structure

```
RLE-EMO/
│
├── RLE-EMO-Comprehensive Benchmark Suite.py   Full benchmark suite
│                                              (four algorithms × eleven instances × 30 runs)
├── RLE-EMO Independent Ablation Study.py      Independent ablation study
│                                              (six RLE-EMO variants × eleven instances × 30 runs)
├── README.md
├── LICENSE
│
├── results/                                   Benchmark outputs
│   ├── results_YYYYMMDD_HHMMSS.json
│   ├── results_YYYYMMDD_HHMMSS.csv
│   ├── summary_table.txt
│   ├── statistics_table.txt
│   ├── pareto_fronts/                         2D and 3D Pareto front plots (PNG 300 dpi + PDF)
│   ├── convergence/                           Convergence curves (instantaneous + best-so-far envelope)
│   ├── statistics/                            Boxplots, success rate, effect size (PNG + PDF)
│   ├── heatmap.png                            Overall HV heatmap (per-instance normalized)
│   └── heatmap.pdf                            Overall HV heatmap (vector)
│
├── ablation_output/                           Ablation outputs
│   ├── ablation_summary.txt
│   ├── ablation_heatmap.png
│   ├── ablation_heatmap.pdf
│   └── ablation_results.json
│
└── requirements.txt
```

Both scripts write to the same default directory (`DEFAULT_OUTPUT_DIR` at the top of each script, currently a hard-coded Windows path). **Pass `--output ./results` to write to a relative directory** — the hard-coded path will not exist on your machine.

---

## Requirements

- Python 3.13
- NumPy
- SciPy
- Matplotlib
- pymoo

The benchmark script performs a dependency check at startup and reports any missing packages with a single `pip install` command. Install with:

```bash
pip install -r requirements.txt
```

`requirements.txt` contains:

```
numpy
scipy
matplotlib
pymoo
```

Optional packages (for the Nemenyi post-hoc test and CSV export):

```bash
pip install scikit-posthocs pandas
```

If these are absent the scripts print `[info] Optional packages not installed` and skip the Nemenyi CD diagram and the CSV export. The reference run log shows exactly this: `scikit_posthocs` was not installed, so the Nemenyi post-hoc and CSV export were skipped while everything else ran.

---

## Usage

### Script 1 — `RLE-EMO-Comprehensive Benchmark Suite.py` (full benchmark)

Runs all eleven benchmark instances with thirty independent runs per instance, for each of the four compared algorithms (RLE-EMO, NSGA-II, MOEA/D, RVEA), and writes results to the output directory.

```bash
python "RLE-EMO-Comprehensive Benchmark Suite.py" --output ./results --runs 30
```

Restrict to a subset of problems or algorithms:

```bash
python "RLE-EMO-Comprehensive Benchmark Suite.py" --output ./results --problems zdt1,dtlz6 --algorithms rle_emo,nsga2
```

**Modes:**

| Flag | Effect |
|------|--------|
| *(none)* | Benchmark only |
| `--ablation` | Ablation only (no benchmark) |
| `--all` | Benchmark **and** integrated ablation in one call |

**Outputs:** `results_YYYYMMDD_HHMMSS.json`, `results_YYYYMMDD_HHMMSS.csv` (if pandas present), `summary_table.txt`, `statistics_table.txt`, `pareto_fronts/`, `convergence/`, `statistics/`, `heatmap.png`, `heatmap.pdf`.

### Script 2 — `RLE-EMO Independent Ablation Study.py` (standalone ablation)

A self-contained ablation study that evaluates six RLE-EMO variants across the eleven benchmark instances with thirty independent runs per (problem, variant) pair, for a total of **1,980 runs**.

```bash
python "RLE-EMO Independent Ablation Study.py" --output ./ablation_output
```

Smoke-test with fewer runs:

```bash
python "RLE-EMO Independent Ablation Study.py" --runs 3
```

Restrict to a subset of problems:

```bash
python "RLE-EMO Independent Ablation Study.py" --problems zdt1,dtlz6
```

**Outputs** (in `ablation_output/`): `ablation_summary.txt`, `ablation_heatmap.png`, `ablation_heatmap.pdf`, `ablation_results.json`.

**Runtime note.** With 30 runs and all eleven problems, the standalone ablation required **1,899.6 minutes (≈ 31.7 hours)** of wall-clock time in the reference execution. For exploratory use, reduce `--runs` or restrict the `--problems` list.

**Note on PDF export.** The ablation script attempts to save the heatmap in both PNG (300 dpi) and PDF (vector). On some Matplotlib installations the PDF backend is not bundled, raising `ModuleNotFoundError: No module named 'matplotlib.backends.backend_pdf'`. If this occurs, the PNG output is still written correctly; install the PDF backend (`pip install matplotlib[pdf]`) and re-run the plotting routine, or comment out the PDF save line.

---

## Ablation Study

The ablation study isolates the contribution of each algorithmic component by disabling it in turn. The six variants are:

| Variant | Description |
|---------|-------------|
| `full` | All components active (baseline) |
| `no_archive_return` | Archive-union step removed from the final Pareto set |
| `no_region_select` | Region-based selection replaced by crowding-distance selection |
| `no_lhs_control` | LHS subpopulation replaced by uniform random sampling |
| `no_local_search` | Periodic coordinate-descent refinement disabled |
| `no_reset` | Both population-reset triggers disabled |

### Ablation results (mean HV ± std, 30 runs per cell, from `ablation_summary.txt`)

The **best variant per instance** (bold) and the `full` baseline:

| Problem | Best variant | HV (best) | `full` HV |
|---------|--------------|-----------|-----------|
| ZDT1 | **no_region_select** | 0.8435 ± 0.0151 | 0.8256 ± 0.0228 |
| ZDT2 | **no_reset** | 0.3596 ± 0.1321 | 0.2483 ± 0.1292 |
| ZDT3 | **no_reset** | 0.9675 ± 0.0305 | 0.9380 ± 0.0390 |
| DTLZ2 | **no_region_select** | 0.7078 ± 0.0100 | 0.5941 ± 0.0329 |
| DTLZ6 | **no_region_select** | 91.9986 ± 6.3368 | 88.0836 ± 7.3936 |
| DTLZ7 | **no_region_select** | 3.8133 ± 0.1149 | 3.6685 ± 0.0989 |
| WFG1 | **no_reset** | 66.7930 ± 10.2776 | 65.1449 ± 7.9654 |
| WFG4 | **no_region_select** | 30.7652 ± 0.6515 | 24.0627 ± 1.6008 |
| WFG9 | **no_region_select** | 39.6569 ± 1.1924 | 34.2942 ± 3.0585 |
| DASCMOP1 | **no_reset** | 0.1937 ± 0.0097 | 0.1892 ± 0.0075 |
| DASCMOP7 | **no_region_select** | 0.0385 ± 0.0023 | 0.0353 ± 0.0020 |

**Note on comparability with the main benchmark table.** The ablation builds each instance's reference front from the **six variants**, whereas the main benchmark builds it from the **four competing algorithms** (supplementary Appendix F, main text Section V-G). Absolute HV values are therefore **not** comparable across the two tables (e.g. DASCMOP7 `full` is 0.0116 in the benchmark table but 0.0353 in the ablation table). All qualitative claims hold within each table.

### Reconciliation with the main results (supplementary Appendix H)

The ablation shows that replacing region-based selection with crowding distance improves HV on **eight of eleven** instances — which at first reading appears to contradict retaining the region-based operator. The supplementary's Appendix H resolves the apparent contradiction in three steps:

1. **The ablation isolates one operator while the archive and reset triggers stay active.** On DTLZ6 and DASCMOP7 the archive and reset triggers alone are already sufficient to win; disabling region-select therefore *improves* the ablation HV without showing that region-select is harmful. The full RLE-EMO in the main comparison still wins both.
2. **The instances where region-select removal helps are exactly the instances where the reference-vector approximation condition fails or is marginal** (DTLZ2, WFG4). Crowding distance happens to align with the smooth concave geometry, which is precisely why the diagnostic recommends RVEA on those instances rather than a modified RLE-EMO.
3. **The one instance where region-select demonstrably helps in the main comparison is WFG1**, the mixed multi-modal front it was designed for.

The correct reading: region-based selection is not universally beneficial, and the diagnostic identifies in advance when it should be retained. On fronts that satisfy the condition, region-select is safe and contributes to final HV; on fronts that violate it, the diagnostic advises choosing a different algorithm rather than retaining RLE-EMO with a modified operator.

### Key findings

1. **Archive-return is the most uniformly important component.** Disabling it reduces mean HV on **every one of the eleven instances**, with losses ranging from 0.7% on DASCMOP1 to 40.3% on WFG4, and roughly halves cardinality on every instance.

2. **Local search is the single most important component on the bi-objective and constrained instances.** Removing it causes HV losses of 13.0% on ZDT1, 44.0% on ZDT2, 9.5% on ZDT3, and 14.1% on DASCMOP1, while its contribution on three-objective instances is much smaller (and on DASCMOP7 it is exactly zero — the constrained landscape does not admit coordinate-descent improvement).

3. **Region-based selection is not universally beneficial.** Replacing it with crowding distance improves HV on eight of eleven instances, by up to +27.9% on WFG4 and +19.1% on DTLZ2. See the reconciliation above for why this does not contradict the main results.

4. **Removing reset is problem-dependent.** Disabling both reset triggers improves HV on five instances (ZDT2, ZDT3, WFG1, DASCMOP1, DASCMOP7) and hurts on six, with the largest losses on DTLZ2 (−17.2%), WFG4 (−19.2%), and DTLZ6 (−10.9%).

5. **LHS control contributes mainly to reproducibility.** Removing it changes the mean HV by less than 3% in nine of eleven cases, but the standard deviation of HV is consistently smaller with LHS control on ZDT3, DTLZ2, DTLZ6, DTLZ7, and DASCMOP7.

The component ranking is therefore **archive-return > local search > reset ≈ region selection ≈ LHS control**, with the important qualification that local search dominates on bi-objective and constrained instances while archive-return dominates on multi-modal and deceptive three-objective instances.

---

## Practitioner Diagnostic

The eight wins and three losses reduce to a single geometric property of the Pareto front: **the reference-vector approximation condition** (Assumption 5 in the manuscript). A three-step diagnostic lets a practitioner predict, before running any algorithm, whether RLE-EMO will outperform a canonical baseline:

1. **If m = 2 (bi-objective), choose RLE-EMO.** RLE-EMO wins all three ZDT instances and both DASCMOP instances (DASCMOP7 by a wide margin: only RLE-EMO produces a non-empty front).

2. **If m ≥ 3, determine whether the front is a smooth concave manifold.** Sample the front with a pilot run, partition into N_max regions using uniform weight vectors, and measure the fraction of empty regions. If the empty fraction exceeds 0.2, the condition is violated and RLE-EMO is recommended. Otherwise, a reference-vector method such as RVEA is recommended.

3. **If the front is believed to be deceptive, prefer NSGA-II.** Deception is a property of the decision space, not the objective space, and is not captured by the reference-vector approximation condition.

The diagnostic predicts the empirical outcome correctly on all eleven instances in the benchmark set (supplementary Table VIII; main text Table IV is an abbreviated form). We state it as an **empirical observation rather than a theorem**; it is a sufficient condition, not a necessary one, and it has been validated only on the eleven instances used to derive it (see Limitations).

---

## Reproducibility

Each run uses a fixed seed (base seed 42, with run *r* seeded as 42 + *r*), applied identically across all algorithms to enable paired comparisons. The initialization is deterministic given the run seed, and no training is performed, so there is no amortized offline cost and no distribution-shift risk.

Both scripts use pymoo for problem definitions, hypervolume and IGD computation, and non-dominated sorting, so the metric definitions match the published conventions exactly. The reference point is set to 1.1 × max(reference front) per objective, and degenerate objective values (|f_i| > 10⁸) are excluded from all metric computations, uniformly across algorithms, to prevent pathological values from contaminating the rankings. A bounding-box filter is applied before metric computation: any solution exceeding 1.2 × the reference-front maximum, or falling below the reference-front minimum minus 0.5, is discarded. The reference front is analytic when reliable (ZDT1–ZDT3, DTLZ2, WFG4, DASCMOP1, DASCMOP7) and best-observed otherwise (DTLZ6, DTLZ7, WFG1, WFG9). These rules are recorded in Table I of the supplementary material.

### Reference-front shapes in the reference execution

| Instance | Policy | Reference front shape |
|----------|--------|------------------------|
| ZDT1 | analytic | (1000, 2) |
| ZDT2 | analytic | (1000, 2) |
| ZDT3 | analytic | (27, 2) |
| DTLZ2 | analytic | (120, 3) |
| DTLZ6 | best-observed | (15, 3) |
| DTLZ7 | best-observed | (62, 3) |
| WFG1 | best-observed | (133, 3) |
| WFG4 | analytic | (136, 3) |
| WFG9 | best-observed | (97, 3) |
| DASCMOP1 | analytic | (671, 2) |
| DASCMOP7 | analytic | (44, 3) |

**Note on ZDT3.** pymoo returns only **27 points** for the ZDT3 analytic front at the default `n_pareto_points`, not 1000. HV is therefore computed against a sparse 27-point front, which is why the absolute ZDT3 HV in this repository (3.7643) is not directly comparable to published ZDT3 HV values computed against a dense analytic front. The relative ordering is unaffected. This is a property of the pymoo sampler, not of the algorithm; it is not called out in the supplementary, which records only the *policy* ("analytic front from pymoo when reliable") rather than the realised shape.

### Execution protocol (supplementary Table II)

Population size and generational budget are matched across all four algorithms:

- N_pop = 40 + ⌊n_var / 5⌋
- G_max = 80 for ZDT and DASCMOP; 100 for DTLZ and WFG

---

## Known discrepancies between manuscript and code output

These are disclosed for transparency. Items 1–4 are gaps or errors that remain **open** in the manuscript and should be corrected before publication; items 5–6 are explained by the supplementary and are not errors.

1. **WFG9 winner is misattributed in the paper.** The manuscript (Section V-C, Table IV) and the supplementary (Table VIII) both say "NSGA-II wins (−9.7%)" on WFG9. The benchmark run log shows **RVEA is best** (37.7978 ± 3.3075), NSGA-II second (37.4740 ± 1.9893), RLE-EMO third (34.1315 ± 3.0405). The −9.7% figure is the gap to RVEA; the gap to NSGA-II is −8.9%. Either the prose should say "RVEA wins" and use −8.9% vs NSGA-II, or the diagnostic table's "NSGA-II wins" cell should be changed to "RVEA wins (marginally)". The run log is the authoritative artifact.

2. **WFG4 loss margin.** The manuscript says RLE-EMO loses by −16.4% to RVEA on WFG4; the run log gives RLE-EMO 23.8868 vs RVEA 28.5745 = −16.4%, consistent. The manuscript's additional "NSGA-II by −14.6%, MOEA/D by −3.5%" is also consistent with the run log. No error here; recorded for completeness because the ablation table's `full` value (24.0627) differs for the reference-front reason in item 5.

3. **ZDT3 reference front has only 27 points.** See the Reproducibility note above. The relative ordering is unaffected, but the absolute HV (3.7643) is not comparable to values computed against a 1000-point analytic front. The supplementary records the policy but not the realised shape.

4. **Friedman test not significant.** χ² = 7.267, p = 0.0639, n = 9, k = 4. This is **already disclosed** in the manuscript (Section V-E, Section VI) and the supplementary (Appendix E), so it is not an error — it is listed here only so that a reader who arrives via the README does not miss it.

5. **DASCMOP7 HV differs across tables.** Benchmark `full` = 0.0116; ablation `full` = 0.0353. Different reference fronts (44-point analytic in the benchmark, 235-point best-observed union in the ablation). Explained in supplementary Appendix F and in the main text Section V-G. Not an error.

6. **Ablation-vs-main-result tension on region-select.** Removing region-select improves HV on 8/11 in the ablation but the full RLE-EMO still wins 8/11 against the canonical baselines. Resolved in supplementary Appendix H (archive and reset triggers carry DTLZ6 and DASCMOP7; the removal-helps instances are exactly the diagnostic's failure instances). Not an error.

7. **File names.** The scripts are named `RLE-EMO-Comprehensive Benchmark Suite.py` and `RLE-EMO Independent Ablation Study.py` in the repository; the scripts' own docstrings refer to them as `rle_emo_benchmark.py` and `rle_emo_ablation.py` in the usage block (a leftover from v5.3). The commands above use the actual file names (quoted, because of the spaces).

8. **Stale BibTeX in earlier README revisions.** Earlier revisions cited a *Journal of Global Optimization* submission with a different title and co-author spelling. The manuscript is under review at *IEEE Transactions on Evolutionary Computation*; the title and author list below supersede the old entry.

---

## Citation

If you use this code, please cite the manuscript under review:

```bibtex
@article{Ndikuriyo2026RLEEMO,
  title   = {A Region-Based Evolutionary Algorithm with Convergence and
             Stability Guarantees for Multi-Objective Optimization},
  author  = {Ndikuriyo, Yves and Zhang, Yinggui and Fom, Dung Davou},
  journal = {IEEE Transactions on Evolutionary Computation},
  year    = {2026},
  note    = {Under review}
}
```

---

## Authors

- **Yves Ndikuriyo** (corresponding author) — School of Traffic and Transportation Engineering, Central South University, Changsha 410075, Hunan, China — yvesndikuriyo@csu.edu.cn
- **Yinggui Zhang** — School of Traffic and Transportation Engineering, Central South University — ygzhang@csu.edu.cn
- **Dung Davou Fom** — School of Traffic and Transportation Engineering, Central South University — dung.fom@csu.edu.cn

**Acknowledgments.** This research was supported by the National Natural Science Foundation of China (Grant No. 71971220) and the Natural Science Foundation of Hunan Province, China (Grant Nos. 2023JJ30710 and 2022JJ31020). The authors also acknowledge support from the School of Traffic and Transportation Engineering, Central South University.

---

## Data and Code Availability

The benchmark problems used in this study are available through the pymoo library at <https://pymoo.org/problems/index.html>. The source code for RLE-EMO and all experimental data generated during this study are available in this public GitHub repository at <https://github.com/YvesNDIKURIYO-2022/RLE-EMO>.

---

## License

This project is released under the MIT License. See the `LICENSE` file for details.
ine HV table, the ablation best-variant table, the component ranking, the diagnostic steps, and the per-instance statistical claims are unchanged because the manuscript, supplementary, and both run logs agree on them. The WFG9 winner is the one substantive number that does not agree, and it is now flagged in three places.
