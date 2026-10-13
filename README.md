<div align="center">

# Physics-Informed Neural Networks for Mechanics

### Reproducing a PINN study, finding where it fails, and fixing it

![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white)
![Tests](https://img.shields.io/badge/tests-49%20passing-2EA44F)
![Seeds](https://img.shields.io/badge/every%20result-5%E2%80%9310%20seeds-6E56CF)

**Daruru Bhuvan Sai** (PES1UG24AM356) · **Vismay H S** (PES1UG24AM325)
UE24CS352A Machine Learning · PES University

[**Report (PDF)**](report/PINN_Mechanics_Report.pdf) · [Results](#results-at-a-glance) · [Quick start](#quick-start) · [Reproduce](#reproducing-every-result) · [Code structure](#repository-structure)

<br>

<img src="results/figures/summary.png" alt="Summary: the paper's method vs our fixes on all three problems" width="100%">

</div>

## Highlights

- **Reproduced all three experiments** of the base paper over 5–10 random seeds each, against exact solutions.
- **Found the cause of each remaining error.** It is a slow-to-learn tail, a corner where the solution jumps, and corners where no smooth network can satisfy the physics.
- **Fixed every one.** The fixes give 10/10 reliable parameter discovery at 30x the speed, about 200x lower error in consolidation, and about 140x lower worst-case error in heat conduction.
- **Tested what the paper didn't:** noisy and sparse data, against a classical fit and the theoretical best (the Cramer-Rao bound).
- **Fully reproducible.** There are 49 unit tests, seeded runs that are bit-identical on the same machine, and one script per result.

## Contents

1. [Background](#background)
2. [Results at a glance](#results-at-a-glance)
3. [Damped oscillator: finding c and k](#1-damped-oscillator-finding-c-and-k)
4. [Consolidation: the error at the corner](#2-consolidation-the-error-at-the-corner)
5. [Heat conduction: edges, then corners](#3-heat-conduction-edges-then-corners)
6. [Noisy and sparse data](#4-noisy-and-sparse-data)
7. [Conclusions](#conclusions)
8. [Quick start](#quick-start)
9. [Reproducing every result](#reproducing-every-result)
10. [Repository structure](#repository-structure)
11. [Design decisions](#design-decisions)
12. [References](#references)

## Background

**What a PINN is.** A physics-informed neural network learns a function that obeys a differential equation. Automatic differentiation gives the network's derivatives, so the equation's residual $r$ can be evaluated at any point. Training minimises that residual together with the boundary, initial-condition and data misfits:

```math
\mathcal{L} \;=\; w_{\text{PDE}}\,\overline{r^{2}} \;+\; w_{\text{BC}}\,\overline{(N-g)^{2}} \;+\; w_{\text{IC}}\,\overline{(N-u_0)^{2}} \;+\; w_{\text{data}}\,\overline{(N-u_{\text{obs}})^{2}}
```

In an *inverse* problem, unknown physical constants are trained alongside the weights.

**The base paper.** Zhang, Chen and Yang, *Data Driven Solutions and Discoveries in Mechanics Using Physics Informed Neural Network* (Stanford CS229), apply PINNs to three problems. They report a single run of each. We asked three questions:

- Does the result hold across seeds?
- Where does the remaining error sit, and why?
- Can it be removed?

## Results at a glance

| Problem (paper section) | Paper's method, reproduced | Our fix | Result |
|:--|:--|:--|:--|
| **Damped oscillator**: find damping *c* and stiffness *k* (4.3) | 3/5 seeds within 1%; 699 s per run | Two-stage warm start | **10/10 seeds; 23 s per run** |
| **Soil consolidation** (4.1) | Relative L2 error 1.4e-2; pressure overshoots to 1.11 | Corner-enriched output | **6.4e-5 (about 200x lower); no overshoot** |
| **Heat conduction** (4.2) | Max error 3.8e-3, on the edges | Exact boundary condition + exact corner terms | **2.7e-5 (about 140x lower)** |
| **Noisy, sparse data** (new) | Not tested | Benchmark vs classical fit and Cramer-Rao bound | **Classical fit wins; PINNs need dense data** |

All numbers are medians over seeds.

---

## 1. Damped oscillator: finding c and k

```math
u'' + c\,u' + k\,u = 0,\qquad u(0)=u'(0)=1,\qquad t\in[0,15]
```

Fifty measurements of $u$ are given; the true values are $c = 0.4$ and $k = 4$, and training starts from $c = 0.2$, $k = 2$.

**Finding.** With the paper's settings only 3 of 5 seeds find both values within 1%. The parameters always match the curve the network currently draws. That curve is learned from left to right, so its tail stays flat for a long time. A flat tail looks like heavy damping and satisfies the ODE exactly, while the late oscillations that encode $c$ carry only 1% of the data's energy.

**Fix: two-stage warm start.**

1. Fit the data without physics.
2. Read $c$ and $k$ off the fitted curve by least squares on its derivatives.
3. Train the PINN with them frozen.
4. Release them and train jointly.

| Method | Seeds within 1% | *c* error, median (worst) | Time per run |
|:--|:-:|:-:|:-:|
| Paper (Adam 100k) | 3/5 | 9.1e-3 (4.0e-2) | 699 s |
| Adam 20k + L-BFGS | 8/10 | 7.3e-3 (1.1e-1) | 59 s |
| **Two-stage (ours)** | **10/10** | **1.4e-3 (5.6e-3)** | **23 s** |

Ablations show that step 2 is the key ingredient: without it, success falls to 7/10.

<details>
<summary><b>Full figure</b>: every seed, ablations and run time</summary>
<br>
<img src="results/figures/oscillator_two_stage.png" alt="Oscillator results" width="100%">
</details>

Code: [`oscillator_reproduce.py`](experiments/oscillator_reproduce.py), [`oscillator_two_stage.py`](experiments/oscillator_two_stage.py) · Tables: [reproduction](results/tables/oscillator_reproduce_summary.md), [fix](results/tables/oscillator_two_stage_summary.md)

## 2. Consolidation: the error at the corner

```math
p_t = p_{xx},\quad 0<x<1,\; 0<t\le 0.5;\qquad p(0,t)=0,\quad p_x(1,t)=0,\quad p(x,0)=1
```

**Finding.** The paper reports a "perfect match". Over 5 seeds the max error is 0.061 (median) and up to 0.166, and the pressure overshoots its physical limit of 1. At $x = 0$, $t = 0$ the exact solution jumps from 1 to 0, an infinitely steep front. The first 2% of the time window holds 11 of the 500 training points but 59% of the squared error. On fresh points the PDE residual is 49, against 5.5e-6 at the training points: the training loss hides the error.

**Fix: corner-enriched output.**

```math
p(x,t) \;=\; \mathrm{erf}\!\left(\frac{x}{2\sqrt{t}}\right) \;+\; x\,t\,N(x,t)
```

The erf term is the exact solution for a layer with no base and carries the jump. The network learns only the smooth correction. We compare it with residual-based adaptive sampling (RAD), the "adaptive re-meshing" that the paper lists as future work.

| Method | Relative L2 error | Highest pressure (limit 1) | Time per run |
|:--|:-:|:-:|:-:|
| Paper (Adam 50k) | 1.4e-2 | 1.108 | 1,143 s |
| Adaptive sampling (RAD) | 1.3e-2 | 1.076 | 395 s |
| **Corner-enriched output (ours)** | **6.4e-5** | **1.000** | **168 s** |

RAD moves points toward the corner, but no placement of points lets a smooth network represent an infinitely steep front.

<details>
<summary><b>Full figure</b>: error over time, profiles near the corner, where RAD puts its points</summary>
<br>
<img src="results/figures/consolidation_enrichment.png" alt="Consolidation results" width="100%">
</details>

Code: [`consolidation_reproduce.py`](experiments/consolidation_reproduce.py), [`consolidation_enrichment.py`](experiments/consolidation_enrichment.py) · Tables: [reproduction](results/tables/consolidation_reproduce_summary.md), [fix](results/tables/consolidation_enrichment_summary.md)

## 3. Heat conduction: edges, then corners

```math
T_{xx} + T_{yy} + 1 = 0 \;\text{ on } (-1,1)^2,\qquad T = 0 \;\text{ on the edges}
```

The paper states that no analytical solution exists. A classical series solution does exist; we implement it and verify it with our own finite-difference solver.

**Finding 1: the edges.** The paper enforces $T = 0$ only as a loss penalty. Its largest error lies on the edges, and the plate dips below 0.
**Fix 1:** $T = (1-x^2)(1-y^2)\,N(x,y)$ makes the boundary condition exact. The paper suggests this but does not test it. The error falls about 30-fold.

**Finding 2: the corners.** Along the top edge $T_{xx} = 0$; along the side edge the PDE forces $T_{xx} = -1$. The exact solution resolves this with a direction-dependent value, $T_{xx} \to -2\theta/\pi$, which no smooth network can reproduce. As a result, every exact-boundary network has a PDE residual of exactly 1 at each corner, whatever its weights. Our unit tests check both facts without training.
**Fix 2:** add the exact local solution of a 90° corner, derived by hand, at each corner, so the network learns only a smooth remainder:

```math
u = r^2\left[-\tfrac14 + \tfrac14\cos 2\theta - \tfrac{1}{\pi}\left(\ln r\,\sin 2\theta + \theta\cos 2\theta\right)\right]
```

| Configuration | Relative L2 error | Max error | Residual, fresh / training points |
|:--|:-:|:-:|:-:|
| Paper: soft boundary condition | 1.3e-3 | 3.8e-3 | 2.2x |
| Exact boundary condition | 4.2e-5 | 7.7e-5 | 97.7x |
| **Exact BC + corner terms (ours)** | **3.9e-5** | **2.7e-5** | **1.8x** |

<details>
<summary><b>Full figure</b>: the corner analysis, residual and error vs distance to a corner, error maps</summary>
<br>
<img src="results/figures/heat_corners.png" alt="Heat corner results" width="100%">
</details>

Code: [`heat_reproduce.py`](experiments/heat_reproduce.py), [`heat_corners.py`](experiments/heat_corners.py) · Tables: [reproduction](results/tables/heat_reproduce_summary.md), [corners](results/tables/heat_corners_summary.md)

## 4. Noisy and sparse data

The paper tests parameter discovery only on clean data at 50 points. We ran 4 methods across 3 data settings × 4 noise levels × 10 draws:

- **Data settings:** 50 points; 15 points; 50 points over $t \le 5$ only.
- **Noise levels:** 0, 1, 3 and 10%.
- **Benchmark:** the Cramer-Rao bound, the best accuracy any unbiased method can reach.

| Method | *c* error, 1% noise, 50 points | *c* error, 1% noise, 15 points | Time per dataset |
|:--|:-:|:-:|:-:|
| Ideal (Cramer-Rao bound) | 4.3e-3 | 8.1e-3 | |
| Classical ODE fit | 5.5e-3 | 1.2e-2 | 0.2 s |
| Standard PINN | 8.6e-3 | 1.5e-2 | 61 s |
| Two-stage PINN (ours) | 4.6e-3 | 2.7e-1 | 38 s |

With a known ODE and two parameters, the classical fit is the right tool. The two-stage PINN comes within 7% of the bound on dense data but fails on sparse data. When data stop at $t = 5$, both PINNs extrapolate far better than a plain network.

<details>
<summary><b>Full figure</b>: error vs noise for every method and data setting</summary>
<br>
<img src="results/figures/oscillator_noise.png" alt="Noise study" width="100%">
</details>

Code: [`oscillator_noise.py`](experiments/oscillator_noise.py) · Table: [summary](results/tables/oscillator_noise_summary.md)

---

## Conclusions

1. **One run is not evidence.** The paper's results hold for only some seeds.
2. **Training loss can mislead.** Near corners the loss looked converged while the equation was badly violated. Residuals on fresh points exposed it every time.
3. **Put known structure in the output, not the loss.** Exact boundary conditions and analytical enrichment beat adaptive sampling and longer training.
4. **Warm starts make inverse problems reliable.** The two-stage method went from 3/5 to 10/10 seeds, at 30x the speed.
5. **A PINN is not always the right tool.** It earns its place when data are incomplete and physics must fill the gaps.

**Limitations.** The enrichment terms need analytical insight into each problem. Uncertainty estimates for *c* and *k* are future work. All timings come from one laptop CPU.

## Quick start

```bash
git clone https://github.com/DBS13062006/pinn-mechanics.git
cd pinn-mechanics
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
# Install PyTorch with the command for your system from https://pytorch.org/get-started/locally/
pip install -r requirements.txt                       # also installs this repo in editable mode
python -m pytest                                      # 49 tests, under a minute
python experiments/oscillator_two_stage.py --quick --device cpu   # about a minute
```

<details>
<summary><b>Other setups</b>: conda and Google Colab</summary>

**Conda**

```bash
conda create -n pinn python=3.12 -y && conda activate pinn
# Install PyTorch from https://pytorch.org/get-started/locally/
pip install -r requirements.txt
python scripts/check_env.py      # checks the GPU, autograd and seeding
```

**Google Colab.** Choose a GPU runtime. The repository is private, so store a GitHub fine-grained token with read access as `GH_TOKEN` in Colab's Secrets panel:

```python
from google.colab import userdata
token = userdata.get("GH_TOKEN")
!git clone https://{token}@github.com/DBS13062006/pinn-mechanics.git
%cd pinn-mechanics
!pip install -q -r requirements.txt
```

On Windows, a plain `pip install torch` gives a CPU-only build, so install PyTorch with the CUDA command from pytorch.org.
</details>

## Reproducing every result

Each script writes its figure to `results/figures/`, its tables to `results/tables/` and raw training logs to `results/runs/` (not committed). Run the reproductions first, because each fix reads its baseline from the reproduction's table.

| Study | Command | Runs | Time* |
|:--|:--|:-:|:-:|
| Oscillator: reproduction | `python experiments/oscillator_reproduce.py --device cpu --jobs 5` | 15 | 15 min |
| Oscillator: two-stage fix | `python experiments/oscillator_two_stage.py --device cpu --jobs 5` | 40 | 5 min |
| Oscillator: noisy and sparse data | `python experiments/oscillator_noise.py --device cpu --jobs 4` | 360 | 50 min |
| Consolidation: reproduction | `python experiments/consolidation_reproduce.py --device cpu --jobs 5` | 10 | 29 min |
| Consolidation: enrichment vs RAD | `python experiments/consolidation_enrichment.py --device cpu --jobs 5` | 15 | 13 min |
| Heat: reproduction | `python experiments/heat_reproduce.py --device cpu --jobs 3` | 15 | 82 min |
| Heat: corners | `python experiments/heat_corners.py --device cpu --jobs 3` | 15 | 50 min |
| Summary figure | `python report/make_summary_figure.py` | 0 | 1 s |

<sub>*Wall time on an Intel i5-13450HX laptop CPU.</sub>

**Options.**

- Every script accepts `--device cpu|cuda`, `--jobs N` and `--quick` (a small smoke test).
- The reproduction scripts accept `--bench`, which times CPU against GPU.
- The heat and noise studies accept `--resume <run folder>` to continue after an interruption.
- Runs are seeded and record library versions and hardware, so reruns on the same machine are bit-identical.

## Repository structure

```
pinn-mechanics/
├── pinn/                         PINN engine shared by every experiment
│   ├── model.py                  tanh MLP, inputs scaled to [-1, 1], Xavier init
│   ├── derivatives.py            grad, nth_grad, laplacian via autograd
│   ├── train.py                  Adam → L-BFGS trainer, per-term loss logging, divergence guard
│   ├── sampling.py               residual-based adaptive sampling (RAD)
│   ├── metrics.py                relative L2, max error, parameter error
│   ├── plotting.py               shared, colourblind-checked figure style
│   └── utils.py                  seeding, float64 setup, run folders, environment record
├── problems/                     one module per physics problem
│   ├── oscillator.py             exact solution, implied parameters, noisy data, Cramer-Rao bound
│   ├── consolidation.py          Fourier and erfc exact solutions, enriched output
│   └── heat.py                   series solution, finite differences, exact-BC and corner outputs
├── experiments/                  one script per study (see the table above)
├── results/
│   ├── figures/                  one figure per study, plus summary.png
│   └── tables/                   per-run CSV and summary table per study
├── tests/                        49 unit tests: engine, exact solutions, corner facts
├── scripts/check_env.py          GPU, autograd and seeding check
├── report/
│   ├── PINN_Mechanics_Report.pdf full project report
│   └── make_summary_figure.py    draws results/figures/summary.png from the tables
├── pyproject.toml
└── requirements.txt
```

The engine (`pinn/`) knows nothing about any one problem. Each `problems/` module supplies a residual, boundary conditions, exact references and the output transforms (hard boundary condition, enrichment). Each `experiments/` script combines the two and measures the result.

## Design decisions

| Decision | Why |
|:--|:--|
| **tanh activations** | The loss needs second derivatives; ReLU's are zero almost everywhere. |
| **float64 throughout** | L-BFGS needs the precision near convergence. For networks this small, a step costs the same as float32 (12.2 vs 12.6 ms on an RTX 4050). |
| **Adam, then L-BFGS** | Adam leaves the random start; L-BFGS converges. If L-BFGS diverges, the last good weights are restored. |
| **Every loss term logged** | A failure can be traced to the PDE, boundary, initial condition or data term. |
| **Exact references for every problem** | Each reference is cross-checked by an independent method (a second exact solution or a finite-difference solver), so errors are measured exactly. |
| **Residuals on fresh points** | The training loss alone hid the corner errors in both forward problems. |
| **5–10 seeds per configuration** | One shortcut looked perfect on 5 seeds (5/5) and reached only 8/10 on 10. |

## References

1. Q. Zhang, Y. Chen, Z. Yang. *Data Driven Solutions and Discoveries in Mechanics Using Physics Informed Neural Network.* Stanford CS229 project report. **Base paper.**
2. M. Raissi, P. Perdikaris, G. E. Karniadakis. Physics-informed neural networks. *Journal of Computational Physics* 378, 686–707, 2019.
3. I. E. Lagaris, A. Likas, D. I. Fotiadis. Artificial neural networks for solving ordinary and partial differential equations. *IEEE Transactions on Neural Networks* 9(5), 987–1000, 1998.
4. C. Wu, M. Zhu, Q. Tan, Y. Kartha, L. Lu. A comprehensive study of non-adaptive and residual-based adaptive sampling for physics-informed neural networks. *Computer Methods in Applied Mechanics and Engineering* 403, 115671, 2023.
5. D. P. Kingma, J. Ba. Adam: a method for stochastic optimization. ICLR, 2015.
6. D. C. Liu, J. Nocedal. On the limited memory BFGS method for large scale optimization. *Mathematical Programming* 45, 503–528, 1989.
7. K. Terzaghi. *Theoretical Soil Mechanics.* Wiley, 1943.
8. H. S. Carslaw, J. C. Jaeger. *Conduction of Heat in Solids*, 2nd ed. Oxford, 1959.
9. S. P. Timoshenko, J. N. Goodier. *Theory of Elasticity*, 3rd ed. McGraw-Hill, 1970.
10. J. M. Varah. A spline least squares method for numerical parameter estimation in differential equations. *SIAM Journal on Scientific and Statistical Computing* 3(1), 28–46, 1982.
11. S. M. Kay. *Fundamentals of Statistical Signal Processing: Estimation Theory.* Prentice Hall, 1993.

<div align="center">
<sub>Daruru Bhuvan Sai and Vismay H S · UE24CS352A Machine Learning · PES University · 2026</sub>
</div>
