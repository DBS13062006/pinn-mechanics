# PINNs for Mechanics: Reproduction and Improvements

UE24CS352A Machine Learning mini-project.

**Base paper:** Q. Zhang, Y. Chen, Z. Yang, *Data Driven Solutions and Discoveries in Mechanics Using Physics Informed Neural Network* (Stanford CS229 report).

**Goal:** reproduce the paper's three experiments, then go beyond them:

1. **Fix the boundary error** in the heat-conduction problem by enforcing the boundary condition exactly (distance-function trial solution), which the paper suggests but never implements; then explain, and fix, the error that remains at the corners.
2. **Fix the error hotspot near the origin** in the consolidation problem: a corner-enriched output, compared against residual-based adaptive sampling (RAD), the paper's own "adaptive re-meshing" future-work item.
3. **Stress-test parameter discovery** (finding damping `c` and stiffness `k`) on noisy and sparse data, with uncertainty estimates and plain-NN / classical baselines. The paper only tested clean, dense data.

## Status

| Step | Description | Status |
|---|---|---|
| 1 | Environment and repo skeleton | done |
| 2 | PINN engine (network, autograd helpers, Adam -> L-BFGS trainer) | done |
| 3 | Reproduce 4.3: damped oscillator, find c and k (5 seeds) + diagnose slow c | done |
| 3b | Improvement: two-stage warm start makes discovery of c reliable and fast | done |
| 4 | Reproduce 4.1: Terzaghi consolidation (5 seeds) + locate the error | done |
| 4b | Improvement: corner-enriched output vs adaptive sampling (RAD) | done |
| 5 | Reproduce 4.2: steady heat conduction (5 seeds) + the paper's suggested hard-BC fix | done |
| 5b | Improvement: why every heat PINN fails at the corners, and a corner-enriched output | done |
| 6 | Improvement 1: exact boundary conditions | done in 5 |
| 7 | Improvement 2: residual-adaptive sampling | done in 4b |
| 8a | Improvement 3: finding c and k from noisy / sparse data, vs classical and plain-NN baselines | done |
| 8b | Improvement 3: uncertainty (error bars that can be trusted) | |
| 9 | Extras (if time) | |
| 10 | Results, write-up, slides, demo | |

## Setup

### Windows laptop (Anaconda Prompt)

```bat
conda create -n pinn python=3.12 -y
conda activate pinn

:: 1. PyTorch with CUDA. Copy the command from https://pytorch.org/get-started/locally/
::    choosing: Stable / Windows / Pip / Python / the newest CUDA option.
::    Do NOT use a plain "pip install torch" on Windows: that is CPU-only.

:: 2. Everything else, plus this repo in editable mode
pip install -r requirements.txt

:: 3. Check that everything works
python scripts/check_env.py
```

If the check reports no GPU, run `nvidia-smi`. The "CUDA Version" it shows (top right) must be at least the CUDA version of the PyTorch build you installed. If it is lower, update the NVIDIA driver or pick an older CUDA option on the PyTorch page.

### Google Colab

Use Runtime -> Change runtime type -> a GPU. Store a GitHub fine-grained token with read access to this repo in Colab's Secrets panel as `GH_TOKEN`, then:

```python
from google.colab import userdata
token = userdata.get("GH_TOKEN")
!git clone https://{token}@github.com/<owner>/pinn-mechanics.git
%cd pinn-mechanics
!pip install -q -r requirements.txt
!python scripts/check_env.py
```

## Running

```bat
python -m pytest                              :: engine unit tests, ~15 s
python experiments/step2_engine_check.py      :: engine check on the heat equation, a few min on GPU
python experiments/step2_engine_check.py --quick   :: smaller version for a CPU

python experiments/step3_oscillator.py --bench      :: time CPU vs GPU for this tiny problem
python experiments/step3_oscillator.py --device cuda             :: 3 configs x 5 seeds
python experiments/step3_oscillator.py --device cpu --jobs 5     :: same, 5 runs in parallel on CPU
python experiments/step3b_oscillator_fix.py --device cpu --jobs 5 :: the fix + ablations, 10 seeds
python experiments/step4_consolidation.py --bench                :: then pick cpu --jobs 5 or cuda
python experiments/step4b_consolidation_fix.py --device cpu --jobs 5 :: the two fixes, 5 seeds
python experiments/step5_heat.py --bench                         :: then pick cpu --jobs 5 or cuda
python experiments/step5_heat.py --device cpu --jobs 3           :: 3 configs x 5 seeds (~2 h on a laptop)
python experiments/step5b_heat_corners.py --device cpu --jobs 3  :: corner diagnostics + fix, 5 seeds (~1 h)
python experiments/step8_oscillator_noise.py --device cpu --jobs 4 :: 4 methods x 3 data settings x 4 noise levels x 10 draws (~1.5 h)
```

Results so far:
[Step 3 summary](results/tables/step3_oscillator_summary.md) ·
[Step 3b summary](results/tables/step3b_oscillator_fix_summary.md) ·
[Step 4 summary](results/tables/step4_consolidation_summary.md) ·
[Step 4b summary](results/tables/step4b_consolidation_fix_summary.md) ·
[Step 5 summary](results/tables/step5_heat_summary.md) ·
[Step 5b summary](results/tables/step5b_heat_corners_summary.md) ·
[Step 8a summary](results/tables/step8_oscillator_noise_summary.md)

## Findings so far

**Oscillator (paper 4.3): why c is found so slowly, and a fix.**
c and k always match the values implied by the curve the network currently draws
(`problems/oscillator.py: implied_parameters`). The network learns the signal left to right, and
the late, small oscillations, which encode the damping c, carry only 1% of the data's energy, while
a flat tail satisfies the ODE exactly. Early on, c looks like ~1 (heavy damping); with the paper's
settings, 2 of 5 seeds still miss c by more than 1% after 100k iterations.
**Fix (two-stage warm start):** fit the data without physics, read c and k off the fitted curve,
then train the full PINN from there (optionally with c and k frozen for a first pass).
Over 10 seeds it succeeded 10/10, versus 3/5 for the paper's settings, about 30x faster.
Ablations: the key ingredient is reading c and k off the data-only fit (without it: 7/10);
freezing c and k first is at most a small safety margin (10/10 without it on one machine,
9/10 on another). A shortcut of Adam 20k + L-BFGS looked perfect on 5 seeds (5/5) but
reached only 8/10 on 10, a reminder of why every result here uses several seeds.

**Consolidation (paper 4.1): the training loss hides where the solution is wrong.**
The exact solution jumps at the corner x = 0, t = 0 (pressure 1 meets the drained top), so the
PINN's error is concentrated at small times. The first 2% of the time window holds about 11 of the
500 random collocation points but most of the squared error, and there the PDE residual on fresh
points is orders of magnitude larger than at the training points. A max error on a uniform grid is
not meaningful here (it only measures how close the grid gets to the jump), so errors are reported
as a function of time. See Step 4 for the numbers.
**Fix:** write the output as p = erf(x / 2 sqrt t) + x t N(x, t). The erf term is the exact solution
for a layer with no base and carries the jump; the network learns only the smooth effect of the base.
This is the paper's own "multiply by x" trick taken one step further. Residual-based adaptive
sampling (RAD) moves points toward the corner but cannot make a smooth network represent an
infinitely steep front, so it helps far less. See Step 4b for the numbers.

**Heat conduction (paper 4.2): the paper's own suggested fix, tested, and what limits every PINN here.**
The paper states that no analytical solution exists; the classic series solution does
(`problems/heat.py`), and our own finite-difference solver converges to it at second order. With the
boundary condition only as a loss penalty (the paper's setup), the largest error sits on the edges and
the plate dips slightly below zero there. Multiplying the network by (1 - x^2)(1 - y^2), as the paper
suggests but never tries, makes the edges exact and cuts the error about 30-fold. See Step 5.
**The corners (Step 5b).** Along the top edge T = 0, so T_xx = 0 there; along the right edge T_yy = 0,
so the PDE forces T_xx = -1. At the corner T_xx would have to be both. The exact solution escapes
because its T_xx depends on the direction you approach from (-2 theta / pi), which no smooth network
can copy, so every PINN has to break something at the four corners. The paper's soft-BC network breaks
the boundary condition: its largest error is at the corners, not along the edges in general. The
hard-BC network breaks the PDE: its residual is exactly 1 at each corner whatever the weights, which
is why its PDE loss on fresh points is ~100x the training loss. All three facts are checked in
`tests/test_heat.py` without any training.
**Fix:** add the exact local solution at each corner (worked out by hand from the corner alone, not
from the series) to the hard-BC output, so the network only learns a smooth remainder: the same idea
as the erf term in consolidation. It removes the corner hotspot, lowers the max error and makes the
training loss trustworthy again; the average error changes little, and training takes about twice as
long. See Step 5b for the numbers.

Every experiment writes its raw output (loss history, model weights, metrics.json) to a new
`results/runs/<timestamp>_<name>/` folder, and its report figure to `results/figures/`.

## Repository layout

```
pinn/          reusable PINN engine
  model.py       fully connected tanh network with input scaling to [-1, 1]
  derivatives.py grad / nth_grad / laplacian via autograd (derivatives w.r.t. inputs)
  train.py       Trainer: Adam then L-BFGS, per-term loss logging, eval hooks
  metrics.py     relative L2 error, max error, parameter error
  plotting.py    shared figure style (colorblind-checked palette)
  sampling.py    residual-based adaptive sampling (RAD) as a Trainer hook
  utils.py       seeding, device and float64 setup, timing, run folders
problems/      one module per physics problem: residual, boundary conditions, reference solution
  oscillator.py  paper 4.3: exact solution, ODE residual, implied-parameter diagnostic,
                 noisy / sparse data, classical ODE least-squares fit, Cramer-Rao bound
  consolidation.py paper 4.1: two independent exact solutions (Fourier, erfc images), residuals
  heat.py        paper 4.2: exact series solution (the paper says none exists) + finite-difference check,
                 hard-BC and corner-enriched outputs
experiments/   runnable scripts that produce every number and figure in the report
tests/         fast unit tests for the engine (pytest)
scripts/       helper scripts (environment check)
notebooks/     demo notebook for the live review
results/       figures/ and tables/ are committed; runs/ (raw logs) is git-ignored
report/        write-up (PDF) and slides
```

## Design decisions

- **float64 everywhere.** On the RTX 4050 one training step took 12.6 ms in float32 and
  12.2 ms in float64 (`scripts/check_env.py`): these networks are too small for the slower
  float64 arithmetic to matter. L-BFGS needs the extra precision near convergence.
- **Adam, then L-BFGS**, as in the paper. Adam escapes the random start; L-BFGS finishes.
  If L-BFGS diverges, the trainer restores the last good parameters instead of losing the run.
- **Every loss term is logged separately**, so a failure can be traced to the PDE, the
  initial condition, the boundary condition or the data term.

## Reproducibility

Every experiment seeds all random generators (`pinn.utils.set_seed`) and saves library versions and hardware (`pinn.utils.env_info`) next to its results.

## Team

| Member | Main responsibility |
|---|---|
| | Forward problems: consolidation, heat, exact BCs, adaptive sampling |
| | Inverse problem: oscillator, robustness study, uncertainty |
