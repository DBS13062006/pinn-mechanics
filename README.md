# PINNs for Mechanics: Reproduction and Improvements

UE24CS352A Machine Learning mini-project.

**Base paper:** Q. Zhang, Y. Chen, Z. Yang, *Data Driven Solutions and Discoveries in Mechanics Using Physics Informed Neural Network* (Stanford CS229 report).

**Goal:** reproduce the paper's three experiments, then go beyond them:

1. **Fix the boundary error** in the heat-conduction problem by enforcing the boundary condition exactly (distance-function trial solution), which the paper suggests but never implements.
2. **Fix the error hotspot near the origin** in the consolidation problem with residual-based adaptive sampling (RAD), the paper's own "adaptive re-meshing" future-work item.
3. **Stress-test parameter discovery** (finding damping `c` and stiffness `k`) on noisy and sparse data, with uncertainty estimates and plain-NN / classical baselines. The paper only tested clean, dense data.

## Status

| Step | Description | Status |
|---|---|---|
| 1 | Environment and repo skeleton | done |
| 2 | PINN engine (network, autograd helpers, Adam -> L-BFGS trainer) | done |
| 3 | Reproduce 4.3: damped oscillator, find c and k (5 seeds) + diagnose slow c | done |
| 4 | Reproduce 4.1: Terzaghi consolidation | |
| 5 | Reproduce 4.2: steady heat conduction | |
| 6 | Improvement 1: exact boundary conditions | |
| 7 | Improvement 2: residual-adaptive sampling | |
| 8 | Improvement 3: noise / sparsity robustness + uncertainty | |
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
```

Results so far: [Step 3 summary table](results/tables/step3_oscillator_summary.md).

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
  utils.py       seeding, device and float64 setup, timing, run folders
problems/      one module per physics problem: residual, boundary conditions, reference solution
  oscillator.py  paper 4.3: exact solution, ODE residual, implied-parameter diagnostic
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
