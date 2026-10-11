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
| 2 | PINN engine (network, autograd helpers, Adam -> L-BFGS trainer) | |
| 3 | Reproduce 4.3: damped oscillator, find c and k | |
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

## Repository layout

```
pinn/          reusable PINN engine (utils now; model, derivatives, training, sampling later)
problems/      one module per physics problem: residual, boundary conditions, reference solution
experiments/   runnable scripts that produce every number and figure in the report
scripts/       helper scripts (environment check)
notebooks/     demo notebook for the live review
results/       figures/ and tables/ are committed; runs/ (raw logs) is git-ignored
report/        write-up (PDF) and slides
```

## Reproducibility

Every experiment seeds all random generators (`pinn.utils.set_seed`) and saves library versions and hardware (`pinn.utils.env_info`) next to its results.

## Team

| Member | Main responsibility |
|---|---|
| | Forward problems: consolidation, heat, exact BCs, adaptive sampling |
| | Inverse problem: oscillator, robustness study, uncertainty |
