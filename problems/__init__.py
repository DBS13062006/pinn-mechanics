"""
problems: one module per physics problem from the paper.

Each module will define the domain, the PDE/ODE residual, the boundary and
initial conditions, and an independent reference solution to measure error:
  consolidation.py  1D Terzaghi consolidation (paper 4.1)  - analytical series
  heat.py           2D steady heat / Poisson (paper 4.2)   - our own fine-grid reference
  oscillator.py     damped SDOF oscillator, find c and k (paper 4.3) - closed form
"""
