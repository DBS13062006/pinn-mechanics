"""
problems: one module per physics problem from the paper.

Each module defines the domain, the PDE/ODE residual, the boundary and
initial conditions, and an independent reference solution to measure error:
  consolidation.py  1D Terzaghi consolidation (paper 4.1)  - exact series and erfc solutions
  heat.py           2D steady heat / Poisson (paper 4.2)   - exact series + finite-difference check
  oscillator.py     damped SDOF oscillator, find c and k (paper 4.3) - closed form
  poromechanics.py  coupled Biot equations, find U, P and c (paper 5) - images + Fourier series
"""
