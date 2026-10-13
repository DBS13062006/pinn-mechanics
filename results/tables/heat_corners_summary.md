# Steady heat conduction: the corners

Errors against the exact series solution. Median over 5 seeds (worst in brackets). Adam 5k then L-BFGS for every configuration.

| configuration | rel. L2 error | max error | max error: distance to the nearest corner | share of squared error within 0.15 of a corner (1.8% of the area) | PDE loss on fresh points near the corners | PDE loss: fresh / training points | lowest temperature (physics: >= 0) | time per run |
|---|---|---|---|---|---|---|---|---|
| soft BC | 1.5e-03 (2.5e-03) | 4.8e-03 (6.5e-03) | 0.00 (at a corner point in 5/5 seeds) | 26% | 9.4e-06 | 1.5x | -4.8e-03 (-6.5e-03) | 356 s |
| hard BC | 4.2e-05 (7.3e-05) | 7.7e-05 (1.5e-04) | 0.06 (at a corner point in 0/5 seeds) | 18% | 1.6e-02 | 97.7x | 0.0e+00 (0.0e+00) | 489 s |
| hard BC + corner terms | 3.9e-05 (1.1e-04) | 2.7e-05 (4.8e-05) | 0.87 (at a corner point in 0/5 seeds) | 2% | 6.1e-05 | 1.8x | 0.0e+00 (0.0e+00) | 945 s |

"Near the corners" for the PDE loss means within 0.1 of both edges, as in the reproduction.

Facts about the problem itself (no training involved; checked in tests/test_heat.py):
- Exact solution: approaching a corner at angle theta from the horizontal edge, T_xx -> -2 theta / pi (0 along that edge, -1 along the other), at every distance. A smooth network has one value there.
- Any hard-BC network T = (1 - x^2)(1 - y^2) N has PDE residual exactly 1 at the four corners.
- The function a hard-BC network must learn, N = T / ((1 - x^2)(1 - y^2)), grows by ln(10) / (2 pi) = 0.366 for every factor of 10 closer to a corner.

Reproducibility: soft5k and hard5k reran the reproduction's training; over 10 runs their rel. L2 and max errors differ from the reproduction's table by at most 0.0e+00 (relative). Exact zeros mean the same machine produced bit-identical training.
