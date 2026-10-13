# 1D consolidation: reproduction (paper section 4.1)

Errors against the exact solution. Median over seeds (worst in brackets).
The paper reports a 'perfect match' with max absolute error about 0.045 (its Fig. 2).

| configuration | runs | rel. L2 error | max error, t >= 0.01 | max error, t >= 0.05 | share of squared error in t < 0.01 (2% of domain) | collocation points in t < 0.01 | PDE loss: training / fresh points | time per run |
|---|---|---|---|---|---|---|---|---|
| paper (Adam 50k + L-BFGS) | 5 | 1.4e-02 (4.3e-02) | 0.061 (0.166) | 0.006 (0.049) | 59% | 11 of 500 | 5.5e-06 / 4.9e+01 | 1143 s |
| Adam 10k + L-BFGS | 5 | 1.7e-02 (3.2e-02) | 0.097 (0.241) | 0.007 (0.016) | 76% | 11 of 500 | 9.6e-05 / 1.9e+02 | 433 s |
