# Step 5: steady heat conduction (paper section 4.2)

Errors against the exact series solution. Median over 5 seeds (worst in brackets).
The paper's Fig. 3 shows a maximum error of about 5.2e-03, on the edges.

| configuration | rel. L2 error | max error | max error on the edges | max error, interior (>= 0.2 from an edge) | share of squared error within 0.05 of an edge (10% of the area) | lowest temperature (physics: >= 0) | PDE loss: training / fresh points | time per run |
|---|---|---|---|---|---|---|---|---|
| paper: soft BC (Adam 50k) | 1.3e-03 (1.6e-03) | 3.8e-03 (5.1e-03) | 3.8e-03 (5.1e-03) | 4.1e-04 (5.3e-04) | 60% | -3.8e-03 (-5.1e-03) | 3.0e-07 / 6.7e-07 | 1983 s |
| soft BC (Adam 5k) | 1.5e-03 (2.5e-03) | 4.8e-03 (6.5e-03) | 4.8e-03 (6.5e-03) | 5.5e-04 (7.3e-04) | 58% | -4.8e-03 (-6.5e-03) | 5.7e-07 / 8.1e-07 | 365 s |
| hard BC (Adam 5k) | 4.2e-05 (7.3e-05) | 7.7e-05 (1.5e-04) | 2.0e-09 (2.0e-09) | 2.0e-05 (3.8e-05) | 13% | 0.0e+00 (0.0e+00) | 1.5e-06 / 1.4e-04 | 428 s |

Hard BC, PDE loss on fresh points near the four corners (within 0.1 of both edges) vs elsewhere: 1.4e-02 vs 1.0e-05 (median).
