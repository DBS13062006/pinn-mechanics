# 1D consolidation: fixing the error near the corner (paper section 4.1)

Median over 5 seeds (worst in brackets). Paper and baseline rows are the reproduction's runs.

| method | rel. L2 error | max error, t >= 0.01 | max error, t >= 0.05 | highest predicted pressure (must be <= 1) | collocation points in t < 0.01 | PDE loss on fresh points | time per run |
|---|---|---|---|---|---|---|---|
| paper (Adam 50k) | 1.4e-02 (4.3e-02) | 6.1e-02 (1.7e-01) | 6.5e-03 (4.9e-02) | 1.108 (1.261) | 11 of 500 | 4.9e+01 | 1143 s |
| baseline (Adam 10k) | 1.7e-02 (3.2e-02) | 9.7e-02 (2.4e-01) | 7.1e-03 (1.6e-02) | 1.191 (1.467) | 11 of 500 | 1.9e+02 | 433 s |
| adaptive sampling (RAD) | 1.3e-02 (3.8e-02) | 3.3e-02 (2.6e-01) | 5.4e-03 (3.1e-02) | 1.076 (1.154) | 37 of 500 | 3.3e+01 | 395 s |
| corner-enriched output | 6.4e-05 (9.6e-05) | 2.9e-04 (5.1e-04) | 2.9e-04 (5.1e-04) | 1.000 (1.000) | 11 of 500 | 8.4e-07 | 168 s |
| enriched + RAD | 5.4e-05 (8.0e-05) | 2.1e-04 (3.5e-04) | 1.5e-04 (3.5e-04) | 1.000 (1.000) | 16 of 500 | 3.6e-07 | 160 s |
