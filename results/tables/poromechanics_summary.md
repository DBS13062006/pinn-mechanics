# Coupled poromechanics, Eqs. (5.1)-(5.2) (paper section 5, future work)

Median over 5 seeds (worst in brackets). Errors against the exact solution.

## Forward problem: displacement U and pore pressure P

| output | rel. L2 error, U | rel. L2 error, P | max error U | max error P, tau >= 0.01 | highest P (physics: <= gamma = 0.638) | time per run |
|---|---|---|---|---|---|---|
| paper recipe | 1.2e-01 (1.7e-01) | 2.1e-01 (2.9e-01) | 7.0e-01 | 1.9e-01 | 0.674 (0.776) | 114 s |
| corner-enriched | 8.8e-05 (1.2e-04) | 2.2e-04 (3.1e-04) | 1.6e-04 | 2.6e-04 | 0.638 (0.638) | 128 s |

## Inverse problem: consolidation coefficient c from 20 settlement readings

True c = 0.5, start c = 0.1. A run succeeds if c ends within 1%.

| method | succeeded | c error, median (worst) | classical curve fit, same readings | rel. L2 error, P | time per run |
|---|---|---|---|---|---|
| paper recipe, clean | 1/5 | 2.0e-02 (1.3e-01) | 1.1e-16 (1.1e-16) | 3.0e-02 | 170 s |
| enriched, clean | 5/5 | 6.2e-05 (2.8e-04) | 1.1e-16 (1.1e-16) | 2.2e-04 | 162 s |
| enriched, 1% noise | 3/5 | 9.3e-03 (3.7e-02) | 9.2e-03 (3.6e-02) | 5.9e-03 | 187 s |
