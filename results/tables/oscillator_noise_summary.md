# Damped oscillator: finding c and k from noisy and sparse data

Truth c = 0.4, k = 4; every method starts from c = 0.2, k = 2. Median relative error over 10 noise draws, worst in brackets.
Noise: standard deviation as a fraction of the initial displacement.
'ideal' = typical (median) error of the best possible unbiased method at this noise level (Cramer-Rao bound); with no noise it is 0.

## Error in c (damping)

| data | noise | ideal | classical | plain NN | standard PINN | two-stage PINN |
|---|---|---|---|---|---|---|
| full50 | 0% | 0.0e+00 | 3.8e-11 (3.8e-11) | 3.3e-03 (1.2e-02) | 7.3e-03 (1.1e-01) | 1.4e-03 (5.6e-03) |
| full50 | 1% | 4.3e-03 | 5.5e-03 (1.5e-02) | 6.8e-03 (1.4e-02) | 8.6e-03 (3.4e-02) | 4.6e-03 (1.7e-02) |
| full50 | 3% | 1.3e-02 | 1.7e-02 (4.4e-02) | 1.7e-02 (3.6e-01) | 3.0e-02 (1.6e-01) | 2.6e-02 (7.4e-02) |
| full50 | 10% | 4.3e-02 | 5.5e-02 (1.6e-01) | 2.0e-01 (1.0e+00) | 1.7e-01 (2.3e+00) | 2.7e-01 (1.4e+00) |
| full15 | 0% | 0.0e+00 | 4.9e-11 (4.9e-11) | 4.2e-01 (8.1e-01) | 1.5e-03 (2.1e-01) | 2.5e-02 (1.5e+00) |
| full15 | 1% | 8.1e-03 | 1.2e-02 (1.7e-02) | 4.8e-01 (6.9e-01) | 1.5e-02 (3.9e-01) | 2.7e-01 (6.2e+00) |
| full15 | 3% | 2.4e-02 | 3.6e-02 (5.2e-02) | 4.3e-01 (9.0e-01) | 5.7e-02 (2.0e-01) | 1.9e-01 (8.7e+00) |
| full15 | 10% | 8.1e-02 | 1.2e-01 (1.7e-01) | 4.1e-01 (8.3e-01) | 3.2e-01 (1.5e+00) | 4.4e-01 (2.6e+00) |
| early50 | 0% | 0.0e+00 | 3.8e-11 (3.8e-11) | 2.3e-02 (3.5e-02) | 2.3e-03 (4.1e-01) | 2.1e-03 (1.1e-02) |
| early50 | 1% | 4.4e-03 | 4.6e-03 (1.3e-02) | 1.6e-02 (5.1e-02) | 3.3e-02 (9.0e-01) | 6.5e-03 (4.7e-02) |
| early50 | 3% | 1.3e-02 | 1.4e-02 (3.9e-02) | 3.2e-02 (1.4e-01) | 7.0e-02 (7.9e-01) | 1.9e-02 (8.9e-02) |
| early50 | 10% | 4.4e-02 | 4.5e-02 (1.3e-01) | 1.3e-01 (7.3e-01) | 2.6e-01 (3.3e+00) | 2.6e-01 (1.8e+00) |

## Error in k (stiffness)

| data | noise | ideal | classical | plain NN | standard PINN | two-stage PINN |
|---|---|---|---|---|---|---|
| full50 | 0% | 0.0e+00 | 1.0e-11 (1.0e-11) | 8.4e-04 (5.8e-03) | 5.5e-04 (4.1e-03) | 4.4e-04 (1.1e-03) |
| full50 | 1% | 8.0e-04 | 1.1e-03 (2.3e-03) | 1.9e-03 (5.3e-03) | 1.4e-03 (6.8e-03) | 8.5e-04 (2.6e-03) |
| full50 | 3% | 2.4e-03 | 3.5e-03 (7.0e-03) | 6.3e-03 (5.8e+00) | 7.1e-03 (1.5e-02) | 5.0e-03 (5.8e+00) |
| full50 | 10% | 8.0e-03 | 1.2e-02 (2.4e-02) | 6.9e-01 (1.5e+02) | 4.1e-02 (8.0e-02) | 3.7e-01 (6.4e+01) |
| full15 | 0% | 0.0e+00 | 8.2e-12 (8.2e-12) | 1.4e-01 (2.6e+00) | 3.1e-04 (1.6e-02) | 8.4e-03 (2.7e+00) |
| full15 | 1% | 1.5e-03 | 7.7e-04 (3.1e-03) | 3.7e-01 (1.3e+00) | 2.2e-03 (1.5e-02) | 3.9e-02 (8.6e-01) |
| full15 | 3% | 4.5e-03 | 2.3e-03 (9.1e-03) | 1.3e-01 (1.4e+00) | 5.7e-03 (1.7e-02) | 3.3e-02 (1.3e+00) |
| full15 | 10% | 1.5e-02 | 7.9e-03 (2.9e-02) | 1.2e-01 (1.0e+00) | 8.0e-02 (1.8e-01) | 9.7e-02 (9.2e-01) |
| early50 | 0% | 0.0e+00 | 9.1e-12 (9.1e-12) | 1.2e-02 (1.6e-02) | 6.5e-04 (3.5e-01) | 3.7e-04 (3.2e-03) |
| early50 | 1% | 7.9e-04 | 8.9e-04 (1.5e-03) | 1.1e-02 (1.9e-02) | 2.1e-03 (2.8e-01) | 1.1e-03 (8.7e-03) |
| early50 | 3% | 2.4e-03 | 2.7e-03 (4.5e-03) | 1.3e-02 (3.2e-02) | 4.7e-03 (2.5e-01) | 2.8e-03 (2.4e-02) |
| early50 | 10% | 7.9e-03 | 8.9e-03 (1.6e-02) | 1.1e-01 (1.6e+01) | 3.7e-02 (3.9e-01) | 5.5e-02 (1.6e+01) |

## Error of the recovered curve u(t) over [0, 15] (relative L2, median)

| data | noise | classical | plain NN | standard PINN | two-stage PINN |
|---|---|---|---|---|---|
| full50 | 0% | 5.3e-11 | 3.1e-02 | 3.1e-03 | 1.3e-03 |
| full50 | 1% | 7.0e-03 | 3.9e-02 | 9.5e-03 | 7.9e-03 |
| full50 | 3% | 2.1e-02 | 6.7e-02 | 2.7e-02 | 2.6e-02 |
| full50 | 10% | 7.3e-02 | 2.6e-01 | 1.4e-01 | 3.0e-01 |
| full15 | 0% | 5.4e-11 | 3.8e-01 | 2.2e-03 | 2.3e-01 |
| full15 | 1% | 1.1e-02 | 5.6e-01 | 1.6e-02 | 1.6e-01 |
| full15 | 3% | 3.3e-02 | 3.8e-01 | 5.2e-02 | 2.4e-01 |
| full15 | 10% | 1.1e-01 | 4.2e-01 | 3.4e-01 | 2.9e-01 |
| early50 | 0% | 5.3e-11 | 8.3e-01 | 1.3e-01 | 2.2e-01 |
| early50 | 1% | 3.7e-03 | 1.1e+00 | 1.3e-01 | 2.2e-01 |
| early50 | 3% | 1.1e-02 | 1.0e+00 | 1.0e-01 | 3.0e-01 |
| early50 | 10% | 3.6e-02 | 1.6e+00 | 1.8e-01 | 2.7e-01 |

## Cost

| method | time per dataset (mean) | runs within 1% for both c and k, no noise |
|---|---|---|
| classical (ODE least squares) | 0.2 s | 30/30 |
| plain NN + regression | 13.4 s | 10/30 |
| standard PINN (Adam 20k + L-BFGS) | 61.1 s | 26/30 |
| two-stage PINN | 38.0 s | 23/30 |

Two-stage runs whose data-only fit implied impossible values (c < 0 or k <= 0), so stage 3 started from the initial guess instead: 1 of 120.
The plain NN's time is its share of the two-stage run (stage 1 only).
