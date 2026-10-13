# Damped oscillator: fixing slow discovery of c (paper section 4.3)

Truth: c = 0.4, k = 4. Start: c = 0.2, k = 2. A run succeeds if both end within 1%.
The paper row is the reproduction's 5 runs with the paper's exact settings.

| method | succeeded | c error (median / worst) | k error (median / worst) | time per run (mean) |
|---|---|---|---|---|
| paper (Adam 100k) | 3/5 | 9.1e-03 / 4.0e-02 | 1.0e-03 / 4.1e-03 | 699 s |
| Adam 20k + L-BFGS | 8/10 | 7.3e-03 / 1.1e-01 | 5.5e-04 / 4.1e-03 | 59 s |
| two-stage without stage 2 | 7/10 | 7.3e-04 / 5.2e-02 | 4.6e-04 / 1.0e-02 | 22 s |
| two-stage without stage 3a | 10/10 | 1.9e-03 / 7.2e-03 | 1.8e-04 / 1.0e-03 | 23 s |
| two-stage (full fix) | 10/10 | 1.4e-03 / 5.6e-03 | 4.4e-04 / 1.1e-03 | 23 s |

Two-stage fix, after stage 2 only (c and k read off the data-only fit, before any physics):
c error median 3.3e-03, worst 1.2e-02; k error median 8.4e-04, worst 5.8e-03.
