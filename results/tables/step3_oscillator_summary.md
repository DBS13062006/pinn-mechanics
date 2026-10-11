# Step 3: damped oscillator (paper section 4.3)

Truth: c = 0.4, k = 4. Start: c = 0.2, k = 2. A run succeeds if both end within 1%.
Paper's single run: c = 0.40125 (0.31% off), k = 3.9933 (0.17% off).

| configuration | succeeded | c error (median / worst) | k error (median / worst) | iterations until both stay within 1% (median) | time per run |
|---|---|---|---|---|---|
| paper (Adam 100k) | 3/5 | 9.1e-03 / 4.0e-02 | 1.0e-03 / 4.1e-03 | 71,000 | 699 s |
| Adam 5k + L-BFGS | 2/5 | 7.8e-02 / 4.4e-01 | 1.5e-03 / 1.5e-02 | 7,649 | 18 s |
| Adam 20k + L-BFGS | 5/5 | 2.2e-03 / 8.9e-03 | 4.3e-04 / 6.6e-04 | 21,900 | 153 s |
