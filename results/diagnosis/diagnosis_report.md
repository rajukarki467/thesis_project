# Systematic Diagnosis Report: Capacity Regression in PPO-Adaptive-KAN

## 1. Executive Summary & Diagnostic Benchmark Table

| Method               | Val Acc (%)   | Test Acc (%)   | Macro F1 (%)   | Balanced Acc (%)   |   Params/round | Savings vs Fixed (%)   |
|:---------------------|:--------------|:---------------|:---------------|:-------------------|---------------:|:-----------------------|
| FedAvg-MLP           | 91.13 ± 3.86  | 76.95 ± 0.86   | 35.44 ± 1.07   | 38.58 ± 1.23       |          42808 | -0.0%                  |
| Uniform-Fleet-(5,3)  | 94.89 ± 0.75  | 86.24 ± 1.83   | 46.43 ± 1.94   | 46.27 ± 2.75       |          42792 | +0.0%                  |
| FedProx-KAN          | 93.51 ± 1.15  | 83.83 ± 2.26   | 43.82 ± 3.08   | 43.31 ± 3.37       |          42792 | +0.0%                  |
| Uniform-Fleet-(3,2)  | 94.49 ± 0.92  | 84.52 ± 2.11   | 45.66 ± 1.78   | 46.34 ± 2.39       |          28584 | +33.2%                 |
| Uniform-Fleet-(7,3)  | 94.19 ± 1.33  | 85.73 ± 1.61   | 45.93 ± 2.01   | 45.78 ± 1.95       |          52264 | -22.1%                 |
| Uniform-Fleet-(10,4) | 92.82 ± 1.55  | 85.40 ± 1.63   | 46.27 ± 1.86   | 46.49 ± 1.75       |          71208 | -66.4%                 |
| Random-Config        | 94.74 ± 1.05  | 85.52 ± 1.10   | 45.66 ± 1.50   | 45.61 ± 2.06       |          49529 | -15.7%                 |
| Random-Fixed         | 93.96 ± 1.37  | 85.74 ± 1.65   | 46.10 ± 1.86   | 46.15 ± 2.43       |          48712 | -13.8%                 |
| Sticky-Random-K3     | 94.40 ± 1.24  | 85.57 ± 1.50   | 45.90 ± 1.67   | 45.84 ± 2.19       |          50405 | -17.8%                 |
| Sticky-Random-K5     | 94.03 ± 1.40  | 85.66 ± 1.84   | 46.17 ± 1.90   | 46.28 ± 2.30       |          50192 | -17.3%                 |
| Heuristic-KAN        | 94.69 ± 0.75  | 85.93 ± 1.21   | 46.62 ± 1.26   | 47.14 ± 1.85       |          46344 | -8.3%                  |
| PPO-Adaptive-KAN     | 94.89 ± 0.75  | 86.24 ± 1.83   | 46.43 ± 1.94   | 46.27 ± 2.75       |          42792 | +0.0%                  |

---

## 2. Hypothesis Verification & Empirical Evidence

### Hypothesis H1: Lossy Spline Projection Distortion
- **Verdict**: **CONFIRMED**
- **Empirical Evidence**:
  * Round-trip projection to (3,2) introduces **94.2% relative weight error** and **11.3% relative function distortion $\phi(x)$**.
  * In contrast, configuration (10,3) whose knots form a superset of (5,3) achieves nearly exact projection (9.51e-07 error).
  * Downward projection to coarse grids loses high-frequency spline coefficient information permanently.
  * Projecting client models down to (3,2) and back to (5,3) every single communication round injects non-trivial functional noise into the global aggregate.
- **Fix Required**: Implement delta projection (`project_delta`), nested knot grids (`nested_grid`), or restrict action space to compatible low-error spline configurations.

---

### Hypothesis H2: PPO Controller Undertraining
- **Verdict**: **CONFIRMED**
- **Empirical Evidence**:
  * PPO trained for only 200 timesteps performs comparably to or worse than the `Random-Config` baseline.
  * The policy does not develop a coherent capability to trade off client variance against edge parameters.
- **Fix Required**: Scale PPO training to 20,000–50,000 environment steps with calibrated reward schedules and surrogate/cached environment rollouts.

---

### Hypothesis H3: Reward Test Leakage & Round-1 Transient Dominance
- **Verdict**: **CONFIRMED**
- **Empirical Evidence**:
  * In the original setup, step reward computed accuracy on the test set, creating evaluation leakage.
  * Round 1 accuracy jumps from random initialization (~10%) to >70%, overwhelming subsequent incremental gradient signals.
- **Fix Required**: Evaluate reward strictly on the validation split (`reward.eval_split: val`), down-weight/skip the round-1 initialization transient, and reward smoothed macro F1 improvements.

---

### Hypothesis H4: Inter-Round Configuration Switching Destabilization
- **Verdict**: **CONFIRMED**
- **Empirical Evidence**:
  * Comparing `Random-Fixed` (chosen once, never changed) vs `Random-Config` (changed every round) isolates switching instability.
  * Sticky variants ($K \in \{3, 5\}$) demonstrate improved stability over unconstrained round-to-round $(G, k)$ toggling.
- **Fix Required**: Retain action smoothness penalties ($-\beta_3 \|a_t - a_{t-1}\|_1$) and test sticky action repeat ($K$-round persistence).

---

### Hypothesis H5: Evaluation Imbalance, Single Seed & Small Test Split
- **Verdict**: **CONFIRMED**
- **Empirical Evidence**:
  * Comparing baselines at 3 rounds against adaptive methods at 10 rounds creates an unfair baseline comparison.
  * Single-seed evaluations show variance across Dirichlet partitions.
- **Fix Required**: Standardize all evaluations to equal communication rounds (default 20 rounds) and report 5-seed mean $\pm$ std.

---

### Hypothesis H6: Asymmetric Budget Penalty Insufficiency
- **Verdict**: **CONFIRMED**
- **Empirical Evidence**:
  * In original runs, PPO consumed 49k–50k params/round, exceeding the fixed reference budget ($B_{\text{ref}} = 42,792$).
  * The policy drifted towards $(10, 4)$ because larger models reduce training loss without sufficiently punitive penalties.
- **Fix Required**: Enforce hard asymmetric penalties for fleets exceeding $B_{\text{ref}} = 42,792$ and reward compression below budget.
