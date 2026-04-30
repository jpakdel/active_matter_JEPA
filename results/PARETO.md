# Pareto Frontier — Best-by-Metric

Top-5 rankings across all runs for each of the four reported probe metrics. The bold cell is the metric value being ranked; the trailing column shows that run's other three metrics for context.

---

## alpha (active dipole strength)

### Top 5 by alpha linear test MSE

| rank | run_id | routing | backbone | target | loss | alpha linear test MSE | (other 3 metrics) |
|---|---|---|---|---|---|---|---|
| 1 | baseline_cnn_ema_vicreg_no_cov_20260428_190830 | baseline | cnn | ema | vicreg_no_cov | **0.0195** | zeta lin=0.3150 / alpha kNN=0.0131 / zeta kNN=0.7559 |
| 2 | baseline_vit_ema_vicreg_20260428_102316 | baseline | vit | ema | vicreg | **0.0316** | zeta lin=0.1332 / alpha kNN=0.0864 / zeta kNN=0.1443 |
| 3 | exp_a_vit_ema_vicreg_20260428_111313 | exp_a | vit | ema | vicreg | **0.0676** | zeta lin=0.1784 / alpha kNN=0.1718 / zeta kNN=0.4109 |
| 4 | djepa_exp_a_vicreg_v0_20260424_005651 | exp_a | vit | shared | vicreg | **0.0869** | zeta lin=0.2455 / alpha kNN=0.2037 / zeta kNN=0.6745 |
| 5 | baseline_cnn_ema_vicreg_20260428_225444 | baseline | cnn | ema | vicreg | **0.1403** | zeta lin=0.3224 / alpha kNN=0.3478 / zeta kNN=0.3870 |

### Top 5 by alpha kNN test MSE

| rank | run_id | routing | backbone | target | loss | alpha kNN test MSE | (other 3 metrics) |
|---|---|---|---|---|---|---|---|
| 1 | baseline_cnn_ema_vicreg_no_cov_20260428_190830 | baseline | cnn | ema | vicreg_no_cov | **0.0131** | alpha lin=0.0195 / zeta lin=0.3150 / zeta kNN=0.7559 |
| 2 | baseline_vit_ema_vicreg_20260428_102316 | baseline | vit | ema | vicreg | **0.0864** | alpha lin=0.0316 / zeta lin=0.1332 / zeta kNN=0.1443 |
| 3 | exp_a_vit_ema_vicreg_20260428_111313 | exp_a | vit | ema | vicreg | **0.1718** | alpha lin=0.0676 / zeta lin=0.1784 / zeta kNN=0.4109 |
| 4 | exp_a_cnn_ema_vicreg_20260428_234329 | exp_a | cnn | ema | vicreg | **0.2027** | alpha lin=0.1601 / zeta lin=0.2832 / zeta kNN=0.3355 |
| 5 | djepa_exp_a_vicreg_v0_20260424_005651 | exp_a | vit | shared | vicreg | **0.2037** | alpha lin=0.0869 / zeta lin=0.2455 / zeta kNN=0.6745 |

---

## zeta (steric alignment)

### Top 5 by zeta linear test MSE

| rank | run_id | routing | backbone | target | loss | zeta linear test MSE | (other 3 metrics) |
|---|---|---|---|---|---|---|---|
| 1 | baseline_vit_ema_vicreg_20260428_102316 | baseline | vit | ema | vicreg | **0.1332** | alpha lin=0.0316 / alpha kNN=0.0864 / zeta kNN=0.1443 |
| 2 | exp_a_vit_ema_vicreg_20260428_111313 | exp_a | vit | ema | vicreg | **0.1784** | alpha lin=0.0676 / alpha kNN=0.1718 / zeta kNN=0.4109 |
| 3 | djepa_exp_b_vicreg_lam001_v0_20260424_023349 | exp_b | vit | shared | vicreg_lam001 | **0.2004** | alpha lin=0.6592 / alpha kNN=0.6235 / zeta kNN=0.1894 |
| 4 | djepa_exp_a_vicreg_v0_20260424_005651 | exp_a | vit | shared | vicreg | **0.2455** | alpha lin=0.0869 / alpha kNN=0.2037 / zeta kNN=0.6745 |
| 5 | exp_b_cnn_ema_vicreg_20260429_003133 | exp_b | cnn | ema | vicreg | **0.2806** | alpha lin=0.2353 / alpha kNN=0.4101 / zeta kNN=0.4874 |

### Top 5 by zeta kNN test MSE

| rank | run_id | routing | backbone | target | loss | zeta kNN test MSE | (other 3 metrics) |
|---|---|---|---|---|---|---|---|
| 1 | baseline_vit_ema_vicreg_20260428_102316 | baseline | vit | ema | vicreg | **0.1443** | alpha lin=0.0316 / zeta lin=0.1332 / alpha kNN=0.0864 |
| 2 | djepa_exp_b_vicreg_lam001_v0_20260424_023349 | exp_b | vit | shared | vicreg_lam001 | **0.1894** | alpha lin=0.6592 / zeta lin=0.2004 / alpha kNN=0.6235 |
| 3 | exp_a_cnn_ema_vicreg_20260428_234329 | exp_a | cnn | ema | vicreg | **0.3355** | alpha lin=0.1601 / zeta lin=0.2832 / alpha kNN=0.2027 |
| 4 | baseline_v0_20260421_152635 | baseline | vit | shared | sigreg | **0.3829** | alpha lin=0.1747 / zeta lin=0.4420 / alpha kNN=0.2314 |
| 5 | baseline_cnn_ema_vicreg_20260428_225444 | baseline | cnn | ema | vicreg | **0.3870** | alpha lin=0.1403 / zeta lin=0.3224 / alpha kNN=0.3478 |

---

## Pareto-optimal on (alpha kNN, zeta kNN)

A run is Pareto-optimal if no other run has both lower alpha kNN AND lower zeta kNN. These are the configurations that are not strictly dominated by any other in the joint (alpha, zeta) space.

| run_id | routing | backbone | target | loss | alpha kNN | zeta kNN |
|---|---|---|---|---|---|---|
| baseline_cnn_ema_vicreg_no_cov_20260428_190830 | baseline | cnn | ema | vicreg_no_cov | **0.0131** | **0.7559** |
| baseline_vit_ema_vicreg_20260428_102316 | baseline | vit | ema | vicreg | **0.0864** | **0.1443** |

*(2 of 26 runs lie on the (alpha, zeta) kNN Pareto frontier.)*

