# Pareto Frontier — Best-by-Metric

Top-5 rankings across all runs for each of the four reported probe metrics. The bold cell is the metric value being ranked; the trailing column shows that run's other three metrics for context.

---

## alpha (active dipole strength)

### Top 5 by alpha linear test MSE

| rank | run_id | routing | backbone | target | loss | alpha linear test MSE | (other 3 metrics) |
|---|---|---|---|---|---|---|---|
| 1 | baseline_vit_ema_vicreg_lam001_20260430_170646 | baseline | vit | ema | vicreg_lam001 | **0.0063** | zeta lin=0.0680 / alpha kNN=0.0147 / zeta kNN=0.1017 |
| 2 | baseline_cnn_ema_vicreg_no_cov_20260428_190830 | baseline | cnn | ema | vicreg_no_cov | **0.0195** | zeta lin=0.3150 / alpha kNN=0.0131 / zeta kNN=0.7559 |
| 3 | baseline_vit_ema_vicreg_20260428_102316 | baseline | vit | ema | vicreg | **0.0316** | zeta lin=0.1332 / alpha kNN=0.0864 / zeta kNN=0.1443 |
| 4 | baseline_vit_ema_vicreg_varw10_20260430_193408 | baseline | vit | ema | vicreg_varw10 | **0.0370** | zeta lin=0.2943 / alpha kNN=0.0455 / zeta kNN=0.4004 |
| 5 | exp_a_vit_ema_vicreg_20260428_111313 | exp_a | vit | ema | vicreg | **0.0676** | zeta lin=0.1784 / alpha kNN=0.1718 / zeta kNN=0.4109 |

### Top 5 by alpha kNN test MSE

| rank | run_id | routing | backbone | target | loss | alpha kNN test MSE | (other 3 metrics) |
|---|---|---|---|---|---|---|---|
| 1 | baseline_cnn_ema_vicreg_no_cov_20260428_190830 | baseline | cnn | ema | vicreg_no_cov | **0.0131** | alpha lin=0.0195 / zeta lin=0.3150 / zeta kNN=0.7559 |
| 2 | baseline_vit_ema_vicreg_lam001_20260430_170646 | baseline | vit | ema | vicreg_lam001 | **0.0147** | alpha lin=0.0063 / zeta lin=0.0680 / zeta kNN=0.1017 |
| 3 | baseline_vit_ema_vicreg_varw10_20260430_193408 | baseline | vit | ema | vicreg_varw10 | **0.0455** | alpha lin=0.0370 / zeta lin=0.2943 / zeta kNN=0.4004 |
| 4 | baseline_vit_ema_vicreg_20260428_102316 | baseline | vit | ema | vicreg | **0.0864** | alpha lin=0.0316 / zeta lin=0.1332 / zeta kNN=0.1443 |
| 5 | exp_a_vit_ema_vicreg_lam001_20260430_202310 | exp_a | vit | ema | vicreg_lam001 | **0.1616** | alpha lin=0.0702 / zeta lin=0.2174 / zeta kNN=0.8070 |

---

## zeta (steric alignment)

### Top 5 by zeta linear test MSE

| rank | run_id | routing | backbone | target | loss | zeta linear test MSE | (other 3 metrics) |
|---|---|---|---|---|---|---|---|
| 1 | baseline_vit_ema_vicreg_lam001_20260430_170646 | baseline | vit | ema | vicreg_lam001 | **0.0680** | alpha lin=0.0063 / alpha kNN=0.0147 / zeta kNN=0.1017 |
| 2 | baseline_vit_ema_vicreg_20260428_102316 | baseline | vit | ema | vicreg | **0.1332** | alpha lin=0.0316 / alpha kNN=0.0864 / zeta kNN=0.1443 |
| 3 | exp_a_vit_ema_vicreg_20260428_111313 | exp_a | vit | ema | vicreg | **0.1784** | alpha lin=0.0676 / alpha kNN=0.1718 / zeta kNN=0.4109 |
| 4 | djepa_exp_b_vicreg_lam001_v0_20260424_023349 | exp_b | vit | shared | vicreg_lam001 | **0.2004** | alpha lin=0.6592 / alpha kNN=0.6235 / zeta kNN=0.1894 |
| 5 | exp_b_cnn_ema_vicreg_lam001_20260430_211210 | exp_b | cnn | ema | vicreg_lam001 | **0.2054** | alpha lin=0.2362 / alpha kNN=0.4753 / zeta kNN=0.3641 |

### Top 5 by zeta kNN test MSE

| rank | run_id | routing | backbone | target | loss | zeta kNN test MSE | (other 3 metrics) |
|---|---|---|---|---|---|---|---|
| 1 | baseline_vit_ema_vicreg_lam001_20260430_170646 | baseline | vit | ema | vicreg_lam001 | **0.1017** | alpha lin=0.0063 / zeta lin=0.0680 / alpha kNN=0.0147 |
| 2 | baseline_vit_ema_vicreg_20260428_102316 | baseline | vit | ema | vicreg | **0.1443** | alpha lin=0.0316 / zeta lin=0.1332 / alpha kNN=0.0864 |
| 3 | djepa_exp_b_vicreg_lam001_v0_20260424_023349 | exp_b | vit | shared | vicreg_lam001 | **0.1894** | alpha lin=0.6592 / zeta lin=0.2004 / alpha kNN=0.6235 |
| 4 | exp_a_cnn_ema_vicreg_20260428_234329 | exp_a | cnn | ema | vicreg | **0.3355** | alpha lin=0.1601 / zeta lin=0.2832 / alpha kNN=0.2027 |
| 5 | exp_b_cnn_ema_vicreg_lam001_20260430_211210 | exp_b | cnn | ema | vicreg_lam001 | **0.3641** | alpha lin=0.2362 / zeta lin=0.2054 / alpha kNN=0.4753 |

---

## Pareto-optimal on (alpha kNN, zeta kNN)

A run is Pareto-optimal if no other run has both lower alpha kNN AND lower zeta kNN. These are the configurations that are not strictly dominated by any other in the joint (alpha, zeta) space.

| run_id | routing | backbone | target | loss | alpha kNN | zeta kNN |
|---|---|---|---|---|---|---|
| baseline_cnn_ema_vicreg_no_cov_20260428_190830 | baseline | cnn | ema | vicreg_no_cov | **0.0131** | **0.7559** |
| baseline_vit_ema_vicreg_lam001_20260430_170646 | baseline | vit | ema | vicreg_lam001 | **0.0147** | **0.1017** |

*(2 of 34 runs lie on the (alpha, zeta) kNN Pareto frontier.)*

