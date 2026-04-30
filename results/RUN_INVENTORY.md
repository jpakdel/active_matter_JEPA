# Run Inventory

All training runs in the project: **26 total** (13 EMA new, 13 shared-encoder pre-refactor).

All MSE values are on z-scored targets (constant-mean baseline = 1.0; lower is better). N_train=700, N_val=96, N_test=104.

---

## Compact table — all runs sorted by alpha kNN test MSE

| run_id | routing | backbone | target | loss | a_lin | z_lin | a_kNN | z_kNN | wall (s) | source |
|---|---|---|---|---|---|---|---|---|---|---|
| baseline_cnn_ema_vicreg_no_cov_20260428_190830 | baseline | cnn | ema | vicreg_no_cov |   0.0195 |   0.3150 |   0.0131 |   0.7559 |   2765 | REFACTORED_CODEBASE/runs/ |
| baseline_vit_ema_vicreg_20260428_102316 | baseline | vit | ema | vicreg |   0.0316 |   0.1332 |   0.0864 |   0.1443 |   2852 | REFACTORED_CODEBASE/runs/ |
| exp_a_vit_ema_vicreg_20260428_111313 | exp_a | vit | ema | vicreg |   0.0676 |   0.1784 |   0.1718 |   0.4109 |   2783 | REFACTORED_CODEBASE/runs/ |
| exp_a_cnn_ema_vicreg_20260428_234329 | exp_a | cnn | ema | vicreg |   0.1601 |   0.2832 |   0.2027 |   0.3355 |   2731 | REFACTORED_CODEBASE/runs/ |
| djepa_exp_a_vicreg_v0_20260424_005651 | exp_a | vit | shared | vicreg |   0.0869 |   0.2455 |   0.2037 |   0.6745 |   2761 | ../runs/ |
| baseline_v0_20260421_152635 | baseline | vit | shared | sigreg |   0.1747 |   0.4420 |   0.2314 |   0.3829 |   2905 | ../runs/ |
| baseline_vit_ema_sigreg_20260428_075508 | baseline | vit | ema | sigreg |   0.1904 |   0.3568 |   0.2744 |   0.4240 |   2754 | REFACTORED_CODEBASE/runs/ |
| exp_a_vit_ema_sigreg_20260428_084330 | exp_a | vit | ema | sigreg |   0.1753 |   0.3754 |   0.3020 |   0.4614 |   2787 | REFACTORED_CODEBASE/runs/ |
| exp_a_v0_20260421_185035 | exp_a | vit | shared | sigreg |   0.2141 |   0.3143 |   0.3352 |   0.3977 |   3034 | ../runs/ |
| baseline_cnn_ema_vicreg_20260428_225444 | baseline | cnn | ema | vicreg |   0.1403 |   0.3224 |   0.3478 |   0.3870 |   2775 | REFACTORED_CODEBASE/runs/ |
| exp_a_cnn_ema_vicreg_no_cov_20260428_204808 | exp_a | cnn | ema | vicreg_no_cov |   0.2998 |   0.6986 |   0.3911 |   0.9473 |   2787 | REFACTORED_CODEBASE/runs/ |
| exp_b_cnn_ema_vicreg_20260429_003133 | exp_b | cnn | ema | vicreg |   0.2353 |   0.2806 |   0.4101 |   0.4874 |   2776 | REFACTORED_CODEBASE/runs/ |
| exp_b_lam1_v0_20260422_121613 | exp_b | vit | shared | sigreg_lam1 |   0.4881 |   0.3955 |   0.5166 |   0.5546 |   2721 | ../runs/ |
| exp_b_v0_20260421_200801 | exp_b | vit | shared | sigreg |   0.5932 |   0.4252 |   0.5402 |   0.5342 |   3015 | ../runs/ |
| exp_b_cnn_ema_vicreg_no_cov_20260428_213707 | exp_b | cnn | ema | vicreg_no_cov |   0.4569 |   0.3153 |   0.5413 |   0.4254 |   2752 | REFACTORED_CODEBASE/runs/ |
| exp_b_vit_ema_sigreg_20260428_093243 | exp_b | vit | ema | sigreg |   0.4478 |   0.3924 |   0.5804 |   0.5428 |   2873 | REFACTORED_CODEBASE/runs/ |
| djepa_exp_b_vicreg_lam001_v0_20260424_023349 | exp_b | vit | shared | vicreg_lam001 |   0.6592 |   0.2004 |   0.6235 |   0.1894 |   2758 | ../runs/ |
| exp_b_vit_ema_vicreg_no_cov_20260428_151528 | exp_b | vit | ema | vicreg_no_cov |   0.5874 |   0.3415 |   0.6361 |   0.4874 |   2998 | REFACTORED_CODEBASE/runs/ |
| exp_b_vit_ema_vicreg_20260428_142657 | exp_b | vit | ema | vicreg |   0.7162 |   0.8886 |   0.7510 |   1.0668 |   2764 | REFACTORED_CODEBASE/runs/ |
| djepa_exp_b_vicreg_covw5_v0_20260424_045839 | exp_b | vit | shared | vicreg_covw5 |   0.7573 |   0.9224 |   0.7616 |   1.0910 |   2735 | ../runs/ |
| djepa_exp_b_vicreg_varw10_v0_20260424_041038 | exp_b | vit | shared | vicreg_varw10 |   0.7160 |   0.9231 |   0.7640 |   1.0292 |   2735 | ../runs/ |
| djepa_baseline_vicreg_v0_20260424_000719 | baseline | vit | shared | vicreg |   0.4169 |   0.4489 |   0.7828 |   1.9799 |   2815 | ../runs/ |
| djepa_exp_b_vicreg_lam1_v0_20260424_032214 | exp_b | vit | shared | vicreg_lam1 |   0.7261 |   0.9038 |   0.7839 |   0.8961 |   2757 | ../runs/ |
| djepa_exp_b_vicreg_v0_20260424_014519 | exp_b | vit | shared | vicreg |   0.7528 |   0.9283 |   0.7872 |   1.4880 |   2758 | ../runs/ |
| exp_b_lam001_v0_20260422_113047 | exp_b | vit | shared | sigreg_lam001 |   0.5912 |   0.3292 |   0.7890 |   1.4616 |   2720 | ../runs/ |
| djepa_exp_b_vicreg_varw50_v0_20260424_054642 | exp_b | vit | shared | vicreg_varw50 |   0.7444 |   0.8826 |   0.8153 |   1.6307 |   2735 | ../runs/ |

---

## Grouped by routing

### baseline

| backbone | target | loss | a_lin | z_lin | a_kNN | z_kNN | a val_lin | z val_lin | final_pmse |
|---|---|---|---|---|---|---|---|---|---|
| cnn | ema | vicreg_no_cov |   0.0195 |   0.3150 |   0.0131 |   0.7559 |   0.0119 |   0.3788 |   0.0054 |
| vit | ema | vicreg |   0.0316 |   0.1332 |   0.0864 |   0.1443 |   0.0533 |   0.1099 |   0.0868 |
| vit | shared | sigreg |   0.1747 |   0.4420 |   0.2314 |   0.3829 |   0.2164 |   0.3663 |   0.9759 |
| vit | ema | sigreg |   0.1904 |   0.3568 |   0.2744 |   0.4240 |   0.2124 |   0.3625 |   0.9662 |
| cnn | ema | vicreg |   0.1403 |   0.3224 |   0.3478 |   0.3870 |   0.1103 |   0.3544 |   1.0374 |
| vit | shared | vicreg |   0.4169 |   0.4489 |   0.7828 |   1.9799 |   0.4363 |   0.4293 |   0.0015 |

### exp_a

| backbone | target | loss | a_lin | z_lin | a_kNN | z_kNN | a val_lin | z val_lin | final_pmse |
|---|---|---|---|---|---|---|---|---|---|
| vit | ema | vicreg |   0.0676 |   0.1784 |   0.1718 |   0.4109 |   0.0662 |   0.1809 |   0.2888 |
| cnn | ema | vicreg |   0.1601 |   0.2832 |   0.2027 |   0.3355 |   0.0866 |   0.3123 |   0.7607 |
| vit | shared | vicreg |   0.0869 |   0.2455 |   0.2037 |   0.6745 |   0.0578 |   0.2704 |   0.2533 |
| vit | ema | sigreg |   0.1753 |   0.3754 |   0.3020 |   0.4614 |   0.1808 |   0.3605 |   0.6757 |
| vit | shared | sigreg |   0.2141 |   0.3143 |   0.3352 |   0.3977 |   0.1747 |   0.3876 |   0.7068 |
| cnn | ema | vicreg_no_cov |   0.2998 |   0.6986 |   0.3911 |   0.9473 |   0.1890 |   0.6764 |   1.2949 |

### exp_b

| backbone | target | loss | a_lin | z_lin | a_kNN | z_kNN | a val_lin | z val_lin | final_pmse |
|---|---|---|---|---|---|---|---|---|---|
| cnn | ema | vicreg |   0.2353 |   0.2806 |   0.4101 |   0.4874 |   0.2211 |   0.2421 |   0.1304 |
| vit | shared | sigreg_lam1 |   0.4881 |   0.3955 |   0.5166 |   0.5546 |   0.5570 |   0.4359 |   0.2080 |
| vit | shared | sigreg |   0.5932 |   0.4252 |   0.5402 |   0.5342 |   0.5445 |   0.4305 |   0.2330 |
| cnn | ema | vicreg_no_cov |   0.4569 |   0.3153 |   0.5413 |   0.4254 |   0.5174 |   0.4110 |   0.5535 |
| vit | ema | sigreg |   0.4478 |   0.3924 |   0.5804 |   0.5428 |   0.5027 |   0.3610 |   0.0799 |
| vit | shared | vicreg_lam001 |   0.6592 |   0.2004 |   0.6235 |   0.1894 |   0.4636 |   0.2795 |   0.0258 |
| vit | ema | vicreg_no_cov |   0.5874 |   0.3415 |   0.6361 |   0.4874 |   0.5770 |   0.3428 |   0.0000 |
| vit | ema | vicreg |   0.7162 |   0.8886 |   0.7510 |   1.0668 |   0.7368 |   0.8145 |   0.0007 |
| vit | shared | vicreg_covw5 |   0.7573 |   0.9224 |   0.7616 |   1.0910 |   0.7396 |   0.8800 |   0.0011 |
| vit | shared | vicreg_varw10 |   0.7160 |   0.9231 |   0.7640 |   1.0292 |   0.7606 |   0.8172 |   0.0007 |
| vit | shared | vicreg_lam1 |   0.7261 |   0.9038 |   0.7839 |   0.8961 |   0.7676 |   0.8386 |   0.0004 |
| vit | shared | vicreg |   0.7528 |   0.9283 |   0.7872 |   1.4880 |   0.7815 |   0.8195 |   0.0010 |
| vit | shared | sigreg_lam001 |   0.5912 |   0.3292 |   0.7890 |   1.4616 |   0.6875 |   0.3340 |   0.0152 |
| vit | shared | vicreg_varw50 |   0.7444 |   0.8826 |   0.8153 |   1.6307 |   0.7417 |   0.8503 |   0.0007 |

---

## Hyperparameters per run

Held-constant across every run: `lr=3e-4`, `batch_size=2`, `num_epochs=30`, `total_steps=10500`, `warmup_steps=700`, `weight_decay=0.05->0.4 cosine`, `grad_clip=1.0`, `seed=0`. ViT runs use AMP fp16; CNN runs use fp32.

| run_id | lambda | vicreg_var_w | vicreg_cov_w | ema_decay | use_amp |
|---|---|---|---|---|---|
| baseline_cnn_ema_vicreg_20260428_225444 |  0.100 |   25.0 |    1.0 |  0.996 | False |
| baseline_cnn_ema_vicreg_no_cov_20260428_190830 |  0.100 |   25.0 |    0.0 |  0.996 | False |
| baseline_v0_20260421_152635 |  0.100 |    n/a |    n/a |    n/a | True |
| baseline_vit_ema_sigreg_20260428_075508 |  0.100 |    n/a |    n/a |  0.996 | True |
| baseline_vit_ema_vicreg_20260428_102316 |  0.100 |   25.0 |    1.0 |  0.996 | True |
| djepa_baseline_vicreg_v0_20260424_000719 |  0.100 |   25.0 |    1.0 |    n/a | True |
| djepa_exp_a_vicreg_v0_20260424_005651 |  0.100 |   25.0 |    1.0 |    n/a | True |
| djepa_exp_b_vicreg_covw5_v0_20260424_045839 |  0.100 |   25.0 |    5.0 |    n/a | True |
| djepa_exp_b_vicreg_lam001_v0_20260424_023349 |  0.010 |   25.0 |    1.0 |    n/a | True |
| djepa_exp_b_vicreg_lam1_v0_20260424_032214 |  1.000 |   25.0 |    1.0 |    n/a | True |
| djepa_exp_b_vicreg_v0_20260424_014519 |  0.100 |   25.0 |    1.0 |    n/a | True |
| djepa_exp_b_vicreg_varw10_v0_20260424_041038 |  0.100 |   10.0 |    1.0 |    n/a | True |
| djepa_exp_b_vicreg_varw50_v0_20260424_054642 |  0.100 |   50.0 |    1.0 |    n/a | True |
| exp_a_cnn_ema_vicreg_20260428_234329 |  0.100 |   25.0 |    1.0 |  0.996 | False |
| exp_a_cnn_ema_vicreg_no_cov_20260428_204808 |  0.100 |   25.0 |    0.0 |  0.996 | False |
| exp_a_v0_20260421_185035 |  0.100 |    n/a |    n/a |    n/a | True |
| exp_a_vit_ema_sigreg_20260428_084330 |  0.100 |    n/a |    n/a |  0.996 | True |
| exp_a_vit_ema_vicreg_20260428_111313 |  0.100 |   25.0 |    1.0 |  0.996 | True |
| exp_b_cnn_ema_vicreg_20260429_003133 |  0.100 |   25.0 |    1.0 |  0.996 | False |
| exp_b_cnn_ema_vicreg_no_cov_20260428_213707 |  0.100 |   25.0 |    0.0 |  0.996 | False |
| exp_b_lam001_v0_20260422_113047 |  0.010 |    n/a |    n/a |    n/a | True |
| exp_b_lam1_v0_20260422_121613 |  1.000 |    n/a |    n/a |    n/a | True |
| exp_b_v0_20260421_200801 |  0.100 |    n/a |    n/a |    n/a | True |
| exp_b_vit_ema_sigreg_20260428_093243 |  0.100 |    n/a |    n/a |  0.996 | True |
| exp_b_vit_ema_vicreg_20260428_142657 |  0.100 |   25.0 |    1.0 |  0.996 | True |
| exp_b_vit_ema_vicreg_no_cov_20260428_151528 |  0.100 |   25.0 |    0.0 |  0.996 | True |

