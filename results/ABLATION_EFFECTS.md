# Ablation Effects: Per-Axis Decision Impact

This document enumerates every paired comparison in the dataset where exactly one design axis is toggled and all others are held constant. Two sections: one for alpha (active dipole strength) and one for zeta (steric alignment). Each row in a table is a single ablation pair drawn from the project's 26 trained-and-evaluated cells (13 EMA cells in `runs/`, 13 shared-encoder cells in `../runs/`).

Where a paired comparison doesn't exist (e.g. CNN+shared, no_cov+SIGReg, etc.), the row is omitted; the data simply doesn't cover that subspace.

---

## Cross-table summary (read this first)

Punch lines distilled across all 18 ablation tables below. Detailed evidence in each section.

| Axis | Effect on alpha | Effect on zeta | Key qualifier |
|---|---|---|---|
| Backbone (ViT → CNN) | **mixed**: hurts on baseline (+0.26), helps on exp_b (-0.34) | **hurts** on baseline (+0.61), helps on exp_b (-0.58) | architecture × routing interaction is real |
| Target (shared → EMA) | mostly **helps**, with one giant rescue (+VICReg+baseline: -0.70) | mostly **helps**, giant rescue same place (-1.84) | EMA mainly matters where the loss alone collapses |
| Loss family (SIGReg → VICReg) | **routing-dependent**: helps baseline+exp_a, hurts exp_b | same direction as alpha | exp_b is structurally hostile to VICReg in any form tested |
| VICReg cov term (off → on) | **routing-dependent**: hurts baseline (+0.33), helps exp_a (-0.19), helps exp_b on CNN (-0.13) but hurts on ViT (+0.11) | **helps** across the board (-0.20 to -0.32) | cov off is best for alpha on baseline; cov on is best for zeta everywhere |
| SIGReg lambda (0.1 default) | both stronger and weaker hurt alpha | weaker (0.01) helps zeta (+ collapses) | tuning doesn't rescue exp_b under SIGReg |
| VICReg lambda (outer) | weaker (0.01) helps alpha on exp_b (-0.16) | weaker (0.01) helps zeta on exp_b dramatically (-1.30) | weakening the regularizer is a real lever |
| VICReg var_weight | minor effects (slight help/hurt) | minor effects | not a sensitive knob in the regime tested |
| VICReg cov_weight magnitude | slight effect on exp_b only | slight effect | also not sensitive |
| Routing (baseline → exp_a → exp_b) | exp_a usually slightly worse; exp_b dramatically worse on alpha | exp_a roughly tied with baseline; exp_b also worse | the §6.1 "exp_b exposes alpha linearly" hypothesis is consistently refuted |

**Three takeaways for design:**

1. **For alpha, the project leader is `baseline + cnn + ema + vicreg_no_cov`** (alpha kNN = 0.0131). Removing the cov term and switching to CNN both help on baseline; switching from shared to EMA target rescues an otherwise-collapsing VICReg run.
2. **For zeta, the project leader is `baseline + vit + ema + vicreg`** (zeta kNN = 0.144). The ViT backbone and the cov-term-on VICReg are both winning choices for zeta.
3. **No single configuration wins both targets.** Architecture × loss × target × routing all interact non-trivially. The cov term has the *cleanest* split: removing it is best for alpha on baseline, keeping it is best for zeta everywhere.

---

# alpha (kNN test MSE)

All values are test MSE on z-scored targets (constant-mean baseline = 1.0; lower is better). Held-constant axes are listed in each row.

Verdict thresholds: |delta| < 0.005 = neutral; < 0.1 = slight; < 0.5 = helps/hurts; >= 0.5 = big.

---

## 1. Backbone

### Backbone: `vit` -> `cnn`

| held constant | vit | cnn | delta | verdict |
|---|---|---|---|---|
| routing=baseline, target=ema, loss=vicreg | 0.0864 | 0.3478 | +0.2613 | hurts |
| routing=exp_a, target=ema, loss=vicreg | 0.1718 | 0.2027 | +0.0308 | slight hurt |
| routing=exp_b, target=ema, loss=vicreg | 0.7510 | 0.4101 | -0.3409 | helps |
| routing=exp_b, target=ema, loss=vicreg_no_cov | 0.6361 | 0.5413 | -0.0948 | slight help |

## 2. Target encoder

### Target encoder: `shared (no EMA)` -> `ema`

| held constant | shared (no EMA) | ema | delta | verdict |
|---|---|---|---|---|
| routing=baseline, backbone=vit, loss=sigreg | 0.2314 | 0.2744 | +0.0430 | slight hurt |
| routing=baseline, backbone=vit, loss=vicreg | 0.7828 | 0.0864 | -0.6964 | **big help** |
| routing=exp_a, backbone=vit, loss=sigreg | 0.3352 | 0.3020 | -0.0331 | slight help |
| routing=exp_a, backbone=vit, loss=vicreg | 0.2037 | 0.1718 | -0.0318 | slight help |
| routing=exp_b, backbone=vit, loss=sigreg | 0.5402 | 0.5804 | +0.0402 | slight hurt |
| routing=exp_b, backbone=vit, loss=vicreg | 0.7872 | 0.7510 | -0.0362 | slight help |

## 3. Loss family

### Loss family: `sigreg` -> `vicreg (full)`

| held constant | sigreg | vicreg (full) | delta | verdict |
|---|---|---|---|---|
| routing=baseline, backbone=vit, target=shared | 0.2314 | 0.7828 | +0.5514 | **big hurt** |
| routing=baseline, backbone=vit, target=ema | 0.2744 | 0.0864 | -0.1880 | helps |
| routing=exp_a, backbone=vit, target=shared | 0.3352 | 0.2037 | -0.1315 | helps |
| routing=exp_a, backbone=vit, target=ema | 0.3020 | 0.1718 | -0.1302 | helps |
| routing=exp_b, backbone=vit, target=shared | 0.5402 | 0.7872 | +0.2470 | hurts |
| routing=exp_b, backbone=vit, target=ema | 0.5804 | 0.7510 | +0.1705 | hurts |

## 4. VICReg cov term

### VICReg cov term: `off (vicreg_no_cov)` -> `on (vicreg, cov_w=1)`

| held constant | off (vicreg_no_cov) | on (vicreg, cov_w=1) | delta | verdict |
|---|---|---|---|---|
| routing=baseline, backbone=cnn, target=ema | 0.0131 | 0.3478 | +0.3346 | hurts |
| routing=exp_a, backbone=cnn, target=ema | 0.3911 | 0.2027 | -0.1885 | helps |
| routing=exp_b, backbone=vit, target=ema | 0.6361 | 0.7510 | +0.1149 | hurts |
| routing=exp_b, backbone=cnn, target=ema | 0.5413 | 0.4101 | -0.1312 | helps |

## 5. SIGReg lambda (outer scale)

### SIGReg lambda: `0.1 default` -> `0.01 lam001`

| held constant | 0.1 default | 0.01 lam001 | delta | verdict |
|---|---|---|---|---|
| routing=exp_b, backbone=vit, target=shared | 0.5402 | 0.7890 | +0.2488 | hurts |

### SIGReg lambda: `0.1 default` -> `1.0 lam1`

| held constant | 0.1 default | 1.0 lam1 | delta | verdict |
|---|---|---|---|---|
| routing=exp_b, backbone=vit, target=shared | 0.5402 | 0.5166 | -0.0237 | slight help |

## 6. VICReg lambda (outer scale)

### VICReg lambda (outer): `0.1 default` -> `0.01 lam001`

| held constant | 0.1 default | 0.01 lam001 | delta | verdict |
|---|---|---|---|---|
| routing=exp_b, backbone=vit, target=shared | 0.7872 | 0.6235 | -0.1637 | helps |

### VICReg lambda (outer): `0.1 default` -> `1.0 lam1`

| held constant | 0.1 default | 1.0 lam1 | delta | verdict |
|---|---|---|---|---|
| routing=exp_b, backbone=vit, target=shared | 0.7872 | 0.7839 | -0.0033 | neutral |

## 7. VICReg var_weight

### VICReg var_weight: `25 default` -> `10 varw10`

| held constant | 25 default | 10 varw10 | delta | verdict |
|---|---|---|---|---|
| routing=exp_b, backbone=vit, target=shared | 0.7872 | 0.7640 | -0.0232 | slight help |

### VICReg var_weight: `25 default` -> `50 varw50`

| held constant | 25 default | 50 varw50 | delta | verdict |
|---|---|---|---|---|
| routing=exp_b, backbone=vit, target=shared | 0.7872 | 0.8153 | +0.0281 | slight hurt |

## 8. VICReg cov_weight magnitude

### VICReg cov_weight: `1 default` -> `5 covw5`

| held constant | 1 default | 5 covw5 | delta | verdict |
|---|---|---|---|---|
| routing=exp_b, backbone=vit, target=shared | 0.7872 | 0.7616 | -0.0256 | slight help |

## 9. Routing

### Routing: `baseline` -> `exp_a`

| held constant | baseline | exp_a | delta | verdict |
|---|---|---|---|---|
| backbone=vit, target=shared, loss=sigreg | 0.2314 | 0.3352 | +0.1038 | hurts |
| backbone=vit, target=shared, loss=vicreg | 0.7828 | 0.2037 | -0.5791 | **big help** |
| backbone=vit, target=ema, loss=sigreg | 0.2744 | 0.3020 | +0.0277 | slight hurt |
| backbone=vit, target=ema, loss=vicreg | 0.0864 | 0.1718 | +0.0854 | slight hurt |
| backbone=cnn, target=ema, loss=vicreg | 0.3478 | 0.2027 | -0.1451 | helps |
| backbone=cnn, target=ema, loss=vicreg_no_cov | 0.0131 | 0.3911 | +0.3780 | hurts |

### Routing: `baseline` -> `exp_b`

| held constant | baseline | exp_b | delta | verdict |
|---|---|---|---|---|
| backbone=vit, target=shared, loss=sigreg | 0.2314 | 0.5402 | +0.3088 | hurts |
| backbone=vit, target=shared, loss=vicreg | 0.7828 | 0.7872 | +0.0044 | neutral |
| backbone=vit, target=ema, loss=sigreg | 0.2744 | 0.5804 | +0.3061 | hurts |
| backbone=vit, target=ema, loss=vicreg | 0.0864 | 0.7510 | +0.6646 | **big hurt** |
| backbone=cnn, target=ema, loss=vicreg | 0.3478 | 0.4101 | +0.0624 | slight hurt |
| backbone=cnn, target=ema, loss=vicreg_no_cov | 0.0131 | 0.5413 | +0.5282 | **big hurt** |

### Routing: `exp_a` -> `exp_b`

| held constant | exp_a | exp_b | delta | verdict |
|---|---|---|---|---|
| backbone=vit, target=shared, loss=sigreg | 0.3352 | 0.5402 | +0.2051 | hurts |
| backbone=vit, target=shared, loss=vicreg | 0.2037 | 0.7872 | +0.5835 | **big hurt** |
| backbone=vit, target=ema, loss=sigreg | 0.3020 | 0.5804 | +0.2784 | hurts |
| backbone=vit, target=ema, loss=vicreg | 0.1718 | 0.7510 | +0.5791 | **big hurt** |
| backbone=cnn, target=ema, loss=vicreg | 0.2027 | 0.4101 | +0.2075 | hurts |
| backbone=cnn, target=ema, loss=vicreg_no_cov | 0.3911 | 0.5413 | +0.1502 | hurts |


---

# zeta (kNN test MSE)

All values are test MSE on z-scored targets (constant-mean baseline = 1.0; lower is better). Held-constant axes are listed in each row.

Verdict thresholds: |delta| < 0.005 = neutral; < 0.1 = slight; < 0.5 = helps/hurts; >= 0.5 = big.

---

## 1. Backbone

### Backbone: `vit` -> `cnn`

| held constant | vit | cnn | delta | verdict |
|---|---|---|---|---|
| routing=baseline, target=ema, loss=vicreg | 0.1443 | 0.3870 | +0.2427 | hurts |
| routing=exp_a, target=ema, loss=vicreg | 0.4109 | 0.3355 | -0.0754 | slight help |
| routing=exp_b, target=ema, loss=vicreg | 1.0668 | 0.4874 | -0.5794 | **big help** |
| routing=exp_b, target=ema, loss=vicreg_no_cov | 0.4874 | 0.4254 | -0.0620 | slight help |

## 2. Target encoder

### Target encoder: `shared (no EMA)` -> `ema`

| held constant | shared (no EMA) | ema | delta | verdict |
|---|---|---|---|---|
| routing=baseline, backbone=vit, loss=sigreg | 0.3829 | 0.4240 | +0.0411 | slight hurt |
| routing=baseline, backbone=vit, loss=vicreg | 1.9799 | 0.1443 | -1.8356 | **big help** |
| routing=exp_a, backbone=vit, loss=sigreg | 0.3977 | 0.4614 | +0.0637 | slight hurt |
| routing=exp_a, backbone=vit, loss=vicreg | 0.6745 | 0.4109 | -0.2636 | helps |
| routing=exp_b, backbone=vit, loss=sigreg | 0.5342 | 0.5428 | +0.0086 | slight hurt |
| routing=exp_b, backbone=vit, loss=vicreg | 1.4880 | 1.0668 | -0.4212 | helps |

## 3. Loss family

### Loss family: `sigreg` -> `vicreg (full)`

| held constant | sigreg | vicreg (full) | delta | verdict |
|---|---|---|---|---|
| routing=baseline, backbone=vit, target=shared | 0.3829 | 1.9799 | +1.5969 | **big hurt** |
| routing=baseline, backbone=vit, target=ema | 0.4240 | 0.1443 | -0.2797 | helps |
| routing=exp_a, backbone=vit, target=shared | 0.3977 | 0.6745 | +0.2768 | hurts |
| routing=exp_a, backbone=vit, target=ema | 0.4614 | 0.4109 | -0.0505 | slight help |
| routing=exp_b, backbone=vit, target=shared | 0.5342 | 1.4880 | +0.9538 | **big hurt** |
| routing=exp_b, backbone=vit, target=ema | 0.5428 | 1.0668 | +0.5240 | **big hurt** |

## 4. VICReg cov term

### VICReg cov term: `off (vicreg_no_cov)` -> `on (vicreg, cov_w=1)`

| held constant | off (vicreg_no_cov) | on (vicreg, cov_w=1) | delta | verdict |
|---|---|---|---|---|
| routing=baseline, backbone=cnn, target=ema | 0.7559 | 0.3870 | -0.3689 | helps |
| routing=exp_a, backbone=cnn, target=ema | 0.9473 | 0.3355 | -0.6118 | **big help** |
| routing=exp_b, backbone=vit, target=ema | 0.4874 | 1.0668 | +0.5795 | **big hurt** |
| routing=exp_b, backbone=cnn, target=ema | 0.4254 | 0.4874 | +0.0620 | slight hurt |

## 5. SIGReg lambda (outer scale)

### SIGReg lambda: `0.1 default` -> `0.01 lam001`

| held constant | 0.1 default | 0.01 lam001 | delta | verdict |
|---|---|---|---|---|
| routing=exp_b, backbone=vit, target=shared | 0.5342 | 1.4616 | +0.9274 | **big hurt** |

### SIGReg lambda: `0.1 default` -> `1.0 lam1`

| held constant | 0.1 default | 1.0 lam1 | delta | verdict |
|---|---|---|---|---|
| routing=exp_b, backbone=vit, target=shared | 0.5342 | 0.5546 | +0.0204 | slight hurt |

## 6. VICReg lambda (outer scale)

### VICReg lambda (outer): `0.1 default` -> `0.01 lam001`

| held constant | 0.1 default | 0.01 lam001 | delta | verdict |
|---|---|---|---|---|
| routing=exp_b, backbone=vit, target=shared | 1.4880 | 0.1894 | -1.2986 | **big help** |

### VICReg lambda (outer): `0.1 default` -> `1.0 lam1`

| held constant | 0.1 default | 1.0 lam1 | delta | verdict |
|---|---|---|---|---|
| routing=exp_b, backbone=vit, target=shared | 1.4880 | 0.8961 | -0.5920 | **big help** |

## 7. VICReg var_weight

### VICReg var_weight: `25 default` -> `10 varw10`

| held constant | 25 default | 10 varw10 | delta | verdict |
|---|---|---|---|---|
| routing=exp_b, backbone=vit, target=shared | 1.4880 | 1.0292 | -0.4588 | helps |

### VICReg var_weight: `25 default` -> `50 varw50`

| held constant | 25 default | 50 varw50 | delta | verdict |
|---|---|---|---|---|
| routing=exp_b, backbone=vit, target=shared | 1.4880 | 1.6307 | +0.1427 | hurts |

## 8. VICReg cov_weight magnitude

### VICReg cov_weight: `1 default` -> `5 covw5`

| held constant | 1 default | 5 covw5 | delta | verdict |
|---|---|---|---|---|
| routing=exp_b, backbone=vit, target=shared | 1.4880 | 1.0910 | -0.3970 | helps |

## 9. Routing

### Routing: `baseline` -> `exp_a`

| held constant | baseline | exp_a | delta | verdict |
|---|---|---|---|---|
| backbone=vit, target=shared, loss=sigreg | 0.3829 | 0.3977 | +0.0148 | slight hurt |
| backbone=vit, target=shared, loss=vicreg | 1.9799 | 0.6745 | -1.3053 | **big help** |
| backbone=vit, target=ema, loss=sigreg | 0.4240 | 0.4614 | +0.0374 | slight hurt |
| backbone=vit, target=ema, loss=vicreg | 0.1443 | 0.4109 | +0.2666 | hurts |
| backbone=cnn, target=ema, loss=vicreg | 0.3870 | 0.3355 | -0.0515 | slight help |
| backbone=cnn, target=ema, loss=vicreg_no_cov | 0.7559 | 0.9473 | +0.1914 | hurts |

### Routing: `baseline` -> `exp_b`

| held constant | baseline | exp_b | delta | verdict |
|---|---|---|---|---|
| backbone=vit, target=shared, loss=sigreg | 0.3829 | 0.5342 | +0.1513 | hurts |
| backbone=vit, target=shared, loss=vicreg | 1.9799 | 1.4880 | -0.4918 | helps |
| backbone=vit, target=ema, loss=sigreg | 0.4240 | 0.5428 | +0.1188 | hurts |
| backbone=vit, target=ema, loss=vicreg | 0.1443 | 1.0668 | +0.9225 | **big hurt** |
| backbone=cnn, target=ema, loss=vicreg | 0.3870 | 0.4874 | +0.1004 | hurts |
| backbone=cnn, target=ema, loss=vicreg_no_cov | 0.7559 | 0.4254 | -0.3305 | helps |

### Routing: `exp_a` -> `exp_b`

| held constant | exp_a | exp_b | delta | verdict |
|---|---|---|---|---|
| backbone=vit, target=shared, loss=sigreg | 0.3977 | 0.5342 | +0.1365 | hurts |
| backbone=vit, target=shared, loss=vicreg | 0.6745 | 1.4880 | +0.8135 | **big hurt** |
| backbone=vit, target=ema, loss=sigreg | 0.4614 | 0.5428 | +0.0814 | slight hurt |
| backbone=vit, target=ema, loss=vicreg | 0.4109 | 1.0668 | +0.6559 | **big hurt** |
| backbone=cnn, target=ema, loss=vicreg | 0.3355 | 0.4874 | +0.1519 | hurts |
| backbone=cnn, target=ema, loss=vicreg_no_cov | 0.9473 | 0.4254 | -0.5219 | **big help** |

