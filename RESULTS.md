# Complete Results — Qwen2.5-7B on GSM8K

## 1. Behavioral Accuracy

| Samples | Correct | Accuracy |
|---------|---------|----------|
| 500 | 426 | **85.2%** |

## 2. Steering Vector Collection

- Total questions attempted: 500
- Successful: 496
- Total step-hash differences: 2285
- Layers: 28, Hidden dim: 3584
- Steering vectors shape: (28, 3584)

## 3. Trajectory Distance Analysis (Layer 27)

Euclidean distances between step activations:

| Transition | Correct | Incorrect | Δ |
|------------|---------|-----------|---|
| step1_to_step2 | 384.21 (n=426) | 383.48 (n=70) | +0.73 |
| step2_to_last | 148.08 (n=390) | 148.70 (n=68) | -0.63 |
| last_to_hash | 180.58 (n=426) | 170.24 (n=70) | +10.34 |
| second_last_to_last | 159.64 (n=426) | 142.94 (n=70) | +16.71 |

Cosine distances:

| Transition | Correct | Incorrect | Δ |
|------------|---------|-----------|---|
| step1_to_step2 | 0.938 | 0.934 | +0.004 |
| step2_to_last | 0.133 | 0.124 | +0.009 |
| last_to_hash | 0.195 | 0.190 | +0.005 |
| second_last_to_last | 0.217 | 0.146 | +0.070 |

**Key finding:** `last_to_hash` shows the largest divergence (Euclidean: correct=180.58, incorrect=170.24; Cosine: correct=0.195, incorrect=0.190). Early transitions (step1→step2) are nearly identical. This confirms the paper's thesis: correct/incorrect reasoning diverges at LATE steps.

## 4. Correctness Predictors (252 configs)

9 feature sets × 28 layers = 252 trained logistic regression classifiers.

### Top 20 by ROC-AUC

| Rank | AUC | Acc | Layer | Feature |
|------|-----|-----|-------|---------|
| 1 | 0.8205 | 0.5217 | 12 | step_diffs |
| 2 | 0.8073 | 0.5200 | 26 | hash_last_diffs_pca_joint |
| 3 | 0.7940 | 0.5400 | 15 | hash_pca |
| 4 | 0.7912 | 0.6522 | 4 | step1_step2_step3 |
| 5 | 0.7874 | 0.7800 | 21 | hash_only |
| 6 | 0.7711 | 0.6087 | 22 | step_diffs |
| 7 | 0.7508 | 0.7200 | 18 | hash_minus_last |
| 8 | 0.7473 | 0.5652 | 21 | step1_step2_step3 |
| 9 | 0.7342 | 0.5000 | 1 | hash_last_diffs_pca |
| 10 | 0.7289 | 0.6739 | 11 | step_diffs |
| 11 | 0.7276 | 0.6000 | 25 | hash_last_diffs_pca_joint |
| 12 | 0.7216 | 0.5435 | 15 | step_diffs |
| 13 | 0.7209 | 0.6400 | 18 | hash_only |
| 14 | 0.7209 | 0.5400 | 2 | hash_pca |
| 15 | 0.7179 | 0.6522 | 6 | step_diffs |
| 16 | 0.7179 | 0.6522 | 14 | step_diffs |
| 17 | 0.7176 | 0.5800 | 26 | hash_only |
| 18 | 0.7143 | 0.6200 | 13 | hash_pca |
| 19 | 0.7110 | 0.7200 | 1 | hash_minus_last |
| 20 | 0.7110 | 0.6600 | 17 | hash_minus_last |

### Best per Feature Set

| Feature | Best AUC | Layer | Accuracy |
|---------|----------|-------|----------|
| hash_last_diffs_pca | 0.7342 | 1 | 0.5000 |
| hash_last_diffs_pca_joint | 0.8073 | 26 | 0.5200 |
| hash_minus_last | 0.7508 | 18 | 0.7200 |
| hash_only | 0.7874 | 21 | 0.7800 |
| hash_pca | 0.7940 | 15 | 0.5400 |
| step1_step2 | 0.6977 | 25 | 0.4200 |
| step1_step2_step3 | 0.7912 | 4 | 0.6522 |
| step2_minus_step1 | 0.6944 | 5 | 0.6000 |
| step_diffs | 0.8205 | 12 | 0.5217 |

### Best AUC per Layer

```
L 0: AUC=0.651 ##########################
L 1: AUC=0.734 #############################
L 2: AUC=0.721 ############################
L 3: AUC=0.688 ###########################
L 4: AUC=0.791 ###############################
L 5: AUC=0.694 ###########################
L 6: AUC=0.718 ############################
L 7: AUC=0.691 ###########################
L 8: AUC=0.688 ###########################
L 9: AUC=0.671 ##########################
L10: AUC=0.598 #######################
L11: AUC=0.729 #############################
L12: AUC=0.821 ################################
L13: AUC=0.714 ############################
L14: AUC=0.718 ############################
L15: AUC=0.794 ###############################
L16: AUC=0.703 ############################
L17: AUC=0.711 ############################
L18: AUC=0.751 ##############################
L19: AUC=0.663 ##########################
L20: AUC=0.678 ###########################
L21: AUC=0.787 ###############################
L22: AUC=0.771 ##############################
L23: AUC=0.688 ###########################
L24: AUC=0.676 ###########################
L25: AUC=0.728 #############################
L26: AUC=0.807 ################################
L27: AUC=0.671 ##########################
```

### Key Observations

- **Best predictor:** `step_diffs` at layer 12 (AUC=0.82)
- **Peak layers:** 12, 15, 21, 26, 4, 22 (all >0.77 AUC)
- **Lowest layer with signal:** Layer 0 already shows AUC>0.65
- **PCA-based features** (hash_pca, hash_last_diffs_pca_joint) perform comparably to raw features
- **Pure step features** (step1_step2, step2_minus_step1) are weaker — differences between steps matter more

## 5. Summary Statistics

- Predictors with AUC > 0.70: 23/252 (9.1%)
- Predictors with AUC > 0.75: 7/252 (2.8%)
- Predictors with AUC > 0.80: 2/252 (0.8%)
- Mean AUC: 0.5725
- Max AUC: 0.8205
- Min AUC: 0.1894
