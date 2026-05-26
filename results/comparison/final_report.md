# Final Comparison Report — EGB-250 Bearing Fault Detection
Generated: 2026-05-13 19:48  |  Device: cuda  |  RANDOM_SEED=42

---

## 1. Checkpoints Used

| Model | Checkpoint | Selection Criterion |
|---|---|---|
| Transformer | `transformer_run_14.pt` | Run 14 (transformers_runs_5_1) |
| GAT | `gat_run_07.pt` | Menor val_loss em gat_runs_summary.csv (Run 7, seed=48, val_loss=0.001391) |
| WAE-GAN+XGB | `wae_gan_run_14.pt` | Diagnoser pré-salvo: `wae_gan_xgboost_diagnoser_run_14.pkl` |

**Test split:** runs {13, 4, 7} por classe (seed=42)  
**WAE-GAN anomaly threshold:** 0.148121 (IQR × 1.5 em janelas normais do treino)

---

## 2. Macro Metrics (Test Set)

| Model       |   Accuracy |   Precision (macro) |   Recall (macro) |   F1 (macro) |   AUC-ROC (macro) |   Latency (ms/sample) |    # Params |
|:------------|-----------:|--------------------:|-----------------:|-------------:|------------------:|----------------------:|------------:|
| Transformer |     0.9990 |              0.9990 |           0.9990 |       0.9990 |            1.0000 |                7.3493 |  67844.0000 |
| GAT         |     0.9969 |              0.9969 |           0.9969 |       0.9969 |            0.9999 |                0.0096 |   5924.0000 |
| WAE-GAN+XGB |     0.9674 |              0.9673 |           0.9674 |       0.9673 |            0.9984 |                0.0179 | 102635.0000 |

---

## 3. Per-Class F1 Score

| Class | Transformer F1 | GAT F1 | WAE-GAN+XGB F1 |
|---|---|---|---|
| Normal (P1) | 1.0000 | 1.0000 | 0.9843 |
| Inner Race (P2) | 0.9979 | 0.9938 | 0.9453 |
| Roller (P3) | 0.9979 | 0.9938 | 0.9500 |
| Outer Race (P4) | 1.0000 | 1.0000 | 0.9898 |

---

## 4. WAE-GAN Binary Anomaly Detection

| Metric | Value |
|---|---|
| AUC-ROC (normal vs any fault) | 1.0000 |
| Average Precision | 1.0000 |
| Threshold (IQR-based) | 0.148121 |

---

## 5. Temporal Consistency (DTW)

Mean DTW distance per run (lower = more temporally consistent):

| Model | Mean DTW |
|---|---|
| Transformer | 0.2500 |
| GAT | 0.7500 |
| WAE-GAN+XGB | 8.8333 |

---

## 6. Error Analysis

- Total test windows: 2916
- Windows with any model disagreement: 106 (3.64%)
- All models agree: 2810 (96.36%)

---

## 7. Final Ranking

Weighted score: **40% F1 + 30% AUC-ROC + 20% inv(DTW) + 10% inv(Latency)**

|    | Model       |   F1 macro |   AUC-ROC |    DTW |   Latency (ms) |   Weighted Score |
|---:|:------------|-----------:|----------:|-------:|---------------:|-----------------:|
|  1 | GAT         |     0.9969 |    0.9999 | 0.7500 |         0.0096 |           0.9523 |
|  2 | Transformer |     0.9990 |    1.0000 | 0.2500 |         7.3493 |           0.9000 |
|  3 | WAE-GAN+XGB |     0.9673 |    0.9984 | 8.8333 |         0.0179 |           0.0999 |

---

## 8. Recommendation

**Recommended model for production: GAT**

### Rationale

- **VibrationTransformer** é um classificador supervisionado de 4 classes que opera diretamente nas janelas brutas. Não requer construção de grafo na inferência, facilitando o deployment. Custo: maior número de parâmetros e latência por amostra em CPU.
- **VibrationGAT** atinge acurácia competitiva com menos parâmetros e inferência mais rápida, ao custo de uma etapa de construção do grafo k-NN na inferência. Ideal para deployment em edge/embedded com restrição de memória.
- **WAE-GAN + XGBoost** oferece tanto scoring de anomalia não-supervisionado (erro de reconstrução) quanto identificação de falha multi-classe (via embeddings → XGBoost). Útil para detectar tipos de falha não vistos no treino. A latência reportada cobre apenas a inferência XGBoost; a codificação pelo encoder WAE-GAN adiciona overhead.

---

## 9. Figures Generated

| File | Description |
|---|---|
| `figures/confusion_matrices.png` | 3-panel confusion matrices (test set) |
| `figures/roc_curves.png` | One-vs-Rest ROC, all 3 models, 4 classes + macro |
| `figures/pr_curves.png` | Precision-Recall curves, all 3 models |
| `figures/wae_anomaly_roc.png` | WAE-GAN binary anomaly ROC with IQR threshold marker |
| `figures/error_analysis.png` | Disagreement rate per class + pairwise counts |
| `figures/dtw_consistency.png` | Mean DTW per model |
