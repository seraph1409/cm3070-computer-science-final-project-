# Deep Learning Breast Cancer Detection

**Project Template 3.2 — Final Year Project**  
BSc Computer Science

Binary mammogram classification using CNNs and VGG16 transfer learning on the DDSM dataset.

---

## What This Project Does

This project trains deep learning models to classify mammogram X-rays as either **normal** or **abnormal**. The goal is to investigate whether AI can help detect breast cancer in screening mammography — the same question being explored in the NHS EDITH trial (2025).

Two approaches are compared:
- **CNN from scratch** — 8 progressive experiments, each changing one thing at a time
- **VGG16 transfer learning** — pretrained on ImageNet, fine-tuned in two phases on mammogram data

---

## Results

| Model | Accuracy | AUC | Sensitivity | Specificity |
|---|---|---|---|---|
| Exp 0 — Baseline CNN | 0.907 | 0.904 | 0.534 | 0.963 |
| Exp 1 — Dropout (best CNN) | 0.841 | 0.906 | 0.793 | 0.849 |
| VGG16 Phase 1 — frozen | 0.719 | 0.911 | 0.941 | 0.685 |
| **VGG16 Phase 2 — fine-tuned** | **0.918** | **0.985** | **0.969** | **0.911** |

> **Sensitivity** is the primary clinical metric — it measures how many real abnormal cases the model catches. Missing a cancer is far more dangerous than a false alarm.

---

## Dataset

**DDSM Mammography** — `skooch/ddsm-mammography` on Kaggle

- Binary classification: `0 = Negative (normal)`, `1 = Positive (abnormal)`
- ~86% negative, ~14% positive — significant class imbalance
- Images stored as TFRecords, streamed via `tf.data`
- Training: files 0–3 | Validation + Test: file 4

> **Note:** The DDSM dataset was collected predominantly from White patients in the US in the 1990s. Performance may not generalise to other demographics or modern digital mammography equipment.

To use the dataset, add `skooch/ddsm-mammography` via the Kaggle **Add Data** button before running either notebook.

---

## Notebooks

| Notebook | Description |
|---|---|
| `cnn-from-scratch-8-experiments.ipynb` | 8 ablation experiments — baseline through L2 regularisation |
| `balanced-vgg16.ipynb` | VGG16 two-phase transfer learning |

Both notebooks are designed to run on **Kaggle** with a GPU accelerator.

---

## How to Run

1. Go to [Kaggle](https://www.kaggle.com) and create a free account
2. Upload the notebook you want to run
3. Add the dataset: **Add Data → skooch/ddsm-mammography**
4. Set accelerator to **GPU T4 x2** (Notebook settings → Accelerator)
5. Run all cells

---

## Preprocessing Pipeline

Every image goes through:

1. **CLAHE** — contrast limited adaptive histogram equalisation for local contrast enhancement
2. **Denoising** — `cv2.fastNlMeansDenoising`
3. **Resize** — to 224×224 pixels
4. **Normalisation** — pixel values scaled to [0, 1]
5. **Augmentation** — horizontal flips and small rotations (training only)
6. **3-channel replication** — grayscale copied to RGB for VGG16 compatibility

---

## CNN Experiments Summary

| Experiment | Change |
|---|---|
| Exp 0 | Baseline — two conv blocks, no regularisation |
| Exp 1 | Dropout(0.5) + SpatialDropout2D(0.2) |
| Exp 2 | Gaussian noise + 50/50 oversampling pipeline |
| Exp 3 | Third conv block + skip connection |
| Exp 4 | Dual parallel dense branches |
| Exp 5 | Cosine annealing learning rate schedule |
| Exp 6 | Fourth conv block + pyramid pooling |
| Exp 7 | L2 regularisation (1e-4) on all layers |

Each experiment changes exactly one thing so the effect can be directly attributed.

---

## VGG16 Transfer Learning

**Phase 1 — Feature extraction**
- All VGG16 layers frozen
- Only new classification head trained
- Learning rate: `1e-4`
- Result: AUC 0.911, Sensitivity 0.941

**Phase 2 — Fine-tuning**
- block4 and block5 unfrozen
- Full model trained end-to-end
- Learning rate: `1e-5` (10x lower to protect pretrained weights)
- Result: AUC 0.985, Sensitivity 0.969

The classification head uses `GlobalAveragePooling2D` instead of `Flatten` to reduce parameters and overfitting risk.

---

## Evaluation

Every model is evaluated on the same held-out test set using:
- Confusion matrix
- ROC curve + AUC score
- Precision-recall curve
- Sensitivity and specificity

Model selection uses **AUC-ROC** as the primary criterion, not accuracy, because class imbalance makes accuracy misleading.

---

## Tech Stack

- Python 3
- TensorFlow / Keras
- OpenCV (CLAHE, denoising)
- scikit-learn (metrics)
- matplotlib / seaborn (plots)
- Kaggle (compute + dataset)

---

## Key Findings

- A naive classifier always predicting "normal" gets 86% accuracy — and catches 0 cancers
- Dropout regularisation was the single most effective intervention in the CNN experiments
- Adding architectural complexity (skip connections, parallel paths) often made sensitivity worse
- VGG16 Phase 1 (frozen base) already outperformed every from-scratch model on sensitivity
- Transfer learning (VGG16 Phase 2) improved all four metrics simultaneously — AUC 0.985, Sensitivity 0.969

---

## Limitations

- Dataset limited to White US patients from the 1990s — results may not generalise across demographics
- No lesion-level annotations — models classify whole images, not specific regions
- No external validation — all evaluation uses the same DDSM distribution as training
- Screen-film mammography from the 1990s — may not transfer to modern digital equipment

---

## References

1. Wang, L. (2024). Mammography with deep learning for breast cancer detection. *Frontiers in Oncology*, 14. https://doi.org/10.3389/fonc.2024.1281922
2. Lee, R.S. et al. (2017). A curated mammography data set for use in computer-aided detection and diagnosis research. *Scientific Data*, 4. https://doi.org/10.1038/sdata.2017.177
3. Chollet, F. (2018). *Deep Learning with Python*. Manning.
4. Simonyan, K. & Zisserman, A. (2015). Very deep convolutional networks for large-scale image recognition. *ICLR 2015*. https://arxiv.org/abs/1409.1556
5. McKinney, S.M. et al. (2020). International evaluation of an AI system for breast cancer screening. *Nature*, 577. https://doi.org/10.1038/s41586-019-1799-6
6. Shen, L. et al. (2019). Deep learning to improve breast cancer detection on screening mammography. *Scientific Reports*, 9. https://doi.org/10.1038/s41598-019-48995-4
