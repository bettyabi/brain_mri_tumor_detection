# Brain MRI Tumor Detection with Deep Learning

Binary classification of brain MRI scans (**tumor vs. no tumor**) using PyTorch.
I compare two CNNs built from scratch with transfer learning on EfficientNet-B0, tune the best model,
and use explainable AI (Grad-CAM, Integrated Gradients, SHAP) to inspect where the model looks.

📓 **[View the notebook](brain_mri_tumor_detection.ipynb)**

## Results

Final model: **hyperparameter-tuned EfficientNet-B0**, evaluated on a held-out test set (n = 34).

| Accuracy | Sensitivity (recall) | Specificity | ROC-AUC |
|---|---|---|---|
| **94.1%** | **100%** (21/21 tumors detected) | 84.6% | 0.978 |

| # | Model | Accuracy | Sensitivity | Specificity | ROC-AUC |
|---|---|---|---|---|---|
| 1 | Scratch CNN (4 conv blocks) | 67.6% | 95.2% | 23.1% | 0.729 |
| 2 | Smaller scratch CNN (3 conv blocks) | 70.6% | 85.7% | 46.2% | 0.656 |
| 3 | EfficientNet-B0, ImageNet preprocessing | 88.2% | 85.7% | 92.3% | 0.974 |
| 4 | EfficientNet-B0, official preprocessing | 91.2% | 90.5% | 92.3% | 0.978 |
| 5 | **EfficientNet-B0, hyperparameter-tuned** | **94.1%** | **100%** | **84.6%** | **0.978** |

In tumor screening, a missed tumor (false negative) is the most costly error, so sensitivity is the key metric.
The tuned model caught every tumor in the test set.

## Pipeline

1. **Data inspection:** integrity checks on all 253 images, plus image sizes, color modes and class balance
2. **Data cleaning:** removed 44 exact and near-duplicate images (MD5 + perceptual hashing) *before* splitting, to prevent train/test leakage
3. **Preprocessing:** stratified 70/15/15 split, RGB conversion, 224×224 resize, ImageNet normalization, light augmentation
4. **Modeling:** scratch CNNs, then EfficientNet-B0 with two-phase transfer learning (frozen backbone, then fine-tuning)
5. **Tuning:** 48-configuration grid search (batch size, dropout, learning rate, weight decay, unfrozen blocks), selected on validation loss
6. **Explainability:** Grad-CAM, Integrated Gradients and SHAP on true/false positives and negatives

## Tech stack

Python · PyTorch · torchvision · scikit-learn · Pillow · imagehash · Captum · SHAP · Matplotlib

## How to run

1. Download the dataset zip from Kaggle:
   [Brain MRI Images for Brain Tumor Detection](https://www.kaggle.com/datasets/navoneel/brain-mri-images-for-brain-tumor-detection) (Navoneel Chakrabarty)
2. Place `brain-mri-images-for-brain-tumor-detection.zip` in the same folder as the notebook.
3. Install the dependencies:
   ```bash
   pip install torch torchvision scikit-learn pillow imagehash captum "shap<0.50" tensorflow matplotlib pandas
   ```
4. Run the notebook from top to bottom. It was developed in Google Colab, but it runs in any Jupyter environment.

## Limitations and next steps

- **Small test set:** 34 images, so each misclassified image changes accuracy by about 3 percentage points. Cross-validation would give more reliable estimates.
- **Explainability (known issue):** the XAI section currently uses a model that was not loaded with the tuned weights. Next step: save the tuned weights after Model 5, reload them, and re-run Grad-CAM, IG and SHAP.
- **Not a diagnostic tool:** this is a learning project. Labels come from the public dataset and were not verified by a radiologist.
