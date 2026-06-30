# Engineering Report: Recaptured Image Detection

**Author:** Candidate (Placement Assignment)  
**Date:** June 30, 2026  
**Target:** SalesCode.ai CV Engineer Role  

---

## 1. Approach Overview

This project implements a classical Computer Vision (CV) and Machine Learning (ML) pipeline to classify whether an image is a **Real Photo** (class `0`) or a **Recaptured Photo of a screen** (class `1`). 

A classical CV + Machine Learning approach was chosen because the dataset is relatively small (102 images) and the assignment preferences favor a small, fast, and lightweight solution. Rather than training a deep learning network, we extract **23 handcrafted features** that target the physical and optical differences between physical screen captures and natural photos (such as color saturation, specular reflections, edge density, micro-texture patterns, moiré frequencies, and high-frequency noise). A constrained Random Forest classifier is trained on these features, yielding a lightweight model of only **145 KB**.

---

## 2. Feature Engineering & Selection

Images are resized to $384 \times 384$ pixels using area interpolation to preserve screen pixel-grid patterns. We extract 23 features across 6 groups:
1. **Luminance & Contrast (2 features)**: Grayscale mean and standard deviation.
2. **Color & Glare (5 features)**: HSV Saturation/Value mean and standard deviation, plus a **Specular Glare Ratio** (percentage of pixels with $S < 30$ and $V > 240$) to identify ambient light reflections on screen glass.
3. **Sharpness & Edges (3 features)**: Laplacian variance (sharpness), Canny edge density (capturing moiré lines and display frames), and Sobel gradient mean (local edge strength).
4. **Texture (8 features)**: 
   * **Local Binary Patterns (LBP)**: Mean, std dev, skewness, and entropy of uniform LBP ($P=8, R=1$) to analyze micro-textures.
   * **Gray-Level Co-occurrence Matrix (GLCM)**: Computed on 16 quantized gray levels. Contrast, correlation, energy, and homogeneity properties are averaged over 4 angles ($0^\circ, 45^\circ, 90^\circ, 135^\circ$) at distance `1` for rotation invariance.
5. **Frequency Domain (4 features)**: FFT spectral entropy and log magnitude statistics (mean mid-frequency, mean high-frequency, and high-frequency energy ratio) to capture moiré patterns.
6. **Noise Estimation (1 feature)**: Immerkær's noise standard deviation using a local Laplacian-like convolution mask.

---

## 3. Model Choice & Rationale

We selected a **Random Forest Classifier** with the following hyperparameters:
`n_estimators=100`, `max_depth=4`, `min_samples_split=5`, `min_samples_leaf=3`, and `max_features="sqrt"`.

**Rationale:**
* **Generalization**: Constraining tree depth and leaf sizes prevents individual trees from overfitting to the small training dataset.
* **Simplicity**: Decision trees handle unscaled features directly, meaning we do not need to save or manage feature scaling parameters in deployment.
* **Footprint**: The model fits in a **145 KB** file and runs quickly on standard CPU environments.

---

## 4. Training Results and Discussion

The model was trained and evaluated on the provided dataset of 102 images (52 real, 50 screen) using a stratified 80/20 train/test split.

* **Measured Holdout Test Accuracy**: **90.48%** (20 out of 21 test images classified correctly).
* **Confusion Matrix**:
  * Actual Real ($N=11$): **10** True Negatives, **1** False Positive.
  * Actual Screen ($N=10$): **10** True Positives, **0** False Negatives.
* **Classification Report**:
  * Real Photo: Precision = $1.00$, Recall = $0.91$, F1-Score = $0.95$.
  * Screen Photo: Precision = $0.91$, Recall = $1.00$, F1-Score = $0.95$.

### Honest Assessment of Limitations:
1. **Dataset Size**: A dataset of 102 images is too small to guarantee global generalization. The obtained holdout test accuracy of 90.48% is a strong initial result, but performance may vary on unseen datasets with different lighting conditions, screen types, and camera resolutions.
2. **False Positives**: The single misclassification (a real photo classified as a screen) indicates that natural textures or reflections can sometimes trigger screen-like features. A larger and more diverse dataset is needed to improve generalization.

---

## 5. Latency & Cost Analysis

Benchmarks were measured on an Apple M2 CPU:
* **Feature Extraction**: 32.24 ms per image.
* **Model Inference**: 1.98 ms per image.
* **Total Prediction Latency**: **34.23 ms per image**.

### Cloud Cost Projection:
If deployed on **AWS Lambda** (1769 MB memory tier, priced at \$0.0000166667 per GB-second):
* Running 1 image takes $0.0342$ seconds.
* Cost per execution = $0.0342 \text{ s} \times 1.769 \text{ GB} \times \$0.0000166667/\text{GB-s} \approx \$0.00000101$ per image.
* This is approximately **\$1 per million images**.

---

## 6. Future Work

To improve accuracy and robustness in future iterations:
1. **Regional Sub-sampling**: Moiré patterns are often localized. Analyzing high-frequency regions at their native resolution instead of downsampling the entire image would prevent loss of fine grid details.
2. **Expand the Dataset**: Collect more diverse images containing different screens, ambient lighting, and angles to reduce the variance of the classifier.
3. **Explore Lightweight CNNs**: If a significantly larger and more diverse dataset becomes available, explore lightweight CNN-based approaches.
