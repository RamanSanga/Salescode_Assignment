# Recaptured Image Detection (Real vs. Screen Photo)

This repository contains a lightweight classical computer vision solution to classify whether an image is a **Real Photo** (class `0`) or a **Photo of a screen** (class `1`, recaptured image). 

Instead of using large deep learning models, this project uses **23 handcrafted spatial and frequency features** paired with a Random Forest classifier. This keeps the solution fast, lightweight, and easy to run on standard CPU environments.

---

## Folder Structure

```text
Salescode_Assignment/
├── Dataset/
│   ├── real/             # 52 Real Photos (Class 0)
│   └── screen/           # 50 Screen Recaptured Photos (Class 1)
├── train.py              # Feature extraction, training, and evaluation script
├── predict.py            # Command-line prediction script
├── requirements.txt      # Project dependencies
├── README.md             # Project documentation (this file)
├── report.md             # Summary report with training results
├── model.pkl             # Trained Random Forest classifier (145 KB)
└── .gitignore            # Git exclusion rules
```

---

## Installation & Setup

Developed and tested using Python 3.13 on macOS (Apple Silicon M2).

1. Navigate to the project directory:
   ```bash
   cd Salescode_Assignment
   ```

2. Set up a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate
   ```

3. Install requirements:
   ```bash
   pip install -r requirements.txt
   ```

---

## Feature Engineering

Images are resized to $384 \times 384$ pixels using area interpolation (`cv2.INTER_AREA`) to preserve screen grid lines, and 23 features are extracted:

1. **Luminance & Contrast** (2 features)
   * **Grayscale Mean**: Measures overall brightness.
   * **Grayscale Standard Deviation**: Measures global image contrast.

2. **Color & Glare** (5 features)
   * **HSV Saturation & Value (Mean & Std Dev)**: Helps detect differences in color saturation and brightness distribution typical of screens.
   * **Specular Glare Ratio**: Calculates the percentage of pixels with saturation $S < 30$ and value $V > 240$ to capture light reflections on screen glass.

3. **Sharpness & Edges** (3 features)
   * **Laplacian Variance**: Evaluates image sharpness to spot blur or focus mismatches.
   * **Canny Edge Density**: Ratio of edge pixels to total pixels, capturing moiré patterns or display edges.
   * **Sobel Gradient Mean**: Average magnitude of Sobel gradients, measuring local edge strength.

4. **Texture** (8 features)
   * **Local Binary Patterns (LBP)**: Computes uniform LBP ($P=8, R=1$) and extracts the mean, std dev, skewness, and entropy to represent micro-texture patterns.
   * **Gray-Level Co-occurrence Matrix (GLCM)**: Grayscale is quantized to 16 levels. Contrast, correlation, energy, and homogeneity properties are computed at distance `1` and averaged across four angles ($0^\circ, 45^\circ, 90^\circ, 135^\circ$) for rotation invariance.

5. **Frequency Domain (FFT)** (4 features)
   * **Spectral Entropy**: Captures frequency spectrum flatness. Moiré grid lines tend to form sharp frequency spikes, which lowers the entropy.
   * **Radial Averages**: Average frequency magnitude in the concentric zones: **Mid-Frequency Mean** (distance $0.15\times$ to $0.5\times$ max radius) and **High-Frequency Mean** (distance $0.5\times$ to $1.0\times$ max radius).
   * **High-Frequency Energy Ratio**: Total high-frequency energy divided by total spectrum energy.

6. **Noise Estimation** (1 feature)
   * **Immerkær Noise Standard Deviation**: A classic noise estimator using a Laplacian-like operator mask to measure high-frequency noise.

---

## Classifier Configuration

We use a `RandomForestClassifier` with constraints to prevent overfitting on the small dataset:
* `n_estimators=100`: Stable ensemble size.
* `max_depth=4`: Restricts tree depth to avoid memorizing specific training images.
* `min_samples_split=5` and `min_samples_leaf=3`: Ensures nodes split and terminate only on groups of samples.
* `max_features="sqrt"`: Limits features evaluated at each split.

Feature scaling is not required, keeping the inference script simple.

---

## Training & Evaluation

To train the model and evaluate it on a holdout test split (80% train, 20% test, stratified):

```bash
python train.py
```

### Performance on Holdout Test Split
* **Test Accuracy**: **90.48%** (20 out of 21 test images classified correctly).
* **Confusion Matrix**:
  ```text
                   Predicted Real  Predicted Screen
  Actual Real       10             1
  Actual Screen     0              10
  ```
* **Precision & Recall**:
  * Real Photo: Recall = **91.0%**, Precision = **100.0%**
  * Screen Photo: Recall = **100.0%**, Precision = **91.0%**

---

## Prediction Interface

Run prediction on a single image file:

```bash
python predict.py <path_to_image>
```

The script prints **ONLY** one floating-point probability between 0 and 1 to `stdout`.

### Examples:
```bash
$ python predict.py Dataset/real/WhatsApp_Image_1.jpeg
0.0870

$ python predict.py Dataset/screen/WhatsApp_Image_2.jpeg
0.7471
```

* **Stdout**: Contains only the probability value.
* **Stderr**: Prints warning or error logs. If a file is corrupt or cannot be read, the script outputs a default value of `0.5` to stdout to avoid crashing.

---

## Latency & Cost Analysis

Benchmarks measured on Apple Silicon M2 (CPU):

| Metric | Measured Value |
| :--- | :--- |
| **Feature Extraction Latency** | 32.24 ms / image |
| **Model Inference Latency** | 1.98 ms / image |
| **Total Prediction Latency** | **34.23 ms / image** |
| **Model Disk Size** | **145 KB** |
| **Estimated Compute Cost** | **~\$1 per million images** (AWS Lambda) |

### Compute Cost Estimate
If deployed on **AWS Lambda** (1769 MB memory tier, priced at \$0.0000166667 per GB-second):
* Running 1 image takes $0.0342$ seconds.
* Cost per execution = $0.0342 \text{ s} \times 1.769 \text{ GB} \times \$0.0000166667/\text{GB-s} \approx \$0.00000101$ per image.
* This is approximately **\$1 per million images**.

---

## Limitations

1. **Small Dataset**: The model was trained and evaluated on 102 images (52 real, 50 screen). Performance may vary under unseen lighting conditions, screen types, or camera sensors.
2. **Generalization**: While the holdout test accuracy is 90.48%, accuracy on a larger, hidden test set might be lower if those images contain environments or screen resolutions not represented in the training set.
3. **Texture Confusions**: High-frequency real-world textures or direct light glare spots can sometimes resemble screen artifacts.

---

## Notes

This project was developed as a student take-home assignment for the SalesCode.ai campus placement. The primary focus was on building a simple, explainable, and maintainable classical machine learning solution rather than using large deep learning models.
# Salescode_Assignment
