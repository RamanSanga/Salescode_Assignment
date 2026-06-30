import os
import time
import sys
import numpy as np
import cv2
import joblib
from tqdm import tqdm
from scipy.stats import skew
from skimage.feature import local_binary_pattern, graycomatrix, graycoprops
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Configuration
DATASET_PATH = "Dataset"
REAL_PATH = os.path.join(DATASET_PATH, "real")
SCREEN_PATH = os.path.join(DATASET_PATH, "screen")
IMAGE_SIZE = (384, 384)


def extract_features(image):
    """
    Extracts a set of 23 handcrafted features from a BGR image.
    The features are designed to capture differences in brightness, color distribution,
    sharpness, micro-texture, spatial co-occurrence, and frequency domain (moiré).
    """
    # 1. Image Preprocessing & Standardization
    h, w = image.shape[:2]
    if (h, w) != IMAGE_SIZE:
        # Use INTER_AREA for downscaling (preserves details without aliasing)
        # Use INTER_CUBIC for upscaling
        interp = cv2.INTER_AREA if min(h, w) > 384 else cv2.INTER_CUBIC
        image = cv2.resize(image, IMAGE_SIZE, interpolation=interp)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    features = []

    # 2. Brightness & Contrast (2 features)
    mean_val = np.mean(gray)
    std_val = np.std(gray)
    features.extend([mean_val, std_val])

    # 3. Color & Specular Glare Statistics (5 features)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    s_channel = hsv[:, :, 1]
    v_channel = hsv[:, :, 2]

    mean_s = np.mean(s_channel)
    std_s = np.std(s_channel)
    mean_v = np.mean(v_channel)
    std_v = np.std(v_channel)

    # Specular Glare Ratio: Saturation < 30 and Value > 240 (normalized count)
    glare_mask = (s_channel < 30) & (v_channel > 240)
    glare_ratio = np.mean(glare_mask)
    features.extend([mean_s, std_s, mean_v, std_v, glare_ratio])

    # 4. Sharpness & Edge Statistics (3 features)
    # Laplacian Variance: measures global sharpness
    lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()

    # Canny Edge Density: ratio of edge pixels to total pixels
    canny = cv2.Canny(gray, 100, 200)
    edge_density = np.mean(canny) / 255.0

    # Sobel Gradient Magnitude Mean: measures average local gradient strength
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    grad_mag = np.sqrt(sobelx**2 + sobely**2)
    mean_grad = np.mean(grad_mag)
    features.extend([lap_var, edge_density, mean_grad])

    # 5. Texture Features: Local Binary Pattern (4 features)
    # Uniform LBP with P=8, R=1 captures micro-textures
    lbp = local_binary_pattern(gray, 8, 1, method='uniform')
    lbp_mean = np.mean(lbp)
    lbp_std = np.std(lbp)

    # Skewness: measures LBP distribution asymmetry
    if lbp_std > 1e-5:
        lbp_skew = skew(lbp.ravel())
    else:
        lbp_skew = 0.0

    # LBP Entropy: measures randomness of uniform micro-patterns
    lbp_hist, _ = np.histogram(lbp, bins=np.arange(11), density=True)
    lbp_hist = lbp_hist + 1e-10  # Avoid division by zero
    lbp_entropy = -np.sum(lbp_hist * np.log2(lbp_hist))
    features.extend([lbp_mean, lbp_std, lbp_skew, lbp_entropy])

    # 6. Texture Features: Gray-Level Co-occurrence Matrix (4 features)
    # Quantize grayscale image to 16 bins to speed up and stabilize GLCM calculation
    gray_quantized = (gray // 16).astype(np.uint8)
    distances = [1]
    angles = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]  # 0, 45, 90, 135 degrees
    glcm = graycomatrix(
        gray_quantized,
        distances=distances,
        angles=angles,
        levels=16,
        symmetric=True,
        normed=True
    )

    # Extract GLCM properties and average across all four angles
    contrast = np.mean(graycoprops(glcm, 'contrast'))
    correlation = np.mean(graycoprops(glcm, 'correlation'))
    energy = np.mean(graycoprops(glcm, 'energy'))
    homogeneity = np.mean(graycoprops(glcm, 'homogeneity'))
    features.extend([contrast, correlation, energy, homogeneity])

    # 7. Frequency Domain (FFT) Statistics (4 features)
    dft = np.fft.fft2(gray)
    dft_shift = np.fft.fftshift(dft)
    mag_spectrum = np.log(np.abs(dft_shift) + 1.0)
    total_mag = np.sum(mag_spectrum)

    # Spectral Entropy: flatness of the frequency spectrum (spikes drop entropy)
    if total_mag > 1e-5:
        prob_spec = mag_spectrum / total_mag
        prob_spec = prob_spec + 1e-10
        spec_entropy = -np.sum(prob_spec * np.log2(prob_spec))
    else:
        spec_entropy = 0.0

    # Concentric frequency radial rings from center coordinate (192, 192)
    cy, cx = IMAGE_SIZE[0] // 2, IMAGE_SIZE[1] // 2
    y_indices, x_indices = np.indices(gray.shape)
    r_coords = np.sqrt((x_indices - cx)**2 + (y_indices - cy)**2)
    max_r = np.sqrt(cy**2 + cx**2)

    # Define mid-frequency and high-frequency radial band masks
    mid_freq_mask = (r_coords >= 0.15 * max_r) & (r_coords < 0.5 * max_r)
    high_freq_mask = (r_coords >= 0.5 * max_r)

    mid_freq_mean = np.mean(mag_spectrum[mid_freq_mask]) if np.any(mid_freq_mask) else 0.0
    high_freq_mean = np.mean(mag_spectrum[high_freq_mask]) if np.any(high_freq_mask) else 0.0
    high_ratio = np.sum(mag_spectrum[high_freq_mask]) / (total_mag + 1e-10)
    features.extend([spec_entropy, mid_freq_mean, high_freq_mean, high_ratio])

    # 8. Noise Estimation (1 feature)
    # Immerkær's fast noise standard deviation estimator
    kernel = np.array([[1, -2, 1],
                       [-2, 4, -2],
                       [1, -2, 1]], dtype=np.float32)
    filtered = cv2.filter2D(gray.astype(np.float32), -1, kernel)
    noise_sigma = np.mean(np.abs(filtered)) * np.sqrt(np.pi / 2.0) / 6.0
    features.append(noise_sigma)

    return features


def load_data_from_dir(directory_path, label):
    """
    Loads all valid images from a directory, extracts features, and returns them with labels.
    """
    X = []
    y = []
    latencies = []

    if not os.path.exists(directory_path):
        print(f"Error: Directory {directory_path} does not exist.", file=sys.stderr)
        return X, y, latencies

    valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}

    # Sort files for deterministic loading
    files = sorted(os.listdir(directory_path))

    for filename in tqdm(files, desc=f"Loading {os.path.basename(directory_path)}"):
        ext = os.path.splitext(filename)[1].lower()
        if ext not in valid_extensions:
            continue

        file_path = os.path.join(directory_path, filename)
        
        try:
            image = cv2.imread(file_path)
            if image is None:
                print(f"\nWarning: Could not read image {file_path}", file=sys.stderr)
                continue

            # Measure feature extraction latency for this image
            start_time = time.perf_counter()
            features = extract_features(image)
            elapsed = time.perf_counter() - start_time

            # Double check feature count correctness
            if len(features) == 23:
                X.append(features)
                y.append(label)
                latencies.append(elapsed)
            else:
                print(f"\nWarning: Feature count mismatch ({len(features)} != 23) for {file_path}", file=sys.stderr)

        except Exception as e:
            print(f"\nWarning: Error processing {file_path}: {str(e)}", file=sys.stderr)
            continue

    return X, y, latencies


def train_and_evaluate():
    print("=== Phase 1: Feature Extraction ===")
    
    start_time = time.perf_counter()
    X_real, y_real, latencies_real = load_data_from_dir(REAL_PATH, 0)
    X_screen, y_screen, latencies_screen = load_data_from_dir(SCREEN_PATH, 1)
    
    X = np.array(X_real + X_screen)
    y = np.array(y_real + y_screen)
    total_load_time = time.perf_counter() - start_time

    if len(X) == 0:
        print("Error: No valid images found in dataset directory.", file=sys.stderr)
        sys.exit(1)

    print(f"\nSuccessfully loaded {len(X)} images:")
    print(f"  - Real photos (Class 0): {len(X_real)}")
    print(f"  - Screen recaptures (Class 1): {len(X_screen)}")
    print(f"  - Features extracted per image: {X.shape[1]}")

    # Latency Stats
    all_latencies = latencies_real + latencies_screen
    avg_extraction_latency = np.mean(all_latencies) * 1000  # in ms
    print(f"  - Average Feature Extraction Latency: {avg_extraction_latency:.2f} ms per image")
    print(f"  - Total Loading & Feature Extraction Time: {total_load_time:.2f} seconds")

    print("\n=== Phase 2: Train/Test Split ===")
    # Stratified split to preserve class ratios
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"Training sample size: {len(X_train)}")
    print(f"Testing sample size:  {len(X_test)}")

    print("\n=== Phase 3: Model Training ===")
    # RandomForest model with constrained parameters to prevent overfitting
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=4,
        min_samples_split=5,
        min_samples_leaf=3,
        max_features="sqrt",
        random_state=42
    )

    t_train_start = time.perf_counter()
    model.fit(X_train, y_train)
    training_time = time.perf_counter() - t_train_start
    print(f"Random Forest trained in {training_time * 1000:.2f} ms")

    print("\n=== Phase 4: Evaluation ===")
    # Predict on testing split
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    class_report = classification_report(
        y_test, y_pred, target_names=["Real Photo", "Screen Photo"]
    )

    print(f"Holdout Test Accuracy: {accuracy * 100:.2f}%")
    print("\nConfusion Matrix:")
    print("                 Predicted Real  Predicted Screen")
    print(f"Actual Real       {conf_matrix[0, 0]:<14} {conf_matrix[0, 1]:<16}")
    print(f"Actual Screen     {conf_matrix[1, 0]:<14} {conf_matrix[1, 1]:<16}")
    
    print("\nClassification Report:")
    print(class_report)

    # Measure inference latency on test set
    t_inf_start = time.perf_counter()
    for row in X_test:
        _ = model.predict_proba(row.reshape(1, -1))[:, 1]
    avg_inf_latency = ((time.perf_counter() - t_inf_start) / len(X_test)) * 1000  # in ms
    print(f"Average model inference latency (without feature extraction): {avg_inf_latency:.4f} ms per image")
    print(f"Total prediction latency (Extraction + Inference): {avg_extraction_latency + avg_inf_latency:.2f} ms per image")

    # Save final model
    joblib.dump(model, "model.pkl")
    print("\nModel successfully saved as model.pkl")


if __name__ == "__main__":
    train_and_evaluate()