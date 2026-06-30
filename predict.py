import os
import sys
import cv2
import joblib

def main():
    # 1. Parse command-line argument
    if len(sys.argv) < 2:
        print("Error: Image path not provided.", file=sys.stderr)
        print("Usage: python predict.py <path_to_image>", file=sys.stderr)
        print("0.5")  # Fallback probability on input error
        sys.exit(1)

    image_path = sys.argv[1]

    # 2. Check if model.pkl exists
    model_path = "model.pkl"
    if not os.path.exists(model_path):
        print(f"Error: Model file '{model_path}' not found. Please run train.py first.", file=sys.stderr)
        print("0.5")  # Fallback probability on model error
        sys.exit(1)

    # 3. Check if image file exists
    if not os.path.exists(image_path):
        print(f"Error: Image file '{image_path}' not found.", file=sys.stderr)
        print("0.5")  # Fallback probability on missing image
        sys.exit(1)

    # 4. Load the model
    try:
        model = joblib.load(model_path)
    except Exception as e:
        print(f"Error loading model: {str(e)}", file=sys.stderr)
        print("0.5")
        sys.exit(1)

    # 5. Process image and predict
    try:
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error: Could not decode image '{image_path}'. Image may be corrupt.", file=sys.stderr)
            print("0.5")
            sys.exit(1)

        # Import feature extraction logic from train.py to ensure 100% parity
        from train import extract_features

        # Extract features
        features = extract_features(image)

        # Predict probability for class 1 (Screen recaptured photo)
        # model.predict_proba returns probability for both [class 0, class 1]
        probs = model.predict_proba([features])[0]
        prob_screen = probs[1]

        # Output ONLY the single floating-point probability to stdout
        print(f"{prob_screen:.4f}")

    except Exception as e:
        print(f"Error during feature extraction or prediction: {str(e)}", file=sys.stderr)
        print("0.5")
        sys.exit(1)

if __name__ == "__main__":
    main()
