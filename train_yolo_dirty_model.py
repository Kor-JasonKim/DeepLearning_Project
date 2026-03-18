#!/usr/bin/env python3
"""
Train a small classifier that predicts clean/dirty from YOLO-derived features.

Workflow:
  1. Dataset: folder with subfolders 'clean' and 'dirty', each containing room images.
  2. For each image: run YOLO (via dirty_scorer.extract_yolo_features), get feature vector.
  3. Label: 0 = clean, 1 = dirty (from folder name).
  4. Train LogisticRegression or MLPClassifier; save model + feature names to yolo_dirty_model.joblib.
     Retrain after changing dirty_scorer.YOLO_FEATURE_NAMES (dimension mismatch otherwise).
  5. (Optional) For a separate folder of test images, run the trained model and
     print per-image probabilities of being dirty/clean.

Usage:
  python train_yolo_dirty_model.py [--dataset PATH] [--test-set PATH] [--model {logistic|mlp}]

Default dataset: ./my_room_dataset  (expected: my_room_dataset/clean/, my_room_dataset/dirty/)
Default test images: ./test_images/normal/  (if it exists), used to print per-image probabilities.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Import after path is set so we resolve project root
def _project_root():
    return Path(__file__).resolve().parent

def _collect_images_and_labels(root: Path, class_to_idx: dict):
    """Collect (image_path, label_index) from root/clean and root/dirty."""
    pairs = []
    for class_name, idx in class_to_idx.items():
        folder = root / class_name
        if not folder.is_dir():
            continue
        for ext in ("*.jpg", "*.jpeg", "*.png", "*.webp", "*.bmp"):
            for path in folder.glob(ext):
                pairs.append((str(path), idx))
    return pairs

def main():
    parser = argparse.ArgumentParser(description="Train YOLO-based dirty classifier")
    parser.add_argument("--dataset", type=Path, default=None,
                        help="Root folder with 'clean' and 'dirty' subfolders (default: ./my_room_dataset)")
    parser.add_argument("--test-set", type=Path, default=None,
                        help="Optional folder of unlabeled test images (default: ./test_images/normal/). "
                             "If present, per-image dirty/clean probabilities are printed after training.")
    parser.add_argument("--model", choices=["logistic", "mlp"], default="logistic",
                        help="Classifier: logistic or mlp")
    parser.add_argument("--out", type=Path, default=None,
                        help="Output path for .joblib (default: project root / yolo_dirty_model.joblib)")
    args = parser.parse_args()

    root = _project_root()
    dataset_path = args.dataset or (root / "train2")
    if not dataset_path.is_dir():
        print(f"Dataset not found: {dataset_path}", file=sys.stderr)
        sys.exit(1)

    # Class mapping: clean=0, dirty=1 (match typical ImageFolder)
    class_to_idx = {"clean": 0, "dirty": 1}
    train_pairs = _collect_images_and_labels(dataset_path, class_to_idx)
    if not train_pairs:
        print("No images found in dataset (expected clean/ and dirty/ subfolders).", file=sys.stderr)
        sys.exit(1)

    print(f"Training images: {len(train_pairs)} from {dataset_path}")

    # Import here so YOLO loads only when needed
    from scoring import dirty_scorer

    X_list = []
    y_list = []
    for i, (path, label) in enumerate(train_pairs):
        if (i + 1) % 100 == 0 or i == 0:
            print(f"  Extracting features {i+1}/{len(train_pairs)} ...")
        try:
            _, vec = dirty_scorer.extract_yolo_features(path)
            X_list.append(vec)
            y_list.append(label)
        except Exception as e:
            print(f"  Skip {path}: {e}", file=sys.stderr)

    X = np.array(X_list, dtype=np.float64)
    y = np.array(y_list, dtype=np.int32)
    print(f"Feature matrix shape: {X.shape}, labels: {y.shape}")

    if args.model == "logistic":
        clf = LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced")
    else:
        clf = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42)

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", clf),
    ])
    pipe.fit(X, y)

    train_acc = (pipe.predict(X) == y).mean()
    print(f"Train accuracy: {train_acc:.4f}")

    out_path = args.out or (root / "yolo_dirty_model.joblib")
    joblib.dump({
        "model": pipe,
        "feature_names": dirty_scorer.YOLO_FEATURE_NAMES,
        "classes": np.array([0, 1]),
    }, out_path)
    print(f"Saved: {out_path}")

    # Optional: run on separate test images and print per-image probabilities.
    # If --test-set is not given, look for ./test_images/normal/ by default.
    test_dir = args.test_set or (root / "test_images" / "normal")
    if test_dir and test_dir.is_dir():
        exts = ("*.jpg", "*.jpeg", "*.png", "*.webp", "*.bmp")
        test_paths = []
        for ext in exts:
            test_paths.extend(test_dir.glob(ext))
        if test_paths:
            print(f"Running on {len(test_paths)} test images in {test_dir} ...")
            for path in sorted(test_paths):
                try:
                    _, vec = dirty_scorer.extract_yolo_features(str(path))
                    vec = vec.reshape(1, -1)
                    proba = pipe.predict_proba(vec)[0]
                    # Labels were 0=clean, 1=dirty during training.
                    # For safety, handle arbitrary class ordering.
                    clf = pipe.named_steps.get("clf", None)
                    classes = getattr(clf, "classes_", np.array([0, 1]))
                    if len(classes) == 2:
                        if classes[0] == 0 and classes[1] == 1:
                            p_clean, p_dirty = float(proba[0]), float(proba[1])
                        elif classes[0] == 1 and classes[1] == 0:
                            p_clean, p_dirty = float(proba[1]), float(proba[0])
                        else:
                            # Unexpected labels; fall back to proba[0] as clean, proba[1] as dirty.
                            p_clean, p_dirty = float(proba[0]), float(proba[1])
                    else:
                        # Multi-class (unexpected here); treat index 0 as clean, 1 as dirty if available.
                        p_clean = float(proba[0])
                        p_dirty = float(proba[1]) if proba.shape[0] > 1 else 1.0 - p_clean
                    print(f"{path.name}: dirty={p_dirty:.3f}, clean={p_clean:.3f}")
                except Exception as e:
                    print(f"  Skip test image {path}: {e}", file=sys.stderr)

if __name__ == "__main__":
    main()
