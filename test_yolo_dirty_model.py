#!/usr/bin/env python3
"""
Run the already-trained YOLO-based dirty classifier on a folder of test images
and print, for each image:
  - dirty / clean probability
  - all engineered YOLO feature values used by the model.

Requires:
  - yolo_dirty_model.joblib in the project root (created by train_yolo_dirty_model.py)

Usage:
  python test_yolo_dirty_model.py                # uses ./test_images/normal/ by default
  python test_yolo_dirty_model.py --images PATH  # use a custom folder of images
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import joblib


def _project_root() -> Path:
    return Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser(description="Run YOLO dirty classifier on test images")
    parser.add_argument(
        "--images",
        type=Path,
        default=None,
        help="Folder of test images (default: ./test_images/normal/)",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=None,
        help="Path to yolo_dirty_model.joblib (default: project root)",
    )
    args = parser.parse_args()

    root = _project_root()
    model_path = args.model_path or (root / "yolo_dirty_model.joblib")
    if not model_path.is_file():
        print(f"Model file not found: {model_path}", file=sys.stderr)
        sys.exit(1)

    data = joblib.load(model_path)
    pipe = data.get("model")
    if pipe is None:
        print(f"No 'model' key in {model_path}", file=sys.stderr)
        sys.exit(1)

    images_dir = args.images or (root / "test_images" / "normal")
    if not images_dir.is_dir():
        print(f"Image folder not found: {images_dir}", file=sys.stderr)
        sys.exit(1)

    # Import lazily to avoid loading YOLO if something is misconfigured above.
    from scoring import dirty_scorer

    exts = ("*.jpg", "*.jpeg", "*.png", "*.webp", "*.bmp")
    test_paths = []
    for ext in exts:
        test_paths.extend(images_dir.glob(ext))

    if not test_paths:
        print(f"No images found in {images_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Running on {len(test_paths)} images from {images_dir} using {model_path} ...")

    clf = pipe.named_steps.get("clf", None)
    classes = getattr(clf, "classes_", np.array([0, 1]))

    for path in sorted(test_paths):
        try:
            feat_dict, vec = dirty_scorer.extract_yolo_features(str(path))
            vec = vec.reshape(1, -1)
            proba = pipe.predict_proba(vec)[0]

            # 0 = clean, 1 = dirty during training; handle arbitrary ordering.
            if len(classes) == 2:
                if classes[0] == 0 and classes[1] == 1:
                    p_clean, p_dirty = float(proba[0]), float(proba[1])
                elif classes[0] == 1 and classes[1] == 0:
                    p_clean, p_dirty = float(proba[1]), float(proba[0])
                else:
                    p_clean, p_dirty = float(proba[0]), float(proba[1])
            else:
                # Unexpected multi-class; assume index 0=clean, 1=dirty if present.
                p_clean = float(proba[0])
                p_dirty = float(proba[1]) if proba.shape[0] > 1 else 1.0 - p_clean

            print(f"{path.name}: dirty={p_dirty:.3f}, clean={p_clean:.3f}")
            # Print all engineered feature values on one indented line
            # (order from YOLO_FEATURE_NAMES for consistency).
            feature_str_parts = []
            for name in dirty_scorer.YOLO_FEATURE_NAMES:
                val = feat_dict.get(name, 0.0)
                feature_str_parts.append(f"{name}={val:.3f}")
            # print("  " + ", ".join(feature_str_parts))
        except Exception as e:
            print(f"  Skip {path}: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()

