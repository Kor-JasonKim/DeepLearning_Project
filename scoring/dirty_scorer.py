"""
YOLO-based dirty score focusing on how objects are spread:
- Same objects spread widely across the image → increases score (dirtier)
- Same objects clustered narrowly → neutral
- Different objects clustered narrowly together → increases score (dirtier)

Returns a single yolo_score in [0, 1] (higher = messier).

When an overlay image is requested, each detection is colored by how much it
contributes to the dirty score: green = cleaner (low contribution), red = dirtier (high contribution).
"""

import numpy as np
from pathlib import Path

# ---------------------------------------------------------------------------
# Config: thresholds and weights
# ---------------------------------------------------------------------------
CONFIDENCE_THRESHOLD = 0.15

# Spread thresholds for "wide" vs "narrow"
WIDE_SPREAD_THRESHOLD = 0.4   # above this per-class spread is considered "wide"
NARROW_GLOBAL_THRESHOLD = 0.5  # below this global spread is considered "narrow"

# Weights for final score:
# yolo_score = w_same_wide * same_wide_factor + w_diff_narrow * diff_narrow_factor
W_SAME_WIDE = 0.6
W_DIFF_NARROW = 0.4

# Feature names for the learned dirty model (order must match feature vector).
# Redundant/weak features removed: global_spread (subsumed by same_wide/diff_narrow logic),
# mean_confidence (lighting/calibration, not placement), frac_classes_multi (correlates with max_class_count).
# Added: floor_band_ratio, max_cell_occupancy, overlap_pile — placement/density proxies.
# Added: messy_* features that only look at a hand-picked subset of COCO classes that typically
#        indicate clutter in a living / personal room (bottle, cup, bowl, laptop, mouse, remote,
#        keyboard, cell phone, book, teddy bear, toothbrush).
YOLO_FEATURE_NAMES = [
    "n_boxes_norm",             # log(1 + n_boxes) / log(1 + 50), cap 1
    "n_classes_norm",           # n_unique_classes / 20, cap 1
    "same_wide_factor",         # same-objects-spread-widely contribution
    "diff_narrow_factor",       # different-objects-clustered-narrowly contribution
    "max_class_count_norm",     # max count of any class / 10, cap 1
    "area_ratio",               # sum(box areas) / image area, cap 1
    "floor_band_ratio",         # area-weighted fraction in bottom band (floor clutter proxy)
    "max_cell_occupancy",       # 4x4 grid; max boxes in one cell / n_boxes (local pile proxy)
    "overlap_pile",             # mean max IoU vs other box (stacked/piled proxy)
    "messy_box_ratio",          # fraction of boxes in messy classes
    "messy_area_ratio",         # fraction of total box area from messy classes
    "messy_floor_band_ratio",   # floor_band_ratio but only for messy classes
]

# Subset of COCO classes that usually behave like "clutter" in an average room.
# 39: bottle, 41: cup, 45: bowl, 63: laptop, 64: mouse, 65: remote,
# 66: keyboard, 67: cell phone, 73: book, 77: teddy bear, 79: toothbrush.
MESSY_CLASS_IDS = {
    39,
    41,
    45,
    63,
    64,
    65,
    66,
    67,
    73,
    75,
    76,
    77,
    78,
    79,
}

_yolo_model = None
_learned_dirty_model = None
_learned_dirty_feature_names = None


def _get_yolo_model():
    """Lazy-load YOLO model once."""
    global _yolo_model
    if _yolo_model is None:
        from ultralytics import YOLO
        _yolo_model = YOLO("yolo26x.pt")
    return _yolo_model


def _compute_spread(boxes_xyxy, width, height):
    """
    Compute spatial spread of box centers, normalized to [0, 1].
    boxes_xyxy: (N, 4) array of x1,y1,x2,y2.
    """
    if boxes_xyxy is None or len(boxes_xyxy) < 2:
        return 0.0
    cx = (boxes_xyxy[:, 0] + boxes_xyxy[:, 2]) / 2
    cy = (boxes_xyxy[:, 1] + boxes_xyxy[:, 3]) / 2
    # Normalized std of centers (relative to image size)
    std_x = np.std(cx) / max(width, 1)
    std_y = np.std(cy) / max(height, 1)
    # Combine and cap to [0, 1]; scale so typical spread ~0.5
    raw = (std_x + std_y) / 2.0
    return min(1.0, raw * 2.0)


def _compute_same_wide_factor(xyxy, cls, width, height):
    """
    Same objects spread widely across the image → positive contribution.
    For each class with at least 2 instances:
        - compute its spread
        - if spread > WIDE_SPREAD_THRESHOLD, it contributes to the factor
    """
    if xyxy is None or len(xyxy) < 2:
        return 0.0

    cls = np.asarray(cls, dtype=int)
    unique_classes, counts = np.unique(cls, return_counts=True)

    per_class_scores = []
    for c, cnt in zip(unique_classes, counts):
        # Need at least 2 instances of the same object type
        if cnt < 2:
            continue
        class_mask = cls == c
        class_boxes = xyxy[class_mask]
        spread_c = _compute_spread(class_boxes, width, height)
        # Only count if it is "wide" enough
        if spread_c <= WIDE_SPREAD_THRESHOLD:
            continue
        # Weight by how many of this object there are (soft cap at 5)
        count_weight = min(1.0, cnt / 5.0)
        per_class_scores.append(spread_c * count_weight)

    if not per_class_scores:
        return 0.0
    return float(np.clip(float(np.mean(per_class_scores)), 0.0, 1.0))


def _compute_diff_narrow_factor(xyxy, cls, width, height):
    """
    Different objects clustered narrowly together → positive contribution.
    Only applies when at least one class has 2+ instances (so single-object
    types like one clock, one chair, one bed do not boost the score).
    """
    if xyxy is None or len(xyxy) < 2:
        return 0.0

    cls = np.asarray(cls, dtype=int)
    unique_classes, counts = np.unique(cls, return_counts=True)
    # Need at least 2 different types and at least one type with 2+ instances
    if len(unique_classes) < 2 or np.max(counts) < 2:
        return 0.0

    global_spread = _compute_spread(xyxy, width, height)
    if global_spread >= NARROW_GLOBAL_THRESHOLD:
        return 0.0

    narrowness = 1.0 - global_spread
    diversity = min(1.0, len(unique_classes) / 5.0)
    return float(np.clip(narrowness * diversity, 0.0, 1.0))


def _box_iou(a, b):
    """IoU of two axis-aligned boxes xyxy."""
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    if x2 <= x1 or y2 <= y1:
        return 0.0
    inter = (x2 - x1) * (y2 - y1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return float(inter / union) if union > 0 else 0.0


def _floor_band_ratio(xyxy, width, height, band_frac=0.35):
    """
    Area-weighted fraction of detections in the bottom band of the image.
    Proxies "stuff on the floor" vs walls/shelves — floor clutter often reads as messier.
    """
    if xyxy is None or len(xyxy) == 0 or height <= 0:
        return 0.0
    # Bottom band: y from (1 - band_frac) * height to height (in pixel space)
    y_cut = (1.0 - band_frac) * float(height)
    areas = (xyxy[:, 2] - xyxy[:, 0]) * (xyxy[:, 3] - xyxy[:, 1])
    areas = np.maximum(areas, 0.0)
    total_area = float(np.sum(areas))
    if total_area <= 0:
        return 0.0
    # Box center y; if center in bottom band, count full area (simple proxy)
    cy = (xyxy[:, 1] + xyxy[:, 3]) / 2.0
    in_band = cy >= y_cut
    band_area = float(np.sum(areas[in_band]))
    return float(np.clip(band_area / total_area, 0.0, 1.0))


def _max_cell_occupancy(xyxy, width, height, grid=4):
    """
    Divide image into grid x grid cells; max boxes in any one cell / n_boxes.
    High value = one region crowded (pile/corner mess) even if global spread is large.
    """
    if xyxy is None or len(xyxy) < 2 or width <= 0 or height <= 0:
        return 0.0
    n = len(xyxy)
    cx = (xyxy[:, 0] + xyxy[:, 2]) / 2.0
    cy = (xyxy[:, 1] + xyxy[:, 3]) / 2.0
    gw = max(width / float(grid), 1e-6)
    gh = max(height / float(grid), 1e-6)
    ix = np.clip((cx / gw).astype(int), 0, grid - 1)
    iy = np.clip((cy / gh).astype(int), 0, grid - 1)
    # Count per cell via 2D bin
    counts = np.bincount(iy * grid + ix, minlength=grid * grid)
    max_count = int(np.max(counts)) if counts.size else 0
    return float(np.clip(max_count / n, 0.0, 1.0))


def _overlap_pile_score(xyxy):
    """
    Mean over boxes of max IoU with any other box — stacked/overlapping proxies.
    Capped to [0, 1]. O(n^2); n is typically small for room scenes.
    """
    if xyxy is None or len(xyxy) < 2:
        return 0.0
    n = len(xyxy)
    # Cap pairs if huge (unlikely)
    if n > 80:
        idx = np.random.RandomState(42).choice(n, size=80, replace=False)
        xyxy = xyxy[idx]
        n = 80
    max_iou_other = []
    for i in range(n):
        best = 0.0
        for j in range(n):
            if i == j:
                continue
            best = max(best, _box_iou(xyxy[i], xyxy[j]))
        max_iou_other.append(best)
    return float(np.clip(np.mean(max_iou_other), 0.0, 1.0))


def _get_raw_yolo_result(image_path_or_array):
    """
    Run YOLO and return (xyxy, conf, cls, width, height) or (None, None, None, W, H).
    Shared by score_image and extract_yolo_features.
    """
    model = _get_yolo_model()
    if isinstance(image_path_or_array, (str, Path)):
        results = model.predict(str(image_path_or_array), conf=CONFIDENCE_THRESHOLD, verbose=False)
    else:
        arr = np.asarray(image_path_or_array)
        if arr.ndim != 3 or arr.shape[2] != 3:
            return None, None, None, 640, 640
        results = model.predict(arr, conf=CONFIDENCE_THRESHOLD, verbose=False)
    if not results or len(results) == 0:
        return None, None, None, 640, 640
    r = results[0]
    boxes = r.boxes
    if boxes is None or len(boxes) == 0:
        if r.orig_shape:
            h, w = r.orig_shape[0], r.orig_shape[1]
            return None, None, None, w, h
        return None, None, None, 640, 640
    xyxy = boxes.xyxy.cpu().numpy()
    conf = boxes.conf.cpu().numpy()
    cls = boxes.cls.cpu().numpy().astype(int)
    if r.orig_shape:
        height, width = r.orig_shape[0], r.orig_shape[1]
    else:
        width = int(xyxy[:, 2].max() - xyxy[:, 0].min()) or 640
        height = int(xyxy[:, 3].max() - xyxy[:, 1].min()) or 640
    return xyxy, conf, cls, width, height


def _zeros_feature_dict():
    """Return feature dict and vector of zeros for no-detection case."""
    d = {k: 0.0 for k in YOLO_FEATURE_NAMES}
    return d, np.zeros(len(YOLO_FEATURE_NAMES), dtype=np.float64)


def _feature_dict_and_vector_from_detections(xyxy, conf, cls, width, height):
    """Build feature dict and 1D vector from YOLO detections (no YOLO call)."""
    if xyxy is None or len(xyxy) == 0:
        return _zeros_feature_dict()

    n_boxes = len(xyxy)
    cls = np.asarray(cls, dtype=int)
    unique_classes, counts = np.unique(cls, return_counts=True)
    n_classes = len(unique_classes)
    same_wide_factor = _compute_same_wide_factor(xyxy, cls, width, height)
    diff_narrow_factor = _compute_diff_narrow_factor(xyxy, cls, width, height)
    max_class_count = int(np.max(counts)) if len(counts) else 0
    areas = (xyxy[:, 2] - xyxy[:, 0]) * (xyxy[:, 3] - xyxy[:, 1])
    image_area = width * height
    area_ratio = min(1.0, float(np.sum(areas)) / max(image_area, 1))

    n_boxes_norm = min(1.0, np.log1p(n_boxes) / np.log1p(50))
    n_classes_norm = min(1.0, n_classes / 20.0)
    max_class_count_norm = min(1.0, max_class_count / 10.0)
    floor_band_ratio = _floor_band_ratio(xyxy, width, height)
    max_cell_occupancy = _max_cell_occupancy(xyxy, width, height)
    overlap_pile = _overlap_pile_score(xyxy)

    # Messy-object-only variants
    messy_mask = np.isin(cls, list(MESSY_CLASS_IDS))
    if np.any(messy_mask):
        messy_box_ratio = float(np.sum(messy_mask)) / float(n_boxes)
        messy_area = float(np.sum(areas[messy_mask]))
        total_area = float(np.sum(areas)) if areas.size else 0.0
        messy_area_ratio = float(np.clip(messy_area / max(total_area, 1.0), 0.0, 1.0))
        messy_floor_band_ratio = _floor_band_ratio(xyxy[messy_mask], width, height)
    else:
        messy_box_ratio = 0.0
        messy_area_ratio = 0.0
        messy_floor_band_ratio = 0.0

    d = {
        "n_boxes_norm": n_boxes_norm,
        "n_classes_norm": n_classes_norm,
        "same_wide_factor": same_wide_factor,
        "diff_narrow_factor": diff_narrow_factor,
        "max_class_count_norm": max_class_count_norm,
        "area_ratio": area_ratio,
        "floor_band_ratio": floor_band_ratio,
        "max_cell_occupancy": max_cell_occupancy,
        "overlap_pile": overlap_pile,
        "messy_box_ratio": messy_box_ratio,
        "messy_area_ratio": messy_area_ratio,
        "messy_floor_band_ratio": messy_floor_band_ratio,
    }
    vec = np.array([d[k] for k in YOLO_FEATURE_NAMES], dtype=np.float64)
    return d, vec


def extract_yolo_features(image_path_or_array):
    """
    Run YOLO and extract a fixed feature vector for training a learned dirty model.
    Use the same dataset (clean/dirty folders) and labels to train a classifier.

    Returns:
        dict: feature names -> values (for inspection).
        np.ndarray: 1D vector in YOLO_FEATURE_NAMES order (for sklearn).
    """
    xyxy, conf, cls, width, height = _get_raw_yolo_result(image_path_or_array)
    return _feature_dict_and_vector_from_detections(xyxy, conf, cls, width, height)


def _load_learned_dirty_model():
    """Lazy-load the learned dirty model (joblib) and feature names if present."""
    global _learned_dirty_model, _learned_dirty_feature_names
    if _learned_dirty_model is not None:
        return _learned_dirty_model is not None
    try:
        import joblib
        model_path = Path(__file__).resolve().parent / "yolo_dirty_model.joblib"
        if not model_path.is_file():
            return False
        data = joblib.load(model_path)
        _learned_dirty_model = data["model"]
        _learned_dirty_feature_names = data.get("feature_names", YOLO_FEATURE_NAMES)
        return True
    except Exception:
        return False


def predict_dirty_from_features(feature_vec):
    """
    Predict dirty probability [0, 1] from a feature vector (same order as YOLO_FEATURE_NAMES).
    Returns None if no learned model is loaded.
    """
    if not _load_learned_dirty_model():
        return None
    vec = np.asarray(feature_vec, dtype=np.float64).reshape(1, -1)
    model = _learned_dirty_model
    # Handle Pipeline (scaler + clf)
    if hasattr(model, "named_steps"):
        model = model.named_steps.get("clf", model)
    if hasattr(model, "predict_proba"):
        proba = _learned_dirty_model.predict_proba(vec)[0]
        classes = getattr(model, "classes_", np.array([0, 1]))
        if len(classes) == 2:
            idx = 1 if classes[1] == 1 else 0
            return float(proba[idx])
        return float(proba[1])
    return float(_learned_dirty_model.predict(vec)[0])


def _categorize_detections(xyxy, cls, width, height):
    """
    Classify each detection as:
        - "same_wide_positive": same objects spread widely across the image
        - "diff_narrow_positive": different objects clustered narrowly together
        - "neutral": does not contribute positively
    Returns a list of category strings aligned with each detection.
    """
    n = 0 if xyxy is None else len(xyxy)
    if n == 0:
        return []

    cls = np.asarray(cls, dtype=int)
    categories = ["neutral"] * n

    # --- Same-objects-wide logic (per-class spread) ---
    unique_classes, counts = np.unique(cls, return_counts=True)
    wide_classes = set()
    for c, cnt in zip(unique_classes, counts):
        if cnt < 2:
            continue
        class_mask = cls == c
        class_boxes = xyxy[class_mask]
        spread_c = _compute_spread(class_boxes, width, height)
        if spread_c > WIDE_SPREAD_THRESHOLD:
            wide_classes.add(c)

    for i, c in enumerate(cls):
        if c in wide_classes:
            categories[i] = "same_wide_positive"

    # --- Different-objects-narrow: only for classes with 2+ instances ---
    global_spread = _compute_spread(xyxy, width, height)
    multi_instance_classes = set(c for c, cnt in zip(unique_classes, counts) if cnt >= 2)
    if (
        global_spread < NARROW_GLOBAL_THRESHOLD
        and len(unique_classes) >= 2
        and len(multi_instance_classes) >= 1
    ):
        for i in range(n):
            if categories[i] == "neutral" and cls[i] in multi_instance_classes:
                categories[i] = "diff_narrow_positive"

    return categories


def _compute_per_detection_contributions(xyxy, cls, conf, width, height, same_wide_factor, diff_narrow_factor):
    """
    Compute each detection's contribution to the final score (before clip).
    Weights are varied so not every box gets the same share:
    - Same-objects-wide: weight by distance from class centroid (farther = more "spread" = higher contribution).
    - Different-objects-narrow: weight by detection confidence (higher conf = higher contribution).
    Returns a list of floats >= 0. Sum over boxes <= yolo_score.
    """
    n = 0 if xyxy is None else len(xyxy)
    if n == 0:
        return []

    cls = np.asarray(cls, dtype=int)
    conf = np.asarray(conf, dtype=float)
    contrib = np.zeros(n, dtype=float)

    # Box centers for distance computation
    cx = (xyxy[:, 0] + xyxy[:, 2]) / 2.0
    cy = (xyxy[:, 1] + xyxy[:, 3]) / 2.0

    unique_classes, counts = np.unique(cls, return_counts=True)

    # --- Same-objects-wide: distribute by distance from class centroid (farther = more contribution) ---
    wide_count = 0
    for cc, cnt in zip(unique_classes, counts):
        if cnt < 2:
            continue
        if _compute_spread(xyxy[cls == cc], width, height) > WIDE_SPREAD_THRESHOLD:
            wide_count += int(np.sum(cls == cc))
    total_same_wide = W_SAME_WIDE * same_wide_factor

    for c, cnt in zip(unique_classes, counts):
        if cnt < 2:
            continue
        class_mask = cls == c
        class_boxes = xyxy[class_mask]
        spread_c = _compute_spread(class_boxes, width, height)
        if spread_c <= WIDE_SPREAD_THRESHOLD:
            continue
        # Class centroid
        c_cx = np.mean(cx[class_mask])
        c_cy = np.mean(cy[class_mask])
        # Distance of each box from centroid (normalized by image diagonal)
        diag = np.sqrt(width ** 2 + height ** 2) or 1.0
        dist = np.sqrt((cx[class_mask] - c_cx) ** 2 + (cy[class_mask] - c_cy) ** 2) / diag
        weights = np.maximum(dist, 1e-6)
        weights = weights / weights.sum()
        # This class gets (n_class / wide_count) of total_same_wide; distribute by distance within class
        class_share = total_same_wide * (int(np.sum(class_mask)) / wide_count) if wide_count else 0
        contrib[class_mask] += weights * class_share

    # --- Different-objects-narrow: only for boxes in a class with 2+ instances ---
    global_spread = _compute_spread(xyxy, width, height)
    multi_instance_classes = set(c for c, cnt in zip(unique_classes, counts) if cnt >= 2)
    if (
        global_spread < NARROW_GLOBAL_THRESHOLD
        and len(unique_classes) >= 2
        and len(multi_instance_classes) >= 1
    ):
        total_diff_narrow = W_DIFF_NARROW * diff_narrow_factor
        # Only boxes in a class that has 2+ instances get this contribution
        multi_mask = np.array([c in multi_instance_classes for c in cls])
        conf_multi = np.where(multi_mask, conf, 0.0)
        conf_sum = np.maximum(conf_multi.sum(), 1e-6)
        weights = np.where(multi_mask, conf_multi / conf_sum, 0.0)
        contrib += weights * total_diff_narrow

    return contrib.tolist()


def _contribution_to_bgr(contribution, max_contribution):
    """
    Map contribution in [0, max_contribution] to BGR color.
    Green = cleaner (low contribution to dirty score).
    Red = dirtier (high contribution to dirty score).
    Linear gradient green -> yellow -> red.
    """
    if max_contribution is None or max_contribution <= 0:
        return (0, 255, 0)  # green (BGR) = no contribution = clean
    t = min(1.0, contribution / max_contribution)
    # BGR: green=(0,255,0), red=(0,0,255). Interpolate: more t -> more red, less green
    b = 0
    g = int(255 * (1.0 - t))
    r = int(255 * t)
    return (b, g, r)


def save_yolo_boxes_only(image_path, yolo_result, out_path):
    """
    Draw YOLO bounding boxes and labels on a transparent background (same size as
    image) and save as PNG. Suitable for use as a toggleable overlay layer.
    """
    import cv2

    img = cv2.imread(str(image_path))
    if img is None:
        return
    orig_h, orig_w = img.shape[:2]

    boxes_xyxy = yolo_result.get("boxes_xyxy") or []
    boxes_names = yolo_result.get("boxes_names") or []
    box_contributions = yolo_result.get("box_contributions") or []
    max_contrib = yolo_result.get("max_contribution") or 0.0

    # Transparent RGBA image (BGRA in OpenCV)
    overlay = np.zeros((orig_h, orig_w, 4), dtype=np.uint8)
    overlay[:, :, 3] = 0  # alpha = 0

    scale = max(orig_w, orig_h) / 640
    thickness = max(1, int(2 * scale))
    font_scale = max(0.4, 0.5 * scale)

    for i, box in enumerate(boxes_xyxy):
        x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
        contrib = box_contributions[i] if i < len(box_contributions) else 0.0
        color = _contribution_to_bgr(contrib, max_contrib)
        color_bgra = (color[0], color[1], color[2], 255)

        cv2.rectangle(overlay, (x1, y1), (x2, y2), color_bgra, thickness)

        name = boxes_names[i] if i < len(boxes_names) else "?"
        ratio = (contrib / max_contrib) if max_contrib > 0 else 0.0
        lbl = f"{name} {ratio * 100:.0f}%"
        (tw, th), _ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
        label_y = max(0, y1 - th - 4)
        cv2.rectangle(
            overlay, (x1, label_y), (x1 + tw + 2, y1), color_bgra, -1
        )
        cv2.putText(
            overlay,
            lbl,
            (x1 + 1, y1 - 3),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    cv2.imwrite(str(out_path), overlay)
    return Path(out_path).name


def score_image(
    image_path_or_array,
    return_detections=False,
    return_overlay_path=None,
    use_learned_model=True,
):
    """
    Compute dirty/messy score from an image using YOLO detection and spatial spread.
    If a learned model (yolo_dirty_model.joblib) exists and use_learned_model is True,
    the returned yolo_score is the learned model's dirty probability; otherwise the
    formula-based score is used.

    Args:
        image_path_or_array: Path to image file (str/Path) or numpy array (H,W,3) BGR or RGB.
        return_detections: If True, include list of detected messy objects in result.
        return_overlay_path: If set, save detection overlay image to this path and include in result.
        use_learned_model: If True and yolo_dirty_model.joblib exists, use it for yolo_score.

    Returns:
        dict with:
            - yolo_score (float): [0, 1], higher = messier.
            - messy_count (int): number of detected messy objects.
            - spread (float): spatial spread [0, 1].
            - detected_objects (list of str): only if return_detections=True.
            - overlay_url (str): only if return_overlay_path was set (filename for URL).
    """
    xyxy, conf, cls, width, height = _get_raw_yolo_result(image_path_or_array)
    if xyxy is None or len(xyxy) == 0:
        return _no_detection_result(return_detections, return_overlay_path)

    model = _get_yolo_model()

    # Collect detected object names (all classes)
    detected_names = []
    for i, c in enumerate(cls):
        name = model.names.get(int(c), "?")
        detected_names.append((name, float(conf[i])))

    # Global spread of all detections (for reference/visualization)
    spread = _compute_spread(xyxy, width, height)

    # New scoring logic:
    # - same_wide_factor: same objects spread widely across the image
    # - diff_narrow_factor: different objects clustered narrowly together
    same_wide_factor = _compute_same_wide_factor(xyxy, cls, width, height)
    diff_narrow_factor = _compute_diff_narrow_factor(xyxy, cls, width, height)

    # Feature dict for UI / learned model (single pass, no extra YOLO inference)
    feature_dict, feature_vec = _feature_dict_and_vector_from_detections(
        xyxy, conf, cls, width, height
    )

    # Per-detection category and contribution for visualization
    categories = _categorize_detections(xyxy, cls, width, height)
    box_contributions = _compute_per_detection_contributions(
        xyxy, cls, conf, width, height, same_wide_factor, diff_narrow_factor
    )
    max_contrib = max(box_contributions) if box_contributions else 0.0

    yolo_score = float(
        np.clip(
            W_SAME_WIDE * same_wide_factor + W_DIFF_NARROW * diff_narrow_factor,
            0.0,
            1.0,
        )
    )
    if use_learned_model:
        learned = predict_dirty_from_features(feature_vec)
        if learned is not None:
            yolo_score = learned

    # Raw box data so callers can redraw / merge with other overlays
    boxes_names = [model.names.get(int(c), "?") for c in cls]

    out = {
        "yolo_score": yolo_score,
        "yolo_features": feature_dict,
        # Legacy fields kept for compatibility but no longer used
        "messy_count": 0,
        "spread": spread,
        "messy_objects": [],
        # Extra debug fields for the new logic
        "same_wide_factor": same_wide_factor,
        "diff_narrow_factor": diff_narrow_factor,
        "box_categories": categories,
        "box_contributions": box_contributions,
        "max_contribution": max_contrib,
        # Raw detection geometry (original image pixel space)
        "boxes_xyxy": xyxy.tolist(),
        "boxes_conf": conf.tolist(),
        "boxes_cls": cls.tolist(),
        "boxes_names": boxes_names,
        "image_width": width,
        "image_height": height,
    }
    if return_detections:
        out["detected_objects"] = list(dict.fromkeys(n for n, _ in detected_names))

    if return_overlay_path:
        try:
            import cv2

            # Base image for drawing
            if isinstance(image_path_or_array, (str, Path)):
                base_img = cv2.imread(str(image_path_or_array))
            else:
                # image_path_or_array is assumed to be RGB; convert to BGR for OpenCV
                base_img = np.asarray(image_path_or_array)
                if base_img.ndim == 3 and base_img.shape[2] == 3:
                    base_img = cv2.cvtColor(base_img, cv2.COLOR_RGB2BGR)

            if base_img is not None:
                img = base_img.copy()

                for i, box in enumerate(xyxy):
                    x1, y1, x2, y2 = box.astype(int)
                    name = model.names.get(int(cls[i]), "?")
                    contrib = box_contributions[i] if i < len(box_contributions) else 0.0
                    color = _contribution_to_bgr(contrib, max_contrib)

                    # Label: name + contribution as percentage of max
                    pct = (100.0 * contrib / max_contrib) if max_contrib > 0 else 0
                    label = f"{name} {pct:.0f}%"

                    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                    cv2.rectangle(img, (x1, max(0, y1 - th - 4)), (x1 + tw + 2, y1), color, -1)
                    cv2.putText(
                        img,
                        label,
                        (x1 + 1, y1 - 3),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255, 255, 255),
                        1,
                        cv2.LINE_AA,
                    )

                cv2.imwrite(str(return_overlay_path), img)
                out["overlay_url"] = Path(return_overlay_path).name
        except Exception:
            pass

    return out


def _no_detection_result(return_detections, return_overlay_path):
    zero_features, _ = _zeros_feature_dict()
    out = {
        "yolo_score": 0.0,
        "yolo_features": zero_features,
        "messy_count": 0,
        "spread": 0.0,
        "messy_objects": [],
        "same_wide_factor": 0.0,
        "diff_narrow_factor": 0.0,
        "box_categories": [],
        "box_contributions": [],
        "max_contribution": 0.0,
        "boxes_xyxy": [],
        "boxes_conf": [],
        "boxes_cls": [],
        "boxes_names": [],
        "image_width": 0,
        "image_height": 0,
    }
    if return_detections:
        out["detected_objects"] = []
    if return_overlay_path:
        out["overlay_url"] = None
    return out
