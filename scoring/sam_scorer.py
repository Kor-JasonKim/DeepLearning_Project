"""
SAM-based object segmentation: get object masks + count + optional labels.

This module uses Meta's Segment Anything Model (SAM) to produce instance masks
for all visible objects in a desk image, then (optionally) runs a classifier
on each mask crop to label what each object is (e.g. laptop, cup, book).
"""

from pathlib import Path
from typing import List, Union, Dict, Any, Optional

import numpy as np

_sam_mask_generator = None
_classifier_model = None
_imagenet_categories: Optional[List[str]] = None


# Path to the downloaded SAM checkpoint.
# Download from the official repo and place in the project root:
#   https://github.com/facebookresearch/segment-anything#model-checkpoints
# For example:
#   sam_vit_h_4b8939.pth
SAM_CHECKPOINT_PATH = Path("sam_vit_b_01ec64.pth")
SAM_MODEL_TYPE = "vit_b"  # can be 'vit_h', 'vit_l', or 'vit_b'

# Detection thresholds: resize before SAM and drop small masks (e.g. individual keys).
# - max_image_size: longer side is capped to this (fewer, coarser segments; faster).
# - min_mask_area: masks with area (px²) below this are discarded.
DEFAULT_MAX_IMAGE_SIZE = 1024
DEFAULT_MIN_MASK_AREA = 2500


def _load_image_rgb(image_path_or_array: Union[str, Path, np.ndarray]) -> np.ndarray:
    """Load image as RGB ndarray (H, W, 3)."""
    if isinstance(image_path_or_array, (str, Path)):
        import cv2
        img_bgr = cv2.imread(str(image_path_or_array))
        if img_bgr is None:
            raise ValueError(f"Failed to read image: {image_path_or_array}")
        return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    arr = np.asarray(image_path_or_array)
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError("Expected image array of shape (H, W, 3).")
    return arr


def _resize_image_if_needed(
    image: np.ndarray,
    max_size: int,
) -> np.ndarray:
    """Resize image so the longer side is at most max_size; keep aspect ratio."""
    if max_size is None or max_size <= 0:
        return image
    h, w = image.shape[:2]
    if max(h, w) <= max_size:
        return image
    import cv2
    scale = max_size / max(h, w)
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))
    return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)


def _get_sam_mask_generator():
    """
    Lazy-load SAM and the automatic mask generator.
    """
    global _sam_mask_generator
    if _sam_mask_generator is not None:
        return _sam_mask_generator

    if not SAM_CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            f"SAM checkpoint not found at {SAM_CHECKPOINT_PATH!s}. "
            "Download a SAM model (e.g., sam_vit_h_4b8939.pth) from the "
            "official repo and place it in the project root."
        )

    from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
    import torch

    sam = sam_model_registry[SAM_MODEL_TYPE](
        checkpoint=str(SAM_CHECKPOINT_PATH)
    )
    sam.to("cuda" if torch.cuda.is_available() else "cpu")

    # You can tweak these parameters for your data (desk photos)
    _sam_mask_generator = SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=32,
        pred_iou_thresh=0.86,
        stability_score_thresh=0.92,
        min_mask_region_area=100,  # filter out tiny fragments
    )
    return _sam_mask_generator


def get_sam_masks(
    image_path_or_array: Union[str, Path, np.ndarray],
    max_image_size: Optional[int] = DEFAULT_MAX_IMAGE_SIZE,
    min_mask_area: Optional[int] = DEFAULT_MIN_MASK_AREA,
) -> tuple:
    """
    Run SAM on an image (optionally resized) and return masks above a size threshold.

    Args:
        image_path_or_array: Path to an RGB image file, or (H,W,3) array.
        max_image_size: If set, resize so the longer side is at most this (faster, fewer tiny segments).
        min_mask_area: Discard masks with area (pixels²) below this (e.g. ignore key-sized blobs).

    Returns:
        (masks, image_rgb): list of mask dicts and the image that was fed to SAM (resized if applicable).
    """
    mask_generator = _get_sam_mask_generator()
    image = _load_image_rgb(image_path_or_array)
    image = _resize_image_if_needed(image, max_size=max_image_size or 0)

    masks = mask_generator.generate(image)
    if min_mask_area and min_mask_area > 0:
        masks = [m for m in masks if m.get("area", 0) >= min_mask_area]
    return masks, image


def _get_classifier():
    """Lazy-load ImageNet classifier and category names."""
    global _classifier_model, _imagenet_categories
    if _classifier_model is not None:
        return _classifier_model, _imagenet_categories

    import torch
    from torchvision import models
    from torchvision.models import ResNet50_Weights

    weights = ResNet50_Weights.IMAGENET1K_V1
    _classifier_model = models.resnet50(weights=weights)
    _classifier_model.eval()
    _classifier_model.to("cuda" if torch.cuda.is_available() else "cpu")
    _imagenet_categories = weights.meta["categories"]
    return _classifier_model, _imagenet_categories


def _crop_to_bbox(image: np.ndarray, bbox: List[int], padding: float = 0.1) -> np.ndarray:
    """Crop image to bbox [x, y, w, h] with optional padding. Returns RGB (H, W, 3)."""
    import cv2
    x, y, w, h = bbox
    H, W = image.shape[:2]
    # add padding
    pad_w = max(1, int(w * padding))
    pad_h = max(1, int(h * padding))
    x1 = max(0, x - pad_w)
    y1 = max(0, y - pad_h)
    x2 = min(W, x + w + pad_w)
    y2 = min(H, y + h + pad_h)
    crop = image[y1:y2, x1:x2]
    return crop


def _classify_crop(crop_rgb: np.ndarray, top_k: int = 1) -> List[str]:
    """Run ImageNet classifier on one crop; return list of top-k label strings."""
    import torch
    from torchvision import transforms

    model, categories = _get_classifier()
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])
    tensor = transform(crop_rgb).unsqueeze(0)
    device = next(model.parameters()).device
    tensor = tensor.to(device)
    with torch.no_grad():
        logits = model(tensor)
    probs = logits.softmax(dim=1).cpu().numpy().ravel()
    top_indices = np.argsort(probs)[::-1][:top_k]
    return [categories[i] for i in top_indices]


def get_sam_masks_with_labels(
    image_path_or_array: Union[str, Path, np.ndarray],
    top_k_per_object: int = 1,
    max_image_size: Optional[int] = DEFAULT_MAX_IMAGE_SIZE,
    min_mask_area: Optional[int] = DEFAULT_MIN_MASK_AREA,
) -> tuple:
    """
    Run SAM (on optionally resized image) to get masks above threshold, then classify each crop.

    Returns:
        (labeled_list, image_rgb): list of {mask, bbox, labels} and the image used (for overlay).
    """
    masks, image = get_sam_masks(
        image_path_or_array,
        max_image_size=max_image_size,
        min_mask_area=min_mask_area,
    )

    result = []
    for m in masks:
        bbox = m["bbox"]
        crop = _crop_to_bbox(image, bbox)
        if crop.size == 0:
            labels = ["unknown"]
        else:
            labels = _classify_crop(crop, top_k=top_k_per_object)
        result.append({
            "mask": m,
            "bbox": bbox,
            "labels": labels,
        })
    return result, image


def count_objects_with_sam(
    image_path_or_array: Union[str, Path, np.ndarray],
    classify: bool = True,
    max_image_size: Optional[int] = DEFAULT_MAX_IMAGE_SIZE,
    min_mask_area: Optional[int] = DEFAULT_MIN_MASK_AREA,
) -> Dict[str, Any]:
    """
    Run SAM (on optionally resized image) and return object count; optionally classify each object.

    Args:
        image_path_or_array: Path to image or (H,W,3) RGB array.
        classify: If True, run ImageNet classifier on each mask crop.
        max_image_size: Resize image so longer side is at most this (faster; fewer tiny segments).
        min_mask_area: Discard masks with area (px²) below this (e.g. ignore key-sized blobs).

    Returns:
        dict with:
          - object_count (int): number of masks (objects) found
          - masks (list): raw mask dicts from SAM
          - detected_objects (list of str): one label per object (only if classify=True)
          - objects_with_labels (list of dicts): each has 'bbox', 'labels' (only if classify=True)
          - image_resized (ndarray): image actually used (for overlay; bboxes/masks are in this coords)
    """
    if classify:
        labeled, image_resized = get_sam_masks_with_labels(
            image_path_or_array,
            top_k_per_object=1,
            max_image_size=max_image_size,
            min_mask_area=min_mask_area,
        )
        detected_objects = [item["labels"][0] for item in labeled]
        objects_with_labels = [
            {"bbox": item["bbox"], "labels": item["labels"]}
            for item in labeled
        ]
        masks = [item["mask"] for item in labeled]
        return {
            "object_count": len(masks),
            "masks": masks,
            "detected_objects": detected_objects,
            "objects_with_labels": objects_with_labels,
            "image_resized": image_resized,
        }
    masks, image_resized = get_sam_masks(
        image_path_or_array,
        max_image_size=max_image_size,
        min_mask_area=min_mask_area,
    )
    return {
        "object_count": len(masks),
        "masks": masks,
        "image_resized": image_resized,
    }


def _get_image_bgr(image_path_or_array: Union[str, Path, np.ndarray]) -> np.ndarray:
    """Load image as BGR ndarray (H, W, 3)."""
    import cv2
    if isinstance(image_path_or_array, (str, Path)):
        img = cv2.imread(str(image_path_or_array))
        if img is None:
            raise ValueError(f"Failed to read image: {image_path_or_array}")
        return img
    arr = np.asarray(image_path_or_array)
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError("Expected image array of shape (H, W, 3).")
    if arr.dtype != np.uint8:
        arr = (np.clip(arr, 0, 1) * 255).astype(np.uint8) if arr.max() <= 1.0 else arr.astype(np.uint8)
    if arr.shape[2] == 3:
        return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    return arr


def draw_sam_overlay(
    image_path_or_array: Union[str, Path, np.ndarray],
    result: Dict[str, Any],
    output_path: Union[str, Path],
    draw_masks: bool = True,
    draw_boxes_and_labels: bool = True,
) -> str:
    """
    Draw detected objects on the image and save to output_path.

    Uses result["image_resized"] if present (so bboxes/masks match); otherwise
    loads from image_path_or_array.

    Args:
        image_path_or_array: Input image path or (H,W,3) array (used if no image_resized in result).
        result: Dict from count_objects_with_sam(..., classify=True), containing
                objects_with_labels, optionally masks, and optionally image_resized.
        output_path: Where to save the overlay image (e.g. uploads/sam_overlay_xyz.jpg).
        draw_masks: If True and result has masks, overlay colored masks.
        draw_boxes_and_labels: If True, draw bbox and label per object.

    Returns:
        Filename of the saved image (e.g. sam_overlay_xyz.jpg).
    """
    import cv2

    if result.get("image_resized") is not None:
        # Bboxes/masks are in resized image coords; draw on that image
        img = cv2.cvtColor(result["image_resized"].copy(), cv2.COLOR_RGB2BGR)
    else:
        img = _get_image_bgr(image_path_or_array).copy()
    H, W = img.shape[:2]

    masks = result.get("masks", [])
    objects_with_labels = result.get("objects_with_labels", [])
    if not objects_with_labels and result.get("masks"):
        objects_with_labels = [{"bbox": m["bbox"], "labels": ["object"]} for m in masks]

    # Optional: draw semi-transparent colored masks
    if draw_masks and masks:
        np.random.seed(42)
        for m in masks:
            seg = m.get("segmentation")
            if seg is None or seg.shape[:2] != (H, W):
                continue
            color = np.array(np.random.randint(50, 255, 3), dtype=np.uint8)
            # Blend mask region with color
            img[seg] = (img[seg].astype(np.float32) * 0.5 + color.astype(np.float32) * 0.5).astype(np.uint8)

    # Draw bboxes and labels
    if draw_boxes_and_labels and objects_with_labels:
        for i, obj in enumerate(objects_with_labels):
            bbox = obj.get("bbox", [0, 0, 0, 0])
            labels = obj.get("labels", ["?"])
            label = labels[0] if labels else "?"
            x, y, w, h = [int(v) for v in bbox]
            x1, y1 = max(0, x), max(0, y)
            x2, y2 = min(W, x + w), min(H, y + h)
            color = (0, 200, 100)  # BGR green
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            # Label background and text
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(img, (x1, y1 - th - 10), (x1 + tw + 4, y1), color, -1)
            cv2.putText(
                img, label, (x1 + 2, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2,
            )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), img)
    return output_path.name


if __name__ == "__main__":
    # CLI: python sam_scorer.py image.jpg [--overlay out.jpg] [--max-size 1024] [--min-area 2500]
    import sys

    if len(sys.argv) < 2:
        print("Usage: python sam_scorer.py path/to/image.jpg [--overlay output.jpg] [--max-size N] [--min-area N]")
        raise SystemExit(1)

    img_path = sys.argv[1]
    overlay_path = None
    max_size = DEFAULT_MAX_IMAGE_SIZE
    min_area = DEFAULT_MIN_MASK_AREA
    args = sys.argv[2:]
    for i, a in enumerate(args):
        if a == "--overlay" and i + 1 < len(args):
            overlay_path = args[i + 1]
        elif a == "--max-size" and i + 1 < len(args):
            try:
                max_size = int(args[i + 1])
            except ValueError:
                pass
        elif a == "--min-area" and i + 1 < len(args):
            try:
                min_area = int(args[i + 1])
            except ValueError:
                pass

    print(f"Rescale: max side {max_size}px, min mask area {min_area} px²")
    print("Processing image (SAM + classifier)...")
    summary = count_objects_with_sam(
        img_path,
        classify=True,
        max_image_size=max_size,
        min_mask_area=min_area,
    )
    print(f"Found {summary['object_count']} objects with SAM.")
    if summary.get("detected_objects"):
        print("Detected objects:")
        for i, label in enumerate(summary["detected_objects"], 1):
            print(f"  {i}. {label}")

    if overlay_path and summary.get("objects_with_labels") is not None:
        draw_sam_overlay(img_path, summary, overlay_path)
        print(f"Overlay image saved to: {overlay_path}")

