# scoring.py - AI 모델 로딩 및 추론 (Keras, YOLO, CLIP)
import logging
import os
import uuid
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)

import tensorflow as tf

tf.config.set_visible_devices([], "GPU")

from tensorflow.keras.preprocessing import image
from . import dirty_scorer
from .heatmap import save_heatmap_only
from .clip import get_clip_analysis

# 템플릿(예: result.html)에서 직접 참조할 수 있도록 재노출
YOLO_FEATURE_NAMES = dirty_scorer.YOLO_FEATURE_NAMES

# =========================================================
# Keras model paths and loading
# =========================================================
MODEL_PATHS = {
    "room": "resnet_room_clean_model.keras",
    "desk": "desk_resnet50_mixup_best.keras",
}

_model_room = None
_model_desk = None


def _get_models():
    global _model_room, _model_desk
    if _model_room is None:
        _model_room = tf.keras.models.load_model(MODEL_PATHS["room"])
    if _model_desk is None:
        _model_desk = tf.keras.models.load_model(MODEL_PATHS["desk"])
    return {"room": _model_room, "desk": _model_desk}


def run_keras_score(file_path, model, mode: str):
    """Return sigmoid output [0,1] from ResNet (same semantics as before)."""
    # For desk scoring, always resize to 224x224 just for inference.
    # The original image file is still used elsewhere (YOLO, CLIP, and final display).
    if mode == "desk":
        target_h, target_w = 224, 224
    else:
        target_h, target_w = model.input_shape[1], model.input_shape[2]

    img = image.load_img(file_path, target_size=(target_h, target_w))
    img_array = image.img_to_array(img)
    img_tensor = tf.convert_to_tensor(img_array)
    # Desk model uses RGB directly; room model keeps existing grayscale pipeline.
    if mode == "desk":
        input_tensor = tf.expand_dims(img_tensor, axis=0)
    else:
        gray_img = tf.image.rgb_to_grayscale(img_tensor)
        gray_3ch_img = tf.image.grayscale_to_rgb(gray_img)
        input_tensor = tf.expand_dims(gray_3ch_img, axis=0)
    input_tensor = tf.keras.applications.resnet50.preprocess_input(input_tensor)
    outputs = model.predict(input_tensor, verbose=0)
    return float(outputs[0][0])


def run_yolo_score(file_path, overlay_path):
    """Run YOLO dirty scorer once; returns result dict (single YOLO inference)."""
    return dirty_scorer.score_image(
        file_path,
        return_detections=True,
        return_overlay_path=overlay_path,
    )


def run_all_analyses(file_path, mode, upload_folder, original_filename):
    """
    Run Keras, YOLO, and CLIP analyses in parallel. Save heatmap and YOLO boxes overlays.
    Returns a dict with all values needed to render the result template (paths as filenames
    for URL building in the app).
    """
    if mode not in ("room", "desk"):
        mode = "room"
    models = _get_models()
    model = models[mode]

    base, ext = os.path.splitext(original_filename)
    suffix = uuid.uuid4().hex[:8]
    overlay_filename = f"yolo_overlay_{base}_{suffix}{ext}"
    overlay_path = os.path.join(upload_folder, overlay_filename)
    heatmap_only_filename = f"keras_heatmap_only_{base}_{suffix}.png"
    heatmap_only_path = os.path.join(upload_folder, heatmap_only_filename)
    yolo_boxes_filename = f"yolo_boxes_{base}_{suffix}.png"
    yolo_boxes_path = os.path.join(upload_folder, yolo_boxes_filename)

    keras_probability = None
    yolo_result = None
    clip_result = None
    error_keras = None
    error_yolo = None
    error_clip = None
    keras_heatmap_guide = None
    heatmap_saved = False
    yolo_boxes_saved = False

    def task_keras():
        return run_keras_score(file_path, model, mode)

    def task_yolo():
        return run_yolo_score(file_path, overlay_path)

    def task_clip():
        return get_clip_analysis(file_path)

    with ThreadPoolExecutor(max_workers=3) as executor:
        f_keras = executor.submit(task_keras)
        f_yolo = executor.submit(task_yolo)
        f_clip = executor.submit(task_clip)
        try:
            keras_probability = f_keras.result()
            try:
                meta = save_heatmap_only(file_path, model, heatmap_only_path, mode=mode)
                keras_heatmap_guide = meta.get("guide_text")
                heatmap_saved = True
            except Exception as e:
                logger.warning("Heatmap generation failed: %s", e, exc_info=True)
        except Exception as e:
            error_keras = str(e)
        try:
            yolo_result = f_yolo.result()
        except Exception as e:
            error_yolo = str(e)
        try:
            clip_result = f_clip.result()
        except Exception as e:
            error_clip = str(e)

    if yolo_result is not None:
        try:
            dirty_scorer.save_yolo_boxes_only(file_path, yolo_result, yolo_boxes_path)
            yolo_boxes_saved = True
        except Exception:
            pass

    # Combined total score
    yolo_score = yolo_result.get("yolo_score") if yolo_result is not None else None
    prob_for_legacy = (
        keras_probability
        if keras_probability is not None
        else (yolo_score if yolo_score is not None else 0.0)
    )
    total_score = None
    if keras_probability is not None and yolo_score is not None:
        k = max(0.0, min(1.0, float(keras_probability)))
        y = max(0.0, min(1.0, float(yolo_score)))
        # t는 Keras 점수(k)가 극값(0 또는 1)에 가까울수록 커지는 값
        # t에 따라 Keras 가중치(w_k)는 0.8<->0.9, YOLO 가중치(w_y)는 0.2<->0.1
        # k가 확실할수록(극값에 가까울수록) Keras 점수에 더 많이, 애매할수록(극값에서 멀어질수록) YOLO 점수에 더 많이 의존하게 된다
        t = (2.0 * k - 1.0) ** 2
        w_k = 0.8 + 0.1 * t
        w_y = 0.2 - 0.1 * t
        total_score = w_k * k + w_y * y
    elif keras_probability is not None:
        total_score = keras_probability
    elif yolo_score is not None:
        total_score = yolo_score

    return {
        "keras_probability": keras_probability,
        "error_keras": error_keras,
        "yolo_result": yolo_result,
        "error_yolo": error_yolo,
        "clip_result": clip_result,
        "error_clip": error_clip,
        "heatmap_only_filename": heatmap_only_filename if heatmap_saved else None,
        "keras_heatmap_guide": keras_heatmap_guide,
        "yolo_boxes_filename": yolo_boxes_filename if yolo_boxes_saved else None,
        "total_score": total_score,
        "prob_for_legacy": prob_for_legacy,
    }

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="Path to input image")
    parser.add_argument("--mode", choices=["room", "desk"], default="room")
    parser.add_argument("--out", required=True, help="Output folder")
    args = parser.parse_args()

    from pathlib import Path
    img_path = Path(args.image)
    result = run_all_analyses(
        file_path=str(img_path),
        mode=args.mode,
        upload_folder=args.out,
        original_filename=img_path.name,
    )
    print(result)