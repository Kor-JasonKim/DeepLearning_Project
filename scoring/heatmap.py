import numpy as np
import tensorflow as tf
import cv2
import os


# 모델 경로 — CLI 실행 시에만 로드 (__main__ 블록)
model_path = "resnet_room_clean_model.keras"
model = None

# ---  데이터 전처리 및 Grad-CAM 함수  ---
def get_img_array(img_path, size=(256, 256), use_grayscale=True):
    """Load and preprocess image. use_grayscale=True for room model, False for desk (RGB)."""
    img = tf.keras.preprocessing.image.load_img(img_path, target_size=size)
    array = tf.keras.preprocessing.image.img_to_array(img)
    img_tensor = tf.convert_to_tensor(array)
    if use_grayscale:
        gray_img = tf.image.rgb_to_grayscale(img_tensor)
        img_tensor = tf.image.grayscale_to_rgb(gray_img)
    array = tf.expand_dims(img_tensor, axis=0)
    return tf.keras.applications.resnet50.preprocess_input(array)

def _get_gradcam_layers(full_model):
    """
    Resolve base (ResNet50), last conv, GAP, and Dense layers for Grad-CAM.
    Works for both:
      - room model (Sequential with nested 'resnet50' base)
      - desk model (where 'conv5_block3_out' may live directly on the top-level model)
    Includes fallback: find first submodel and its last Conv2D when name lookup fails.
    """
    base_model = None
    last_conv_layer = None

    # 1) Prefer an explicitly named ResNet50 submodel if present
    try:
        base_model = full_model.get_layer("resnet50")
    except (ValueError, AttributeError):
        base_model = None

    # 2) If conv5_block3_out exists directly on the top-level model, use that
    if last_conv_layer is None:
        for layer in full_model.layers:
            if layer.name == "conv5_block3_out":
                last_conv_layer = layer
                if base_model is None:
                    base_model = full_model
                break

    # 3) Otherwise, search nested layers for conv5_block3_out (room model fallback)
    if last_conv_layer is None and base_model is None:
        for layer in full_model.layers:
            try:
                nested = layer.get_layer("conv5_block3_out")
                base_model = layer
                last_conv_layer = nested
                break
            except (ValueError, AttributeError):
                continue

    # 4) Fallback: first layer that is a Model (submodel), use its last Conv2D as heatmap source
    if base_model is None or last_conv_layer is None:
        for layer in full_model.layers:
            if not isinstance(layer, tf.keras.Model):
                continue
            # Submodel: find last Conv2D (ResNet-style backbone)
            last_conv = None
            for sub in layer.layers:
                if isinstance(sub, tf.keras.layers.Conv2D):
                    last_conv = sub
            if last_conv is not None:
                base_model = layer
                last_conv_layer = last_conv
                break

    if base_model is None or last_conv_layer is None:
        raise ValueError(
            "Could not find ResNet50 base or conv5_block3_out layer in model. "
            "Top-level layer names: %s"
            % ([l.name for l in full_model.layers],)
        )

    # Resolve GAP and Dense by type (works for Sequential: base, GAP, Dense)
    gap_layer = None
    dense_layer = None
    for layer in full_model.layers:
        if layer is base_model or (hasattr(layer, "name") and layer.name == getattr(base_model, "name", None)):
            continue
        if isinstance(layer, tf.keras.layers.GlobalAveragePooling2D):
            gap_layer = layer
        if isinstance(layer, tf.keras.layers.Dense):
            dense_layer = layer
    if gap_layer is None or dense_layer is None:
        try:
            idx_base = full_model.layers.index(base_model)
            if idx_base + 2 < len(full_model.layers):
                gap_layer = gap_layer or full_model.layers[idx_base + 1]
                dense_layer = dense_layer or full_model.layers[idx_base + 2]
        except (ValueError, AttributeError):
            pass
        if gap_layer is None or dense_layer is None:
            raise ValueError("Could not find GAP and Dense layers after base model")

    return base_model, last_conv_layer, gap_layer, dense_layer


def make_gradcam_heatmap(img_array, full_model):
    """
    Compute Grad-CAM for either room or desk model.

    We avoid manually reconstructing the head (GAP/Dense) and instead use the
    original computation graph from full_model so it works for both:
      - simple Sequential([base, GAP, Dense(1)]) room model
      - deeper heads (e.g. GAP -> Dense(128) -> Dense(1)) desk model
    """
    base_model, last_conv_layer, gap_layer, dense_layer = _get_gradcam_layers(full_model)

    x = base_model.output
    passed_base = False
    for layer in full_model.layers:
        if layer is base_model or (
            hasattr(layer, "name") and layer.name == getattr(base_model, "name", None)
        ):
            passed_base = True
            continue
        if not passed_base:
            continue
        x = layer(x)

    # Build a model that maps from the base model input to:
    #   - the last conv feature maps
    #   - the final prediction scalar (recomputed head)
    grad_model = tf.keras.models.Model(
        inputs=[base_model.input],
        outputs=[last_conv_layer.output, x],
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[:, 0]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-10)
    return heatmap.numpy()

# --- 청소 가이드 텍스트 생성 ---
def generate_clean_guide(heatmap, label):
    if label == "Clean":
        return "상태가 아주 좋습니다! 유지 관리만 신경 써주세요."

    h, w = heatmap.shape
    y_center, x_center = np.unravel_index(np.argmax(heatmap), heatmap.shape)
    
    vertical = "침대 위나 책상 상단" if y_center < h * 0.4 else "바닥 구석이나 하단" if y_center > h * 0.7 else "방 중앙 부근"
    horizontal = "왼쪽" if x_center < w * 0.3 else "오른쪽" if x_center > w * 0.7 else "중앙"

    return f"📍 [{vertical} {horizontal}] 구역이 어지럽습니다. 해당 위치를 정리해보세요!"

# --- 시각화 함수 ---
def visualize_single_heatmap(img_path, heatmap, label, confidence):
    import matplotlib.pyplot as plt

    img = cv2.imread(img_path)
    img = cv2.resize(img, (256, 256))
    
    # 가이드 텍스트 생성
    guide_text = generate_clean_guide(heatmap, label)
    
    # 히트맵 처리 및 합성
    heatmap_resized = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap_colored = np.uint8(255 * heatmap_resized)
    heatmap_colored = cv2.applyColorMap(heatmap_colored, cv2.COLORMAP_JET)

    superimposed_img = heatmap_colored * 0.4 + img
    superimposed_img = np.clip(superimposed_img, 0, 255).astype(np.uint8)
    superimposed_img = cv2.cvtColor(superimposed_img, cv2.COLOR_BGR2RGB)

    # 1. 콘솔에 텍스트 결과 출력
    print("\n" + "="*50)
    print(f"판독 파일: {os.path.basename(img_path)}")
    print(f"판정 결과: {label} ({confidence*100:.1f}%)")
    print(f"분석 가이드: {guide_text}")
    print("="*50 + "\n")

    # 2. 히트맵 이미지만 단일 출력
    plt.figure(figsize=(8, 8))
    plt.imshow(superimposed_img)
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()
    plt.close()


def save_merged_overlay(img_path, full_model, yolo_result, out_path):
    """
    Produce a single image at the original image's native resolution that layers:
      1. original image (native resolution, not downscaled)
      2. Grad-CAM heatmap (computed at 256×256 model input, upscaled back to original size, 40% alpha)
      3. YOLO bounding boxes (drawn at original pixel coords, green→red by contribution)

    yolo_result is the dict returned by dirty_scorer.score_image — must include
    boxes_xyxy, boxes_names, box_contributions, max_contribution.
    """
    GRADCAM_INPUT = (full_model.input_shape[1], full_model.input_shape[2])

    # --- Grad-CAM (model must receive input at its expected size) ---
    prepared_img = get_img_array(img_path, size=GRADCAM_INPUT)
    heatmap = make_gradcam_heatmap(prepared_img, full_model)
    prediction = full_model.predict(prepared_img, verbose=0)
    score = float(prediction[0][0])
    label = "Dirty" if score > 0.5 else "Clean"
    confidence = score if score > 0.5 else (1.0 - score)
    guide_text = generate_clean_guide(heatmap, label)

    # --- Load original image at native resolution (BGR) ---
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {img_path}")
    orig_h, orig_w = img.shape[:2]

    # --- Upscale heatmap to original resolution and alpha-blend ---
    heatmap_resized = cv2.resize(heatmap, (orig_w, orig_h))
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
    merged = np.clip(heatmap_colored * 0.4 + img, 0, 255).astype(np.uint8)

    # --- Draw YOLO boxes at their native original-image coordinates (no scaling) ---
    boxes_xyxy = yolo_result.get("boxes_xyxy") or []
    boxes_names = yolo_result.get("boxes_names") or []
    box_contributions = yolo_result.get("box_contributions") or []
    max_contrib = yolo_result.get("max_contribution") or 0.0

    # Scale box thickness / font proportionally so they look reasonable regardless of image size
    scale = max(orig_w, orig_h) / 640
    thickness = max(1, int(2 * scale))
    font_scale = max(0.4, 0.5 * scale)

    for i, box in enumerate(boxes_xyxy):
        x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])

        contrib = box_contributions[i] if i < len(box_contributions) else 0.0
        ratio = (contrib / max_contrib) if max_contrib > 0 else 0.0
        color = (0, int(255 * (1 - ratio)), int(255 * ratio))  # BGR: green → red

        name = boxes_names[i] if i < len(boxes_names) else "?"
        lbl = f"{name} {ratio * 100:.0f}%"

        cv2.rectangle(merged, (x1, y1), (x2, y2), color, thickness)
        (tw, th), _ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
        cv2.rectangle(merged, (x1, max(0, y1 - th - 4)), (x1 + tw + 2, y1), color, -1)
        cv2.putText(
            merged, lbl, (x1 + 1, y1 - 3),
            cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 1, cv2.LINE_AA,
        )

    cv2.imwrite(out_path, merged)
    return {"label": label, "confidence": confidence, "guide_text": guide_text}


def save_heatmap_only(img_path, full_model, out_path, mode="room"):
    """
    Generate Grad-CAM heatmap only (no original image), at the original image's
    resolution, and save as PNG. Suitable for use as a toggleable overlay layer.
    mode: "room" (grayscale input) or "desk" (RGB input), must match model training.
    """
    size = (full_model.input_shape[1], full_model.input_shape[2])
    use_grayscale = mode != "desk"
    prepared_img = get_img_array(img_path, size=size, use_grayscale=use_grayscale)
    heatmap = make_gradcam_heatmap(prepared_img, full_model)
    prediction = full_model.predict(prepared_img, verbose=0)
    score = float(prediction[0][0])
    label = "Dirty" if score > 0.5 else "Clean"
    confidence = score if score > 0.5 else (1.0 - score)
    guide_text = generate_clean_guide(heatmap, label)

    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {img_path}")
    orig_h, orig_w = img.shape[:2]

    heatmap_resized = cv2.resize(heatmap, (orig_w, orig_h))
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    heatmap_colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    # Add alpha channel: low heat = transparent, high heat = opaque (for overlay on base image)
    alpha = (heatmap_resized * 255).astype(np.uint8)
    bgra = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2BGRA)
    bgra[:, :, 3] = alpha
    cv2.imwrite(out_path, bgra)
    return {"label": label, "confidence": confidence, "guide_text": guide_text}


def save_heatmap_overlay(img_path, full_model, out_path):
    """
    Generate Grad-CAM overlay (same pipeline as visualize_single_heatmap) and
    save to out_path so it can be served by Flask. Does not call plt.show().
    """
    size = (full_model.input_shape[1], full_model.input_shape[2])
    prepared_img = get_img_array(img_path, size=size)
    heatmap = make_gradcam_heatmap(prepared_img, full_model)
    prediction = full_model.predict(prepared_img, verbose=0)
    score = float(prediction[0][0])
    label = "Dirty" if score > 0.5 else "Clean"
    confidence = score if score > 0.5 else (1.0 - score)
    guide_text = generate_clean_guide(heatmap, label)

    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {img_path}")
    img = cv2.resize(img, (full_model.input_shape[2], full_model.input_shape[1]))

    heatmap_resized = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap_colored = np.uint8(255 * heatmap_resized)
    heatmap_colored = cv2.applyColorMap(heatmap_colored, cv2.COLORMAP_JET)

    superimposed = heatmap_colored * 0.4 + img
    superimposed = np.clip(superimposed, 0, 255).astype(np.uint8)
    # cv2.imwrite expects BGR
    cv2.imwrite(out_path, superimposed)
    return {"label": label, "confidence": confidence, "guide_text": guide_text}


# --- 실행부 (단일 이미지 테스트) ---
if __name__ == "__main__":
    model = tf.keras.models.load_model(model_path)
    target_image_path = "test_images/normal0026.jpg"
    if os.path.exists(target_image_path):
        prepared_img = get_img_array(target_image_path)
        prediction = model.predict(prepared_img, verbose=0)
        score = prediction[0][0]
        label = "Dirty" if score > 0.5 else "Clean"
        confidence = score if score > 0.5 else (1.0 - score)
        heatmap = make_gradcam_heatmap(prepared_img, model)
        visualize_single_heatmap(target_image_path, heatmap, label, confidence)