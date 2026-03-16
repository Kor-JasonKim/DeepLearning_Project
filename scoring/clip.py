"""
CLIP-based room state analysis: reasons (labels) for messiness.
Used by app.py; no ResNet here (Keras model is in app/heatmap).
"""
import os
import numpy as np
import tensorflow as tf
from PIL import Image
from transformers import TFCLIPModel, CLIPProcessor

# --- CLIP model (lazy load to avoid startup cost when not used) ---
_model_name = "openai/clip-vit-base-patch32"
_clip_model = None
_processor = None


def _get_clip():
    global _clip_model, _processor
    if _clip_model is None:
        _clip_model = TFCLIPModel.from_pretrained(_model_name)
        _processor = CLIPProcessor.from_pretrained(_model_name)
    return _clip_model, _processor


# --- Candidate labels and Korean mapping ---
CANDIDATE_LABELS = [
    "messy clothes on the floor",
    "unmade bed with messy sheets",
    "trash and waste",
    "cluttered desk with many items",
    "a clean and organized room",
]

KOREAN_LABELS = {
    "messy clothes on the floor": "바닥의 옷가지",
    "unmade bed with messy sheets": "정리되지 않은 침구",
    "trash and waste": "쓰레기 및 잡동사니",
    "cluttered desk with many items": "어질러진 책상/선반",
    "a clean and organized room": "깨끗한 상태",
}


def get_clip_analysis(img_path):
    """
    Run CLIP on image and return status + reasons (no ResNet).
    Returns: dict with "status" ("✅ 상태 양호" | "🚨 정리 필요") and "reasons" (list of str).
    """
    clip_model, processor = _get_clip()
    raw_img = Image.open(img_path).convert("RGB")

    inputs = processor(
        text=CANDIDATE_LABELS,
        images=raw_img,
        return_tensors="tf",
        padding=True,
    )
    outputs = clip_model(**inputs)
    probs = tf.nn.softmax(outputs.logits_per_image, axis=-1).numpy()[0]

    # All labels in confidence order (desc)
    order = probs.argsort()[::-1]
    all_labels = [(KOREAN_LABELS[CANDIDATE_LABELS[i]], float(probs[i])) for i in order]

    top_idx = int(order[0])
    top_label = CANDIDATE_LABELS[top_idx]
    top_conf = float(probs[top_idx])

    # Status from CLIP: clean label with high prob → 양호
    if top_label == "a clean and organized room" and top_conf >= 0.9:
        return {"status": "✅ 상태 양호", "reasons": None, "all_labels": all_labels}

    # Build reasons: top 2 non-clean labels with conf >= 0.15
    reasons = []
    for idx in order:
        label = CANDIDATE_LABELS[idx]
        conf = probs[idx]
        if label == "a clean and organized room" or conf < 0.15:
            continue
        reasons.append(f"{KOREAN_LABELS[label]}({conf*100:.1f}%)")
        if len(reasons) >= 2:
            break
    if not reasons:
        reasons.append("기타 미분류 잡동사니")

    return {"status": "🚨 정리 필요", "reasons": reasons, "all_labels": all_labels}


# --- Standalone test (folder loop) ---
if __name__ == "__main__":
    test_folder = "imgs"
    print("\n" + "=" * 50)
    print("🧠 CLIP 기반 지능형 방 상태 분석기")
    print("=" * 50)
    if os.path.exists(test_folder):
        for file_name in sorted(os.listdir(test_folder)):
            if file_name.lower().endswith((".jpg", ".jpeg", ".png")):
                path = os.path.join(test_folder, file_name)
                out = get_clip_analysis(path)
                print(f"[{file_name}] {out['status']}")
                if out.get("reasons"):
                    print(f"   ㄴ 감지된 문제: {', '.join(out['reasons'])}")
                print("-" * 30)
