# app.py (흑백 전처리 통합본)
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from flask import Flask, render_template, request

# ⭐️ 외부 파일(preprocessing.py)에서 불러오지 않고 직접 정의합니다.
classes = ['clean', 'dirty']

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# =========================================================
# 1. Keras 모델 로드 (.keras 포맷)
# =========================================================
model_path = 'resnet_room_clean_model.keras'
model = tf.keras.models.load_model(model_path)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files.get("image")
        if not file:
            return "파일이 없습니다."

        file_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(file_path)

        # =========================================================
        # 2. Keras AI 판독 프로세스 (흑백 전처리 포함)
        # =========================================================
        try:
            # 1) 이미지 로드 및 크기 조정
            img = image.load_img(file_path, target_size=(256, 256))
            img_array = image.img_to_array(img)
            
            # ⭐️ 2) 흑백 변환 후 3채널 복제 (학습 및 검증 코드와 동일하게 맞춤!)
            img_tensor = tf.convert_to_tensor(img_array)
            gray_img = tf.image.rgb_to_grayscale(img_tensor)
            gray_3ch_img = tf.image.grayscale_to_rgb(gray_img)
            
            # 3) 차원 추가 및 ResNet 전처리
            input_tensor = tf.expand_dims(gray_3ch_img, axis=0)
            input_tensor = tf.keras.applications.resnet50.preprocess_input(input_tensor)

            # 4) 모델 예측
            outputs = model.predict(input_tensor, verbose=0)
            
            # 5) 이진 분류(Sigmoid) 결과 해석
            prob_value = outputs[0][0] 

            if prob_value >= 0.5:
                result = "Dirty (더러움)"
                confidence = prob_value * 100
            else:
                result = "Clean (깨끗함)"
                confidence = (1.0 - prob_value) * 100

        except Exception as e:
            return f"이미지 처리 중 에러가 발생했습니다: {e}"

        # 결과 화면
        return f"""
        <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
        <div class="container text-center" style="margin-top: 100px; max-width: 500px;">
            <div class="card p-5 shadow">
                <h2 class="mb-4">판독 결과: <span class="text-primary">{result}</span></h2>
                <h4 class="text-muted mb-4">확신도: {confidence:.1f}%</h4>
                <a href="/" class="btn btn-outline-secondary">다시 검사하기</a>
            </div>
        </div>
        """

    return render_template("index.html")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
