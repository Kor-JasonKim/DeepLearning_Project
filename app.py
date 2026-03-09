# app.py
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from flask import Flask, render_template, request
from preprocessing import classes, prepare_image_for_keras

# 👉 주의: 파이토치용 'data_transforms'는 케라스에서 쓸 수 없으므로 제거했습니다!
# 대신 'classes' 리스트만 가져옵니다. (예: classes = ['clean', 'dirty'])
from preprocessing import classes 

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# =========================================================
# 1. Keras 모델 로드 (.keras 포맷)
# =========================================================
# 파이토치처럼 껍데기(모델 구조)를 미리 만들 필요 없이, 파일 하나만 부르면 끝납니다!
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
        # 2. Keras AI 판독 프로세스
        # =========================================================
        try:
            input_tensor = prepare_image_for_keras(file_path)

            # 4) 모델 예측 (verbose=0으로 터미널 로그를 깔끔하게 유지)
            outputs = model.predict(input_tensor, verbose=0)
            probabilities = outputs[0]
            
            # 5) 확신도 및 결과 클래스 추출
            confidence = np.max(probabilities) * 100
            predicted_index = np.argmax(probabilities)
            result = classes[predicted_index]

        except Exception as e:
            return f"이미지 처리 중 에러가 발생했습니다: {e}"

        # 결과 화면 (HTML 구조는 그대로 유지했습니다)
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
