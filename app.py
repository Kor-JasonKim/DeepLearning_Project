import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
# ⭐️ 전처리 함수 이름 충돌을 막기 위해 'as'로 이름을 명확히 구분합니다.
from tensorflow.keras.applications.xception import preprocess_input as xception_preprocess, Xception
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess

from flask import Flask, render_template, request

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# =========================================================
# 1. 모델 로드 (방 모델 & 책상 모델)
# =========================================================
# [방 모델] Kaggle Xception (몸통 + 분류기)
room_head_model = tf.keras.models.load_model('room_model.h5')
room_base_model = Xception(weights='imagenet', include_top=False, pooling='avg', input_shape=(299, 299, 3))

# [책상 모델] 직접 학습한 ResNet50
desk_model = tf.keras.models.load_model('desk_resnet50.keras')

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files.get("image")
        # ⭐️ HTML 폼에서 전송된 '방'인지 '책상'인지 구분하는 값을 받아옵니다. (기본값은 'room')
        target_type = request.form.get("target_type", "room") 
        
        if not file:
            return "파일이 없습니다."

        file_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(file_path)

        # =========================================================
        # 2. 선택된 대상에 따른 AI 판독 프로세스 분기
        # =========================================================
        try:
            if target_type == "room":
                # --- [방 검사 파이프라인 (Xception)] ---
                img = image.load_img(file_path, target_size=(299, 299))
                img_array = image.img_to_array(img)
                input_tensor = np.expand_dims(img_array, axis=0)
                input_tensor = xception_preprocess(input_tensor) # Xception 전처리

                features = room_base_model.predict(input_tensor, verbose=0)
                outputs = room_head_model.predict(features, verbose=0)
                
                # 결과 해석
                if room_head_model.output_shape[-1] == 1:
                    prob_value = outputs[0][0]
                    if prob_value >= 0.5:
                        result = "Dirty (더러움)"
                        confidence = prob_value * 100
                    else:
                        result = "Clean (깨끗함)"
                        confidence = (1.0 - prob_value) * 100
                else:
                    predicted_index = np.argmax(outputs[0])
                    if predicted_index == 0:
                        result = "Clean (깨끗함)"
                        confidence = outputs[0][0] * 100
                    else:
                        result = "Dirty (더러움)"
                        confidence = outputs[0][1] * 100
                        
                target_name = "방"

            elif target_type == "desk":
                # --- [책상 검사 파이프라인 (ResNet50)] ---
                img = image.load_img(file_path, target_size=(224, 224))
                img_array = image.img_to_array(img)
                input_tensor = np.expand_dims(img_array, axis=0)
                input_tensor = resnet_preprocess(input_tensor) # ResNet50 전처리
                
                outputs = desk_model.predict(input_tensor, verbose=0)
                score = outputs[0][0]
                
                # 결과 해석 (Sigmoid 출력 기준)
                if score > 0.5:
                    result = "Dirty (더러움)"
                    confidence = score * 100
                else:
                    result = "Clean (깨끗함)"
                    confidence = (1.0 - score) * 100
                    
                target_name = "책상"

        except Exception as e:
            return f"이미지 처리 중 에러가 발생했습니다: {e}"

        # 결과 화면
        return f"""
        <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
        <div class="container text-center" style="margin-top: 100px; max-width: 500px;">
            <div class="card p-5 shadow">
                <h4 class="mb-3 text-secondary">[{target_name} 검사 결과]</h4>
                <h2 class="mb-4">판독 결과: <span class="text-primary">{result}</span></h2>
                <h4 class="text-muted mb-4">확신도: {confidence:.1f}%</h4>
                <a href="/" class="btn btn-outline-secondary">다시 검사하기</a>
            </div>
        </div>
        """

    return render_template("index.html")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)