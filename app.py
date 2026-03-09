# app.py
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from PIL import Image
from flask import Flask, render_template, request

# 👉 방금 만든 파일(preprocessing.py)에서 필요한 것들을 가져옵니다.
from preprocessing import data_transforms, classes 

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# 1. 모델 설정 및 로드
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet18()
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 3)
model.load_state_dict(torch.load('resnet_room_clean_model.keras', map_location=device))
model.eval()

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files.get("image")
        if not file:
            return "파일이 없습니다."

        file_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(file_path)

        # 3. AI 판독 프로세스 (가져온 data_transforms 와 classes 를 그대로 사용!)
        img = Image.open(file_path).convert('RGB')
        input_tensor = data_transforms(img).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = F.softmax(outputs, dim=1)[0]
            confidence = torch.max(probabilities).item() * 100
            _, predicted = torch.max(outputs, 1)
            result = classes[predicted.item()]

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