## AI Room & Desk Cleanliness Scorer

AI 방/책상 청결도 평가 웹 앱입니다. 사용자가 방이나 책상 사진을 업로드하면, Keras 기반 분류 모델과 YOLO 기반 객체 탐지, CLIP 텍스트-이미지 모델을 조합해 **얼마나 지저분한지(또는 얼마나 깨끗한지)** 를 점수로 보여주고 시각화해 줍니다.

웹 UI에서는 카메라 촬영 또는 갤러리에서 이미지를 선택할 수 있고, 로그인한 사용자는 결과를 게시물로 공유하고 다른 사람의 사진을 평가할 수 있습니다.

---

## Features

- **Room / Desk 모드**
  - `room`: 방 전체 청결도 평가 (흑백 기반 ResNet 모델)
  - `desk`: 책상/작업대 청결도 평가 (RGB ResNet 모델)
- **다중 AI 분석 파이프라인**
  - **Keras ResNet 이진 분류 모델**
    - `resnet_room_clean_model.keras` (방)
    - `desk_resnet50_mixup_best.keras` (책상)
    - Grad-CAM을 이용한 **청소가 필요한 영역 히트맵** 생성
  - **YOLO 기반 지저분함(dirty) 스코어**
    - `yolo26x.pt` YOLO 모델을 사용해 객체 탐지
    - 한 종류의 물건이 넓게 퍼져 있거나, 여러 종류가 좁은 영역에 몰려 있을수록 더 **지저분** 하다고 판단
    - 선택적으로 `yolo_dirty_model.joblib` 학습 모델을 사용해 탐지 피처 기반 dirty 확률을 예측
    - 투명 배경 YOLO 박스 레이어 및 원본 이미지 위 오버레이 이미지 생성
  - **CLIP 기반 “이유” 설명**
    - `openai/clip-vit-base-patch32` 로 이미지와 텍스트 후보를 비교
    - “바닥의 옷가지”, “정리되지 않은 침구”, “쓰레기 및 잡동사니”, “어질러진 책상/선반”, “깨끗한 상태” 등의 한국어 레이블 출력
- **Grad-CAM Heatmap**
  - ResNet 출력에 대한 Grad-CAM을 생성해 **어느 영역이 더럽다고 판단되는지** 시각적으로 표시
  - 투명한 히트맵-only PNG를 만들어, 프론트엔드에서 토글 가능한 레이어로 사용 가능
- **커뮤니티 기능**
  - 이메일/비밀번호 기반 회원가입 및 로그인 (`flask-login`)
  - AI 결과를 게시글로 공유 (이미지, AI score, mode 저장)
  - 다른 사용자가 게시물에 대해 **청결도 %를 평가**하고 평균을 확인

---

## Architecture Overview

- **`app.py`**
  - Flask 애플리케이션 엔트리포인트
  - 사용자 인증, 게시글 CRUD(피드/상세조회/평가), 이미지 업로드 및 분석 결과 렌더링
  - `/`:
    - 이미지 업로드 후 `scoring.run_all_analyses(...)` 호출
    - Keras / YOLO / CLIP 결과와 히트맵, 박스 오버레이, 설명 텍스트를 `result.html` 템플릿으로 전달
  - `/auth/*`:
    - 회원가입, 로그인/로그아웃, 프로필
  - `/posts`, `/post/<id>`:
    - AI 점수 및 사용자 평가가 포함된 게시글 피드/상세 페이지
  - `/post/<id>/score`:
    - 로그인 유저가 게시글에 대해 청결도 %를 남기고, 서버는 이를 기반으로 0~1 dirty score 로 저장

- **`db.py`**
  - MySQL 연결 유틸
  - `.env` 또는 환경변수 기반 설정 (`MYSQL_HOST`, `MYSQL_USER`, `MYSQL_PASSWORD`, `MYSQL_DATABASE`)

- **`scoring/` 패키지**
  - `scoring.py`:
    - Keras, YOLO, CLIP 세 분석을 **ThreadPoolExecutor** 로 병렬 실행
    - Keras probability, YOLO dirty score 를 가중합으로 통합한 `total_score` 계산
    - Grad-CAM heatmap-only PNG 및 YOLO 박스-only PNG 저장
  - `dirty_scorer.py`:
    - YOLO 모델(`yolo26x.pt`)을 사용해 객체 탐지
    - 탐지 결과에서 다양한 피처(`YOLO_FEATURE_NAMES`)를 계산한 뒤 dirty score 산출
    - 선택적으로 학습된 dirty 모델(`yolo_dirty_model.joblib`)을 로드하여 dirty 확률 예측
    - 각 박스별 기여도(색상: green→red)와 오버레이 이미지 생성
  - `heatmap.py`:
    - ResNet 기반 Grad-CAM 구현 (room/desk 모델 모두 지원)
    - 원본 해상도에 맞는 heatmap-only PNG 및 오버레이 이미지 생성
    - 가장 더러운 영역을 기준으로 간단한 한국어 청소 가이드 텍스트 생성
  - `clip.py`:
    - HuggingFace `transformers` 의 TFCLIPModel + CLIPProcessor 사용
    - 미리 정의된 영어 프롬프트와의 유사도를 계산해 한국어로 상태/이유를 리턴

- **템플릿 & 정적 파일**
  - `templates/base.html`, `templates/index.html`, `templates/result.html`, `templates/posts_feed.html` 등
    - Bootstrap 5 + Font Awesome 기반 반응형 UI
    - 카메라 스트림/갤러리 업로드, 처리 중 모달, 결과 시각화 등
  - `static/css/app.css`, `static/js/app.js`

---

## Requirements

Python 3.10+ 권장.

핵심 의존성은 `requirements.txt` 에 정의되어 있으며, 주요 라이브러리는 다음과 같습니다.

- 웹 / 인증
  - `flask`
  - `flask-login`
  - `python-dotenv`
  - `pymysql`
- 딥러닝 / 모델링
  - `tensorflow` (Keras ResNet 분류 모델)
  - `ultralytics` (YOLOv8 계열)
  - `opencv-python-headless`
  - `numpy`, `scikit-learn`, `joblib`
- 기타
  - `transformers`, `Pillow` (CLIP)
  - `matplotlib` (heatmap 시각화)

설치는 다음과 같이 할 수 있습니다.

```bash
pip install -r requirements.txt
```

GPU 가속을 사용하려면 TensorFlow에 대해 CUDA 지원 버전을 별도로 설치해야 합니다.

---

## Model Files

이 프로젝트는 여러 개의 사전 학습된/학습된 모델 파일을 필요로 합니다. Git LFS 또는 별도 스토리지에 보관하고, 아래와 같은 경로로 배치해야 합니다.

- 프로젝트 루트:
  - `resnet_room_clean_model.keras`
  - `desk_resnet50_mixup_best.keras`
  - `yolo26x.pt`
  - `yolo_dirty_model.joblib`

---

## Database Setup

애플리케이션은 MySQL을 사용합니다.

- `.env` 또는 환경 변수에서 다음 값을 설정합니다.

```bash
MYSQL_HOST=localhost
MYSQL_USER=root
MYSQL_PASSWORD=your_password
MYSQL_DATABASE=room_cleanliness
SECRET_KEY=your_flask_secret_key
```

- 초기 스키마와 예시 데이터를 생성하기 위한 `init_db.py` 스크립트가 포함되어 있다면, 다음처럼 실행합니다 (실제 옵션/내용은 스크립트 참고).

```bash
python init_db.py
```

앱을 실행하기 전에 `app.py` 의 `__main__` 블록에서 MySQL 연결 테스트를 수행하므로, 연결 실패 시 경고 메시지와 함께 환경 변수 또는 DB 초기화 스크립트를 확인하라는 안내가 출력됩니다.

---

## Running the App

1. 가상환경 생성 (선택)

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
```

2. 의존성 설치

```bash
pip install -r requirements.txt
```

3. 환경 변수 설정 (예시)

```bash
export MYSQL_HOST=localhost
export MYSQL_USER=root
export MYSQL_PASSWORD=...
export MYSQL_DATABASE=room_cleanliness
export SECRET_KEY=change-me
export FLASK_HOST=0.0.0.0
export FLASK_PORT=5001
export FLASK_DEBUG=1
```

4. DB 초기화 (`init_db.py` 가 있는 경우)

```bash
python init_db.py
```

5. Flask 앱 실행

```bash
python app.py
```

브라우저에서 `http://localhost:5001/` 에 접속하면 메인 페이지(`AI 방 검사관`)를 사용할 수 있습니다.

---

## Usage Flow

1. **모드 선택**
   - 메인 화면에서 `방` 또는 `책상` 버튼을 선택합니다.
2. **이미지 업로드**
   - 카메라를 켜서 사진을 촬영하거나, 갤러리에서 이미지를 선택합니다.
3. **AI 판독 시작**
   - `AI 판독 시작` 버튼을 누르면 서버에서 Keras + YOLO + CLIP 분석을 수행합니다.
4. **결과 확인**
   - 전체 점수(`total_score`)와 Keras 확률, YOLO dirty score, CLIP 분석 결과(한국어)를 확인합니다.
   - Grad-CAM 히트맵 및 YOLO 오버레이 이미지를 통해 어떤 영역/객체가 지저분함에 기여하는지 시각적으로 파악할 수 있습니다.
5. **로그인 후 공유 (선택)**
   - 회원가입/로그인 후에는 결과 이미지를 게시물로 저장하고, 다른 사용자의 피드를 확인하며 평가할 수 있습니다.

---

## Project Structure (Simplified)

```text
DeepLearning_Project/
├─ app.py                # Flask app (routes, auth, posts, scoring integration)
├─ db.py                 # MySQL connection helper
├─ init_db.py            # DB initialization script (if provided)
├─ scoring/
│  ├─ __init__.py
│  ├─ scoring.py         # run_all_analyses: Keras + YOLO + CLIP
│  ├─ dirty_scorer.py    # YOLO dirty scoring, feature extraction, overlays
│  ├─ heatmap.py         # Grad-CAM heatmap generation and overlays
│  ├─ clip.py            # CLIP-based cleanliness reason analysis
├─ templates/
│  ├─ base.html
│  ├─ index.html
│  ├─ result.html
│  ├─ posts_feed.html
│  ├─ post_detail.html
│  ├─ login.html
│  └─ register.html
├─ static/
│  ├─ css/app.css
│  ├─ js/app.js
│  └─ logo.png
├─ uploads/              # (runtime) uploaded images & overlays
├─ requirements.txt
└─ README.md
```
