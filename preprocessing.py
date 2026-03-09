# preprocessing.py
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image

# 결과 출력을 위한 클래스 목록 (클래스가 3개였군요!)
classes = ['Clean (깨끗함)', 'Dirty (더러움)']

def prepare_image_for_keras(file_path):
    """
    저장된 이미지 파일을 읽어와서 Keras ResNet 모델이 
    소화할 수 있는 텐서(Tensor) 형태로 변환해주는 함수입니다.
    """
    # 1. 이미지 불러오기 및 크기 조정 (256x256)
    img = image.load_img(file_path, target_size=(256, 256))
    
    # 2. 이미지를 숫자 배열(Numpy Array)로 변환
    img_array = image.img_to_array(img)
    
    # 3. 모델이 요구하는 배치(Batch) 차원 추가: (256, 256, 3) ➡️ (1, 256, 256, 3)
    input_tensor = np.expand_dims(img_array, axis=0)
    
    # 4. Keras ResNet50 전용 전처리 (파이토치의 Normalize 역할을 알아서 해줌)
    processed_tensor = tf.keras.applications.resnet50.preprocess_input(input_tensor)
    
    return processed_tensor
