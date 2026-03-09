# preprocessing.py
from torchvision import transforms

# 다른 곳에서 불러다 쓸 수 있도록 변수로 만들어둡니다.
data_transforms = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

classes = ['Clean (깨끗함)', 'Dirty (더러움)', 'Normal (평범함)']