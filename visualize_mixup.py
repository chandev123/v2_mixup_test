import torch
from torchvision.transforms import v2
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
import random

# 데이터셋 디렉토리 위치 지정
data_dir = Path(__file__).parent
data_path = data_dir.joinpath("v2.mixup_test_251126_png")
train_path = data_path.joinpath("train_data")

# 이미지 파일 리스트 가져오기
img_paths = list(train_path.rglob("*.png"))

# 10개 랜덤 선택 (인접하지 않도록 랜덤 셔플 후 선택)
random.shuffle(img_paths)
selected_paths = img_paths[:10]

# Transform 정의
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

transform = v2.Compose([
    v2.ToImage(),
    v2.ToDtype(dtype=torch.float32, scale=True),
    v2.Resize((224, 224)),
    v2.Normalize(mean=mean, std=std),
])

# 이미지 로드 및 전처리
images = []
for p in selected_paths:
    img = Image.open(p).convert('RGB')
    img = transform(img)
    images.append(img)

# 배치 생성
batch = torch.stack(images)
labels = torch.zeros(10, dtype=torch.long) # 더미 라벨

# Mixup 적용
mixup = v2.MixUp(alpha=0.2, num_classes=10)
mixed_batch, mixed_labels = mixup(batch, labels)

# 시각화
def denormalize(tensor):
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    return tensor * std + mean

plt.figure(figsize=(20, 5))

for i in range(10):
    # 원본 이미지
    ax = plt.subplot(2, 10, i + 1)
    orig_img = denormalize(batch[i]).permute(1, 2, 0).clamp(0, 1)
    plt.imshow(orig_img)
    plt.axis('off')
    if i == 0:
        plt.title("Original")

    # Mixup 이미지
    ax = plt.subplot(2, 10, i + 11)
    mixed_img = denormalize(mixed_batch[i]).permute(1, 2, 0).clamp(0, 1)
    plt.imshow(mixed_img)
    plt.axis('off')
    if i == 0:
        plt.title("Mixup")

plt.tight_layout()
plt.savefig('model_mixup.png')
print("model_mixup.png saved.")
