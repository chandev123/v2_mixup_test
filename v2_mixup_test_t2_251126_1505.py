# v2.mixup_test_251126.ipynb
# 목표: v2.mixup의 기능 확인
# 과제: v2.mixup이 이미지를 섞는 연산 확인
# - v2.mixup이 이미지와 라벨을 섞는 기능 확인
# - v2.mixup을 통과한 결과가 손실함수에 반영되는 기능 확인
# - v2.mixup을 통과한 결과가 optimizer에 반영되는 기능 확인

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.cuda.amp import GradScaler, autocast
from torchvision.transforms import v2
from torchvision.models import resnet18

from pathlib import Path
from PIL import Image
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
import os

def get_unique_filename(filename):
    if not os.path.exists(filename):
        return filename
    
    name, ext = os.path.splitext(filename)
    counter = 1
    while True:
        new_filename = f"{name}_{counter}{ext}"
        if not os.path.exists(new_filename):
            return new_filename
        counter += 1

# 커스텀 데이터셋 클래스
class MyDataset(Dataset):
    def __init__(self, data_path, transform=None, train=True):
        self.train = train
        train_df = pd.read_csv(cvs_path)
        self.name2label = dict(zip(train_df["name"], train_df["label"]))

        if self.train:
            self.img_path = list(data_path.joinpath("train_data").rglob( "*.png"))
            self.labels =  [self.name2label[d.name] for d in self.img_path]
        else:
            self.img_path = list(data_path.joinpath("test_data").rglob("*.png"))

        self.transform = transform

    def __len__(self):
        return len(self.img_path)   

    def __getitem__(self, index):
        img = Image.open(self.img_path[index])
        if img.mode != 'RGB':
            img = img.convert('RGB')

        if self.transform:
            img = self.transform(img)

        if self.train:
            return img, self.labels[index]
        else:
            return img, self.img_path[index].name

# 데이터셋 디렉토리 위치 지정
data_dir = Path(__file__).parent
data_path = data_dir.joinpath("v2.mixup_test_251126_png")
cvs_path = data_path.joinpath("train_data.csv")

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

transform =  v2.Compose([
    v2.ToImage(),
    v2.ToDtype(dtype=torch.float32, scale=True),
    v2.Normalize(mean=mean, std=std),
])

test_transform =  v2.Compose([
    v2.ToImage(),
    v2.ToDtype(dtype=torch.long)
])

train_data = MyDataset(data_path, train=True, transform=transform)
test_data = MyDataset(data_path, train=False, transform=test_transform)

train_size = int(len(train_data) * 0.8)
train_data, val_data = torch.utils.data.random_split(train_data, [train_size, len(train_data) - train_size])
train_data, train_vl_data = torch.utils.data.random_split(train_data, [train_size, len(train_data) - train_size])

train_loader = torch.utils.data.DataLoader(train_data, batch_size=128, shuffle=True)
train_vl_loader = torch.utils.data.DataLoader(train_vl_data, batch_size=128, shuffle=False)
val_loader = torch.utils.data.DataLoader(val_data, batch_size=128, shuffle=False)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=128, shuffle=False)

model = resnet18(pretrained=False)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.fc = nn.Linear(512, 10, bias=True)
model = model.to(device)

scaler = GradScaler()
# criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, weight_decay=0.0001)
scheduler = CosineAnnealingLR(optimizer, T_max=10, eta_min=0.001)
mixup = v2.MixUp(alpha=3.0, num_classes=10)

# Mixup 스냅샷 저장 함수
def save_mixup_snapshot(orig_x, orig_y, mixed_x, mixed_y, mean, std, num_samples=5):
    inv_mean = [-m/s for m, s in zip(mean, std)]
    inv_std = [1/s for s in std]
    inv_normalize = v2.Normalize(mean=inv_mean, std=inv_std)
    
    orig_x = inv_normalize(orig_x).cpu()
    mixed_x = inv_normalize(mixed_x).cpu()
    
    plt.figure(figsize=(15, 8))
    for i in range(min(num_samples, orig_x.size(0))):
        # 원본
        plt.subplot(2, num_samples, i + 1)
        plt.imshow(orig_x[i].permute(1, 2, 0).clamp(0, 1))
        plt.axis('off')
        plt.title(f"Original\nLabel: {orig_y[i].item()}")
        
        # Mixup
        plt.subplot(2, num_samples, num_samples + i + 1)
        plt.imshow(mixed_x[i].permute(1, 2, 0).clamp(0, 1))
        plt.axis('off')
        
        probs, indices = torch.topk(mixed_y[i], k=2)
        label_str = f"Mixup\nCls {indices[0].item()}: {probs[0].item():.2f}\nCls {indices[1].item()}: {probs[1].item():.2f}"
        plt.title(label_str, fontsize=9)
        
    plt.tight_layout()
    save_path = get_unique_filename('mixup_training_snapshot.png')
    plt.savefig(save_path)
    plt.close()
    print(f"훈련 중 Mixup 스냅샷 저장 완료: {save_path}")

# train 함수
def train_one_epoch(model, loader, epoch):
    model.train()                 
    tot_loss, tot_acc, tot_cnt = 0.0, 0.0, 0

    pbar = tqdm(loader, desc='훈련')
    for i, (x, y) in enumerate(pbar):
        x, y = x.to(device), y.to(device)
        
        # 첫 에폭의 첫 배치에서만 스냅샷 저장
        if epoch == 1 and i == 0:
            x_orig = x.clone()
            y_orig = y.clone()
            
        x, y = mixup(x, y)
        
        if epoch == 1 and i == 0:
            save_mixup_snapshot(x_orig, y_orig, x, y, mean, std)

        optimizer.zero_grad()              

        out = model(x)         
        loss = criterion(out, y)
        loss.backward()           
        optimizer.step()          
        scheduler.step()                  

        tot_loss += loss.item() * y.size(0)
        pred = out.argmax(dim=1)
        y_true = y.argmax(dim=1)
        tot_acc += (pred == y_true).sum().item()
        tot_cnt  += y.size(0)

    return tot_loss/tot_cnt, tot_acc/tot_cnt

def evaluate(model, loader):
    model.eval()
    tot_loss, tot_acc, tot_cnt = 0.0, 0.0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            y = F.one_hot(y, num_classes=10).float()
            
            out = model(x)
            loss = criterion(out, y)

            tot_loss += loss.item() * y.size(0)
            pred = out.argmax(dim=1)
            y_true = y.argmax(dim=1)
            tot_acc += (pred == y_true).sum().item()
            tot_cnt  += y.size(0)

    return tot_loss/tot_cnt, tot_acc/tot_cnt

tr_hist, te_hist, lr_hist = [], [], []
te_loss, best_te_loss, patience_counter = 0, 0, 0
for ep in range(1, 6):
    tr_loss, tr_acc = train_one_epoch(model, train_loader, ep) # tr
    te_loss, te_acc = evaluate(model, val_loader) # va

    tr_hist.append((tr_loss, tr_acc))
    te_hist.append((te_loss, te_acc))

    if te_loss < best_te_loss:
        best_te_loss = te_loss
        patience_counter = 0
    else:
        patience_counter += 1

    if patience_counter >= 3:
        print(f"Early stopping at epoch {ep}")
        break

print('학습완료')

# 시각화
tr_loss = [t[0] for t in tr_hist]
tr_acc = [t[1] for t in tr_hist]
te_loss = [t[0] for t in te_hist]
te_acc = [t[1] for t in te_hist]

epochs = range(1, len(tr_loss) + 1)

plt.figure(figsize=(12, 5))

# Loss Plot
plt.subplot(1, 2, 1)
plt.plot(epochs, tr_loss, 'b-', label='Training Loss')
plt.plot(epochs, te_loss, 'r-', label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

# Accuracy Plot
plt.subplot(1, 2, 2)
plt.plot(epochs, tr_acc, 'b-', label='Training Accuracy')
plt.plot(epochs, te_acc, 'r-', label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
save_path = get_unique_filename('model_acc.png')
plt.savefig(save_path)
plt.show()

# ---------------------------------------------------------
# Mixup 이미지 시각화 (수정됨)
# ---------------------------------------------------------
def visualize_mixup_samples(loader, mixup_fn, mean, std, num_samples=5):
    print("Mixup 이미지 시각화 생성 중...")
    
    # 1. 데이터 배치 가져오기
    x, y = next(iter(loader))
    x, y = x.to(device), y.to(device)
    
    # 원본 복사 (시각화용)
    orig_x = x.clone()
    orig_y = y.clone()
    
    # 2. Mixup 적용
    mixed_x, mixed_y = mixup_fn(x, y)
    
    # 3. 역정규화 (Un-normalize) 설정
    inv_mean = [-m/s for m, s in zip(mean, std)]
    inv_std = [1/s for s in std]
    inv_normalize = v2.Normalize(mean=inv_mean, std=inv_std)
    
    # 4. 시각화 데이터 준비
    orig_x = inv_normalize(orig_x).cpu()
    mixed_x = inv_normalize(mixed_x).cpu()
    
    plt.figure(figsize=(15, 8))
    
    for i in range(min(num_samples, x.size(0))):
        # --- 원본 이미지 ---
        img_orig = orig_x[i].permute(1, 2, 0).clamp(0, 1)
        plt.subplot(2, num_samples, i + 1)
        plt.imshow(img_orig)
        plt.axis('off')
        plt.title(f"Original\nLabel: {orig_y[i].item()}")
        
        # --- Mixup 이미지 ---
        img_mix = mixed_x[i].permute(1, 2, 0).clamp(0, 1)
        plt.subplot(2, num_samples, num_samples + i + 1)
        plt.imshow(img_mix)
        plt.axis('off')
        
        # Mixup 라벨 확인 (Top 2 클래스 및 비율 표시)
        # mixed_y[i]는 one-hot vector (soft label) 상태임
        probs, indices = torch.topk(mixed_y[i], k=2)
        label_str = f"Mixup\nCls {indices[0].item()}: {probs[0].item():.2f}\nCls {indices[1].item()}: {probs[1].item():.2f}"
        plt.title(label_str, fontsize=9)
        
    plt.tight_layout()
    save_path = get_unique_filename('mixup_samples_comparison.png')
    plt.savefig(save_path)
    plt.show()
    print(f"Mixup 비교 이미지 저장 완료: {save_path}")

# 실행
visualize_mixup_samples(train_loader, mixup, mean, std)
