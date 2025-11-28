
## 🔬 실험 기록

### 실험 #1: v2.mixup 기본 기능 테스트
**날짜**: 2025-11-26

#### 목표
- v2.mixup이 이미지와 라벨을 올바르게 섞는지 확인
- 손실 함수(Loss Function)에 믹스된 결과가 반영되는지 확인
- Optimizer에 믹스된 결과가 반영되는지 확인

#### 설정
- **모델**: ResNet-18 (pretrained=False)
- **손실 함수**: CrossEntropyLoss (label_smoothing=0.1)
- **Optimizer**: AdamW
- **Scheduler**: CosineAnnealingLR (T_max=100, eta_min=0.001)
- **Mixed Precision**: GradScaler 사용
- **데이터**: 커스텀 이미지 데이터셋

#### 발견한 문제점 🐛
1. **이미지-라벨 불일치 문제**
   - CSV 파일의 파일 수와 실제 이미지 수가 불일치
   - 클래스 수도 불일치 발생
   - **에러**: `CUDA error: device-side assert triggered`
   
2. **원인 분석**
   - CSV에 정의된 라벨 범위가 모델의 출력 클래스 수를 초과
   - 일부 이미지 파일이 CSV에 누락되거나, 반대로 CSV에는 있지만 실제 파일이 없는 경우 발생

#### 해결 방법 ✅
```python
# 1. 실제 클래스 수 확인
num_classes = train_df['label'].nunique()
model.fc = nn.Linear(512, num_classes, bias=True)

# 2. CSV에 있는 파일만 필터링
csv_names = set(train_df['name'])
self.img_path = [p for p in glob(f"{data_path}/train_data/*.png") 
                 if Path(p).name in csv_names]
```

#### 코드 개선사항
- `glob()` 대신 `Path.glob()` 사용으로 경로 처리 개선
- 파일명 추출 시 `.split()` 대신 `Path.name` 속성 활용

#### 다음 단계 📌
- [ ] 데이터셋 정합성 검증 코드 추가
- [ ] v2.mixup 적용 전후 이미지 시각화
- [ ] 학습 결과 비교 (mixup 사용 vs 미사용)
- [ ] 성능 메트릭 기록 (accuracy, loss)

---

### 실험 #2: Mixup 설정 오류 디버깅
**날짜**: 2025-11-26

#### 목표
- CUDA device-side assert 오류 원인 파악
- Mixup 적용 시 발생하는 문제 해결

#### 설정
- **GPU**: NVIDIA GeForce GTX 1050
- **CUDA_LAUNCH_BLOCKING**: 1 (디버깅 모드)
- **데이터셋**: CIFAR-10 (10 classes)
- **모델 출력**: 10 classes

#### 발견한 문제점 🐛
1. **CUDA Error 발생**
   - **에러**: `CUDA error: device-side assert triggered`
   - 데이터 로딩 및 모델 구조는 정상 확인:
     - Labels: 0~9 (10개 클래스) ✅
     - Model output: 10 classes ✅
   
2. **추정 원인**
   - **Mixup 설정이 잘못된 것으로 추정**
   - Mixup 적용 후 레이블 처리 방식 문제 가능성
   - CrossEntropyLoss와 mixup된 레이블 간 호환성 문제 의심

#### 현재 상태
- 🔍 디버깅 진행 중
- 데이터 및 모델 검증 완료
- Mixup 코드 부분 점검 필요

#### 다음 단계 📌
- [ ] Mixup 적용 코드 검토
- [ ] 손실 함수 계산 부분 확인
- [ ] Mixup 후 레이블 형식 검증 (float vs int)
- [ ] One-hot encoding 필요 여부 확인

---

#### 구현 내용 🛠️
1. **Mixup 비교 시각화 (`visualize_mixup_samples`)**
   - 원본 이미지 5장 vs Mixup 이미지 5장 비교
   - 역정규화(Un-normalize)를 통해 원본 색상 복원
   - Mixup된 라벨의 상위 2개 클래스와 확률 표시 (예: `Cls 3: 0.80, Cls 7: 0.20`)
   - 결과 저장: `mixup_samples_comparison.png`

2. **학습 중 스냅샷 저장 (`save_mixup_snapshot`)**
   - `train_one_epoch` 함수 수정하여 첫 에폭의 첫 배치 데이터 캡처
   - 실제 학습 루프 내에서 Mixup이 적용되는 순간을 포착
   - 결과 저장: `mixup_training_snapshot.png`

#### 결과 ✅
- Mixup이 정상적으로 작동함을 시각적으로 검증 완료
- 두 이미지가 겹쳐 보이고, 라벨 확률도 그에 맞게 혼합됨을 확인

#### 발견한 문제점 (Issues) 🐛
- **시각화의 한계**: 현재 시각화된 이미지는 최종 출력(Un-normalized) 형태이므로, 실제 연산에 사용되는 Raw Mixup 텐서 상태를 완벽하게 대변하지 못할 수 있음. (추후 수정 예정)

---

## 💡 학습 내용 및 인사이트

### CUDA Assert Error 디버깅
- `CUDA_LAUNCH_BLOCKING=1` 환경변수로 상세 에러 확인 가능
- 라벨 값이 클래스 범위를 벗어나면 device-side assert 발생
- 데이터 로딩 전 항상 라벨 범위 검증 필요


## 🔬 실험 기록

### 실험 #1: v2.mixup 기본 기능 테스트
**날짜**: 2025-11-26

#### 목표
- v2.mixup이 이미지와 라벨을 올바르게 섞는지 확인
- 손실 함수(Loss Function)에 믹스된 결과가 반영되는지 확인
- Optimizer에 믹스된 결과가 반영되는지 확인

#### 설정
- **모델**: ResNet-18 (pretrained=False)
- **손실 함수**: CrossEntropyLoss (label_smoothing=0.1)
- **Optimizer**: AdamW
- **Scheduler**: CosineAnnealingLR (T_max=100, eta_min=0.001)
- **Mixed Precision**: GradScaler 사용
- **데이터**: 커스텀 이미지 데이터셋

#### 발견한 문제점 🐛
1. **이미지-라벨 불일치 문제**
   - CSV 파일의 파일 수와 실제 이미지 수가 불일치
   - 클래스 수도 불일치 발생
   - **에러**: `CUDA error: device-side assert triggered`
   
2. **원인 분석**
   - CSV에 정의된 라벨 범위가 모델의 출력 클래스 수를 초과
   - 일부 이미지 파일이 CSV에 누락되거나, 반대로 CSV에는 있지만 실제 파일이 없는 경우 발생

#### 해결 방법 ✅
```python
# 1. 실제 클래스 수 확인
num_classes = train_df['label'].nunique()
model.fc = nn.Linear(512, num_classes, bias=True)

# 2. CSV에 있는 파일만 필터링
csv_names = set(train_df['name'])
self.img_path = [p for p in glob(f"{data_path}/train_data/*.png") 
                 if Path(p).name in csv_names]
```

#### 코드 개선사항
- `glob()` 대신 `Path.glob()` 사용으로 경로 처리 개선
- 파일명 추출 시 `.split()` 대신 `Path.name` 속성 활용

#### 다음 단계 📌
- [ ] 데이터셋 정합성 검증 코드 추가
- [ ] v2.mixup 적용 전후 이미지 시각화
- [ ] 학습 결과 비교 (mixup 사용 vs 미사용)
- [ ] 성능 메트릭 기록 (accuracy, loss)

---

### 실험 #2: Mixup 설정 오류 디버깅
**날짜**: 2025-11-26

#### 목표
- CUDA device-side assert 오류 원인 파악
- Mixup 적용 시 발생하는 문제 해결

#### 설정
- **GPU**: NVIDIA GeForce GTX 1050
- **CUDA_LAUNCH_BLOCKING**: 1 (디버깅 모드)
- **데이터셋**: CIFAR-10 (10 classes)
- **모델 출력**: 10 classes

#### 발견한 문제점 🐛
1. **CUDA Error 발생**
   - **에러**: `CUDA error: device-side assert triggered`
   - 데이터 로딩 및 모델 구조는 정상 확인:
     - Labels: 0~9 (10개 클래스) ✅
     - Model output: 10 classes ✅
   
2. **추정 원인**
   - **Mixup 설정이 잘못된 것으로 추정**
   - Mixup 적용 후 레이블 처리 방식 문제 가능성
   - CrossEntropyLoss와 mixup된 레이블 간 호환성 문제 의심

#### 현재 상태
- 🔍 디버깅 진행 중
- 데이터 및 모델 검증 완료
- Mixup 코드 부분 점검 필요

#### 다음 단계 📌
- [ ] Mixup 적용 코드 검토
- [ ] 손실 함수 계산 부분 확인
- [ ] Mixup 후 레이블 형식 검증 (float vs int)
- [ ] One-hot encoding 필요 여부 확인

---

#### 구현 내용 🛠️
1. **Mixup 비교 시각화 (`visualize_mixup_samples`)**
   - 원본 이미지 5장 vs Mixup 이미지 5장 비교
   - 역정규화(Un-normalize)를 통해 원본 색상 복원
   - Mixup된 라벨의 상위 2개 클래스와 확률 표시 (예: `Cls 3: 0.80, Cls 7: 0.20`)
   - 결과 저장: `mixup_samples_comparison.png`

2. **학습 중 스냅샷 저장 (`save_mixup_snapshot`)**
   - `train_one_epoch` 함수 수정하여 첫 에폭의 첫 배치 데이터 캡처
   - 실제 학습 루프 내에서 Mixup이 적용되는 순간을 포착
   - 결과 저장: `mixup_training_snapshot.png`

#### 결과 ✅
- Mixup이 정상적으로 작동함을 시각적으로 검증 완료
- 두 이미지가 겹쳐 보이고, 라벨 확률도 그에 맞게 혼합됨을 확인

#### 발견한 문제점 (Issues) 🐛
- **시각화의 한계**: 현재 시각화된 이미지는 최종 출력(Un-normalized) 형태이므로, 실제 연산에 사용되는 Raw Mixup 텐서 상태를 완벽하게 대변하지 못할 수 있음. (추후 수정 예정)

---

## 💡 학습 내용 및 인사이트

### CUDA Assert Error 디버깅
- `CUDA_LAUNCH_BLOCKING=1` 환경변수로 상세 에러 확인 가능
- 라벨 값이 클래스 범위를 벗어나면 device-side assert 발생
- 데이터 로딩 전 항상 라벨 범위 검증 필요

### Path 객체 활용
- `pathlib.Path` 사용으로 OS 독립적인 경로 처리
- `.name` 속성으로 파일명 추출이 더 명확하고 안전

---

## 📋 프로젝트 개요
이 프로젝트는 PyTorch의 `v2.mixup` 기능을 활용하여 데이터 증강(Data Augmentation) 효과를 실험하고 검증합니다. 이미지와 라벨을 섞는 Mixup 연산이 올바르게 수행되는지 시각적으로 확인하고, 학습 과정에서 손실 함수와 옵티마이저에 미치는 영향을 분석합니다.

## 🔗 참고 자료
- [PyTorch v2.mixup 문서](https://pytorch.org/vision/stable/transforms.html)
- [Mixup 논문](https://arxiv.org/abs/1710.09412)
