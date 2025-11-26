# v2.mixup 구현 실험 로그 🧪

## 📋 프로젝트 개요
PyTorch의 `v2.mixup` 기능을 테스트하고 검증하는 프로젝트

---

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

## 📊 실험 결과 요약

| 실험 # | 날짜 | 주요 변경사항 | 결과 | 비고 |
|--------|------|---------------|------|------|
| 1 | 2025-11-26 | 초기 설정 및 문제 발견 | 진행 중 | 데이터 정합성 문제 해결 필요 |

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

## 🔗 참고 자료
- [PyTorch v2.mixup 문서](https://pytorch.org/vision/stable/transforms.html)
- [Mixup 논문](https://arxiv.org/abs/1710.09412)
