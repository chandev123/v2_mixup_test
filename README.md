# v2.mixup Test Project 🧪

이 프로젝트는 PyTorch의 `v2.mixup` 기능을 테스트하고 검증하기 위해 만들어졌습니다! 🚀

## 🎯 목표
`v2.mixup`이 이미지를 섞는 연산을 제대로 수행하는지 확인합니다.
- 🖼️ **이미지와 라벨 믹스**: `v2.mixup`이 이미지와 라벨을 올바르게 섞는지 확인
- 📉 **손실 함수 반영**: 믹스된 결과가 손실 함수(Loss Function)에 잘 반영되는지 확인
- ⚙️ **Optimizer 반영**: 믹스된 결과가 Optimizer에 잘 반영되는지 확인

## 🛠️ 사용 방법
누구나 쉽게 사용할 수 있습니다!

1. **필요한 라이브러리 설치** 📦
   ```bash
   pip install torch torchvision pandas pillow tqdm
   ```

2. **노트북 실행** 🏃‍♂️
   `v2.mixup_test_251126.ipynb` 파일을 열고 셀을 순서대로 실행하세요.

## 📂 파일 구조
- `v2.mixup_test_251126.ipynb`: 메인 테스트 코드가 포함된 주피터 노트북
- `v2_mixup_test_251126.py`: 파이썬 스크립트 버전 (Mixup 시각화 기능 포함)
- `model_acc.png`: 학습 결과 (Loss/Accuracy) 그래프
- `mixup_samples_comparison.png`: Mixup 전후 비교 이미지
- `mixup_training_snapshot.png`: 학습 도중 캡처된 Mixup 스냅샷

---
Happy Coding! ✨
