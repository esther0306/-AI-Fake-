# SW중심대학 디지털 경진대회_SW와 생성AI의 만남 : AI 부문
> **주제:** 생성 AI의 가짜(Fake) 음성 검출 및 탐지  
> **팀원:** 이신화, 이지수, 김지우, 김소연  
> **참여 기간:** 2024.07.01 ~ 2024.07.19 09:59


## 🔍 문제 설명

5초 분량의 입력 오디오 샘플에서 영어 음성의 진짜(Real) 사람 목소리와 생성 AI의 가짜(Fake) 사람 목소리를 동시에 검출해내는 AI 모델을 개발해야 합니다.

- **Train Set:**  
  55,438개의 오디오(ogg) 샘플 (32kHz 샘플링)  
  → 각 샘플은 방음 환경에서 수집된 진짜 또는 가짜 목소리 1개 포함

- **Test Set:**  
  50,000개의 오디오 샘플 (32kHz, 5초)  
  → 최대 2개의 목소리 (Real/Fake) 포함, 다양한 환경에서 녹음됨

- **Unlabeled Data:**  
  1,264개의 오디오 샘플 (Label 미제공, 평가 데이터와 환경 동일)

자세한 내용 및 데이터 출처: [Dacon 대회 페이지](https://dacon.io/competitions/official/236253/data)

---
# 모델 설계 및 학습 전략 (총 4개 실험)
본 프로젝트에서는 총 **4개의 1D ResNet18 기반 모델**을 실험한 후, **soft voting 앙상블**을 적용하여 최종 성능을 향상
모든 모델은 **원시 waveform 입력**을 사용하며, **multi-label 이진 분류 구조**로 Fake/Real 음성 동시 예측 가능

## 🔄 실험 모델 요약

| 버전 | 이름 | 주요 특징 |
|------|------|-----------|
| A | `jiwoo's_test_method`[보기](./models/jiwoo's_test_method.py) | 무음 샘플 포함, 일반적인 mixup, 기본 추론 |
| B | `jisoo's_test` | 무음 + 1초 단위 분할 → 평균 추론 |
| C | `resnet_sinhwa's_test_method` | segment 평균 기반 추론, 5세그먼트 처리 |
| D | `sinhwa's_method_e8` | sinhwa_v1에서 **Epoch 증가(8회)** → 최종 성능 개선 |

## 🧪 Noise 제거 처리

- **사용 모델:** [DeepFilterNet V3](https://github.com/Rikorose/DeepFilterNet)  
- **방법:**  
  Dacon에서 제공한 사전 학습된 가중치를 이용해 **테스트 데이터의 노이즈 제거 수행**  
  (추가적인 학습은 진행하지 않음)

---
## 공통 데이터 전처리

- **샘플 길이 통일:**  
  입력 오디오가 5초(=32,000 samples)보다 길 경우 무작위로 5초를 **Crop**, 짧은 경우 **Zero-padding** 수행
- **무음 샘플 생성:**  
  학습의 일반화와 강건함을 위해 **무작위 5%의 샘플을 무음(zero waveform)으로 대체**
- **Waveform 증강:**  
  `RandomCrop(1초)` 을 포함한 파형 기반 증강 기법 적용

---

## 모델 구성

- **1D ResNet-18** 아키텍처 기반
  - 입력: 1채널 waveform (Conv1D)
  - 특징 추출: 4개 블록 (64 → 128 → 256 → 512)
  - 출력: 2개의 sigmoid 활성화로 **[Fake, Real] 다중 레이블** 확률 예측

- **Mixup 학습 전략**:
  - 두 waveform과 라벨을 섞어 `Fake-Fake`, `Real-Real`, `Fake-Real` 조합 학습 가능
  - 무음([0,0]) 샘플은 mixup 대상에서 제외하여 label 왜곡 방지







