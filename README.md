<div align="center">

# 🎨 FlowAttention: Lightweight Cross-Covariance Attention for Resource-Efficient Semi-Supervised Semantic Segmentation

**Single-GPU Trainable Lightweight Semi-Supervised Semantic Segmentation Framework**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-3776AB.svg?style=flat-square&logo=python&logoColor=white)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C.svg?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![CUDA 11.8+](https://img.shields.io/badge/CUDA-11.8+-76B900.svg?style=flat-square&logo=nvidia&logoColor=white)](#)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=flat-square)](https://opensource.org/licenses/MIT)

<p align="center">
  <b>FlowAttention</b>은 단일 GPU 환경에서도 고성능의 준지도 세그멘테이션 학습이 가능하도록 설계된<br>자원 효율적 Cross-Covariance Attention 기반 프레임워크입니다.
</p>

[연구 목표](#-연구-목표-research-goal) •
[환경 설정](#-환경-설정-environment-setup) •
[사용법](#-사용법-usage) •
[정량적 성능 비교](#-정량적-성능-비교-benchmark-results) •
[프로젝트 구조](#-프로젝트-구조-project-structure)

---

</div>

<br>

## 🎯 연구 목표 (Research Goal)

본 연구는 **단일 GPU (NVIDIA RTX 3090, 24GB) 환경에서도 고성능 학습이 가능한 경량 Semi-Supervised Semantic Segmentation 프레임워크** 구축을 목표로 합니다.

* **ImageNet Pretrained 모델 미사용**: 사전 학습된 가중치(ImageNet Pretrained) 없이 ResNet-50을 백본(Backbone)으로 직접 학습
* **Cross-Covariance Attention (XCA) 메커니즘**: 레이블(Labeled) 피처의 핵심 정보와 표현력을 언레이블(Unlabeled) 피처로 교차 전달하여 의도적인 일관성 강화
* **극대화된 자원 효율성**: 대규모 멀티 GPU 클러스터가 필수적이었던 **CorrMatch (CVPR 2024)** 대비 현저히 적은 컴퓨팅 자원(단일 RTX 3090)만으로 높은 세그멘테이션 성능 달성

<br>

## 🛠️ 환경 설정 (Environment Setup)

### 1. 권장 사양

- **Hardware**: NVIDIA RTX 3090 (VRAM 24GB) 1장
- **OS**: Ubuntu 22.04 LTS (Docker base image)
- **CUDA**: 12.6.3
- **cuDNN**: 9
- **Python**: Python 3
- **PyTorch**: 2.9.0
- **torchvision**: 0.24.0
- **torchaudio**: 2.9.0
- **PyTorch CUDA Wheel**: cu126

### 2. Docker 기반 환경 구성

실험 환경은 [Semantic-Segmentation-Research/docker](https://github.com/Semantic-Segmentation-Research/docker)의 Dockerfile을 기반으로 이미지를 빌드하여 구성


## 📂 프로젝트 구조 (Project Structure)

```text
ssl-semantic-segmentation/
├── .vscode/                  # VS Code 설정
├── configs/                  # 실험 설정 파일
├── dataset/                  # 데이터셋 처리
├── model/                    # 모델 및 네트워크 구성
├── partitions/                # 데이터 분할 관련 파일
├── util/                     # 학습 및 평가 유틸리티
│
├── .gitignore                # Git 제외 항목
├── LICENSE                   # 라이선스
├── README.md                 # 프로젝트 문서
│
├── cityscapesLabels.py       # Cityscapes 클래스 및 레이블 정의
├── clean.sh                  # 정리 작업 스크립트
├── configuration.py          # 기본 학습 설정
├── corrmatch.py              # CorrMatch 관련 구현
├── download.sh               # 다운로드 스크립트
├── evaluate.py               # 모델 평가
├── losses.py                 # Loss 함수
├── ssl_inference.py          # 추론 실행
├── ssl_main.py               # Semi-supervised 학습 진입점
├── ssl_tensorboard.py        # TensorBoard 관련 기능
├── vis.py                    # 결과 시각화
└── vis_mask.py               # Segmentation mask 시각화
```
<br>

## 💡 사용법 (Usage)

### 1. 데이터셋 준비

지원하는 데이터셋(Pascal VOC 2012, Cityscapes 등)을 아래 디렉토리 구조에 맞춰 구성합니다.

<details>
<summary><b>📂 데이터셋 디렉토리 구조 예시 (Click)</b></summary>

```text
data/
├── PascalVOC2012/
│   ├── JPEGImages/
│   ├── SegmentationClass/
│   └── ImageSets/Segmentation/
└── Cityscapes/
    ├── leftImg8bit/
    └── gtFine/
```
</details>

### 2. 파라미터 설정

학습 및 평가에 필요한 하이퍼파라미터, 데이터셋 경로, 모델 옵션 등은 `configuration.py`에서 수정할 수 있습니다.

```python
# configuration.py (예시)
from dataclasses import dataclass

@dataclass
class ModelConfig:
    backbone: str = "resnet50"
    num_classes: int = 19  # Cityscapes: 19, Pascal VOC: 21

@dataclass
class TrainConfig:
    dataset: str = "cityscapes"
    model_name: str = "flowAtt_v1.7.0"
    
    # Optimizer & Scheduler
    optimizer: str = "Adam"
    scheduler: str = "cosineDecay"
    
    # Hyperparameters
    batch_size: int = 4          # Step당 Batch Size
    accumulation_steps: int = 4  # Effective Batch Size = 4 x 4 = 16
    lr: float = 5e-4             # Learning Rate (0.0005)
    num_epochs: int = 800
    crop_size: int = 448
...
```

### 3. 학습 및 평가 실행 (Main Pipeline)

단일 RTX 3090 (24GB) 환경에서 `ssl_main.py`를 통해 파이프라인을 실행합니다.

```bash
# 학습 실행
python ssl_main.py

# 추론 실행
python ssl_inference.py
```

<br>
## 📊 정량적 성능 비교 (Benchmark Results)

| Method | Backbone | Pretrained | Resolution | Effective Batch | LR | Precision | Optimizer | mIoU (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **CorrMatch†** | ResNet-101 | O | 801 | 64 | 0.005 | FP32 | SGD+Poly (multi-GPU) | 79.40 |
| **CorrMatch ver.1** | ResNet-101 | O | 448 | 16 | 0.00125 | AMP | SGD+Poly (single-GPU) | 74.72 |
| **CorrMatch ver.2** | ResNet-50 | O | 448 | 16 | 0.00125 | AMP | SGD+Poly (single-GPU) | 72.88 |
| **CorrMatch ver.3** | ResNet-50 | X | 448 | 16 | 0.00125 | AMP | SGD+Poly (single-GPU) | 36.49 |
| **Ours‡** | ResNet-50 | X | 448 | 16 | 0.0005 | AMP | Adam+CosineDecay (single-GPU) | **41.68** <font color="#28a745">(**+5.19%p**)</font> |

> - `†` : 논문 원본 설정 (Multi-GPU 기반)
> - `‡` : FlowAttention (Single RTX 3090, Scratch Training)

<br>
