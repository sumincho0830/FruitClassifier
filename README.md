# 모바일 카메라를 활용한 킥보드 불법 주정차 판별
양인호 정보공학전공 2021074220<br>
조수민 국제학부 2020031849 

### 목차

[I. Proposal](#i-proposal)<br>
[II. Dataset](#ii-dataset)<br>
[III. Methodology](#iii-methodology)<br>
[IV. Evaluation & Analysis](#iv-evaluation--analysis)<br>
[V. Related Work](#v-related-work)<br>
[VI. Conclusion & Discussion](#vi-conclusion--discussion)<br>

# I. Proposal
<p>
최근 전동 킥보드의 불법 주차로 인한 시민 불편 사례가 증가하고 있으며, 이는 공공 안전과 재산 피해의 주요 원인으로 작용하고 있습니다. 특히, 시각장애인용 보도 블록 위에 주차된 킥보드, 좁은 골목이나 인도에 사선으로 주차되어 통행을 방해하는 사례는 보행자와 차량 운전자의 안전을 위협할 뿐만 아니라 자동차 손상 등 재산상의 손해를 초래할 수 있습니다. 그러나 이러한 문제를 효과적으로 규제하거나 해결할 수 있는 명확한 방안은 현재 마련되어 있지 않은 상황입니다.
 </p>
 <p>
현행 신고 시스템은 신고자가 사진을 촬영하여 제출하고 관리자가 이를 확인하는 방식으로 이루어지고 있으나, 신고 과정에 많은 시간이 소요되고 개개인의 판단 기준이 명확하지 않아 원활한 신고와 관리가 어렵다는 한계점이 있습니다.
  </p>
<p>
이에 본 프로젝트는 시민이 동일한 기준을 바탕으로 불법 주차된 킥보드를 신속하고 정확하게 판별하고 신고할 수 있는 시스템을 구축하여, 신고와 관리의 활성화를 도모하고자 합니다. 이를 통해 공공 안전을 증진하고 불법 주차 문제를 체계적으로 해결하는 데 기여하는 것을 목표로 합니다.
또한, 본 모델델은 바코드 없는 상품 분류 및 계산, 텍스트 음성 변환(TTS) 기술과 결합한 시각장애인용 식별 기능, 자동화 로봇에 탑재 가능한 모델 등 향후 다양한 분야에서 활용될 수 있을 것으로 기대됩니다. 
</p>

# II. Dataset
* **불법 주차 분류 기준**: 사람이나 자전거 등의 통행에 현저하게 방해가 될 가능성이 있는 위치에 주차되어 있는 킥보드를 **불법**으로, 벽에 가깝게 주차되어 있거나 통행로를 크게 침해하지 않는 킥보드를 **정석**으로 규정하였습니다.
* **데이터 수집 방법**:
* * 사용 기기: 휴대폰 카메라 (갤럭시 노트20)
  * 이미지 촬영 기준: 휴대폰 카메라의 3x3 격자의 중하부에 킥보드가 위치하도록 촬영
  * 데이터 수: 약 200장의 이미지를 수집한 뒤 모델의 성능 개선을 위하여 100장 중복 (총 300장)

# III. Methodology

### **사용 기술 및 도구**<br>
* **환경**: T4 GPU 가속 기능을 활용하기 위해 Google Colab Pro 환경에서 학습을 진행하였습니다. 
* **모델**: 이미지 분류 분야에서 높은 성능을 보이는 ResNet(Residual Networks) 모델을 사용하여 딥러닝 모델에서 발생하는 기울기 소실 문제를 해결하고 90% 이상의 정확도를 달성하였습니다.<br> 
* **안드로이드 앱**: 코틀린 기반의 안드로이드 앱에 TorchScript 모듈을 접목하여 실시간으로 촬영 및 판별이 가능하도록 하였습니다.<br>

### **개발 과정**<br>
1.  **데이터 전처리**: 모델 학습에 적절한 224px * 224px의 크기로 이미지를 변환한 뒤 3차원 벡터(Tensor) 형태로 처리하였습니다. <br>
2.  **모델 성능 최적화**: 다양한 하이퍼파라미터 튜닝 및 batch와 epoch 조정, 정규화 등을 통해 모델의 정확도를 최대화하였습니다.<br>

### **Step by Step Guide**<br>
1. **라이브러리 불러오기**
  * torch: 딥러닝 학습을 위한 PyTorch 라이브러리.
  * torchvision: 데이터 처리 및 사전 학습된 모델을 로드하는 도구.
  * matplotlib: 데이터 시각화를 위한 라이브러리.
  * PIL: 이미지 로드 및 전처리를 위한 도구.

   ```python
   import torch
   import torch.nn as nn
   from torchvision import datasets, models, transforms
   from torch.utils.data import DataLoader, WeightedRandomSampler, random_split
   from collections import Counter
   import matplotlib.pyplot as plt
   from PIL import Image
   import os
   ```


# IV. Evaluation & Analysis

# V. Related Work

# VI. Coclusion & Discussion

