# 모바일 카메라를 활용한 과일 분류기

[I. Proposal<br>](#proposal)
II. Datasets<br>
III. Methodology<br>
IV. Evaluation & Analysis<br>
V. Related Work<br>
VI. Coclusion & Discussion<br>

# I. Proposal

**1. 프로젝트 개요**<br>
본 프로젝트는 이미지 분류 분야에서 높은 성능을 보이는 ResNet(Residual Networks) 모델을 사용하여 딥러닝 모델에서 발생하는 기울기 소실 문제를 해결하고 80% 이상의 정확도를 달성하는 것을 목표로 합니다. 이와 같은 모델은 바코드 없는 상품 분류 및 계산, 농산물 품질 분류, TTS등의 기술과 결합한 시각장애인용 식별 기능, 자동화 로봇 탑재용 모델 등 향후 다양한 분야에서 활용될 수 있을 것으로 기대됩니다. 

**2. 프로젝트 목표**<br>
과일 분류 모델 개발: ResNet 기반의 과일 분류 모델을 설계하고 훈련하여, 다양한 과일 종류를 높은 정확도로 분류합니다.
데이터셋 구성: 오렌지, 사과, 바나나, 딸기 등 일반적으로 소비되는 과일의 이미지로 구성된 학습 데이터셋을 사용합니다.
모델 성능 최적화: 다양한 하이퍼파라미터 튜닝 및 데이터 증강 기법을 통해 모델의 정확도를 최대화합니다.
안드로이드 앱 통합: 훈련된 모델을 TFLite 형식으로 변환하여 안드로이드 환경에서 사용할 수 있는 과일 분류 프로그램을 개발합니다.

**3. 사용 기술 및 도구**<br>
프레임워크: TensorFlow 및 PyTorch (모델 학습에 사용)
모델 구조: ResNet-50, ResNet-101 등 다양한 ResNet 아키텍처를 실험하여 최적 성능을 도출합니다.
데이터 전처리: 이미지 크기 조정, 색상 보정 및 데이터 증강 기술 (회전, 확대, 축소 등) 사용
모델 변환: TFLite를 사용하여 안드로이드에 최적화된 모델 형식으로 변환
안드로이드 개발: Android Studio를 사용하여 앱 개발, 카메라로 실시간 과일 분류 가능하게 구현

# II. Dataset
[Kaggle Fruit Recognition Dataset][dataset]

# III. Methodology

# IV. Evaluation & Analysis

# V. Related Work

# VI. Coclusion & Discussion

[dataset]: https://www.kaggle.com/datasets/chrisfilo/fruit-recognition
