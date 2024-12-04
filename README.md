# 모바일 카메라를 활용한 킥보드 불법 주정차 판별
양인호 정보공학전공 2021074220<br>
조수민 국제학부 2020031849 

### 목차

[I. Proposal](#i-proposal)<br>
[II. Dataset](#ii-dataset)<br>
[III. Methodology](#iii-methodology)<br>
[IV. Step by Step Guide](#iv-step-by-step-guide)<br>
[V. Evaluation & Analysis](#v-evaluation--analysis)<br>
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
  * 이미 다양성: 다양한 무작위 변환(뒤집기, 색조정, 회전 등)을 적용하여 데이터 다양성을 확보
  * 데이터 수: 약 200장의 이미지를 수집한 뒤 모델의 성능 개선을 위하여 100장 중복 (총 300장)

# III. Methodology

### **사용 기술 및 도구**<br>
* **환경**: T4 GPU 가속 기능을 활용하기 위해 Google Colab Pro 환경에서 학습을 진행하였습니다. 
* **모델**: 이미지 분류 분야에서 높은 성능을 보이는 ResNet(Residual Networks) 모델을 사용하여 딥러닝 모델에서 발생하는 기울기 소실 문제를 해결하고 90% 이상의 정확도를 달성하였습니다.<br> 
* **안드로이드 앱**: 코틀린 기반의 안드로이드 앱에 TorchScript 모듈을 접목하여 실시간으로 촬영 및 판별이 가능하도록 하였습니다.<br>

### **개발 과정**<br>
* **데이터 전처리**: 이미지 크기를 모델 학습에 적절한 224px * 224px로 조정한 뒤 3차원 벡터(Tensor)의 숫자값으로 변환해 컴퓨터가 인식할 수 있도록 하였습니다. <br>
* **모델 성능 최적화**: 다양한 하이퍼파라미터 튜닝 및 batch와 epoch 조정, 정규화 등을 통해 모델의 정확도를 최대화하였습니다.<br>

# IV. Step by Step Guide
### **1. 라이브러리 불러오기**<br>
* <code>torch</code>: 딥러닝 학습을 위한 PyTorch 라이브러리.
* <code>torchvision</code>: 데이터 처리 및 사전 학습된 모델을 로드하는 도구.
* <code>matplotlib</code>: 데이터 시각화를 위한 라이브러리.
* <code>PIL</code>: 이미지 로드 및 전처리를 위한 도구.

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

### **2. 장치 설정 (GPU/CPU)**<br>
Colab Pro의 T4 GPU를 활용하여 학습 속도를 가속화합니다. GPU가 없는 경우 CPU를 사용하도록 설정합니다.

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

### **3. 데이터 전처리 및 변환 설정**<br>
* 이미지를 224x224 크기로 조정.
* 다양한 무작위 변환(뒤집기, 색조정, 회전 등)을 적용하여 데이터 다양성을 확보.
* 정규화를 통해 모델 학습 안정화.
  
```python
data_path = '/content/drive/MyDrive/인공지능2/인공지능2프로젝트/킥보드사진/train'

transform = transforms.Compose([
    transforms.Resize((224, 224)), # 이미지 크기 조정
    transforms.RandomHorizontalFlip(), # 뒤집기
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2), # 색 조정
    transforms.RandomRotation(15), # 회전
    transforms.ToTensor(), # 텐서 변환
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # 정규화
])

```

### **4. 데이터셋 로드 및 학습/테스트 데이터 분할**<br>
데이터를 불러온 뒤 학습용(80%)과 테스트용(20%)으로 나눕니다.<br>

```python
full_dataset = datasets.ImageFolder(root=data_path, transform=transform)

train_size = int(0.8 * len(full_dataset))
test_size = len(full_dataset) - train_size
train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])

```


### **5. 클래스 불균형 조정**<br>
<code>WeightedRandomSampler</code>를 사용해 학습 데이터셋을 샘플링한 뒤, 데이터가 적은 클래스의 샘플이 더 자주 선택될 수 있도록 조정합니다. 이 과정은 학습 데이터를 균형 있게 제공하여 모델이 특정 클래스에 편향되지 않도록 합니다.

```python
# 학습 데이터의 라벨 수집
train_labels = [full_dataset.targets[i] for i in train_dataset.indices]
# 클래스별 데이터 개수 계산
class_counts = Counter(train_labels)

#  클래스별 가중치 계산 (샘플이 적은 클래스일수록 높은 가중치 부여)
class_weights = [1.0 / class_counts[c] for c in range(len(class_counts))]
# 샘플별 가중치 설정
sample_weights = [class_weights[label] for label in train_labels]
# 가중치 기반 샘플링 설정
sampler = WeightedRandomSampler(sample_weights, num_samples=len(train_dataset), replacement=True)
```
### **6. 데이터 로더 생성**<br>
데이터를 배치 크기(32)로 나누어 모델에 전달할 준비를 합니다.
```python
train_loader = DataLoader(train_dataset, batch_size=32, sampler=sampler)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
```
### **7. ResNet34 모델 설정 및 수정**<br>
* **ResNet34 모델 로드**: 사전 학습된 ResNet34 모델을 가져옵니다.
* **출력층 수정**: 원래의 출력층을 이진 분류를 위한 Fully Connected (FC) 레이어로 교체합니다.
* **장치 설정**: 모델을 GPU 또는 CPU로 전송하여 학습 준비를 완료합니다.

```python
model = models.resnet34(pretrained=True)
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 2)  # 이진 분류를 위해 출력층 수정
model = model.to(device)
```

### **8. 손실 함수 및 최적화기 정의**<br>
* **손실 함수**: <code>CrossEntropyLoss</code>를 사용해 모델 예측과 실제 값 간의 차이를 계산합니다.
* **최적화기**: Adam 옵티마이저를 사용하여 모델 가중치를 업데이트합니다. 학습률은 0.001로 설정되었습니다.

```python
criterion = nn.CrossEntropyLoss()  # 손실 함수 설정
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  # Adam 옵티마이저
```
### **9. 모델 학습 함수 정의 및 모델 학습**<br>
지정된 epoch 만큼 학습을 진행하며, 매 epoch마다 학습 손실과 정확도, 검증 정확도를 출력합니다.<br>
1. **Training Phase:**
   * 학습 데이터를 사용해 모델을 훈련.
   * 손실(loss) 계산 후 역전파(backpropagation)와 최적화를 통해 가중치 업데이트.
   * 학습 정확도 계산.
     
2. **Validation Phase:**
   * 검증 데이터를 사용해 모델의 성능 평가.
   * 학습 정확도와 검증 정확도를 출력.
     <br>
     
```python
# Training loop
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=30):
    model.train()
    for epoch in range(num_epochs):
        # Training phase
        running_loss = 0.0
        correct = 0
        total = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_accuracy = correct / total * 100

        # Validation phase
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        val_accuracy = val_correct / val_total * 100
        model.train()

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}, "
              f"Train Accuracy: {train_accuracy:.2f}%, Val Accuracy: {val_accuracy:.2f}%")
```

```python
print("Training started...")
train_model(model, train_loader, test_loader, criterion, optimizer, num_epochs=30)
```

### **9. 모델 학습**<br>



# V. Evaluation & Analysis

# VI. Coclusion & Discussion


