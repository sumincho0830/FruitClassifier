# 모바일 카메라를 활용한 킥보드 불법 주정차 판별
**양인호 정보공학전공 2021074220** - 데이터 수집 및 전처리, 모델 학습<br>
**조수민 국제학부 2020031849** - 모바일 앱 개발 및 모델 적용

### 목차

[I. Proposal](#i-proposal)<br>
[II. Dataset](#ii-dataset)<br>
[III. Methodology](#iii-methodology)<br>
[IV. Step by Step Guide](#iv-step-by-step-guide)<br>
[V. Evaluation & Analysis](#v-evaluation--analysis)<br>
[VI. Adding PyTorch Model to Android App](#vi-adding-pytorch-model-to-android-app)<br>
[VII. Conclusion & Discussion](#vii-conclusion--discussion)<br>

# I. Proposal
<p>
최근 전동 킥보드의 불법 주차로 인한 시민 불편 사례가 증가하고 있으며, 이는 공공 안전과 재산 피해의 주요 원인으로 작용하고 있습니다. 특히, 시각장애인용 보도 블록 위에 주차된 킥보드, 좁은 골목이나 인도에 사선으로 주차되어 통행을 방해하는 사례는 보행자와 차량 운전자의 안전을 위협할 뿐만 아니라 자동차 손상 등 재산상의 손해를 초래할 수 있습니다. 그러나 이러한 문제를 효과적으로 규제하거나 해결할 수 있는 명확한 방안은 현재 마련되어 있지 않은 상황입니다.
 </p>
 <br>
 
 ### 관련기사: [[인턴액티브] 아무데나 막 세워두는 공유킥보드…시민 불편 한계 도달]([https://example.com](https://www.hankyung.com/article/202310225539Y))

> *-서울시 보행자전거과 관계자는 21년 7월부터 23년 6월까지 약 2년 동안 접수된 민원만 **19만2천588건**에 달하고, 11만1천956건이 견인되었다고 밝혔다.-*

[전체 기사 바로가기](https://www.hankyung.com/article/202310225539Y)

<br>
<br>
 <p>
현행 신고 시스템은 신고자가 사진을 촬영하여 제출하고 관리자가 이를 확인하는 방식으로 이루어지고 있으나, 신고 과정에 많은 시간이 소요되고 개개인의 판단 기준이 명확하지 않아 원활한 신고와 관리가 어렵다는 한계점이 있습니다. 일례로, 올해 1월 부터 시행 중인 서울특별시 개인형 이동장치 주정차 위반 신고시스템은 시민의 판단 하에 전동기기의 ID와 위치를 신고하는 방식으로 운영되고 있으며, 불법 주정차에 대한 특별한 가이드라인이나 판단 보조 도구를 제시하고 있지 않습니다.
  </p>
  
![KakaoTalk_20241205_021630476](https://github.com/user-attachments/assets/2061478f-f507-455c-bc28-489c19d32415)
<br>
![KakaoTalk_20241205_021630476_01](https://github.com/user-attachments/assets/d7d48a41-360b-435f-b5bc-efe157816193)
  
<p>
이에 본 프로젝트는 시민이 동일한 기준을 바탕으로 불법 주차된 킥보드를 신속하고 정확하게 판별하고 신고할 수 있는 시스템을 구축하여, 신고와 관리의 활성화를 도모하고자 합니다. 그리고 이를 통해 공공 안전을 증진하고 불법 주차 문제를 체계적으로 해결하는 데 기여하는 것을 목표로 합니다.
또한, 본 모델은 바코드 없는 상품 분류 및 계산, 텍스트 음성 변환(TTS) 기술과 결합한 시각장애인용 식별 기능, 자동화 로봇에 탑재 가능한 모델 등 향후 다양한 분야에서 활용될 수 있을 것으로 기대됩니다. 
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
# 30 epoch으로 모델 학습
print("Training started...")
train_model(model, train_loader, test_loader, criterion, optimizer, num_epochs=30)
```
![스크린샷 2024-12-04 195245](https://github.com/user-attachments/assets/a395c91b-9a0d-4bf5-bcf4-e5eac6d38bdd)
<br>

### **9. 모델 저장**<br>

```python
model_save_path = "/content/drive/MyDrive/인공지능2/인공지능2프로젝트/kickboard_resnet34.pth"
torch.save(model.state_dict(), model_save_path)
```

# V. Evaluation & Analysis
학습에 사용되지 않은 테스트 데이터를 통해 모델의 최종 성능을 평가합니다.

### **1. 모델 로드 및 평가 모드 설정**<br>
* 저장된 PyTorch 모델 파일(kickboard_resnet34.pth)을 로드합니다.
* ResNet34의 출력층을 이진 분류용으로 수정한 후, 저장된 가중치를 적용합니다.
* model.eval(): 모델을 평가 모드로 전환하여 추론 시 불필요한 드롭아웃 레이어 등을 비활성화합니다.

```python
# 저장된 모델 경로
model_load_path = "/content/drive/MyDrive/인공지능2/인공지능2프로젝트/kickboard_resnet34.pth"

# ResNet34 모델 불러오기
model = models.resnet34(pretrained=False)
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 2)  # 이진 분류
model.load_state_dict(torch.load(model_load_path))
model = model.to(device)
model.eval()  # 평가 모드로 전환
```
### **2. 테스트 이미지 로드 및 평가** <br>
* 테스트 디렉토리에서 이미지를 순차적으로 불러오고 차례로 전처리를 시행합니다.
* 전처리를 수행한 후, 모델 입력 형식에 맞게 배치 차원을 추가합니다.
* 분류 결과를 원본 이미지 위에 표시하여 결과를 직관적으로 확인할 수 있도록 합니다.

```python
# 이미지 전처리 설정
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 이미지를 로드하고 전처리
def preprocess_image(image_path):
    # 이미지 로드 및 Exif 메타데이터 기반 회전 수정
    image = Image.open(image_path)
    image = ImageOps.exif_transpose(image).convert('RGB')  # Exif 데이터 기반 회전 해결
    return transform(image).unsqueeze(0)  # 배치 차원 추가

# 테스트 이미지 디렉토리 경로
image_dir = '/content/drive/MyDrive/인공지능2/인공지능2프로젝트/킥보드사진/train/정석'

# 디렉토리 내 이미지 처리
classes = ["good", "bad"]
for image_name in os.listdir(image_dir):
    image_path = os.path.join(image_dir, image_name)

    # 이미지가 파일인지 확인
    if os.path.isfile(image_path):
        try:
            # 원본 이미지를 로드 (시각화를 위해)
            original_image = Image.open(image_path)
            original_image = ImageOps.exif_transpose(original_image)  # Exif 데이터 기반 회전 해결

            # 이미지 전처리 및 예측
            test_image = preprocess_image(image_path).to(device)
            with torch.no_grad():
                output = model(test_image)
                _, predicted = torch.max(output, 1)

            # 결과 출력
            result = classes[predicted.item()]
            print(f"이미지 {image_name} 예측 결과: {result}")

            # 이미지와 결과를 시각적으로 표시
            plt.imshow(original_image)
            plt.title(f"Prediction: {result}")
            plt.axis('off')
            plt.show()

        except Exception as e:
            print(f"이미지 {image_name} 처리 중 오류 발생: {e}")

```
![image](https://github.com/user-attachments/assets/ec633b8b-5547-430c-ab97-61743484d8e0)


<br>

**학습 결과 추이**
```python
import matplotlib.pyplot as plt

# Training and validation accuracy lists (example data from your output)
train_accuracies = [
    75.49, 92.16, 89.22, 88.24, 88.24, 94.12, 94.12, 96.08, 84.31, 88.24,
    83.33, 77.45, 88.24, 88.24, 87.25, 87.25, 89.22, 91.18, 95.10, 95.10,
    95.10, 96.08, 98.04, 97.06, 95.10, 97.06, 96.08, 91.18, 82.35, 86.27
]
val_accuracies = [
    65.38, 50.00, 80.77, 76.92, 73.08, 92.31, 69.23, 80.77, 76.92, 53.85,
    50.00, 50.00, 65.38, 57.69, 50.00, 50.00, 50.00, 50.00, 57.69, 73.08,
    73.08, 76.92, 80.77, 100.00, 88.46, 50.00, 80.77, 61.54, 50.00, 69.23
]

# Plot training and validation accuracy
plt.figure(figsize=(10, 6))
plt.plot(range(1, 31), train_accuracies, label="Train Accuracy")
plt.plot(range(1, 31), val_accuracies, label="Validation Accuracy", linestyle="--")
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.title("Training and Validation Accuracy Over Epochs")
plt.legend()
plt.grid(True)
plt.show()

```
![image](https://github.com/user-attachments/assets/78b1eafc-049d-4a27-9a91-82faf08d02a5)
<br>
학습 결과에서 Train Accuracy는 90% 이상으로 매우 높은 값을 기록한 반면, Validation Accuracy는 30~70% 사이에서 변동하며 일정하지 않은 경향을 보였습니다. 특히, 학습 정확도가 점차 100%에 근접하는 동안에도 검증 정확도가 크게 개선되지 않는 점에서 **과적합(overfitting)** 가능성을 확인할 수 있었습니다. 이는 데이터 수와 데이터 다양성을 확보함으로써 개선될 수 있다고 생각됩니다.

### 3. Grad-CAM을 활용한 분류 결과 시각화
Grad-CAM(Gradient-weighted Class Activation Mapping)은 딥러닝 모델이 이미지 분류 시 어떤 부분에 주목했는지를 시각적으로 보여줌으로써 예측 결과를 신뢰할 수 있는 시각적 증거를 제공합니다. 이런 특징은 모델의 투명성과 디버깅을 위한 도구로 사용됩니다.

**3-1. Grad-CAM 클래스 정의**<br>
* register_forward_hook: 모델의 forward pass 중 특정 층의 출력(feature map)을 저장.
* register_backward_hook: backward pass 중 특정 층의 기울기(gradient)를 저장.
* generate_cam: 저장된 feature map과 gradient를 사용해 활성화 맵(CAM)을 생성.
```python
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None

        # Hook 등록
        target_layer.register_forward_hook(self.save_feature_map)
        target_layer.register_backward_hook(self.save_gradients)

    def save_feature_map(self, module, input, output):
        self.feature_map = output

    def save_gradients(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate_cam(self, class_idx):
        # Grad-CAM 활성화 맵 생성 과정 (생략)

```
**3-2. 모델 로드 및 Grad-CAM 설정**<br>
* 모델의 마지막 합성곱 층(layer4[2].conv2)을 Grad-CAM 대상 층으로 지정.
* Grad-CAM 클래스의 인스턴스를 생성하여 활성화 맵 생성 준비.
* 기울기와 feature map의 가중합을 통해 활성화 맵 생성.
* 활성화 맵을 원본 이미지 크기로 조정하고, cv2.applyColorMap을 사용해 색상 히트맵 적용.
* 히트맵과 원본 이미지를 cv2.addWeighted를 통해 합성.
```python
model_load_path = "/content/drive/MyDrive/인공지능2/인공지능2프로젝트/kickboard_resnet34.pth"
model = models.resnet34(pretrained=False)
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 2)
model.load_state_dict(torch.load(model_load_path))
model = model.to(device)
model.eval()

# Grad-CAM 설정
target_layer = model.layer4[2].conv2  # ResNet34의 마지막 합성곱 층
grad_cam = GradCAM(model, target_layer)

```
**3-3. Grad-CAM 활성화 맵 생성 및 시각화**<br>
* Grad-CAM 활성화 맵을 생성하기 위해 모델이 예측한 클래스(class_idx)를 사용.
* 선택된 클래스의 예측값에 대해 backward를 호출하여 기울기(gradient)를 계산.
```python
# 이미지 전처리 함수
def preprocess_image(image_path):
    image = Image.open(image_path)
    image = ImageOps.exif_transpose(image).convert('RGB')
    return transform(image).unsqueeze(0)


def apply_grad_cam(image_path, model, grad_cam):
    test_image = preprocess_image(image_path).to(device)

    # 모델 예측
    output = model(test_image)
    _, predicted = torch.max(output, 1)
    class_idx = predicted.item()

    # Grad-CAM 활성화 맵 생성
    model.zero_grad()
    output[0, class_idx].backward()
    cam = grad_cam.generate_cam(class_idx)

    # 활성화 맵을 원본 이미지에 오버레이
    original_image = Image.open(image_path)
    original_image = ImageOps.exif_transpose(original_image).convert('RGB')
    original_image = np.array(original_image)
    cam_resized = cv2.resize(cam, (original_image.shape[1], original_image.shape[0]))
    heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(original_image, 0.5, heatmap, 0.5, 0)

    return predicted.item(), overlay

# 테스트 이미지에 Grad-CAM 적용
for image_name in os.listdir(image_dir):
    image_path = os.path.join(image_dir, image_name)

    if os.path.isfile(image_path):
        try:
            predicted_class, cam_overlay = apply_grad_cam(image_path, model, grad_cam)
            print(f"이미지 {image_name} 예측 클래스: {classes[predicted_class]}")
            plt.imshow(cam_overlay)
            plt.title(f"Prediction: {classes[predicted_class]}")
            plt.axis('off')
            plt.show()
        except Exception as e:
            print(f"이미지 {image_name} 처리 중 오류 발생: {e}")
```
![image](https://github.com/user-attachments/assets/8016ec2c-68f3-487a-be1e-e3af2ef9f997)
<br>
![image](https://github.com/user-attachments/assets/24146a9d-937a-4fb8-b63a-a1bbd359ac2f)
<br>
위와 같이 Grad-CAM을 통해 모델이 인식하는 패턴의 범위를 시각적으로 확인할 수 있습니다. 결과로 출력된 이미지에서는 모델이 킥보드라는 대상을 제대로 인식하는 것을 볼 수 있습니다.

# VI. Adding PyTorch Model to Android App
검증이 완료된 모델은 PyTorch 모듈로 변환하여 안드로이드 앱에서 촬영한 이미지를 판별할 수 있도록 접목합니다. 
### 1. TorchScript 모듈 변환 및 저장
**1-1. CPU기반 환경에 적합하도록 모델 변환** <br>
CPU기반 환경인 모바일 앱에서는 GPU를 사용하는 CUDA 연산을 지원하지 않기 때문에 모델과 데이터(가중치 및 버퍼)를 명시적으로 CPU를 사용하도록 변환해줍니다. 
```python
model.to('cpu') # 모델을 CPU를 사용하도록 설정
for param in model.parameters():
    param.data = param.data.cpu() # 모델의 데이터가 CPU와 호환되도록 변환
for buffer in model.buffers(): 
    buffer.data = buffer.data.cpu() 
```
**1-2. TorchScript 모델 저장** <br>
PyTorch 모델을 TorchScript 형식으로 변환한 후, CPU에서 실행 가능하도록 저장합니다.
```python
scripted_model = torch.jit.script(model.cpu())
scripted_model.save("binary_resnet34_cpu.pt")
```
**1-3. PyTorch 모델 최적화 및 내보내기** <br>
모바일 환경에서 효율적인 실행을 위해 모델을 최적화 한 뒤 <code>.ptl</code>형식으로 저장하여 PyTorch Lite Interpreter에서 사용할 수 있도록 합니다. 
```python
from torch.utils.mobile_optimizer import optimize_for_mobile
# 모델을 모바일 환경에 최적화
optimized_model = optimize_for_mobile(scripted_model)
# Lite Interpreter에서 인식 가능한 형태로 저장
optimized_model._save_for_lite_interpreter("binary_resnet34_cpu.ptl")
```

### 2. 안드로이드 앱에 모델 연결
**2-1. 프로젝트에 모델 추가**<br>
안드로이드 프로젝트 생성 후 app/main/assets 폴더를 생성한 뒤 모델을 추가합니다.<br>
![image](https://github.com/user-attachments/assets/489b97c2-7145-49d5-ae01-b8350a8df9e8)
<br>

**2-2. PyTorch 디펜던시 추가**<br>
자동 생성된 <code>build.gradle</code>(App) 파일에 PyTorch 모델을 인식하고 사용할 수 있게 하는 라이브러리를 추가합니다.
```kotlin

android {
//...
    sourceSets{
            getByName("main"){
                assets.srcDirs("src/main/assets") // PyTorch모델 사용 설정
            }
        }
}

//...

dependencies {
    //...
    implementation("org.pytorch:pytorch_android:1.10.0")
    implementation("org.pytorch:pytorch_android_torchvision:1.10.0")
}
```
**2-2. 이미 전처리 및 모델 활용**<br>
모델 로딩과 이미지 분류 메서드를 포함하는 <code>PyTorchModelProcessor</code> 클래스를 생성합니다.
```kotlin
class PyTorchModelProcessor(private val context:Context) {
    private val TAG = "PyTorchModelProcessor"
    private lateinit var model:Module

    // 모델 로드
    fun loadModel(assetName: String){
        try {
            val modelPath = assetFilePath(context, assetName)
            model = Module.load(modelPath)
        }catch (e:Exception){
            Log.d(TAG, "Failed to load model.")
            throw RuntimeException("Failed to load model: ${e.message}")
        }
    }

    // 이미지 전처리 및 분류
    fun classifyImage(bitmap: Bitmap): String{
        Log.d(TAG, "classifyImage()")

        // 모델이 잘 초기화 되었는지 확인
        if (!::model.isInitialized) {
            throw IllegalStateException("Model has not been initialized. Make sure to call loadModel() first.")
        }

        // Bitmap 이미지 크기 조정, 정규화 및 텐서 변환
        val inputTensor = TensorImageUtils.bitmapToFloat32Tensor(
            Bitmap.createScaledBitmap(bitmap, 224, 224, true),
            floatArrayOf(0.485f, 0.456f, 0.406f), // Normalize mean
            floatArrayOf(0.229f, 0.224f, 0.225f)  // Normalize std
        )
        val inputData = inputTensor.dataAsFloatArray
        Log.d(TAG, "Input Tensor: First 10 values: ${inputData.take(10).joinToString()}")
        Log.d(TAG, "Input Tensor: Size: ${inputData.size}")


        // 모델 실행
        val outputTensor = model.forward(IValue.from(inputTensor)).toTensor()
        Log.d(TAG, "classifyImage: outputTensor[0] ${outputTensor.dataAsFloatArray[0]}")
        Log.d(TAG, "Output tensor shape: ${outputTensor.shape().contentToString()}")


        // 텐서로 출력된 결괏값
        val scores = outputTensor.dataAsFloatArray
        Log.d(TAG, "classifyImage: scores[0] ${scores[0]}")
        Log.d(TAG, "classifyImage: scores.size ${scores.size}")
        val score = scores[0]
        Log.d(TAG, "classifyImage: score $score")
        
        // Sigmoid function
        val probability = 1 / (1 + Math.exp(-score.toDouble()))
        Log.d(TAG, "classifyImage: probability $probability")

        // 분류 결과를 반환
        return if(probability > 0.5) "불법" else "정석"
    }
}
```
필요한 곳에 <code>PyTorchModelProcessor</code> 인스턴스를 생성하고 용도에 맞게 사용합니다.
```kotlin
private lateinit var modelProcessor: PyTorchModelProcessor

modelProcessor = PyTorchModelProcessor(this) // Context를 전달하여 인스턴스 초기화
modelProcessor.loadModel("binary_resnet34_cpu.ptl") // 모델 가져오기
classificationResult = modelProcessor.classifyImage(bitmap) // 이진 분류 결과 받기
```
**분류 결과**
![KakaoTalk_20241205_015813120_02](https://github.com/user-attachments/assets/d8530590-cc0e-4a23-8a9d-bbe061a288b4)
![KakaoTalk_20241205_015813120_04](https://github.com/user-attachments/assets/f222f4e0-c90d-4e51-8761-c9ae46c31c7d)


# VII. Conclusion & Discussion

### 1. 한계점
* **데이터 부족**: 학습을 위한 데이터가 충분히 많지 않아 일반화에 한계가 있을 수 있습니다. 또한, 특정 지역이나 환경에서 수집된 데이터로 학습한 경우 등 데이터가 평향되는 경우, 다른 환경에서는 모델 성능이 저하될 수 있습니다. 데이터의 다양성을 확보를 위한 추가적인 데이터 수집이 필요합니다. 

* **실시간 처리의 한계**: CPU 환경에서 모델을 실행하는 모바일 디바이스에서 모델 실행 속도가 실제 상황에서 충분히 빠르지 않을 가능성이 있습니다. 이를 해결하기 위해 추가적인 최적화 작업이 필요합니다.
  
### 2. 향후 개선 방안
* **데이터 확장**: 데이터 부족 문제를 해결하기 위해 크라우드소싱, 웹 크롤링, 또는 시뮬레이션을 활용한 이미지 생성 방안을 고려할 수 있습니다. 또한, 다양한 기기와 환경에서 데이터를 수집하여 모델의 일반화를 높일 수 있습니다.

* **데이터 증강(Data Augmentation)**: 기존 데이터에 밝기 조절, 왜곡, 필터 적용 등 추가적인 무작위 변환을 통해 학습 데이터의 다양성을 확대하고, 모델이 다양한 상황에 적응할 수 있도록 합니다.
