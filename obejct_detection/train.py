# USAGE
# python train.py

import config
import torch
from torchvision import models, transforms
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
from PIL import Image


torch.cuda.set_device(3)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using Device:", DEVICE)

# CSV 파일 로딩
print("[INFO] loading dataset...")
rows = open(config.ANNOTS_PATH).read().strip().split("\n")

data = []		# 이미지 데이터 리스트
targets = []	# bbox 좌표
filenames = []	# 각 이미지 파일명

# 훈련 데이터 파일 입력
for row in rows:
    # 파일명 파싱
    row = row.split(",")
    (filename, startX, startY, endX, endY) = row
    
    # opencv로 파일 로딩
    imagePath = os.path.sep.join([config.IMAGES_PATH, filename])
    image = cv2.imread(imagePath)
    (h, w) = image.shape[:2]		# 이미지 높이 폭 획득
    
    # 경계 박스 정규화
    startX = float(startX) / w
    startY = float(startY) / h
    endX = float(endX) / w
    endY = float(endY) / h
    
    # 파일 로딩. 타겟 크기는 224, 224
    image = Image.open(imagePath)
    image = image.resize((224,224))
    image = np.array(image)	# 이미지를 배열로 변환
    
    # 이미지를 데이터배열에 추가. targets에 bbox 추가. 파일명 리스트 추가
    data.append(image)
    targets.append((startX, startY, endX, endY))
    filenames.append(filename)

# 데이터를 numpy 형식으로 변환. 255로 나누어 0 - 1 사이값으로 정규화
data = np.array(data, dtype="float32") / 255.0
targets = np.array(targets, dtype="float32")

# 90% 훈련용 데이터 분할. 10%는 테스트용 분할
split = train_test_split(data, targets, filenames, test_size=0.10,
	random_state=42)	# https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html

# 데이터셋 train, test 의 이미지, target label, fname 획득
(trainImages, testImages) = split[:2]	# 224x224 이미지. 720개. 80개
(trainTargets, testTargets) = split[2:4]	# bbox 데이터	
(trainFilenames, testFilenames) = split[4:]	# 파일명

trainImages = torch.tensor(trainImages, dtype = torch.float32)
testImages = torch.tensor(testImages, dtype = torch.float32)
trainTargets = torch.tensor(trainTargets, dtype = torch.float32)
testTargets = torch.tensor(testTargets, dtype = torch.float32)

train_dataset = TensorDataset(trainImages, trainTargets)
test_dataset = TensorDataset(testImages, testTargets)

train_dataloader = DataLoader(train_dataset, batch_size = config.BATCH_SIZE, shuffle = True)
test_dataloader = DataLoader(test_dataset, batch_size = config.BATCH_SIZE, shuffle = False)

# 저장될 훈련 모델 파일명
print("[INFO] saving testing filenames...")
f = open(config.TEST_FILENAMES, "w")
f.write("\n".join(testFilenames))
f.close()

# 사전 학습된 VGG16 네트워크 로딩. 전이학습. 
# FC 층만 제외하고 전체 레이어 모델 로딩.
model = models.vgg16(pretrained = True)

# FC층을 밀집층(128)-(64)-(32)-(4)로 연결
# 결론적으로 밀집층(4)가 각 bbox 좌표의 예측값이 되도록, 모델의 출력을 만듬. 
# 마지막층은 sigmoid로 활성화시킴.
for param in model.features.parameters():
    param.requires_grad = False

model.classifier = nn.Sequential(
    nn.Linear(25088, 128),
    nn.ReLU(),
    nn.Linear(128, 64),
    nn.ReLU(),
    nn.Linear(64,32),
    nn.ReLU(),
    nn.Linear(32,4),
    nn.Sigmoid()
)

# 모델 입력은 vgg.input 층 사용. 
# bbox 회귀식 계산을 위해, 모델은 미세 조정 될 것임. 
# vgg.input.shape = (None, 224, 224, 3)
# output=bboxhead.shape = (None, 4)

# ADAM으로 loss 경계하강, 최적화. 
# loss 함수는 Mean Square Error
opt = torch.optim.Adam(model.parameters(), lr=config.INIT_LR)
loss = nn.MSELoss()
model = model.to(DEVICE)
print(model)

# bbox 회귀모델 fitting을 위한 네트워크 훈련
print("[INFO] training bounding box regressor...")

all_loss = []

for i in range(config.NUM_EPOCHS):
    avg_loss = []
    print(f"Epoch {i+1} \n------------------------")
    for batch, (X, y) in enumerate(train_dataloader):
        X, y = X.to(DEVICE), y.to(DEVICE)
        X  = X.permute(0,3,1,2)
        pred = model(X)
    
        loss_value = loss(pred, y)
        opt.zero_grad()
        loss_value.backward()
        opt.step()
        avg_loss.append(loss_value.item())
        
        
        if batch % 20 ==0 :
            loss_value, current = loss_value.item(), batch * len(X)
            print(f'loss: {loss_value:>7f}, batch : {batch}')
    mean = sum(avg_loss) / len(avg_loss)
    all_loss.append(mean)    

# 학습 종료 후 훈련 모델 저장
print("[INFO] saving object detector model...")
torch.save(model,config.MODEL_PATH)	# https://www.tensorflow.org/guide/keras/save_and_serialize
# https://portal.hdfgroup.org/display/support/Download+HDFView

# 훈련 이력 저장.
N = config.NUM_EPOCHS
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0,N), all_loss, label = "train_loss")
plt.title("Bounding Box Regression Loss on Training Set")
plt.xlabel("Epoch #")
plt.ylabel("Loss")
plt.legend(loc="lower left")
plt.savefig(config.PLOT_PATH) 
