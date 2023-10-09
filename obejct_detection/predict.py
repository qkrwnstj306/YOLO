# USAGE
# python predict.py --input output/test_images.txt

import config
import torch
import numpy as np
import mimetypes
import argparse
import imutils
import cv2
import os
from PIL import Image
torch.cuda.set_device(3)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using Device:", DEVICE)

curd = os.path.dirname(os.path.abspath(__file__)) + '/'

# 파이썬 입력 인자 파싱. 
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", default=curd+'dataset/images',
	help="path to input image/text file of image filenames")
args = vars(ap.parse_args())

# 입력 폴더에서 파일명 획득
input_path = args['input']
imagePaths = []
for path in os.listdir(input_path):
    if os.path.isfile(os.path.join(input_path, path)):
        imagePaths.append(input_path + '/' + path)

# 학습 모델 로딩
print("[INFO] loading object detector...")
model = torch.load(config.MODEL_PATH)
model = model.to(DEVICE)
model.eval()
# 이미지 파일을 로딩해 학습 모델에 입력. prediction 함.
for imagePath in imagePaths:
    # 이미지 로딩 후 정규화. 
    # 0축 차원 확장해, 학습 모델 입력 텐서 차원과 일치시킴.
    # 파일 로딩. 타겟 크기는 224, 224
    image = Image.open(imagePath)
    image = image.resize((224,224))
    image = np.array(image)	/ 255.0 # 이미지를 배열로 변환
    image = np.expand_dims(image, axis=0)	# (1,224,244,3)
    
    image = torch.tensor(image, dtype= torch.float).permute(0,3,1,2)
    # 예측한 후, 첫번째 출력 bbox값을 획득
    image = image.to(DEVICE)
    preds = model(image)[0]
    (startX, startY, endX, endY) = preds
    
    # 이미지 로딩해 width 리사이즈 후, 이미지 폭, 너비 획득
    image = cv2.imread(imagePath)
    image = imutils.resize(image, width=600)
    (h, w) = image.shape[:2]
    
    # bbox는 0 - 1 사이값으로 정규화되어 있으므로, 이미지 폭, 너비로 스케일 적용
    startX = int(startX * w)
    startY = int(startY * h)
    endX = int(endX * w)
    endY = int(endY * h)
    
    # 예측된 bbox를 이미지 위에 그려줌
    cv2.rectangle(image, (startX, startY), (endX, endY),
        (0, 255, 0), 2)
    
    # 이미지 출력
    cv2.imwrite("Output.jpg", image)
    print("Done")
    #cv2.imshow("Output", image)
    #cv2.waitKey(0)