"""
Creates a Pytorch dataset to load the Pascal VOC dataset
"""

import torch
import os
import pandas as pd
from PIL import Image


class VOCDataset(torch.utils.data.Dataset): # Dataset 파생 클래스 정의
    def __init__(
        self, csv_file, img_dir, label_dir, S=7, B=2, C=20, transform=None,
    ):  # 생성자
        # 라벨링 데이터 파일 읽음
        self.annotations = pd.read_csv(csv_file)
        self.img_dir = img_dir      # 학습 데이터 파일 폴더
        self.label_dir = label_dir  # 라벨 데이터 파일 폴더
        self.transform = transform  # 데이터 변환기 설정
        self.S = S                  # YOLO 격자 갯수, original image -> 7 by 7 Grid cell로 compression
        self.B = B                  # 격자별 경계상자 갯수. 이후 버전에서 앵커박스로 변경됨
        self.C = C                  # 경계상자별 클래스 수

    def __len__(self):  # Overriding
        return len(self.annotations)    # 학습 데이터 갯수

    def __getitem__(self, index):   # Overriding
        # 라벨 파일 폴더에서 index에 대한 라벨 파일명 경로 획득
        label_path = os.path.join(self.label_dir, self.annotations.iloc[index, 1])

        # 라벨 파일 파싱해 bbox 리스트 획득
        boxes = []
        with open(label_path) as f:
            for label in f.readlines():
                # 클래스 라벨, x, y, 폭, 높이
                class_label, x, y, width, height = [
                    float(x) if float(x) != int(float(x)) else int(x)
                    for x in label.replace("\n", "").split()
                ]

                # 경계박스 리스트 추가
                boxes.append([class_label, x, y, width, height])

        # 이미지 파일명 획득
        img_path = os.path.join(self.img_dir, self.annotations.iloc[index, 0])
        image = Image.open(img_path)
        boxes = torch.tensor(boxes) # 학습될 수 있도록 bbox 텐서 변환

        # 이미지 448x448 해상도로 변환 후 텐서 변환
        # 경계박스형식 = class, cx, cy, w, h
        if self.transform:
            image, boxes = self.transform(image, boxes) 

        # 로딩된 각 bbox별로 학습 데이터를 모델 예측 결과와 비교해 손실값 계산할 수 있도록
        # 모델 계산 출력과 동일한 텐서 차원으로 맞춰주어, 데이터를 label_matrix에 넣어줌.
        label_matrix = torch.zeros((self.S, self.S, self.C + 5 * self.B))   # 격자 7x7, 클래스수 + (bbox 4개 + objectness) * 경계상자수 2개
        for box in boxes:
            class_label, x, y, width, height = box.tolist() # x,y 는 grid cell 기준 | width, heigh는 전체 이미지 기준이다.
            class_label = int(class_label)

            # x, y는 정규화된 bbox 중심좌표 값. 격자갯수를 곱해서 x, y에 대응되는 격자 index를 구함.
            i, j = int(self.S * y), int(self.S * x)
            # 격자 인덱스에 대한, x, y의 상대좌표를 구함. 이를 위해, 격자좌표계로 변환된 x, y에서 구해진 격자 인덱스 i, j를 빼줌. 
            x_cell, y_cell = self.S * x - j, self.S * y - i     

            # 정규화된 bbox의 width, height를 격자좌표계로 변환.
            width_cell, height_cell = (
                width * self.S,
                height * self.S,
            )

            # 라벨링된 bbox에 x, y에 대응하는 각 격자 i, j의 텐서에 bbox, class를 값을 설정함. 
            # 이를 위해, 먼저 label_matrix의 objectness값이 이미 할당되어 있는 지 확인.
            if label_matrix[i, j, 20] == 0: # 해당 i, j 격자에 bbox 객체 정보가 할당된 것이 없으면
                # i, j에 bbox와 객체가 있으므로, objectness 값은 1로 할당.
                label_matrix[i, j, 20] = 1
                
                # 객체의 bbox 값을 격자 i, j의 상대좌표 x_cell, y_cell로 설정하고, 격자 좌표계로 bbox의 폭, 너비 설정.
                box_coordinates = torch.tensor(
                    [x_cell, y_cell, width_cell, height_cell]
                )

                # 객체 bbox값은 21-25까지 텐서로 저장
                label_matrix[i, j, 21:25] = box_coordinates

                # one hot 인코딩으로 class 라벨에 해당하는 벡터 요소 부분에 1로 설정
                label_matrix[i, j, class_label] = 1

        return image, label_matrix      # 학습 데이터와 라벨 데이터 리턴.
