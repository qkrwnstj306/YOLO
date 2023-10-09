import os
import torch
import torchvision.transforms as transforms
import torch.optim as optim
import torchvision.transforms.functional as FT
from tqdm import tqdm
from torch.utils.data import DataLoader
from model import Yolov1
from dataset import VOCDataset
from utils import (
    non_max_suppression,
    mean_average_precision,
    intersection_over_union,
    cellboxes_to_boxes,
    get_bboxes,
    plot_image,
    save_checkpoint,
    load_checkpoint,
)
from PIL import Image

curd = os.path.dirname(os.path.abspath(__file__)) + '/'

DEVICE = "cuda" if torch.cuda.is_available else "cpu"

class PredictCompose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):
        for t in self.transforms:
            img = t(img)

        return img

if __name__ == '__main__':
    # 테스트 파일들 정의
    img_fnames= ['data/images/000256.jpg', 'data/images/000231.jpg', 'data/images/000215.jpg', 'data/images/000035.jpg', 'data/images/000028.jpg']

    # YOLO 모델 생성 및 학습 모델 파일 로딩
    model = Yolov1(split_size=7, num_boxes=2, num_classes=20).to(DEVICE)
    model.load_state_dict(torch.load(curd + 'yolo_last.pth'))

    for img_fname in img_fnames:
        image = Image.open(curd + img_fname)

        # 객체 탐지를 위해, 입력 데이터 448x448 변환 및 텐서 처리함.
        transform = PredictCompose([transforms.Resize((448, 448)), transforms.ToTensor(),])
        x = transform(image)

        # 변환된 입력 데이터 x를, 모델에 입력할 수 있도록 텐서 차원 변환 (배치, 채널, 폭, 높이)
        c, w, h = x.shape
        input = torch.reshape(x, (1, c, w, h))  
        input = input.to(DEVICE)    # GPU 메모리 사용

        # 모델 예측
        pred = model(input)

        # 예측된 bbox는 정규화된 값을 가지므로, 이를 디코딩함.
        # 실제 픽셀좌표로 변환.  
        batch_bboxes = cellboxes_to_boxes(pred)

        # 각 bbox들 중에서 비최대억제(NMS) 알고리즘 이용해, 중복된 bbox 중 객체 클래스 확률 제일 높은 순으로 박스 정렬. 이때 IoU 허용치는 0.5, 확률 허용치는 0.4 이상임.
        for bboxes in batch_bboxes:
            bboxes = non_max_suppression(bboxes, iou_threshold=0.5, threshold=0.4, box_format="midpoint")
            
            # 이미지를 출력할 수 있는 차원인 (w,h,c) 순열 변환. cpu 메모리 이동. bbox 출력할 수 있도록 함수에 전달
            plot_image(x.permute(1,2,0).to("cpu"), bboxes)
