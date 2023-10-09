"""
이 코드는 YOLOv1의 구조, 알고리즘을 잘 이해할 수 있도록, 논문을 바탕으로 개발된 PyTorch기반 오픈소스 코드임. 
Pascal VOC 데이터로 학습되며, 클래스는 20개, 격자는 7 * 7, 엥커박스는 2개임. Pascal VOC 데이터셋 사용.
참고 - Aladdin Persson, 2020, Machine Learning Collection, https://github.com/aladdinpersson/Machine-Learning-Collection/tree/master/ML/Pytorch/object_detection/YOLO
"""
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
from loss import YoloLoss
from torch.utils.tensorboard import SummaryWriter
cwd = os.getcwd()
curd = os.path.dirname(os.path.abspath(__file__)) + '/'
  
# 텐서보드 저장 폴더 설정
writer = SummaryWriter(log_dir=curd + '/runs')      # 훈련 과정 모니터링은 tensorboard에서 확인할 수 있도록 원 오픈소스를 수정함. 

seed = 123
torch.manual_seed(seed)

# 하이퍼 파라메터 설정. YOLOv5부터는 튜닝 과정이 코드에 포함되어 제공됨.
LEARNING_RATE = 2e-5
DEVICE = "cuda" if torch.cuda.is_available else "cpu"
BATCH_SIZE = 16 # GPU램이 크다면 배치크기를 높일 것. 64 in original paper but I don't have that much vram, grad accum?
WEIGHT_DECAY = 0
EPOCHS = 1000
NUM_WORKERS = 2
PIN_MEMORY = True
LOAD_MODEL = False
LOAD_MODEL_FILE = "overfit.pth.tar"
IMG_DIR = "data/images"     # 학습용 이미지, 라벨 폴더 설정
LABEL_DIR = "data/labels"

# 이미지 학습할 수 있도록 이미지 데이터 변환
class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, bboxes):
        for t in self.transforms:
            img, bboxes = t(img), bboxes

        return img, bboxes

# 학습 데이터는 448x448 크기로 변환된 후, 텐서로 변환됨. 참고로, 448은 PASCAL VOC 이미지 파일 해상도임.
transform = Compose([transforms.Resize((448, 448)), transforms.ToTensor(),])

# 학습 함수 정의
def train_fn(epoch, train_loader, model, optimizer, loss_fn):
    loop = tqdm(train_loader, leave=True)   # 훈련 과정 모니터링
    mean_loss = []

    # 배치 크기만큼 학습 루프 반복 실행
    for batch_idx, (x, y) in enumerate(loop):
        x, y = x.to(DEVICE), y.to(DEVICE)   # 텐서를 GPU 메모리로 이동시킴.
        out = model(x)                      # 이미지 데이터 입력 및 예측값 획득
        loss = loss_fn(out, y)              # 손실함수 호출해 손실계산
        mean_loss.append(loss.item())       # 손실값 배열 저장
        optimizer.zero_grad()               # 옵티마이저에서 기존 손실 경사하강을 위한 값이 이번에 누적되지 않도록 초기화
        loss.backward()                     # 손실값을 최소화하기 위한, 역전파. 가중치 조절해 각 층별 손실값 경사하강시킴.
        optimizer.step()

        if epoch % 20 == 0 and batch_idx == len(loop) - 1:  # epoch 20개 마다 학습된 모델을 pth파일로 저장함.
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss
                }, curd + f'yolo_{epoch}.pth')

        # 진행바 업데이트
        loop.set_postfix(loss=loss.item())

    m_loss = sum(mean_loss)/len(mean_loss)
    print(f"Mean loss was {m_loss}")

    return m_loss


def main():
    # YOLO 모델 생성. 논문과 동일하게 7x7 격자, bbox는 격자당 2개. 클래스는 PASCAL VOC와 동일하게 20개.
    model = Yolov1(split_size=7, num_boxes=2, num_classes=20).to(DEVICE)
    optimizer = optim.Adam(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    loss_fn = YoloLoss()    # YOLO 손실함수 생성

    if LOAD_MODEL:
        load_checkpoint(torch.load(LOAD_MODEL_FILE), model, optimizer)  # 기존 학습 모델 있으면 로딩해, 이후 전이학습처리함.

    # 학습 데이터 준비. 데이터 정규화 및 텐서 변환을 위한 트랜스폼 설정.
    train_dataset = VOCDataset(
        curd + "data/100examples.csv",
        transform=transform,
        img_dir=curd + IMG_DIR,
        label_dir=curd + LABEL_DIR,
    )   # 학습은 100개에 대해서만 처리함. 정확도 높일려면, 학습 데이터수를 늘일것.

    test_dataset = VOCDataset(
        curd + "data/test.csv", transform=transform, img_dir=curd + IMG_DIR, label_dir=curd + LABEL_DIR,
    )   # 테스트 데이터셋 로딩.

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        shuffle=True,
        drop_last=True,
    )   # 데이터 로더 정의. 배치크기, 배치데이터 로딩시 워커(작업 프로세스 수) 설정. 데이터 로딩 시 Pageable 메모리가 아닌 CPU의 고정된 PIN메모리 사용해 CUDA 데이터 복사 성능을 개선시킴(데이터 양이 많을 때 유용함). 매 에폭마다 데이터 뒤섞음. 만약, 배치의 마지막이 배치크기와 다를경우 drop시킴. 
    # https://pytorch.org/docs/stable/data.html

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        shuffle=True,
        drop_last=True,
    )

    # 학습
    for epoch in range(EPOCHS):
        # 데이터로더에서 모든 데이터 가져와, 에폭 학습 전에 모델을 통해 계산 bbox와 라벨링된 target bbox를 가져옴. 이때 IoU 허용치는 0.5, 객체유무 허용치는 0.4임.
        # 실제 학습이 진행되진 않고, model.eval()을 통해 당시의 mAP를 계산하기 위해서 호출되는 function이다.
        pred_boxes, target_boxes = get_bboxes(
            train_loader, model, iou_threshold=0.5, threshold=0.4
        )      # target_boxes=(batch index, class, objectness, cx, cy, w, h)

        # 이때의 mAP를 계산함.
        mean_avg_prec = mean_average_precision(
            pred_boxes, target_boxes, iou_threshold=0.5, box_format="midpoint"
        )
        print(f"Train mAP: {mean_avg_prec}")

        # mAP가 0.99를 넘으면 학습 종료
        base = 0.99
        if mean_avg_prec > base:   
            break

        # 학습함. 
        m_loss = train_fn(epoch, train_loader, model, optimizer, loss_fn)

        # 텐서보드 로그 파일 저장. mAP, Loss 데이터 저장.
        writer.add_scalar("mAP", mean_avg_prec, epoch)
        writer.add_scalar("Loss/train", m_loss, epoch)

    writer.close()

    # 학습 최종 모델 저장
    torch.save(model.state_dict(), curd + 'yolo_last.pth')      

if __name__ == "__main__":
    main()
