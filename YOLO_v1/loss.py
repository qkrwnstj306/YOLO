"""
이 구현 소스는 YOLOv1의 논문을 바탕으로, Loss 함수를 정의한 레이어로, 토치 Module에서 파생받아 정의된 것임.
"""

import torch
import torch.nn as nn
from utils import intersection_over_union


class YoloLoss(nn.Module):
    """
    YOLOv1 모델의 손실함수를 정의
    """

    def __init__(self, S=7, B=2, C=20):
        super(YoloLoss, self).__init__()
        self.mse = nn.MSELoss(reduction="sum")

        """
        S 격자 갯수
        B 엥커 박스 갯수
        C 클래스 수. VOC 데이터의 경우 20임.
        """
        self.S = S
        self.B = B
        self.C = C

        # YOLO 논문에서 사용한 Loss 다중함수의 가중치 조절 값. 매직 넘버임. noobj 손실함수 람다값은 0.5, bbox 손실함수 람다값은 5.0임
        self.lambda_noobj = 0.5
        self.lambda_coord = 5

    def forward(self, predictions, target):
        # 다크넷에서 계산된 결과는 predictions에 담김. 
        # 앞서 모델 출력 정의된 대로, predictios는 (BATCH_SIZE, S*S(C+B*5) 형식으로 표현됨. 
        # target은 라벨링 데이터로 dataset loader에서 같은 텐서 차원으로 정의되어 있음에 주의할 것.
        # predictions 결과는 모델의 가중치가 곱해진 결과값임. 
        predictions = predictions.reshape(-1, self.S, self.S, self.C + self.B * 5)  

        # 계산된 bbox 1과 라벨링된 bbox2 사이의 IoU 값 계산함
        # 게산되어야 할 bbox는 2개이므로(B=2), 두개의 bbox에 대해 IoU를 계산해야 함.
        iou_b1 = intersection_over_union(predictions[..., 21:25], target[..., 21:25])  
        iou_b2 = intersection_over_union(predictions[..., 26:30], target[..., 21:25])
        ious = torch.cat([iou_b1.unsqueeze(0), iou_b2.unsqueeze(0)], dim=0) # 각 b1, b2중 최대 IoU를 가진 IoUs 최대값 Max 인덱스 얻기 위해, unsqueeze(0)처리 후, 차원0을 기준으로 연접함. https://pytorch.org/docs/stable/generated/torch.unsqueeze.html

        # 두 bbox 중 IoU 값이 가장 큰 bbox를 취함.
        iou_maxes, bestbox = torch.max(ious, dim=0) # 차원 0을 기준으로 최대 IoU를 가진 iou_maxes값과 bestbox 인덱스를 얻음. 이때 iou_maxes와 bestbox 차원은 (16, 7, 7, 1)임. https://pytorch.org/docs/stable/generated/torch.max.html
        exists_box = target[..., 20].unsqueeze(3)  # 각 격자의 objectness 얻음 (16, 7, 7). 이를 예측된 bbox 중 최대 IoU의 bbox를 얻기 위해, bestbox 차원과 맞춰야 함. 그러므로, 계산할 수 있는 (16, 7, 7, 1) 형태로 unsqueeze이용해 텐서 차원 변환 처리함.
        #격자 내에, object가 있으면 1, 아니면 0으로 설정되어 있음.
        
        # 손실을 줄일 기준 데이터인 타겟(목표) 데이터에 객체(objectness)가 존재하면 
        # 타겟 데이터와 가장 높은 IoU 가진 bbox인 box_predictions (16, 7, 7, 4)를 얻음. 
        box_predictions = exists_box * (
            (
                bestbox * predictions[..., 26:30]
                + (1 - bestbox) * predictions[..., 21:25]
            )
        )

        # 타겟 데이터의 bbox를 얻음. objectness가 없다면 box_targets은 0이 되어, 아래 코드는 계산되지 않음.
        box_targets = exists_box * target[..., 21:25]  

        # Loss 함수 정의. 각 Loss는 sample단위로 loss계산되도록 텐서 차원이 적절히 변경된 후 loss게산됨. 
        #
        # bbox W, H loss = MSE(pred box, target box). Take sqrt of width, height of boxes to ensure that
        # Image 데이터 입력된 모델을 통해 얻은 bbox의 W, H를 절대값>SQRT처리 후, 원래 box_predictions값에 대한 +/-1 부호 적용.
        box_predictions[..., 2:4] = torch.sign(box_predictions[..., 2:4]) * torch.sqrt(
            torch.abs(box_predictions[..., 2:4] + 1e-6)     # https://pytorch.org/docs/stable/generated/torch.sign.html
        )
        box_targets[..., 2:4] = torch.sqrt(box_targets[..., 2:4])   # 타겟 bbox W, H를 SQRT함

        # bbox에 대한 MSE 계산. 이를 위해, 먼저 box_predictions 텐서 (16,7,7,4)를 첫 차원부터 마지막차원 -2까지 평활화하여 텐서 (784,4)로 만든 후, 각 784의 격자별로 종속된 bbox와 타겟 bbox와 차이값을 MSE로 계산함.
        # bbox의 x, y는 MSE로 손실값 계산. W, H는 SQRT처리된 값으로 MSE 손실값 계산됨.   
        box_loss = self.mse(
            torch.flatten(box_predictions, end_dim=-2),
            torch.flatten(box_targets, end_dim=-2),
        )

        # 예측 계산된 bbox 두개 중 라벨링된 bbox와 IoU가 가장 높은 objectness값인 pred_box 얻음. 
        pred_box = (
            bestbox * predictions[..., 25:26] + (1 - bestbox) * predictions[..., 20:21]
        )

        # IoU 가장 높은 pred_box objectness 값과 라벨링된 타겟 데이터의 objectness 값과의 MSE계산을 통한 손실값 object_loss 얻음.
        object_loss = self.mse(
            torch.flatten(exists_box * pred_box),
            torch.flatten(exists_box * target[..., 20:21]),
        )

        # YOLO손실함수에 정의된 no object loss 계산. 앞서 objectness 게산한것과 반대로 계산하므로 exists_box에 -1값을 취하여 계산된 predictions, target 텐서(16,7,7,1)을 평활화해 (16,49)텐서로 변환한 후 MSE값을 계산함.
        # no object loss는 계산된 bbox 두개에 대해 처리함.
        no_object_loss = self.mse(
            torch.flatten((1 - exists_box) * predictions[..., 20:21], start_dim=1),
            torch.flatten((1 - exists_box) * target[..., 20:21], start_dim=1),
        )   # 첫번째 계산된 bbox의 no object loss

        no_object_loss += self.mse(
            torch.flatten((1 - exists_box) * predictions[..., 25:26], start_dim=1),
            torch.flatten((1 - exists_box) * target[..., 20:21], start_dim=1)
        )   # 두번째 계산된 bbox의 no object loss

        # 클래스 Loss 함수 정의. MSE계산을 위해, predictions[..., :20] 텐서(16,7,7,20)을 마지막 차원 축 -2까지 평활화해 텐서(784,20)으로 변환험, 그리고, 변환된 텐서의 각784요소별 클래스 확률20개에 대해서, 계산된 예측 클래스 벡터와 라벨링된 타겟 클래스 확률값 차이를 MSE처리해 손실값 계산.
        class_loss = self.mse(
            torch.flatten(exists_box * predictions[..., :20], end_dim=-2,),
            torch.flatten(exists_box * target[..., :20], end_dim=-2,),
        )  

        # 손실값 전체 합함. 각 손실함수의 가중치를 고려해, bbox 좌표 손실과 no_object_loss에 람다값(경험치. 매직넘버) 곱함.
        loss = (
            self.lambda_coord * box_loss  # first two rows in paper
            + object_loss  # third row in paper
            + self.lambda_noobj * no_object_loss  # forth row
            + class_loss  # fifth row
        )

        return loss # 전체 손실값 리턴해, 손실을 최소화하도록 Torch가 다크넷 네트워크 가중치를 조절할 수 있도록 함.
