"""
YOLOv1 모델은 이후 개발되는 YOLOv3, v5, x에 영향을 준 주요 개념을 포함하고 있음.
"""

import torch
import torch.nn as nn

""" 
YOLOv1 네트워크 구성을 유연하게 추가하도록, 그 구조를 배열과 튜플로 정의함.
튜플 구조는 (kernel_size, filters, stride, padding) 임
M 문자열은 2x2 max pooling을 의미함.  
리스트 요소는 그 안에 정의된 튜플로 표현된 층의 반복 추가를 정의함.
"""

# YOLOv1 모델 정보를 배열, 튜플 형식으로 정의함
architecture_config = [
    (7, 64, 2, 3),
    "M",
    (3, 192, 1, 1),
    "M",
    (1, 128, 1, 0),
    (3, 256, 1, 1),
    (1, 256, 1, 0),
    (3, 512, 1, 1),
    "M",
    [(1, 256, 1, 0), (3, 512, 1, 1), 4],
    (1, 512, 1, 0),
    (3, 1024, 1, 1),
    "M",
    [(1, 512, 1, 0), (3, 1024, 1, 1), 2],
    (3, 1024, 1, 1),
    (3, 1024, 2, 1),
    (3, 1024, 1, 1),
    (3, 1024, 1, 1),
]

# conv net block을 정의함. 
class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(CNNBlock, self).__init__()
        # YOLO의 conv net 구조는 conv2d, batch normal, leaky relu로 구성됨.
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)  # https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
        self.batchnorm = nn.BatchNorm2d(out_channels)
        self.leakyrelu = nn.LeakyReLU(0.1)

    def forward(self, x):
        # 모델 학습시 아래와 같은 구조로 입력 데이터 계산
        return self.leakyrelu(self.batchnorm(self.conv(x)))


# 앞서 정의된 YOLO 구조 리스트 튜플 해석해 모델 레이어 층 생성
class Yolov1(nn.Module):
    def __init__(self, in_channels=3, **kwargs):
        super(Yolov1, self).__init__()

        self.architecture = architecture_config
        self.in_channels = in_channels  # 입력 채널. 디폴트값 3.
        self.darknet = self._create_conv_layers(self.architecture)  # YOLO 구조 해석해 다크넷 모델 레이어 생성 추가.
        self.fcs = self._create_fcs(**kwargs)   # 마지막 FC 연결층 생성

    def forward(self, x):
        x = self.darknet(x) # 생성된 다크넷 모델에 데이터 입력 및 계산. x 최종 출력층 shape은 입력 (16, 3, 448, 448)에서 (16, 1024, 7, 7)이 되도록 전체 다크넷의 conv 층 파라메터가 구성되어 있음. 
        return self.fcs(torch.flatten(x, start_dim=1))  # 리턴받은 x를 입력받아, 1차원 이후로(0차원은 배치크기임) 평활화(flatten)처리 후, FC 연결층 실행. shape는 (16, 50176)임. 50176=1024*7*7

    def _create_conv_layers(self, architecture):
        # 다크넷 모델 레이어 배열 정의
        layers = []
        in_channels = self.in_channels

        # YOLO 구조 각 튜플, 리스트 열거하며 처리.
        for x in architecture:
            if type(x) == tuple:    # 튜플일 경우, CNNBLOCK 생성
                layers += [
                    CNNBlock(
                        in_channels, x[1], kernel_size=x[0], stride=x[2], padding=x[3],
                    )
                ]
                in_channels = x[1]  # 다음 추가할 레이어를 위해 입력 채널은 최종 채널로 설정. Conv2d()로 생성된 레이어의 이전 채널과 다음 채널수를 서로 일치하도록 함.

            elif type(x) == str:
                layers += [nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))] # 2x2 max pooling. https://pytorch.org/docs/stable/generated/torch.nn.MaxPool2d.html

            elif type(x) == list:   # 리스트 형식이면. 두개 튜플 입력과 추가 반복횟수 처리함
                conv1 = x[0]
                conv2 = x[1]
                num_repeats = x[2]

                for _ in range(num_repeats):
                    layers += [
                        CNNBlock(
                            in_channels,
                            conv1[1],
                            kernel_size=conv1[0],
                            stride=conv1[2],
                            padding=conv1[3],
                        )   # 각 튜플의 conv 파라메터 입력해, 레이어 생성.
                    ]
                    layers += [
                        CNNBlock(
                            conv1[1],
                            conv2[1],
                            kernel_size=conv2[0],
                            stride=conv2[2],
                            padding=conv2[3],
                        )   # 이전 채널수는 conv1[1]에서 입력받음. 나머지는 동일하게 처리.
                    ]
                    in_channels = conv2[1]

        return nn.Sequential(*layers)   # 시퀀스 모델에 생성된 레이어들 추가.

    def _create_fcs(self, split_size, num_boxes, num_classes):
        S, B, C = split_size, num_boxes, num_classes

        # In original paper this should be
        # nn.Linear(1024*S*S, 4096),
        # nn.LeakyReLU(0.1),
        # nn.Linear(4096, S*S*(B*5+C))

        # 마지막 FC 레이어 생성. 
        # 우선, 바로 이전 출력을 FC처리를 위해 평활화(Flatten)함.
        # 이와 가중치 행렬이 다음층과 밀집 연결되도록 함(Linear 입력=1024 x 7 x 7, 출력=496)
        # LeakyReLu(0.1) 사용.
        # 최종 출력은 7 x 7 x (20 + 2 * 5) 형식임. 
        # Dataset loader에서 생성된 라벨데이터와 다중손실함수로 손실값 계산되어야 하므로, 같은 텐서 차원이 되도록 하여야 함. 
        return nn.Sequential(
            nn.Flatten(),
            nn.Linear(1024 * S * S, 496),
            # nn.Dropout(0.0),
            nn.LeakyReLU(0.1),
            nn.Linear(496, S * S * (C + B * 5)),
        )
