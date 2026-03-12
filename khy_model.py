import torch
import torch.nn as nn
import torch.nn.functional as F

class OmokResBlock(nn.Module):
    def __init__(self, channels):
        super(OmokResBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual 
        out = F.relu(out)
        return out

class DualHeadResOmokCNN(nn.Module):
    def __init__(self, board_size=15):
        super(DualHeadResOmokCNN, self).__init__()
        self.board_size = board_size
        
        # 1. 공통 특성 추출기 (작성하신 몸통 구조)
        self.conv_input = nn.Conv2d(1, 128, kernel_size=3, padding=1)
        self.bn_input = nn.BatchNorm2d(128)
        
        self.res1 = OmokResBlock(128)
        self.res2 = OmokResBlock(128)
        self.res3 = OmokResBlock(128)
        
        # 2. Policy Network (다음 수 후보 확률 제안)
        # 1x1 Conv를 사용해 채널을 확 줄여 연산량을 최적화하는 것이 정석입니다.
        self.policy_head = nn.Sequential(
            nn.Conv2d(128, 2, kernel_size=1), 
            nn.BatchNorm2d(2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(2 * board_size * board_size, board_size * board_size)
        )
        
        # 3. Value Network (현재 판세 승률 추정: -1.0 ~ 1.0)
        self.value_head = nn.Sequential(
            nn.Conv2d(128, 1, kernel_size=1), 
            nn.BatchNorm2d(1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(1 * board_size * board_size, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Tanh()
        )
        
        self._initialize_weights()

    def forward(self, x):
        # 텐서 차원 보정
        if x.dim() == 2: x = x.unsqueeze(0).unsqueeze(0)
        elif x.dim() == 3: x = x.unsqueeze(1)
            
        # 몸통 통과
        x = F.relu(self.bn_input(self.conv_input(x)))
        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        
        # 머리 두 개로 나뉘어 출력
        policy_logits = self.policy_head(x)
        value = self.value_head(x)
        
        return policy_logits, value

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)