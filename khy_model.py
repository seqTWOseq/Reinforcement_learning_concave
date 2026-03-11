import torch
import torch.nn as nn
import torch.nn.functional as F

class OmokResBlock(nn.Module):
    def __init__(self, channels):
        super(OmokResBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)  # 배치 정규화 추가
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)  # 배치 정규화 추가

    def forward(self, x):
        residual = x  # 입력값을 기억해둠 (지름길)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        
        out += residual  # 핵심: 찌그러진 출력값에 원래 입력값을 더해줌!
        out = F.relu(out)
        return out

class OmokCNN(nn.Module):
    def __init__(self):
        super(OmokCNN, self).__init__()
        
        # 1. 초기 시각 피질 (채널 수를 128로 넉넉하게 시작)
        self.conv_input = nn.Conv2d(1, 128, kernel_size=3, padding=1)
        self.bn_input = nn.BatchNorm2d(128)
        
        # 2. 잔차 블록 연결 (여기서 패턴을 깊게 인식하지만 기울기는 소실되지 않음)
        self.res1 = OmokResBlock(128)
        self.res2 = OmokResBlock(128)
        self.res3 = OmokResBlock(128) # 필요시 더 깊게 쌓아도 안전합니다
        
        # 3. 가치 판단 네트워크
        self.fc1 = nn.Linear(128 * 15 * 15, 512)
        self.bn_fc = nn.BatchNorm1d(512) # FC 레이어에도 정규화 적용
        self.fc2 = nn.Linear(512, 225)
        
        # ReLU에 최적화된 He 가중치 초기화 적용
        self._initialize_weights()

    def forward(self, x):
        if x.dim() == 2: x = x.unsqueeze(0).unsqueeze(0)
        elif x.dim() == 3: x = x.unsqueeze(1)
            
        x = F.relu(self.bn_input(self.conv_input(x)))
        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        
        x = x.view(x.size(0), -1)
        x = F.relu(self.bn_fc(self.fc1(x)))
        q_values = self.fc2(x)
        
        return q_values

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)