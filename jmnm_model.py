"""
NamiAI 모델 정의
- GomokuNet: 4 ResBlock, 128 filters, 3채널 입력 (내돌/상대돌/턴)
- 가중치: C:/nami/gomoku/weights/ppo_model.pth (PPO 강화학습)
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

SIZE    = 15
IN_CH   = 3
FILTERS = 128
RES_BLOCKS = 4

DEFAULT_WEIGHT = "nami.pth"


class ResBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch, ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(ch),
        )

    def forward(self, x):
        return F.relu(self.net(x) + x, inplace=True)


class NamiNet(nn.Module):
    """
    Policy-Value 듀얼헤드 ResNet
    input : (B, 3, 15, 15)
      ch0 - 내 돌 (1=있음)
      ch1 - 상대 돌 (1=있음)
      ch2 - 턴 (all 1.0)
    output: policy logits (B, 225), value (B, 1)
    """
    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(IN_CH, FILTERS, 3, padding=1, bias=False),
            nn.BatchNorm2d(FILTERS),
            nn.ReLU(inplace=True),
        )
        self.res = nn.Sequential(*[ResBlock(FILTERS) for _ in range(RES_BLOCKS)])

        self.pol_conv = nn.Sequential(
            nn.Conv2d(FILTERS, 2, 1, bias=False),
            nn.BatchNorm2d(2),
            nn.ReLU(inplace=True),
        )
        self.pol_fc = nn.Linear(2 * SIZE * SIZE, SIZE * SIZE)

        self.val_conv = nn.Sequential(
            nn.Conv2d(FILTERS, 1, 1, bias=False),
            nn.BatchNorm2d(1),
            nn.ReLU(inplace=True),
        )
        self.val_fc = nn.Sequential(
            nn.Linear(SIZE * SIZE, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 1),
            nn.Tanh(),
        )

    def forward(self, x):
        x = self.res(self.stem(x))
        p = self.pol_fc(self.pol_conv(x).flatten(1))
        v = self.val_fc(self.val_conv(x).flatten(1))
        return p, v


def board_to_tensor(state_12):
    """
    OmokEnvGUI 보드 (0=빈칸, 1=내돌, 2=상대돌) →
    NamiNet 입력 텐서 (3, 15, 15)
    """
    ch0 = (state_12 == 1).astype(np.float32)   # 내 돌
    ch1 = (state_12 == 2).astype(np.float32)   # 상대 돌
    ch2 = np.ones((SIZE, SIZE), dtype=np.float32)  # 턴 채널
    return np.stack([ch0, ch1, ch2], axis=0)


def load_nami_model(path=DEFAULT_WEIGHT, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = NamiNet().to(device)
    try:
        model.load_state_dict(torch.load(path, map_location=device))
        print(f"✅ NamiNet 가중치 로드: {path}  [{device}]")
    except FileNotFoundError:
        print(f"⚠️  가중치 없음: {path} → 랜덤 초기화로 실행")
    model.eval()
    return model, device
