import torch.nn as nn
import torch.nn.functional as F

class OmokCNN(nn.Module):
    def __init__(self):
        super(OmokCNN, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        
        self.fc1 = nn.Linear(128 * 15 * 15, 512)
        self.fc2 = nn.Linear(512, 225) 

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        
        x = x.view(x.size(0), -1) 
        
        x = F.relu(self.fc1(x))
        x = self.fc2(x) # 마지막 출력은 Q-value이므로 활성화 함수를 쓰지 않음
        
        return x