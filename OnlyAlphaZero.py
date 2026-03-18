from collections import deque
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os

# ==========================================
# 0. 기본 설정
# ==========================================
MEMORY_SIZE = 10000  # 최근 만 번의 움직임을 기억할 뇌 용량
BATCH_SIZE = 128     # 한 번 공부할 때 기억 상자에서 꺼내볼 문제 개수
BOARD_SIZE = 15
ACTION_SIZE = BOARD_SIZE * BOARD_SIZE
EPISODES = 10000         # 자기 자신과 몇 판을 싸울 것인가?
EPOCHS = 10              # 모인 데이터로 신경망을 몇 번 복습할 것인가?

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🔥 학습 장치: {device}")

# ==========================================
# 1. 오목 게임 환경
# ==========================================
class GomokuGame:
    def __init__(self, size=BOARD_SIZE):
        self.size = size

    def get_initial_state(self):
        return np.zeros((self.size, self.size), dtype=np.int8)

    def get_next_state(self, state, action, player):
        row = action // self.size
        col = action % self.size
        next_state = np.copy(state)
        next_state[row, col] = player
        return next_state

    def get_valid_moves(self, state):
        return (state.reshape(-1) == 0).astype(np.uint8)

    def check_win(self, state, action):
        if action is None:
            return False
        row, col = action // self.size, action % self.size
        player = state[row, col]
        if player == 0:
            return False

        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
        for dr, dc in directions:
            count = 1
            for step in (1, -1):
                r, c = row + dr * step, col + dc * step
                while 0 <= r < self.size and 0 <= c < self.size and state[r, c] == player:
                    count += 1
                    r += dr * step
                    c += dc * step
            if count >= 5:
                return True
        return False

    def get_reward_and_ended(self, state, action):
        if self.check_win(state, action):
            return 1.0, True # 승리
        if np.sum(self.get_valid_moves(state)) == 0:
            return 0.0, True # 무승부
        return 0.0, False

    def get_canonical_form(self, state, player):
        return state * player

# ==========================================
# 2. 듀얼 헤드 ResNet (The Brain)
# ==========================================
class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        res = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x += res
        return F.relu(x)

class NeuralNet(nn.Module):
    def __init__(self, num_blocks=3, channels=64):
        super().__init__()
        self.start_conv = nn.Sequential(
            nn.Conv2d(1, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU()
        )
        
        self.res_blocks = nn.ModuleList([ResBlock(channels) for _ in range(num_blocks)])
        
        self.policy_conv = nn.Conv2d(channels, 2, kernel_size=1)
        self.policy_bn = nn.BatchNorm2d(2)
        self.policy_fc = nn.Linear(2 * BOARD_SIZE * BOARD_SIZE, ACTION_SIZE)
        
        self.value_conv = nn.Conv2d(channels, 1, kernel_size=1)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(BOARD_SIZE * BOARD_SIZE, 64)
        self.value_fc2 = nn.Linear(64, 1)

    def forward(self, x):
        x = x.view(-1, 1, BOARD_SIZE, BOARD_SIZE)
        x = self.start_conv(x)
        for block in self.res_blocks:
            x = block(x)
            
        p = F.relu(self.policy_bn(self.policy_conv(x)))
        p = p.view(-1, 2 * BOARD_SIZE * BOARD_SIZE)
        policy = self.policy_fc(p)
        
        v = F.relu(self.value_bn(self.value_conv(x)))
        v = v.view(-1, BOARD_SIZE * BOARD_SIZE)
        v = F.relu(self.value_fc1(v))
        value = torch.tanh(self.value_fc2(v))
        
        return F.log_softmax(policy, dim=1), value

# ==========================================
# 3. 순수 신경망 기반 학습 루프
# ==========================================
def execute_episode(game, model):
    """MCTS 없이 신경망의 예측 확률만으로 플레이하며 데이터를 수집합니다."""
    train_examples = []
    state = game.get_initial_state()
    current_player = 1

    step = 0
    while True:
        step += 1
        canonical_state = game.get_canonical_form(state, current_player)
        state_tensor = torch.FloatTensor(canonical_state).unsqueeze(0).to(device)
        
        # MCTS 없이 신경망이 바로 확률 계산
        with torch.no_grad():
            policy, _ = model(state_tensor)
            policy = torch.exp(policy).cpu().numpy().flatten()
            
        # 불법 수 제거 및 확률 정규화
        valid_moves = game.get_valid_moves(canonical_state)
        policy = policy * valid_moves
        sum_policy = np.sum(policy)
        
        if sum_policy > 0:
            policy /= sum_policy
        else:
            policy = valid_moves / np.sum(valid_moves)
            
        # 확률 분포에 따라 행동 선택 (탐색 유지)
        action = np.random.choice(len(policy), p=policy)
        
        # 데이터 저장 (현재 상태, '선택한 행동', 턴)
        train_examples.append([canonical_state, action, current_player])
        
        state = game.get_next_state(state, action, current_player)
        reward, is_terminal = game.get_reward_and_ended(state, action)

        if is_terminal:
            print(f"   └ 🏁 {step}수 만에 게임 종료. 보상: {reward}")
            # 승자가 정해지면, 각 턴에서 선택했던 행동이 좋은 행동이었는지(보상) 반환
            return [(x[0], x[1], reward * (1 if x[2] == current_player else -1)) for x in train_examples]
        
        current_player *= -1

def train_pure_neural():
    game = GomokuGame()
    model = NeuralNet().to(device)
    model_path = "neural_omok_latest.pth"
    
    if os.path.exists(model_path):
        print(f"🔄 '{model_path}' 파일 발견! 이어서 학습합니다.")
        model.load_state_dict(torch.load(model_path, map_location=device))
    else:
        print("🔥 처음부터 학습을 시작합니다!")

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # 🌟 핵심 1: 과거의 경험을 저장할 기억 상자 생성
    memory = deque(maxlen=MEMORY_SIZE) 

    print("🚀 순수 신경망 자가 학습을 시작합니다! (Replay Buffer 장착 완료)")
    
    for iteration in range(1, EPISODES + 1):
        print(f"\n====================================")
        print(f"🥊 [Self-Play] 에피소드 {iteration}/{EPISODES} 진행 중...")
        model.eval()
        
        # 방금 한 게임 데이터 수집 후 기억 상자에 쏟아 붓기
        episode_data = execute_episode(game, model)
        memory.extend(episode_data)
        
        # 기억 상자에 데이터가 충분히 안 쌓였으면 아직 공부 안 함!
        if len(memory) < BATCH_SIZE:
            print(f"   └ 📦 데이터 모으는 중... ({len(memory)}/{BATCH_SIZE})")
            continue
            
        print(f"🧠 [Train] 신경망 가중치 업데이트 중... (총 기억: {len(memory)}개)")
        model.train()
        
        # 🌟 핵심 2: 최근 게임 하나만 파는 게 아니라, 기억 상자에서 랜덤으로 뽑아옴!
        mini_batch = random.sample(memory, BATCH_SIZE)
        
        states, actions, target_vs = list(zip(*mini_batch))
        states = torch.FloatTensor(np.array(states)).to(device)
        actions = torch.LongTensor(np.array(actions)).to(device)
        target_vs = torch.FloatTensor(np.array(target_vs).reshape(-1, 1)).to(device)

        # 에포크를 10번에서 3번으로 줄임 (과적합 방지)
        for epoch in range(3):
            out_pis, out_vs = model(states)
            
            log_probs = out_pis.gather(1, actions.unsqueeze(1))
            
            # 🌟 핵심 3: Advantage 적용 (실제 결과 - 나의 원래 예상치)
            advantage = target_vs - out_vs.detach() 
            loss_pi = -torch.mean(log_probs * advantage)
            
            loss_v = torch.mean((target_vs - out_vs) ** 2)
            
            # 🌟 핵심 4: Entropy 추가 (다양한 수를 시도하도록 장려하는 패널티)
            probs = torch.exp(out_pis)
            entropy = -torch.mean(torch.sum(probs * out_pis, dim=1))
            
            # 엔트로피를 빼줘서(-0.01) Loss를 낮추려면 더 탐색하게 만듦
            total_loss = loss_pi + loss_v - 0.01 * entropy

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

        print(f"   └ ✅ 업데이트 완료! (Loss: {total_loss.item():.4f})")
        torch.save(model.state_dict(), model_path)

if __name__ == "__main__":
    train_pure_neural()