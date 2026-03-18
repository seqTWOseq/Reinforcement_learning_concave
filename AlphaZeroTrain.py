import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math
import random
import os
from torch.utils.tensorboard import SummaryWriter  # 🌟 텐서보드 추가

# ==========================================
# 0. 기본 설정
# ==========================================
BOARD_SIZE = 15
ACTION_SIZE = BOARD_SIZE * BOARD_SIZE
MCTS_SIMULATIONS = 600  # 한 수를 두기 위해 미리 상상해볼 미래의 수 (실전엔 400~800 권장)
EPISODES = 10000          # 자기 자신과 몇 판을 싸울 것인가?
EPOCHS = 10              # 모인 데이터로 신경망을 몇 번 복습할 것인가?

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🔥 학습 장치: {device}")

# ==========================================
# 1. 오목 게임 환경 (알파제로 맞춤형)
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
# 2. 알파제로 듀얼 헤드 ResNet (The Brain)
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

class AlphaZeroNet(nn.Module):
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
# 3. MCTS (몬테카를로 트리 탐색 - 수읽기 엔진)
# ==========================================
class Node:
    def __init__(self, parent=None, prior_prob=1.0):
        self.parent = parent
        self.children = {}
        self.visits = 0
        self.value_sum = 0
        self.prior = prior_prob

    def is_expanded(self):
        return len(self.children) > 0

    def get_ucb(self, c_puct=1.0):
        if self.visits == 0:
            q_value = 0
        else:
            q_value = self.value_sum / self.visits
        u_value = c_puct * self.prior * math.sqrt(self.parent.visits) / (1 + self.visits)
        return q_value + u_value

class MCTS:
    def __init__(self, game, model):
        self.game = game
        self.model = model

    @torch.no_grad()
    def search(self, state):
        root = Node()
        
        for _ in range(MCTS_SIMULATIONS):
            node = root
            current_state = np.copy(state)
            current_player = 1
            action_history = []

            while node.is_expanded():
                best_action = max(node.children.keys(), key=lambda a: node.children[a].get_ucb())
                node = node.children[best_action]
                current_state = self.game.get_next_state(current_state, best_action, current_player)
                action_history.append(best_action)
                current_player *= -1

            last_action = action_history[-1] if action_history else None
            reward, is_terminal = self.game.get_reward_and_ended(current_state, last_action)

            if not is_terminal:
                canonical_state = self.game.get_canonical_form(current_state, current_player)
                state_tensor = torch.FloatTensor(canonical_state).to(device)
                policy, value = self.model(state_tensor)
                policy = torch.exp(policy).cpu().numpy().flatten()
                value = value.item()

                valid_moves = self.game.get_valid_moves(current_state)
                policy = policy * valid_moves
                sum_policy = np.sum(policy)
                if sum_policy > 0:
                    policy /= sum_policy
                else:
                    policy = valid_moves / np.sum(valid_moves)

                for action, prob in enumerate(policy):
                    if valid_moves[action]:
                        node.children[action] = Node(parent=node, prior_prob=prob)
            else:
                value = reward

            while node is not None:
                node.visits += 1
                node.value_sum += value
                value = -value
                node = node.parent

        action_probs = np.zeros(ACTION_SIZE)
        for action, child in root.children.items():
            action_probs[action] = child.visits
        action_probs /= np.sum(action_probs)
        return action_probs

# ==========================================
# 4. 알파제로 학습 루프 (Self-Play & Train)
# ==========================================
def execute_episode(game, model):
    train_examples = []
    state = game.get_initial_state()
    current_player = 1
    mcts = MCTS(game, model)

    step = 0
    while True:
        step += 1
        canonical_state = game.get_canonical_form(state, current_player)
        
        pi = mcts.search(canonical_state)
        train_examples.append([canonical_state, pi, current_player])
        
        action = np.random.choice(len(pi), p=pi) if step < 10 else np.argmax(pi)
        state = game.get_next_state(state, action, current_player)
        reward, is_terminal = game.get_reward_and_ended(state, action)

        if is_terminal:
            print(f"   └ 🏁 {step}수 만에 게임 종료. 보상: {reward}")
            return [(x[0], x[1], reward * (1 if x[2] == current_player else -1)) for x in train_examples]
        
        current_player *= -1

def train_alphazero():
    game = GomokuGame()
    model = AlphaZeroNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    model_path = "alphazero_omok_latest.pth"
    start_iteration = 1 # 🌟 끊긴 부분부터 이어가기 위한 변수

    # 🌟 모델 로드 로직 수정 (기존 버전 호환 및 iteration 복구)
    if os.path.exists(model_path):
        print(f"🔄 오! '{model_path}' 파일 발견! 똑똑해진 뇌를 불러와서 이어서 학습합니다.")
        checkpoint = torch.load(model_path, map_location=device)
        
        # 딕셔너리 형태로 저장된 최신 포맷인 경우
        if isinstance(checkpoint, dict) and 'iteration' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_iteration = checkpoint['iteration'] + 1
            print(f"📈 이전 학습 기록을 찾아 {start_iteration}번째 에피소드부터 그래프를 이어갑니다!")
        else:
            # 예전에 state_dict만 저장했던 구버전 포맷인 경우
            model.load_state_dict(checkpoint)
            print("⚠️ 구버전 저장 파일을 불러왔습니다. 에피소드는 1부터 다시 카운트합니다.")
    else:
        print("🔥 저장된 가중치가 없네요. 완전 처음부터 쌩초보 상태로 학습을 시작합니다!")

    # 🌟 텐서보드 기록 매니저 생성
    writer = SummaryWriter('logs/alphazero_omok')

    print("🚀 AlphaZero 자기 주도 학습을 시작합니다!")
    
    # 🌟 start_iteration 부터 시작하도록 변경
    for iteration in range(start_iteration, EPISODES + 1):
        print(f"\n====================================")
        print(f"🥊 [Self-Play] 에피소드 {iteration}/{EPISODES} 진행 중...")
        model.eval()
        
        memory = execute_episode(game, model)
        
        print(f"🧠 [Train] 신경망 가중치 업데이트 중... (데이터 {len(memory)}개)")
        model.train()
        
        states, target_pis, target_vs = list(zip(*memory))
        states = torch.FloatTensor(np.array(states)).to(device)
        target_pis = torch.FloatTensor(np.array(target_pis)).to(device)
        target_vs = torch.FloatTensor(np.array(target_vs).reshape(-1, 1)).to(device)

        # loss 변수를 반복문 밖에서도 쓰기 위해 초기화
        loss_pi_val, loss_v_val, total_loss_val = 0, 0, 0

        for epoch in range(EPOCHS):
            out_pis, out_vs = model(states)
            
            loss_pi = -torch.sum(target_pis * out_pis) / len(states)
            loss_v = torch.sum((target_vs - out_vs) ** 2) / len(states)
            total_loss = loss_pi + loss_v

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            
            # 마지막 에포크의 loss 값을 기록하기 위해 저장
            loss_pi_val = loss_pi.item()
            loss_v_val = loss_v.item()
            total_loss_val = total_loss.item()

        print(f"   └ ✅ 업데이트 완료! (Loss: {total_loss_val:.4f})")
        
        # 🌟 텐서보드에 지표 기록 (그래프가 이어짐)
        writer.add_scalar('Loss/Total', total_loss_val, iteration)
        writer.add_scalar('Loss/Policy', loss_pi_val, iteration)
        writer.add_scalar('Loss/Value', loss_v_val, iteration)
        writer.add_scalar('Game/Steps', len(memory), iteration)
        
        # 🌟 모델 저장 로직 수정 (iteration과 optimizer 상태까지 함께 저장)
        torch.save({
            'iteration': iteration,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
        }, "alphazero_omok_latest.pth")

    # 학습이 모두 끝나면 writer 닫기
    writer.close()

if __name__ == "__main__":
    train_alphazero()