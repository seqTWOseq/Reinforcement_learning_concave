import numpy as np
import gymnasium as gym
from gymnasium import spaces
import tkinter as tk
import time
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# ==========================================
# 0. 기본 설정 및 장치
# ==========================================
BOARD_SIZE = 15
ACTION_SIZE = BOARD_SIZE * BOARD_SIZE
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================
# 1. 알파제로 두뇌 (Network) & 수읽기 (MCTS)
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

class GomokuGame:
    """MCTS가 머릿속으로 시뮬레이션할 때 쓸 가상의 게임판"""
    def __init__(self, size=BOARD_SIZE):
        self.size = size

    def get_next_state(self, state, action, player):
        next_state = np.copy(state)
        next_state[action // self.size, action % self.size] = player
        return next_state

    def get_valid_moves(self, state):
        return (state.reshape(-1) == 0).astype(np.uint8)

    def check_win(self, state, action):
        if action is None: return False
        row, col = action // self.size, action % self.size
        player = state[row, col]
        if player == 0: return False
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
        for dr, dc in directions:
            count = 1
            for step in (1, -1):
                r, c = row + dr * step, col + dc * step
                while 0 <= r < self.size and 0 <= c < self.size and state[r, c] == player:
                    count += 1
                    r += dr * step; c += dc * step
            if count >= 5: return True
        return False

    def get_reward_and_ended(self, state, action):
        if self.check_win(state, action): return 1.0, True
        if np.sum(self.get_valid_moves(state)) == 0: return 0.0, True
        return 0.0, False

    def get_canonical_form(self, state, player):
        return state * player

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
        q_value = 0 if self.visits == 0 else self.value_sum / self.visits
        u_value = c_puct * self.prior * math.sqrt(self.parent.visits) / (1 + self.visits)
        return q_value + u_value

class MCTS:
    def __init__(self, game, model, simulations=100):
        self.game = game
        self.model = model
        self.simulations = simulations

    @torch.no_grad()
    def search(self, state):
        root = Node()
        for _ in range(self.simulations):
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
                if sum_policy > 0: policy /= sum_policy
                else: policy = valid_moves / np.sum(valid_moves)

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
# 2. 오목 강화학습 환경 (GUI - 그대로 사용!)
# ==========================================
class OmokEnvGUI(gym.Env):
    metadata = {"render_modes": ["ansi", "human"], "render_fps": 4}
    def __init__(self, render_mode="human"):
        super().__init__()
        self.board_size = BOARD_SIZE
        self.render_mode = render_mode
        self.action_space = spaces.Discrete(self.board_size ** 2)
        self.observation_space = spaces.Box(low=0, high=2, shape=(self.board_size, self.board_size), dtype=np.int8)
        self.board = None
        self.current_player = 1
        self.window = None
        self.cell_size, self.margin = 40, 30

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.board = np.zeros((self.board_size, self.board_size), dtype=np.int8)
        self.current_player = 1
        return self.board.copy(), {"current_player": self.current_player}

    def step(self, action):
        r, c = action // self.board_size, action % self.board_size
        if self.board[r, c] != 0:
            return self.board.copy(), -1.0, False, False, {"reason": "invalid_move", "current_player": self.current_player}
        self.board[r, c] = self.current_player
        if self._check_win(r, c, self.current_player):
            return self.board.copy(), 1.0, True, False, {"reason": "win", "winner": self.current_player}
        if not np.any(self.board == 0):
            return self.board.copy(), 0.0, True, False, {"reason": "draw", "winner": 0}
        self.current_player = 3 - self.current_player
        return self.board.copy(), 0.0, False, False, {"current_player": self.current_player}

    def _check_win(self, row, col, player):
        directions = [(0, 1), (1, 0), (1, 1), (-1, 1)]
        for dr, dc in directions:
            count = 1
            for step in (1, -1):
                r, c = row + dr * step, col + dc * step
                while 0 <= r < self.board_size and 0 <= c < self.board_size and self.board[r, c] == player:
                    count += 1
                    r += dr * step; c += dc * step
            if count >= 5: return True
        return False

    def render(self):
        if self.render_mode != "human": return
        if self.window is None:
            self.window = tk.Tk()
            self.window.title("AI 오목 대전 시뮬레이터 (AlphaZero)")
            size = (self.board_size - 1) * self.cell_size + self.margin * 2
            self.canvas = tk.Canvas(self.window, width=size, height=size, bg="#DCB35C")
            self.canvas.pack()
        self.canvas.delete("all")
        for i in range(self.board_size):
            start, end = self.margin + i * self.cell_size, self.margin + (self.board_size - 1) * self.cell_size
            self.canvas.create_line(self.margin, start, end, start)
            self.canvas.create_line(start, self.margin, start, end)
        for r in range(self.board_size):
            for c in range(self.board_size):
                if self.board[r, c] != 0:
                    x, y = self.margin + c * self.cell_size, self.margin + r * self.cell_size
                    rad = self.cell_size // 2 - 2
                    color = "black" if self.board[r, c] == 1 else "white"
                    self.canvas.create_oval(x - rad, y - rad, x + rad, y + rad, fill=color, outline="black")
        self.window.update_idletasks()
        self.window.update()

    def close(self):
        if self.window: self.window.destroy(); self.window = None

# ==========================================
# 3. 에이전트 클래스 (사용자 vs 알파제로)
# ==========================================
class HumanAgent:
    def __init__(self, env, name="Human(👤)"):
        self.name = name
        self.env = env
        self.clicked_action = None
        self.current_state = None

    def select_action(self, state):
        self.clicked_action = None
        self.current_state = state
        self.env.canvas.bind("<Button-1>", self._click_handler)
        while self.clicked_action is None:
            self.env.window.update()
            time.sleep(0.05)
        self.env.canvas.unbind("<Button-1>")
        return self.clicked_action

    def _click_handler(self, event):
        c = round((event.x - self.env.margin) / self.env.cell_size)
        r = round((event.y - self.env.margin) / self.env.cell_size)
        if 0 <= r < self.env.board_size and 0 <= c < self.env.board_size:
            action = r * self.env.board_size + c
            if self.current_state.flatten()[action] == 0:
                self.clicked_action = action

# 🌟 알파제로 에이전트 등장!
class AlphaZeroAgent:
    def __init__(self, model_path="alphazero_omok_latest.pth", name="AlphaZero(🤖)"):
        self.name = name
        self.game = GomokuGame()
        
        # 신경망 뼈대를 만들고, 저장된 가중치(.pth)를 불러와서 덮어씌움
        self.model = AlphaZeroNet().to(device)
        try:
            self.model.load_state_dict(torch.load(model_path, map_location=device))
            self.model.eval() # 평가 모드
            print(f"✅ '{model_path}' 알파제로 두뇌 장착 완료! (장치: {device})")
        except Exception as e:
            print(f"❌ 가중치 로드 실패: {e}\n임시로 랜덤하게 둡니다.")
            self.model = None

    def select_action(self, state, player_id):
        if self.model is None:
            valid = np.where(state.flatten() == 0)[0]
            return int(np.random.choice(valid)) if len(valid) > 0 else 0

        # 🌟 핵심 1: GUI 상태를 알파제로가 이해하는 상태(내돌:1, 상대:-1)로 변환
        canonical_state = np.zeros_like(state, dtype=np.int8)
        canonical_state[state == player_id] = 1        # 내 돌
        canonical_state[state == (3 - player_id)] = -1 # 상대 돌

        print("🤔 알파제로가 100번의 수읽기를 진행 중입니다...")
        
        # 🌟 핵심 2: MCTS 수읽기를 통해 가장 확실한 수를 찾음
        mcts = MCTS(self.game, self.model, simulations=100)
        action_probs = mcts.search(canonical_state)
        
        # 확률이 가장 높은 곳 선택
        best_action = np.argmax(action_probs)
        return int(best_action)


# ==========================================
# 4. 대결 실행 루프 (Arena)
# ==========================================
def main():
    env = OmokEnvGUI(render_mode="human")
    agent1 = HumanAgent(env, name="Human_Black(●)")
    agent2 = AlphaZeroAgent(model_path="alphazero_omok_latest.pth", name="AlphaZero_White(○)")
    
    state, info = env.reset()
    env.render()
    terminated = False
    
    print(f"=== ⚔️ {agent1.name} vs {agent2.name} 대결 시작 ===")
    
    while not terminated:
        current_player = info["current_player"]
        
        if current_player == 1:
            action = agent1.select_action(state)
        else:
            # 알파제로에게는 현재 판(state)과 자신이 누군지(2번 플레이어) 알려줌
            action = agent2.select_action(state, player_id=2)
            
        state, reward, terminated, _, info = env.step(action)
        env.render()
        time.sleep(0.1)

    print("\n=== 🏁 대결 종료 ===")
    winner = info.get("winner")
    if winner == 1: print(f"🎉 {agent1.name} 승리!")
    elif winner == 2: print(f"🎉 {agent2.name} 승리! (알파제로 무서움...)")
    else: print("🤝 무승부!")
        
    time.sleep(3)
    env.close()

if __name__ == "__main__":
    main()