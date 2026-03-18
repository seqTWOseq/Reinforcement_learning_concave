import numpy as np
import gymnasium as gym
from gymnasium import spaces
import tkinter as tk
import time
import torch as th
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

# ==========================================
# 0. 오목 전용 맞춤형 CNN 뇌 (에러 방지용 맞춤 정장)
# ==========================================
class OmokCNN(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 256):
        super().__init__(observation_space, features_dim)
        n_input_channels = observation_space.shape[0] # 1채널
        
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )

        with th.no_grad():
            sample_tensor = th.as_tensor(observation_space.sample()[None]).float()
            n_flatten = self.cnn(sample_tensor).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.linear(self.cnn(observations))

# ==========================================
# 1. 스파링 파트너 (유치원생 & 고수)
# ==========================================
class Agent0_Random:
    def select_action(self, state):
        board_size = state.shape[1]
        valid_moves = np.argwhere(state[0] == 0)
        if len(valid_moves) == 0: return 0
        idx = np.random.choice(len(valid_moves))
        return valid_moves[idx][0] * board_size + valid_moves[idx][1]

class Agent2_White:
    def select_action(self, state):
        board_size = state.shape[1]
        valid_moves = np.argwhere(state[0] == 0)
        if len(valid_moves) == 0: return 0
        if len(valid_moves) >= board_size * board_size - 1:
            center = board_size // 2
            if state[0, center, center] == 0: return center * board_size + center
            else: return center * board_size + (center + 1)
        best_score = -float('inf')
        best_action = valid_moves[0]
        for move in valid_moves:
            r, c = move
            attack_score = self._evaluate_position(state[0], r, c, player=2) 
            defense_score = self._evaluate_position(state[0], r, c, player=1)
            total_score = attack_score + defense_score * 1.2 + np.random.uniform(0, 1)
            if total_score > best_score:
                best_score = total_score; best_action = move
        return best_action[0] * board_size + best_action[1]

    def _evaluate_position(self, state, r, c, player):
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
        score = 0
        board_size = state.shape[0]
        for dr, dc in directions:
            consecutive = 1; open_ends = 0
            for step in (1, -1):
                nr, nc = r + dr * step, c + dc * step
                while 0 <= nr < board_size and 0 <= nc < board_size and state[nr, nc] == player:
                    consecutive += 1; nr += dr * step; nc += dc * step
                if 0 <= nr < board_size and 0 <= nc < board_size and state[nr, nc] == 0: open_ends += 1
            if consecutive >= 5: score += 100000 
            elif consecutive == 4 and open_ends == 2: score += 10000
            elif consecutive == 4 and open_ends == 1: score += 1000
            elif consecutive == 3 and open_ends == 2: score += 1000
            elif consecutive == 3 and open_ends == 1: score += 100
            elif consecutive == 2 and open_ends == 2: score += 10
        return score

# ==========================================
# 2. 강화학습 체육관 (보상 강화 & CNN 시야)
# ==========================================
class OmokTrainingEnv(gym.Env):
    def __init__(self, render_mode="none", opponent_type="smart"):
        super().__init__()
        self.board_size = 15
        self.render_mode = render_mode
        self.action_space = spaces.Discrete(self.board_size ** 2)
        self.observation_space = spaces.Box(low=0, high=2, shape=(1, self.board_size, self.board_size), dtype=np.uint8)
        self.opponent = Agent0_Random() if opponent_type == "random" else Agent2_White()
        self.last_opponent_move = (7, 7)
        self.window = None
        self.cell_size, self.margin = 40, 30

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.board = np.zeros((self.board_size, self.board_size), dtype=np.uint8)
        self.last_opponent_move = (7, 7)
        return self.board.reshape(1, 15, 15), {}

    def step(self, action):
        r, c = action // self.board_size, action % self.board_size
        if self.board[r, c] != 0:
            # 이미 돌이 있는 자리면 게임을 끝내지 않고, 벌점만 주고 같은 턴을 유지
            return self.board.reshape(1, 15, 15), -1.0, False, False, {"reason": "invalid_move"}
        self.board[r, c] = 1 
        
        # 🚨 [덩우의 보너스] 상대방 마지막 돌 근처에 두면 보상 (+0.2)
        dist = abs(r - self.last_opponent_move[0]) + abs(c - self.last_opponent_move[1])
        near_bonus = 0.2 if dist <= 2 else 0.0
        
        # 패턴 보너스 (3목, 4목 등)
        pattern_score = self._evaluate_pattern(self.board, r, c, 1)
        reward = 0.1 + (pattern_score * 0.0001) + near_bonus

        if self._check_win(r, c, 1):
            return self.board.reshape(1, 15, 15), 10.0 + reward, True, False, {"reason": "AI_win"}
        if not np.any(self.board == 0):
            return self.board.reshape(1, 15, 15), 0.0, True, False, {"reason": "draw"}

        opp_action = self.opponent.select_action(self.board.reshape(1, 15, 15))
        opp_r, opp_c = opp_action // self.board_size, opp_action % self.board_size
        self.board[opp_r, opp_c] = 2
        self.last_opponent_move = (opp_r, opp_c) # 상대방 위치 기억
        
        if self._check_win(opp_r, opp_c, 2):
            return self.board.reshape(1, 15, 15), -5.0, True, False, {"reason": "Agent2_win"}
        return self.board.reshape(1, 15, 15), reward, False, False, {}

    def _check_win(self, row, col, player):
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
        for dr, dc in directions:
            count = 1
            for step in (1, -1):
                r, c = row + dr * step, col + dc * step
                while 0 <= r < self.board_size and 0 <= c < self.board_size and self.board[r, c] == player:
                    count += 1; r += dr * step; c += dc * step
            if count >= 5: return True
        return False

    def _evaluate_pattern(self, state, r, c, player):
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
        score = 0
        for dr, dc in directions:
            consecutive = 1; open_ends = 0
            for step in (1, -1):
                nr, nc = r + dr * step, c + dc * step
                while 0 <= nr < self.board_size and 0 <= nc < self.board_size and state[nr, nc] == player:
                    consecutive += 1; nr += dr * step; nc += dc * step
                if 0 <= nr < self.board_size and 0 <= nc < self.board_size and state[nr, nc] == 0: open_ends += 1
            if consecutive >= 5: score += 100000 
            elif consecutive == 4: score += 10000 if open_ends == 2 else 1000
            elif consecutive == 3: score += 1000 if open_ends == 2 else 100
        return score

    def render(self):
        if self.render_mode != "human": return
        if self.window is None:
            self.window = tk.Tk(); self.window.title("궁극의 AI 오목")
            size = (self.board_size - 1) * self.cell_size + self.margin * 2
            self.canvas = tk.Canvas(self.window, width=size, height=size, bg="#DCB35C"); self.canvas.pack()
        self.canvas.delete("all")
        for i in range(self.board_size):
            start, end = self.margin + i * self.cell_size, self.margin + (self.board_size - 1) * self.cell_size
            self.canvas.create_line(self.margin, start, end, start); self.canvas.create_line(start, self.margin, start, end)
        for r in range(self.board_size):
            for c in range(self.board_size):
                if self.board[r, c] != 0:
                    x, y = self.margin + c * self.cell_size, self.margin + r * self.cell_size
                    rad = self.cell_size // 2 - 2; color = "black" if self.board[r, c] == 1 else "white"
                    self.canvas.create_oval(x - rad, y - rad, x + rad, y + rad, fill=color, outline="black")
        self.window.update()

# ==========================================
# 3. 지휘 통제실 (학습 및 텐서보드 시각화)
# ==========================================
def train_curriculum():
    # 시각화(TensorBoard) 및 맞춤형 뇌 설정
    policy_kwargs = dict(features_extractor_class=OmokCNN, features_extractor_kwargs=dict(features_dim=256), normalize_images=False)
    TB_LOG_PATH = "./tb_omok_final/"

    print("🔥 [1단계] 유치원생 상대 특훈 시작! (텐서보드 기록 중...)")
    env1 = OmokTrainingEnv(render_mode="none", opponent_type="random")
    model = PPO("CnnPolicy", env1, verbose=1, tensorboard_log=TB_LOG_PATH, policy_kwargs=policy_kwargs)
    model.learn(total_timesteps=500000) # 50만 번
    model.save("omok_ppo_stage1")
    env1.close()
    
    print("\n🔥 [2단계] 고수 상대 실전 특훈 시작!")
    env2 = OmokTrainingEnv(render_mode="none", opponent_type="smart")
    # 1단계의 뇌를 가져와서 텐서보드 로그도 이어서 기록!
    model = PPO.load("omok_ppo_stage1", env=env2, tensorboard_log=TB_LOG_PATH)
    model.learn(total_timesteps=1000000) # 100만 번
    model.save("omok_ppo_ultimate")
    env2.close()

def play():
    env = OmokTrainingEnv(render_mode="human", opponent_type="smart")
    model = PPO.load("omok_ppo_ultimate")
    obs, _ = env.reset()
    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, _, _ = env.step(action)
        env.render(); time.sleep(0.3)
        if done: break
    time.sleep(3); env.close()

if __name__ == "__main__":
    train_curriculum() # 훈련 시작!
    play() # 훈련 끝나면 결과 보기!