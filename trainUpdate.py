import os
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import tkinter as tk
import time
import torch as th
import torch.nn as nn

from sb3_contrib import MaskablePPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.callbacks import CheckpointCallback # 🌟 중간 저장을 위한 콜백 추가!

# =========================================================
# 0. 공통 설정
# =========================================================
SEED = 2026
np.random.seed(SEED)
th.manual_seed(SEED)

BOARD_SIZE = 15
EMPTY = 0
BLACK = 1   # 학습 에이전트
WHITE = 2   # 상대

# 패턴 점수 기준
WIN_SCORE = 1_000_000
OPEN_FOUR = 100_000
CLOSED_FOUR = 10_000
OPEN_THREE = 5_000
CLOSED_THREE = 500
OPEN_TWO = 50

DIRECTIONS = [(0, 1), (1, 0), (1, 1), (1, -1)]


# =========================================================
# 1. 유틸 함수
# =========================================================
def in_bounds(r, c, board_size=BOARD_SIZE):
    return 0 <= r < board_size and 0 <= c < board_size


def check_win_on_board(board, row, col, player):
    for dr, dc in DIRECTIONS:
        count = 1
        for step in (1, -1):
            r, c = row + dr * step, col + dc * step
            while in_bounds(r, c, board.shape[0]) and board[r, c] == player:
                count += 1
                r += dr * step
                c += dc * step
        if count >= 5:
            return True
    return False


def _line_count_and_open_ends(board, row, col, player, dr, dc):
    count = 1
    open_ends = 0
    n = board.shape[0]

    for step in (1, -1):
        r, c = row + dr * step, col + dc * step
        while 0 <= r < n and 0 <= c < n and board[r, c] == player:
            count += 1
            r += dr * step
            c += dc * step

        if 0 <= r < n and 0 <= c < n and board[r, c] == EMPTY:
            open_ends += 1

    return count, open_ends


def score_placed_stone(board, row, col, player):
    score = 0
    for dr, dc in DIRECTIONS:
        count, open_ends = _line_count_and_open_ends(board, row, col, player, dr, dc)

        if count >= 5:
            score += WIN_SCORE
        elif count == 4 and open_ends == 2:
            score += OPEN_FOUR
        elif count == 4 and open_ends == 1:
            score += CLOSED_FOUR
        elif count == 3 and open_ends == 2:
            score += OPEN_THREE
        elif count == 3 and open_ends == 1:
            score += CLOSED_THREE
        elif count == 2 and open_ends == 2:
            score += OPEN_TWO

    return score


def score_move(board, row, col, player):
    if board[row, col] != EMPTY:
        return -1

    board[row, col] = player
    score = score_placed_stone(board, row, col, player)
    board[row, col] = EMPTY
    return score


def move_wins(board, row, col, player):
    if board[row, col] != EMPTY:
        return False
    board[row, col] = player
    win = check_win_on_board(board, row, col, player)
    board[row, col] = EMPTY
    return win


def threat_level_from_score(score):
    if score >= WIN_SCORE:
        return 5
    if score >= OPEN_FOUR:
        return 4
    if score >= CLOSED_FOUR:
        return 3
    if score >= OPEN_THREE:
        return 2
    if score >= CLOSED_THREE:
        return 1
    return 0


def center_bonus(row, col, board_size=BOARD_SIZE):
    center = board_size // 2
    dist = abs(row - center) + abs(col - center)
    return max(0.0, (board_size - dist) * 0.1)


def adjacent_stone_bonus(board, row, col):
    bonus = 0.0
    n = board.shape[0]
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            nr, nc = row + dr, col + dc
            if 0 <= nr < n and 0 <= nc < n and board[nr, nc] != EMPTY:
                bonus += 0.15
    return bonus


# =========================================================
# 2. 오목 전용 ResNet (업그레이드된 신경망)
# =========================================================
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        out = self.relu(out)
        return out


class OmokResNet(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 256, num_blocks: int = 4):
        super().__init__(observation_space, features_dim)

        n_input_channels = observation_space.shape[0]
        base_channels = 64

        self.conv_initial = nn.Sequential(
            nn.Conv2d(n_input_channels, base_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True)
        )

        blocks = []
        for _ in range(num_blocks):
            blocks.append(ResidualBlock(base_channels))
        self.res_blocks = nn.Sequential(*blocks)

        with th.no_grad():
            sample_tensor = th.as_tensor(observation_space.sample()[None]).float()
            n_flatten = self.res_blocks(self.conv_initial(sample_tensor)).shape.numel() // sample_tensor.shape[0]

        self.flatten = nn.Flatten()
        self.linear = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, observations: th.Tensor) -> th.Tensor:
        x = self.conv_initial(observations)
        x = self.res_blocks(x)
        x = self.flatten(x)
        return self.linear(x)


# =========================================================
# 3. 스파링 파트너
# =========================================================
class Agent0_Random:
    def select_action(self, board):
        valid_moves = np.argwhere(board == EMPTY)
        if len(valid_moves) == 0:
            return 0
        idx = np.random.choice(len(valid_moves))
        r, c = valid_moves[idx]
        return r * board.shape[0] + c


class Agent2_RuleBasedTutor:
    def select_action(self, board):
        n = board.shape[0]
        valid_moves = [tuple(x) for x in np.argwhere(board == EMPTY)]
        if not valid_moves:
            return 0

        if len(valid_moves) == n * n:
            center = n // 2
            return center * n + center

        winning_moves = [(r, c) for (r, c) in valid_moves if move_wins(board, r, c, WHITE)]
        if winning_moves:
            r, c = self._pick_best_scored_move(board, winning_moves, WHITE)
            return r * n + c

        opp_winning_moves = [(r, c) for (r, c) in valid_moves if move_wins(board, r, c, BLACK)]
        if opp_winning_moves:
            r, c = self._pick_best_block(board, opp_winning_moves)
            return r * n + c

        attack_scores = {}
        defense_scores = {}
        for r, c in valid_moves:
            attack_scores[(r, c)] = score_move(board, r, c, WHITE)
            defense_scores[(r, c)] = score_move(board, r, c, BLACK)

        self_open_four = [mv for mv in valid_moves if attack_scores[mv] >= OPEN_FOUR]
        if self_open_four:
            r, c = max(self_open_four, key=lambda mv: attack_scores[mv] + center_bonus(*mv))
            return r * n + c

        opp_open_four = [mv for mv in valid_moves if defense_scores[mv] >= OPEN_FOUR]
        if opp_open_four:
            r, c = max(opp_open_four, key=lambda mv: defense_scores[mv] + center_bonus(*mv))
            return r * n + c

        self_open_three = [mv for mv in valid_moves if attack_scores[mv] >= OPEN_THREE]
        if self_open_three:
            best_self = max(self_open_three, key=lambda mv: attack_scores[mv] + center_bonus(*mv))
            if defense_scores[best_self] < OPEN_FOUR:
                r, c = best_self
                return r * n + c

        opp_open_three = [mv for mv in valid_moves if defense_scores[mv] >= OPEN_THREE]
        if opp_open_three:
            r, c = max(opp_open_three, key=lambda mv: defense_scores[mv] + center_bonus(*mv))
            return r * n + c

        best_score = -float("inf")
        best_move = valid_moves[0]

        for r, c in valid_moves:
            total_score = (
                attack_scores[(r, c)]
                + defense_scores[(r, c)] * 1.25
                + center_bonus(r, c, n)
                + adjacent_stone_bonus(board, r, c)
                + np.random.uniform(0, 0.05)
            )
            if total_score > best_score:
                best_score = total_score
                best_move = (r, c)

        return best_move[0] * n + best_move[1]

    def _pick_best_scored_move(self, board, moves, player):
        return max(
            moves,
            key=lambda mv: score_move(board, mv[0], mv[1], player) + center_bonus(mv[0], mv[1], board.shape[0])
        )

    def _pick_best_block(self, board, moves):
        return max(
            moves,
            key=lambda mv: score_move(board, mv[0], mv[1], BLACK) + center_bonus(mv[0], mv[1], board.shape[0])
        )


# =========================================================
# 4. 강화학습 환경
# =========================================================
class OmokTrainingEnv(gym.Env):
    metadata = {"render_modes": ["none", "human"]}

    def __init__(self, render_mode="none", opponent_type="smart"):
        super().__init__()

        self.board_size = BOARD_SIZE
        self.render_mode = render_mode
        self.max_steps = self.board_size * self.board_size

        self.action_space = spaces.Discrete(self.board_size ** 2)
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(2, self.board_size, self.board_size),
            dtype=np.float32
        )

        if opponent_type == "random":
            self.opponent = Agent0_Random()
        else:
            self.opponent = Agent2_RuleBasedTutor()

        self.board = np.zeros((self.board_size, self.board_size), dtype=np.uint8)
        self.window = None
        self.canvas = None
        self.cell_size, self.margin = 40, 30
        self.step_count = 0

    def _get_obs(self):
        obs = np.zeros((2, self.board_size, self.board_size), dtype=np.float32)
        obs[0] = (self.board == BLACK).astype(np.float32)
        obs[1] = (self.board == WHITE).astype(np.float32)
        return obs

    def action_masks(self):
        return (self.board.reshape(-1) == EMPTY)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.board = np.zeros((self.board_size, self.board_size), dtype=np.uint8)
        self.step_count = 0
        return self._get_obs(), {}

    def step(self, action):
        self.step_count += 1
        r, c = action // self.board_size, action % self.board_size

        if self.board[r, c] != EMPTY:
            truncated = self.step_count >= self.max_steps
            return self._get_obs(), -2.5, False, truncated, {"reason": "invalid_move"}

        opp_threat_before = self._max_potential_threat_level(WHITE)
        self.board[r, c] = BLACK

        my_move_score = score_placed_stone(self.board, r, c, BLACK)
        my_attack_level = threat_level_from_score(my_move_score)

        opp_threat_after = self._max_potential_threat_level(WHITE)
        blocked_levels = max(0, opp_threat_before - opp_threat_after)

        reward = -0.01
        reward += 0.20 * my_attack_level
        reward += 0.60 * blocked_levels

        if opp_threat_before >= 4 and blocked_levels > 0:
            reward += 1.2

        if check_win_on_board(self.board, r, c, BLACK):
            reward += 15.0
            return self._get_obs(), reward, True, False, {"reason": "AI_win"}

        if not np.any(self.board == EMPTY):
            return self._get_obs(), 0.0, True, False, {"reason": "draw"}

        if self.step_count >= self.max_steps:
            return self._get_obs(), reward, False, True, {"reason": "time_limit"}

        opp_action = self.opponent.select_action(self.board.copy())
        opp_r, opp_c = opp_action // self.board_size, opp_action % self.board_size

        if self.board[opp_r, opp_c] != EMPTY:
            valid = np.argwhere(self.board == EMPTY)
            if len(valid) == 0:
                return self._get_obs(), 0.0, True, False, {"reason": "draw"}
            opp_r, opp_c = valid[0]

        self.board[opp_r, opp_c] = WHITE

        opp_move_score = score_placed_stone(self.board, opp_r, opp_c, WHITE)
        opp_attack_level = threat_level_from_score(opp_move_score)
        reward -= 0.15 * max(0, opp_attack_level - 1)

        if check_win_on_board(self.board, opp_r, opp_c, WHITE):
            reward -= 15.0
            return self._get_obs(), reward, True, False, {"reason": "Agent2_win"}

        if not np.any(self.board == EMPTY):
            return self._get_obs(), 0.0, True, False, {"reason": "draw"}

        if self.step_count >= self.max_steps:
            return self._get_obs(), reward, False, True, {"reason": "time_limit"}

        return self._get_obs(), reward, False, False, {}

    def _max_potential_threat_level(self, player):
        valid_moves = np.argwhere(self.board == EMPTY)
        if len(valid_moves) == 0:
            return 0
        best_score = 0
        for r, c in valid_moves:
            s = score_move(self.board, r, c, player)
            if s > best_score:
                best_score = s
        return threat_level_from_score(best_score)

    def render(self):
        if self.render_mode != "human":
            return
        if self.window is None:
            self.window = tk.Tk()
            self.window.title("궁극의 AI 오목 - ResNet Ver.")
            size = (self.board_size - 1) * self.cell_size + self.margin * 2
            self.canvas = tk.Canvas(self.window, width=size, height=size, bg="#DCB35C")
            self.canvas.pack()

        self.canvas.delete("all")
        for i in range(self.board_size):
            start = self.margin + i * self.cell_size
            end = self.margin + (self.board_size - 1) * self.cell_size
            self.canvas.create_line(self.margin, start, end, start)
            self.canvas.create_line(start, self.margin, start, end)

        for r in range(self.board_size):
            for c in range(self.board_size):
                if self.board[r, c] != EMPTY:
                    x = self.margin + c * self.cell_size
                    y = self.margin + r * self.cell_size
                    rad = self.cell_size // 2 - 2
                    color = "black" if self.board[r, c] == BLACK else "white"
                    self.canvas.create_oval(
                        x - rad, y - rad, x + rad, y + rad,
                        fill=color, outline="black"
                    )
        self.window.update()

    def close(self):
        if self.window is not None:
            self.window.destroy()
            self.window = None
            self.canvas = None


# =========================================================
# 5. 추가 이어서 학습 (무한 반복 업데이트 전용)
# =========================================================
def train_continue():
    TB_LOG_PATH = "./tb_omok_resnet/"
    CHECKPOINT_DIR = "./checkpoints/"
    BASE_MODEL = "omok_resnet_ultimate"
    LATEST_MODEL = "omok_resnet_latest" # 🌟 덮어쓰기용 최신 파일 이름
    
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    env = OmokTrainingEnv(render_mode="none", opponent_type="smart")

    checkpoint_callback = CheckpointCallback(
        save_freq=50_000,
        save_path=CHECKPOINT_DIR,
        name_prefix="omok_update"
    )

    # 🌟 핵심: 최신 파일이 있으면 그걸 부르고, 없으면 베이스 모델 부르기!
    if os.path.exists(f"{LATEST_MODEL}.zip"):
        print(f"🔄 [무한 특훈] 최신 모델 '{LATEST_MODEL}'을 불러와서 이어서 학습합니다!")
        model = MaskablePPO.load(LATEST_MODEL, env=env, device="auto", tensorboard_log=TB_LOG_PATH)
    else:
        print(f"🔥 [추가 특훈 1회차] 베이스 모델 '{BASE_MODEL}'을 불러와서 첫 추가 학습을 시작합니다!")
        try:
            model = MaskablePPO.load(BASE_MODEL, env=env, device="auto", tensorboard_log=TB_LOG_PATH)
        except Exception as e:
            print(f"❌ '{BASE_MODEL}.zip' 파일이 없어! 150만 번 기본 학습부터 먼저 끝내야 해: {e}")
            return

    # 100만 번 이어 달리기
    model.learn(
        total_timesteps=1_000_000, 
        callback=checkpoint_callback, 
        reset_num_timesteps=False
    )

    # 🌟 다 끝나면 최신 모델 이름으로 덮어쓰기!
    model.save(LATEST_MODEL)
    print(f"\n🎉 100만 번 추가 학습 완수! '{LATEST_MODEL}' 파일로 무사히 저장(업데이트)되었습니다.")
    env.close()


# =========================================================
# 6. 플레이
# =========================================================
def play():
    env = OmokTrainingEnv(render_mode="human", opponent_type="smart")
    
    # 플레이할 때는 가장 최신 버전(v2)을 불러오도록 수정
    try:
        model = MaskablePPO.load("omok_resnet_ultimate_v2", env=env, device="auto")
        print("✅ 최신 모델(v2)로 대결을 시작합니다!")
    except:
        model = MaskablePPO.load("omok_resnet_ultimate", env=env, device="auto")
        print("✅ 기존 모델로 대결을 시작합니다!")

    obs, _ = env.reset()
    env.render()
    time.sleep(0.5)

    while True:
        action, _ = model.predict(
            obs,
            deterministic=True,
            action_masks=env.action_masks()
        )

        obs, reward, terminated, truncated, info = env.step(int(action))
        env.render()
        time.sleep(0.25)

        if terminated or truncated:
            print("게임 종료:", info, "reward =", reward)
            break

    time.sleep(2)
    env.close()


# =========================================================
# 7. 실행
# =========================================================
if __name__ == "__main__":
    # 훈련 처음부터 돌리는 건 주석 처리!
    # train_curriculum() 
    
    # 이어서 훈련하는 함수 실행
    train_continue()

    play()