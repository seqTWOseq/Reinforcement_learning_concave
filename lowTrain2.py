import numpy as np
import gymnasium as gym
from gymnasium import spaces
import tkinter as tk
import time
import torch as th
import torch.nn as nn

from sb3_contrib import MaskablePPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

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
    """
    board[row, col]에 이미 player 돌이 놓여 있다고 가정하고 점수 계산
    """
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
    """
    빈 칸 (row, col)에 player가 둔다고 가정한 점수
    """
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
# 2. 오목 전용 CNN
# =========================================================
class OmokCNN(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 256):
        super().__init__(observation_space, features_dim)

        n_input_channels = observation_space.shape[0]  # 2채널

        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        with th.no_grad():
            sample_tensor = th.as_tensor(observation_space.sample()[None]).float()
            n_flatten = self.cnn(sample_tensor).shape[1]

        self.linear = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.linear(self.cnn(observations))


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
    """
    우선순위:
    1) 내가 바로 이길 수 있으면 둠
    2) 상대가 바로 이기는 수 있으면 무조건 막음
    3) 내가 열린 4 만들 수 있으면 둠
    4) 상대 열린 4 있으면 막음
    5) 내가 열린 3 만들 수 있으면 둠
    6) 상대 열린 3 있으면 막음
    7) 나머지는 공격+수비+중앙+인접 보너스로 선택
    """
    def select_action(self, board):
        n = board.shape[0]
        valid_moves = [tuple(x) for x in np.argwhere(board == EMPTY)]
        if not valid_moves:
            return 0

        # 초반 첫 수는 중앙 선호
        if len(valid_moves) == n * n:
            center = n // 2
            return center * n + center

        # 1) 즉승 수
        winning_moves = [(r, c) for (r, c) in valid_moves if move_wins(board, r, c, WHITE)]
        if winning_moves:
            r, c = self._pick_best_scored_move(board, winning_moves, WHITE)
            return r * n + c

        # 2) 상대 즉승 차단
        opp_winning_moves = [(r, c) for (r, c) in valid_moves if move_wins(board, r, c, BLACK)]
        if opp_winning_moves:
            r, c = self._pick_best_block(board, opp_winning_moves)
            return r * n + c

        # 각 수의 공격/수비 점수 계산
        attack_scores = {}
        defense_scores = {}
        for r, c in valid_moves:
            attack_scores[(r, c)] = score_move(board, r, c, WHITE)
            defense_scores[(r, c)] = score_move(board, r, c, BLACK)

        # 3) 내가 열린 4 만들기
        self_open_four = [mv for mv in valid_moves if attack_scores[mv] >= OPEN_FOUR]
        if self_open_four:
            r, c = max(self_open_four, key=lambda mv: attack_scores[mv] + center_bonus(*mv))
            return r * n + c

        # 4) 상대 열린 4 막기
        opp_open_four = [mv for mv in valid_moves if defense_scores[mv] >= OPEN_FOUR]
        if opp_open_four:
            r, c = max(opp_open_four, key=lambda mv: defense_scores[mv] + center_bonus(*mv))
            return r * n + c

        # 5) 내가 열린 3 만들기
        self_open_three = [mv for mv in valid_moves if attack_scores[mv] >= OPEN_THREE]
        if self_open_three:
            best_self = max(self_open_three, key=lambda mv: attack_scores[mv] + center_bonus(*mv))
            # 너무 급한 수비가 아니면 공격 진행
            if defense_scores[best_self] < OPEN_FOUR:
                r, c = best_self
                return r * n + c

        # 6) 상대 열린 3 막기
        opp_open_three = [mv for mv in valid_moves if defense_scores[mv] >= OPEN_THREE]
        if opp_open_three:
            r, c = max(opp_open_three, key=lambda mv: defense_scores[mv] + center_bonus(*mv))
            return r * n + c

        # 7) 종합 휴리스틱
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
        self.max_steps = self.board_size * self.board_size  # 에이전트 행동 기준 상한

        self.action_space = spaces.Discrete(self.board_size ** 2)
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(2, self.board_size, self.board_size),  # 내 돌 / 상대 돌
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

    # -------------------------
    # 관측 / 마스크
    # -------------------------
    def _get_obs(self):
        obs = np.zeros((2, self.board_size, self.board_size), dtype=np.float32)
        obs[0] = (self.board == BLACK).astype(np.float32)
        obs[1] = (self.board == WHITE).astype(np.float32)
        return obs

    def action_masks(self):
        # True = 선택 가능
        return (self.board.reshape(-1) == EMPTY)

    # -------------------------
    # Gym API
    # -------------------------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.board = np.zeros((self.board_size, self.board_size), dtype=np.uint8)
        self.step_count = 0
        return self._get_obs(), {}

    def step(self, action):
        self.step_count += 1

        r, c = action // self.board_size, action % self.board_size

        # 불법 착수: 게임을 바로 끝내지는 않지만 큰 패널티
        if self.board[r, c] != EMPTY:
            truncated = self.step_count >= self.max_steps
            return self._get_obs(), -2.5, False, truncated, {"reason": "invalid_move"}

        # 내 수를 두기 전, 상대의 최대 위협 수준
        opp_threat_before = self._max_potential_threat_level(WHITE)

        # 내 착수
        self.board[r, c] = BLACK

        # 공격 점수 / 방어 점수(막기)
        my_move_score = score_placed_stone(self.board, r, c, BLACK)
        my_attack_level = threat_level_from_score(my_move_score)

        opp_threat_after = self._max_potential_threat_level(WHITE)
        blocked_levels = max(0, opp_threat_before - opp_threat_after)

        # 기본 보상 설계
        reward = -0.01  # 오래 끄는 것 방지용 작은 step penalty
        reward += 0.20 * my_attack_level
        reward += 0.60 * blocked_levels

        # 상대의 큰 위협(4목급 이상)을 막았으면 추가 보너스
        if opp_threat_before >= 4 and blocked_levels > 0:
            reward += 1.2

        # 내가 승리
        if check_win_on_board(self.board, r, c, BLACK):
            reward += 15.0
            return self._get_obs(), reward, True, False, {"reason": "AI_win"}

        # 무승부
        if not np.any(self.board == EMPTY):
            return self._get_obs(), 0.0, True, False, {"reason": "draw"}

        # 시간 제한
        if self.step_count >= self.max_steps:
            return self._get_obs(), reward, False, True, {"reason": "time_limit"}

        # 상대 턴
        opp_action = self.opponent.select_action(self.board.copy())
        opp_r, opp_c = opp_action // self.board_size, opp_action % self.board_size

        # 방어 코드: 상대가 이상한 수를 골라도 빈칸 중 하나로 대체
        if self.board[opp_r, opp_c] != EMPTY:
            valid = np.argwhere(self.board == EMPTY)
            if len(valid) == 0:
                return self._get_obs(), 0.0, True, False, {"reason": "draw"}
            opp_r, opp_c = valid[0]

        self.board[opp_r, opp_c] = WHITE

        # 상대가 강한 패턴을 만들면 약간 감점
        opp_move_score = score_placed_stone(self.board, opp_r, opp_c, WHITE)
        opp_attack_level = threat_level_from_score(opp_move_score)
        reward -= 0.15 * max(0, opp_attack_level - 1)

        # 상대 승리
        if check_win_on_board(self.board, opp_r, opp_c, WHITE):
            reward -= 15.0
            return self._get_obs(), reward, True, False, {"reason": "Agent2_win"}

        # 무승부
        if not np.any(self.board == EMPTY):
            return self._get_obs(), 0.0, True, False, {"reason": "draw"}

        # 시간 제한
        if self.step_count >= self.max_steps:
            return self._get_obs(), reward, False, True, {"reason": "time_limit"}

        return self._get_obs(), reward, False, False, {}

    # -------------------------
    # 보조 계산
    # -------------------------
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

    # -------------------------
    # 렌더링
    # -------------------------
    def render(self):
        if self.render_mode != "human":
            return

        if self.window is None:
            self.window = tk.Tk()
            self.window.title("궁극의 AI 오목")
            size = (self.board_size - 1) * self.cell_size + self.margin * 2
            self.canvas = tk.Canvas(self.window, width=size, height=size, bg="#DCB35C")
            self.canvas.pack()

        self.canvas.delete("all")

        # 격자
        for i in range(self.board_size):
            start = self.margin + i * self.cell_size
            end = self.margin + (self.board_size - 1) * self.cell_size
            self.canvas.create_line(self.margin, start, end, start)
            self.canvas.create_line(start, self.margin, start, end)

        # 돌
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
# 5. 학습
# =========================================================
def train_curriculum():
    policy_kwargs = dict(
        features_extractor_class=OmokCNN,
        features_extractor_kwargs=dict(features_dim=256),
        normalize_images=False,
    )

    TB_LOG_PATH = "./tb_omok_masked/"

    print("🔥 [1단계] 랜덤 상대 특훈 시작")
    env1 = OmokTrainingEnv(render_mode="none", opponent_type="random")

    model = MaskablePPO(
        "CnnPolicy",
        env1,
        verbose=1,
        tensorboard_log=TB_LOG_PATH,
        policy_kwargs=policy_kwargs,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=256,
        gamma=0.995,
        gae_lambda=0.95,
        ent_coef=0.01,
        device="auto",
        seed=SEED,
    )
    model.learn(total_timesteps=50_000)
    model.save("omok_maskable_stage1")
    env1.close()

    print("\n🔥 [2단계] 규칙형 튜터 상대 실전 특훈 시작")
    env2 = OmokTrainingEnv(render_mode="none", opponent_type="smart")

    model = MaskablePPO.load(
        "omok_maskable_stage1",
        env=env2,
        device="auto",
        tensorboard_log=TB_LOG_PATH,
    )
    model.learn(total_timesteps=1_00_000)
    model.save("omok_maskable_ultimate")
    env2.close()


# =========================================================
# 6. 플레이
# =========================================================
def play():
    env = OmokTrainingEnv(render_mode="human", opponent_type="smart")
    model = MaskablePPO.load("omok_maskable_ultimate", env=env, device="auto")

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
    train_curriculum()
    play()