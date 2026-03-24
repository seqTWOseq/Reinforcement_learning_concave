import argparse
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import tkinter as tk
from tkinter import font as tkfont
import time
import os
import glob
import random
from collections import deque

# PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

# =============================================================================
# 1. 오목 환경 (Multi-channel 관측, Action Masking, 착수 순서 추적)
# =============================================================================

def make_obs_from_board(board, current_player):
    """보드와 현재 플레이어로 Multi-channel 관측 생성. (3, 15, 15)"""
    size = board.shape[0]
    my_stones = (board == current_player).astype(np.float32)
    opp_stones = (board == 3 - current_player).astype(np.float32)
    turn_channel = np.full((size, size), float(current_player == 1), dtype=np.float32)
    return np.stack([my_stones, opp_stones, turn_channel], axis=0)

def get_action_mask(board):
    """빈 칸만 1, 나머지 0. shape (225,)"""
    return (board.flatten() == 0).astype(np.float32)


def _max_line_length(board: np.ndarray, player: int, row: int, col: int) -> int:
    """해당 위치에 놓였다고 가정할 때, 4방향 중 가장 긴 연속 돌 개수."""
    board_size = board.shape[0]
    directions = [(0, 1), (1, 0), (1, 1), (-1, 1)]
    max_len = 1
    for dr, dc in directions:
        count = 1
        for step in (1, -1):
            r, c = row + dr * step, col + dc * step
            while 0 <= r < board_size and 0 <= c < board_size and board[r, c] == player:
                count += 1
                r += dr * step
                c += dc * step
        if count > max_len:
            max_len = count
    return max_len


def _find_opponent_open_four_blocks(board: np.ndarray, opp: int, board_size: int) -> set[int]:
    """
    상대가 OOㅁO (4목인데 가운데 빈 칸) 형태인 위치 ㅁ을 찾아, 그 빈 칸의 action 인덱스 집합 반환.
    ㅁ에 두면 상대가 다음 수에 5목이 되므로 반드시 막아야 함.
    """
    blocks = set()
    directions = [(0, 1), (1, 0), (1, 1), (-1, 1)]
    for dr, dc in directions:
        for r in range(board_size):
            for c in range(board_size):
                # (r,c)부터 5칸이 유효한지
                cells = []
                for i in range(5):
                    rr, cc = r + i * dr, c + i * dc
                    if 0 <= rr < board_size and 0 <= cc < board_size:
                        cells.append((rr, cc, board[rr, cc]))
                    else:
                        cells = []
                        break
                if len(cells) != 5:
                    continue
                # 5칸 중 정확히 4개가 상대 돌, 1개가 빈 칸
                opp_count = sum(1 for _, _, v in cells if v == opp)
                empty_count = sum(1 for _, _, v in cells if v == 0)
                if opp_count != 4 or empty_count != 1:
                    continue
                # 빈 칸 위치가 "가운데" 형태: OOㅁO, OㅁOO, OOOㅁ 중 하나
                empty_pos = None
                for i, (rr, cc, v) in enumerate(cells):
                    if v == 0:
                        empty_pos = (rr, cc)
                        break
                if empty_pos is not None:
                    blocks.add(empty_pos[0] * board_size + empty_pos[1])
    return blocks


def _find_opponent_open_three_blocks(board: np.ndarray, opp: int, board_size: int) -> set[int]:
    """
    상대가 '열린 3'(양끝 빈 3연속)인 줄을 찾아, 그 양끝 빈 칸(막을 수 있는 위치)의 action 집합 반환.
    열린 3은 다음 수에 열린 4가 되므로 반드시 한쪽 끝을 막아야 함. 닫힌 3보다 우선 막는다.
    """
    blocks = set()
    directions = [(0, 1), (1, 0), (1, 1), (-1, 1)]
    for dr, dc in directions:
        for r in range(board_size):
            for c in range(board_size):
                cells = []
                for i in range(5):
                    rr, cc = r + i * dr, c + i * dc
                    if 0 <= rr < board_size and 0 <= cc < board_size:
                        cells.append((rr, cc, board[rr, cc]))
                    else:
                        cells = []
                        break
                if len(cells) != 5:
                    continue
                # 패턴: 빈칸-상대-상대-상대-빈칸 (열린 3)
                vals = [v for _, _, v in cells]
                if vals[0] == 0 and vals[1] == opp and vals[2] == opp and vals[3] == opp and vals[4] == 0:
                    blocks.add(cells[0][0] * board_size + cells[0][1])
                    blocks.add(cells[4][0] * board_size + cells[4][1])
    return blocks


def _creates_open_three(board: np.ndarray, player: int, row: int, col: int) -> bool:
    """
    (row, col)에 돌을 두었을 때 '열린 3' 패턴(양끝이 비어있는 3연속)이 생기는지 대략 판정.
    정확한 전수검사보다는, 한 방향에서 길이 3이고 양끝이 비어있는 경우를 찾는다.
    """
    board_size = board.shape[0]
    directions = [(0, 1), (1, 0), (1, 1), (-1, 1)]
    for dr, dc in directions:
        count = 1
        empty_before = False
        empty_after = False

        # 정방향
        r, c = row + dr, col + dc
        while 0 <= r < board_size and 0 <= c < board_size and board[r, c] == player:
            count += 1
            r += dr
            c += dc
        if 0 <= r < board_size and 0 <= c < board_size and board[r, c] == 0:
            empty_after = True

        # 역방향
        r, c = row - dr, col - dc
        while 0 <= r < board_size and 0 <= c < board_size and board[r, c] == player:
            count += 1
            r -= dr
            c -= dc
        if 0 <= r < board_size and 0 <= c < board_size and board[r, c] == 0:
            empty_before = True

        if count == 3 and empty_before and empty_after:
            return True
    return False


def _get_rule_move_lists(board: np.ndarray, current_player: int, mask: np.ndarray):
    """
    규칙 우선순위 1~7에 해당하는 행동 목록들을 반환.
    반환: (winning_moves, block_win_moves, block_open_four_moves, block_four_moves,
           block_open_three_moves, create_four_moves, create_open_three_moves)
    board는 복사본으로 넘길 것 (내부에서 수정함).
    """
    board_size = board.shape[0]
    valid_actions = np.where(mask > 0.5)[0]
    me = current_player
    opp = 3 - current_player

    winning_moves = []
    block_win_moves = []
    block_open_four_moves = []
    block_four_moves = []
    block_open_three_moves = []
    create_four_moves = []
    create_open_three_moves = []

    open_four_blocks = _find_opponent_open_four_blocks(board, opp, board_size)
    open_three_blocks = _find_opponent_open_three_blocks(board, opp, board_size)
    valid_set = set(valid_actions)
    block_open_four_moves = [a for a in open_four_blocks if a in valid_set]
    block_open_three_moves = [a for a in open_three_blocks if a in valid_set]

    for a in valid_actions:
        r, c = divmod(a, board_size)
        board[r, c] = me
        if _max_line_length(board, me, r, c) >= 5:
            winning_moves.append(a)
        else:
            max_len_me = _max_line_length(board, me, r, c)
            if max_len_me == 4:
                create_four_moves.append(a)
            elif _creates_open_three(board, me, r, c):
                create_open_three_moves.append(a)
        board[r, c] = 0

        board[r, c] = opp
        max_len_opp = _max_line_length(board, opp, r, c)
        if max_len_opp >= 5:
            block_win_moves.append(a)
        elif max_len_opp == 4:
            block_four_moves.append(a)
        elif _creates_open_three(board, opp, r, c):
            block_open_three_moves.append(a)
        board[r, c] = 0

    return (
        winning_moves,
        block_win_moves,
        block_open_four_moves,
        block_four_moves,
        block_open_three_moves,
        create_four_moves,
        create_open_three_moves,
    )


# 규칙 1~7에 해당하는 행동을 했을 때 추가로 줄 보상 (쉐이핑). tier 0 = 해당 없음.
# 1=승리수, 2=상대 승리 차단, 3=OOㅁO 차단, 4=상대 4줄 차단, 5=상대 열린3 차단, 6=내 4줄, 7=내 열린3
SHAPING_REWARD_BY_TIER = {
    1: 0.12,   # 내 즉시 승리 수 (환경에서 이미 +1 주므로 소량)
    2: 0.28,   # 상대 즉시 승리 수 차단
    3: 0.22,   # 상대 OOㅁO 차단
    4: 0.18,   # 상대 4줄 차단
    5: 0.14,   # 상대 열린 3 차단
    6: 0.16,   # 내 4줄 만들기
    7: 0.08,   # 내 열린 3 만들기
}


def get_rule_tier(board: np.ndarray, current_player: int, action: int, mask: np.ndarray) -> int:
    """
    주어진 행동이 규칙 1~7 중 어디에 해당하는지 반환.
    반환: 1~7 (해당 규칙), 0 (해당 없음).
    """
    lists = _get_rule_move_lists(board.copy(), current_player, mask)
    tier_order = (
        (1, lists[0]),  # winning_moves
        (2, lists[1]),  # block_win_moves
        (3, lists[2]),  # block_open_four_moves
        (4, lists[3]),  # block_four_moves
        (5, lists[4]),  # block_open_three_moves
        (6, lists[5]),  # create_four_moves
        (7, lists[6]),  # create_open_three_moves
    )
    for tier, move_list in tier_order:
        if action in move_list:
            return tier
    return 0


def select_action_with_rules(board: np.ndarray, current_player: int, mask: np.ndarray) -> int | None:
    """
    오목 규칙 기반 우선순위:
    1) 내 즉시 승리 수 (5목 완성)
    2) 상대 즉시 승리 수 차단
    3) 상대 OOㅁO(가운데 빈 4목) 차단 — ㅁ에 두면 다음 수 5목
    4) 상대 4줄(다음 수에 5목) 차단
    5) 상대 열린 3 차단
    6) 내 4줄 만들기
    7) 내 열린 3 만들기
    반환값: 규칙으로 결정한 action 인덱스, 없으면 None.
    """
    board_size = board.shape[0]
    valid_actions = np.where(mask > 0.5)[0]
    if len(valid_actions) == 0:
        return None

    lists = _get_rule_move_lists(board.copy(), current_player, mask)
    (
        winning_moves,
        block_win_moves,
        block_open_four_moves,
        block_four_moves,
        block_open_three_moves,
        create_four_moves,
        create_open_three_moves,
    ) = lists

    rng = np.random.default_rng()
    for moves in (
        winning_moves,
        block_win_moves,
        block_open_four_moves,
        block_four_moves,
        block_open_three_moves,
        create_four_moves,
        create_open_three_moves,
    ):
        if moves:
            return int(rng.choice(moves))

    return None

class OmokEnv(gym.Env):
    """Gymnasium 환경: Multi-channel 관측, action_mask, 렌주룰 없음."""
    metadata = {"render_modes": ["ansi", "human"], "render_fps": 4}

    def __init__(self, board_size=15, render_mode=None):
        super().__init__()
        self.board_size = board_size
        self.render_mode = render_mode
        self.action_space = spaces.Discrete(board_size ** 2)
        # 관측: (3, 15, 15) = 내 돌, 상대 돌, 턴 채널
        self.observation_space = spaces.Box(
            low=0.0, high=1.0,
            shape=(3, board_size, board_size),
            dtype=np.float32
        )
        self.board = None
        self.current_player = 1
        self.move_order = None  # (15,15) 각 칸에 몇 번째 수였는지 (1-based), 0이면 빈칸
        self.move_count = 0     # 전체 착수 순서 카운터
        # GUI
        self.window = None
        self.cell_size, self.margin = 40, 30

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.board = np.zeros((self.board_size, self.board_size), dtype=np.int8)
        self.current_player = 1
        self.move_order = np.zeros((self.board_size, self.board_size), dtype=np.int32)
        self.move_count = 0
        obs = make_obs_from_board(self.board, self.current_player)
        mask = get_action_mask(self.board)
        return obs, {"action_mask": mask, "current_player": self.current_player}

    def step(self, action):
        r, c = action // self.board_size, action % self.board_size
        if self.board[r, c] != 0:
            obs = make_obs_from_board(self.board, self.current_player)
            mask = get_action_mask(self.board)
            return obs, -10.0, True, False, {
                "action_mask": mask, "reason": "invalid_move", "winner": 3 - self.current_player
            }

        # 전체 게임 기준 n번째 수 (1, 2, 3, ...)
        self.move_count += 1
        move_number = self.move_count
        self.board[r, c] = self.current_player
        self.move_order[r, c] = move_number

        if self._check_win(r, c, self.current_player):
            obs = make_obs_from_board(self.board, self.current_player)
            mask = get_action_mask(self.board)
            return obs, 1.0, True, False, {
                "action_mask": mask, "reason": "win", "winner": self.current_player
            }
        if not np.any(self.board == 0):
            obs = make_obs_from_board(self.board, self.current_player)
            mask = get_action_mask(self.board)
            return obs, 0.0, True, False, {"action_mask": mask, "reason": "draw", "winner": 0}

        self.current_player = 3 - self.current_player
        obs = make_obs_from_board(self.board, self.current_player)
        mask = get_action_mask(self.board)
        return obs, 0.0, False, False, {"action_mask": mask, "current_player": self.current_player}

    def _check_win(self, row, col, player):
        directions = [(0, 1), (1, 0), (1, 1), (-1, 1)]
        for dr, dc in directions:
            count = 1
            for step in (1, -1):
                r, c = row + dr * step, col + dc * step
                while 0 <= r < self.board_size and 0 <= c < self.board_size and self.board[r, c] == player:
                    count += 1
                    r += dr * step
                    c += dc * step
            if count >= 5:
                return True
        return False

    def get_move_order(self):
        return self.move_order.copy() if self.move_order is not None else None

    def render(self):
        if self.render_mode != "human":
            return
        if self.window is None:
            self.window = tk.Tk()
            self.window.title("AI 오목 대전")
            size = (self.board_size - 1) * self.cell_size + self.margin * 2
            self.canvas = tk.Canvas(self.window, width=size, height=size, bg="#DCB35C")
            self.canvas.pack()
            self._font_small = tkfont.Font(family="Consolas", size=9, weight="bold")
            self._font_medium = tkfont.Font(family="Consolas", size=10, weight="bold")

        self.canvas.delete("all")
        for i in range(self.board_size):
            start = self.margin + i * self.cell_size
            end = self.margin + (self.board_size - 1) * self.cell_size
            self.canvas.create_line(self.margin, start, end, start)
            self.canvas.create_line(start, self.margin, start, end)

        move_order = self.get_move_order()
        for r in range(self.board_size):
            for c in range(self.board_size):
                if self.board[r, c] != 0:
                    x = self.margin + c * self.cell_size
                    y = self.margin + r * self.cell_size
                    rad = self.cell_size // 2 - 2
                    color = "black" if self.board[r, c] == 1 else "white"
                    self.canvas.create_oval(x - rad, y - rad, x + rad, y + rad, fill=color, outline="black")
                    num = move_order[r, c] if move_order is not None else 0
                    if num > 0:
                        text_color = "white" if self.board[r, c] == 1 else "black"
                        self.canvas.create_text(x, y, text=str(num), fill=text_color, font=self._font_small)

        self.window.update_idletasks()
        self.window.update()

    def close(self):
        if self.window:
            self.window.destroy()
            self.window = None


# =============================================================================
# 2. Data Augmentation (8방향: 회전 4 + 반전 2 조합)
# =============================================================================

def augment_obs_action(obs, action, board_size=15):
    """(obs, action) 쌍을 8가지 대칭으로 부풀리기. obs (3,15,15), action 스칼라."""
    out = []
    # 0: 원본
    out.append((obs.copy(), action))
    # 1: 90도
    o1 = np.rot90(obs, k=1, axes=(1, 2))
    r, c = action // board_size, action % board_size
    r1, c1 = c, board_size - 1 - r
    out.append((o1, r1 * board_size + c1))
    # 2: 180도
    o2 = np.rot90(obs, k=2, axes=(1, 2))
    r2, c2 = board_size - 1 - r, board_size - 1 - c
    out.append((o2, r2 * board_size + c2))
    # 3: 270도
    o3 = np.rot90(obs, k=3, axes=(1, 2))
    r3, c3 = board_size - 1 - c, r
    out.append((o3, r3 * board_size + c3))
    # 4: 좌우 반전
    o4 = np.flip(obs, axis=2).copy()
    r4, c4 = r, board_size - 1 - c
    out.append((o4, r4 * board_size + c4))
    # 5: 상하 반전
    o5 = np.flip(obs, axis=1).copy()
    r5, c5 = board_size - 1 - r, c
    out.append((o5, r5 * board_size + c5))
    # 6: 좌우 반전 후 90도 (대각 반전과 동일)
    o6 = np.rot90(np.flip(obs, axis=2), k=1, axes=(1, 2))
    # (r,c) -> flip_c -> (r, bs-1-c) -> rot90 -> (bs-1-c, r)
    out.append((o6, (board_size - 1 - c) * board_size + r))
    # 7: 상하 반전 후 90도
    o7 = np.rot90(np.flip(obs, axis=1), k=1, axes=(1, 2))
    out.append((o7, c * board_size + (board_size - 1 - r)))
    return out


def augment_mask(mask, action, board_size=15):
    """mask는 (225,) 불리언/플로트. augmentation 시 각 변환에 맞게 mask 재배치."""
    masks = []
    m = mask.reshape(board_size, board_size)
    # 0
    masks.append(mask.copy())
    # 1: 90
    masks.append(np.rot90(m, k=1).flatten())
    # 2: 180
    masks.append(np.rot90(m, k=2).flatten())
    # 3: 270
    masks.append(np.rot90(m, k=3).flatten())
    # 4: flip col
    masks.append(np.flip(m, axis=1).flatten())
    # 5: flip row
    masks.append(np.flip(m, axis=0).flatten())
    # 6
    masks.append(np.rot90(np.flip(m, axis=1), k=1).flatten())
    # 7
    masks.append(np.rot90(np.flip(m, axis=0), k=1).flatten())
    return masks


# =============================================================================
# 3. CNN Policy / Value 네트워크 (3x3 커널, Multi-channel 입력)
# =============================================================================

class GomokuCNN(nn.Module):
    """공유 CNN 백본 후 policy head / value head 분리."""

    def __init__(self, board_size=15, channels=3, hidden=128, n_actions=225):
        super().__init__()
        self.board_size = board_size
        self.n_actions = n_actions
        self.conv = nn.Sequential(
            nn.Conv2d(channels, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
        )
        flat_size = 64 * board_size * board_size
        self.policy_fc = nn.Sequential(
            nn.Linear(flat_size, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )
        self.value_fc = nn.Sequential(
            nn.Linear(flat_size, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        b = x.size(0)
        h = self.conv(x)
        h_flat = h.view(b, -1)
        logits = self.policy_fc(h_flat)
        value = self.value_fc(h_flat).squeeze(-1)
        return logits, value

    def get_action(self, obs, mask, deterministic=False):
        """obs: (1,3,15,15), mask: (225,) 1=가능 0=불가. action 스칼라."""
        with torch.no_grad():
            logits, value = self.forward(obs)
            logits = logits.cpu().numpy().reshape(-1)
            mask_np = mask.cpu().numpy() if torch.is_tensor(mask) else mask
            logits = np.where(mask_np > 0.5, logits, -1e9)
            if deterministic:
                action = int(np.argmax(logits))
            else:
                probs = np.exp(logits - logits.max())
                probs = probs * mask_np
                if probs.sum() <= 0:
                    action = int(np.where(mask_np > 0)[0][0])
                else:
                    probs = probs / probs.sum()
                    action = int(np.random.choice(self.n_actions, p=probs))
            return action, value.cpu().numpy().item()


# =============================================================================
# 4. PPO (Action Masking + Augmentation 적용)
# =============================================================================

class PPOBuffer:
    """Trajectory 버퍼. 수집 후 8방향 augmentation 적용하여 저장."""

    def __init__(self, board_size=15, gamma=0.99, gae_lambda=0.95, use_augmentation=True):
        self.board_size = board_size
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.use_augmentation = use_augmentation
        self.obs_list = []
        self.act_list = []
        self.mask_list = []
        self.ret_list = []
        self.adv_list = []
        self.logp_old_list = []

    def add(self, obs, action, mask, reward, value, logp_old):
        self.obs_list.append(obs)
        self.act_list.append(action)
        self.mask_list.append(mask)
        self.ret_list.append(reward)
        self.adv_list.append(0.0)
        self.logp_old_list.append(logp_old)

    def set_last_reward(self, reward):
        """에피소드가 상대 승리로 끝났을 때, 우리 마지막 수의 보상을 -1로 설정."""
        if self.ret_list:
            self.ret_list[-1] = reward

    def finish_trajectory(self, last_value=0.0, last_done=True):
        """한 에피소드 끝: GAE로 advantage 계산 후 augmentation 적용해 버퍼에 반영."""
        if not self.ret_list:
            return
        rews = np.array(self.ret_list, dtype=np.float32)
        R = last_value if not last_done else 0.0
        returns = []
        for r in rews[::-1]:
            R = r + self.gamma * R
            returns.append(R)
        returns = np.array(returns[::-1], dtype=np.float32)
        adv = returns - np.mean(returns)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        for i in range(len(self.obs_list)):
            self.adv_list[i] = adv[i]
            self.ret_list[i] = returns[i]

        if self.use_augmentation:
            aug_obs, aug_act, aug_mask, aug_ret, aug_adv, aug_logp = [], [], [], [], [], []
            for i in range(len(self.obs_list)):
                pairs = augment_obs_action(self.obs_list[i], self.act_list[i], self.board_size)
                masks_aug = augment_mask(self.mask_list[i], self.act_list[i], self.board_size)
                for k, (o, a) in enumerate(pairs):
                    aug_obs.append(o)
                    aug_act.append(a)
                    aug_mask.append(masks_aug[k].astype(np.float32))
                    aug_ret.append(self.ret_list[i])
                    aug_adv.append(self.adv_list[i])
                    aug_logp.append(self.logp_old_list[i])
            self.obs_list = aug_obs
            self.act_list = aug_act
            self.mask_list = aug_mask
            self.ret_list = aug_ret
            self.adv_list = aug_adv
            self.logp_old_list = aug_logp

    def get_batches(self, batch_size):
        """배치로 넘겨줌."""
        n = len(self.obs_list)
        idx = np.random.permutation(n)
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            ind = idx[start:end]
            obs_b = np.stack([self.obs_list[i] for i in ind], axis=0)
            act_b = np.array([self.act_list[i] for i in ind], dtype=np.int64)
            mask_b = np.stack([self.mask_list[i] for i in ind], axis=0)
            ret_b = np.array([self.ret_list[i] for i in ind], dtype=np.float32)
            adv_b = np.array([self.adv_list[i] for i in ind], dtype=np.float32)
            logp_old_b = np.array([self.logp_old_list[i] for i in ind], dtype=np.float32)
            yield obs_b, act_b, mask_b, ret_b, adv_b, logp_old_b

    def clear(self):
        self.obs_list = []
        self.act_list = []
        self.mask_list = []
        self.ret_list = []
        self.adv_list = []
        self.logp_old_list = []

    def __len__(self):
        return len(self.obs_list)


def ppo_update(model, optimizer, batch, clip_eps=0.2, value_coef=0.5, ent_coef=0.01, device="cpu"):
    """PPO 업데이트. logits에 nan/inf가 있으면 업데이트하지 않고 None 반환."""
    obs, act, mask, ret, adv, logp_old = batch
    obs_t = torch.tensor(obs, dtype=torch.float32, device=device)
    act_t = torch.tensor(act, dtype=torch.long, device=device)
    mask_t = torch.tensor(mask, dtype=torch.float32, device=device)
    ret_t = torch.tensor(ret, dtype=torch.float32, device=device)
    adv_t = torch.tensor(adv, dtype=torch.float32, device=device)
    logp_old_t = torch.tensor(logp_old, dtype=torch.float32, device=device)

    logits, value = model(obs_t)
    if torch.isnan(logits).any() or torch.isinf(logits).any():
        return None
    logits_masked = logits.masked_fill(mask_t < 0.5, -1e9)
    dist = Categorical(logits=logits_masked)
    logp = dist.log_prob(act_t)
    entropy = dist.entropy().mean()

    ratio = torch.exp(logp - logp_old_t)
    surr1 = ratio * adv_t
    surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * adv_t
    policy_loss = -torch.min(surr1, surr2).mean()
    value_loss = F.mse_loss(value, ret_t)
    loss = policy_loss + value_coef * value_loss - ent_coef * entropy
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
    optimizer.step()
    return loss.item(), policy_loss.item(), value_loss.item()


# =============================================================================
# 5. Self-play + Opponent Pool
# =============================================================================

def save_model(model, path):
    torch.save(model.state_dict(), path)

def load_model(model, path, device="cpu"):
    state = torch.load(path, map_location=device)
    model.load_state_dict(state)
    return model

class OpponentPool:
    """과거 저장된 모델 경로 리스트. 랜덤 샘플하여 상대로 사용."""

    def __init__(self, model_dir, model_class, device, max_pool_size=10):
        self.model_dir = model_dir
        self.model_class = model_class
        self.device = device
        self.max_pool_size = max_pool_size
        self.paths = []
        self._refresh()

    def _refresh(self):
        os.makedirs(self.model_dir, exist_ok=True)
        files = sorted(glob.glob(os.path.join(self.model_dir, "*.pth")), key=os.path.getmtime, reverse=True)
        self.paths = files[: self.max_pool_size]

    def add(self, path):
        if path not in self.paths:
            self.paths.insert(0, path)
            self.paths = self.paths[: self.max_pool_size]

    def sample_opponent(self, current_path=None):
        """현재 모델 경로 제외하고 과거 모델 중 하나 반환. 없으면 None (최신끼리)."""
        self._refresh()
        candidates = [p for p in self.paths if p != current_path]
        if not candidates:
            return None
        return random.choice(candidates)


# =============================================================================
# 6. Human Agent (GUI 클릭)
# =============================================================================

class HumanAgent:
    def __init__(self, env, name="Human"):
        self.env = env
        self.name = name
        self.clicked_action = None
        self.current_state = None
        self.current_mask = None

    def select_action(self, state, mask=None):
        self.clicked_action = None
        self.current_state = state
        self.current_mask = mask
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
            flat = self.current_state.flatten() if self.current_state.ndim == 3 else self.current_state
            if len(flat) == 225 and flat[action] == 0:
                self.clicked_action = action


# =============================================================================
# 7. 학습 모드 (A) / 플레이 모드 (B) 진입점
# =============================================================================

def _resolve_device(device_str: str) -> str:
    """'auto' / 'cpu' / 'cuda' 문자열을 실제 디바이스 문자열로 변환."""
    device_str = (device_str or "auto").lower()
    if device_str == "cpu":
        return "cpu"
    if device_str == "cuda":
        return "cuda" if torch.cuda.is_available() else "cpu"
    # auto
    return "cuda" if torch.cuda.is_available() else "cpu"


def run_training(
    total_timesteps=200_000,
    save_interval=10_000,
    opponent_pool_size=5,
    lr=3e-4,
    n_steps=512,
    batch_size=64,
    n_epochs=4,
    model_dir="gomoku_models",
    seed=42,
    render=False,
    device_str: str = "auto",
    resume_from: str | None = None,
):
    device = _resolve_device(device_str)
    print(f"[Train] Using device: {device}")
    np.random.seed(seed)
    torch.manual_seed(seed)
    env = OmokEnv(render_mode="human" if render else None)
    board_size = env.board_size
    n_actions = board_size ** 2

    model = GomokuCNN(board_size=board_size, n_actions=n_actions).to(device)
    if resume_from and os.path.isfile(resume_from):
        load_model(model, resume_from, device)
        print(f"[Train] Resumed from checkpoint: {resume_from}")
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    pool = OpponentPool(model_dir, GomokuCNN, device, max_pool_size=opponent_pool_size)
    buffer = PPOBuffer(board_size=board_size, use_augmentation=True)

    os.makedirs(model_dir, exist_ok=True)
    global_step = 0
    n_episodes = 0
    current_save_path = None

    while global_step < total_timesteps:
        # 상대 선택: 일정 확률로 풀에서 과거 모델 선택
        opp_path = pool.sample_opponent(current_save_path) if random.random() < 0.5 else None
        if opp_path and os.path.isfile(opp_path):
            opponent = GomokuCNN(board_size=board_size, n_actions=n_actions).to(device)
            load_model(opponent, opp_path, device)
            opponent.eval()
        else:
            opponent = model

        obs, info = env.reset()
        mask = info["action_mask"]
        done = False
        ep_reward = 0.0

        while not done and global_step < total_timesteps:
            # 규칙 기반 우선 선택 (수비 / 공격 우선순위)
            board_view = env.board.copy()
            rule_action = select_action_with_rules(board_view, env.current_player, mask)

            obs_t = torch.tensor(obs[np.newaxis, ...], dtype=torch.float32, device=device)
            mask_t = torch.tensor(mask, dtype=torch.float32, device=device)
            with torch.no_grad():
                logits, value = model(obs_t)
                logits_np = logits.cpu().numpy().reshape(-1)
                logits_np = np.where(mask > 0.5, logits_np, -1e9)
                probs = np.exp(logits_np - logits_np.max())
                probs = probs * mask
                if probs.sum() <= 0:
                    ppo_action = int(np.where(mask > 0)[0][0])
                else:
                    probs = probs / probs.sum()
                    ppo_action = int(np.random.choice(n_actions, p=probs))

                # 규칙이 제안한 수가 있으면 그 수를 사용, 없으면 PPO 선택 사용
                action = rule_action if rule_action is not None else ppo_action

                logits_masked = logits.masked_fill(mask_t.unsqueeze(0) < 0.5, -1e9)
                dist = Categorical(logits=logits_masked)
                logp_old = dist.log_prob(torch.tensor([action], device=device)).item()

            board_before = env.board.copy()
            player_before = env.current_player
            next_obs, reward, terminated, truncated, info = env.step(action)
            tier = get_rule_tier(board_before, player_before, action, mask)
            shaping = SHAPING_REWARD_BY_TIER.get(tier, 0.0)
            total_reward = reward + shaping
            done = terminated or truncated
            ep_reward += total_reward
            buffer.add(obs, action, mask, total_reward, value.cpu().item(), logp_old)
            obs, mask = next_obs, info["action_mask"]
            global_step += 1

            # 진행률 출력 (대략 1% 단위)
            if total_timesteps > 0:
                interval = max(1, total_timesteps // 100)
                if global_step % interval == 0 or global_step == total_timesteps:
                    percent = global_step / total_timesteps * 100.0
                    print(f"[Train] progress: {global_step}/{total_timesteps} ({percent:.1f}%)")

            if done:
                buffer.finish_trajectory(last_value=0.0, last_done=True)
                n_episodes += 1
                # PPO update
                if len(buffer) >= batch_size:
                    nan_detected = False
                    for _ in range(n_epochs):
                        for batch in buffer.get_batches(batch_size):
                            result = ppo_update(model, optimizer, batch, device=device)
                            if result is None:
                                nan_detected = True
                                break
                        if nan_detected:
                            break
                    if nan_detected:
                        print("[Train] NaN detected in logits. Stopping. Resume from last saved checkpoint (e.g. --resume gomoku_models\\gomoku_ppo_XXXXX.pth)")
                        buffer.clear()
                        env.close()
                        return
                    buffer.clear()
                break

            # 상대 턴 (self-play)
            if not done:
                inv_obs = np.stack(
                    [next_obs[1], next_obs[0], next_obs[2]],
                    axis=0,
                )
                inv_mask = get_action_mask(env.board)

                # 상대도 동일한 규칙 기반 우선순위를 따르도록 함
                opp_board_view = env.board.copy()
                opp_rule_action = select_action_with_rules(opp_board_view, env.current_player, inv_mask)

                opp_obs_t = torch.tensor(inv_obs[np.newaxis, ...], dtype=torch.float32, device=device)
                with torch.no_grad():
                    logits_opp, _ = opponent(opp_obs_t)
                    logits_opp_np = logits_opp.cpu().numpy().reshape(-1)
                    logits_opp_np = np.where(inv_mask > 0.5, logits_opp_np, -1e9)
                    probs_opp = np.exp(logits_opp_np - logits_opp_np.max())
                    probs_opp = probs_opp * inv_mask
                    if probs_opp.sum() <= 0:
                        ppo_opp_action = int(np.where(inv_mask > 0)[0][0])
                    else:
                        probs_opp = probs_opp / probs_opp.sum()
                        ppo_opp_action = int(np.random.choice(n_actions, p=probs_opp))

                opp_action = opp_rule_action if opp_rule_action is not None else ppo_opp_action

                next_obs2, reward2, terminated, truncated, info = env.step(opp_action)
                done = terminated or truncated
                if done:
                    # 우리(흑)가 진 경우: 우리 버퍼의 마지막 보상을 -1로
                    if info.get("winner") == 2:
                        buffer.set_last_reward(-1.0)
                    buffer.finish_trajectory(last_value=0.0, last_done=True)
                    n_episodes += 1
                    if len(buffer) >= batch_size:
                        nan_detected = False
                        for _ in range(n_epochs):
                            for batch in buffer.get_batches(batch_size):
                                result = ppo_update(model, optimizer, batch, device=device)
                                if result is None:
                                    nan_detected = True
                                    break
                            if nan_detected:
                                break
                        if nan_detected:
                            print("[Train] NaN detected in logits. Stopping. Resume from last saved checkpoint (e.g. --resume gomoku_models\\gomoku_ppo_XXXXX.pth)")
                            buffer.clear()
                            env.close()
                            return
                        buffer.clear()
                    break
                obs, mask = next_obs2, info["action_mask"]

        if render:
            env.render()

        if save_interval > 0 and global_step > 0 and global_step % save_interval == 0:
            path = os.path.join(model_dir, f"gomoku_ppo_{global_step}.pth")
            save_model(model, path)
            pool.add(path)
            current_save_path = path
            print(f"[Train] step={global_step} episodes={n_episodes} saved {path}")

    path_final = os.path.join(model_dir, "gomoku_ppo_final.pth")
    save_model(model, path_final)
    print(f"Training done. Final model: {path_final}")
    env.close()


def run_play(model_path, model_dir="gomoku_models", human_plays_black=True, device_str: str = "auto"):
    device = _resolve_device(device_str)
    print(f"[Play] Using device: {device}")
    env = OmokEnv(render_mode="human")
    board_size = env.board_size
    model = GomokuCNN(board_size=board_size, n_actions=board_size ** 2).to(device)
    if not os.path.isfile(model_path):
        model_path = os.path.join(model_dir, "gomoku_ppo_final.pth")
    load_model(model, model_path, device)
    model.eval()
    human = HumanAgent(env, name="Human")

    state, info = env.reset()
    env.render()
    terminated = False
    current_player = 1

    while not terminated:
        mask = info["action_mask"]
        if (current_player == 1 and human_plays_black) or (current_player == 2 and not human_plays_black):
            # state (3,15,15): 채널0=내 돌, 채널1=상대 돌. Human 턴이면 보드 보여주기
            board_flat = np.zeros(board_size * board_size, dtype=np.int8)
            board_flat[(state[0] > 0.5).flatten()] = current_player
            board_flat[(state[1] > 0.5).flatten()] = 3 - current_player
            action = human.select_action(board_flat, mask)
        else:
            # 규칙 기반 우선 선택 (실전 플레이에서도 동일 전략 사용)
            # state는 현재 플레이어 관점이므로, 보드 복원 후 규칙 적용
            board = np.zeros((board_size, board_size), dtype=np.int8)
            board[state[0] > 0.5] = current_player
            board[state[1] > 0.5] = 3 - current_player
            rule_action = select_action_with_rules(board, current_player, mask)

            if rule_action is not None:
                action = rule_action
            else:
                obs_t = torch.tensor(state[np.newaxis, ...], dtype=torch.float32, device=device)
                mask_t = torch.tensor(mask, dtype=torch.float32, device=device)
                action, _ = model.get_action(obs_t, mask_t, deterministic=True)
        state, reward, terminated, truncated, info = env.step(action)
        if not terminated:
            current_player = info["current_player"]
        env.render()
        time.sleep(0.1)

    winner = info.get("winner", 0)
    print("=== 대결 종료 ===")
    if winner == 0:
        print("무승부")
    elif (winner == 1 and human_plays_black) or (winner == 2 and not human_plays_black):
        print("Human 승리!")
    else:
        print("AI 승리!")
    time.sleep(2)
    env.close()


# =============================================================================
# 8. 메인 진입점
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Gomoku PPO Self-play")
    parser.add_argument("--mode", type=str, default="play", choices=["train", "play"], help="train: 학습 모드(A), play: 플레이 모드(B)")
    parser.add_argument("--model_path", type=str, default=None, help="play 모드에서 불러올 모델 경로")
    parser.add_argument("--model_dir", type=str, default="gomoku_models", help="모델 저장/로드 디렉터리")
    parser.add_argument("--total_timesteps", type=int, default=200_000, help="학습 총 스텝")
    parser.add_argument("--save_interval", type=int, default=10_000, help="가중치 저장 주기")
    parser.add_argument("--human_black", action="store_true", default=True, help="플레이 모드에서 인간이 흑인지")
    parser.add_argument("--render_train", action="store_true", help="학습 중 GUI 렌더링")
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="학습/플레이에 사용할 디바이스 (auto/cpu/cuda)",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="학습 시 저장된 가중치(.pth) 경로를 지정하면 해당 모델에서 이어서 학습",
    )
    args = parser.parse_args()

    if args.mode == "train":
        run_training(
            total_timesteps=args.total_timesteps,
            save_interval=args.save_interval,
            model_dir=args.model_dir,
            render=args.render_train,
            device_str=args.device,
            resume_from=args.resume,
        )
    else:
        run_play(
            model_path=args.model_path,
            model_dir=args.model_dir,
            human_plays_black=args.human_black,
            device_str=args.device,
        )


if __name__ == "__main__":
    main()
