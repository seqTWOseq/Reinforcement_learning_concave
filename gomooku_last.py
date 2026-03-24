import os
import glob
import time
import random
from typing import List, Tuple, Optional

import numpy as np
import gymnasium as gym
from gymnasium import spaces

import tkinter as tk
from tkinter import font as tkfont

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


# ============================================================
# Config
# ============================================================
BOARD_SIZE = 15
N_ACTIONS = BOARD_SIZE * BOARD_SIZE
OBS_SHAPE = (3, BOARD_SIZE, BOARD_SIZE)


# ============================================================
# 1) Observation / Action Mask
# ============================================================
def make_obs_from_board(board: np.ndarray, current_player: int) -> np.ndarray:
    """
    Create CNN input with 3 channels:
      channel 0: current player's stones (1.0)
      channel 1: opponent's stones (1.0)
      channel 2: turn indicator (current_player == 1 ? 1.0 : 0.0)
    Shape: (3, 15, 15), dtype float32.
    """
    size = board.shape[0]
    my_stones = (board == current_player).astype(np.float32)
    opp_stones = (board == 3 - current_player).astype(np.float32)
    turn_channel = np.full(
        (size, size),
        float(current_player == 1),
        dtype=np.float32,
    )
    return np.stack([my_stones, opp_stones, turn_channel], axis=0)


def get_action_mask(board: np.ndarray) -> np.ndarray:
    """
    Mask for invalid actions: 1 for empty cells, 0 for occupied.
    Shape: (225,), dtype float32.
    """
    return (board.flatten() == 0).astype(np.float32)


def terminal_reward_from_winner(winner: int, our_player_id: int) -> float:
    if winner == 0:
        return 0.0
    return 1.0 if winner == our_player_id else -1.0


# ============================================================
# 2) Gymnasium + Tkinter Omok Environment
# ============================================================
class OmokEnv(gym.Env):
    """
    Gomoku (15x15) with:
      - Multi-channel obs (3, 15, 15)
      - Action masking via info["action_mask"]
      - GUI render: show move order numbers on each stone
    """

    metadata = {"render_modes": ["ansi", "human"], "render_fps": 4}

    def __init__(self, render_mode: Optional[str] = None):
        super().__init__()
        self.board_size = BOARD_SIZE
        self.render_mode = render_mode

        self.action_space = spaces.Discrete(self.board_size**2)
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=OBS_SHAPE,
            dtype=np.float32,
        )

        self.board: Optional[np.ndarray] = None
        self.current_player: int = 1  # 1: Black, 2: White

        # Move order matrix (1-based). 0 means empty.
        self.move_order: Optional[np.ndarray] = None
        self.move_count: int = 0

        # GUI
        self.window = None
        self.canvas = None
        self._font_small = None
        self._font_medium = None
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

    def step(self, action: int):
        r, c = divmod(int(action), self.board_size)
        if self.board is None:
            raise RuntimeError("Environment not reset.")

        # Invalid move
        if self.board[r, c] != 0:
            obs = make_obs_from_board(self.board, self.current_player)
            mask = get_action_mask(self.board)
            winner = 3 - self.current_player
            return (
                obs,
                -10.0,
                True,
                False,
                {"action_mask": mask, "reason": "invalid_move", "winner": winner},
            )

        # Apply move
        self.move_count += 1
        move_number = self.move_count
        self.board[r, c] = self.current_player
        self.move_order[r, c] = move_number

        # Win check
        if self._check_win(r, c, self.current_player):
            obs = make_obs_from_board(self.board, self.current_player)
            mask = get_action_mask(self.board)
            return (
                obs,
                1.0,
                True,
                False,
                {"action_mask": mask, "reason": "win", "winner": self.current_player},
            )

        # Draw check
        if not np.any(self.board == 0):
            obs = make_obs_from_board(self.board, self.current_player)
            mask = get_action_mask(self.board)
            return (
                obs,
                0.0,
                True,
                False,
                {"action_mask": mask, "reason": "draw", "winner": 0},
            )

        # Switch player
        self.current_player = 3 - self.current_player
        obs = make_obs_from_board(self.board, self.current_player)
        mask = get_action_mask(self.board)
        return (
            obs,
            0.0,
            False,
            False,
            {"action_mask": mask, "current_player": self.current_player},
        )

    def _check_win(self, row: int, col: int, player: int) -> bool:
        directions = [(0, 1), (1, 0), (1, 1), (-1, 1)]
        for dr, dc in directions:
            count = 1
            for step in (1, -1):
                r, c = row + dr * step, col + dc * step
                while (
                    0 <= r < self.board_size
                    and 0 <= c < self.board_size
                    and self.board[r, c] == player
                ):
                    count += 1
                    r += dr * step
                    c += dc * step
            if count >= 5:
                return True
        return False

    def render(self):
        if self.render_mode != "human":
            return
        if self.window is None:
            self.window = tk.Tk()
            self.window.title("Gomoku (AI Self-play / Training)")
            size = (self.board_size - 1) * self.cell_size + self.margin * 2
            self.canvas = tk.Canvas(self.window, width=size, height=size, bg="#DCB35C")
            self.canvas.pack()
            self._font_small = tkfont.Font(family="Consolas", size=9, weight="bold")
            self._font_medium = tkfont.Font(family="Consolas", size=10, weight="bold")

        self.canvas.delete("all")

        # Grid
        for i in range(self.board_size):
            start = self.margin + i * self.cell_size
            end = self.margin + (self.board_size - 1) * self.cell_size
            self.canvas.create_line(self.margin, start, end, start)
            self.canvas.create_line(start, self.margin, start, end)

        # Stones and move order numbers
        move_order = self.move_order if self.move_order is not None else None
        for r in range(self.board_size):
            for c in range(self.board_size):
                if self.board[r, c] != 0:
                    x = self.margin + c * self.cell_size
                    y = self.margin + r * self.cell_size
                    rad = self.cell_size // 2 - 2
                    color = "black" if self.board[r, c] == 1 else "white"
                    self.canvas.create_oval(
                        x - rad, y - rad, x + rad, y + rad, fill=color, outline="black"
                    )

                    if move_order is not None:
                        num = int(move_order[r, c])
                        if num > 0:
                            # Requirement: black stone -> white number, white stone -> black number
                            text_color = "white" if self.board[r, c] == 1 else "black"
                            self.canvas.create_text(
                                x, y, text=str(num), fill=text_color, font=self._font_small
                            )

        self.window.update_idletasks()
        self.window.update()

    def close(self):
        if self.window is not None:
            self.window.destroy()
            self.window = None
            self.canvas = None


# ============================================================
# 3) Human Agent (click to place stone)
# ============================================================
class HumanAgent:
    def __init__(self, env: OmokEnv, name: str = "Human"):
        self.env = env
        self.name = name
        self.clicked_action: Optional[int] = None
        self._waiting: bool = False

    def select_action(self, mask: np.ndarray) -> int:
        """
        Blocks until user clicks an empty cell.
        `mask` is expected shape (225,), 1 for valid.
        """
        if self.env.window is None or self.env.canvas is None:
            # Make sure GUI is initialized.
            self.env.render()

        self.clicked_action = None
        self._waiting = True

        self.env.canvas.bind("<Button-1>", lambda event: self._click_handler(event, mask))
        while self.clicked_action is None and self._waiting:
            self.env.window.update()
            time.sleep(0.05)
        self.env.canvas.unbind("<Button-1>")
        return int(self.clicked_action)

    def _click_handler(self, event, mask: np.ndarray):
        bs = self.env.board_size
        c = round((event.x - self.env.margin) / self.env.cell_size)
        r = round((event.y - self.env.margin) / self.env.cell_size)

        if 0 <= r < bs and 0 <= c < bs:
            action = r * bs + c
            if mask[action] > 0.5 and self.env.board[r, c] == 0:
                self.clicked_action = action
                self._waiting = False


# ============================================================
# 4) CNN Policy/Value Network (3x3 kernels)
# ============================================================
class GomokuCNN(nn.Module):
    """
    Shared CNN backbone + policy head (Actor) + value head (Critic).
    Input: (B, 3, 15, 15)
    Output: logits (B, 225), value (B,)
    """

    def __init__(
        self,
        board_size: int = BOARD_SIZE,
        channels: int = 3,
        hidden: int = 128,
        n_actions: int = N_ACTIONS,
    ):
        super().__init__()
        self.board_size = board_size
        self.n_actions = n_actions

        self.conv = nn.Sequential(
            nn.Conv2d(channels, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
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

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.conv(x)
        h_flat = h.view(h.size(0), -1)
        logits = self.policy_fc(h_flat)
        value = self.value_fc(h_flat).squeeze(-1)
        return logits, value


# ============================================================
# 5) PPO Buffer + Data Augmentation (8x symmetry)
# ============================================================
def augment_obs_action(
    obs: np.ndarray, action: int, board_size: int = BOARD_SIZE
) -> List[Tuple[np.ndarray, int]]:
    """
    Augment (obs, action) by 8 board symmetries:
      0: original
      1: 90 deg
      2: 180 deg
      3: 270 deg
      4: flip left-right
      5: flip up-down
      6: flip left-right then 90 deg
      7: flip up-down then 90 deg
    obs shape: (3, bs, bs)
    """
    out: List[Tuple[np.ndarray, int]] = []
    r, c = divmod(int(action), board_size)

    # 0
    out.append((obs.copy(), action))

    # 1: 90 (numpy rot90 is counter-clockwise)
    o1 = np.rot90(obs, k=1, axes=(1, 2))
    r1, c1 = c, board_size - 1 - r
    out.append((o1, r1 * board_size + c1))

    # 2: 180
    o2 = np.rot90(obs, k=2, axes=(1, 2))
    r2, c2 = board_size - 1 - r, board_size - 1 - c
    out.append((o2, r2 * board_size + c2))

    # 3: 270
    o3 = np.rot90(obs, k=3, axes=(1, 2))
    r3, c3 = board_size - 1 - c, r
    out.append((o3, r3 * board_size + c3))

    # 4: flip left-right (axis=2)
    o4 = np.flip(obs, axis=2).copy()
    r4, c4 = r, board_size - 1 - c
    out.append((o4, r4 * board_size + c4))

    # 5: flip up-down (axis=1)
    o5 = np.flip(obs, axis=1).copy()
    r5, c5 = board_size - 1 - r, c
    out.append((o5, r5 * board_size + c5))

    # 6: flip left-right then 90
    o6 = np.rot90(np.flip(obs, axis=2), k=1, axes=(1, 2))
    out.append((o6, (board_size - 1 - c) * board_size + r))

    # 7: flip up-down then 90
    o7 = np.rot90(np.flip(obs, axis=1), k=1, axes=(1, 2))
    out.append((o7, c * board_size + (board_size - 1 - r)))

    return out


def augment_mask(mask: np.ndarray, action: int, board_size: int = BOARD_SIZE) -> List[np.ndarray]:
    """
    Augment action mask with same 8 symmetries.
    mask shape: (225,) where 1 = empty(valid), 0 = occupied(invalid).
    Returns list of 8 masks each shape (225,).
    """
    m = mask.reshape(board_size, board_size)
    masks: List[np.ndarray] = []

    # 0
    masks.append(mask.copy())
    # 1: 90
    masks.append(np.rot90(m, k=1).flatten())
    # 2: 180
    masks.append(np.rot90(m, k=2).flatten())
    # 3: 270
    masks.append(np.rot90(m, k=3).flatten())
    # 4: flip left-right (columns)
    masks.append(np.flip(m, axis=1).flatten())
    # 5: flip up-down (rows)
    masks.append(np.flip(m, axis=0).flatten())
    # 6: flip left-right then 90
    masks.append(np.rot90(np.flip(m, axis=1), k=1).flatten())
    # 7: flip up-down then 90
    masks.append(np.rot90(np.flip(m, axis=0), k=1).flatten())
    return masks


class PPOBuffer:
    """
    Trajectory buffer.
    After an episode ends, it computes returns/advantage, then applies
    8x symmetry augmentation for (obs, action, mask).
    """

    def __init__(
        self,
        board_size: int = BOARD_SIZE,
        gamma: float = 0.99,
        use_augmentation: bool = True,
    ):
        self.board_size = board_size
        self.gamma = gamma
        self.use_augmentation = use_augmentation

        self.obs_list: List[np.ndarray] = []
        self.act_list: List[int] = []
        self.mask_list: List[np.ndarray] = []
        self.rew_list: List[float] = []
        self.adv_list: List[float] = []
        self.logp_old_list: List[float] = []

    def add(
        self,
        obs: np.ndarray,
        action: int,
        mask: np.ndarray,
        reward: float,
        value: float,
        logp_old: float,
    ):
        # value is accepted for compatibility, but this implementation uses
        # Monte-Carlo style returns and normalizes advantages.
        self.obs_list.append(obs.astype(np.float32))
        self.act_list.append(int(action))
        self.mask_list.append(mask.astype(np.float32))
        self.rew_list.append(float(reward))
        self.adv_list.append(0.0)
        self.logp_old_list.append(float(logp_old))

    def set_last_reward(self, reward: float):
        if self.rew_list:
            self.rew_list[-1] = float(reward)

    def finish_trajectory(self, last_done: bool = True):
        if not self.rew_list:
            return

        rews = np.array(self.rew_list, dtype=np.float32)
        # Since we end episode fully, last_done=True sets bootstrap to 0.
        R = 0.0
        returns = []
        for r in rews[::-1]:
            R = r + self.gamma * R
            returns.append(R)
        returns = np.array(returns[::-1], dtype=np.float32)

        adv = returns - returns.mean()
        adv = adv / (adv.std() + 1e-8)

        self.adv_list = adv.tolist()
        self.rew_list = returns.tolist()

        if not self.use_augmentation:
            return

        aug_obs: List[np.ndarray] = []
        aug_act: List[int] = []
        aug_mask: List[np.ndarray] = []
        aug_ret: List[float] = []
        aug_adv: List[float] = []
        aug_logp: List[float] = []

        # Apply 8x symmetry to each time-step sample.
        for i in range(len(self.obs_list)):
            pairs = augment_obs_action(self.obs_list[i], self.act_list[i], self.board_size)
            masks_aug = augment_mask(self.mask_list[i], self.act_list[i], self.board_size)
            for k, (o, a) in enumerate(pairs):
                aug_obs.append(o.astype(np.float32))
                aug_act.append(int(a))
                aug_mask.append(masks_aug[k].astype(np.float32))
                aug_ret.append(float(self.rew_list[i]))
                aug_adv.append(float(self.adv_list[i]))
                aug_logp.append(float(self.logp_old_list[i]))

        self.obs_list = aug_obs
        self.act_list = aug_act
        self.mask_list = aug_mask
        self.rew_list = aug_ret
        self.adv_list = aug_adv
        self.logp_old_list = aug_logp

    def get_batches(self, batch_size: int):
        n = len(self.obs_list)
        if n == 0:
            return

        idx = np.random.permutation(n)
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            ind = idx[start:end]

            obs_b = np.stack([self.obs_list[i] for i in ind], axis=0)
            act_b = np.array([self.act_list[i] for i in ind], dtype=np.int64)
            mask_b = np.stack([self.mask_list[i] for i in ind], axis=0)
            ret_b = np.array([self.rew_list[i] for i in ind], dtype=np.float32)
            adv_b = np.array([self.adv_list[i] for i in ind], dtype=np.float32)
            logp_old_b = np.array([self.logp_old_list[i] for i in ind], dtype=np.float32)

            yield obs_b, act_b, mask_b, ret_b, adv_b, logp_old_b

    def clear(self):
        self.obs_list.clear()
        self.act_list.clear()
        self.mask_list.clear()
        self.rew_list.clear()
        self.adv_list.clear()
        self.logp_old_list.clear()

    def __len__(self):
        return len(self.obs_list)


def masked_categorical_from_logits(
    logits: torch.Tensor, mask: torch.Tensor
) -> Tuple[Categorical, torch.Tensor]:
    """
    logits: (B, A)
    mask: (B, A) float where 1=valid, 0=invalid
    """
    logits_masked = logits.masked_fill(mask < 0.5, -1e9)
    dist = Categorical(logits=logits_masked)
    return dist, logits_masked


def ppo_update(
    model: GomokuCNN,
    optimizer: torch.optim.Optimizer,
    batch: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    clip_eps: float = 0.2,
    value_coef: float = 0.5,
    ent_coef: float = 0.01,
    device: str = "cpu",
) -> Optional[float]:
    """
    Minimal PPO update (single model, action masking).
    Returns loss value or None when NaN/inf is detected.
    """
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

    dist, _ = masked_categorical_from_logits(logits, mask_t)
    logp = dist.log_prob(act_t)
    entropy = dist.entropy().mean()

    ratio = torch.exp(logp - logp_old_t)
    surr1 = ratio * adv_t
    surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * adv_t

    policy_loss = -torch.min(surr1, surr2).mean()
    value_loss = F.mse_loss(value, ret_t)

    loss = policy_loss + value_coef * value_loss - ent_coef * entropy
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
    optimizer.step()
    return float(loss.item())


def sample_action(
    model: GomokuCNN,
    obs: np.ndarray,
    mask: np.ndarray,
    device: str,
    deterministic: bool = False,
) -> Tuple[int, float, float]:
    """
    Returns: action, value, logp_old
    obs: (3, 15, 15)
    mask: (225,)
    """
    obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)  # (1,3,15,15)
    mask_t = torch.tensor(mask, dtype=torch.float32, device=device).unsqueeze(0)  # (1,225)

    with torch.no_grad():
        logits, value = model(obs_t)
        dist, logits_masked = masked_categorical_from_logits(logits, mask_t)

        if deterministic:
            action = int(torch.argmax(logits_masked, dim=1).item())
        else:
            action = int(dist.sample().item())

        logp_old = float(dist.log_prob(torch.tensor([action], device=device)).item())
        return action, float(value.item()), logp_old


# ============================================================
# 6) Opponent Pool
# ============================================================
class OpponentPool:
    def __init__(self, model_dir: str, max_pool_size: int = 10):
        self.model_dir = model_dir
        self.max_pool_size = max_pool_size
        self.paths: List[str] = []
        self.refresh()

    def refresh(self):
        os.makedirs(self.model_dir, exist_ok=True)
        files = glob.glob(os.path.join(self.model_dir, "*.pth"))
        # sort by mtime desc
        files = sorted(files, key=os.path.getmtime, reverse=True)
        self.paths = files[: self.max_pool_size]

    def add(self, path: str):
        self.refresh()
        if path not in self.paths:
            self.paths.insert(0, path)
            self.paths = self.paths[: self.max_pool_size]

    def sample_opponent(self, current_path: Optional[str] = None) -> Optional[str]:
        self.refresh()
        candidates = [p for p in self.paths if p != current_path]
        if not candidates:
            return None
        return random.choice(candidates)


# ============================================================
# 7) Mode 1: Data Collection (Human vs Human)
# ============================================================
def run_data_collection(
    dataset_dir: str = "gomoku_human_dataset",
    n_games: int = 30,
    seed: int = 42,
):
    random.seed(seed)
    np.random.seed(seed)

    os.makedirs(dataset_dir, exist_ok=True)
    existing = sorted(glob.glob(os.path.join(dataset_dir, "game_*.npz")))
    start_idx = len(existing)

    env = OmokEnv(render_mode="human")
    human = HumanAgent(env, name="Player")

    print(f"[Mode 1] Data collection into: {dataset_dir}")
    print(f"[Mode 1] Existing games: {len(existing)}. New games start at index: {start_idx}")

    for game_i in range(start_idx, start_idx + n_games):
        print(f"\n[Mode 1] Game {game_i + 1}/{start_idx + n_games}: playing...")
        obs, info = env.reset()
        env.render()

        traj_obs: List[np.ndarray] = []
        traj_actions: List[int] = []

        terminated = False
        truncated = False
        while not (terminated or truncated):
            current_player = int(info["current_player"])
            mask = info["action_mask"]

            if current_player == 1:
                print("[Mode 1] Black turn. Click an empty cell.")
            else:
                print("[Mode 1] White turn. Click an empty cell.")

            action = human.select_action(mask)

            # Record state-action BEFORE stepping
            traj_obs.append(obs.astype(np.float32))
            traj_actions.append(int(action))

            obs, reward, terminated, truncated, info = env.step(action)
            env.render()

        winner = int(info.get("winner", 0))
        out_path = os.path.join(dataset_dir, f"game_{game_i:05d}.npz")
        obs_arr = np.stack(traj_obs, axis=0).astype(np.float32)  # (T,3,15,15)
        act_arr = np.array(traj_actions, dtype=np.int64)  # (T,)

        np.savez_compressed(out_path, obs=obs_arr, actions=act_arr, winner=winner)
        print(f"[Mode 1] Saved: {out_path} | winner={winner}")

        # Short delay to see the final board
        time.sleep(1.0)

    env.close()
    print("[Mode 1] Finished data collection.")


# ============================================================
# 8) Mode 2: Behavioral Cloning (Supervised pre-training)
# ============================================================
def _obs_batch_to_action_mask(obs_batch: np.ndarray) -> np.ndarray:
    """
    obs_batch: (B, 3, 15, 15)
    mask: (B, 225) where 1=empty(valid), 0=occupied(invalid)
    """
    occ = (obs_batch[:, 0] > 0.5) | (obs_batch[:, 1] > 0.5)
    mask = (~occ).reshape(obs_batch.shape[0], -1).astype(np.float32)
    return mask


def load_human_dataset(dataset_dir: str) -> Tuple[np.ndarray, np.ndarray]:
    files = sorted(glob.glob(os.path.join(dataset_dir, "game_*.npz")))
    if not files:
        raise FileNotFoundError(f"No dataset files found in: {dataset_dir}")

    obs_list: List[np.ndarray] = []
    act_list: List[np.ndarray] = []
    for fp in files:
        data = np.load(fp)
        obs_list.append(data["obs"].astype(np.float32))
        act_list.append(data["actions"].astype(np.int64))

    X = np.concatenate(obs_list, axis=0)  # (N,3,15,15)
    y = np.concatenate(act_list, axis=0)  # (N,)
    return X, y


def run_behavioral_cloning(
    dataset_dir: str = "gomoku_human_dataset",
    pretrained_out_path: str = "pretrained_model.pth",
    seed: int = 42,
    epochs: int = 5,
    batch_size: int = 128,
    lr: float = 1e-3,
    device_str: str = "auto",
):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if device_str == "cpu":
        device = "cpu"
    elif device_str == "cuda":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"[Mode 2] Device: {device}")
    X, y = load_human_dataset(dataset_dir)
    print(f"[Mode 2] Loaded dataset: X={X.shape}, y={y.shape}")

    model = GomokuCNN().to(device)

    # Freeze value head; only train conv + policy head.
    for p in model.value_fc.parameters():
        p.requires_grad = False

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    criterion = nn.CrossEntropyLoss()

    model.train()
    n = X.shape[0]
    steps_per_epoch = max(1, (n + batch_size - 1) // batch_size)

    for ep in range(epochs):
        # Shuffle indices each epoch
        idx = np.random.permutation(n)
        total_loss = 0.0

        for step in range(steps_per_epoch):
            start = step * batch_size
            end = min(start + batch_size, n)
            if start >= end:
                continue
            b_idx = idx[start:end]

            obs_b = X[b_idx]
            y_b = y[b_idx]

            mask_b = _obs_batch_to_action_mask(obs_b)  # (B,225)

            obs_t = torch.tensor(obs_b, dtype=torch.float32, device=device)
            y_t = torch.tensor(y_b, dtype=torch.long, device=device)
            mask_t = torch.tensor(mask_b, dtype=torch.float32, device=device)

            logits, _ = model(obs_t)
            logits_masked = logits.masked_fill(mask_t < 0.5, -1e9)
            loss = criterion(logits_masked, y_t)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()
            total_loss += float(loss.item()) * (end - start)

        avg_loss = total_loss / n
        print(f"[Mode 2] Epoch {ep + 1}/{epochs} - avg_loss={avg_loss:.6f}")

    # Save full model weights (conv + actor + value head params)
    os.makedirs(os.path.dirname(pretrained_out_path) or ".", exist_ok=True)
    torch.save(model.state_dict(), pretrained_out_path)
    print(f"[Mode 2] Saved pre-trained model: {pretrained_out_path}")


# ============================================================
# 9) Mode 3: PPO Self-play Reinforcement Learning
# ============================================================
def _resolve_device(device_str: str) -> str:
    device_str = (device_str or "auto").lower()
    if device_str == "cpu":
        return "cpu"
    if device_str == "cuda":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


def run_self_play_ppo(
    pretrained_model_path: str = "pretrained_model.pth",
    model_dir: str = "gomoku_models",
    total_timesteps: int = 200000,
    save_interval: int = 10000,
    batch_size: int = 128,
    n_epochs: int = 4,
    lr: float = 3e-4,
    gamma: float = 0.99,
    opponent_pool_size: int = 10,
    render: bool = False,
    device_str: str = "auto",
    seed: int = 42,
):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    device = _resolve_device(device_str)
    print(f"[Mode 3] Device: {device}")

    env = OmokEnv(render_mode="human" if render else None)
    model = GomokuCNN().to(device)

    if pretrained_model_path and os.path.isfile(pretrained_model_path):
        state = torch.load(pretrained_model_path, map_location=device)
        model.load_state_dict(state, strict=False)
        print(f"[Mode 3] Loaded pretrained weights: {pretrained_model_path}")
    else:
        print(f"[Mode 3] WARNING: pretrained_model_path not found: {pretrained_model_path}")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    buffer = PPOBuffer(board_size=BOARD_SIZE, gamma=gamma, use_augmentation=True)
    pool = OpponentPool(model_dir=model_dir, max_pool_size=opponent_pool_size)

    os.makedirs(model_dir, exist_ok=True)
    global_step = 0
    ep_count = 0
    current_save_path: Optional[str] = None

    print("[Mode 3] Training start...")
    start_time = time.time()

    # Train until global_step (count of "our model" actions) reaches target.
    while global_step < total_timesteps:
        # Choose opponent snapshot if available
        opp_path = pool.sample_opponent(current_path=current_save_path)
        if opp_path and os.path.isfile(opp_path):
            opponent = GomokuCNN().to(device)
            opponent.load_state_dict(torch.load(opp_path, map_location=device), strict=False)
            opponent.eval()
        else:
            opponent = model  # fallback: self-play against itself

        # Randomly decide which color the current model plays in this episode
        current_model_is_black = random.random() < 0.5
        our_player_id = 1 if current_model_is_black else 2

        obs, info = env.reset()
        mask = info["action_mask"]
        terminated = False
        truncated = False

        # Collect only transitions where "our model" is acting.
        while not (terminated or truncated) and global_step < total_timesteps:
            env_current_player = int(info["current_player"])
            if env_current_player == our_player_id:
                action, value, logp_old = sample_action(
                    model=model,
                    obs=obs,
                    mask=mask,
                    device=device,
                    deterministic=False,
                )

                next_obs, reward, terminated, truncated, info = env.step(action)
                next_mask = info["action_mask"]

                buffer.add(
                    obs=obs,
                    action=action,
                    mask=mask,
                    reward=reward,
                    value=value,
                    logp_old=logp_old,
                )

                obs = next_obs
                mask = next_mask
                global_step += 1
                if render:
                    env.render()

                # If done, assign terminal reward based on winner
                if terminated or truncated:
                    winner = int(info.get("winner", 0))
                    term_r = terminal_reward_from_winner(winner=winner, our_player_id=our_player_id)
                    buffer.set_last_reward(term_r)
                    buffer.finish_trajectory(last_done=True)
                    break
            else:
                # Opponent acts; we do not store transitions in our buffer.
                with torch.no_grad():
                    opp_action, _, _ = sample_action(
                        model=opponent,
                        obs=obs,
                        mask=mask,
                        device=device,
                        deterministic=False,
                    )
                next_obs, reward, terminated, truncated, info = env.step(opp_action)
                obs = next_obs
                mask = info["action_mask"]
                if render:
                    env.render()

                if terminated or truncated:
                    winner = int(info.get("winner", 0))
                    # Assign terminal reward to our last action in buffer.
                    term_r = terminal_reward_from_winner(winner=winner, our_player_id=our_player_id)
                    buffer.set_last_reward(term_r)
                    buffer.finish_trajectory(last_done=True)
                    break

        # If we have data, run PPO updates
        if len(buffer) >= batch_size:
            loss_values = []
            for _ep in range(n_epochs):
                for batch in buffer.get_batches(batch_size):
                    loss = ppo_update(
                        model=model,
                        optimizer=optimizer,
                        batch=batch,
                        device=device,
                    )
                    if loss is None:
                        print("[Mode 3] NaN/inf detected. Stopping early for safety.")
                        break
                    loss_values.append(loss)
            buffer.clear()

        ep_count += 1

        # Periodic checkpointing
        if save_interval > 0 and global_step > 0 and (global_step % save_interval == 0):
            ckpt_path = os.path.join(model_dir, f"gomoku_ppo_{global_step}.pth")
            torch.save(model.state_dict(), ckpt_path)
            pool.add(ckpt_path)
            current_save_path = ckpt_path
            elapsed = time.time() - start_time
            print(
                f"[Mode 3] step={global_step} episodes={ep_count} saved={ckpt_path} "
                f"elapsed={elapsed/60:.1f}m"
            )

    final_path = os.path.join(model_dir, "gomoku_ppo_final.pth")
    torch.save(model.state_dict(), final_path)
    print(f"[Mode 3] Training done. Final model: {final_path}")
    env.close()


# ============================================================
# 10) Mode 4: AI vs Human Play Mode
# ============================================================
def run_ai_vs_human(
    model_path: str = "gomoku_models/gomoku_ppo_final.pth",
    human_plays_black: bool = True,
    device_str: str = "auto",
):
    device = _resolve_device(device_str)
    print(f"[Mode 4] Device: {device}")

    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")

    env = OmokEnv(render_mode="human")
    model = GomokuCNN().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device), strict=False)
    model.eval()

    human = HumanAgent(env, name="Human")

    obs, info = env.reset()
    mask = info["action_mask"]
    terminated = False
    truncated = False
    env.render()

    our_human_player_id = 1 if human_plays_black else 2

    current_player = int(info["current_player"])
    print(
        f"[Mode 4] Starting: human plays {'Black' if human_plays_black else 'White'} "
        f"(player_id={our_human_player_id})"
    )

    while not (terminated or truncated):
        current_player = int(info["current_player"])
        mask = info["action_mask"]

        if current_player == our_human_player_id:
            action = human.select_action(mask)
        else:
            action, _, _ = sample_action(
                model=model, obs=obs, mask=mask, device=device, deterministic=True
            )

        obs, reward, terminated, truncated, info = env.step(action)
        env.render()
        if not terminated and not truncated:
            time.sleep(0.05)

    winner = int(info.get("winner", 0))
    print("[Mode 4] Game finished.")
    if winner == 0:
        print("[Mode 4] Result: draw")
    elif winner == our_human_player_id:
        print("[Mode 4] Result: human wins")
    else:
        print("[Mode 4] Result: AI wins")

    time.sleep(2.0)
    env.close()


# ============================================================
# 11) Main interactive runner
# ============================================================
def _prompt_int(msg: str, default: int) -> int:
    s = input(msg).strip()
    if not s:
        return default
    return int(s)


def _prompt_str(msg: str, default: str) -> str:
    s = input(msg).strip()
    if not s:
        return default
    return s


def main():
    print("=== Gomoku Pipeline: Data Collection -> Behavioral Cloning -> PPO Self-play ===")
    print("Select mode:")
    print("  1) Data collection (Human vs Human, save trajectories)")
    print("  2) Behavioral cloning pre-training (supervised learning)")
    print("  3) PPO self-play reinforcement learning (opponent pool + augmentation)")
    print("  4) AI vs Human play mode")

    mode = _prompt_str("Enter mode number (1-4): ", "1")

    if mode == "1":
        dataset_dir = _prompt_str("Dataset dir: ", "gomoku_human_dataset")
        n_games = _prompt_int("Number of games to collect: ", 30)
        run_data_collection(dataset_dir=dataset_dir, n_games=n_games)
    elif mode == "2":
        dataset_dir = _prompt_str("Dataset dir: ", "gomoku_human_dataset")
        out_path = _prompt_str("Pretrained out path (.pth): ", "pretrained_model.pth")
        epochs = _prompt_int("BC epochs: ", 5)
        batch_size = _prompt_int("BC batch size: ", 128)
        lr = float(_prompt_str("BC learning rate (e.g. 0.001): ", "0.001"))
        device_str = _prompt_str("Device (auto/cpu/cuda): ", "auto")
        run_behavioral_cloning(
            dataset_dir=dataset_dir,
            pretrained_out_path=out_path,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            device_str=device_str,
        )
    elif mode == "3":
        pretrained_path = _prompt_str("Pretrained model path (.pth): ", "pretrained_model.pth")
        model_dir = _prompt_str("Model dir to save checkpoints: ", "gomoku_models")
        total_timesteps = _prompt_int("PPO total timesteps (our actions): ", 200000)
        save_interval = _prompt_int("Save interval (our actions): ", 10000)
        batch_size = _prompt_int("PPO batch size: ", 128)
        n_epochs = _prompt_int("PPO update epochs per episode: ", 4)
        lr = float(_prompt_str("PPO learning rate: ", "0.0003"))
        gamma = float(_prompt_str("Gamma: ", "0.99"))
        opponent_pool_size = _prompt_int("Opponent pool size: ", 10)
        render = _prompt_str("Render training GUI? (y/n): ", "n").lower().startswith("y")
        device_str = _prompt_str("Device (auto/cpu/cuda): ", "auto")
        run_self_play_ppo(
            pretrained_model_path=pretrained_path,
            model_dir=model_dir,
            total_timesteps=total_timesteps,
            save_interval=save_interval,
            batch_size=batch_size,
            n_epochs=n_epochs,
            lr=lr,
            gamma=gamma,
            opponent_pool_size=opponent_pool_size,
            render=render,
            device_str=device_str,
        )
    elif mode == "4":
        model_path = _prompt_str("Model path (.pth): ", os.path.join("gomoku_models", "gomoku_ppo_final.pth"))
        human_color = _prompt_str("Human plays black? (y/n): ", "y").lower().startswith("y")
        device_str = _prompt_str("Device (auto/cpu/cuda): ", "auto")
        run_ai_vs_human(model_path=model_path, human_plays_black=human_color, device_str=device_str)
    else:
        print("Invalid mode. Exiting.")


if __name__ == "__main__":
    main()

