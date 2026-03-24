"""
Gomoku (오목) PPO + Self-play AI — 규칙 없음 (순수 정책 학습)
- gomoku2.py와 동일한 환경/CNN/PPO/Self-play 구조, 규칙 기반 수 선택만 제거
- gomoku2에서 저장한 가중치를 --resume으로 불러와 학습 가능
"""
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


class OmokEnv(gym.Env):
    """Gymnasium 환경: Multi-channel 관측, action_mask, 렌주룰 없음."""
    metadata = {"render_modes": ["ansi", "human"], "render_fps": 4}

    def __init__(self, board_size=15, render_mode=None):
        super().__init__()
        self.board_size = board_size
        self.render_mode = render_mode
        self.action_space = spaces.Discrete(board_size ** 2)
        self.observation_space = spaces.Box(
            low=0.0, high=1.0,
            shape=(3, board_size, board_size),
            dtype=np.float32
        )
        self.board = None
        self.current_player = 1
        self.move_order = None
        self.move_count = 0
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
            self.window.title("AI 오목 대전 (규칙 없음)")
            size = (self.board_size - 1) * self.cell_size + self.margin * 2
            self.canvas = tk.Canvas(self.window, width=size, height=size, bg="#DCB35C")
            self.canvas.pack()
            self._font_small = tkfont.Font(family="Consolas", size=9, weight="bold")
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
# 2. Data Augmentation (8방향)
# =============================================================================

def augment_obs_action(obs, action, board_size=15):
    out = []
    out.append((obs.copy(), action))
    r, c = action // board_size, action % board_size
    o1 = np.rot90(obs, k=1, axes=(1, 2))
    out.append((o1, c * board_size + (board_size - 1 - r)))
    o2 = np.rot90(obs, k=2, axes=(1, 2))
    out.append((o2, (board_size - 1 - r) * board_size + (board_size - 1 - c)))
    o3 = np.rot90(obs, k=3, axes=(1, 2))
    out.append((o3, (board_size - 1 - c) * board_size + r))
    o4 = np.flip(obs, axis=2).copy()
    out.append((o4, r * board_size + (board_size - 1 - c)))
    o5 = np.flip(obs, axis=1).copy()
    out.append((o5, (board_size - 1 - r) * board_size + c))
    o6 = np.rot90(np.flip(obs, axis=2), k=1, axes=(1, 2))
    out.append((o6, (board_size - 1 - c) * board_size + r))
    o7 = np.rot90(np.flip(obs, axis=1), k=1, axes=(1, 2))
    out.append((o7, c * board_size + (board_size - 1 - r)))
    return out

def augment_mask(mask, action, board_size=15):
    m = mask.reshape(board_size, board_size)
    return [
        mask.copy(),
        np.rot90(m, k=1).flatten(),
        np.rot90(m, k=2).flatten(),
        np.rot90(m, k=3).flatten(),
        np.flip(m, axis=1).flatten(),
        np.flip(m, axis=0).flatten(),
        np.rot90(np.flip(m, axis=1), k=1).flatten(),
        np.rot90(np.flip(m, axis=0), k=1).flatten(),
    ]


# =============================================================================
# 3. CNN Policy / Value
# =============================================================================

class GomokuCNN(nn.Module):
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
        self.policy_fc = nn.Sequential(nn.Linear(flat_size, hidden), nn.ReLU(), nn.Linear(hidden, n_actions))
        self.value_fc = nn.Sequential(nn.Linear(flat_size, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def forward(self, x):
        b = x.size(0)
        h = self.conv(x)
        h_flat = h.view(b, -1)
        logits = self.policy_fc(h_flat)
        value = self.value_fc(h_flat).squeeze(-1)
        return logits, value

    def get_action(self, obs, mask, deterministic=False):
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
# 4. PPO Buffer & Update
# =============================================================================

class PPOBuffer:
    def __init__(self, board_size=15, gamma=0.99, use_augmentation=True):
        self.board_size = board_size
        self.gamma = gamma
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
        if self.ret_list:
            self.ret_list[-1] = reward

    def finish_trajectory(self, last_value=0.0, last_done=True):
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
            self.obs_list, self.act_list = aug_obs, aug_act
            self.mask_list, self.ret_list = aug_mask, aug_ret
            self.adv_list, self.logp_old_list = aug_adv, aug_logp

    def get_batches(self, batch_size):
        n = len(self.obs_list)
        idx = np.random.permutation(n)
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            ind = idx[start:end]
            yield (
                np.stack([self.obs_list[i] for i in ind], axis=0),
                np.array([self.act_list[i] for i in ind], dtype=np.int64),
                np.stack([self.mask_list[i] for i in ind], axis=0),
                np.array([self.ret_list[i] for i in ind], dtype=np.float32),
                np.array([self.adv_list[i] for i in ind], dtype=np.float32),
                np.array([self.logp_old_list[i] for i in ind], dtype=np.float32),
            )

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
    def __init__(self, model_dir, model_class, device, max_pool_size=10):
        self.model_dir = model_dir
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
        self._refresh()
        candidates = [p for p in self.paths if p != current_path]
        return random.choice(candidates) if candidates else None


# =============================================================================
# 6. Human Agent
# =============================================================================

class HumanAgent:
    def __init__(self, env, name="Human"):
        self.env = env
        self.name = name
        self.clicked_action = None
        self.current_state = None

    def select_action(self, state, mask=None):
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
            flat = self.current_state.flatten() if self.current_state.ndim == 3 else self.current_state
            if len(flat) == 225 and flat[action] == 0:
                self.clicked_action = action


# =============================================================================
# 7. Train / Play (규칙 없음 — PPO만 사용)
# =============================================================================

def _resolve_device(device_str: str) -> str:
    device_str = (device_str or "auto").lower()
    if device_str == "cpu":
        return "cpu"
    if device_str == "cuda":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


def run_training(
    total_timesteps=200_000,
    save_interval=10_000,
    opponent_pool_size=5,
    lr=3e-4,
    batch_size=64,
    n_epochs=4,
    model_dir="gomoku3_models",
    seed=42,
    render=False,
    device_str: str = "auto",
    resume_from: str | None = None,
):
    device = _resolve_device(device_str)
    print(f"[Train] Using device: {device} (no rules, PPO only)")
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

        while not done and global_step < total_timesteps:
            obs_t = torch.tensor(obs[np.newaxis, ...], dtype=torch.float32, device=device)
            mask_t = torch.tensor(mask, dtype=torch.float32, device=device)
            with torch.no_grad():
                logits, value = model(obs_t)
                logits_np = logits.cpu().numpy().reshape(-1)
                logits_np = np.where(mask > 0.5, logits_np, -1e9)
                probs = np.exp(logits_np - logits_np.max())
                probs = probs * mask
                if probs.sum() <= 0:
                    action = int(np.where(mask > 0)[0][0])
                else:
                    probs = probs / probs.sum()
                    action = int(np.random.choice(n_actions, p=probs))
                logits_masked = logits.masked_fill(mask_t.unsqueeze(0) < 0.5, -1e9)
                dist = Categorical(logits=logits_masked)
                logp_old = dist.log_prob(torch.tensor([action], device=device)).item()

            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            buffer.add(obs, action, mask, reward, value.cpu().item(), logp_old)
            obs, mask = next_obs, info["action_mask"]
            global_step += 1

            if total_timesteps > 0:
                interval = max(1, total_timesteps // 100)
                if global_step % interval == 0 or global_step == total_timesteps:
                    print(f"[Train] progress: {global_step}/{total_timesteps} ({100.0 * global_step / total_timesteps:.1f}%)")

            if done:
                buffer.finish_trajectory(last_value=0.0, last_done=True)
                n_episodes += 1
                if len(buffer) >= batch_size:
                    nan_detected = False
                    for _ in range(n_epochs):
                        for batch in buffer.get_batches(batch_size):
                            if ppo_update(model, optimizer, batch, device=device) is None:
                                nan_detected = True
                                break
                        if nan_detected:
                            break
                    if nan_detected:
                        print("[Train] NaN detected. Stopping. Resume from last saved checkpoint.")
                        buffer.clear()
                        env.close()
                        return
                    buffer.clear()
                break

            if not done:
                inv_obs = np.stack([next_obs[1], next_obs[0], next_obs[2]], axis=0)
                inv_mask = get_action_mask(env.board)
                opp_obs_t = torch.tensor(inv_obs[np.newaxis, ...], dtype=torch.float32, device=device)
                opp_action, _ = opponent.get_action(opp_obs_t, torch.tensor(inv_mask, device=device), deterministic=False)
                next_obs2, reward2, terminated, truncated, info = env.step(opp_action)
                done = terminated or truncated
                if done:
                    if info.get("winner") == 2:
                        buffer.set_last_reward(-1.0)
                    buffer.finish_trajectory(last_value=0.0, last_done=True)
                    n_episodes += 1
                    if len(buffer) >= batch_size:
                        nan_detected = False
                        for _ in range(n_epochs):
                            for batch in buffer.get_batches(batch_size):
                                if ppo_update(model, optimizer, batch, device=device) is None:
                                    nan_detected = True
                                    break
                            if nan_detected:
                                break
                        if nan_detected:
                            print("[Train] NaN detected. Stopping.")
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


def run_play(model_path, model_dir="gomoku3_models", human_plays_black=True, device_str: str = "auto"):
    device = _resolve_device(device_str)
    print(f"[Play] Using device: {device} (no rules)")
    env = OmokEnv(render_mode="human")
    board_size = env.board_size
    model = GomokuCNN(board_size=board_size, n_actions=board_size ** 2).to(device)
    if model_path is None or not os.path.isfile(model_path):
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
            board_flat = np.zeros(board_size * board_size, dtype=np.int8)
            board_flat[(state[0] > 0.5).flatten()] = current_player
            board_flat[(state[1] > 0.5).flatten()] = 3 - current_player
            action = human.select_action(board_flat, mask)
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
# 8. Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Gomoku PPO Self-play (no rules)")
    parser.add_argument("--mode", type=str, default="play", choices=["train", "play"])
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--model_dir", type=str, default="gomoku3_models")
    parser.add_argument("--total_timesteps", type=int, default=200_000)
    parser.add_argument("--save_interval", type=int, default=10_000)
    parser.add_argument("--human_black", action="store_true", default=True)
    parser.add_argument("--render_train", action="store_true")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--resume", type=str, default=None, help="이어서 학습할 가중치 경로 (예: gomoku_models\\gomoku_ppo_final.pth)")
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
