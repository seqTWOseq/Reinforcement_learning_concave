import gymnasium as gym
from gymnasium import spaces
import tkinter as tk

import numpy as np
from numba import njit
import random
from collections import deque
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import time
from tqdm import tqdm
import matplotlib.pyplot as plt

from khy_model import DualHeadResOmokCNN

# ==========================================
# 1. 오목 강화학습 환경 (GUI 포함)
# ==========================================
class OmokEnvGUI(gym.Env):
    metadata = {"render_modes": ["ansi", "human"], "render_fps": 4}

    def __init__(self, render_mode="human"):
        super().__init__()
        self.board_size = 15
        self.render_mode = render_mode
        
        # 행동 공간 (0~224) 및 상태 공간 (15x15 배열, 0:빈칸, 1:흑, 2:백)
        self.action_space = spaces.Discrete(self.board_size ** 2)
        self.observation_space = spaces.Box(low=0, high=2, shape=(self.board_size, self.board_size), dtype=np.int8)
        
        self.board = None
        self.current_player = 1 # 1: 흑, 2: 백
        
        # GUI 설정
        self.window = None
        self.cell_size, self.margin = 40, 30

    def reset(self, seed=None, options=None):
        """보드를 초기화하고 흑돌 턴으로 시작합니다."""
        super().reset(seed=seed)
        self.board = np.zeros((self.board_size, self.board_size), dtype=np.int8)
        self.current_player = 1
        return self.board.copy(), {"current_player": self.current_player}

    def step(self, action):
        """에이전트의 행동을 적용하고 상태, 보상, 종료 여부를 반환합니다."""
        r, c = action // self.board_size, action % self.board_size
        
        # 1. 반칙 (이미 돌이 있는 곳) 처리
        if self.board[r, c] != 0:
            return self.board.copy(), -10.0, True, False, {"reason": "invalid_move", "winner": 3 - self.current_player}

        # 2. 착수 및 승리/무승부 판정
        self.board[r, c] = self.current_player
        if self._check_win(r, c, self.current_player):
            return self.board.copy(), 1.0, True, False, {"reason": "win", "winner": self.current_player}
        if not np.any(self.board == 0):
            return self.board.copy(), 0.0, True, False, {"reason": "draw", "winner": 0}

        # 3. 턴 교체
        self.current_player = 3 - self.current_player
        return self.board.copy(), 0.0, False, False, {"current_player": self.current_player}

    def _check_win(self, row, col, player):
        """4방향 탐색으로 5목 완성 여부를 논리적으로 확인합니다."""
        directions = [(0, 1), (1, 0), (1, 1), (-1, 1)]
        for dr, dc in directions:
            count = 1
            for step in (1, -1): # 정방향(1), 역방향(-1) 탐색
                r, c = row + dr * step, col + dc * step
                while 0 <= r < self.board_size and 0 <= c < self.board_size and self.board[r, c] == player:
                    count += 1
                    r += dr * step; c += dc * step
            if count >= 5: return True
        return False

    def render(self):
        """Tkinter 창을 새로고침하여 현재 보드를 시각화합니다."""
        if self.render_mode != "human": return
        
        if self.window is None:
            self.window = tk.Tk()
            self.window.title("AI 오목 대전 시뮬레이터")
            size = (self.board_size - 1) * self.cell_size + self.margin * 2
            self.canvas = tk.Canvas(self.window, width=size, height=size, bg="#DCB35C")
            self.canvas.pack()

        self.canvas.delete("all")
        
        # 격자선 그리기
        for i in range(self.board_size):
            start, end = self.margin + i * self.cell_size, self.margin + (self.board_size - 1) * self.cell_size
            self.canvas.create_line(self.margin, start, end, start)
            self.canvas.create_line(start, self.margin, start, end)

        # 돌 그리기
        for r in range(self.board_size):
            for c in range(self.board_size):
                if self.board[r, c] != 0:
                    x, y = self.margin + c * self.cell_size, self.margin + r * self.cell_size
                    rad = self.cell_size // 2 - 2
                    color = "black" if self.board[r, c] == 1 else "white"
                    self.canvas.create_oval(x - rad, y - rad, x + rad, y + rad, fill=color, outline="black")

        # 화면 비동기 업데이트 (mainloop 대체)
        self.window.update_idletasks()
        self.window.update()

    def close(self):
        if self.window: self.window.destroy(); self.window = None

# ==========================================
# 2. 에이전트 클래스 (여기서부터 알고리즘 작성 필요)
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
        
        # 마우스 왼쪽 클릭 이벤트 활성화
        self.env.canvas.bind("<Button-1>", self._click_handler)
        
        # 클릭될 때까지 무한 대기 (화면은 멈추지 않도록 update 호출)
        while self.clicked_action is None:
            self.env.window.update()
            time.sleep(0.05) # CPU 과부하 방지
            
        # 행동 결정 완료 시 이벤트 해제 (AI 턴에 클릭 방지)
        self.env.canvas.unbind("<Button-1>")
        
        return self.clicked_action

    def _click_handler(self, event):
        # 클릭한 픽셀 위치를 오목판 논리적 좌표(행, 열)로 변환
        c = round((event.x - self.env.margin) / self.env.cell_size)
        r = round((event.y - self.env.margin) / self.env.cell_size)
        
        # 보드 범위 내인지 확인
        if 0 <= r < self.env.board_size and 0 <= c < self.env.board_size:
            action = r * self.env.board_size + c
            
            # 유효한 빈칸(0)을 클릭했을 때만 값 업데이트 (반칙 클릭 무시)
            if self.current_state.flatten()[action] == 0:
                self.clicked_action = action

class HeuristicAgent:
    def __init__(self, name="Heuristic_AI", mistake_prob=0.0):
        self.name = name
        self.mistake_prob = mistake_prob

    def select_action(self, state, move_count=0, **kwargs):
        board_size = state.shape[0]
        valid_moves = np.where(state.flatten() == 0)[0]
        if len(valid_moves) == 0: return 0

        # [추가] 실수 로직: 설정된 확률에 따라 완전히 무작위로 착수
        # AI에게 "빈틈"을 만들어주는 핵심 장치입니다.
        if random.random() < self.mistake_prob:
            return np.random.choice(valid_moves)

        # 보드가 비어있다면 중앙(7, 7)에 착수
        if np.sum(state != 0) == 0:
            return (board_size // 2) * board_size + (board_size // 2)

        best_score = -float('inf')
        best_actions = []

        # 기존 돌 주변 탐색 (반경 2칸으로 최적화)
        candidates = self._get_candidate_moves(state, board_size)
        if not candidates:
            candidates = valid_moves

        for action in candidates:
            r, c = action // board_size, action % board_size
            
            # 내가 착수했을 때의 공격 점수 (현재 플레이어 기준)
            offense_score = self._evaluate_position(state, r, c, player=1)
            # 상대가 착수했을 때의 방어 점수 (상대 플레이어 기준)
            defense_score = self._evaluate_position(state, r, c, player=2)
            
            # 방어 가중치를 높여서 AI의 공격을 끈질기게 막도록 유도
            total_score = offense_score + (defense_score * 1.2)

            if total_score > best_score:
                best_score = total_score
                best_actions = [action]
            elif total_score == best_score:
                best_actions.append(action)

        return np.random.choice(best_actions)

    def _get_candidate_moves(self, state, board_size):
        candidates = set()
        occupied = np.argwhere(state != 0)
        # 반경 2칸 이내면 오목의 모든 주요 패턴을 커버하기에 충분합니다. (연산량 절감)
        for r, c in occupied:
            for dr in range(-2, 3):
                for dc in range(-2, 3):
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < board_size and 0 <= nc < board_size and state[nr, nc] == 0:
                        candidates.add(nr * board_size + nc)
        return list(candidates)

    def _evaluate_position(self, state, row, col, player):
        directions = [(0, 1), (1, 0), (1, 1), (-1, 1)]
        score = 0
        board_size = state.shape[0]

        for dr, dc in directions:
            consecutive = 1
            open_ends = 0

            # 양방향 탐색 로직은 그대로 유지하되 가중치 밸런스 조정
            for step in (1, -1):
                r, c = row + dr * step, col + dc * step
                while 0 <= r < board_size and 0 <= c < board_size:
                    if state[r, c] == player:
                        consecutive += 1
                    elif state[r, c] == 0:
                        open_ends += 1
                        break
                    else:
                        break
                    r += dr * step
                    c += dc * step

            # 패턴별 가중치 (0.0 ~ 1.0 스케일로 정규화 및 우선순위 엄격 유지)
            if consecutive >= 5:
                score += 1.0       # 5목 완성 (승리/패배 직결, 최우선)
            elif consecutive == 4:
                if open_ends == 2: 
                    score += 0.1   # 열린 4목 (다음 턴 무조건 승리)
                elif open_ends == 1: 
                    score += 0.01  # 닫힌 4목 (방어 필수 유도)
            elif consecutive == 3:
                if open_ends == 2: 
                    score += 0.001 # 열린 3목 (공격 전개)
            elif consecutive == 2:
                if open_ends == 2: 
                    score += 0.0001 # 열린 2목 (초반 자리 선점)
                
        return score

# =================================================================

@njit
def check_pattern_fast(state, r, c, player, target, open_ends_req):
    """C언어 속도로 동작하는 패턴 확인 함수"""
    directions = np.array([[0, 1], [1, 0], [1, 1], [-1, 1]])
    board_size = state.shape[0]

    for d in range(4):
        dr = directions[d, 0]
        dc = directions[d, 1]
        consecutive = 1
        open_ends = 0

        for step in (1, -1):
            nr = r + dr * step
            nc = c + dc * step
            while 0 <= nr < board_size and 0 <= nc < board_size:
                if state[nr, nc] == player:
                    consecutive += 1
                elif state[nr, nc] == 0:
                    open_ends += 1
                    break
                else:
                    break
                nr += dr * step
                nc += dc * step
        
        if consecutive >= target and open_ends >= open_ends_req:
            return True
    return False

@njit
def find_urgent_move_fast(state, valid_moves, player):
    """C언어 속도로 동작하는 위급 수 탐색 함수 (Numba는 None을 쓸 수 없어 -1 반환)"""
    board_size = state.shape[0]
    opponent = 3 - player
    best_move = -1

    for i in range(len(valid_moves)):
        move = valid_moves[i]
        r = move // board_size
        c = move % board_size
        
        # 1순위: 내가 두면 바로 승리 (5목 완성)
        if check_pattern_fast(state, r, c, player, 5, 0):
            return move
        
        # 2순위: 상대가 두면 바로 패배 (상대 5목 완성 차단)
        # 단, 내가 당장 이길 수 있는 수가 있다면 그게 우선이므로 1순위 아래에 배치
        if best_move == -1 and check_pattern_fast(state, r, c, opponent, 5, 0):
            best_move = move

    return best_move

@njit
def fast_rollout_fast(state, action, max_depth, max_moves=100):
    """극단적으로 최적화된 초고속 MCTS 시뮬레이션 엔진"""
    board_size = state.shape[0]
    sim_state = state.copy()
    r = action // board_size
    c = action % board_size
    sim_state[r, c] = 1 
    
    current_stones = 0
    for i in range(board_size):
        for j in range(board_size):
            if sim_state[i, j] != 0:
                current_stones += 1
                
    # 내가 방금 둔 수로 즉시 승리
    if check_pattern_fast(sim_state, r, c, 1, 5, 0):
        return 1.0 
        
    current_player = 2
    depth_penalty_weight = 0.01

    valid_moves = np.where(sim_state.flatten() == 0)[0]
    num_valid = len(valid_moves)

    for depth in range(max_depth):
        # 100수 도달 시 (사용자 요청: -0.4)
        if current_stones >= max_moves:
            return -0.4
            
        # 판이 꽉 찼을 때 (자연 무승부)
        if num_valid == 0:
            return -0.2 
        
        idx = np.random.randint(num_valid)
        sim_action = valid_moves[idx]
        
        valid_moves[idx] = valid_moves[num_valid - 1]
        num_valid -= 1
            
        sr = sim_action // board_size
        sc = sim_action % board_size
        sim_state[sr, sc] = current_player
        current_stones += 1 
        
        if check_pattern_fast(sim_state, sr, sc, current_player, 5, 0):
            penalty = depth * depth_penalty_weight
            if current_player == 1:
                # 늦게 이길수록 보상이 깎여서 100수 제한(-0.4)에 가까워짐
                return max(-0.39, 1.0 - penalty) 
            else:
                return min(0.39, -1.0 + penalty) 
            
        current_player = 3 - current_player
        
    # 메모리의 무승부 기준(-0.2)을 반환하여 중립적인 가치를 부여
    return -0.2
    
class KhyAgent:
    def __init__(self, model):
        self.name = "Khy_AI"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.0005, weight_decay=1e-4)

        self.is_training = True
        
        # 탐험 파라미터
        self.epsilon = 1.0
        self.epsilon_min = 0.00
        self.epsilon_decay = 0.999
        
        # 경험 재생 메모리
        self.memory = deque(maxlen=50000)
        self.batch_size = 1024
        self.gamma = 0.99
    
    # ====================
    # 행동 선택 로직
    def _normalize_to_range(self, values, valid_moves, target_min=-1.0, target_max=1.0):
        board_size = int(np.sqrt(len(values)))
        result = np.full(len(values), -float('inf'), dtype=np.float32)
        
        valid_values = values[valid_moves]
        if len(valid_values) == 0:
            return result
            
        v_min = np.min(valid_values)
        v_max = np.max(valid_values)
        
        if v_max > v_min:
            # 1. 0 ~ 1 사이로 Min-Max 스케일링
            scaled = (valid_values - v_min) / (v_max - v_min)
            # 2. 목표 범위(target_min ~ target_max)로 변환
            result[valid_moves] = (scaled * (target_max - target_min)) + target_min
        else:
            # 모든 값의 차이가 없다면 특혜나 페널티 없이 중립(0.0) 부여
            result[valid_moves] = 0.0 
            
        return result
    
    def select_action(self, state, move_count=0):
        board_size = state.shape[0]
        total_grids = board_size * board_size 
        
        raw_valid_moves = np.where(state.flatten() == 0)[0]
        if len(raw_valid_moves) == 0:
            return 0
        
        occupied = np.argwhere(state != 0)
        if len(occupied) == 0:
            return (board_size // 2) * board_size + (board_size // 2)
        
        # 인접 빈칸 탐색, 위급 수 방어
        sensible_moves = set()
        for r, c in occupied:
            for dr in range(-2, 3):
                for dc in range(-2, 3):
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < board_size and 0 <= nc < board_size and state[nr, nc] == 0:
                        sensible_moves.add(nr * board_size + nc)
        valid_moves = np.array(list(sensible_moves))
            
        urgent_move = find_urgent_move_fast(state, valid_moves, player=1)
        if urgent_move != -1: 
            return urgent_move

        self.model.eval()

        # 현재 상태의 Policy(정책) 평가
        state_tensor = torch.FloatTensor(state).unsqueeze(0).unsqueeze(0).to(self.device)
        with torch.no_grad():
            policy_logits, _ = self.model(state_tensor)
            policy_logits = policy_logits.squeeze()

        valid_mask = torch.ones(total_grids, dtype=torch.bool).to(self.device)
        valid_mask[valid_moves] = False
        policy_logits[valid_mask] = -float('inf')
        policy_probs = F.softmax(policy_logits, dim=0).cpu().numpy()

        # Rollout 결과
        num_simulations = 400
        action_visits = np.zeros(total_grids)
        action_wins = np.zeros(total_grids)

        for _ in range(num_simulations):
            probs = policy_probs[valid_moves]
            probs /= np.sum(probs)
            sim_action = np.random.choice(valid_moves, p=probs)
            
            reward = fast_rollout_fast(state, sim_action, max_depth=30)
            action_visits[sim_action] += 1
            action_wins[sim_action] += reward
        
        raw_rollout_values = np.divide(action_wins, action_visits, out=np.zeros_like(action_wins), where=action_visits!=0)

        # 가치 일괄 평가
        num_valid = len(valid_moves)
        next_states_batch = np.zeros((num_valid, 1, board_size, board_size), dtype=np.float32)
        
        for i, move in enumerate(valid_moves):
            r, c = move // board_size, move % board_size
            next_state = state.copy()
            next_state[r, c] = 1 
            canonical_next_state = np.where(next_state != 0, 3 - next_state, 0)
            next_states_batch[i, 0] = canonical_next_state
            
        batch_tensor = torch.FloatTensor(next_states_batch).to(self.device)
        with torch.no_grad():
            _, next_values = self.model(batch_tensor)
            next_values = next_values.flatten().cpu().numpy()
            
        raw_cnn_values = np.zeros(total_grids)
        raw_cnn_values[valid_moves] = -1.0 * next_values # 내 입장에서 부호 반전

        # 내재적 보상(Intrinsic)
        raw_intrinsic_rewards = np.zeros(total_grids)
        for move in valid_moves:
            raw_intrinsic_rewards[move] = self.get_intrinsic_reward(state, move)

        # 정규화
        norm_policy   = self._normalize_to_range(policy_probs, valid_moves)
        norm_rollout  = self._normalize_to_range(raw_rollout_values, valid_moves)
        norm_value    = self._normalize_to_range(raw_cnn_values, valid_moves)
        norm_int      = self._normalize_to_range(raw_intrinsic_rewards, valid_moves)
        
        w_policy  = 0.25
        w_rollout = 0.20
        w_value   = 0.25
        w_int     = 0.30

        # 최종 스코어 계산 (모든 값이 -1 ~ 1 스케일을 가짐)
        final_score = np.where(
            np.isin(np.arange(total_grids), valid_moves), 
            (w_policy * norm_policy) + 
            (w_rollout * norm_rollout) + 
            (w_value * norm_value) + 
            (w_int * norm_int), 
            -float('inf')
        )
        
        if self.is_training and np.random.rand() < self.epsilon:
            top_k = min(5, len(valid_moves))
            sorted_indices = np.argsort(final_score)[-top_k:]
            chosen_idx = np.random.choice(sorted_indices)
            return int(chosen_idx)
        else:
            return int(np.argmax(final_score))
    
    # ====================
    # 내재적 보상
    def get_intrinsic_reward(self, state, action):
        board_size = state.shape[0]
        r, c = action // board_size, action % board_size
        
        def evaluate_for_player(target_player):
            sim_state = state.copy()
            sim_state[r, c] = target_player 
            
            score = 0.0
            directions = [(0, 1), (1, 0), (1, 1), (-1, 1)]
            pattern_counts = {'open_3': 0, 'four': 0}
            
            for dr, dc in directions:
                consecutive = 1
                open_ends = 0
                
                for step in (1, -1):
                    nr, nc = r + dr * step, c + dc * step
                    while 0 <= nr < board_size and 0 <= nc < board_size:
                        if sim_state[nr, nc] == target_player:
                            consecutive += 1
                        elif sim_state[nr, nc] == 0:
                            open_ends += 1
                            break 
                        else:
                            break 
                        nr += dr * step
                        nc += dc * step
                
                # 1.0 스케일에 맞춘 점수 재조정
                if consecutive >= 5:
                    score += 0.5      # 승리/패배 직결 (최고점)
                elif consecutive == 4 and open_ends >= 1:
                    score += 0.2      # 열린 4목 (매우 높음)
                    pattern_counts['four'] += 1
                elif consecutive == 3 and open_ends == 2:
                    score += 0.08       # 열린 3목
                    pattern_counts['open_3'] += 1
            
            # 양수겸장 판단 (최대 1.0을 넘지 않도록 조정)
            if pattern_counts['four'] >= 2 or (pattern_counts['four'] >= 1 and pattern_counts['open_3'] >= 1) or pattern_counts['open_3'] >= 2:
                score = max(score, 0.25)
                
            return score

        # 공격 가치 (내가 두었을 때의 파괴력)
        attack_value = evaluate_for_player(1) 
        
        # 수비 가치 (상대가 두었을 때의 파괴력을 사전에 차단)
        defense_value = evaluate_for_player(2) 
        
        # 공격과 수비를 합치되, 절대 1.0을 넘지 않도록 강력한 상한선(Clipping) 적용
        total_reward = min((attack_value * 1.1) + defense_value, 1.0)
        
        return total_reward
    
    # ====================
    # 기억 장치 (데이터 증강 적용)
    def memorize_episode(self, episode_memory, final_reward):
        discounted_reward = final_reward
        step_cost = 0.005 
        intrinsic_weight = 0.05 
        
        for state, action, step_reward in reversed(episode_memory):
            # 내재적 보상 결합
            total_reward = discounted_reward + (step_reward * intrinsic_weight)
            
            # [수정] Value 헤드(tanh)의 범위인 -1 ~ 1 사이로 강제 고정 (학습 안정화)
            total_reward = np.clip(total_reward, -1.0, 1.0)
            
            # 2. 데이터 증강 (8방향 대칭)
            board_size = state.shape[0]
            action_matrix = np.zeros((board_size, board_size), dtype=np.int8)
            action_matrix[action // board_size, action % board_size] = 1
            
            for i in range(4):
                # 회전
                rot_s = np.rot90(state, k=i)
                rot_a = np.argmax(np.rot90(action_matrix, k=i))
                self.memory.append((rot_s.copy(), rot_a, total_reward))
                
                # 좌우 반전 후 회전
                flip_s = np.fliplr(rot_s)
                flip_a = np.argmax(np.fliplr(np.rot90(action_matrix, k=i)))
                self.memory.append((flip_s.copy(), flip_a, total_reward))
                
            # 다음(이전 턴) 계산을 위해 보상 업데이트
            if final_reward > 0:
                # 승리: 늦게 이길수록 가치 하락 (스텝 비용 차감)
                discounted_reward = discounted_reward * self.gamma - step_cost
            elif final_reward < 0:
                # 패배: 늦게 질수록 덜 나쁨 (즉, 이전 턴일수록 책임이 적음 -> 0에 가깝게 수렴)
                discounted_reward = discounted_reward * self.gamma + step_cost 
            else:
                # 무승부
                discounted_reward = discounted_reward * self.gamma
            
            # 하한선 방어
            discounted_reward = max(discounted_reward, -1.0)
    
    # ====================
    # 복습 엔진
    def replay_experience(self):
        if len(self.memory) < self.batch_size:
            return 0.0, 0.0 # 학습을 안 했을 때는 0 반환
        
        minibatch = random.sample(self.memory, self.batch_size)
        states, actions, targets = zip(*minibatch)

        states_tensor = torch.FloatTensor(np.array(states)).unsqueeze(1).to(self.device)
        actions_tensor = torch.LongTensor(actions).to(self.device) 
        targets_tensor = torch.FloatTensor(targets).unsqueeze(1).to(self.device) 

        policy_logits, values = self.model(states_tensor)
        
        value_loss = F.mse_loss(values, targets_tensor)
        policy_loss = F.cross_entropy(policy_logits, actions_tensor)
        
        # [업데이트] Loss 스케일 균형을 맞추기 위한 명시적 가중치 설정
        # Policy Loss의 기본 스케일이 훨씬 크므로 가중치를 0.5로 낮추고 Value에 집중
        weight_value = 1.5
        weight_policy = 0.3 

        total_loss = (value_loss * weight_value) + (policy_loss * weight_policy)

        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        
        # 외부에서 로깅할 수 있도록 순수 Python float 값으로 반환
        return value_loss.item(), policy_loss.item()
    
    # ====================
    # 유틸
    def train_mode(self):
        self.model.train()
        self.is_training = True
    
    def eval_mode(self):
        self.model.eval()
        self.is_training = False
        self.epsilon = 0.0
    
    def save_model(self, filepath):
        torch.save(self.model.state_dict(), filepath)

    def load_model(self, filepath):
        self.model.load_state_dict(torch.load(filepath, map_location=self.device, weights_only=True))
    
    def decay_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            
# ==========================================
# 3. 대결 실행 루프 (Arena)
# ==========================================
def main():
    env = OmokEnvGUI(render_mode="human")
    agent2 = HeuristicAgent()
    
    
    model = DualHeadResOmokCNN()
    agent1 = KhyAgent(model)
    agent1.load_model("khy_omok_heu.pth")
    agent1.eval_mode()
    
    state, info = env.reset()
    env.render()
    terminated = False
    
    print(f"=== ⚔️ {agent1.name} vs {agent2.name} 대결 시작 ===")
    
    while not terminated:
        # 턴에 따른 상태 반전 논리 (상대는 항상 자신이 흑돌인 것처럼 착각하게 만듦)
        if info["current_player"] == 1:
            start_time = time.time()
            action = agent1.select_action(state)
            end_time = time.time()
            print(f"   ▶ {agent1.name} 착수 완료! (소요 시간: {end_time - start_time:.2f}초)")
        else:
            inverted_state = np.where(state == 1, 2, np.where(state == 2, 1, 0))
            start_time = time.time()
            action = agent2.select_action(inverted_state)
            end_time = time.time()
            print(f"   ▶ {agent2.name} 착수 완료! (소요 시간: {end_time - start_time:.2f}초)")
            
        state, reward, terminated, _, info = env.step(action)
        env.render()
        time.sleep(0.5) # 시각적 확인을 위한 지연

    # 결과 판정
    print("\n=== 🏁 대결 종료 ===")
    winner = info.get("winner")
    if winner == 1: print(f"🎉 {agent1.name} 승리!")
    elif winner == 2: print(f"🎉 {agent2.name} 승리!")
    else: print("🤝 무승부!")
        
    time.sleep(3)
    env.close()

# ===================================================================    
plt.rcParams['axes.unicode_minus'] = False

class LivePlotter:
    def __init__(self, title="Real-time Training Status"):
        plt.ion()  # 대화형 모드 활성화 (창이 떠 있는 상태로 코드 진행 가능)
        self.fig, self.ax = plt.subplots(figsize=(8, 5))
        self.v_losses, self.p_losses = [], []
        self.ax.set_title(title)
        self.ax.set_xlabel("Episode")
        self.ax.set_ylabel("Loss")
        self.line_v, = self.ax.plot([], [], label="Value Loss", color='blue', alpha=0.6)
        self.line_p, = self.ax.plot([], [], label="Policy Loss", color='red', alpha=0.6)
        self.ax.legend()

    def update(self, v_loss, p_loss):
        self.v_losses.append(v_loss)
        self.p_losses.append(p_loss)
        
        # 데이터 업데이트
        x_data = range(len(self.v_losses))
        self.line_v.set_data(x_data, self.v_losses)
        self.line_p.set_data(x_data, self.p_losses)
        
        # 화면 축 자동 조정 및 갱신
        self.ax.relim()
        self.ax.autoscale_view()
        
        plt.pause(0.5) # 짧게 멈춰서 그래프 그릴 시간 확보
        plt.draw()

def train_main():
    env = OmokEnvGUI(render_mode=None)
    
    # [추가] 텐서보드 Writer 초기화 (실행할 때마다 runs 폴더 내에 기록됨)
    # writer = SummaryWriter("runs/omok_training_v1")
    plotter = LivePlotter(title="Real-time Training Status")
    
    # 학습할 메인 에이전트
    model1 = DualHeadResOmokCNN()  
    agent1 = KhyAgent(model1)
    agent1.load_model("khy_omok_heu.pth")
    print(f"[Device 확인] {agent1.device}")
    agent1.train_mode()
    
    # 셀프 대결을 위한 상대방 에이전트 (과거의 나)
    model2 = DualHeadResOmokCNN()
    agent2_self = KhyAgent(model2)
    agent2_self.model.load_state_dict(agent1.model.state_dict())
    agent2_self.eval_mode()
    
    N = 10
    EPISODES = 10000
    UPDATE_INTERVAL = 500 # 통계 리셋 및 상대방 진화 주기
    
    global_episode = 0 # [추가] 세대와 상관없이 누적되는 전체 에피소드 카운터 (텐서보드 X축)
    
    for gen in range(1, N + 1):
        print(f"\n{'='*40}\n[Generation {gen}/{N}] 제 {gen}세대\n{'='*40}")
        
        # 세대 시작 시 탐험률 초기화
        agent1.epsilon, agent1.epsilon_decay = 0.15, 0.9855
        
        # 10,000판을 500판 단위로 쪼개어 루프 실행
        for phase_start in range(1, EPISODES + 1, UPDATE_INTERVAL):
            phase_end = phase_start + UPDATE_INTERVAL - 1
            
            # 통계 초기화
            agent1_wins, draws, agent1_losses, total_steps = 0, 0, 0, 0
            pbar = tqdm(total=UPDATE_INTERVAL, desc=f"[Gen {gen}] {phase_start}~{phase_end}판", position=0, leave=True)
            
            for episode in range(phase_start, phase_end + 1):
                global_episode += 1 # [추가] 전체 누적 카운트 1 증가
                
                state, info = env.reset()
                terminated = False
                memory_b, memory_w = [], []
                current_episode_steps = 0 
                
                agent1_color = 1 if np.random.rand() < 0.5 else 2

                # --- 단일 에피소드(게임) 진행 ---
                while not terminated:
                    current_player = info["current_player"]
                    
                    # 100수 제한 강제 패배 로직
                    if current_episode_steps >= 100:
                        terminated = True
                        info["winner"] = 0 
                        break 
                    
                    if current_player == 2:
                        canonical_state = np.where(state != 0, 3 - state, 0)
                    else:
                        canonical_state = state.copy()
                        
                    is_opening = current_episode_steps < 2

                    is_agent1_turn = (current_player == agent1_color)
                    active_agent = agent1 if is_agent1_turn else agent2_self
                    
                    if is_opening:
                        valid_moves = np.where(canonical_state.flatten() == 0)[0]
                        action = np.random.choice(valid_moves)
                    else:
                        action = active_agent.select_action(canonical_state, move_count=current_episode_steps)
                        
                    step_reward = agent1.get_intrinsic_reward(canonical_state, action)
                    
                    if current_player == 1:
                        memory_b.append((canonical_state, action, step_reward))
                    else:
                        memory_w.append((canonical_state, action, step_reward))

                    next_state, reward, terminated, _, info = env.step(action)
                    state = next_state
                    current_episode_steps += 1

                total_steps += current_episode_steps

                # --- 게임 종료 후: 승패 기록 및 복습 ---
                winner = info.get("winner")
                if winner == agent1_color:
                    agent1_wins += 1
                    final_reward = 1.0  
                elif winner == 0: 
                    draws += 1
                    final_reward = -0.2
                else: 
                    agent1_losses += 1
                    final_reward = -1.0 

                if agent1_color == 1:
                    agent1.memorize_episode(memory_b, final_reward)
                else:
                    agent1.memorize_episode(memory_w, final_reward)
                    
                # [수정] 복습 시 반환된 Loss를 합산하여 에피소드 평균 Loss 계산
                ep_v_loss, ep_p_loss = 0.0, 0.0
                valid_trains = 0
                
                for _ in range(8):
                    v_loss, p_loss = agent1.replay_experience()
                    # 학습이 이루어졌을 때만 누적 (초반 메모리 부족 시 0 반환됨)
                    if v_loss > 0 or p_loss > 0: 
                        ep_v_loss += v_loss
                        ep_p_loss += p_loss
                        valid_trains += 1
                
                # [추가] 텐서보드에 네트워크 Loss 기록 (학습이 1번이라도 진행된 경우)
                if valid_trains > 0:
                    ep_v_loss /= valid_trains
                    ep_p_loss /= valid_trains
                    # writer.add_scalar("1_Loss/Value_Loss", ep_v_loss, global_episode)
                    # writer.add_scalar("1_Loss/Policy_Loss", ep_p_loss, global_episode)
                    
                    if global_episode % 10 == 0:
                        plotter.update(ep_v_loss, ep_p_loss)

                # --- 통계 계산 ---
                current_phase_ep = episode - phase_start + 1
                decisive_games = agent1_wins + agent1_losses 
                
                if decisive_games > 0:
                    win_rate = (agent1_wins / decisive_games) * 100 
                else:
                    win_rate = 0.0
                    
                avg_steps = total_steps // current_phase_ep
                
                # [추가] 텐서보드에 훈련 지표 기록
                # writer.add_scalar("2_Metrics/Win_Rate", win_rate, global_episode)
                # writer.add_scalar("2_Metrics/Steps_per_Game", current_episode_steps, global_episode)
                # writer.add_scalar("3_Hyperparameters/Epsilon", agent1.epsilon, global_episode)

                pbar.set_postfix({
                    "승/무/패": f"{agent1_wins}/{draws}/{agent1_losses}",
                    "유효승률": f"{win_rate:.1f}%",
                    "현재": f"{current_episode_steps}수",
                    "평균": f"{avg_steps}수",
                    "입실론": f"{agent1.epsilon:.3f}",
                    "메모리": f"{len(agent1.memory)}"
                })
                pbar.update(1)
                
                agent1.decay_epsilon()

            pbar.close()
            
            # --- 500판 종료 직후: 상대방 진화 및 탐험률 롤백 ---
            if phase_end < EPISODES: 
                if win_rate >= 55.0 and decisive_games >= 100:
                    agent1.save_model(f"khy_omok_levelup.pth")
                    agent2_self.model.load_state_dict(agent1.model.state_dict())
                    agent1.memory.clear()
                    update_msg = "상대방 진화 완료 (승률 55% 돌파)"
                else:
                    update_msg = "상대방 유지 (승률 부족으로 진화 보류)"
                    
                agent1.epsilon, agent1.epsilon_decay = 0.15, 0.9855
                print(f"[업데이트] {phase_end}판 종료: {update_msg} / [입실론 롤백: {agent1.epsilon:.3f}]\n")
                
            agent1.save_model(f"khy_omok_ep{phase_end}.pth")
        
    print(f"\n=== 총 {N}세대({N * EPISODES}판)의 대장정 완료 ===")
    # writer.close() # [추가] 모든 학습 완료 시 텐서보드 Writer 닫기
    env.close()
    
    plt.ioff()
    plt.show()
    
# ==========================================
# 4. 메인
# ==========================================
if __name__ == "__main__":
    # main()
    train_main()