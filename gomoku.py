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
import time
from tqdm import tqdm

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
    def __init__(self, name="Heuristic_AI"):
        self.name = name

    def select_action(self, state):
        board_size = state.shape[0]
        valid_moves = np.where(state.flatten() == 0)[0]
        if len(valid_moves) == 0: return 0

        # 보드가 비어있다면 중앙(7, 7)에 착수
        if np.sum(state != 0) == 0:
            return (board_size // 2) * board_size + (board_size // 2)

        best_score = -float('inf')
        best_actions = []

        # 연산 시간을 줄이기 위해 기존 돌 주변 반경 2칸 이내의 빈칸만 탐색 후보로 선정
        candidates = self._get_candidate_moves(state, board_size)
        if not candidates:
            candidates = valid_moves

        for action in candidates:
            r, c = action // board_size, action % board_size
            
            # 내가 착수했을 때의 공격 점수 (Player 1 기준)
            offense_score = self._evaluate_position(state, r, c, player=1)
            # 상대가 착수했을 때의 방어 점수 (Player 2 기준)
            defense_score = self._evaluate_position(state, r, c, player=2)
            
            # 방어에 약간의 가중치를 더 주어 안정적으로 플레이 (1.1배)
            total_score = offense_score + (defense_score * 1.1)

            if total_score > best_score:
                best_score = total_score
                best_actions = [action]
            elif total_score == best_score:
                best_actions.append(action) # 동점일 경우 후보에 추가

        # 동점인 행동 중 무작위 선택하여 패턴 고착화 방지
        return np.random.choice(best_actions)

    def _get_candidate_moves(self, state, board_size):
        candidates = set()
        occupied = np.argwhere(state != 0)
        for r, c in occupied:
            for dr in range(-4, 5):
                for dc in range(-4, 5):
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

            # 양방향 탐색
            for step in (1, -1):
                r, c = row + dr * step, col + dc * step
                while 0 <= r < board_size and 0 <= c < board_size:
                    if state[r, c] == player:
                        consecutive += 1
                    elif state[r, c] == 0:
                        open_ends += 1
                        break # 빈칸을 만나면 열린 끝으로 간주하고 중단
                    else:
                        break # 상대 돌을 만나면 막힌 것으로 간주하고 중단
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
def fast_rollout_fast(state, action, max_depth, max_moves=800):
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
    depth_penalty_weight = 0.05 

    # [최적화 1] 매 턴 np.where를 호출하지 않고, 처음에 한 번만 빈칸 목록을 추출
    valid_moves = np.where(sim_state.flatten() == 0)[0]
    num_valid = len(valid_moves)

    for depth in range(max_depth):
        if current_stones >= max_moves or num_valid == 0:
            return -0.8 if current_stones >= max_moves else 0.0
        
        # [최적화 3] O(1) 속도로 빈칸 뽑고 삭제하기 (배열 새로고침 방지)
        idx = np.random.randint(num_valid)
        sim_action = valid_moves[idx]
        
        # 뽑은 빈칸을 배열 맨 뒤의 값과 바꾸고, 길이를 1 줄임 (초고속 삭제 기법)
        valid_moves[idx] = valid_moves[num_valid - 1]
        num_valid -= 1
            
        sr = sim_action // board_size
        sc = sim_action % board_size
        sim_state[sr, sc] = current_player
        current_stones += 1 
        
        # 승패 확인 (이 연산 하나만 남김)
        if check_pattern_fast(sim_state, sr, sc, current_player, 5, 0):
            penalty = depth * depth_penalty_weight
            if current_player == 1:
                return 1.0 - penalty 
            else:
                return -1.0 + penalty 
            
        current_player = 3 - current_player
        
    return 0.0
    
class KhyAgent:
    def __init__(self, model):
        self.name = "Khy_AI"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        
        # 탐험 파라미터
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.999
        
        # 경험 재생 메모리
        self.memory = deque(maxlen=150000)
        self.batch_size = 1024
        self.gamma = 0.99

    # 행동 선택 로직
    def select_action(self, state, move_count=0):
        board_size = state.shape[0]
        total_grids = board_size * board_size 
        
        raw_valid_moves = np.where(state.flatten() == 0)[0]
        if len(raw_valid_moves) == 0:
            return 0
        
        occupied = np.argwhere(state != 0)
        if len(occupied) == 0:
            return (board_size // 2) * board_size + (board_size // 2) 
        
        # 인접 빈칸 탐색
        sensible_moves = set()
        for r, c in occupied:
            for dr in range(-2, 3):
                for dc in range(-2, 3):
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < board_size and 0 <= nc < board_size and state[nr, nc] == 0:
                        sensible_moves.add(nr * board_size + nc)
        valid_moves = np.array(list(sensible_moves))
            
        # 위급 수 우선 확인 (Numba 함수 호출)
        urgent_move = find_urgent_move_fast(state, valid_moves, player=1)
        if urgent_move != -1: 
            return urgent_move

        # CNN 가치 및 정책 평가
        state_tensor = torch.FloatTensor(state).unsqueeze(0).unsqueeze(0).to(self.device)
        self.model.eval()
        with torch.no_grad():
            policy_logits, value = self.model(state_tensor)
            policy_logits = policy_logits.squeeze()
        self.model.train()

        # 불가능 수 마스킹
        valid_mask = torch.ones(total_grids, dtype=torch.bool).to(self.device)
        valid_mask[valid_moves] = False
        policy_logits[valid_mask] = -float('inf')

        # Softmax를 적용해 행동 확률 분포 획득
        policy_probs = F.softmax(policy_logits, dim=0).cpu().numpy()

        # [핵심 수정 1] Epsilon 무작위 탐험을 버리고 알파제로식 디리클레 노이즈 주입
        if self.model.training:
            noise = np.random.dirichlet([0.3] * len(valid_moves))
            # 기존 신경망의 추천 확률 75%에 상상력(노이즈) 25%를 섞음
            policy_probs[valid_moves] = 0.75 * policy_probs[valid_moves] + 0.25 * noise
            policy_probs = policy_probs / np.sum(policy_probs[valid_moves]) # 확률 정규화

        # MCTS 시뮬레이션
        num_simulations = 800 # [핵심 수정 2] 시뮬레이션 횟수 상향 (수읽기 파워 증폭)
        action_visits = np.zeros(total_grids)
        action_wins = np.zeros(total_grids)

        for _ in range(num_simulations):
            # [핵심 수정 3] Epsilon 분기문 완전히 삭제 (오직 신경망의 지능적 확률로만 선택)
            probs = policy_probs[valid_moves]
            probs = probs / np.sum(probs)
            sim_action = np.random.choice(valid_moves, p=probs)
            
            # [핵심 수정 4] 깊이 연장 (최대 20수 앞까지 내다봄)
            reward = fast_rollout_fast(state, sim_action, max_depth=20)
            action_visits[sim_action] += 1
            action_wins[sim_action] += reward
        
        sim_q_values = np.divide(action_wins, action_visits, out=np.zeros_like(action_wins), where=action_visits!=0)
        
        # 내재적 보상(공격 포메이션)을 계산하여 합산할 배열 준비
        intrinsic_rewards = np.zeros(total_grids)
        for move in valid_moves:
            # 행동마다 유발되는 공격 + 수비 포메이션 가치 평가
            intrinsic_rewards[move] = self.get_intrinsic_reward(state, move)
        
        # Policy와 Rollout Value 결합
        alpha = 0.4  # 신경망 정책 가중치
        beta = 0.4   # MCTS 롤아웃 가중치
        weight_int = 0.2 # 내재적 보상(공격성) 가중치

        final_score = np.where(
            action_visits > 0, 
            (alpha * policy_probs) + (beta * sim_q_values) + (weight_int * intrinsic_rewards), 
            policy_probs + (weight_int * intrinsic_rewards) # 롤아웃 안 된 곳도 내재적 보상은 평가
        )
        final_score[~np.isin(np.arange(total_grids), valid_moves)] = -float('inf')

        if self.model.training:
            # 웜 스타트 직후이므로, 탐험을 충분히 할 수 있게 초반 20수까지 온도를 1.0으로 둡니다.
            tau = 1.0 if move_count < 20 else 0.1
            
            valid_scores = final_score[valid_moves]
            
            # 지수 함수 오버플로우를 막기 위해 최댓값을 빼고 Softmax 계산
            valid_scores = valid_scores - np.max(valid_scores) 
            exp_scores = np.exp(valid_scores / tau)
            action_probs = exp_scores / np.sum(exp_scores)
            
            # 온도에 의해 계산된 확률 분포(action_probs)를 바탕으로 행동 무작위(가중치) 선택
            return np.random.choice(valid_moves, p=action_probs)
        else:
            # 실전 평가 모드(과거의 나)일 때는 얄짤없이 무조건 가장 좋은 수만 둡니다.
            return np.argmax(final_score)
    
    # 내재적 보상 (1:1 공수 완벽 밸런스 평가)
    def get_intrinsic_reward(self, state, action):
        board_size = state.shape[0]
        r, c = action // board_size, action % board_size
        
        # 특정 플레이어(나 또는 상대) 입장에서 해당 위치의 패턴 가치를 계산하는 내부 함수
        def evaluate_for_player(target_player):
            sim_state = state.copy()
            sim_state[r, c] = target_player # 평가하려는 플레이어의 돌을 놓아봄
            
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
                
                # 순수 포메이션 가치 평가 (돌이 놓였을 때의 파괴력)
                if consecutive >= 5:
                    score += 2.0  
                elif consecutive == 4 and open_ends >= 1:
                    score += 1.0  
                    pattern_counts['four'] += 1
                elif consecutive == 3 and open_ends == 2:
                    score += 0.4  
                    pattern_counts['open_3'] += 1
            
            # 양수겸장(3-3, 4-3 등) 평가
            if pattern_counts['four'] >= 2 or (pattern_counts['four'] >= 1 and pattern_counts['open_3'] >= 1) or pattern_counts['open_3'] >= 2:
                score += 2.0
                
            return score

        # 1. 공격 가치: 내가(1) 두었을 때 얻는 포메이션 점수
        attack_value = evaluate_for_player(1) 
        
        # 2. 수비 가치: 상대(2)가 두었다면 얻었을 포메이션 점수를 빼앗음
        defense_value = evaluate_for_player(2) 
        
        # 공격과 수비의 가치를 1.0 대 1.0으로 동등하게 합산
        total_reward = (attack_value * 1.1) + defense_value
        
        return total_reward
    
    # 기억 장치 (데이터 증강 적용)
    def memorize_episode(self, episode_memory, final_reward):
        discounted_reward = final_reward
        step_cost = 0.005 # 턴이 길어질수록 깎이는 '시간 지연 페널티' (유지)
        intrinsic_weight = 0.05 # 내재적 보상의 비중을 5%로 대폭 축소 (파밍 방지)
        
        for state, action, step_reward in reversed(episode_memory):
            # 최종 승패 보상에 '매우 작게 축소된' 모양 만들기 보상을 더함
            total_reward = discounted_reward + (step_reward * intrinsic_weight)
            
            board_size = state.shape[0]
            action_matrix = np.zeros((board_size, board_size), dtype=np.int8)
            action_matrix[action // board_size, action % board_size] = 1
            
            # (8방향 대칭 데이터 생성 코드는 기존과 완벽히 동일하므로 그대로 유지)
            for i in range(4):
                rot_state = np.rot90(state, k=i)
                rot_action_mat = np.rot90(action_matrix, k=i)
                rot_action = np.argmax(rot_action_mat) 
                self.memory.append((rot_state.copy(), rot_action, total_reward))
                
                flip_state = np.fliplr(rot_state)
                flip_action_mat = np.fliplr(rot_action_mat)
                flip_action = np.argmax(flip_action_mat)
                self.memory.append((flip_state.copy(), flip_action, total_reward))
                
            # 핵심 논리: 다음 스텝(더 이전 턴)으로 갈수록 시간 감가 적용 및 페널티 부여
            discounted_reward = discounted_reward * self.gamma - step_cost
            
            # 보상이 -1.0 밑으로 무한히 떨어져서 학습이 붕괴되는 것을 방어
            discounted_reward = max(discounted_reward, -1.0)
    
    # 복습 엔진
    def replay_experience(self):
        if len(self.memory) < self.batch_size:
            return
        
        minibatch = random.sample(self.memory, self.batch_size)
        states, actions, targets = zip(*minibatch)

        states_tensor = torch.FloatTensor(np.array(states)).unsqueeze(1).to(self.device)
        actions_tensor = torch.LongTensor(actions).to(self.device) 
        targets_tensor = torch.FloatTensor(targets).unsqueeze(1).to(self.device) 

        policy_logits, values = self.model(states_tensor)
        
        value_loss = F.mse_loss(values, targets_tensor)
        policy_loss = F.cross_entropy(policy_logits, actions_tensor)
        total_loss = value_loss + policy_loss

        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
    
    # 유틸
    def train_mode(self):
        self.model.train()
    
    def eval_mode(self):
        self.model.eval()
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
    agent1.load_model("khy_omok_final.pth")
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

def train_main():
    env = OmokEnvGUI(render_mode=None)
    
    # 학습할 메인 에이전트
    model1 = DualHeadResOmokCNN()  
    agent1 = KhyAgent(model1)
    agent1.load_model("khy_omok_final.pth")
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
    
    for gen in range(1, N + 1):
        print(f"\n{'='*40}\n[Generation {gen}/{N}] 제 {gen}세대\n{'='*40}")
        
        # 세대 시작 시 탐험률 초기화 및 진행률 표시줄 생성
        agent1.epsilon, agent1.epsilon_decay = 0.0, 1.0
        
        # 10,000판을 500판 단위로 쪼개어 루프 실행 (총 20개의 구간)
        for phase_start in range(1, EPISODES + 1, UPDATE_INTERVAL):
            phase_end = phase_start + UPDATE_INTERVAL - 1
            
            # 핵심: 200판마다 통계(승리 횟수, 턴 수)를 0으로 초기화하여 '현재 상대'에 대한 진짜 실력만 측정
            agent1_wins, total_steps = 0, 0
            
            # 새로운 200판 단위의 진행바 생성
            agent1_wins, draws, agent1_losses, total_steps = 0, 0, 0, 0
            pbar = tqdm(total=UPDATE_INTERVAL, desc=f"[Gen {gen}] {phase_start}~{phase_end}판", position=0, leave=True)
            
            for episode in range(phase_start, phase_end + 1):
                state, info = env.reset()
                terminated = False
                memory_b, memory_w = [], []
                current_episode_steps = 0 
                
                agent1_color = 1 if np.random.rand() < 0.5 else 2

                # --- 단일 에피소드(게임) 진행 ---
                while not terminated:
                    current_player = info["current_player"]
                    
                    # 50수 제한 강제 패배 로직
                    if current_episode_steps >= 100:
                        terminated = True
                        info["winner"] = 0 
                        break 
                    
                    # 핵심: 현재 플레이어의 시점에 맞게 보드판 정규화 (내 돌은 항상 1, 상대는 2)
                    if current_player == 2:
                        canonical_state = np.where(state != 0, 3 - state, 0)
                    else:
                        canonical_state = state.copy()
                        
                    is_opening = current_episode_steps < 2

                    # 현재 턴이 메인 에이전트인지, 과거의 나(상대방)인지 판별
                    is_agent1_turn = (current_player == agent1_color)
                    active_agent = agent1 if is_agent1_turn else agent2_self
                    
                    # 항상 '내 돌이 1'인 정규화된 보드판(canonical_state)으로 판단
                    if is_opening:
                        valid_moves = np.where(canonical_state.flatten() == 0)[0]
                        action = np.random.choice(valid_moves)
                    else:
                        action = active_agent.select_action(canonical_state, move_count=current_episode_steps)
                        
                    # 보상 계산 역시 정규화된 보드판 기준 (agent1의 로직 공유)
                    step_reward = agent1.get_intrinsic_reward(canonical_state, action)
                    
                    # 메모리 저장: CNN이 헷갈리지 않게 무조건 '정규화된 상태'를 저장
                    # 현재 흑 차례면 memory_b에, 백 차례면 memory_w에 분류
                    if current_player == 1:
                        memory_b.append((canonical_state, action, step_reward))
                    else:
                        memory_w.append((canonical_state, action, step_reward))

                    # 실제 게임 환경은 원래 action대로 진행
                    next_state, reward, terminated, _, info = env.step(action)
                    state = next_state
                    current_episode_steps += 1

                total_steps += current_episode_steps

                # --- 게임 종료 후: 승패 기록 및 복습 ---
                winner = info.get("winner")
                if winner == agent1_color:
                    agent1_wins += 1
                elif winner == 0: # 무승부
                    draws += 1
                else: # 패배
                    agent1_losses += 1

                # 데이터 증강 및 양방향 학습 로직
                if winner == agent1_color:
                    final_reward = 1.0  # 내가 이김
                elif winner == 0:
                    final_reward = -1.0 # 무승부 (난전 강제를 위한 공멸 페널티 적용 시)
                else:
                    final_reward = -1.0 # 내가 짐

                # 오직 내가(agent1) 플레이했던 색깔의 기보만 저장
                if agent1_color == 1:
                    agent1.memorize_episode(memory_b, final_reward)
                else:
                    agent1.memorize_episode(memory_w, final_reward)
                    
                for _ in range(8):
                    agent1.replay_experience()

                # --- 통계 계산 (1~10000판 누적이 아닌, 현재 200판 구간 내의 통계) ---
                current_phase_ep = episode - phase_start + 1
                decisive_games = agent1_wins + agent1_losses # 승패가 갈린 게임 수
                
                if decisive_games > 0:
                    # 무승부를 제외하고, 순수하게 이기거나 진 게임 중 이긴 비율
                    win_rate = (agent1_wins / decisive_games) * 100 
                else:
                    win_rate = 0.0
                    
                avg_steps = total_steps // current_phase_ep

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

            # 500판 구간 종료 시 진행바 닫기
            pbar.close()
            
            # --- 200판 종료 직후: 상대방 진화 및 탐험률 롤백 ---
            if phase_end < EPISODES: # 마지막 판이 아닐 때만 업데이트 수행
                if win_rate >= 55.0 and decisive_games >= 100:
                    agent1.save_model(f"khy_omok_ep{phase_end}.pth")
                    agent2_self.model.load_state_dict(agent1.model.state_dict())
                    agent1.memory.clear()
                    update_msg = "상대방 진화 완료 (승률 55% 돌파)"
                else:
                    update_msg = "상대방 유지 (승률 부족으로 진화 보류)"
                    
                agent1.epsilon = 0.0
                agent1.epsilon_decay = 1.0
                print(f"[업데이트] {phase_end}판 종료: {update_msg} / [입실론 롤백: {agent1.epsilon:.3f}]\n")

        # 10,000판 (1세대) 종료 후 최종 모델 저장
        agent1.save_model(f"khy_omok_gen{gen}_final.pth")
        
    print(f"\n=== 총 {N}세대({N * EPISODES}판)의 대장정 완료 ===")
    env.close()
    
# ==========================================
# 4. 메인
# ==========================================
if __name__ == "__main__":
    # main()
    train_main()