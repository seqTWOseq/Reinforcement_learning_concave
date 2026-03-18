import numpy as np
import gymnasium as gym
from gymnasium import spaces
import tkinter as tk
import time

import numpy as np
from numba import njit
import random
from collections import deque
import torch
import torch.nn.functional as F
import torch.optim as optim

from khy_model import DualHeadResOmokCNN
from hjw_model import AlphaZeroNet
from hjw_model import MCTS
from hjw_model import GomokuGame
from jmnm_model import board_to_tensor as nami_board_to_tensor, load_nami_model


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
    """
    마우스 클릭 이벤트를 통해 사용자가 직접 착수하는 에이전트입니다.
    """
    def __init__(self, env, name="Human(👤)"):
        self.name = name
        self.env = env
        self.clicked_action = None
        self.current_state = None

    def select_action(self, state):
        """사용자가 화면을 클릭할 때까지 코드 실행을 일시 정지하고 대기합니다."""
        self.clicked_action = None
        self.current_state = state
        
        # 1. 마우스 왼쪽 클릭 이벤트 활성화
        self.env.canvas.bind("<Button-1>", self._click_handler)
        
        # 2. 클릭될 때까지 무한 대기 (화면은 멈추지 않도록 update 호출)
        while self.clicked_action is None:
            self.env.window.update()
            time.sleep(0.05) # CPU 과부하 방지
            
        # 3. 행동 결정 완료 시 이벤트 해제 (AI 턴에 클릭 방지)
        self.env.canvas.unbind("<Button-1>")
        
        return self.clicked_action

    def _click_handler(self, event):
        """마우스 클릭 시 좌표를 행동(Action) 인덱스로 변환합니다."""
        # 클릭한 픽셀 위치를 오목판 논리적 좌표(행, 열)로 변환
        c = round((event.x - self.env.margin) / self.env.cell_size)
        r = round((event.y - self.env.margin) / self.env.cell_size)
        
        # 보드 범위 내인지 확인
        if 0 <= r < self.env.board_size and 0 <= c < self.env.board_size:
            action = r * self.env.board_size + c
            
            # 유효한 빈칸(0)을 클릭했을 때만 값 업데이트 (반칙 클릭 무시)
            if self.current_state.flatten()[action] == 0:
                self.clicked_action = action

# ==========================================
# 김현용
# ==========================================
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
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.0005)

        self.is_training = True
        
        # 탐험 파라미터
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.999
        
        # 경험 재생 메모리
        self.memory = deque(maxlen=150000)
        self.batch_size = 1024
        self.gamma = 0.99

    # ====================
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

        # 탐험을 위한 디리클레 노이즈 주입 (Train 모드일 때만)
        if self.is_training:
            noise = np.random.dirichlet([0.3] * len(valid_moves))
            policy_probs[valid_moves] = 0.75 * policy_probs[valid_moves] + 0.25 * noise
            policy_probs /= np.sum(policy_probs[valid_moves])

        # MinMax 정규화
        p_min = policy_probs[valid_moves].min()
        p_max = policy_probs[valid_moves].max()

        if p_max > p_min:
            policy_scaled = (policy_probs - p_min) / (p_max - p_min)
        else:
            policy_scaled = policy_probs

        # Rollout Q-Value
        num_simulations = 400
        action_visits = np.zeros(total_grids)
        action_wins = np.zeros(total_grids)

        for _ in range(num_simulations):
            probs = policy_probs[valid_moves]
            probs /= np.sum(probs)
            sim_action = np.random.choice(valid_moves, p=probs)
            
            # Numba로 최적화된 20수 무작위 롤아웃 실행 (빠른 속도)
            reward = fast_rollout_fast(state, sim_action, max_depth=30)
            action_visits[sim_action] += 1
            action_wins[sim_action] += reward
        
        # 롤아웃 결과 평균 승률 산출
        sim_q_values = np.divide(action_wins, action_visits, out=np.zeros_like(action_wins), where=action_visits!=0)

        # 가치 일괄 평가
        num_valid = len(valid_moves)
        next_states_batch = np.zeros((num_valid, 1, board_size, board_size), dtype=np.float32)
        
        for i, move in enumerate(valid_moves):
            r, c = move // board_size, move % board_size
            next_state = state.copy()
            next_state[r, c] = 1 
            # 다음 턴(상대) 시점으로 보드판 정규화 (시점 반전)
            canonical_next_state = np.where(next_state != 0, 3 - next_state, 0)
            next_states_batch[i, 0] = canonical_next_state
            
        batch_tensor = torch.FloatTensor(next_states_batch).to(self.device)
        with torch.no_grad():
            _, next_values = self.model(batch_tensor)
            next_values = next_values.flatten().cpu().numpy()
            
        cnn_values_full = np.zeros(total_grids)
        # 내 입장에서의 승률로 부호 반전(-)
        cnn_values_full[valid_moves] = -1.0 * next_values

        # 내재적 보상
        intrinsic_rewards = np.zeros(total_grids)
        for move in valid_moves:
            intrinsic_rewards[move] = self.get_intrinsic_reward(state, move)

        w_policy  = 0.2  # 직관의 비중
        w_rollout = 0.3  # 난전/전술 검증의 비중
        w_value   = 0.3  # 대국적인 형세 판단의 비중
        w_int     = 0.2  # 당장의 포메이션(안전) 비중

        # 방문하지 않은 곳도 Value와 Intrinsic, Policy는 평가가 가능하므로 합산
        final_score = np.where(
            np.isin(np.arange(total_grids), valid_moves), 
            (w_policy * policy_scaled) + (w_rollout * sim_q_values) + (w_value * cnn_values_full) + (w_int * intrinsic_rewards), 
            -float('inf')
        )

        return np.argmax(final_score)
    
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
            if final_reward > 0: # 이긴 게임: 이전 턴으로 갈수록 가치 감소
                discounted_reward = discounted_reward * self.gamma - step_cost
            else: # 진 게임: 이전 턴으로 갈수록 가치 하락 (더 나쁘게 평가)
                discounted_reward = discounted_reward * self.gamma - step_cost
            
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
        
        # [핵심] 두 Loss를 합치되, 스케일에 따라 가중치(예: c=1.0)를 둘 수 있습니다.
        total_loss = (value_loss * 1.5) + policy_loss

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
# NamiAgent (정민 & 나미팀)
# ==========================================
class NamiAgent:
    """
    SL(지도학습) + PPO(강화학습) 으로 훈련된 NamiNet 기반 에이전트.
    - 3채널 입력 (내돌 / 상대돌 / 턴): 흑백 구분 명확
    - 긴급수 우선 처리 (5목 완성 / 상대 5목 차단)
    - Policy head argmax 착수
    """
    def __init__(self, model_path="nami.pth",
                 name="NamiAI(🐟)"):
        self.name = name
        self.model, self.device = load_nami_model(model_path)

    def _pattern_score(self, board, r, c, player):
        """
        r,c에 player 돌을 놓은 뒤 생기는 가장 강한 패턴 점수 반환.
        10=5목  4=열린4  3=사(닫힌4)  2=33/34  1=열린3
        """
        size = board.shape[0]
        directions = [(0,1),(1,0),(1,1),(-1,1)]
        open_threes = fours = open_fours = 0

        for dr, dc in directions:
            consecutive = 1
            open_ends = 0
            for step in (1, -1):
                nr, nc = r + dr*step, c + dc*step
                while 0 <= nr < size and 0 <= nc < size:
                    if board[nr, nc] == player:
                        consecutive += 1
                    elif board[nr, nc] == 0:
                        open_ends += 1
                        break
                    else:
                        break
                    nr += dr*step; nc += dc*step

            if consecutive >= 5:
                return 10                        # 5목 완성
            elif consecutive == 4 and open_ends == 2:
                open_fours += 1                  # 열린 4
            elif consecutive == 4 and open_ends == 1:
                fours += 1                       # 닫힌 4 (사)
            elif consecutive == 3 and open_ends == 2:
                open_threes += 1                 # 열린 3

        if open_fours >= 1:    return 4          # 열린4 → 다음 수 무조건 이김
        if fours >= 2:         return 3          # 44
        if fours >= 1 and open_threes >= 1: return 2   # 34
        if open_threes >= 2:   return 2          # 33
        if fours >= 1:         return 1          # 단순 사
        if open_threes >= 1:   return 0          # 단순 열린3 (긴급 아님)
        return -1

    def _urgent_move(self, state):
        """
        우선순위별 긴급수 반환. 없으면 -1.
        1순위 내 5목 → 2순위 상대 5목 차단 → 3순위 내 열린4 →
        4순위 상대 열린4/사 차단 → 5순위 내 33/34 → 6순위 상대 33/34 차단
        """
        size = state.shape[0]
        valid = np.where(state.flatten() == 0)[0]

        best   = [-1, -1, -1, -1, -1, -1]  # 각 우선순위별 후보 (인덱스)

        for move in valid:
            r, c = move // size, move % size

            # 내 착수 시뮬레이션
            my_board = state.copy(); my_board[r, c] = 1
            my_score = self._pattern_score(my_board, r, c, 1)

            # 상대 착수 시뮬레이션
            op_board = state.copy(); op_board[r, c] = 2
            op_score = self._pattern_score(op_board, r, c, 2)

            if my_score == 10 and best[0] == -1: best[0] = move  # 내 5목
            if op_score == 10 and best[1] == -1: best[1] = move  # 상대 5목 차단
            if my_score == 4  and best[2] == -1: best[2] = move  # 내 열린4
            if op_score >= 3  and best[3] == -1: best[3] = move  # 상대 열린4/44 차단
            if my_score == 2  and best[4] == -1: best[4] = move  # 내 33/34
            if op_score == 2  and best[5] == -1: best[5] = move  # 상대 33/34 차단

        for candidate in best:
            if candidate != -1:
                return candidate
        return -1

    def select_action(self, state):
        """
        state: (15,15) numpy, 0=빈칸 1=내돌 2=상대돌
        반환: 착수 위치 (int, 0~224)
        """
        # 1. 긴급수 (5목 완성 / 차단)
        urgent = self._urgent_move(state)
        if urgent != -1:
            return urgent

        # 2. 모델 추론
        state_t = nami_board_to_tensor(state)          # (3,15,15)
        t = torch.tensor(state_t).unsqueeze(0).to(self.device)  # (1,3,15,15)
        with torch.no_grad():
            pol_logits, _ = self.model(t)
        pol_logits = pol_logits.squeeze(0)             # (225,)

        # 빈 칸만 착수 가능
        valid_mask = torch.tensor(
            (state.flatten() == 0), dtype=torch.bool, device=self.device)
        pol_logits[~valid_mask] = -1e9

        return int(torch.argmax(pol_logits).item())


# ==========================================
# 홍정우AlphaZeroAgent
# ==========================================
BOARD_SIZE = 15
ACTION_SIZE = BOARD_SIZE * BOARD_SIZE
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
# 3. 대결 실행 루프 (Arena)
# ==========================================
def main():
    env = OmokEnvGUI(render_mode="human")
    # agent1 = HumanAgent(env, name="Human_Black(●)")
    # 김현용
    khy_model = DualHeadResOmokCNN()
    agent2 = KhyAgent(khy_model)
    # agent1.load_model("khy_omok_gen2_final.pth")
    agent2.eval_mode()

    # ================================================================
    # NamiAI와 대결하고 싶으면:
    #   1. nami.pth 가중치 파일을 이 폴더에 받아주세요
    #   2. 아래 주석(agent1 = NamiAgent(...))을 풀고
    #      그 아래 agent1 = AlphaZeroAgent(...) 줄을 주석 처리하세요
    #   3. while 루프 안의 주석도 풀고 아래 줄을 주석 처리하세요
    #      # action = agent1.select_action(state)  ← 이 줄 주석 해제
    #      action = agent1.select_action(state, player_id=1)  ← 이 줄 주석 처리
    # ================================================================
    # agent1 = NamiAgent(name="NamiAI(🐟)")
    agent1 = AlphaZeroAgent(name="AlphaZero(🤖)")

    state, info = env.reset()
    env.render()
    terminated = False

    print(f"=== ⚔️ {agent1.name} vs {agent2.name} 대결 시작 ===")

    while not terminated:
        # 턴에 따른 상태 반전 논리 (상대는 항상 자신이 흑돌인 것처럼 착각하게 만듦)
        if info["current_player"] == 1:
            # NamiAgent 사용 시 아래 주석을 풀고 그 아래 줄을 주석 처리하세요
            # action = agent1.select_action(state)
            action = agent1.select_action(state, player_id=1)
        else:
            inverted_state = np.where(state == 1, 2, np.where(state == 2, 1, 0))
            action = agent2.select_action(inverted_state)
            # action = agent2.select_action(state, player_id=2)
            
        state, reward, terminated, _, info = env.step(action)
        env.render()
        time.sleep(1) # 시각적 확인을 위한 지연

    # 결과 판정
    print("\n=== 🏁 대결 종료 ===")
    winner = info.get("winner")
    if winner == 1: print(f"🎉 {agent1.name} 승리!")
    elif winner == 2: print(f"🎉 {agent2.name} 승리! (중앙 선호 전략)")
    else: print("🤝 무승부!")
        
    time.sleep(3)
    env.close()

if __name__ == "__main__":
    main()