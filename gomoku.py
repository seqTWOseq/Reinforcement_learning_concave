import gymnasium as gym
from gymnasium import spaces
import tkinter as tk

import numpy as np
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
        self.memory = deque(maxlen=50000)
        self.batch_size = 64
        self.gamma = 0.99

    # 행동 선택 로직
    def select_action(self, state):
        board_size = state.shape[0]
        total_grids = board_size * board_size # 하드코딩 제거 (동적 할당)
        
        raw_valid_moves = np.where(state.flatten() == 0)[0]
        if len(raw_valid_moves) == 0:
            return 0
        
        occupied = np.argwhere(state != 0)
        if len(occupied) == 0:
            return (board_size // 2) * board_size + (board_size // 2) # 천원 착수
        
        # 인접 빈칸 탐색
        sensible_moves = set()
        for r, c in occupied:
            for dr in range(-2, 3):
                for dc in range(-2, 3):
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < board_size and 0 <= nc < board_size and state[nr, nc] == 0:
                        sensible_moves.add(nr * board_size + nc)
        valid_moves = np.array(list(sensible_moves))
            
        # 위급 수 우선 확인
        urgent_move = self._find_urgent_move(state, valid_moves, player=1)
        if urgent_move is not None:
            return urgent_move

        # CNN 가치 및 정책 평가
        state_tensor = torch.FloatTensor(state).unsqueeze(0).unsqueeze(0).to(self.device)
        self.model.eval()
        with torch.no_grad():
            # 듀얼 헤드의 두 가지 출력값을 분리해서 받음
            policy_logits, value = self.model(state_tensor)
            policy_logits = policy_logits.squeeze()
        self.model.train()

        # 불가능 수 마스킹
        valid_mask = torch.ones(total_grids, dtype=torch.bool).to(self.device)
        valid_mask[valid_moves] = False
        policy_logits[valid_mask] = -float('inf')

        # Softmax를 적용해 모델이 제안하는 '행동 확률 분포(Policy)' 획득
        policy_probs = F.softmax(policy_logits, dim=0).cpu().numpy()

        # MCTS 시뮬레이션
        num_simulations = 50
        action_visits = np.zeros(total_grids)
        action_wins = np.zeros(total_grids)

        for _ in range(num_simulations):
            if np.random.rand() <= self.epsilon:
                sim_action = np.random.choice(valid_moves)
            else:
                # Value가 아닌 Policy(신경망의 직관)를 기반으로 후보 수 좁히기
                probs = policy_probs[valid_moves]
                probs = probs / np.sum(probs) # 확률 정규화
                sim_action = np.random.choice(valid_moves, p=probs)
            
            reward = self._fast_rollout(state, sim_action, max_depth=5)
            action_visits[sim_action] += 1
            action_wins[sim_action] += reward
        
        sim_q_values = np.divide(action_wins, action_visits, out=np.zeros_like(action_wins), where=action_visits!=0)
        
        # Policy(직관)와 Rollout Value(수읽기 승률)를 결합하여 최종 판단
        alpha = 0.5
        final_score = np.where(action_visits > 0, (alpha * policy_probs) + ((1 - alpha) * sim_q_values), policy_probs)
        final_score[~np.isin(np.arange(total_grids), valid_moves)] = -float('inf')

        return np.argmax(final_score)
    
    # 시뮬레이션 엔진
    def _fast_rollout(self, state, action, max_depth=5):
        board_size = state.shape[0]
        sim_state = state.copy()
        r, c = action // board_size, action % board_size
        sim_state[r, c] = 1 
        
        if self._check_pattern(sim_state, r, c, player=1, target=5):
            return 1.0 
            
        current_player = 2
        for _ in range(max_depth):
            valid_moves = np.where(sim_state.flatten() == 0)[0]
            if len(valid_moves) == 0:
                break 
                
            # 배열 복사 없이 플레이어 식별자만 전달하여 속도 최적화
            urgent_move = self._find_urgent_move(sim_state, valid_moves, player=current_player)
            sim_action = urgent_move if urgent_move is not None else np.random.choice(valid_moves)
                
            sr, sc = sim_action // board_size, sim_action % board_size
            sim_state[sr, sc] = current_player
            
            if self._check_pattern(sim_state, sr, sc, current_player, target=5):
                return 1.0 if current_player == 1 else -1.0 
                
            current_player = 3 - current_player
            
        return 0.0
    
    # 내재적 보상 (공격 포메이션 평가)
    def get_intrinsic_reward(self, state, action):
        board_size = state.shape[0]
        r, c = action // board_size, action % board_size
        sim_state = state.copy()
        sim_state[r, c] = 1
        
        reward = 0.0
        directions = [(0, 1), (1, 0), (1, 1), (-1, 1)]
        pattern_counts = {'open_3': 0, 'four': 0}
        
        for dr, dc in directions:
            consecutive = 1
            open_ends = 0
            
            for step in (1, -1):
                nr, nc = r + dr * step, c + dc * step
                while 0 <= nr < board_size and 0 <= nc < board_size:
                    if sim_state[nr, nc] == 1:
                        consecutive += 1
                    elif sim_state[nr, nc] == 0:
                        open_ends += 1
                        break 
                    else:
                        break 
                    nr += dr * step
                    nc += dc * step
            
            if consecutive >= 5:
                reward += 1.0  
            elif consecutive == 4 and open_ends >= 1:
                reward += 0.5  
                pattern_counts['four'] += 1
            elif consecutive == 3 and open_ends == 2:
                reward += 0.2  
                pattern_counts['open_3'] += 1
        
        if pattern_counts['four'] >= 2 or (pattern_counts['four'] >= 1 and pattern_counts['open_3'] >= 1) or pattern_counts['open_3'] >= 2:
            reward += 0.8
            
        return reward
    
    # 무조건 해야 되는 행동 하드코딩
    def _find_urgent_move(self, state, valid_moves, player):
        board_size = state.shape[0]
        opponent = 3 - player
        best_move = None
        best_priority = 5 # 낮을수록 높음

        # 단일 루프로 탐색 최적화
        for move in valid_moves:
            r, c = move // board_size, move % board_size
            
            # 1순위: 승리 확정
            if self._check_pattern(state, r, c, player=player, target=5):
                return move
            
            # 2순위: 상대 승리 방어
            if best_priority > 2 and self._check_pattern(state, r, c, player=opponent, target=5):
                best_move = move
                best_priority = 2
                continue
            
            # 3순위: 상대 양수겸장 차단
            if best_priority > 3:
                threat_count = 0
                if self._check_pattern(state, r, c, player=opponent, target=4): threat_count += 1
                if self._check_pattern(state, r, c, player=opponent, target=3, open_ends_req=2): threat_count += 1
                if threat_count >= 2:
                    best_move = move
                    best_priority = 3
                    continue
            
            # 4순위: 상대 열린 4목 방어
            if best_priority > 4 and self._check_pattern(state, r, c, player=opponent, target=4, open_ends_req=2):
                best_move = move
                best_priority = 4

        return best_move
    
    # 패턴 파악
    def _check_pattern(self, state, r, c, player, target, open_ends_req=0):
        directions = [(0, 1), (1, 0), (1, 1), (-1, 1)]
        board_size = state.shape[0]

        for dr, dc in directions:
            consecutive = 1
            open_ends = 0

            for step in (1, -1):
                nr, nc = r + dr * step, c + dc * step
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
    
    # 기억 장치
    def memorize_episode(self, episode_memory, final_reward):
        discounted_reward = final_reward 
        for state, action, step_reward in reversed(episode_memory):
            total_reward = step_reward + discounted_reward
            self.memory.append((state, action, total_reward))
            discounted_reward *= self.gamma
    
    # 복습 엔진
    def replay_experience(self):
        if len(self.memory) < self.batch_size:
            return
        
        minibatch = random.sample(self.memory, self.batch_size)
        states, actions, targets = zip(*minibatch)

        # 채널 차원(unsqueeze(1)) 추가로 Conv2d 에러 방지
        states_tensor = torch.FloatTensor(np.array(states)).unsqueeze(1).to(self.device)
        actions_tensor = torch.LongTensor(actions).to(self.device) # Policy 정답: (Batch,) 유지
        targets_tensor = torch.FloatTensor(targets).unsqueeze(1).to(self.device) # Value 정답: (Batch, 1)로 변환

        # 듀얼 헤드 추론
        policy_logits, values = self.model(states_tensor)
        
        # Value Loss (MSE): 현재 판세의 승률 예측 오차
        value_loss = F.mse_loss(values, targets_tensor)

        # Policy Loss (Cross-Entropy): 내가 실제로 둔 좋은 수를 모델도 높은 확률로 추천했는가?
        policy_loss = F.cross_entropy(policy_logits, actions_tensor)

        # Total Loss: 두 오차의 합산
        total_loss = value_loss + policy_loss

        # 역전파 및 가중치 업데이트
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
    
    model1 = DualHeadResOmokCNN()  
    agent1 = KhyAgent(model1)
    print(f"[Device 확인] {agent1.device}")
    agent1.train_mode()
    
    model2 = DualHeadResOmokCNN()
    agent2_self = KhyAgent(model2)
    agent2_self.eval_mode()
    
    agent_heur = HeuristicAgent(name="Heuristic_White")
    
    N = 10
    EPISODES = 10000 
    
    for gen in range(1, N + 1):
        print(f"\n{'='*40}\n[Generation {gen}/{N}] 제 {gen}세대\n{'='*40}")
        
        # 1막 초기화
        agent1.epsilon, agent1.epsilon_decay = 0.01, 0.998
        agent1_wins, total_phase_steps = 0, 0
        pbar = None
        
        for episode in range(1, EPISODES + 1):
            # --- 막(Phase) 전환 논리 ---
            if episode == 1:
                pbar = tqdm(total=2000, desc=f"[Gen {gen}] 1막 VS 휴", position=0, leave=True)
            elif episode == 2001:
                pbar.close()
                agent1_wins, total_phase_steps = 0, 0
                agent1.epsilon, agent1.epsilon_decay = 0.2, 0.999
                pbar = tqdm(total=6000, desc=f"[Gen {gen}] 2막 VS 셀프", position=0, leave=True)
            elif episode == 8001:
                pbar.close()
                agent1_wins, total_phase_steps = 0, 0
                agent1.epsilon, agent1.epsilon_decay = 0.0, 1.0 
                pbar = tqdm(total=2000, desc=f"[Gen {gen}] 3막 VS 휴", position=0, leave=True)
                
            state, info = env.reset()
            terminated = False
            memory_b, memory_w = [], []
            current_episode_steps = 0 
            
            # --- 단일 에피소드(게임) 진행 ---
            while not terminated:
                current_player = info["current_player"]
                
                if current_player == 1:
                    action = agent1.select_action(state)
                    step_reward = agent1.get_intrinsic_reward(state, action)
                    memory_b.append((state.copy(), action, step_reward))
                    next_state, reward, terminated, _, info = env.step(action)
                    state = next_state
                else:
                    # 속도 최적화: 수식을 이용한 상태 반전 (1->2, 2->1)
                    inverted_state = np.where(state != 0, 3 - state, 0)
                    
                    if episode <= 2000 or episode >= 8001:
                        action = agent_heur.select_action(inverted_state) 
                    else:
                        action = agent2_self.select_action(inverted_state) 
                        
                    step_reward = agent1.get_intrinsic_reward(inverted_state, action)
                    memory_w.append((inverted_state.copy(), action, step_reward))
                    next_state, reward, terminated, _, info = env.step(action)
                    state = next_state
                    
                current_episode_steps += 1 
            
            # 논리 오류 수정: 총 스텝 수는 게임이 끝난 후 한 번만 더함
            total_phase_steps += current_episode_steps

            # --- 게임 종료 후: 승패 기록 및 복습 ---
            winner = info.get("winner")
            if winner == 1:
                agent1_wins += 1
                agent1.memorize_episode(memory_b, 1.0)   
                agent1.memorize_episode(memory_w, -1.0)  
            elif winner == 2:
                agent1.memorize_episode(memory_b, -1.0)  
                agent1.memorize_episode(memory_w, 1.0)   
            else: # 무승부
                agent1.memorize_episode(memory_b, -0.5)
                agent1.memorize_episode(memory_w, -0.5)
                
            for _ in range(4):
                agent1.replay_experience()

            # --- 통계 계산 및 진행바 업데이트 ---
            current_ep = episode if episode <= 2000 else (episode - 2000 if episode <= 8000 else episode - 8000)
            win_rate = (agent1_wins / current_ep) * 100
            avg_steps = total_phase_steps // current_ep

            pbar.set_postfix({
                "승률": f"{agent1_wins}/{current_ep} ({win_rate:.2f}%)",
                "평균": f"{avg_steps}수",
                "입실론": f"{agent1.epsilon:.3f}",
                "메모리": f"{len(agent1.memory)}"
            })
            pbar.update(1)
            
            if episode <= 8000:
                agent1.decay_epsilon()

            # --- 체크포인트 저장 및 셀프 플레이 갱신 ---
            if episode % 1000 == 0:
                agent1.save_model(f"khy_omok_gen{gen}_ep{episode}.pth") # 세대 변수 추가
                if 2000 <= episode < 8000:
                    agent1.epsilon = 0.2  
                    agent2_self.model.load_state_dict(agent1.model.state_dict())
                
        if pbar is not None:
            pbar.close()
        
        # 덮어쓰기 방지: 세대별 최종 모델 분리 저장
        agent1.save_model(f"khy_omok_gen{gen}_final.pth")
        
    print(f"\n=== 총 {N}세대({N * EPISODES}판)의 대장정 완료 ===")
    env.close()
    
# ==========================================
# 4. 메인
# ==========================================
if __name__ == "__main__":
    # main()
    train_main()