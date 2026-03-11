import gymnasium as gym
from gymnasium import spaces
import tkinter as tk

import numpy as np
import random
from collections import deque
import torch
import torch.optim as optim
import time
from tqdm import tqdm

from khy_model import OmokCNN

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
        
        # 탐험률 설정
        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.9995
        
        # 경험 재생 메모리 (3단계에서 다룰 예정)
        self.memory = deque(maxlen=50000)
        self.batch_size = 64
        self.gamma = 0.99

    # 행동 선택 로직
    def select_action(self, state):
        raw_valid_moves = np.where(state.flatten() == 0)[0]
        if len(raw_valid_moves) == 0:
            return 0
        
        board_size = state.shape[0]

        # 반사 신경 (확실한 킬각/위기는 즉시 착수)
        occupied = np.argwhere(state != 0)
        if len(occupied) == 0:
            # 첫 수는 무조건 바둑판의 정중앙(천원)에 착수!
            return (board_size // 2) * board_size + (board_size // 2)
        else:
            # 기존 돌들 주변 반경 2칸 이내만 탐색 후보로 선정
            sensible_moves = set()
            for r, c in occupied:
                for dr in range(-2, 3):
                    for dc in range(-2, 3):
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < board_size and 0 <= nc < board_size and state[nr, nc] == 0:
                            sensible_moves.add(nr * board_size + nc)
            # 좁혀진 후보군을 valid_moves로 확정
            valid_moves = np.array(list(sensible_moves))
            
        urgent_move = self._find_urgent_move(state, valid_moves)
        if urgent_move is not None:
            return urgent_move

        # 탐험 (Epsilon)
        if np.random.rand() <= self.epsilon:
            return np.random.choice(valid_moves)
        
        # CNN을 통해 모든 칸의 가치(Q-Value) 초기 평가
        state_tensor = torch.FloatTensor(state).to(self.device)
        self.model.eval()
        with torch.no_grad():
            q_values = self.model(state_tensor).squeeze()
        self.model.train()

        # 불가능한 수 마스킹 (-무한대)
        mask = torch.ones(225, dtype=torch.bool).to(self.device)
        mask[valid_moves] = False
        q_values[mask] = -float('inf')

        # 가장 점수가 높은 K의 수
        K = min(3, len(valid_moves))
        top_k_actions = torch.topk(q_values, K).indices.cpu().numpy()

        best_action = top_k_actions[0]
        best_q_value = -float('inf')

        # Top-10 후보들을 각각 50번씩 가상 대국 돌려보기
        for action in top_k_actions:
            # 시뮬레이션을 통해 진짜 승률(Q값)을 계산
            simulated_q = self._simulate_rollouts(state, action, num_rollouts=5, max_depth=5)
            my_initial_q = q_values[action].item()
            final_score = (simulated_q * 0.5) + (my_initial_q * 0.5)
            
            # 가장 가치가 높은 수를 최종 선택
            if final_score > best_q_value:
                best_q_value = final_score
                best_action = action

        return best_action
    
    # 시뮬레이트
    def _simulate_rollouts(self, state, action, num_rollouts=5, max_depth=5):
        board_size = state.shape[0]
        total_score = 0.0
        
        for _ in range(num_rollouts):
            sim_state = state.copy()
            r, c = action // board_size, action % board_size
            sim_state[r, c] = 1 
            
            current_player = 2 
            terminated_by_win = False
            
            for _ in range(max_depth):
                valid_moves = np.where(sim_state.flatten() == 0)[0]
                if len(valid_moves) == 0:
                    break
                
                if current_player == 1:
                    eval_state = sim_state
                else:
                    eval_state = np.where(sim_state == 1, 2, np.where(sim_state == 2, 1, 0))
                
                # 강력해진 4단계 반사 신경으로 뻔한 수 방어
                urgent_move = self._find_urgent_move(eval_state, valid_moves)
                    
                if urgent_move is not None:
                    sim_action = urgent_move
                else:
                    # 무거운 파이썬 난수 탐색 대신, 초고속 C++ 기반 CNN 가이드 부활!
                    # (빨리 승패를 결정지어 10수 루프를 조기 종료시킵니다)
                    sim_tensor = torch.FloatTensor(eval_state).to(self.device)
                    self.model.eval()
                    with torch.no_grad():
                        q_values = self.model(sim_tensor).squeeze()
                    self.model.train()
                
                    valid_mask = torch.ones(225, dtype=torch.bool).to(self.device)
                    valid_mask[valid_moves] = False
                    q_values[valid_mask] = -float('inf')
                    
                    K = min(3, len(valid_moves))
                    top_k_indices = torch.topk(q_values, K).indices.cpu().numpy()
                    
                    probabilities = [0.5, 0.3, 0.2][:K]
                    probabilities /= np.sum(probabilities) 
                    
                    sim_action = np.random.choice(top_k_indices, p=probabilities)
        
                sr, sc = sim_action // board_size, sim_action % board_size
                sim_state[sr, sc] = current_player
                
                # 누군가 이기면 루프 즉시 폭파 (속도 향상의 핵심)
                if self._check_pattern(sim_state, sr, sc, current_player, target=5):
                    if current_player == 1:
                        total_score += 1.0   
                    else:
                        total_score -= 1.0   
                    terminated_by_win = True
                    break
                
                current_player = 3 - current_player
                
            # 10수가 끝난 뒤 최종 형세 판단
            if not terminated_by_win:
                if current_player == 1:
                    final_eval_state = sim_state
                    sign = 1.0 
                else:
                    final_eval_state = np.where(sim_state == 1, 2, np.where(sim_state == 2, 1, 0))
                    sign = -1.0 
                    
                sim_tensor = torch.FloatTensor(final_eval_state).to(self.device)
                self.model.eval()
                with torch.no_grad():
                    leaf_q_values = self.model(sim_tensor).squeeze()
                self.model.train()
                
                valid_mask = torch.ones(225, dtype=torch.bool).to(self.device)
                valid_mask[np.where(final_eval_state.flatten() == 0)[0]] = False
                leaf_q_values[valid_mask] = -float('inf')
                
                total_score += (sign * leaf_q_values.max().item())
        
        return total_score / num_rollouts
    
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
                        break # 빈칸이면 열린 끝으로 간주하고 중단
                    else:
                        break # 막혔으면 중단
                    nr += dr * step
                    nc += dc * step
            
            # 가치 평가 및 즉각적인 도파민(점수) 분비!
            if consecutive >= 5:
                reward += 1.0  # 승리 확정타 (최고점)
            elif consecutive == 4 and open_ends >= 1:
                reward += 0.5  # 4목 (치명적 위협)
                pattern_counts['four'] += 1
            elif consecutive == 3 and open_ends == 2:
                reward += 0.2  # 열린 3목 (아주 훌륭한 공격 포메이션)
                pattern_counts['open_3'] += 1
            
        # 오목의 꽃: '양수겸장(4-3, 3-3 등)' 달성 시 엄청난 콤보 보너스!
        if pattern_counts['four'] >= 2 or (pattern_counts['four'] >= 1 and pattern_counts['open_3'] >= 1) or pattern_counts['open_3'] >= 2:
            reward += 0.8
        
        return reward
    
    # 무조건 해야 되는 행동 하드코딩
    def _find_urgent_move(self, state, valid_moves):
        board_size = state.shape[0]

        # 1순위: 내가 두면 바로 5목이 되는 자리 (승리 확정)
        for move in valid_moves:
            r, c = move // board_size, move % board_size
            if self._check_pattern(state, r, c, player=1, target=5):
                return move
        
        # 2순위: 상대방이 두면 5목이 되는 자리 (즉시 방어)
        for move in valid_moves:
            r, c = move // board_size, move % board_size
            if self._check_pattern(state, r, c, player=2, target=5):
                return move
        
        # 3순위: 상대방의 '양수겸장(4-3, 3-3)' 자리를 미리 차단
        for move in valid_moves:
            r, c = move // board_size, move % board_size
            threat_count = 0
            # 가상으로 상대방이 두었다고 가정하고 패턴 체크
            if self._check_pattern(state, r, c, player=2, target=4): # 4목 위협
                threat_count += 1
            if self._check_pattern(state, r, c, player=2, target=3, open_ends_req=2): # 열린 3 위협
                threat_count += 1
            
            if threat_count >= 2:
                return move
        
        # 4순위: 상대방이 두면 '양쪽이 열린 4목'이 되는 자리 (열린 3 방어)
        for move in valid_moves:
            r, c = move // board_size, move % board_size
            if self._check_pattern(state, r, c, player=2, target=4, open_ends_req=2):
                return move
        
        # 긴급한 상황이 아니면 None을 반환하여 CNN에게 결정권을 넘김
        return None
    
    # 무조건 해야 되는 행동의 패턴
    def _check_pattern(self, state, r, c, player, target, open_ends_req=0):
        # 가로, 세로, 대각선 방향 탐색
        directions = [(0, 1), (1, 0), (1, 1), (-1, 1)]
        board_size = state.shape[0]

        for dr, dc in directions:
            consecutive = 1
            open_ends = 0

            # 양방향(+, -)으로 돌이 얼마나 이어져 있는지 확인
            for step in (1, -1):
                nr, nc = r + dr * step, c + dc * step
                while 0 <= nr < board_size and 0 <= nc < board_size:
                    if state[nr, nc] == player:
                        consecutive += 1
                    elif state[nr, nc] == 0:
                        open_ends += 1
                        break # 빈칸을 만나면 열린 끝으로 간주하고 탐색 중단
                    else:
                        break # 상대 돌이나 벽을 만나면 탐색 중단
                    nr += dr * step
                    nc += dc * step
            # 목표한 개수(target)와 열린 끝(open_ends) 조건을 모두 만족하는가?
            if consecutive >= target and open_ends >= open_ends_req:
                return True
        return False
    
    # 기억 장치(Episode Memory 저장)
    def memorize_episode(self, episode_memory, final_reward):
        discounted_reward = final_reward # (상태, 행동, 내재적 보상)
        for state, action, step_reward in reversed(episode_memory):
            # 최종 승패(보상)에 내재적 보상을 더함
            total_reward = step_reward + discounted_reward
            # 메인 메모리(deque)에 기보를 영구 저장
            self.memory.append((state, action, total_reward))
            # 과거로 갈수록 보상의 영향력을 할인 (Gamma = 0.99)
            discounted_reward *= self.gamma
    
    # 복습 엔진
    def replay_experience(self):
        if len(self.memory) < self.batch_size:
            return
        
        minibatch = random.sample(self.memory, self.batch_size)
        states = []
        actions = []
        targets = []

        for state, action, reward in minibatch:
            states.append(state)
            actions.append(action)
            targets.append(reward)
        
        # 파이토치 텐서로 변환하여 GPU 연산 준비
        states_tensor = torch.FloatTensor(np.array(states)).to(self.device)
        actions_tensor = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        targets_tensor = torch.FloatTensor(targets).to(self.device)

        # 현재 뇌(CNN)가 평가하는 바둑판의 가치(Q-Value) 예측
        q_values = self.model(states_tensor)
        # 모델이 예측한 225개의 가치 중, '실제로 두었던 위치(action)'의 가치만 추출
        current_q = q_values.gather(1, actions_tensor).squeeze()
        # (내가 예측한 가치 - 실제 얻은 보상)의 평균 제곱 오차(MSE) 계산
        loss = torch.nn.MSELoss()(current_q, targets_tensor)

        # 오차를 줄이는 방향으로 뇌세포(가중치) 업데이트
        self.optimizer.zero_grad()
        loss.backward()
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
    
    
    model = OmokCNN()
    agent1 = KhyAgent(model)
    
    agent1.load_model("khy_omok_model_gen2_final.pth")
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
    
# def train_main():
#     env = OmokEnvGUI(render_mode=None) 
    
#     model1 = OmokCNN()
#     agent1 = KhyAgent(model1)
#     agent1.load_model("khy_omok_model_final.pth")
#     agent1.train_mode()
    
#     agent1.epsilon = 0.3
#     agent1.epsilon_decay = 0.9982
    
#     model2 = OmokCNN()
#     agent2 = KhyAgent(model2)
    
#     agent2.model.load_state_dict(agent1.model.state_dict())
#     agent2.eval_mode()
    
#     EPISODES = 10000
#     agent1_wins = 0
#     pbar = tqdm(range(1, EPISODES + 1), desc="학습 진행률")
    
#     for episode in pbar:
#         state, info = env.reset()
#         terminated = False
#         episode_memory = []
        
#         while not terminated:
#             current_player = info["current_player"]
            
#             if current_player == 1:
#                 action = agent1.select_action(state)
#                 # 내적 보상 계산 (이전에 추가하신 함수가 있다면 사용, 없다면 0으로 두셔도 무방합니다)
#                 step_reward = agent1.get_intrinsic_reward(state, action) if hasattr(agent1, 'get_intrinsic_reward') else 0.0
#                 episode_memory.append((state.copy(), action, step_reward))
                
#                 next_state, reward, terminated, _, info = env.step(action)
#                 state = next_state
#             else:
#                 inverted_state = np.where(state == 1, 2, np.where(state == 2, 1, 0))
#                 # 파트너는 agent2의 뇌로 생각해서 둡니다.
#                 action = agent2.select_action(inverted_state) 
#                 next_state, reward, terminated, _, info = env.step(action)
#                 state = next_state

#         # ==========================================
#         # 게임 종료 후: 기억 저장 및 대규모 복습 진행
#         # ==========================================
#         winner = info.get("winner")
#         if winner == 1:
#             final_reward = 1.0     # 승리
#             agent1_wins += 1
#         elif winner == 2:
#             final_reward = -1.0    # 패배
#         else:
#             final_reward = -1.0    # 무승부 (패배 처리)
            
#         # agent1만 학습을 진행합니다. (agent2는 맞으면서 데이터만 제공하는 샌드백 역할)
#         agent1.memorize_episode(episode_memory, final_reward)
#         for _ in range(4):
#             agent1.replay_experience()

#         # 진행 상황 표시 (메모리에 데이터가 얼마나 쌓였는지도 확인 가능)
#         win_rate = (agent1_wins / episode) * 100
#         pbar.set_postfix({
#             "승리": f"{agent1_wins}/{episode} | {win_rate:.2f}%",
#             "앱실론": f"{agent1.epsilon:.3f}",
#             "메모리": f"{len(agent1.memory)}" # 뇌 용량이 차오르는 것을 볼 수 있습니다
#         })
        
#         # 탐험률 감소
#         agent1.decay_epsilon()
        
#         # 1000판마다 중간 저장
#         if episode % 1000 == 0:
#             agent1.save_model(f"khy_omok_model_ep{episode}.pth")
#             agent2.model.load_state_dict(agent1.model.state_dict())
            
#             agent1.epsilon = 0.3
#             agent1.epsilon_decay = 0.9982
            
#     # 전체 학습 종료 후 최종 뇌 구조 저장
#     agent1.save_model("khy_omok_model_final.pth")
#     print("\n=== 1만 판의 수읽기 완료 ===")
#     env.close()

def train_main():
    env = OmokEnvGUI(render_mode=None)
    
    model1 = OmokCNN()  
    agent1 = KhyAgent(model1)
    print(f"[Device 확인] {agent1.device}")
    agent1.train_mode()
    
    model2 = OmokCNN()
    agent2_self = KhyAgent(model2)
    agent2_self.eval_mode()
    
    agent_heur = HeuristicAgent(name="Heuristic_White")
    
    N = 10
    EPISODES = 10000 
    
    for gen in range(1, N + 1):
        print(f"\n{'='*40}")
        print(f"[Generation {gen}/{N}] 제 {gen}세대")
        print(f"{'='*40}")
        
        # 매 세대가 시작될 때마다 1막(기초 훈련) 셋팅으로 초기화
        agent1.epsilon = 0.05
        agent1.epsilon_decay = 0.998
        agent1_wins = 0
        total_phase_steps = 0
        
        # 원칙 2. tqdm 진행바는 매 세대마다 새로 생성!
        pbar = None
        
        for episode in range(1, EPISODES + 1):
            if episode == 1:
                    # 1막 시작 (1 ~ 2000)
                    pbar = tqdm(total=2000, desc=f"[Gen {gen}] 1막 VS 휴", position=0, leave=True)
                    
            elif episode == 2001:
                pbar.close() # 1막 바 닫고 화면에 남김
                agent1_wins = 0
                total_phase_steps = 0
                agent1.epsilon = 0.2  
                agent1.epsilon_decay = 0.9995
                # 2막 시작 (2001 ~ 8000)
                pbar = tqdm(total=6000, desc=f"[Gen {gen}] 2막 VS 셀프", position=0, leave=True)
                
            elif episode == 8001:
                pbar.close() # 2막 바 닫고 화면에 남김
                agent1_wins = 0 
                total_phase_steps = 0 
                agent1.epsilon = 0.0  
                agent1.epsilon_decay = 1.0 
                # 3막 시작 (8001 ~ 10000)
                pbar = tqdm(total=2000, desc=f"[Gen {gen}] 3막 VS 휴", position=0, leave=True)
                
            state, info = env.reset()
            terminated = False
            memory_b = [] 
            memory_w = []
            current_episode_steps = 0 
            
            while not terminated:
                current_player = info["current_player"]
                
                if current_player == 1:
                    action = agent1.select_action(state)
                    step_reward = agent1.get_intrinsic_reward(state, action)
                    memory_b.append((state.copy(), action, step_reward))
                    next_state, reward, terminated, _, info = env.step(action)
                    state = next_state
                else:
                    inverted_state = np.where(state == 1, 2, np.where(state == 2, 1, 0))
                    if episode <= 2000 or episode >= 8001:
                        action = agent_heur.select_action(inverted_state) 
                    else:
                        action = agent2_self.select_action(inverted_state) 
                        
                    step_reward = agent1.get_intrinsic_reward(inverted_state, action)
                    memory_w.append((inverted_state.copy(), action, step_reward))
                    
                    next_state, reward, terminated, _, info = env.step(action)
                    state = next_state
                    
                current_episode_steps += 1 
                
            total_phase_steps += current_episode_steps

            # --- 게임 종료 후: 기억 저장 및 복습 ---
            winner = info.get("winner")
            if winner == 1:
                agent1_wins += 1
                agent1.memorize_episode(memory_b, 1.0)   
                agent1.memorize_episode(memory_w, -1.0)  
            elif winner == 2:
                agent1.memorize_episode(memory_b, -1.0)  
                agent1.memorize_episode(memory_w, 1.0)   
            else:
                agent1.memorize_episode(memory_b, -0.5)
                agent1.memorize_episode(memory_w, -0.5)
                
            for _ in range(4):
                agent1.replay_experience()

            # --- 통계 계산 ---
            if episode <= 2000:
                current_ep = episode
            elif episode <= 8000:
                current_ep = episode - 2000
            else:
                current_ep = episode - 8000

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

            if episode % 1000 == 0:
                agent1.save_model(f"khy_omok_model_ep{episode}.pth")
                if 2000 <= episode < 8000:
                    agent1.epsilon = 0.2  
                    agent2_self.model.load_state_dict(agent1.model.state_dict())
                
        # 한 세대(1만 판)가 끝날 때마다 최종 결과물 저장
        if pbar is not None:
            pbar.close()
        
        agent1.save_model(f"khy_omok_model_final.pth")
        
    print(f"\n=== 총 {N}세대({N * EPISODES}판)의 대장정 완료 ===")
    env.close()
    
# ==========================================
# 4. 메인
# ==========================================
if __name__ == "__main__":
    # main()
    train_main()