import numpy as np
import gymnasium as gym
from gymnasium import spaces
import tkinter as tk
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from khy_model import OmokCNN
from collections import deque
import random
from tqdm import tqdm

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
    pass

# ==========================================
# 3. 대결 실행 루프 (Arena)
# ==========================================
def main():
    env = OmokEnvGUI(render_mode="human")
    agent1 = HumanAgent(env)
    agent2 = HeuristicAgent(name="Heuristic_AI")
    
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
            action = agent2.select_action(inverted_state)
            print(f"   ▶ {agent2.name} 착수 완료! (즉시 착수)")
            
        state, reward, terminated, _, info = env.step(action)
        env.render()
        time.sleep(0.1) # 시각적 확인을 위한 지연

    # 결과 판정
    print("\n=== 🏁 대결 종료 ===")
    winner = info.get("winner")
    if winner == 1: print(f"🎉 {agent1.name} 승리!")
    elif winner == 2: print(f"🎉 {agent2.name} 승리!")
    else: print("🤝 무승부!")
        
    time.sleep(3)
    env.close()

if __name__ == "__main__":
    main()