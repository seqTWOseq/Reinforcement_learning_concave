import numpy as np
import gymnasium as gym
from gymnasium import spaces
import tkinter as tk
import time

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

class Agent1:
    """무작위 위치에 착수하는 에이전트"""
    def __init__(self, name="Random_Black(●)"): self.name = name
    def select_action(self, state):
        valid = np.where(state.flatten() == 0)[0]
        return np.random.choice(valid) if len(valid) > 0 else 0

class Agent2:
    """중앙(7, 7)과 가장 가까운 빈칸에 착수하는 에이전트"""
    def __init__(self, name="Center_White(○)"): self.name = name
    def select_action(self, state):
        board_size, center = state.shape[0], state.shape[0] // 2
        valid = np.where(state.flatten() == 0)[0]
        if len(valid) == 0: return 0
        
        # 맨해튼 거리가 가장 짧은 액션 선택
        best_action = min(valid, key=lambda a: abs(a // board_size - center) + abs(a % board_size - center))
        return best_action

class SmartHeuristicAgent:
    """패턴 인식을 통해 공격과 방어를 수행하는 똑똑한 에이전트"""
    def __init__(self, name="Smart_AI(○)"):
        self.name = name

    def select_action(self, state):
        board_size = state.shape[0]
        valid_moves = np.argwhere(state == 0)
        
        if len(valid_moves) == 0:
            return 0

        # 🚨 [수정된 부분] 사람이 이미 중앙에 두었는지 확인하고 피하기
        if len(valid_moves) >= board_size * board_size - 1:
            center = board_size // 2
            if state[center, center] == 0:
                return center * board_size + center  # 중앙이 비어있으면 중앙에
            else:
                return center * board_size + (center + 1) # 중앙이 막혔으면 바로 옆에 착수

        best_score = -float('inf')
        best_action = valid_moves[0]

        # 모든 빈칸에 대해 가치(Score) 평가
        for move in valid_moves:
            r, c = move
            attack_score = self._evaluate_position(state, r, c, player=1)
            defense_score = self._evaluate_position(state, r, c, player=2)
            
            # 방어에 약간 더 높은 가중치 부여
            total_score = attack_score + defense_score * 1.2 
            total_score += np.random.uniform(0, 1)

            if total_score > best_score:
                best_score = total_score
                best_action = move

        return best_action[0] * board_size + best_action[1]

    def _evaluate_position(self, state, r, c, player):
        """특정 위치에 돌을 두었을 때 만들어지는 패턴의 점수를 계산합니다."""
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
        score = 0
        board_size = state.shape[0]
        
        for dr, dc in directions:
            consecutive = 1
            open_ends = 0
            
            for step in (1, -1):
                nr, nc = r + dr * step, c + dc * step
                while 0 <= nr < board_size and 0 <= nc < board_size and state[nr, nc] == player:
                    consecutive += 1
                    nr += dr * step
                    nc += dc * step
                
                if 0 <= nr < board_size and 0 <= nc < board_size and state[nr, nc] == 0:
                    open_ends += 1
            
            if consecutive >= 5: 
                score += 100000 
            elif consecutive == 4 and open_ends == 2: 
                score += 10000
            elif consecutive == 4 and open_ends == 1: 
                score += 1000
            elif consecutive == 3 and open_ends == 2: 
                score += 1000
            elif consecutive == 3 and open_ends == 1: 
                score += 100
            elif consecutive == 2 and open_ends == 2: 
                score += 10
                
        return score

# ==========================================
# 3. 대결 실행 루프 (Arena)
# ==========================================
def main():
    env = OmokEnvGUI(render_mode="human")
    agent1 = HumanAgent(env, name="Human_Black(●)")
    agent2 = SmartHeuristicAgent()
    
    state, info = env.reset()
    env.render()
    terminated = False
    
    print(f"=== ⚔️ {agent1.name} vs {agent2.name} 대결 시작 ===")
    
    while not terminated:
        # 턴에 따른 상태 반전 논리 (상대는 항상 자신이 흑돌인 것처럼 착각하게 만듦)
        if info["current_player"] == 1:
            action = agent1.select_action(state)
        else:
            inverted_state = np.where(state == 1, 2, np.where(state == 2, 1, 0))
            action = agent2.select_action(inverted_state)
            
        state, reward, terminated, _, info = env.step(action)
        env.render()
        time.sleep(0.1) # 시각적 확인을 위한 지연

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