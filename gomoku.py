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


class AgentHeuristic:
    """
    규칙 기반 에이전트 (train.py의 HeuristicBot과 동일한 로직).
    RL 모델 없이도 즉시 사용 가능하며, 기본 공격·수비를 확실히 수행한다.

    착수 우선순위:
      1순위: 즉시 5목 승리
      2순위: 상대 3·4목 차단
      3순위: 내 3목 → 4목 확장
      4순위: 기존 돌 인접 빈칸 무작위
    """

    def __init__(self, name="Heuristic_White(○)"):
        self.name = name
        self.bs = 15

    def _max_consecutive(self, board, r, c, player):
        best = 0
        for dr, dc in [(0, 1), (1, 0), (1, 1), (-1, 1)]:
            cnt = 1
            for s in (1, -1):
                nr, nc = r + dr * s, c + dc * s
                while 0 <= nr < self.bs and 0 <= nc < self.bs and board[nr, nc] == player:
                    cnt += 1; nr += dr * s; nc += dc * s
            best = max(best, cnt)
        return best

    def select_action(self, state):
        # state는 항상 '자신=1, 상대=2' 관점으로 전달된다 (inverted_state)
        board = state.copy()
        bs = self.bs
        empty = [(r, c) for r in range(bs) for c in range(bs) if board[r, c] == 0]
        if not empty:
            return 0

        win_moves, block_moves, extend_moves = [], [], []

        for r, c in empty:
            board[r, c] = 1
            my_cnt = self._max_consecutive(board, r, c, 1)
            board[r, c] = 0

            board[r, c] = 2
            opp_cnt = self._max_consecutive(board, r, c, 2)
            board[r, c] = 0

            if my_cnt >= 5:
                win_moves.append((r, c))
            elif opp_cnt >= 3:
                block_moves.append((opp_cnt, r, c))
            elif my_cnt == 3:
                extend_moves.append((r, c))

        if win_moves:
            r, c = win_moves[0]
        elif block_moves:
            block_moves.sort(reverse=True)
            _, r, c = block_moves[0]
        elif extend_moves:
            r, c = extend_moves[0]
        else:
            adjacent = set()
            for br in range(bs):
                for bc in range(bs):
                    if board[br, bc] != 0:
                        for dr in (-1, 0, 1):
                            for dc in (-1, 0, 1):
                                nr, nc = br + dr, bc + dc
                                if 0 <= nr < bs and 0 <= nc < bs and board[nr, nc] == 0:
                                    adjacent.add((nr, nc))
            pool = list(adjacent) if adjacent else empty
            r, c = pool[np.random.randint(len(pool))]

        action = r * bs + c
        print(f"[{self.name}] 착수: {action} (행={r}, 열={c})")
        return action


class Agent2:
    """
    RL + 규칙 하이브리드 에이전트.

    [동작 방식]
    1. 상대 4목(즉시 승리 위협) 또는 자신의 5목 완성이 가능하면
       규칙(HeuristicBot 로직)이 강제로 그 수를 둔다. (치명적 실수 방지)
    2. 위 조건이 없을 때만 RL 모델의 예측을 사용한다.
    3. 모델 로드에 실패하면 AgentHeuristic 전체로 폴백한다.

    [순환 임포트 방지]
    OmokCNN은 __init__ 안에서 지연 임포트(lazy import)한다.
    """

    def __init__(self, model_path="omok_model.zip", name="PPO_White(○)"):
        self.name = name
        self.model = None
        self.bs = 15
        self._heuristic = AgentHeuristic(name=name + "_fallback")

        try:
            from train import OmokCNN
            try:
                from sb3_contrib import MaskablePPO as _PPO
            except ImportError:
                from stable_baselines3 import PPO as _PPO

            self.model = _PPO.load(
                model_path,
                custom_objects={"features_extractor_class": OmokCNN},
            )
            print(f"[{self.name}] 모델 로드 성공: {model_path}")
        except FileNotFoundError:
            print(f"[{self.name}] 경고: '{model_path}' 없음 → 규칙 기반으로 대체합니다.")
        except Exception as e:
            print(f"[{self.name}] 모델 로드 실패: {e}")
            print(f"[{self.name}] → 규칙 기반(HeuristicBot)으로 대체합니다.")

    def _critical_move(self, state):
        """
        우선순위대로 강제 착수 위치를 반환, 없으면 None.
          P1: 내가 두면 5목 (즉시 승리)
          P2: 상대가 두면 5목 (즉시 차단)
          P3: 내가 두면 4목 (승리 준비)
          P4: 상대가 두면 4목 (4목 차단 — RL이 놓치면 규칙이 막음)
        """
        bs = self.bs
        board = state.copy()
        my_four, opp_four = None, None

        for r in range(bs):
            for c in range(bs):
                if board[r, c] != 0:
                    continue

                board[r, c] = 1
                my_cnt = self._heuristic._max_consecutive(board, r, c, 1)
                board[r, c] = 0

                board[r, c] = 2
                opp_cnt = self._heuristic._max_consecutive(board, r, c, 2)
                board[r, c] = 0

                if my_cnt >= 5:           return r * bs + c   # P1: 즉시 승리
                if opp_cnt >= 5:          return r * bs + c   # P2: 즉시 차단
                if my_cnt == 4 and my_four is None:
                    my_four = r * bs + c
                if opp_cnt == 4 and opp_four is None:
                    opp_four = r * bs + c

        if my_four  is not None: return my_four   # P3: 내 4목 완성
        if opp_four is not None: return opp_four  # P4: 상대 4목 차단
        return None

    def select_action(self, state):
        """
        state: 2D numpy 배열 (15×15), 반전된 뷰 (자신=1, 상대=2)
        """
        obs_flat = state.flatten()
        valid = np.where(obs_flat == 0)[0]
        if len(valid) == 0:
            return 0

        # 모델 없으면 전체 규칙 기반으로 폴백
        if self.model is None:
            return self._heuristic.select_action(state)

        # 1순위: 즉시 승리 or 상대 5목 차단 (규칙 강제 적용)
        critical = self._critical_move(state)
        if critical is not None:
            print(f"[{self.name}] 긴급 착수(규칙): {critical} (행={critical // 15}, 열={critical % 15})")
            return critical

        # 2순위: RL 모델 예측
        obs_cnn = np.stack([
            (state == 1).astype(np.float32),  # Ch0: 내 돌
            (state == 2).astype(np.float32),  # Ch1: 상대 돌
            (state == 0).astype(np.float32),  # Ch2: 빈칸
        ], axis=0)  # (3, 15, 15)

        valid_mask = (obs_flat == 0)

        try:
            action, _ = self.model.predict(obs_cnn, deterministic=True, action_masks=valid_mask)
        except TypeError:
            action, _ = self.model.predict(obs_cnn, deterministic=True)
        action = int(action)

        if obs_flat[action] != 0:
            print(f"[{self.name}] 유효하지 않은 수({action}) → 규칙 기반 폴백")
            return self._heuristic.select_action(state)

        print(f"[{self.name}] 착수(RL): {action} (행={action // 15}, 열={action % 15})")
        return action

# ==========================================
# 3. 대결 실행 루프 (Arena)
# ==========================================
def main():
    env = OmokEnvGUI(render_mode="human")
    agent1 = HumanAgent(env, name="Human_Black(●)")

    # ── AI 선택 ─────────────────────────────────────────────────────
    # RL 모델이 잘 학습됐을 때: Agent2()          (RL + 규칙 하이브리드)
    # RL 모델 없이 규칙만 사용: AgentHeuristic()  (즉시 강한 플레이)
    agent2 = Agent2()                    # omok_model.zip 사용 (학습 완료 후 자동 저장)
    # agent2 = AgentHeuristic()          # RL 없이 규칙만 사용 시
    
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
    elif winner == 2: print(f"🎉 {agent2.name} 승리!")
    else: print("🤝 무승부!")
        
    time.sleep(3)
    env.close()

if __name__ == "__main__":
    main()