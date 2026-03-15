"""
train.py — 오목 강화학습 최종 훈련 스크립트
===============================================
[구조 개요]
  관찰값:  (3, 15, 15) CNN 입력 — [내 돌 / 상대 돌 / 빈칸] 이진 채널
  정책:    MaskablePPO + 커스텀 CNN (15x15 공간 특징 최적화)
  보상:    극단적 Dense Reward — 승패·공격·수비 전 상황 보상 부여
  커리큘럼:
    Phase 1 (0 ~ 5M steps): 휴리스틱 봇 상대로 기본 전략 습득
    Phase 2 (5M ~ 10M):     자기 자신(Self-Play)과 대결하며 실력 향상
  마스킹:  Action Masking 유지 (유효하지 않은 수 완전 차단)

[사용법]
  pip install stable-baselines3 sb3-contrib
  python train.py
"""

import os
import re
import numpy as np
import gymnasium as gym
from gymnasium import spaces

import torch
import torch.nn as nn

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, CallbackList

from gomoku import OmokEnvGUI


# ==========================================
# 공통 유틸: 보드 → CNN 입력 변환
# ==========================================
def board_to_cnn_obs(board: np.ndarray) -> np.ndarray:
    """
    2D 보드 (15,15) → 3채널 CNN 입력 (3, 15, 15)
    항상 에이전트가 player=1 관점으로 변환됨.
      Ch0: 내 돌   (board == 1) → 1.0
      Ch1: 상대 돌 (board == 2) → 1.0
      Ch2: 빈칸    (board == 0) → 1.0
    """
    return np.stack([
        (board == 1).astype(np.float32),
        (board == 2).astype(np.float32),
        (board == 0).astype(np.float32),
    ], axis=0)  # (3, 15, 15)


# ==========================================
# 1. 커스텀 CNN 특징 추출기 (15x15 최적화)
# ==========================================
class OmokCNN(BaseFeaturesExtractor):
    """
    15x15 오목 보드 전용 CNN.
    NatureCNN은 84x84 이미지용이라 15x15에서 공간 정보 손실 심각.
    이 CNN은 padding=1로 15x15 해상도를 최대한 유지하며 특징을 추출한다.

    [구조]
      (3,15,15) → Conv×3 (64→128→256, 15x15 유지) →
      Conv (stride=2, 8x8로 압축) → Flatten → Linear(features_dim)
    """

    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 512):
        super().__init__(observation_space, features_dim)
        n_ch = observation_space.shape[0]  # 3

        self.cnn = nn.Sequential(
            nn.Conv2d(n_ch, 64,  kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(64,  128, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, padding=1), nn.ReLU(),
            # 15x15 → 8x8 다운샘플 (stride=2)
            nn.Conv2d(256, 256, kernel_size=3, stride=2,  padding=1), nn.ReLU(),
            nn.Flatten(),
        )

        with torch.no_grad():
            n_flat = self.cnn(torch.zeros(1, *observation_space.shape)).shape[1]

        self.head = nn.Sequential(
            nn.Linear(n_flat, features_dim),
            nn.ReLU(),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.head(self.cnn(obs))


# ==========================================
# 2. 휴리스틱 봇 (Phase 1 상대방)
# ==========================================
class HeuristicBot:
    """
    규칙 기반 오목 봇. 착수 우선순위:
      1순위: 자신이 놓으면 바로 5목이 되어 승리하는 자리
      2순위: 상대방의 열린 3목 또는 4목을 차단하는 자리
             (위협이 클수록 우선 — opp_count 내림차순)
      3순위: 자신의 3목을 4목으로 확장하는 자리
      4순위: 기존 돌에 인접한 빈칸 중 무작위 선택
    """

    def __init__(self, board_size: int = 15):
        self.board_size = board_size

    # ── 내부 헬퍼: 4방향 최대 연속 돌 수 ───────────────────────────
    def _max_consecutive(self, board: np.ndarray, r: int, c: int, player: int) -> int:
        bs = self.board_size
        best = 0
        for dr, dc in [(0, 1), (1, 0), (1, 1), (-1, 1)]:
            cnt = 1
            for s in (1, -1):
                nr, nc = r + dr * s, c + dc * s
                while 0 <= nr < bs and 0 <= nc < bs and board[nr, nc] == player:
                    cnt += 1
                    nr += dr * s
                    nc += dc * s
            best = max(best, cnt)
        return best

    def get_action(self, board: np.ndarray, player: int = 2) -> int:
        """
        현재 보드(2D int8 배열)를 분석해 최적 착수 위치를 정수(row*bs+col)로 반환.
        player: 이 봇이 사용할 돌 번호 (기본값 2 = 흰 돌)
        """
        bs  = self.board_size
        opp = 3 - player

        empty = [(r, c) for r in range(bs) for c in range(bs) if board[r, c] == 0]
        if not empty:
            return 0

        win_moves    = []   # (r, c)          1순위: 즉시 승리
        block_moves  = []   # (opp_cnt, r, c) 2순위: 상대 3·4목 차단
        extend_moves = []   # (r, c)          3순위: 내 3목 → 4목 확장

        for r, c in empty:
            # 내가 놓으면 몇 목이 되는가?
            board[r, c] = player
            my_cnt = self._max_consecutive(board, r, c, player)
            board[r, c] = 0

            # 상대가 놓으면 몇 목이 되는가?
            board[r, c] = opp
            opp_cnt = self._max_consecutive(board, r, c, opp)
            board[r, c] = 0

            if my_cnt >= 5:          # 1순위
                win_moves.append((r, c))
            elif opp_cnt >= 3:       # 2순위 (3·4·5목 모두 포함, 내림차순 정렬)
                block_moves.append((opp_cnt, r, c))
            elif my_cnt == 3:        # 3순위
                extend_moves.append((r, c))

        if win_moves:
            r, c = win_moves[0]
            return r * bs + c

        if block_moves:
            block_moves.sort(reverse=True)   # 위협 큰 순서(5>4>3)
            _, r, c = block_moves[0]
            return r * bs + c

        if extend_moves:
            r, c = extend_moves[0]
            return r * bs + c

        # 4순위: 기존 돌에 인접한 빈칸 (없으면 전체 빈칸) 중 무작위 선택
        adjacent: set = set()
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
        return r * bs + c


# ==========================================
# 3. 상대방 모델 공유 컨테이너
# ==========================================
class OpponentHolder:
    """
    모든 병렬 환경이 동일한 상대방을 참조하는 공유 객체.

    use_heuristic=True  → Phase 1: 휴리스틱 봇
    use_heuristic=False → Phase 2: Self-Play 체크포인트 모델
    """

    def __init__(self):
        self.model = None
        self.use_heuristic = True          # Phase 1 기본값
        self._heuristic_bot = HeuristicBot()  # HeuristicBot 인스턴스

    def get_action(self, board: np.ndarray, valid_mask: np.ndarray) -> int:
        if self.use_heuristic:
            return self._heuristic_bot.get_action(board.copy(), player=2)

        # Self-Play: 30% 확률로 HeuristicBot 유지 (3목 차단 학습 신호 공급)
        # → 에이전트가 3목을 만들면 반드시 막히는 경험을 꾸준히 제공
        if self.model is None or np.random.random() < 0.30:
            return self._heuristic_bot.get_action(board.copy(), player=2)

        # 상대방 관점 (색상 반전) CNN 입력
        inv_board = np.where(board == 1, 2, np.where(board == 2, 1, 0)).astype(np.int8)
        obs_cnn = board_to_cnn_obs(inv_board)

        try:
            action, _ = self.model.predict(obs_cnn, deterministic=False, action_masks=valid_mask)
        except TypeError:
            action, _ = self.model.predict(obs_cnn, deterministic=False)
        action = int(action)

        if not valid_mask[action]:
            action = int(np.random.choice(np.where(valid_mask)[0]))
        return action


# ==========================================
# 4. 학습 환경 (CNN 관찰 + Dense Reward + Action Masking)
# ==========================================
class OmokTrainEnv(gym.Env):
    """
    MaskablePPO + CnnPolicy 학습 전용 환경.

    [관찰값]  (3, 15, 15) float32
    [행동]    Discrete(225), Action Masking으로 빈칸만 선택 가능

    [보상 테이블]
      승리               +10.0
      패배               -10.0
      무승부              0.0
      내 4목 달성        +2.0
      내 3목 달성        +0.5
      상대 승리수 차단   +5.0  (승리에 준하는 수비 보상)
      상대 4목 차단      +2.5
      상대 3목 차단      +0.8
    """

    metadata = {"render_modes": []}

    def __init__(self, opponent_holder: OpponentHolder = None):
        super().__init__()
        self._env = OmokEnvGUI(render_mode=None)
        self.bs = self._env.board_size       # 15
        self._n = self.bs ** 2               # 225
        self._holder = opponent_holder

        self.action_space = spaces.Discrete(self._n)
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(3, self.bs, self.bs), dtype=np.float32
        )

    # ── Gymnasium 인터페이스 ──────────────────────────────────────────
    def reset(self, seed=None, options=None):
        obs, info = self._env.reset(seed=seed)
        return board_to_cnn_obs(obs), info

    def step(self, action):
        action = int(action)
        board_before = self._env.board.copy()  # 에이전트 착수 前 보드

        # ── [A] 에이전트(흑돌) 착수 ─────────────────────────────────
        obs, raw_reward, terminated, truncated, info = self._env.step(action)

        if terminated:
            winner = info.get("winner", 0)
            if winner == 1:   final_reward = 10.0   # 승리
            elif winner == 0: final_reward = 0.0    # 무승부
            else:             final_reward = -10.0  # 반칙(마스킹으로 드물게 발생)
            return board_to_cnn_obs(obs), final_reward, terminated, truncated, info

        # ── [B] 에이전트 착수 보상 계산 ─────────────────────────────
        r, c = action // self.bs, action % self.bs
        my_count = self._max_consecutive(obs, r, c, player=1)
        my_reward = {3: 1.0, 4: 2.0}.get(min(my_count, 4), 0.0)

        # ── [B2] 무적수 보상: 쌍삼(33) / 사삼(43) / 쌍사(44) ─────────
        # 돌 하나로 두 방향 이상에서 동시에 위협을 만들면 상대가 막을 수 없음
        three_dirs = self._count_threats(obs, r, c, player=1, threshold=3)
        four_dirs  = self._count_threats(obs, r, c, player=1, threshold=4)

        if four_dirs >= 2:                        # 쌍사(44): 최강 무적수
            multi_threat_reward = 4.0
        elif four_dirs == 1 and three_dirs >= 2:  # 사삼(43): 강한 무적수
            multi_threat_reward = 3.0
        elif three_dirs >= 2:                     # 쌍삼(33): 기본 무적수
            multi_threat_reward = 2.0
        else:
            multi_threat_reward = 0.0

        # ── [C] 차단 보상: 이 자리에 상대가 놓았다면? ───────────────
        tmp = board_before.copy()
        tmp[r, c] = 2  # 상대가 놓는다고 가정
        blocked_count = self._max_consecutive(tmp, r, c, player=2)
        if blocked_count >= 5:   block_reward = 5.0   # 상대 승리수 차단!
        elif blocked_count == 4: block_reward = 2.5
        elif blocked_count == 3: block_reward = 2.2   # 1.5 → 2.2: 3목 차단 추가 강화
        else:                    block_reward = 0.0

        # ── [D] 상대방 응수 ──────────────────────────────────────────
        valid_mask = (obs.flatten() == 0)
        if not valid_mask.any():
            return board_to_cnn_obs(obs), my_reward + block_reward, True, False, {"winner": 0}

        opp_action = self._holder.get_action(self._env.board.copy(), valid_mask) \
                     if self._holder else int(np.random.choice(np.where(valid_mask)[0]))

        obs, _, terminated, truncated, info = self._env.step(opp_action)

        if terminated:
            winner = info.get("winner", 0)
            opp_reward = -10.0 if winner == 2 else 0.0
            threat_penalty = 0.0
        else:
            opp_reward = 0.0
            # ── [E] 차단 실패 패널티: 상대가 3목+ 완성 시 즉시 페널티 ──
            # 차단 보상(양수)만으로는 학습 불충분 → 실패 신호(음수)로 보완
            opp_r = opp_action // self.bs
            opp_c = opp_action % self.bs
            opp_consec = self._max_consecutive(obs, opp_r, opp_c, player=2)
            if opp_consec >= 4:   threat_penalty = -6.0   # -3.0 → -6.0 (4목 방치 = 거의 패배)
            elif opp_consec == 3: threat_penalty = -4.0   # -2.0 → -4.0 (3목 방치 강한 경고)
            else:                 threat_penalty = 0.0

        final_reward = my_reward + block_reward + multi_threat_reward + opp_reward + threat_penalty
        return board_to_cnn_obs(obs), final_reward, terminated, truncated, info

    # ── Action Masking (MaskablePPO 전용) ───────────────────────────
    def action_masks(self) -> np.ndarray:
        return (self._env.board.flatten() == 0)  # (225,) bool

    # ── 내부 헬퍼 ────────────────────────────────────────────────────
    def _max_consecutive(self, board_2d: np.ndarray, r: int, c: int, player: int) -> int:
        best = 0
        for dr, dc in [(0, 1), (1, 0), (1, 1), (-1, 1)]:
            cnt = 1
            for s in (1, -1):
                nr, nc = r + dr * s, c + dc * s
                while 0 <= nr < self.bs and 0 <= nc < self.bs and board_2d[nr, nc] == player:
                    cnt += 1; nr += dr * s; nc += dc * s
            best = max(best, cnt)
        return best

    def _count_threats(self, board_2d: np.ndarray, r: int, c: int,
                       player: int, threshold: int) -> int:
        """
        (r,c)에 player 돌이 놓인 상태에서, threshold 이상의 연속수가
        나오는 방향이 몇 개인지 반환.
        쌍삼·사삼·쌍사 감지에 사용.
        """
        count = 0
        for dr, dc in [(0, 1), (1, 0), (1, 1), (-1, 1)]:
            cnt = 1
            for s in (1, -1):
                nr, nc = r + dr * s, c + dc * s
                while 0 <= nr < self.bs and 0 <= nc < self.bs and board_2d[nr, nc] == player:
                    cnt += 1; nr += dr * s; nc += dc * s
            if cnt >= threshold:
                count += 1
        return count

    def render(self): pass
    def close(self): self._env.close()


# ==========================================
# 5. 커리큘럼 콜백 (Phase 1 → Phase 2 자동 전환)
# ==========================================
class CurriculumCallback(BaseCallback):
    """
    Phase 1: 휴리스틱 봇과 대결 (기본 공격/수비 패턴 학습)
    Phase 2: Self-Play — 주기적으로 현재 모델을 저장 후 상대방으로 교체

    phase2_start 타임스텝 도달 시 자동으로 전환.
    """

    def __init__(self, holder: OpponentHolder, phase2_start: int,
                 selfplay_freq: int, save_dir: str = "./opponent_pool/", verbose: int = 1):
        super().__init__(verbose)
        self.holder = holder
        self.phase2_start = phase2_start
        self.selfplay_freq = selfplay_freq
        self.save_dir = save_dir
        self._in_phase2 = False
        os.makedirs(save_dir, exist_ok=True)

    def _on_step(self) -> bool:
        t = self.num_timesteps

        # ── Phase 1 → Phase 2 전환 ────────────────────────────────
        if not self._in_phase2 and t >= self.phase2_start:
            self._in_phase2 = True
            self.holder.use_heuristic = False
            self._save_opponent(t)
            if self.verbose:
                print(f"\n{'='*55}")
                print(f"  [커리큘럼] Phase 2 돌입! Self-Play 시작 (step {t:,})")
                print(f"{'='*55}")

        # ── Phase 2: 주기적 상대방 업데이트 ─────────────────────────
        elif self._in_phase2 and self.n_calls % self.selfplay_freq == 0:
            self._save_opponent(t)
            if self.verbose:
                print(f"\n[Self-Play] 상대방 업데이트 완료 (step {t:,})")

        return True

    def _save_opponent(self, step: int):
        path = os.path.join(self.save_dir, f"opponent_{step}")
        self.model.save(path)
        try:
            from sb3_contrib import MaskablePPO
            self.holder.model = MaskablePPO.load(path)
        except Exception:
            from stable_baselines3 import PPO
            self.holder.model = PPO.load(path)


# ==========================================
# 6. 메인 훈련 파이프라인
# ==========================================
if __name__ == "__main__":
    # ── MaskablePPO 임포트 ──────────────────────────────────────────
    try:
        from sb3_contrib import MaskablePPO
        from sb3_contrib.common.wrappers import ActionMasker
        USE_MASKING = True
        print("[INFO] sb3-contrib 감지 → MaskablePPO + Action Masking 활성화")
    except ImportError:
        from stable_baselines3 import PPO as MaskablePPO
        USE_MASKING = False
        print("[INFO] sb3-contrib 없음 → 일반 PPO 사용 (pip install sb3-contrib 권장)")

    # ── GPU / CPU 디바이스 자동 감지 ────────────────────────────────
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] 학습 디바이스: {DEVICE.upper()}")
    if DEVICE == "cuda":
        print(f"[INFO] GPU: {torch.cuda.get_device_name(0)}")

    print("=" * 55)
    print("  오목 최종 훈련 (CNN + Heuristic → Self-Play)")
    print("=" * 55)

    # ── 커리큘럼 설정 ────────────────────────────────────────────────
    # [3목 차단 집중 학습]
    # 5M 최고 모델에서 이어서, Self-Play 없이 HeuristicBot 100%로 3M 추가 학습
    # → HeuristicBot은 항상 3목을 확장하므로 차단 학습 신호가 지속 발생
    TOTAL_TIMESTEPS = 8_000_000           # 5M 체크포인트 기준 +3M 추가 (~1.5시간)
    PHASE2_START    = 99_999_999          # Self-Play 진입 비활성화 (HeuristicBot 100%)
    SELFPLAY_FREQ   = 100_000

    # ── 상대방 컨테이너 생성 ────────────────────────────────────────
    holder = OpponentHolder()

    # ── 환경 팩토리 ─────────────────────────────────────────────────
    def make_env():
        env = OmokTrainEnv(opponent_holder=holder)
        if USE_MASKING:
            env = ActionMasker(env, lambda e: e.action_masks())
        return env

    N_ENVS = 8
    print(f"\n[1/3] 병렬 환경 초기화 중 (DummyVecEnv × {N_ENVS})...")
    vec_env = make_vec_env(make_env, n_envs=N_ENVS)

    # ── 모델 초기화 (체크포인트 자동 감지) ──────────────────────────
    print("\n[2/3] 모델 초기화 중...")

    # ── 5M 체크포인트 명시적 고정 ────────────────────────────────────
    # 9M 모델은 HeuristicBot 30% 혼합으로 퇴보(ep_rew_mean=-5.88)
    # → 5M(ep_rew_mean=37, explained_variance=0.15) 에서 재출발
    FIXED_CKPT = "./checkpoints/omok_ppo_5000000_steps.zip"

    latest_ckpt = None
    ckpt_steps  = 0
    if os.path.isfile(FIXED_CKPT):
        latest_ckpt = FIXED_CKPT
        ckpt_steps  = 5_000_000
        print(f"  [체크포인트 고정] {FIXED_CKPT}")
    elif os.path.isdir("./checkpoints"):
        zips = sorted(
            [f for f in os.listdir("./checkpoints") if f.endswith(".zip")],
            key=lambda f: os.path.getmtime(os.path.join("./checkpoints", f)),
        )
        if zips:
            latest_ckpt = os.path.join("./checkpoints", zips[-1])
            m = re.search(r'_(\d+)_steps', zips[-1])
            ckpt_steps = int(m.group(1)) if m else 0

    def _new_model():
        """CNN 구조의 새 MaskablePPO 모델 생성"""
        print("  새 모델로 학습 시작 (Phase 1: 휴리스틱 봇)")
        return MaskablePPO(
            policy="CnnPolicy",
            env=vec_env,
            verbose=1,
            learning_rate=0.00005,  # 0.0001 → 0.00005: 안정적 수렴
            n_steps=4096,
            batch_size=512,
            n_epochs=5,             # 10 → 5: 과적합 방지, 가치함수 안정화
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.1,
            ent_coef=0.02,
            vf_coef=2.0,            # 1.0 → 2.0: 가치함수 학습 강화 (explained_variance 음수 수정)
            max_grad_norm=0.5,
            device=DEVICE,
            target_kl=0.015,        # 0.03 → 0.015: KL 더 엄격하게 제한
            policy_kwargs=dict(
                features_extractor_class=OmokCNN,
                features_extractor_kwargs=dict(features_dim=512),
                net_arch=[256, 256],
            ),
        )

    # ── 체크포인트 로드 (관찰값 공간 불일치 시 새 모델로 폴백) ──────
    reset_num_timesteps = True
    if latest_ckpt:
        try:
            model = MaskablePPO.load(latest_ckpt, env=vec_env)
            model.ent_coef = 0.05   # 탐색 약간 늘려 새 패널티 구조 적응
            reset_num_timesteps = False
            holder.use_heuristic = True   # Self-Play 비활성화: HeuristicBot 100% 고정
            print(f"  체크포인트 로드 성공 ({ckpt_steps:,} steps) → 이어서 학습")
            print(f"  → [3목 차단 집중] Phase 1 전용 (HeuristicBot 100%, Self-Play 없음)")
        except ValueError as e:
            # 관찰값 공간 불일치(MLP→CNN 전환 등) → 새 모델로 시작
            print(f"  ⚠️  체크포인트 호환 불가 → 새 모델로 시작합니다.")
            print(f"     (원인: {e})")
            print(f"  💡 기존 checkpoints/ 폴더를 삭제하거나 백업 후 재실행하세요.")
            model = _new_model()
            ckpt_steps = 0
    else:
        model = _new_model()

    # ── 콜백 설정 ────────────────────────────────────────────────────
    curriculum_cb = CurriculumCallback(
        holder,
        phase2_start=PHASE2_START,
        selfplay_freq=SELFPLAY_FREQ // N_ENVS,
        verbose=1,
    )
    checkpoint_cb = CheckpointCallback(
        save_freq=500_000 // N_ENVS,
        save_path="./checkpoints/",
        name_prefix="omok_ppo",
        verbose=1,
    )

    # ── 학습 실행 ────────────────────────────────────────────────────
    remaining = max(TOTAL_TIMESTEPS - ckpt_steps, 1_000_000)
    print(f"\n[3/3] 학습 시작 — 총 {remaining:,} 타임스텝")
    if PHASE2_START >= TOTAL_TIMESTEPS:
        print(f"  Phase 1 (휴리스틱 봇):  전 구간 고정 (Self-Play 비활성화)\n")
    else:
        print(f"  Phase 1 (휴리스틱 봇):  0 ~ {PHASE2_START:,} steps")
        print(f"  Phase 2 (Self-Play):    {PHASE2_START:,} ~ {TOTAL_TIMESTEPS:,} steps\n")

    model.learn(
        total_timesteps=remaining,
        callback=CallbackList([curriculum_cb, checkpoint_cb]),
        reset_num_timesteps=reset_num_timesteps,
        progress_bar=True,
    )

    model.save("omok_model")
    print("\n" + "=" * 55)
    print("  학습 완료! 모델 저장: omok_model.zip")
    print("=" * 55)
    vec_env.close()
