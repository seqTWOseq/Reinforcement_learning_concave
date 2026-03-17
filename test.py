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

        
        return np.argmax(final_score)