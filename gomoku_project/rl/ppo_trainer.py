from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Categorical

from gomoku_project.core.constants import (
    BLACK,
    BOARD_SIZE,
    DRAW,
    DRAW_REWARD,
    INVALID_MOVE_PENALTY,
    WHITE,
    WIN_REWARD,
)
from gomoku_project.core.utils import action_to_pos, opponent, orient_board_to_player
from gomoku_project.envs.gomoku_env import GomokuEnv
from gomoku_project.players.heuristic_player import HeuristicPlayer, _analyze_move, build_heuristic_scores
from gomoku_project.players.ppo_player import PPOPlayer
from gomoku_project.rl.ppo_buffer import PPOBuffer
from gomoku_project.rl.ppo_network import (
    PPOActorCritic,
    mix_policy_logits_with_prior,
    normalize_policy_prior_scores,
    observations_to_tensor,
    valid_actions_to_mask,
)

_SUPPORTED_OPPONENTS = frozenset({"self", "heuristic"})
_SUPPORTED_OPPONENT_MODES = frozenset({"self_play_only", "self_play_with_heuristic_pool"})


@dataclass(frozen=True)
class PPORolloutStep:
    observation: np.ndarray
    action_mask: np.ndarray
    action: int
    log_prob: float
    value: float
    player_id: int
    heuristic_beta: float


@dataclass(frozen=True)
class _SampledPolicyAction:
    action: int
    log_prob: float
    value: float
    action_mask: np.ndarray
    player_id: int
    heuristic_beta: float
    selected_action_prob: float
    top1_action_prob: float
    top3_action_prob_sum: float
    entropy: float
    valid_action_count: int


class PPOTrainer:
    def __init__(
        self,
        model: Optional[torch.nn.Module] = None,
        *,
        board_size: int = BOARD_SIZE,
        device: str = "cpu",
        learning_rate: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_epsilon: float = 0.2,
        entropy_coef: float = 0.01,
        value_coef: float = 0.5,
        update_epochs: int = 4,
        minibatch_size: int = 128,
        max_grad_norm: float = 0.5,
        use_gae: bool = True,
        opponent_mode: str = "self_play_only",
        opponent_probs: Optional[dict[str, float]] = None,
        seed: int | None = None,
        opening_ply_cutoff: int = 6,
        center_radius: int = 2,
        use_heuristic_prior: bool = False,
        heuristic_prior_beta_start: float = 0.0,
        heuristic_prior_beta_end: float = 0.0,
        heuristic_prior_decay_updates: int = 0,
        heuristic_prior_score_clip: float = 3.0,
    ) -> None:
        self.board_size = board_size
        self.action_space_size = board_size * board_size
        self.device = torch.device(device)
        self.model = model or PPOActorCritic(board_size=board_size)
        self.model.to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.update_epochs = update_epochs
        self.minibatch_size = minibatch_size
        self.max_grad_norm = max_grad_norm
        self.use_gae = use_gae
        self.opponent_mode = opponent_mode
        self.opponent_probs = self._resolve_opponent_probs(opponent_mode, opponent_probs)
        self.rng = np.random.default_rng(seed)
        self.opening_ply_cutoff = max(int(opening_ply_cutoff), 1)
        self.center_radius = max(int(center_radius), 0)
        self.use_heuristic_prior = bool(use_heuristic_prior)
        self.heuristic_prior_beta_start = max(float(heuristic_prior_beta_start), 0.0)
        self.heuristic_prior_beta_end = max(float(heuristic_prior_beta_end), 0.0)
        self.heuristic_prior_decay_updates = max(int(heuristic_prior_decay_updates), 0)
        self.heuristic_prior_score_clip = max(float(heuristic_prior_score_clip), 0.0)
        self.buffer = PPOBuffer()
        self.training_updates = 0
        self.self_play_games = 0
        self.heuristic_pool_games = 0
        self.training_games = 0

    def build_player(
        self,
        *,
        deterministic: bool,
        name: str,
    ) -> PPOPlayer:
        return PPOPlayer(
            model=self.model,
            device=str(self.device),
            deterministic=deterministic,
            use_heuristic_prior=self.use_heuristic_prior,
            heuristic_prior_beta=self.current_heuristic_prior_beta(),
            heuristic_prior_score_clip=self.heuristic_prior_score_clip,
            name=name,
        )

    def current_heuristic_prior_beta(self) -> float:
        if not self.use_heuristic_prior:
            return 0.0
        if self.heuristic_prior_decay_updates <= 0:
            return float(self.heuristic_prior_beta_end)

        progress = min(float(self.training_updates) / float(self.heuristic_prior_decay_updates), 1.0)
        beta = self.heuristic_prior_beta_start + (
            self.heuristic_prior_beta_end - self.heuristic_prior_beta_start
        ) * progress
        return float(max(beta, 0.0))

    def record_external_episode(
        self,
        steps: Sequence[PPORolloutStep],
        *,
        controlled_player: int,
        winner: int | None,
        reason: str,
        opponent_type: str,
    ) -> int:
        if not steps:
            return 0

        resolved_steps = [
            PPORolloutStep(
                observation=np.asarray(step.observation, dtype=np.int8).copy(),
                action_mask=np.asarray(step.action_mask, dtype=bool).copy(),
                action=int(step.action),
                log_prob=float(step.log_prob),
                value=float(step.value),
                player_id=int(step.player_id),
                heuristic_beta=float(step.heuristic_beta),
            )
            for step in steps
        ]
        resolved_winner = DRAW if winner is None else int(winner)
        return self._store_trajectory(
            steps=resolved_steps,
            controlled_player=int(controlled_player),
            winner=resolved_winner,
            reason=str(reason),
            opponent_type=str(opponent_type),
        )

    def collect_self_play_games(self, num_games: int = 1) -> dict[str, Any]:
        summary = self._new_collection_summary()

        for game_index in range(1, num_games + 1):
            opponent_type = self._sample_opponent_type()
            episode_summary = self._run_episode(
                opponent_type,
                deterministic_policy=False,
                collect_trajectories=True,
                update_counters=True,
                game_index=game_index,
            )
            self._accumulate_collection_summary(summary, episode_summary)

        summary["buffer_size"] = len(self.buffer)
        return self._finalize_collection_summary(summary)

    def evaluate_against_heuristic(
        self,
        num_games: int = 10,
        *,
        deterministic: bool = True,
    ) -> dict[str, Any]:
        summary = self._new_collection_summary()
        for game_index in range(1, num_games + 1):
            episode_summary = self._run_episode(
                "heuristic",
                deterministic_policy=deterministic,
                collect_trajectories=False,
                update_counters=False,
                game_index=game_index,
            )
            self._accumulate_collection_summary(summary, episode_summary)

        summary["buffer_size"] = len(self.buffer)
        return self._finalize_collection_summary(summary)

    def update(self) -> Optional[dict[str, float]]:
        if len(self.buffer) == 0:
            return None

        batch = self.buffer.as_batch()
        raw_advantages = batch["advantages"].astype(np.float32, copy=True)
        raw_returns = batch["returns"].astype(np.float32, copy=False)
        raw_values = batch["values"].astype(np.float32, copy=False)
        advantages = raw_advantages.copy()
        if len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        num_samples = len(batch["actions"])
        minibatch_size = min(self.minibatch_size, num_samples)
        metric_history: list[dict[str, float]] = []

        self.model.train()
        for _ in range(self.update_epochs):
            permutation = np.random.permutation(num_samples)
            for start in range(0, num_samples, minibatch_size):
                batch_indices = permutation[start : start + minibatch_size]
                observation_tensor = observations_to_tensor(batch["observations"][batch_indices], device=self.device)
                action_mask_tensor = torch.as_tensor(
                    batch["action_masks"][batch_indices],
                    dtype=torch.bool,
                    device=self.device,
                )
                action_tensor = torch.as_tensor(batch["actions"][batch_indices], dtype=torch.int64, device=self.device)
                old_log_prob_tensor = torch.as_tensor(
                    batch["log_probs"][batch_indices],
                    dtype=torch.float32,
                    device=self.device,
                )
                return_tensor = torch.as_tensor(batch["returns"][batch_indices], dtype=torch.float32, device=self.device)
                advantage_tensor = torch.as_tensor(advantages[batch_indices], dtype=torch.float32, device=self.device)
                heuristic_prior_scores, heuristic_beta_values = self._build_heuristic_prior_batch(
                    observations=batch["observations"][batch_indices],
                    action_masks=batch["action_masks"][batch_indices],
                    player_ids=batch["player_ids"][batch_indices],
                    heuristic_betas=batch["heuristic_betas"][batch_indices],
                )

                logits, values = self.model(observation_tensor)
                masked_logits = mix_policy_logits_with_prior(
                    logits,
                    action_mask_tensor,
                    prior_scores=heuristic_prior_scores,
                    beta=heuristic_beta_values,
                )
                distribution = Categorical(logits=masked_logits)
                new_log_probs = distribution.log_prob(action_tensor)
                entropy = distribution.entropy().mean()

                ratios = torch.exp(new_log_probs - old_log_prob_tensor)
                unclipped = ratios * advantage_tensor
                clipped = torch.clamp(ratios, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * advantage_tensor
                policy_loss = -torch.min(unclipped, clipped).mean()
                value_loss = F.mse_loss(values, return_tensor)
                total_loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy
                approx_kl = (old_log_prob_tensor - new_log_probs).mean()
                clip_fraction = (torch.abs(ratios - 1.0) > self.clip_epsilon).float().mean()

                self.optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.optimizer.step()

                metric_history.append(
                    {
                        "total_loss": float(total_loss.item()),
                        "policy_loss": float(policy_loss.item()),
                        "value_loss": float(value_loss.item()),
                        "entropy": float(entropy.item()),
                        "approx_kl": float(approx_kl.item()),
                        "clip_fraction": float(clip_fraction.item()),
                    }
                )

        self.training_updates += 1
        self.buffer.clear()
        return {
            "total_loss": float(np.mean([metric["total_loss"] for metric in metric_history])),
            "policy_loss": float(np.mean([metric["policy_loss"] for metric in metric_history])),
            "value_loss": float(np.mean([metric["value_loss"] for metric in metric_history])),
            "entropy": float(np.mean([metric["entropy"] for metric in metric_history])),
            "approx_kl": float(np.mean([metric["approx_kl"] for metric in metric_history])),
            "clip_fraction": float(np.mean([metric["clip_fraction"] for metric in metric_history])),
            "advantage_mean": float(np.mean(raw_advantages)),
            "advantage_std": float(np.std(raw_advantages)),
            "return_mean": float(np.mean(raw_returns)),
            "return_std": float(np.std(raw_returns)),
            "value_prediction_mean": float(np.mean(raw_values)),
            "value_prediction_std": float(np.std(raw_values)),
            "num_samples": float(num_samples),
        }

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state": self.model.state_dict(),
                "optimizer_state": self.optimizer.state_dict(),
                "board_size": self.board_size,
                "gamma": self.gamma,
                "gae_lambda": self.gae_lambda,
                "clip_epsilon": self.clip_epsilon,
                "entropy_coef": self.entropy_coef,
                "value_coef": self.value_coef,
                "update_epochs": self.update_epochs,
                "minibatch_size": self.minibatch_size,
                "training_updates": self.training_updates,
                "self_play_games": self.self_play_games,
                "heuristic_pool_games": self.heuristic_pool_games,
                "training_games": self.training_games,
                "use_heuristic_prior": self.use_heuristic_prior,
                "heuristic_prior_beta_start": self.heuristic_prior_beta_start,
                "heuristic_prior_beta_end": self.heuristic_prior_beta_end,
                "heuristic_prior_decay_updates": self.heuristic_prior_decay_updates,
                "heuristic_prior_score_clip": self.heuristic_prior_score_clip,
            },
            path,
        )

    def load(self, path: str | Path) -> None:
        checkpoint = torch.load(Path(path), map_location=self.device)
        if "model_state" not in checkpoint:
            raise ValueError("Checkpoint format is not compatible with PPOTrainer.")

        self.model.load_state_dict(checkpoint["model_state"])
        optimizer_state = checkpoint.get("optimizer_state")
        if optimizer_state is not None:
            self.optimizer.load_state_dict(optimizer_state)
        self.training_updates = int(checkpoint.get("training_updates", 0))
        self.self_play_games = int(checkpoint.get("self_play_games", 0))
        self.heuristic_pool_games = int(checkpoint.get("heuristic_pool_games", 0))
        self.training_games = int(checkpoint.get("training_games", self.self_play_games + self.heuristic_pool_games))
        self.model.to(self.device)
        self.model.eval()

    def load_if_exists(self, path: str | Path) -> bool:
        path = Path(path)
        if not path.exists():
            return False

        try:
            self.load(path)
        except Exception as exc:
            print(f"checkpoint_load_failed={exc}")
            return False
        return True

    def _run_episode(
        self,
        opponent_type: str,
        *,
        deterministic_policy: bool,
        collect_trajectories: bool,
        update_counters: bool,
        game_index: int,
    ) -> dict[str, Any]:
        if opponent_type not in _SUPPORTED_OPPONENTS:
            raise ValueError(f"Unsupported opponent_type: {opponent_type}")

        env = GomokuEnv(board_size=self.board_size)
        observation, info = env.reset()
        terminated = False
        truncated = False
        move_count = 0

        ppo_players, opponent_player, matchup_label, ppo_color_label = self._build_episode_participants(
            opponent_type,
            game_index=game_index,
        )
        trajectories = {player: [] for player in ppo_players}
        episode_summary = self._new_episode_summary(
            opponent_type=opponent_type,
            matchup_label=matchup_label,
            ppo_color_label=ppo_color_label,
            ppo_players=ppo_players,
        )

        self.model.eval()
        while not (terminated or truncated):
            current_player = int(info["current_player"])
            valid_actions = list(info["valid_actions"])
            if current_player in trajectories:
                policy_observation = np.asarray(observation, dtype=np.int8)
                raw_board = np.asarray(info.get("board", orient_board_to_player(policy_observation, current_player)), dtype=np.int8)
                sampled_action = self._sample_policy_action(
                    observation=policy_observation,
                    board=raw_board,
                    player_id=current_player,
                    valid_actions=valid_actions,
                    deterministic=deterministic_policy,
                )
                trajectories[current_player].append(
                    PPORolloutStep(
                        observation=policy_observation.copy(),
                        action_mask=sampled_action.action_mask.copy(),
                        action=sampled_action.action,
                        log_prob=sampled_action.log_prob,
                        value=sampled_action.value,
                        player_id=sampled_action.player_id,
                        heuristic_beta=sampled_action.heuristic_beta,
                    )
                )
                self._record_policy_stats(episode_summary, sampled_action)
                self._record_move_quality(
                    episode_summary,
                    board=policy_observation,
                    valid_actions=valid_actions,
                    action=sampled_action.action,
                    ply_index=move_count,
                )
                action = sampled_action.action
            elif opponent_player is not None:
                action = opponent_player.select_action(observation, valid_actions, info)
            else:
                raise RuntimeError("Opponent player was not configured for a non-self episode.")

            observation, _, terminated, truncated, info = env.step(action)
            move_count += 1

        winner = int(info.get("winner", DRAW))
        reason = str(info.get("reason", "unknown"))
        invalid_actor = int(info["current_player"]) if reason == "invalid_move" else None
        if collect_trajectories:
            for controlled_player, steps in trajectories.items():
                self._store_trajectory(
                    steps=steps,
                    controlled_player=controlled_player,
                    winner=winner,
                    reason=reason,
                    opponent_type=opponent_type,
                )

        env.close()

        if update_counters:
            self.training_games += 1
            if opponent_type == "heuristic":
                self.heuristic_pool_games += 1
            else:
                self.self_play_games += 1

        return self._finalize_episode_summary(
            episode_summary,
            move_count=move_count,
            winner=winner,
            reason=reason,
            invalid_actor=invalid_actor,
            trajectories=trajectories,
        )

    def _store_trajectory(
        self,
        *,
        steps: Sequence[PPORolloutStep],
        controlled_player: int,
        winner: int,
        reason: str,
        opponent_type: str,
    ) -> int:
        if not steps:
            return 0

        terminal_reward = self._terminal_reward_for_player(
            controlled_player=controlled_player,
            winner=winner,
            reason=reason,
        )
        final_index = len(steps) - 1

        for index, step in enumerate(steps):
            self.buffer.add(
                observation=step.observation,
                action_mask=step.action_mask,
                action=step.action,
                log_prob=step.log_prob,
                value=step.value,
                player_id=step.player_id,
                heuristic_beta=step.heuristic_beta,
                reward=terminal_reward if index == final_index else 0.0,
                done=index == final_index,
                info={
                    "controlled_player": controlled_player,
                    "winner": winner,
                    "reason": reason,
                    "opponent_type": opponent_type,
                },
            )

        return self.buffer.finish_episode(gamma=self.gamma, gae_lambda=self.gae_lambda, use_gae=self.use_gae)

    def _build_episode_participants(
        self,
        opponent_type: str,
        *,
        game_index: int,
    ) -> tuple[tuple[int, ...], HeuristicPlayer | None, str, str]:
        if opponent_type == "self":
            return (BLACK, WHITE), None, "ppo_vs_ppo", "black+white"

        if opponent_type != "heuristic":
            raise ValueError(f"Unsupported opponent_type: {opponent_type}")

        ppo_color = BLACK if game_index % 2 == 1 else WHITE
        heuristic_player = HeuristicPlayer(
            name="HeuristicPoolOpponent",
            seed=int(self.rng.integers(0, 2**31 - 1)),
        )
        heuristic_player.set_player_id(opponent(ppo_color))
        heuristic_player.reset()
        return (ppo_color,), heuristic_player, "ppo_vs_heuristic", self._player_label(ppo_color)

    def _sample_policy_action(
        self,
        *,
        observation: np.ndarray,
        board: np.ndarray,
        player_id: int,
        valid_actions: list[int],
        deterministic: bool,
    ) -> _SampledPolicyAction:
        action_mask = valid_actions_to_mask(valid_actions, self.action_space_size)
        observation_tensor = observations_to_tensor(observation, device=self.device)
        action_mask_tensor = torch.as_tensor(action_mask[None, :], dtype=torch.bool, device=self.device)
        heuristic_beta = self.current_heuristic_prior_beta()
        prior_scores = self._build_heuristic_prior_scores(
            board=board,
            valid_actions=valid_actions,
            player_id=player_id,
        )

        with torch.no_grad():
            logits, value = self.model(observation_tensor)
            masked_logits = mix_policy_logits_with_prior(
                logits,
                action_mask_tensor,
                prior_scores=prior_scores,
                beta=heuristic_beta if prior_scores is not None else 0.0,
            )
            distribution = Categorical(logits=masked_logits)
            probabilities = torch.softmax(masked_logits, dim=-1).squeeze(0)
            if deterministic:
                action_tensor = torch.argmax(masked_logits, dim=-1)
            else:
                action_tensor = distribution.sample()
            log_prob_tensor = distribution.log_prob(action_tensor)
            selected_action_prob = probabilities[int(action_tensor.item())].item()
            top_k = min(3, probabilities.shape[0])
            top3_action_prob_sum = torch.topk(probabilities, k=top_k).values.sum().item()
            top1_action_prob = probabilities.max().item()
            entropy = distribution.entropy().item()

        return _SampledPolicyAction(
            action=int(action_tensor.item()),
            log_prob=float(log_prob_tensor.item()),
            value=float(value.squeeze(0).item()),
            action_mask=action_mask,
            player_id=int(player_id),
            heuristic_beta=float(heuristic_beta if prior_scores is not None else 0.0),
            selected_action_prob=float(selected_action_prob),
            top1_action_prob=float(top1_action_prob),
            top3_action_prob_sum=float(top3_action_prob_sum),
            entropy=float(entropy),
            valid_action_count=len(valid_actions),
        )

    def _build_heuristic_prior_scores(
        self,
        *,
        board: np.ndarray,
        valid_actions: Sequence[int],
        player_id: int,
    ) -> np.ndarray | None:
        if not self.use_heuristic_prior or player_id not in {BLACK, WHITE}:
            return None

        heuristic_scores = build_heuristic_scores(np.asarray(board, dtype=np.int8), valid_actions, int(player_id))
        return normalize_policy_prior_scores(
            heuristic_scores,
            valid_actions,
            clip_value=self.heuristic_prior_score_clip,
        )

    def _build_heuristic_prior_batch(
        self,
        *,
        observations: np.ndarray,
        action_masks: np.ndarray,
        player_ids: np.ndarray,
        heuristic_betas: np.ndarray,
    ) -> tuple[np.ndarray | None, float | np.ndarray]:
        if not self.use_heuristic_prior:
            return None, 0.0

        prior_rows: list[np.ndarray] = []
        has_nonzero_prior = False
        beta_values = np.asarray(heuristic_betas, dtype=np.float32)
        for observation, action_mask, player_id, beta in zip(observations, action_masks, player_ids, beta_values):
            beta_value = float(beta)
            if beta_value <= 0.0 or int(player_id) not in {BLACK, WHITE}:
                prior_rows.append(np.zeros(self.action_space_size, dtype=np.float32))
                continue

            has_nonzero_prior = True
            raw_board = orient_board_to_player(np.asarray(observation, dtype=np.int8), int(player_id))
            valid_actions = np.flatnonzero(np.asarray(action_mask, dtype=bool)).astype(int).tolist()
            prior_scores = self._build_heuristic_prior_scores(
                board=raw_board,
                valid_actions=valid_actions,
                player_id=int(player_id),
            )
            prior_rows.append(
                np.zeros(self.action_space_size, dtype=np.float32)
                if prior_scores is None
                else prior_scores.astype(np.float32, copy=False)
            )

        if not has_nonzero_prior:
            return None, 0.0
        return np.stack(prior_rows).astype(np.float32), beta_values

    def _sample_opponent_type(self) -> str:
        opponent_types = list(self.opponent_probs.keys())
        probabilities = np.asarray([self.opponent_probs[name] for name in opponent_types], dtype=np.float64)
        sampled_index = int(self.rng.choice(len(opponent_types), p=probabilities))
        return opponent_types[sampled_index]

    def _record_policy_stats(self, episode_summary: dict[str, Any], sampled_action: _SampledPolicyAction) -> None:
        episode_summary["ppo_action_count"] += 1
        episode_summary["selected_action_prob_sum"] += sampled_action.selected_action_prob
        episode_summary["top1_action_prob_sum"] += sampled_action.top1_action_prob
        episode_summary["top3_action_prob_sum"] += sampled_action.top3_action_prob_sum
        episode_summary["action_entropy_sum"] += sampled_action.entropy
        episode_summary["valid_action_count_sum"] += float(sampled_action.valid_action_count)

    def _record_move_quality(
        self,
        episode_summary: dict[str, Any],
        *,
        board: np.ndarray,
        valid_actions: list[int],
        action: int,
        ply_index: int,
    ) -> None:
        chosen_attack = _analyze_move(board, action, BLACK)
        chosen_defense = _analyze_move(board, action, WHITE)
        immediate_win_available = False
        immediate_block_available = False
        for candidate_action in valid_actions:
            if not immediate_win_available and _analyze_move(board, candidate_action, BLACK).immediate_win:
                immediate_win_available = True
            if not immediate_block_available and _analyze_move(board, candidate_action, WHITE).immediate_win:
                immediate_block_available = True
            if immediate_win_available and immediate_block_available:
                break

        row, col = action_to_pos(action, self.board_size)
        center = self.board_size // 2
        is_center_near = abs(row - center) <= self.center_radius and abs(col - center) <= self.center_radius
        if is_center_near:
            episode_summary["center_near_count"] += 1
        if ply_index < self.opening_ply_cutoff:
            episode_summary["opening_action_count"] += 1
            if is_center_near:
                episode_summary["opening_center_near_count"] += 1

        if immediate_win_available:
            episode_summary["immediate_win_available_count"] += 1
            if chosen_attack.immediate_win:
                episode_summary["immediate_win_chosen_count"] += 1
        if immediate_block_available:
            episode_summary["immediate_block_available_count"] += 1
            if chosen_defense.immediate_win:
                episode_summary["immediate_block_chosen_count"] += 1
        if chosen_attack.open_three_count > 0:
            episode_summary["open_three_created_count"] += 1
        if chosen_attack.open_four_count > 0 or chosen_attack.strong_four_count > 0:
            episode_summary["strong_four_created_count"] += 1

    def _new_episode_summary(
        self,
        *,
        opponent_type: str,
        matchup_label: str,
        ppo_color_label: str,
        ppo_players: tuple[int, ...],
    ) -> dict[str, Any]:
        return {
            "opponent_type": opponent_type,
            "matchup": matchup_label,
            "ppo_color": ppo_color_label,
            "ppo_players": ppo_players,
            "ppo_action_count": 0,
            "selected_action_prob_sum": 0.0,
            "top1_action_prob_sum": 0.0,
            "top3_action_prob_sum": 0.0,
            "action_entropy_sum": 0.0,
            "valid_action_count_sum": 0.0,
            "center_near_count": 0,
            "opening_action_count": 0,
            "opening_center_near_count": 0,
            "immediate_win_available_count": 0,
            "immediate_win_chosen_count": 0,
            "immediate_block_available_count": 0,
            "immediate_block_chosen_count": 0,
            "open_three_created_count": 0,
            "strong_four_created_count": 0,
        }

    def _finalize_episode_summary(
        self,
        episode_summary: dict[str, Any],
        *,
        move_count: int,
        winner: int,
        reason: str,
        invalid_actor: int | None,
        trajectories: dict[int, list[_CollectedStep]],
    ) -> dict[str, Any]:
        controlled_players = tuple(sorted(trajectories.keys()))
        ppo_black_episodes = 1 if BLACK in controlled_players else 0
        ppo_white_episodes = 1 if WHITE in controlled_players else 0
        if winner == DRAW:
            ppo_wins = 0
            ppo_losses = 0
            ppo_draws = len(controlled_players)
        else:
            ppo_wins = sum(1 for player in controlled_players if player == winner)
            ppo_losses = len(controlled_players) - ppo_wins
            ppo_draws = 0

        ppo_action_count = max(int(episode_summary["ppo_action_count"]), 1)
        return {
            **episode_summary,
            "moves": move_count,
            "winner": winner,
            "winner_label": self._winner_label(winner),
            "reason": reason,
            "black_win": int(winner == BLACK),
            "white_win": int(winner == WHITE),
            "draw": int(winner == DRAW),
            "ppo_black_episodes": ppo_black_episodes,
            "ppo_white_episodes": ppo_white_episodes,
            "ppo_wins": ppo_wins,
            "ppo_losses": ppo_losses,
            "ppo_draws": ppo_draws,
            "invalid_move": int(reason == "invalid_move"),
            "ppo_invalid_move": int(reason == "invalid_move" and invalid_actor in controlled_players),
            "opponent_invalid_move": int(reason == "invalid_move" and invalid_actor not in controlled_players),
            "selected_action_prob_mean": episode_summary["selected_action_prob_sum"] / ppo_action_count,
            "top1_action_prob_mean": episode_summary["top1_action_prob_sum"] / ppo_action_count,
            "top3_action_prob_sum_mean": episode_summary["top3_action_prob_sum"] / ppo_action_count,
            "action_entropy_mean": episode_summary["action_entropy_sum"] / ppo_action_count,
            "valid_action_count_mean": episode_summary["valid_action_count_sum"] / ppo_action_count,
            "center_near_ratio": episode_summary["center_near_count"] / ppo_action_count,
            "opening_center_near_ratio": (
                episode_summary["opening_center_near_count"] / episode_summary["opening_action_count"]
                if episode_summary["opening_action_count"] > 0
                else 0.0
            ),
            "immediate_win_pick_rate": (
                episode_summary["immediate_win_chosen_count"] / episode_summary["immediate_win_available_count"]
                if episode_summary["immediate_win_available_count"] > 0
                else 0.0
            ),
            "immediate_block_pick_rate": (
                episode_summary["immediate_block_chosen_count"] / episode_summary["immediate_block_available_count"]
                if episode_summary["immediate_block_available_count"] > 0
                else 0.0
            ),
            "open_three_create_rate": episode_summary["open_three_created_count"] / ppo_action_count,
            "strong_four_create_rate": episode_summary["strong_four_created_count"] / ppo_action_count,
            "ppo_result_count": ppo_wins + ppo_losses + ppo_draws,
            "episode_log": self._format_episode_log(
                matchup=episode_summary["matchup"],
                ppo_color=episode_summary["ppo_color"],
                moves=move_count,
                winner=winner,
                reason=reason,
                ppo_wins=ppo_wins,
                ppo_losses=ppo_losses,
                ppo_draws=ppo_draws,
            ),
        }

    def _new_collection_summary(self) -> dict[str, Any]:
        return {
            "games": 0,
            "moves": 0,
            "black_wins": 0,
            "white_wins": 0,
            "draws": 0,
            "self_games": 0,
            "heuristic_games": 0,
            "self_moves": 0,
            "heuristic_moves": 0,
            "ppo_black_episodes": 0,
            "ppo_white_episodes": 0,
            "self_ppo_black_episodes": 0,
            "self_ppo_white_episodes": 0,
            "heuristic_ppo_black_episodes": 0,
            "heuristic_ppo_white_episodes": 0,
            "ppo_wins": 0,
            "ppo_losses": 0,
            "ppo_draws": 0,
            "self_ppo_wins": 0,
            "self_ppo_losses": 0,
            "self_ppo_draws": 0,
            "heuristic_ppo_wins": 0,
            "heuristic_ppo_losses": 0,
            "heuristic_ppo_draws": 0,
            "invalid_move_games": 0,
            "self_invalid_move_games": 0,
            "heuristic_invalid_move_games": 0,
            "ppo_invalid_moves": 0,
            "opponent_invalid_moves": 0,
            "ppo_action_count": 0,
            "selected_action_prob_sum": 0.0,
            "top1_action_prob_sum": 0.0,
            "top3_action_prob_sum": 0.0,
            "action_entropy_sum": 0.0,
            "valid_action_count_sum": 0.0,
            "center_near_count": 0,
            "opening_action_count": 0,
            "opening_center_near_count": 0,
            "immediate_win_available_count": 0,
            "immediate_win_chosen_count": 0,
            "immediate_block_available_count": 0,
            "immediate_block_chosen_count": 0,
            "open_three_created_count": 0,
            "strong_four_created_count": 0,
            "buffer_size": len(self.buffer),
            "episode_logs": [],
        }

    def _accumulate_collection_summary(self, summary: dict[str, Any], episode_summary: dict[str, Any]) -> None:
        prefix = str(episode_summary["opponent_type"])
        summary["games"] += 1
        summary["moves"] += int(episode_summary["moves"])
        summary["black_wins"] += int(episode_summary["black_win"])
        summary["white_wins"] += int(episode_summary["white_win"])
        summary["draws"] += int(episode_summary["draw"])
        summary[f"{prefix}_games"] += 1
        summary[f"{prefix}_moves"] += int(episode_summary["moves"])

        summary["ppo_black_episodes"] += int(episode_summary["ppo_black_episodes"])
        summary["ppo_white_episodes"] += int(episode_summary["ppo_white_episodes"])
        summary[f"{prefix}_ppo_black_episodes"] += int(episode_summary["ppo_black_episodes"])
        summary[f"{prefix}_ppo_white_episodes"] += int(episode_summary["ppo_white_episodes"])
        summary["ppo_wins"] += int(episode_summary["ppo_wins"])
        summary["ppo_losses"] += int(episode_summary["ppo_losses"])
        summary["ppo_draws"] += int(episode_summary["ppo_draws"])
        summary[f"{prefix}_ppo_wins"] += int(episode_summary["ppo_wins"])
        summary[f"{prefix}_ppo_losses"] += int(episode_summary["ppo_losses"])
        summary[f"{prefix}_ppo_draws"] += int(episode_summary["ppo_draws"])
        summary["invalid_move_games"] += int(episode_summary["invalid_move"])
        summary[f"{prefix}_invalid_move_games"] += int(episode_summary["invalid_move"])
        summary["ppo_invalid_moves"] += int(episode_summary["ppo_invalid_move"])
        summary["opponent_invalid_moves"] += int(episode_summary["opponent_invalid_move"])

        summary["ppo_action_count"] += int(episode_summary["ppo_action_count"])
        summary["selected_action_prob_sum"] += float(episode_summary["selected_action_prob_sum"])
        summary["top1_action_prob_sum"] += float(episode_summary["top1_action_prob_sum"])
        summary["top3_action_prob_sum"] += float(episode_summary["top3_action_prob_sum"])
        summary["action_entropy_sum"] += float(episode_summary["action_entropy_sum"])
        summary["valid_action_count_sum"] += float(episode_summary["valid_action_count_sum"])
        summary["center_near_count"] += int(episode_summary["center_near_count"])
        summary["opening_action_count"] += int(episode_summary["opening_action_count"])
        summary["opening_center_near_count"] += int(episode_summary["opening_center_near_count"])
        summary["immediate_win_available_count"] += int(episode_summary["immediate_win_available_count"])
        summary["immediate_win_chosen_count"] += int(episode_summary["immediate_win_chosen_count"])
        summary["immediate_block_available_count"] += int(episode_summary["immediate_block_available_count"])
        summary["immediate_block_chosen_count"] += int(episode_summary["immediate_block_chosen_count"])
        summary["open_three_created_count"] += int(episode_summary["open_three_created_count"])
        summary["strong_four_created_count"] += int(episode_summary["strong_four_created_count"])
        summary["episode_logs"].append(str(episode_summary["episode_log"]))

    def _finalize_collection_summary(self, summary: dict[str, Any]) -> dict[str, Any]:
        games = max(int(summary["games"]), 1)
        ppo_result_count = summary["ppo_wins"] + summary["ppo_losses"] + summary["ppo_draws"]
        ppo_action_count = max(int(summary["ppo_action_count"]), 1)

        summary["avg_moves"] = float(summary["moves"] / games)
        summary["self_avg_moves"] = float(summary["self_moves"] / summary["self_games"]) if summary["self_games"] > 0 else 0.0
        summary["heuristic_avg_moves"] = (
            float(summary["heuristic_moves"] / summary["heuristic_games"]) if summary["heuristic_games"] > 0 else 0.0
        )
        summary["ppo_result_count"] = int(ppo_result_count)
        summary["ppo_win_rate"] = float(summary["ppo_wins"] / ppo_result_count) if ppo_result_count > 0 else 0.0
        summary["ppo_loss_rate"] = float(summary["ppo_losses"] / ppo_result_count) if ppo_result_count > 0 else 0.0
        summary["ppo_draw_rate"] = float(summary["ppo_draws"] / ppo_result_count) if ppo_result_count > 0 else 0.0
        for prefix in ("self", "heuristic"):
            result_count = summary[f"{prefix}_ppo_wins"] + summary[f"{prefix}_ppo_losses"] + summary[f"{prefix}_ppo_draws"]
            summary[f"{prefix}_ppo_result_count"] = int(result_count)
            summary[f"{prefix}_ppo_win_rate"] = float(summary[f"{prefix}_ppo_wins"] / result_count) if result_count > 0 else 0.0
            summary[f"{prefix}_ppo_loss_rate"] = (
                float(summary[f"{prefix}_ppo_losses"] / result_count) if result_count > 0 else 0.0
            )
            summary[f"{prefix}_ppo_draw_rate"] = (
                float(summary[f"{prefix}_ppo_draws"] / result_count) if result_count > 0 else 0.0
            )

        summary["invalid_move_rate"] = float(summary["invalid_move_games"] / games)
        summary["selected_action_prob_mean"] = float(summary["selected_action_prob_sum"] / ppo_action_count)
        summary["top1_action_prob_mean"] = float(summary["top1_action_prob_sum"] / ppo_action_count)
        summary["top3_action_prob_sum_mean"] = float(summary["top3_action_prob_sum"] / ppo_action_count)
        summary["action_entropy_mean"] = float(summary["action_entropy_sum"] / ppo_action_count)
        summary["valid_action_count_mean"] = float(summary["valid_action_count_sum"] / ppo_action_count)
        summary["center_near_ratio"] = float(summary["center_near_count"] / ppo_action_count)
        summary["opening_center_near_ratio"] = (
            float(summary["opening_center_near_count"] / summary["opening_action_count"])
            if summary["opening_action_count"] > 0
            else 0.0
        )
        summary["immediate_win_pick_rate"] = (
            float(summary["immediate_win_chosen_count"] / summary["immediate_win_available_count"])
            if summary["immediate_win_available_count"] > 0
            else 0.0
        )
        summary["immediate_block_pick_rate"] = (
            float(summary["immediate_block_chosen_count"] / summary["immediate_block_available_count"])
            if summary["immediate_block_available_count"] > 0
            else 0.0
        )
        summary["open_three_create_rate"] = float(summary["open_three_created_count"] / ppo_action_count)
        summary["strong_four_create_rate"] = float(summary["strong_four_created_count"] / ppo_action_count)
        return summary

    def _terminal_reward_for_player(
        self,
        *,
        controlled_player: int,
        winner: int,
        reason: str,
    ) -> float:
        if reason == "invalid_move":
            return WIN_REWARD if winner == controlled_player else INVALID_MOVE_PENALTY
        if winner == DRAW:
            return DRAW_REWARD
        return WIN_REWARD if winner == controlled_player else -WIN_REWARD

    @staticmethod
    def _resolve_opponent_probs(
        opponent_mode: str,
        opponent_probs: Optional[dict[str, float]],
    ) -> dict[str, float]:
        if opponent_mode not in _SUPPORTED_OPPONENT_MODES:
            raise ValueError(
                f"Unsupported opponent_mode '{opponent_mode}'. "
                f"Expected one of {sorted(_SUPPORTED_OPPONENT_MODES)}."
            )

        resolved = {"self": 1.0} if opponent_mode == "self_play_only" else {"self": 0.7, "heuristic": 0.3}
        if opponent_probs is not None:
            resolved = {str(name): float(prob) for name, prob in opponent_probs.items()}

        unknown = set(resolved) - _SUPPORTED_OPPONENTS
        if unknown:
            raise ValueError(f"Unsupported opponent types in opponent_probs: {sorted(unknown)}")

        total = float(sum(resolved.values()))
        if total <= 0.0:
            raise ValueError("opponent_probs must sum to a positive value.")
        if any(prob < 0.0 for prob in resolved.values()):
            raise ValueError("opponent_probs cannot contain negative probabilities.")

        normalized = {name: prob / total for name, prob in resolved.items() if prob > 0.0}
        if not normalized:
            raise ValueError("At least one opponent probability must be greater than zero.")
        if opponent_mode == "self_play_only" and set(normalized) != {"self"}:
            raise ValueError("self_play_only mode only supports the 'self' opponent.")

        return normalized

    @staticmethod
    def _player_label(player: int) -> str:
        return "black" if player == BLACK else "white"

    @staticmethod
    def _winner_label(winner: int) -> str:
        if winner == BLACK:
            return "black"
        if winner == WHITE:
            return "white"
        return "draw"

    @staticmethod
    def _format_episode_log(
        *,
        matchup: str,
        ppo_color: str,
        moves: int,
        winner: int,
        reason: str,
        ppo_wins: int,
        ppo_losses: int,
        ppo_draws: int,
    ) -> str:
        return (
            f"matchup={matchup} "
            f"ppo_color={ppo_color} "
            f"moves={moves} "
            f"winner={PPOTrainer._winner_label(winner)} "
            f"reason={reason} "
            f"ppo_result=win:{ppo_wins}/lose:{ppo_losses}/draw:{ppo_draws}"
        )
