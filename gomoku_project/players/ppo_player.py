from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from gomoku_project.core.constants import BLACK, WHITE
from gomoku_project.core.utils import orient_board_to_player
from gomoku_project.players.base import BasePlayer
from gomoku_project.players.heuristic_player import build_heuristic_scores
from gomoku_project.rl.ppo_network import (
    mix_policy_logits_with_prior,
    normalize_policy_prior_scores,
    observations_to_tensor,
    valid_actions_to_mask,
)

try:
    import torch
    from torch.distributions import Categorical
except ImportError:  # pragma: no cover - optional dependency fallback
    torch = None
    Categorical = None


@dataclass(frozen=True)
class PPOPlayerAction:
    action: int
    action_mask: np.ndarray
    log_prob: float
    value: float
    player_id: int
    heuristic_beta: float


class PPOPlayer(BasePlayer):
    def __init__(
        self,
        model: Any = None,
        *,
        device: str = "cpu",
        deterministic: bool = True,
        use_heuristic_prior: bool = False,
        heuristic_prior_beta: float = 0.0,
        heuristic_prior_score_clip: float = 3.0,
        name: str = "PPOPlayer",
    ) -> None:
        super().__init__(name=name)
        self.model = model
        self.device = device
        self.deterministic = deterministic
        self.use_heuristic_prior = bool(use_heuristic_prior)
        self.heuristic_prior_beta = max(float(heuristic_prior_beta), 0.0)
        self.heuristic_prior_score_clip = max(float(heuristic_prior_score_clip), 0.0)
        self.last_action_info: PPOPlayerAction | None = None

        if self.model is not None and torch is not None and hasattr(self.model, "to"):
            self.model.to(self.device)
            self.model.eval()

    def reset(self) -> None:
        self.last_policy_target = None
        self.last_action_info = None
        if self.model is not None and torch is not None and hasattr(self.model, "eval"):
            self.model.eval()

    def select_action(
        self,
        observation: np.ndarray,
        valid_actions: Sequence[int],
        info: dict[str, Any],
    ) -> int:
        if not valid_actions:
            self.last_action_info = None
            return 0

        action_space_size = observation.shape[0] * observation.shape[1]
        mask = valid_actions_to_mask(valid_actions, action_space_size)
        current_player = info.get("current_player", self.player_id)
        player_id = int(current_player) if current_player in {BLACK, WHITE} else 0
        if self.model is None or torch is None or Categorical is None:
            action = int(np.random.choice(valid_actions))
            self.last_policy_target = self._one_hot(action, action_space_size)
            self.last_action_info = PPOPlayerAction(
                action=action,
                action_mask=mask.copy(),
                log_prob=float(np.log(1.0 / len(valid_actions))),
                value=0.0,
                player_id=player_id,
                heuristic_beta=0.0,
            )
            return action

        observation_tensor = observations_to_tensor(np.asarray(observation, dtype=np.int8), device=self.device)
        mask_tensor = torch.as_tensor(mask[None, :], dtype=torch.bool, device=self.device)
        prior_scores = None
        if self.use_heuristic_prior and self.heuristic_prior_beta > 0.0 and player_id in {BLACK, WHITE}:
            raw_board = info.get("board")
            if isinstance(raw_board, np.ndarray):
                board = np.asarray(raw_board, dtype=np.int8)
            else:
                board = orient_board_to_player(np.asarray(observation, dtype=np.int8), player_id)
            heuristic_scores = build_heuristic_scores(board, valid_actions, player_id)
            prior_scores = normalize_policy_prior_scores(
                heuristic_scores,
                valid_actions,
                clip_value=self.heuristic_prior_score_clip,
            )

        with torch.no_grad():
            logits, value = self.model(observation_tensor)
            masked_logits = mix_policy_logits_with_prior(
                logits,
                mask_tensor,
                prior_scores=prior_scores,
                beta=self.heuristic_prior_beta if prior_scores is not None else 0.0,
            )
            probabilities = torch.softmax(masked_logits, dim=-1).squeeze(0)
            self.last_policy_target = probabilities.detach().cpu().numpy().astype(np.float32)
            distribution = Categorical(logits=masked_logits)
            if self.deterministic:
                action_tensor = torch.argmax(masked_logits, dim=-1)
            else:
                action_tensor = distribution.sample()

            action = int(action_tensor.item())
            self.last_action_info = PPOPlayerAction(
                action=action,
                action_mask=mask.copy(),
                log_prob=float(distribution.log_prob(action_tensor).item()),
                value=float(value.squeeze(0).item()),
                player_id=player_id,
                heuristic_beta=float(self.heuristic_prior_beta if prior_scores is not None else 0.0),
            )

        return action

    @staticmethod
    def _one_hot(action: int, action_space_size: int) -> np.ndarray:
        target = np.zeros(action_space_size, dtype=np.float32)
        target[int(action)] = 1.0
        return target
