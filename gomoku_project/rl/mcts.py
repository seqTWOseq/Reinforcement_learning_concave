from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import torch

from gomoku_project.core.constants import DRAW
from gomoku_project.core.game import GomokuGame
from gomoku_project.core.utils import opponent


@dataclass
class SearchNode:
    game: GomokuGame
    to_play: int
    prior: float
    visit_count: int = 0
    value_sum: float = 0.0
    children: dict[int, "SearchNode"] = field(default_factory=dict)

    def expanded(self) -> bool:
        return bool(self.children)

    def value(self) -> float:
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count


class AlphaZeroMCTS:
    def __init__(
        self,
        model: torch.nn.Module,
        *,
        device: str = "cpu",
        num_simulations: int = 32,
        c_puct: float = 1.5,
        dirichlet_alpha: float = 0.03,
        dirichlet_epsilon: float = 0.25,
    ) -> None:
        self.model = model
        self.device = device
        self.num_simulations = num_simulations
        self.c_puct = c_puct
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_epsilon = dirichlet_epsilon

    def run(self, game: GomokuGame, *, add_exploration_noise: bool = False) -> np.ndarray:
        root = SearchNode(game=game.clone(), to_play=game.current_player, prior=1.0)
        policy, _ = self._evaluate(root)
        self._expand(root, policy)

        if add_exploration_noise and root.children:
            self._add_exploration_noise(root)

        for _ in range(self.num_simulations):
            node = root
            search_path = [node]

            while node.expanded():
                _, node = self._select_child(node)
                search_path.append(node)

            policy, value = self._evaluate(node)
            if not node.game.done:
                self._expand(node, policy)

            self._backpropagate(search_path, value, node.to_play)

        action_space = game.board_size * game.board_size
        visit_probs = np.zeros(action_space, dtype=np.float32)
        total_visits = sum(child.visit_count for child in root.children.values())
        if total_visits == 0:
            valid_actions = game.get_valid_actions()
            if valid_actions:
                uniform = 1.0 / len(valid_actions)
                for action in valid_actions:
                    visit_probs[action] = uniform
            return visit_probs

        for action, child in root.children.items():
            visit_probs[action] = child.visit_count / total_visits
        return visit_probs

    def _evaluate(self, node: SearchNode) -> tuple[np.ndarray, float]:
        action_space = node.game.board_size * node.game.board_size

        if node.game.done:
            return np.zeros(action_space, dtype=np.float32), self._terminal_value(node.game.winner, node.to_play)

        observation = node.game.get_observation(player=node.to_play)
        observation_tensor = torch.as_tensor(
            observation,
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0).unsqueeze(0)

        self.model.eval()
        with torch.no_grad():
            policy_logits, value = self.model(observation_tensor)

        logits = policy_logits[0].detach().cpu().numpy().astype(np.float32)
        value_scalar = float(value[0].item())

        valid_actions = node.game.get_valid_actions()
        policy = np.zeros(action_space, dtype=np.float32)
        if valid_actions:
            masked_logits = logits[valid_actions]
            masked_logits = masked_logits - np.max(masked_logits)
            exp_logits = np.exp(masked_logits)
            total = float(exp_logits.sum())
            if total <= 0.0 or not np.isfinite(total):
                uniform = 1.0 / len(valid_actions)
                for action in valid_actions:
                    policy[action] = uniform
            else:
                policy[valid_actions] = exp_logits / total
        return policy, value_scalar

    def _expand(self, node: SearchNode, policy: np.ndarray) -> None:
        if node.game.done:
            return

        for action in node.game.get_valid_actions():
            child_game = node.game.clone()
            child_game.step(action)
            child_to_play = opponent(node.to_play)
            node.children[action] = SearchNode(
                game=child_game,
                to_play=child_to_play,
                prior=float(policy[action]),
            )

    def _select_child(self, node: SearchNode) -> tuple[int, SearchNode]:
        best_score = -float("inf")
        best_action = -1
        best_child: SearchNode | None = None
        parent_visits = max(1, node.visit_count)

        for action, child in node.children.items():
            prior_score = self.c_puct * child.prior * math.sqrt(parent_visits) / (1 + child.visit_count)
            value_score = -child.value()
            score = value_score + prior_score
            if score > best_score:
                best_score = score
                best_action = action
                best_child = child

        if best_child is None:
            raise RuntimeError("MCTS selection failed because no child node was available.")
        return best_action, best_child

    def _backpropagate(self, search_path: list[SearchNode], value: float, leaf_to_play: int) -> None:
        for node in reversed(search_path):
            if node.to_play == leaf_to_play:
                node.value_sum += value
            else:
                node.value_sum -= value
            node.visit_count += 1

    def _add_exploration_noise(self, node: SearchNode) -> None:
        actions = list(node.children.keys())
        noise = np.random.dirichlet([self.dirichlet_alpha] * len(actions)).astype(np.float32)
        for action, sample in zip(actions, noise):
            child = node.children[action]
            child.prior = (1.0 - self.dirichlet_epsilon) * child.prior + self.dirichlet_epsilon * float(sample)

    @staticmethod
    def _terminal_value(winner: int | None, to_play: int) -> float:
        if winner is None or winner == DRAW:
            return 0.0
        return 1.0 if winner == to_play else -1.0
