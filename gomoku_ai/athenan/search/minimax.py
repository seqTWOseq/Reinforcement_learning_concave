"""Inference-focused stronger search for Athenan."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

import torch
from torch import nn

from gomoku_ai.athenan.network import AthenanValueNet
from gomoku_ai.athenan.search.move_ordering import order_actions
from gomoku_ai.athenan.search.tactical_rules import (
    find_immediate_blocking_actions,
    find_immediate_winning_actions,
    generate_proximity_candidates,
)
from gomoku_ai.athenan.utils import env_to_tensor
from gomoku_ai.common.agents import BaseSearcher, SearchResult
from gomoku_ai.env import BLACK, DRAW, GomokuEnv, WHITE


@dataclass(frozen=True)
class _NodeResult:
    """Internal negamax node payload."""

    value: float
    pv: list[int]
    nodes: int
    depth_reached: int


class AthenanInferenceSearcher(BaseSearcher):
    """Inference-oriented stronger negamax searcher.

    Differences from training search:
    - deeper default depth
    - iterative deepening support
    - optional wall-clock time budget
    - preserves the same `SearchResult` contract
    """

    def __init__(
        self,
        *,
        model: AthenanValueNet | nn.Module | None = None,
        max_depth: int = 4,
        candidate_limit: int | None = 64,
        candidate_radius: int = 2,
        use_alpha_beta: bool = True,
        iterative_deepening: bool = True,
        time_budget_sec: float | None = None,
        device: str | None = None,
    ) -> None:
        if max_depth <= 0:
            raise ValueError("max_depth must be positive.")
        if candidate_limit is not None and candidate_limit <= 0:
            raise ValueError("candidate_limit must be positive when provided.")
        if candidate_radius < 0:
            raise ValueError("candidate_radius must be non-negative.")
        if time_budget_sec is not None and time_budget_sec < 0.0:
            raise ValueError("time_budget_sec must be non-negative when provided.")
        if not isinstance(use_alpha_beta, bool):
            raise ValueError("use_alpha_beta must be bool.")
        if not isinstance(iterative_deepening, bool):
            raise ValueError("iterative_deepening must be bool.")

        self.model = model
        self.max_depth = int(max_depth)
        self.candidate_limit = candidate_limit
        self.candidate_radius = int(candidate_radius)
        self.use_alpha_beta = use_alpha_beta
        self.iterative_deepening = iterative_deepening
        self.time_budget_sec = time_budget_sec
        self.device = device

    def search(self, env: GomokuEnv) -> SearchResult:
        """Run stronger search and return best move info under `SearchResult`."""

        model_was_training: bool | None = None
        if self.model is not None:
            model_was_training = self.model.training
            if model_was_training:
                self.model.eval()

        try:
            terminal_value = self._evaluate_terminal(env, perspective_player=env.current_player)
            if terminal_value is not None:
                return SearchResult(
                    best_action=-1,
                    root_value=terminal_value,
                    action_values={},
                    principal_variation=[],
                    nodes=1,
                    depth_reached=0,
                    forced_tactical=False,
                )

            immediate_wins = find_immediate_winning_actions(env)
            if immediate_wins:
                ordered_wins = order_actions(
                    env,
                    immediate_wins,
                    candidate_limit=self.candidate_limit,
                )
                best_action = ordered_wins[0]
                return SearchResult(
                    best_action=best_action,
                    root_value=1.0,
                    action_values={action: 1.0 for action in ordered_wins},
                    principal_variation=[best_action],
                    nodes=max(1, len(ordered_wins)),
                    depth_reached=1,
                    forced_tactical=True,
                )

            immediate_blocks = find_immediate_blocking_actions(env)
            if immediate_blocks:
                root_actions = order_actions(
                    env,
                    immediate_blocks,
                    candidate_limit=self.candidate_limit,
                )
                forced_tactical = True
            else:
                root_actions = self._ordered_candidates(env)
                forced_tactical = False

            if not root_actions:
                raise RuntimeError("No legal candidates were generated for inference search.")

            search_start_time = time.perf_counter()
            depth_schedule = (
                list(range(1, self.max_depth + 1))
                if self.iterative_deepening
                else [self.max_depth]
            )

            best_complete_result: SearchResult | None = None
            for depth in depth_schedule:
                if self._is_time_budget_exceeded(search_start_time) and best_complete_result is not None:
                    break
                depth_result = self._search_root_at_depth(
                    env,
                    root_actions=root_actions,
                    depth=depth,
                    search_start_time=search_start_time,
                    forced_tactical=forced_tactical,
                )
                if depth_result is None:
                    break
                best_complete_result = depth_result
                if abs(depth_result.root_value) >= 1.0 - 1e-8:
                    break

            if best_complete_result is not None:
                return best_complete_result

            # Fallback when time budget is too strict to complete any depth.
            fallback_action = root_actions[0]
            fallback_value = self._evaluate_leaf(env, perspective_player=env.current_player)
            return SearchResult(
                best_action=fallback_action,
                root_value=float(fallback_value),
                action_values={fallback_action: float(fallback_value)},
                principal_variation=[fallback_action],
                nodes=1,
                depth_reached=1,
                forced_tactical=forced_tactical,
            )
        finally:
            if self.model is not None and model_was_training:
                self.model.train()

    def _search_root_at_depth(
        self,
        env: GomokuEnv,
        *,
        root_actions: list[int],
        depth: int,
        search_start_time: float,
        forced_tactical: bool,
    ) -> SearchResult | None:
        """Run one full root negamax pass at the requested depth."""

        root_player = int(env.current_player)
        opponent = self._opponent_of(root_player)
        alpha = -math.inf
        beta = math.inf
        nodes = 1
        depth_reached = 0
        best_action = root_actions[0]
        best_value = -math.inf
        best_child_pv: list[int] = []
        action_values: dict[int, float] = {}

        for action in root_actions:
            if self._is_time_budget_exceeded(search_start_time):
                break

            child_env = env.clone()
            child_env.apply_move(action)
            child_result = self._negamax(
                child_env,
                depth_remaining=depth - 1,
                alpha=-beta,
                beta=-alpha,
                perspective_player=opponent,
                ply=1,
                search_start_time=search_start_time,
            )
            if child_result is None:
                break

            value_from_root = -child_result.value
            action_values[action] = float(value_from_root)
            nodes += child_result.nodes
            depth_reached = max(depth_reached, child_result.depth_reached)

            if value_from_root > best_value:
                best_value = value_from_root
                best_action = action
                best_child_pv = child_result.pv

            if self.use_alpha_beta:
                alpha = max(alpha, value_from_root)
                if alpha >= beta:
                    break

        if not action_values:
            return None
        if best_value == -math.inf:
            best_value = 0.0

        principal_variation = [best_action] + best_child_pv
        return SearchResult(
            best_action=best_action,
            root_value=float(best_value),
            action_values=action_values,
            principal_variation=principal_variation if principal_variation else [best_action],
            nodes=nodes,
            depth_reached=max(1, min(depth, depth_reached)),
            forced_tactical=forced_tactical,
        )

    def _negamax(
        self,
        env: GomokuEnv,
        *,
        depth_remaining: int,
        alpha: float,
        beta: float,
        perspective_player: int,
        ply: int,
        search_start_time: float,
    ) -> _NodeResult | None:
        """Depth-limited negamax with optional alpha-beta and time cut."""

        if self._is_time_budget_exceeded(search_start_time):
            return None

        terminal_value = self._evaluate_terminal(env, perspective_player=perspective_player)
        if terminal_value is not None:
            return _NodeResult(value=terminal_value, pv=[], nodes=1, depth_reached=ply)

        if depth_remaining <= 0:
            return _NodeResult(
                value=self._evaluate_leaf(env, perspective_player=perspective_player),
                pv=[],
                nodes=1,
                depth_reached=ply,
            )

        actions = self._ordered_candidates(env)
        if not actions:
            return _NodeResult(
                value=self._evaluate_leaf(env, perspective_player=perspective_player),
                pv=[],
                nodes=1,
                depth_reached=ply,
            )

        best_value = -math.inf
        best_pv: list[int] = []
        nodes = 1
        depth_reached = ply
        local_alpha = alpha

        for action in actions:
            if self._is_time_budget_exceeded(search_start_time):
                break

            child_env = env.clone()
            child_env.apply_move(action)
            child_result = self._negamax(
                child_env,
                depth_remaining=depth_remaining - 1,
                alpha=-beta,
                beta=-local_alpha,
                perspective_player=self._opponent_of(perspective_player),
                ply=ply + 1,
                search_start_time=search_start_time,
            )
            if child_result is None:
                break

            value = -child_result.value
            nodes += child_result.nodes
            depth_reached = max(depth_reached, child_result.depth_reached)

            if value > best_value:
                best_value = value
                best_pv = [action] + child_result.pv

            if self.use_alpha_beta:
                local_alpha = max(local_alpha, value)
                if local_alpha >= beta:
                    break

        if best_value == -math.inf:
            # Could happen under strict time budget; return no result so caller can fallback.
            return None

        return _NodeResult(
            value=float(best_value),
            pv=best_pv,
            nodes=nodes,
            depth_reached=depth_reached,
        )

    def _ordered_candidates(self, env: GomokuEnv) -> list[int]:
        candidates = generate_proximity_candidates(
            env,
            radius=self.candidate_radius,
            candidate_limit=None,
        )
        return order_actions(env, candidates, candidate_limit=self.candidate_limit)

    def _evaluate_terminal(self, env: GomokuEnv, *, perspective_player: int) -> float | None:
        if not env.done:
            return None
        if env.winner == DRAW:
            return 0.0
        if env.winner not in {BLACK, WHITE}:
            raise ValueError(f"Unexpected terminal winner value: {env.winner!r}.")
        return 1.0 if env.winner == perspective_player else -1.0

    def _evaluate_leaf(self, env: GomokuEnv, *, perspective_player: int) -> float:
        terminal_value = self._evaluate_terminal(env, perspective_player=perspective_player)
        if terminal_value is not None:
            return terminal_value
        if self.model is None:
            return 0.0

        tensor = env_to_tensor(env, device=self._resolve_model_device())
        with torch.no_grad():
            value = float(self.model(tensor).reshape(-1)[0].item())
        if int(env.current_player) != int(perspective_player):
            value = -value
        return value

    def _is_time_budget_exceeded(self, search_start_time: float) -> bool:
        if self.time_budget_sec is None:
            return False
        return (time.perf_counter() - search_start_time) >= self.time_budget_sec

    def _resolve_model_device(self) -> str | torch.device:
        if self.device is not None:
            return self.device
        if self.model is None:
            return "cpu"
        try:
            return next(self.model.parameters()).device
        except StopIteration:
            return "cpu"

    @staticmethod
    def _opponent_of(player: int) -> int:
        return WHITE if player == BLACK else BLACK
