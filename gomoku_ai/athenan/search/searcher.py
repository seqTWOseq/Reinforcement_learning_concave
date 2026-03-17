"""Value-guided depth-limited negamax search for Athenan."""

from __future__ import annotations

import math
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
    """Internal negamax return payload."""

    value: float
    pv: list[int]
    nodes: int  # visited search nodes including the current node
    depth_reached: int


class AthenanSearcher(BaseSearcher):
    """Depth-limited negamax searcher with optional alpha-beta pruning.

    Value sign convention:
    - each node value is from that node's current-player perspective
    - parent receives `-child_value` (negamax symmetry)
    - root `action_values` are always in root-player perspective
    - `nodes` counts visited negamax nodes, including root and leaf nodes
    """

    def __init__(
        self,
        *,
        model: AthenanValueNet | nn.Module | None = None,
        max_depth: int = 2,
        candidate_limit: int | None = 32,
        candidate_radius: int = 2,
        use_alpha_beta: bool = True,
        device: str | None = None,
    ) -> None:
        if max_depth <= 0:
            raise ValueError("max_depth must be positive.")
        if candidate_limit is not None and candidate_limit <= 0:
            raise ValueError("candidate_limit must be positive when provided.")
        if candidate_radius < 0:
            raise ValueError("candidate_radius must be non-negative.")
        if not isinstance(use_alpha_beta, bool):
            raise ValueError("use_alpha_beta must be a bool.")

        self.model = model
        self.max_depth = max_depth
        self.candidate_limit = candidate_limit
        self.candidate_radius = candidate_radius
        self.use_alpha_beta = use_alpha_beta
        self.device = device

    def search(self, env: GomokuEnv) -> SearchResult:
        """Search one move and return root-level action values + PV.

        Terminal-root behavior:
        - `best_action = -1` sentinel (no legal move to play)
        - empty `action_values` and empty `principal_variation`
        - `nodes = 1` because only root terminal check is visited
        """

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

            root_player = int(env.current_player)
            if not root_actions:
                raise RuntimeError("No legal candidates were generated for root search.")

            alpha = -math.inf
            beta = math.inf
            nodes = 1
            depth_reached = 0
            best_action = root_actions[0]
            best_value = -math.inf
            best_child_pv: list[int] = []
            action_values: dict[int, float] = {}

            for action in root_actions:
                child_env = env.clone()
                child_env.apply_move(action)
                opponent = self._opponent_of(root_player)
                child_result = self._negamax(
                    child_env,
                    depth_remaining=self.max_depth - 1,
                    alpha=-beta,
                    beta=-alpha,
                    perspective_player=opponent,
                    ply=1,
                )
                value_from_root = -child_result.value
                action_values[action] = value_from_root

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

            if best_value == -math.inf:
                best_value = 0.0
            principal_variation = [best_action] + best_child_pv
            return SearchResult(
                best_action=best_action,
                root_value=float(best_value),
                action_values=action_values,
                principal_variation=principal_variation if principal_variation else [best_action],
                nodes=nodes,
                depth_reached=max(1, depth_reached),
                forced_tactical=forced_tactical,
            )
        finally:
            if self.model is not None and model_was_training:
                self.model.train()

    def _negamax(
        self,
        env: GomokuEnv,
        *,
        depth_remaining: int,
        alpha: float,
        beta: float,
        perspective_player: int,
        ply: int,
    ) -> _NodeResult:
        """Depth-limited negamax with optional alpha-beta pruning."""

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
            child_env = env.clone()
            child_env.apply_move(action)
            child_result = self._negamax(
                child_env,
                depth_remaining=depth_remaining - 1,
                alpha=-beta,
                beta=-local_alpha,
                perspective_player=self._opponent_of(perspective_player),
                ply=ply + 1,
            )
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
            best_value = self._evaluate_leaf(env, perspective_player=perspective_player)

        return _NodeResult(
            value=float(best_value),
            pv=best_pv,
            nodes=nodes,
            depth_reached=depth_reached,
        )

    def _ordered_candidates(self, env: GomokuEnv) -> list[int]:
        """Generate and order candidates using tactical neighborhood heuristic."""

        candidates = generate_proximity_candidates(
            env,
            radius=self.candidate_radius,
            candidate_limit=None,
        )
        ordered_actions = order_actions(
            env,
            candidates,
            candidate_limit=self.candidate_limit,
        )
        return ordered_actions

    def _evaluate_terminal(self, env: GomokuEnv, *, perspective_player: int) -> float | None:
        """Return terminal value in `perspective_player` view, otherwise `None`."""

        if not env.done:
            return None
        if env.winner == DRAW:
            return 0.0
        if env.winner not in {BLACK, WHITE}:
            raise ValueError(f"Unexpected terminal winner value: {env.winner!r}.")
        return 1.0 if env.winner == perspective_player else -1.0

    def _evaluate_leaf(self, env: GomokuEnv, *, perspective_player: int) -> float:
        """Evaluate non-terminal leaf from `perspective_player` perspective.

        Fallback mode:
        - when `self.model is None`, return `0.0` (neutral value)
        - when model exists, always use network inference
        """

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

    def _resolve_model_device(self) -> str | torch.device:
        """Resolve inference device once per leaf without moving model repeatedly."""

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
