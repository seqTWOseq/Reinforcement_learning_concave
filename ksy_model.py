from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

BOARD_SIZE = 15
EMPTY = 0
BLACK = 1
WHITE = 2
DRAW = -1
DIRECTIONS = (
    (0, 1),
    (1, 0),
    (1, 1),
    (1, -1),
)
STONE_SYMBOLS = {
    EMPTY: ".",
    BLACK: "X",
    WHITE: "O",
}


@dataclass(frozen=True)
class MoveResult:
    action: int
    row: int
    col: int
    player: int
    next_player: int | None
    winner: int | None
    done: bool
    reason: str
    move_count: int
    last_move: tuple[int, int] | None

    def as_dict(self) -> dict[str, Any]:
        return {
            "action": self.action,
            "row": self.row,
            "col": self.col,
            "player": self.player,
            "next_player": self.next_player,
            "winner": self.winner,
            "done": self.done,
            "reason": self.reason,
            "move_count": self.move_count,
            "last_move": self.last_move,
        }


class GomokuEnv:
    def __init__(self, board_size: int = BOARD_SIZE) -> None:
        self.board_size = int(board_size)
        self.board: np.ndarray = np.zeros((self.board_size, self.board_size), dtype=np.int8)
        self.current_player: int = BLACK
        self.last_move: tuple[int, int] | None = None
        self.winner: int | None = None
        self.done: bool = False
        self.move_count: int = 0

    def reset(self) -> np.ndarray:
        self.board.fill(EMPTY)
        self.current_player = BLACK
        self.last_move = None
        self.winner = None
        self.done = False
        self.move_count = 0
        return self.board.copy()

    def clone(self) -> "GomokuEnv":
        cloned = GomokuEnv(board_size=self.board_size)
        cloned.board = self.board.copy()
        cloned.current_player = self.current_player
        cloned.last_move = self.last_move
        cloned.winner = self.winner
        cloned.done = self.done
        cloned.move_count = self.move_count
        return cloned

    def action_to_coord(self, action: int) -> tuple[int, int]:
        if not 0 <= int(action) < self.board_size * self.board_size:
            raise ValueError(
                f"Action {action} is out of range for a {self.board_size}x{self.board_size} board."
            )
        return divmod(int(action), self.board_size)

    def coord_to_action(self, row: int, col: int) -> int:
        if not (0 <= int(row) < self.board_size and 0 <= int(col) < self.board_size):
            raise ValueError(
                f"Coordinate {(row, col)} is out of range for a {self.board_size}x{self.board_size} board."
            )
        return int(row) * self.board_size + int(col)

    def get_valid_moves(self) -> np.ndarray:
        if self.done:
            return np.zeros(self.board_size * self.board_size, dtype=bool)
        return (self.board.reshape(-1) == EMPTY).astype(bool, copy=False)

    def is_legal_action(self, action: int) -> bool:
        if self.done or not isinstance(action, (int, np.integer)):
            return False
        try:
            row, col = self.action_to_coord(int(action))
        except ValueError:
            return False
        return bool(self.board[row, col] == EMPTY)

    def get_legal_actions(self) -> list[int]:
        return np.flatnonzero(self.get_valid_moves()).astype(int, copy=False).tolist()

    def apply_move(self, action: int) -> dict[str, Any]:
        if self.done:
            raise RuntimeError("Cannot apply a move after the game has already ended.")

        row, col = self.action_to_coord(action)
        if self.board[row, col] != EMPTY:
            raise ValueError(f"Cell {(row, col)} is already occupied.")

        player = self.current_player
        self.board[row, col] = player
        self.last_move = (row, col)
        self.move_count += 1

        if self.check_win_from_move(row, col, player):
            self.done = True
            self.winner = player
            reason = "win"
        elif self.move_count == self.board_size * self.board_size:
            self.done = True
            self.winner = DRAW
            reason = "draw"
        else:
            self.current_player = self._opponent_of(player)
            reason = "ongoing"

        return MoveResult(
            action=int(action),
            row=row,
            col=col,
            player=player,
            next_player=None if self.done else self.current_player,
            winner=self.winner,
            done=self.done,
            reason=reason,
            move_count=self.move_count,
            last_move=self.last_move,
        ).as_dict()

    def check_win_from_move(self, row: int, col: int, player: int) -> bool:
        if self.board[row, col] != player:
            return False
        for delta_row, delta_col in DIRECTIONS:
            count = 1
            count += self._count_direction(row, col, player, delta_row, delta_col)
            count += self._count_direction(row, col, player, -delta_row, -delta_col)
            if count >= 5:
                return True
        return False

    def is_terminal(self) -> bool:
        return self.done

    def encode_state(self) -> np.ndarray:
        current_player_stones = (self.board == self.current_player).astype(np.float32)
        opponent_stones = (
            (self.board != EMPTY) & (self.board != self.current_player)
        ).astype(np.float32)
        last_move_plane = np.zeros((self.board_size, self.board_size), dtype=np.float32)
        if self.last_move is not None:
            last_row, last_col = self.last_move
            last_move_plane[last_row, last_col] = 1.0
        black_to_move_plane = np.full(
            (self.board_size, self.board_size),
            1.0 if self.current_player == BLACK else 0.0,
            dtype=np.float32,
        )
        return np.stack(
            (
                current_player_stones,
                opponent_stones,
                last_move_plane,
                black_to_move_plane,
            ),
            axis=0,
        ).astype(np.float32, copy=False)

    def render(self) -> str:
        header = "   " + " ".join(f"{column:>2}" for column in range(self.board_size))
        rows = [header]
        for row_index in range(self.board_size):
            stones = " ".join(STONE_SYMBOLS[int(value)] for value in self.board[row_index])
            rows.append(f"{row_index:>2} {stones}")
        return "\n".join(rows)

    def _count_direction(
        self,
        row: int,
        col: int,
        player: int,
        delta_row: int,
        delta_col: int,
    ) -> int:
        count = 0
        next_row = row + delta_row
        next_col = col + delta_col
        while (
            0 <= next_row < self.board_size
            and 0 <= next_col < self.board_size
            and self.board[next_row, next_col] == player
        ):
            count += 1
            next_row += delta_row
            next_col += delta_col
        return count

    @staticmethod
    def _opponent_of(player: int) -> int:
        return WHITE if int(player) == BLACK else BLACK


@dataclass(frozen=True)
class SearchResult:
    best_action: int
    root_value: float
    action_values: dict[int, float]
    principal_variation: list[int]
    nodes: int
    depth_reached: int
    forced_tactical: bool = False
    cutoffs: int = 0
    pruned_branches: int = 0
    tt_hits: int = 0
    tt_stores: int = 0


@dataclass(frozen=True)
class GreedyHeuristicConfig:
    terminal_win_score: float = 1_000_000.0
    opponent_weight: float = 1.1
    five_score: float = 250_000.0
    open_four_score: float = 40_000.0
    closed_four_score: float = 12_000.0
    open_three_score: float = 2_500.0
    closed_three_score: float = 600.0
    open_two_score: float = 120.0
    center_weight: float = 0.15
    connectivity_weight: float = 1.0


@dataclass(frozen=True)
class PatternSummary:
    five: int = 0
    open_four: int = 0
    closed_four: int = 0
    open_three: int = 0
    closed_three: int = 0
    open_two: int = 0

    def weighted_score(self, config: GreedyHeuristicConfig) -> float:
        return float(
            (self.five * config.five_score)
            + (self.open_four * config.open_four_score)
            + (self.closed_four * config.closed_four_score)
            + (self.open_three * config.open_three_score)
            + (self.closed_three * config.closed_three_score)
            + (self.open_two * config.open_two_score)
        )


class GreedyHeuristicEvaluator:
    def __init__(self, config: GreedyHeuristicConfig | None = None) -> None:
        self.config = config or GreedyHeuristicConfig()

    def evaluate_for_player(self, env: GomokuEnv, player: int) -> float:
        if player not in {BLACK, WHITE}:
            raise ValueError(f"player must be BLACK({BLACK}) or WHITE({WHITE}).")

        if env.done:
            if env.winner == DRAW:
                return 0.0
            if env.winner == player:
                return float(self.config.terminal_win_score)
            return float(-self.config.terminal_win_score)

        opponent = WHITE if player == BLACK else BLACK
        player_score = self._score_player(env, player)
        opponent_score = self._score_player(env, opponent)
        return float(player_score - (self.config.opponent_weight * opponent_score))

    def count_patterns_for_player(self, env: GomokuEnv, player: int) -> PatternSummary:
        if player not in {BLACK, WHITE}:
            raise ValueError(f"player must be BLACK({BLACK}) or WHITE({WHITE}).")

        counts = {
            "five": 0,
            "open_four": 0,
            "closed_four": 0,
            "open_three": 0,
            "closed_three": 0,
            "open_two": 0,
        }
        for row in range(env.board_size):
            for col in range(env.board_size):
                if int(env.board[row, col]) != player:
                    continue
                for delta_row, delta_col in DIRECTIONS:
                    prev_row = row - delta_row
                    prev_col = col - delta_col
                    if (
                        0 <= prev_row < env.board_size
                        and 0 <= prev_col < env.board_size
                        and int(env.board[prev_row, prev_col]) == player
                    ):
                        continue

                    length, end_row, end_col = self._line_extent(
                        env, row, col, player, delta_row, delta_col
                    )
                    open_ends = self._count_open_ends(
                        env,
                        prev_row=prev_row,
                        prev_col=prev_col,
                        end_row=end_row,
                        end_col=end_col,
                    )
                    pattern_name = self._classify_contiguous_pattern(length, open_ends)
                    if pattern_name is not None:
                        counts[pattern_name] += 1

        return PatternSummary(**counts)

    def score_patterns_for_player(self, env: GomokuEnv, player: int) -> float:
        return self.count_patterns_for_player(env, player).weighted_score(self.config)

    def score_action_for_player(self, env: GomokuEnv, action: int, player: int) -> float:
        if env.done:
            raise RuntimeError("Cannot score child actions from a terminal position.")
        if player not in {BLACK, WHITE}:
            raise ValueError(f"player must be BLACK({BLACK}) or WHITE({WHITE}).")
        if int(env.current_player) != int(player):
            raise ValueError("score_action_for_player requires player to equal env.current_player.")
        if not env.is_legal_action(action):
            raise ValueError(f"Action {action} is not legal in the current position.")

        child = env.clone()
        child.apply_move(int(action))
        return self.evaluate_for_player(child, player)

    def would_action_win_for_player(self, env: GomokuEnv, action: int, player: int) -> bool:
        if player not in {BLACK, WHITE}:
            raise ValueError(f"player must be BLACK({BLACK}) or WHITE({WHITE}).")
        if env.done or not env.is_legal_action(action):
            return False

        row, col = env.action_to_coord(int(action))
        return self._projected_max_line_length(env, row, col, player) >= 5

    def _score_player(self, env: GomokuEnv, player: int) -> float:
        return float(
            self.score_patterns_for_player(env, player)
            + (self.config.center_weight * self._score_center_control(env, player))
            + (self.config.connectivity_weight * self._score_connectivity(env, player))
        )

    def _score_center_control(self, env: GomokuEnv, player: int) -> float:
        center = (env.board_size - 1) / 2.0
        score = 0.0
        for row, col in np.argwhere(env.board == player):
            distance = abs(float(row) - center) + abs(float(col) - center)
            score += float((env.board_size * 2) - distance)
        return score

    def _score_connectivity(self, env: GomokuEnv, player: int) -> float:
        score = 0.0
        for row, col in np.argwhere(env.board == player):
            neighbors = 0
            for delta_row in (-1, 0, 1):
                for delta_col in (-1, 0, 1):
                    if delta_row == 0 and delta_col == 0:
                        continue
                    next_row = int(row + delta_row)
                    next_col = int(col + delta_col)
                    if 0 <= next_row < env.board_size and 0 <= next_col < env.board_size:
                        if int(env.board[next_row, next_col]) == player:
                            neighbors += 1
            score += neighbors * 0.5
        return score

    def _projected_max_line_length(self, env: GomokuEnv, row: int, col: int, player: int) -> int:
        max_length = 1
        for delta_row, delta_col in DIRECTIONS:
            length = 1
            length += self._count_direction(env, row, col, player, delta_row, delta_col)
            length += self._count_direction(env, row, col, player, -delta_row, -delta_col)
            if length > max_length:
                max_length = length
        return max_length

    @staticmethod
    def _count_open_ends(
        env: GomokuEnv,
        *,
        prev_row: int,
        prev_col: int,
        end_row: int,
        end_col: int,
    ) -> int:
        open_ends = 0
        if (
            0 <= prev_row < env.board_size
            and 0 <= prev_col < env.board_size
            and int(env.board[prev_row, prev_col]) == EMPTY
        ):
            open_ends += 1
        if (
            0 <= end_row < env.board_size
            and 0 <= end_col < env.board_size
            and int(env.board[end_row, end_col]) == EMPTY
        ):
            open_ends += 1
        return open_ends

    @staticmethod
    def _line_extent(
        env: GomokuEnv,
        row: int,
        col: int,
        player: int,
        delta_row: int,
        delta_col: int,
    ) -> tuple[int, int, int]:
        length = 1
        next_row = row + delta_row
        next_col = col + delta_col
        while (
            0 <= next_row < env.board_size
            and 0 <= next_col < env.board_size
            and int(env.board[next_row, next_col]) == player
        ):
            length += 1
            next_row += delta_row
            next_col += delta_col
        return length, next_row, next_col

    @staticmethod
    def _count_direction(
        env: GomokuEnv,
        row: int,
        col: int,
        player: int,
        delta_row: int,
        delta_col: int,
    ) -> int:
        count = 0
        next_row = row + delta_row
        next_col = col + delta_col
        while 0 <= next_row < env.board_size and 0 <= next_col < env.board_size:
            if int(env.board[next_row, next_col]) != player:
                break
            count += 1
            next_row += delta_row
            next_col += delta_col
        return count

    @staticmethod
    def _classify_contiguous_pattern(length: int, open_ends: int) -> str | None:
        if length >= 5:
            return "five"
        if length == 4:
            if open_ends == 2:
                return "open_four"
            if open_ends == 1:
                return "closed_four"
            return None
        if length == 3:
            if open_ends == 2:
                return "open_three"
            if open_ends == 1:
                return "closed_three"
            return None
        if length == 2:
            return "open_two" if open_ends == 2 else None
        return None


def generate_candidate_actions(
    env: GomokuEnv,
    *,
    radius: int = 2,
    max_candidates: int | None = None,
    evaluator: GreedyHeuristicEvaluator | None = None,
    player: int | None = None,
) -> list[int]:
    if radius < 0:
        raise ValueError("radius must be non-negative.")
    if max_candidates is not None and max_candidates <= 0:
        raise ValueError("max_candidates must be positive when provided.")

    legal_actions = env.get_legal_actions()
    if not legal_actions:
        return []

    if _is_board_empty(env):
        center_action = env.coord_to_action(env.board_size // 2, env.board_size // 2)
        ordered_center = [center_action] if env.is_legal_action(center_action) else [legal_actions[0]]
        return ordered_center[:max_candidates] if max_candidates is not None else ordered_center

    candidate_set = _collect_proximity_candidates(env, radius=radius)
    if not candidate_set:
        return order_candidate_actions(
            env,
            legal_actions,
            max_candidates=max_candidates,
            evaluator=evaluator,
            player=player,
        )

    return order_candidate_actions(
        env,
        sorted(candidate_set),
        max_candidates=max_candidates,
        evaluator=evaluator,
        player=player,
    )


def order_candidate_actions(
    env: GomokuEnv,
    actions: list[int] | tuple[int, ...],
    *,
    max_candidates: int | None = None,
    evaluator: GreedyHeuristicEvaluator | None = None,
    player: int | None = None,
) -> list[int]:
    if max_candidates is not None and max_candidates <= 0:
        raise ValueError("max_candidates must be positive when provided.")
    if env.done:
        return []

    player_to_move = int(env.current_player if player is None else player)
    if player_to_move not in {BLACK, WHITE}:
        raise ValueError(f"player must be BLACK({BLACK}) or WHITE({WHITE}).")
    if player is not None and player_to_move != int(env.current_player):
        raise ValueError("order_candidate_actions requires player to equal env.current_player.")

    resolved_actions = _resolve_unique_legal_actions(env, actions)
    if not resolved_actions:
        return []

    resolved_evaluator = _resolve_evaluator(evaluator)
    opponent = WHITE if player_to_move == BLACK else BLACK
    scored_actions = []
    for action in resolved_actions:
        own_win = resolved_evaluator.would_action_win_for_player(env, action, player_to_move)
        block = resolved_evaluator.would_action_win_for_player(env, action, opponent)
        heuristic_score = float(resolved_evaluator.score_action_for_player(env, action, player_to_move))
        scored_actions.append(
            (
                int(action),
                bool(own_win),
                bool(block),
                heuristic_score,
                _center_distance(env, int(action)),
            )
        )

    scored_actions.sort(
        key=lambda item: (
            0 if item[1] else 1,
            0 if item[2] else 1,
            -item[3],
            item[4],
            item[0],
        )
    )
    ordered = [action for action, _, _, _, _ in scored_actions]
    return ordered[:max_candidates] if max_candidates is not None else ordered


def score_candidate_action(
    env: GomokuEnv,
    action: int,
    *,
    evaluator: GreedyHeuristicEvaluator | None = None,
    player: int | None = None,
) -> float:
    if env.done:
        raise ValueError("Cannot score actions on a finished game.")

    player_to_move = int(env.current_player if player is None else player)
    if player_to_move not in {BLACK, WHITE}:
        raise ValueError(f"player must be BLACK({BLACK}) or WHITE({WHITE}).")
    if player is not None and player_to_move != int(env.current_player):
        raise ValueError("score_candidate_action requires player to equal env.current_player.")
    if not env.is_legal_action(action):
        raise ValueError(f"Action {action} is not legal in the current position.")

    resolved_evaluator = _resolve_evaluator(evaluator)
    opponent = WHITE if player_to_move == BLACK else BLACK
    own_win_bonus = (
        float(resolved_evaluator.config.terminal_win_score)
        if resolved_evaluator.would_action_win_for_player(env, action, player_to_move)
        else 0.0
    )
    block_bonus = (
        float(resolved_evaluator.config.terminal_win_score * 0.5)
        if resolved_evaluator.would_action_win_for_player(env, action, opponent)
        else 0.0
    )
    heuristic_score = float(resolved_evaluator.score_action_for_player(env, action, player_to_move))
    return float(own_win_bonus + block_bonus + heuristic_score)


def score_action(
    env: GomokuEnv,
    action: int,
    *,
    player: int | None = None,
) -> float:
    return score_candidate_action(env, action, player=player)


def order_actions(
    env: GomokuEnv,
    actions: list[int] | tuple[int, ...],
    *,
    player: int | None = None,
    candidate_limit: int | None = None,
) -> list[int]:
    return order_candidate_actions(
        env,
        actions,
        max_candidates=candidate_limit,
        player=player,
    )


def find_immediate_winning_actions(
    env: GomokuEnv,
    *,
    player: int | None = None,
    candidate_actions: list[int] | None = None,
) -> list[int]:
    if env.done:
        return []

    player_to_move = env.current_player if player is None else int(player)
    actions = _resolve_action_pool(env, candidate_actions)
    winning_actions: list[int] = []
    for action in actions:
        if _is_immediate_win_for_player(env, action, player_to_move):
            winning_actions.append(action)
    return sorted(winning_actions)


def find_immediate_blocking_actions(
    env: GomokuEnv,
    *,
    defender: int | None = None,
    candidate_actions: list[int] | None = None,
) -> list[int]:
    if env.done:
        return []

    defender_player = env.current_player if defender is None else int(defender)
    attacker = WHITE if defender_player == BLACK else BLACK
    opponent_immediate_wins = find_immediate_winning_actions(
        env,
        player=attacker,
        candidate_actions=candidate_actions,
    )
    return _blocking_actions_from_opponent_immediate_wins(opponent_immediate_wins)


def generate_proximity_candidates(
    env: GomokuEnv,
    *,
    radius: int = 2,
    candidate_limit: int | None = None,
) -> list[int]:
    return generate_candidate_actions(
        env,
        radius=radius,
        max_candidates=candidate_limit,
    )


def apply_forced_tactical_rule(
    env: GomokuEnv,
    *,
    candidate_limit: int | None = None,
) -> SearchResult | None:
    immediate_wins = find_immediate_winning_actions(env)
    if immediate_wins:
        ordered_wins = order_actions(env, immediate_wins, candidate_limit=candidate_limit)
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
        ordered_blocks = order_actions(env, immediate_blocks, candidate_limit=candidate_limit)
        best_action = ordered_blocks[0]
        return SearchResult(
            best_action=best_action,
            root_value=0.0,
            action_values={action: 0.0 for action in ordered_blocks},
            principal_variation=[best_action],
            nodes=max(1, len(ordered_blocks)),
            depth_reached=1,
            forced_tactical=True,
        )

    return None


def _resolve_unique_legal_actions(
    env: GomokuEnv,
    actions: list[int] | tuple[int, ...],
) -> list[int]:
    legal_mask = env.get_valid_moves()
    resolved: list[int] = []
    seen: set[int] = set()
    for action in actions:
        normalized = int(action)
        if normalized in seen:
            continue
        if 0 <= normalized < legal_mask.size and bool(legal_mask[normalized]):
            resolved.append(normalized)
            seen.add(normalized)
    return resolved


def _collect_proximity_candidates(env: GomokuEnv, *, radius: int) -> set[int]:
    legal_mask = env.get_valid_moves()
    occupied_coords = np.argwhere(env.board != EMPTY)
    candidate_set: set[int] = set()
    for row, col in occupied_coords:
        for delta_row in range(-radius, radius + 1):
            for delta_col in range(-radius, radius + 1):
                next_row = int(row + delta_row)
                next_col = int(col + delta_col)
                if not (0 <= next_row < env.board_size and 0 <= next_col < env.board_size):
                    continue
                action = env.coord_to_action(next_row, next_col)
                if bool(legal_mask[action]):
                    candidate_set.add(action)
    return candidate_set


def _is_board_empty(env: GomokuEnv) -> bool:
    return not bool(np.any(env.board != EMPTY))


def _center_distance(env: GomokuEnv, action: int) -> float:
    row, col = env.action_to_coord(action)
    center = (env.board_size - 1) / 2.0
    return float(abs(row - center) + abs(col - center))


def _resolve_evaluator(evaluator: GreedyHeuristicEvaluator | None) -> GreedyHeuristicEvaluator:
    return evaluator if evaluator is not None else GreedyHeuristicEvaluator()


def _resolve_action_pool(env: GomokuEnv, candidate_actions: list[int] | None) -> list[int]:
    legal_moves = np.asarray(env.get_valid_moves(), dtype=bool)
    if candidate_actions is None:
        return np.flatnonzero(legal_moves).astype(int, copy=False).tolist()

    resolved: list[int] = []
    seen: set[int] = set()
    for action in candidate_actions:
        normalized = int(action)
        if normalized in seen:
            continue
        if 0 <= normalized < legal_moves.size and legal_moves[normalized]:
            resolved.append(normalized)
            seen.add(normalized)
    return resolved


def _blocking_actions_from_opponent_immediate_wins(opponent_wins: list[int]) -> list[int]:
    return sorted({int(action) for action in opponent_wins})


def _is_immediate_win_for_player(env: GomokuEnv, action: int, player: int) -> bool:
    return GreedyHeuristicEvaluator().would_action_win_for_player(env, action, int(player))


TTFlag = Literal["EXACT", "LOWER", "UPPER"]


@dataclass(frozen=True)
class NegamaxSearchSummary:
    value: float
    best_action: int | None
    action_values: dict[int, float]
    principal_variation: list[int]
    nodes: int
    depth_reached: int
    cutoffs: int
    pruned_branches: int
    tt_hits: int
    tt_stores: int


@dataclass(frozen=True)
class _NegamaxNodeResult:
    value: float
    pv: list[int]
    nodes: int
    depth_reached: int
    cutoffs: int
    pruned_branches: int
    tt_hits: int
    tt_stores: int


@dataclass(frozen=True)
class TTEntry:
    key: tuple[bytes, int]
    value: float
    depth: int
    flag: TTFlag
    best_action: int | None = None
    pv: tuple[int, ...] = ()


def negamax(
    env: GomokuEnv,
    depth: int,
    evaluator: GreedyHeuristicEvaluator | None = None,
    *,
    radius: int = 2,
    max_candidates: int | None = None,
    use_alpha_beta: bool = True,
    use_transposition_table: bool = True,
    preferred_action: int | None = None,
    transposition_table: dict[tuple[bytes, int], TTEntry] | None = None,
) -> tuple[float, int | None]:
    result = run_negamax_search(
        env,
        depth=depth,
        evaluator=evaluator,
        radius=radius,
        max_candidates=max_candidates,
        use_alpha_beta=use_alpha_beta,
        use_transposition_table=use_transposition_table,
        preferred_action=preferred_action,
        transposition_table=transposition_table,
    )
    return float(result.value), result.best_action


def run_negamax_search(
    env: GomokuEnv,
    *,
    depth: int,
    evaluator: GreedyHeuristicEvaluator | None = None,
    radius: int = 2,
    max_candidates: int | None = None,
    use_alpha_beta: bool = True,
    use_transposition_table: bool = True,
    preferred_action: int | None = None,
    transposition_table: dict[tuple[bytes, int], TTEntry] | None = None,
) -> NegamaxSearchSummary:
    if not isinstance(env, GomokuEnv):
        raise TypeError("env must be a GomokuEnv instance.")
    if depth < 0:
        raise ValueError("depth must be non-negative.")
    if radius < 0:
        raise ValueError("radius must be non-negative.")
    if max_candidates is not None and max_candidates <= 0:
        raise ValueError("max_candidates must be positive when provided.")
    if not isinstance(use_alpha_beta, bool):
        raise ValueError("use_alpha_beta must be a bool.")
    if not isinstance(use_transposition_table, bool):
        raise ValueError("use_transposition_table must be a bool.")

    resolved_evaluator = _resolve_greedy_evaluator(evaluator)
    root_player = int(env.current_player)
    resolved_tt = None if not use_transposition_table else ({} if transposition_table is None else transposition_table)

    terminal_value = _evaluate_terminal_for_player(
        env,
        player_to_move=root_player,
        depth_remaining=depth,
        evaluator=resolved_evaluator,
    )
    if terminal_value is not None:
        return NegamaxSearchSummary(
            value=float(terminal_value),
            best_action=None,
            action_values={},
            principal_variation=[],
            nodes=1,
            depth_reached=0,
            cutoffs=0,
            pruned_branches=0,
            tt_hits=0,
            tt_stores=0,
        )

    if depth == 0:
        return NegamaxSearchSummary(
            value=float(_evaluate_leaf_for_player(env, player=root_player, evaluator=resolved_evaluator)),
            best_action=None,
            action_values={},
            principal_variation=[],
            nodes=1,
            depth_reached=0,
            cutoffs=0,
            pruned_branches=0,
            tt_hits=0,
            tt_stores=0,
        )

    actions = generate_candidate_actions(
        env,
        radius=radius,
        max_candidates=max_candidates,
        evaluator=resolved_evaluator,
    )
    actions = _prioritize_preferred_action(actions, _tt_best_action_for_key(resolved_tt, _make_transposition_key(env)))
    actions = _prioritize_preferred_action(actions, preferred_action)
    if not actions:
        return NegamaxSearchSummary(
            value=float(_evaluate_leaf_for_player(env, player=root_player, evaluator=resolved_evaluator)),
            best_action=None,
            action_values={},
            principal_variation=[],
            nodes=1,
            depth_reached=0,
            cutoffs=0,
            pruned_branches=0,
            tt_hits=0,
            tt_stores=0,
        )

    best_action = actions[0]
    best_value = -math.inf
    best_child_pv: list[int] = []
    action_values: dict[int, float] = {}
    nodes = 1
    depth_reached = 0
    cutoffs = 0
    pruned_branches = 0
    tt_hits = 0
    tt_stores = 0
    alpha = -math.inf
    beta = math.inf
    local_alpha = alpha

    for index, action in enumerate(actions):
        child_env = env.clone()
        child_env.apply_move(action)
        child_result = _negamax_node(
            child_env,
            depth_remaining=depth - 1,
            player_to_move=_opponent_of(root_player),
            evaluator=resolved_evaluator,
            radius=radius,
            max_candidates=max_candidates,
            ply=1,
            alpha=-beta,
            beta=-local_alpha,
            use_alpha_beta=use_alpha_beta,
            use_transposition_table=use_transposition_table,
            transposition_table=resolved_tt,
        )
        value = -child_result.value
        action_values[int(action)] = float(value)
        nodes += child_result.nodes
        depth_reached = max(depth_reached, child_result.depth_reached)
        cutoffs += child_result.cutoffs
        pruned_branches += child_result.pruned_branches
        tt_hits += child_result.tt_hits
        tt_stores += child_result.tt_stores

        if value > best_value:
            best_value = float(value)
            best_action = int(action)
            best_child_pv = child_result.pv

        if use_alpha_beta:
            local_alpha = max(local_alpha, value)
            if local_alpha >= beta:
                cutoffs += 1
                pruned_branches += max(0, len(actions) - index - 1)
                break

    principal_variation = [best_action] + best_child_pv
    tt_stores += _store_tt_entry(
        resolved_tt,
        _make_transposition_key(env),
        TTEntry(
            key=_make_transposition_key(env),
            value=float(best_value),
            depth=depth,
            flag="EXACT",
            best_action=int(best_action),
            pv=tuple(principal_variation),
        ),
    )
    return NegamaxSearchSummary(
        value=float(best_value),
        best_action=int(best_action),
        action_values=action_values,
        principal_variation=principal_variation,
        nodes=nodes,
        depth_reached=max(1, depth_reached),
        cutoffs=cutoffs,
        pruned_branches=pruned_branches,
        tt_hits=tt_hits,
        tt_stores=tt_stores,
    )


def run_iterative_deepening_search(
    env: GomokuEnv,
    *,
    max_depth: int,
    evaluator: GreedyHeuristicEvaluator | None = None,
    radius: int = 2,
    max_candidates: int | None = None,
    use_alpha_beta: bool = True,
    use_transposition_table: bool = True,
    transposition_table: dict[tuple[bytes, int], TTEntry] | None = None,
) -> NegamaxSearchSummary:
    if max_depth < 0:
        raise ValueError("max_depth must be non-negative.")

    if max_depth == 0:
        return run_negamax_search(
            env,
            depth=0,
            evaluator=evaluator,
            radius=radius,
            max_candidates=max_candidates,
            use_alpha_beta=use_alpha_beta,
            use_transposition_table=use_transposition_table,
            transposition_table=transposition_table,
        )

    best_completed_result: NegamaxSearchSummary | None = None
    preferred_action: int | None = None
    shared_tt = None if not use_transposition_table else ({} if transposition_table is None else transposition_table)
    for depth in range(1, max_depth + 1):
        result = run_negamax_search(
            env,
            depth=depth,
            evaluator=evaluator,
            radius=radius,
            max_candidates=max_candidates,
            use_alpha_beta=use_alpha_beta,
            use_transposition_table=use_transposition_table,
            preferred_action=preferred_action,
            transposition_table=shared_tt,
        )
        best_completed_result = result
        preferred_action = result.best_action

    if best_completed_result is None:
        raise RuntimeError("Iterative deepening did not complete any search depth.")
    return best_completed_result


def _negamax_node(
    env: GomokuEnv,
    *,
    depth_remaining: int,
    player_to_move: int,
    evaluator: GreedyHeuristicEvaluator,
    radius: int,
    max_candidates: int | None,
    ply: int,
    alpha: float,
    beta: float,
    use_alpha_beta: bool,
    use_transposition_table: bool,
    transposition_table: dict[tuple[bytes, int], TTEntry] | None,
) -> _NegamaxNodeResult:
    original_alpha = alpha
    original_beta = beta
    key = _make_transposition_key(env)
    tt_entry = None if transposition_table is None else transposition_table.get(key)
    tt_hits = 1 if tt_entry is not None else 0
    tt_stores = 0

    terminal_value = _evaluate_terminal_for_player(
        env,
        player_to_move=player_to_move,
        depth_remaining=depth_remaining,
        evaluator=evaluator,
    )
    if terminal_value is not None:
        tt_stores += _store_tt_entry(
            transposition_table if use_transposition_table else None,
            key,
            TTEntry(key=key, value=float(terminal_value), depth=depth_remaining, flag="EXACT"),
        )
        return _NegamaxNodeResult(
            value=float(terminal_value),
            pv=[],
            nodes=1,
            depth_reached=ply,
            cutoffs=0,
            pruned_branches=0,
            tt_hits=tt_hits,
            tt_stores=tt_stores,
        )

    if int(env.current_player) != int(player_to_move):
        raise ValueError("player_to_move must match env.current_player on non-terminal nodes.")

    if use_transposition_table and tt_entry is not None and tt_entry.depth >= depth_remaining:
        if tt_entry.flag == "EXACT":
            return _NegamaxNodeResult(
                value=float(tt_entry.value),
                pv=_pv_from_tt_entry(tt_entry),
                nodes=1,
                depth_reached=ply,
                cutoffs=0,
                pruned_branches=0,
                tt_hits=tt_hits,
                tt_stores=tt_stores,
            )
        if tt_entry.flag == "LOWER":
            alpha = max(alpha, float(tt_entry.value))
        else:
            beta = min(beta, float(tt_entry.value))
        if alpha >= beta:
            return _NegamaxNodeResult(
                value=float(tt_entry.value),
                pv=_pv_from_tt_entry(tt_entry),
                nodes=1,
                depth_reached=ply,
                cutoffs=1,
                pruned_branches=0,
                tt_hits=tt_hits,
                tt_stores=tt_stores,
            )

    if depth_remaining <= 0:
        leaf_value = float(_evaluate_leaf_for_player(env, player=player_to_move, evaluator=evaluator))
        tt_stores += _store_tt_entry(
            transposition_table if use_transposition_table else None,
            key,
            TTEntry(key=key, value=leaf_value, depth=depth_remaining, flag="EXACT"),
        )
        return _NegamaxNodeResult(
            value=leaf_value,
            pv=[],
            nodes=1,
            depth_reached=ply,
            cutoffs=0,
            pruned_branches=0,
            tt_hits=tt_hits,
            tt_stores=tt_stores,
        )

    actions = generate_candidate_actions(
        env,
        radius=radius,
        max_candidates=max_candidates,
        evaluator=evaluator,
    )
    actions = _prioritize_preferred_action(actions, None if tt_entry is None else tt_entry.best_action)
    if not actions:
        leaf_value = float(_evaluate_leaf_for_player(env, player=player_to_move, evaluator=evaluator))
        tt_stores += _store_tt_entry(
            transposition_table if use_transposition_table else None,
            key,
            TTEntry(key=key, value=leaf_value, depth=depth_remaining, flag="EXACT"),
        )
        return _NegamaxNodeResult(
            value=leaf_value,
            pv=[],
            nodes=1,
            depth_reached=ply,
            cutoffs=0,
            pruned_branches=0,
            tt_hits=tt_hits,
            tt_stores=tt_stores,
        )

    best_value = -math.inf
    best_pv: list[int] = []
    nodes = 1
    depth_reached = ply
    cutoffs = 0
    pruned_branches = 0
    local_alpha = alpha

    for index, action in enumerate(actions):
        child_env = env.clone()
        child_env.apply_move(action)
        child_result = _negamax_node(
            child_env,
            depth_remaining=depth_remaining - 1,
            player_to_move=_opponent_of(player_to_move),
            evaluator=evaluator,
            radius=radius,
            max_candidates=max_candidates,
            ply=ply + 1,
            alpha=-beta,
            beta=-local_alpha,
            use_alpha_beta=use_alpha_beta,
            use_transposition_table=use_transposition_table,
            transposition_table=transposition_table,
        )
        value = -child_result.value
        nodes += child_result.nodes
        depth_reached = max(depth_reached, child_result.depth_reached)
        cutoffs += child_result.cutoffs
        pruned_branches += child_result.pruned_branches
        tt_hits += child_result.tt_hits
        tt_stores += child_result.tt_stores

        if value > best_value:
            best_value = float(value)
            best_pv = [int(action)] + child_result.pv

        if use_alpha_beta:
            local_alpha = max(local_alpha, value)
            if local_alpha >= beta:
                cutoffs += 1
                pruned_branches += max(0, len(actions) - index - 1)
                break

    best_action = None if not best_pv else int(best_pv[0])
    tt_flag = _resolve_tt_flag(best_value, original_alpha, original_beta)
    tt_stores += _store_tt_entry(
        transposition_table if use_transposition_table else None,
        key,
        TTEntry(
            key=key,
            value=float(best_value),
            depth=depth_remaining,
            flag=tt_flag,
            best_action=best_action,
            pv=tuple(best_pv),
        ),
    )

    return _NegamaxNodeResult(
        value=float(best_value),
        pv=best_pv,
        nodes=nodes,
        depth_reached=depth_reached,
        cutoffs=cutoffs,
        pruned_branches=pruned_branches,
        tt_hits=tt_hits,
        tt_stores=tt_stores,
    )


def _evaluate_leaf_for_player(
    env: GomokuEnv,
    *,
    player: int,
    evaluator: GreedyHeuristicEvaluator,
) -> float:
    return float(evaluator.evaluate_for_player(env, int(player)))


def _evaluate_terminal_for_player(
    env: GomokuEnv,
    *,
    player_to_move: int,
    depth_remaining: int,
    evaluator: GreedyHeuristicEvaluator,
) -> float | None:
    if not env.done:
        return None
    if env.winner == DRAW:
        return 0.0
    if env.winner not in {BLACK, WHITE}:
        raise ValueError(f"Unexpected terminal winner value: {env.winner!r}.")

    base = float(evaluator.config.terminal_win_score + max(0, depth_remaining))
    return base if int(env.winner) == int(player_to_move) else -base


def _resolve_greedy_evaluator(
    evaluator: GreedyHeuristicEvaluator | None,
) -> GreedyHeuristicEvaluator:
    return evaluator if evaluator is not None else GreedyHeuristicEvaluator()


def _opponent_of(player: int) -> int:
    return WHITE if int(player) == BLACK else BLACK


def _prioritize_preferred_action(
    actions: list[int],
    preferred_action: int | None,
) -> list[int]:
    if preferred_action is None or preferred_action not in actions:
        return actions
    if actions and actions[0] == preferred_action:
        return actions
    return [int(preferred_action)] + [action for action in actions if int(action) != int(preferred_action)]


def _make_transposition_key(env: GomokuEnv) -> tuple[bytes, int]:
    return env.board.tobytes(), int(env.current_player)


def _resolve_tt_flag(value: float, alpha: float, beta: float) -> TTFlag:
    if value <= alpha:
        return "UPPER"
    if value >= beta:
        return "LOWER"
    return "EXACT"


def _pv_from_tt_entry(entry: TTEntry) -> list[int]:
    if entry.pv:
        return [int(action) for action in entry.pv]
    if entry.best_action is None:
        return []
    return [int(entry.best_action)]


def _store_tt_entry(
    transposition_table: dict[tuple[bytes, int], TTEntry] | None,
    key: tuple[bytes, int],
    entry: TTEntry,
) -> int:
    if transposition_table is None:
        return 0
    existing = transposition_table.get(key)
    if existing is not None:
        if existing.depth > entry.depth:
            return 0
        if existing.depth == entry.depth and existing.flag == "EXACT" and entry.flag != "EXACT":
            return 0
    transposition_table[key] = entry
    return 1


def _tt_best_action_for_key(
    transposition_table: dict[tuple[bytes, int], TTEntry] | None,
    key: tuple[bytes, int],
) -> int | None:
    if transposition_table is None:
        return None
    entry = transposition_table.get(key)
    if entry is None:
        return None
    return None if entry.best_action is None else int(entry.best_action)


class AthenanGreedyHeuristicAgent:
    def __init__(
        self,
        evaluator: GreedyHeuristicEvaluator | None = None,
        *,
        candidate_radius: int = 2,
        max_candidates: int | None = None,
    ) -> None:
        self.evaluator = evaluator or GreedyHeuristicEvaluator()
        self.candidate_radius = int(candidate_radius)
        self.max_candidates = max_candidates

    def select_action(self, env: GomokuEnv) -> int:
        if not isinstance(env, GomokuEnv):
            raise TypeError("env must be a GomokuEnv instance.")

        candidate_actions = generate_candidate_actions(
            env,
            radius=self.candidate_radius,
            max_candidates=self.max_candidates,
            evaluator=self.evaluator,
        )
        if not candidate_actions:
            raise RuntimeError("No legal moves are available in the current position.")
        return candidate_actions[0]


class AthenanNegamaxSearcher:
    def __init__(
        self,
        *,
        evaluator: GreedyHeuristicEvaluator | None = None,
        max_depth: int = 2,
        candidate_radius: int = 2,
        max_candidates: int | None = None,
        use_alpha_beta: bool = True,
        use_iterative_deepening: bool = False,
        use_transposition_table: bool = True,
    ) -> None:
        if max_depth < 0:
            raise ValueError("max_depth must be non-negative.")
        if candidate_radius < 0:
            raise ValueError("candidate_radius must be non-negative.")
        if max_candidates is not None and max_candidates <= 0:
            raise ValueError("max_candidates must be positive when provided.")
        if not isinstance(use_alpha_beta, bool):
            raise ValueError("use_alpha_beta must be a bool.")
        if not isinstance(use_iterative_deepening, bool):
            raise ValueError("use_iterative_deepening must be a bool.")
        if not isinstance(use_transposition_table, bool):
            raise ValueError("use_transposition_table must be a bool.")

        self.evaluator = evaluator
        self.max_depth = int(max_depth)
        self.candidate_radius = int(candidate_radius)
        self.max_candidates = max_candidates
        self.use_alpha_beta = use_alpha_beta
        self.use_iterative_deepening = use_iterative_deepening
        self.use_transposition_table = use_transposition_table

    def search(self, env: GomokuEnv) -> SearchResult:
        if self.use_iterative_deepening:
            result = run_iterative_deepening_search(
                env,
                max_depth=self.max_depth,
                evaluator=self.evaluator,
                radius=self.candidate_radius,
                max_candidates=self.max_candidates,
                use_alpha_beta=self.use_alpha_beta,
                use_transposition_table=self.use_transposition_table,
            )
        else:
            result = run_negamax_search(
                env,
                depth=self.max_depth,
                evaluator=self.evaluator,
                radius=self.candidate_radius,
                max_candidates=self.max_candidates,
                use_alpha_beta=self.use_alpha_beta,
                use_transposition_table=self.use_transposition_table,
            )
        best_action = -1 if result.best_action is None else int(result.best_action)
        return SearchResult(
            best_action=best_action,
            root_value=float(result.value),
            action_values=dict(result.action_values),
            principal_variation=list(result.principal_variation),
            nodes=int(result.nodes),
            depth_reached=int(result.depth_reached),
            forced_tactical=False,
            cutoffs=int(result.cutoffs),
            pruned_branches=int(result.pruned_branches),
            tt_hits=int(result.tt_hits),
            tt_stores=int(result.tt_stores),
        )

    def select_action(self, env: GomokuEnv) -> int:
        return int(self.search(env).best_action)


class NegamaxAthenanAgent:
    def __init__(
        self,
        *,
        name: str = "Negamax-Athenan",
        evaluator: GreedyHeuristicEvaluator | None = None,
        max_depth: int = 4,
        candidate_radius: int = 4,
        max_candidates: int | None = 16,
        use_alpha_beta: bool = True,
        use_iterative_deepening: bool = True,
        use_transposition_table: bool = True,
    ) -> None:
        self.name = name
        self.searcher = AthenanNegamaxSearcher(
            evaluator=evaluator,
            max_depth=max_depth,
            candidate_radius=candidate_radius,
            max_candidates=max_candidates,
            use_alpha_beta=use_alpha_beta,
            use_iterative_deepening=use_iterative_deepening,
            use_transposition_table=use_transposition_table,
        )

    def search(self, state: np.ndarray, player_id: int) -> SearchResult:
        env = build_env_from_state(state, player_id=player_id)
        return self.searcher.search(env)

    def select_action(self, state: np.ndarray, player_id: int) -> int:
        result = self.search(state, player_id=player_id)
        if result.best_action != -1:
            return int(result.best_action)

        board = np.asarray(state)
        valid = np.flatnonzero(board.reshape(-1) == EMPTY)
        return int(valid[0]) if valid.size > 0 else 0


def build_env_from_state(state: np.ndarray, *, player_id: int) -> GomokuEnv:
    board = np.asarray(state, dtype=np.int8)
    if board.ndim != 2 or board.shape[0] != board.shape[1]:
        raise ValueError("state must be a square 2D board array.")
    if player_id not in {BLACK, WHITE}:
        raise ValueError(f"player_id must be BLACK({BLACK}) or WHITE({WHITE}).")

    unique_values = set(np.unique(board).tolist())
    if not unique_values.issubset({EMPTY, BLACK, WHITE}):
        raise ValueError("state contains unsupported stone values.")

    env = GomokuEnv(board_size=int(board.shape[0]))
    env.board = board.copy()
    env.current_player = int(player_id)
    env.move_count = int(np.count_nonzero(env.board != EMPTY))
    env.done, env.winner = _infer_terminal_status(env)
    return env


def _infer_terminal_status(env: GomokuEnv) -> tuple[bool, int | None]:
    for player in (BLACK, WHITE):
        coords = np.argwhere(env.board == player)
        for row, col in coords:
            row_i = int(row)
            col_i = int(col)
            for delta_row, delta_col in DIRECTIONS:
                prev_row = row_i - delta_row
                prev_col = col_i - delta_col
                if (
                    0 <= prev_row < env.board_size
                    and 0 <= prev_col < env.board_size
                    and int(env.board[prev_row, prev_col]) == player
                ):
                    continue

                length = 1
                next_row = row_i + delta_row
                next_col = col_i + delta_col
                while (
                    0 <= next_row < env.board_size
                    and 0 <= next_col < env.board_size
                    and int(env.board[next_row, next_col]) == player
                ):
                    length += 1
                    next_row += delta_row
                    next_col += delta_col
                if length >= 5:
                    return True, player

    if env.move_count == env.board_size * env.board_size:
        return True, DRAW
    return False, None


__all__ = [
    "AthenanGreedyHeuristicAgent",
    "AthenanNegamaxSearcher",
    "BLACK",
    "BOARD_SIZE",
    "DRAW",
    "EMPTY",
    "GomokuEnv",
    "GreedyHeuristicConfig",
    "GreedyHeuristicEvaluator",
    "MoveResult",
    "NegamaxAthenanAgent",
    "NegamaxSearchSummary",
    "PatternSummary",
    "SearchResult",
    "TTEntry",
    "WHITE",
    "apply_forced_tactical_rule",
    "build_env_from_state",
    "find_immediate_blocking_actions",
    "find_immediate_winning_actions",
    "generate_candidate_actions",
    "generate_proximity_candidates",
    "negamax",
    "order_actions",
    "order_candidate_actions",
    "run_iterative_deepening_search",
    "run_negamax_search",
    "score_action",
]
