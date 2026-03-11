from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from gomoku_project.core.constants import BLACK, EMPTY, WHITE
from gomoku_project.core.utils import action_to_pos
from gomoku_project.players.base import BasePlayer

_DIRECTIONS = ((0, 1), (1, 0), (1, 1), (1, -1))
_LINE_RADIUS = 4


@dataclass(frozen=True)
class _PatternSummary:
    immediate_win: bool
    strong_four_count: int
    open_four_count: int
    open_three_count: int
    max_run: int


@dataclass(frozen=True)
class _ScoredMove:
    action: int
    total_score: int
    center_distance: int


def _line_tokens(
    board: np.ndarray,
    row: int,
    col: int,
    dr: int,
    dc: int,
    player: int,
) -> tuple[list[str], int]:
    board_size = board.shape[0]
    tokens: list[str] = []
    center_index = _LINE_RADIUS

    for offset in range(-_LINE_RADIUS, _LINE_RADIUS + 1):
        target_row = row + dr * offset
        target_col = col + dc * offset
        if 0 <= target_row < board_size and 0 <= target_col < board_size:
            value = int(board[target_row, target_col])
            if value == player:
                tokens.append("S")
            elif value == EMPTY:
                tokens.append("E")
            else:
                tokens.append("O")
        else:
            tokens.append("O")

    return tokens, center_index


def _iter_windows(tokens: list[str], center_index: int, window_size: int) -> list[tuple[int, list[str]]]:
    windows: list[tuple[int, list[str]]] = []
    for start in range(0, len(tokens) - window_size + 1):
        end = start + window_size
        if start <= center_index < end:
            windows.append((start, tokens[start:end]))
    return windows


def _is_open_four(window: list[str]) -> bool:
    return (
        len(window) == 6
        and window[0] == "E"
        and window[-1] == "E"
        and window.count("O") == 0
        and window.count("S") == 4
    )


def _is_strong_four(window: list[str]) -> bool:
    return len(window) == 5 and window.count("O") == 0 and window.count("S") == 4 and window.count("E") == 1


def _is_open_three(window: list[str]) -> bool:
    if (
        len(window) != 6
        or window[0] != "E"
        or window[-1] != "E"
        or window.count("O") > 0
        or window.count("S") != 3
    ):
        return False

    for index, token in enumerate(window):
        if token != "E":
            continue
        candidate = window.copy()
        candidate[index] = "S"
        if _is_open_four(candidate):
            return True
    return False


def _contiguous_run(board: np.ndarray, row: int, col: int, dr: int, dc: int, player: int) -> int:
    board_size = board.shape[0]
    count = 1
    for direction in (1, -1):
        target_row = row + dr * direction
        target_col = col + dc * direction
        while 0 <= target_row < board_size and 0 <= target_col < board_size:
            if int(board[target_row, target_col]) != player:
                break
            count += 1
            target_row += dr * direction
            target_col += dc * direction
    return count


def _analyze_move(board: np.ndarray, action: int, player: int) -> _PatternSummary:
    board_size = board.shape[0]
    row, col = action_to_pos(action, board_size)
    if int(board[row, col]) != EMPTY:
        return _PatternSummary(
            immediate_win=False,
            strong_four_count=0,
            open_four_count=0,
            open_three_count=0,
            max_run=0,
        )

    candidate_board = board.copy()
    candidate_board[row, col] = player

    immediate_win = False
    strong_four_windows: set[tuple[int, int]] = set()
    open_four_windows: set[tuple[int, int]] = set()
    open_three_windows: set[tuple[int, int]] = set()
    max_run = 1

    for direction_index, (dr, dc) in enumerate(_DIRECTIONS):
        max_run = max(max_run, _contiguous_run(candidate_board, row, col, dr, dc, player))
        tokens, center_index = _line_tokens(candidate_board, row, col, dr, dc, player)

        for start, window in _iter_windows(tokens, center_index, 5):
            if window.count("S") == 5:
                immediate_win = True
            elif _is_strong_four(window):
                strong_four_windows.add((direction_index, start))

        for start, window in _iter_windows(tokens, center_index, 6):
            if _is_open_four(window):
                open_four_windows.add((direction_index, start))
            if _is_open_three(window):
                open_three_windows.add((direction_index, start))

    return _PatternSummary(
        immediate_win=immediate_win or max_run >= 5,
        strong_four_count=len(strong_four_windows),
        open_four_count=len(open_four_windows),
        open_three_count=len(open_three_windows),
        max_run=max_run,
    )


def _center_bonus(board_size: int, row: int, col: int) -> int:
    center = board_size // 2
    max_distance = max(center * 2, 1)
    distance = abs(row - center) + abs(col - center)
    return int(round(1000.0 * (1.0 - min(distance / max_distance, 1.0))))


def _connection_bonus(board: np.ndarray, row: int, col: int, player: int) -> int:
    score = 0
    nearby_occupied = 0
    nearby_allies = 0

    for dr in range(-2, 3):
        for dc in range(-2, 3):
            if dr == 0 and dc == 0:
                continue
            target_row = row + dr
            target_col = col + dc
            if not (0 <= target_row < board.shape[0] and 0 <= target_col < board.shape[1]):
                continue

            value = int(board[target_row, target_col])
            if value != EMPTY:
                nearby_occupied += 1
            if value == player:
                nearby_allies += 1
                score += 160 if max(abs(dr), abs(dc)) == 1 else 70

    if nearby_allies == 0 and np.any(board == player):
        score -= 220
    if nearby_occupied == 0 and np.any(board != EMPTY):
        score -= 420

    return score


def _score_patterns(attack: _PatternSummary, defense: _PatternSummary) -> int:
    attack_score = 0
    defense_score = 0

    if attack.immediate_win:
        attack_score += 1_000_000
    if defense.immediate_win:
        defense_score += 900_000

    attack_score += 120_000 * attack.open_four_count
    attack_score += 100_000 * attack.strong_four_count
    defense_score += 95_000 * defense.open_four_count
    defense_score += 90_000 * defense.strong_four_count

    attack_score += 20_000 * attack.open_three_count
    defense_score += 18_000 * defense.open_three_count

    attack_score += 2_500 * attack.max_run
    defense_score += 1_800 * defense.max_run

    return attack_score + defense_score


class HeuristicPlayer(BasePlayer):
    def __init__(
        self,
        name: str = "HeuristicPlayer",
        *,
        seed: int | None = None,
    ) -> None:
        super().__init__(name=name)
        self.rng = np.random.default_rng(seed)

    def reset(self) -> None:
        self.last_policy_target = None

    def select_action(
        self,
        observation: np.ndarray,
        valid_actions: Sequence[int],
        info: dict[str, Any],
    ) -> int:
        if not valid_actions:
            return 0

        board = np.asarray(observation, dtype=np.int8)
        board_size = board.shape[0]
        scored_moves = [self._score_move(board, int(action)) for action in valid_actions]
        best_score = max(move.total_score for move in scored_moves)
        best_moves = [move for move in scored_moves if move.total_score == best_score]
        closest_distance = min(move.center_distance for move in best_moves)
        tied_moves = [move for move in best_moves if move.center_distance == closest_distance]
        chosen_move = tied_moves[int(self.rng.integers(len(tied_moves)))]

        self.last_policy_target = np.zeros(board_size * board_size, dtype=np.float32)
        self.last_policy_target[chosen_move.action] = 1.0
        return int(chosen_move.action)

    def _score_move(self, board: np.ndarray, action: int) -> _ScoredMove:
        board_size = board.shape[0]
        row, col = action_to_pos(action, board_size)
        attack = _analyze_move(board, action, BLACK)
        defense = _analyze_move(board, action, WHITE)
        pattern_score = _score_patterns(attack, defense)
        positional_score = _center_bonus(board_size, row, col) + _connection_bonus(board, row, col, BLACK)
        center = board_size // 2
        center_distance = abs(row - center) + abs(col - center)
        return _ScoredMove(
            action=action,
            total_score=pattern_score + positional_score,
            center_distance=center_distance,
        )
