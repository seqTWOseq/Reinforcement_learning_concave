from __future__ import annotations

from typing import Iterable

import numpy as np

from gomoku_project.core.constants import BLACK, EMPTY, WHITE


def action_to_pos(action: int, board_size: int) -> tuple[int, int]:
    return divmod(int(action), board_size)


def pos_to_action(row: int, col: int, board_size: int) -> int:
    return int(row * board_size + col)


def opponent(player: int) -> int:
    return WHITE if player == BLACK else BLACK


def get_valid_actions(board: np.ndarray) -> list[int]:
    return np.flatnonzero(board.reshape(-1) == EMPTY).astype(int).tolist()


def is_action_in_bounds(action: int, board_size: int) -> bool:
    return 0 <= int(action) < board_size * board_size


def orient_board_to_player(board: np.ndarray, player: int) -> np.ndarray:
    if player == BLACK:
        return board.copy()

    oriented = np.where(board == BLACK, WHITE, np.where(board == WHITE, BLACK, EMPTY))
    return oriented.astype(board.dtype, copy=False)


def board_to_ansi(board: np.ndarray) -> str:
    symbols = {
        EMPTY: ".",
        BLACK: "X",
        WHITE: "O",
    }
    rows = [" ".join(symbols[int(value)] for value in row) for row in board]
    return "\n".join(rows)


def copy_info_dict(info: dict) -> dict:
    copied = dict(info)
    board = copied.get("board")
    if isinstance(board, np.ndarray):
        copied["board"] = board.copy()
    valid_actions = copied.get("valid_actions")
    if isinstance(valid_actions, Iterable) and not isinstance(valid_actions, (str, bytes)):
        copied["valid_actions"] = list(valid_actions)
    return copied

