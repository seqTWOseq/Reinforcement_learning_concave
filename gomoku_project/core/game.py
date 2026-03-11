from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from gomoku_project.core.constants import (
    BLACK,
    BOARD_SIZE,
    CONTINUE_REWARD,
    DRAW,
    DRAW_REWARD,
    EMPTY,
    INVALID_MOVE_PENALTY,
    WIN_LENGTH,
    WIN_REWARD,
)
from gomoku_project.core.utils import (
    action_to_pos,
    get_valid_actions,
    is_action_in_bounds,
    opponent,
    orient_board_to_player,
)


@dataclass(frozen=True)
class GameStepResult:
    reward: float
    done: bool
    winner: Optional[int]
    reason: str
    last_action: Optional[int]


class GomokuGame:
    def __init__(self, board_size: int = BOARD_SIZE, win_length: int = WIN_LENGTH) -> None:
        self.board_size = board_size
        self.win_length = win_length
        self.board = np.zeros((self.board_size, self.board_size), dtype=np.int8)
        self.current_player = BLACK
        self.winner: Optional[int] = None
        self.done = False
        self.last_action: Optional[int] = None
        self.last_reason = "reset"

    @classmethod
    def from_state(
        cls,
        board: np.ndarray,
        current_player: int,
        *,
        done: bool = False,
        winner: Optional[int] = None,
        last_action: Optional[int] = None,
        last_reason: str = "manual",
        win_length: int = WIN_LENGTH,
    ) -> "GomokuGame":
        game = cls(board_size=board.shape[0], win_length=win_length)
        game.board = board.astype(np.int8, copy=True)
        game.current_player = int(current_player)
        game.done = bool(done)
        game.winner = winner
        game.last_action = last_action
        game.last_reason = last_reason
        return game

    def reset(self) -> np.ndarray:
        self.board.fill(EMPTY)
        self.current_player = BLACK
        self.winner = None
        self.done = False
        self.last_action = None
        self.last_reason = "reset"
        return self.board.copy()

    def get_valid_actions(self) -> list[int]:
        if self.done:
            return []
        return get_valid_actions(self.board)

    def get_observation(self, player: Optional[int] = None) -> np.ndarray:
        perspective = self.current_player if player is None else player
        return orient_board_to_player(self.board, perspective)

    def get_info(self) -> dict:
        return {
            "current_player": self.current_player,
            "valid_actions": self.get_valid_actions(),
            "winner": self.winner,
            "last_action": self.last_action,
            "done": self.done,
            "reason": self.last_reason,
            "board": self.board.copy(),
        }

    def clone(self) -> "GomokuGame":
        return GomokuGame.from_state(
            board=self.board,
            current_player=self.current_player,
            done=self.done,
            winner=self.winner,
            last_action=self.last_action,
            last_reason=self.last_reason,
            win_length=self.win_length,
        )

    def step(self, action: int) -> GameStepResult:
        if self.done:
            raise RuntimeError("Cannot call step() on a finished game. Call reset() first.")

        action = int(action)
        self.last_action = action

        if not is_action_in_bounds(action, self.board_size):
            self.done = True
            self.winner = opponent(self.current_player)
            self.last_reason = "invalid_move"
            return GameStepResult(
                reward=INVALID_MOVE_PENALTY,
                done=True,
                winner=self.winner,
                reason=self.last_reason,
                last_action=self.last_action,
            )

        row, col = action_to_pos(action, self.board_size)
        if self.board[row, col] != EMPTY:
            self.done = True
            self.winner = opponent(self.current_player)
            self.last_reason = "invalid_move"
            return GameStepResult(
                reward=INVALID_MOVE_PENALTY,
                done=True,
                winner=self.winner,
                reason=self.last_reason,
                last_action=self.last_action,
            )

        self.board[row, col] = self.current_player

        if self._check_win(row, col, self.current_player):
            self.done = True
            self.winner = self.current_player
            self.last_reason = "win"
            return GameStepResult(
                reward=WIN_REWARD,
                done=True,
                winner=self.winner,
                reason=self.last_reason,
                last_action=self.last_action,
            )

        if not np.any(self.board == EMPTY):
            self.done = True
            self.winner = DRAW
            self.last_reason = "draw"
            return GameStepResult(
                reward=DRAW_REWARD,
                done=True,
                winner=self.winner,
                reason=self.last_reason,
                last_action=self.last_action,
            )

        self.current_player = opponent(self.current_player)
        self.last_reason = "continue"
        return GameStepResult(
            reward=CONTINUE_REWARD,
            done=False,
            winner=None,
            reason=self.last_reason,
            last_action=self.last_action,
        )

    def _check_win(self, row: int, col: int, player: int) -> bool:
        directions = ((0, 1), (1, 0), (1, 1), (-1, 1))
        for dr, dc in directions:
            count = 1
            for direction in (1, -1):
                r = row + dr * direction
                c = col + dc * direction
                while 0 <= r < self.board_size and 0 <= c < self.board_size:
                    if self.board[r, c] != player:
                        break
                    count += 1
                    r += dr * direction
                    c += dc * direction
            if count >= self.win_length:
                return True
        return False
