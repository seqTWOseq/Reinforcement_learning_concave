from __future__ import annotations

import time
import tkinter as tk
from typing import Optional, Sequence

import numpy as np

from gomoku_project.core.constants import BLACK, BOARD_SIZE, EMPTY, PLAYER_LABELS
from gomoku_project.core.utils import action_to_pos, pos_to_action


class TkinterRenderer:
    def __init__(
        self,
        board_size: int = BOARD_SIZE,
        cell_size: int = 40,
        margin: int = 30,
        title: str = "Gomoku",
    ) -> None:
        self.board_size = board_size
        self.cell_size = cell_size
        self.margin = margin
        self.title = title
        self.window: Optional[tk.Tk] = None
        self.canvas: Optional[tk.Canvas] = None
        self._clicked_action: Optional[int] = None
        self._valid_action_set: set[int] = set()
        self._board: Optional[np.ndarray] = None

    def open(self) -> None:
        if self.window is not None:
            return

        size = (self.board_size - 1) * self.cell_size + self.margin * 2
        self.window = tk.Tk()
        self.window.title(self.title)
        self.canvas = tk.Canvas(self.window, width=size, height=size, bg="#DCB35C")
        self.canvas.pack()

    def render(
        self,
        board: np.ndarray,
        current_player: Optional[int] = None,
        last_action: Optional[int] = None,
    ) -> None:
        self.open()
        assert self.canvas is not None
        assert self.window is not None

        self._board = board.copy()
        self.canvas.delete("all")

        for index in range(self.board_size):
            start = self.margin + index * self.cell_size
            end = self.margin + (self.board_size - 1) * self.cell_size
            self.canvas.create_line(self.margin, start, end, start)
            self.canvas.create_line(start, self.margin, start, end)

        for row in range(self.board_size):
            for col in range(self.board_size):
                value = int(board[row, col])
                if value == EMPTY:
                    continue
                x = self.margin + col * self.cell_size
                y = self.margin + row * self.cell_size
                radius = self.cell_size // 2 - 2
                fill = "black" if value == BLACK else "white"
                self.canvas.create_oval(
                    x - radius,
                    y - radius,
                    x + radius,
                    y + radius,
                    fill=fill,
                    outline="black",
                )

        if last_action is not None and 0 <= last_action < self.board_size * self.board_size:
            row, col = action_to_pos(last_action, self.board_size)
            x = self.margin + col * self.cell_size
            y = self.margin + row * self.cell_size
            marker = 5
            self.canvas.create_rectangle(
                x - marker,
                y - marker,
                x + marker,
                y + marker,
                outline="red",
                width=2,
            )

        title = self.title
        if current_player in PLAYER_LABELS:
            title = f"{self.title} - Turn: {PLAYER_LABELS[current_player]}"
        self.window.title(title)
        self.process_events()

    def wait_for_action(self, valid_actions: Sequence[int]) -> int:
        self.open()
        assert self.canvas is not None

        self._clicked_action = None
        self._valid_action_set = {int(action) for action in valid_actions}
        self.canvas.bind("<Button-1>", self._click_handler)

        while self._clicked_action is None:
            self.process_events()
            time.sleep(0.05)

        self.canvas.unbind("<Button-1>")
        return self._clicked_action

    def process_events(self) -> None:
        if self.window is None:
            return
        self.window.update_idletasks()
        self.window.update()

    def close(self) -> None:
        if self.window is not None:
            self.window.destroy()
            self.window = None
            self.canvas = None
            self._board = None
            self._clicked_action = None
            self._valid_action_set.clear()

    def _click_handler(self, event: tk.Event) -> None:
        col = round((event.x - self.margin) / self.cell_size)
        row = round((event.y - self.margin) / self.cell_size)

        if not (0 <= row < self.board_size and 0 <= col < self.board_size):
            return

        action = pos_to_action(row, col, self.board_size)
        if action in self._valid_action_set:
            self._clicked_action = action
