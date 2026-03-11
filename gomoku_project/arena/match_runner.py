from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

from gomoku_project.core.constants import BLACK, DRAW, WHITE
from gomoku_project.envs.gomoku_env import GomokuEnv
from gomoku_project.players.base import BasePlayer


@dataclass
class MatchResult:
    black_name: str
    white_name: str
    winner: Optional[int]
    winner_name: str
    reason: str
    move_count: int
    final_reward: float
    actions: list[int]


def run_match(
    black_player: BasePlayer,
    white_player: BasePlayer,
    env: Optional[GomokuEnv] = None,
    *,
    render: bool = False,
    renderer: Optional[object] = None,
    move_delay: float = 0.0,
    post_game_delay: float = 0.0,
    close_renderer: bool = True,
) -> MatchResult:
    env = env or GomokuEnv()

    if render and renderer is None:
        from gomoku_project.ui.tkinter_renderer import TkinterRenderer

        renderer = TkinterRenderer(board_size=env.board_size)

    black_player.set_player_id(BLACK)
    white_player.set_player_id(WHITE)
    black_player.set_renderer(renderer if render else None)
    white_player.set_renderer(renderer if render else None)
    black_player.reset()
    white_player.reset()

    observation, info = env.reset()
    terminated = False
    truncated = False
    reward = 0.0
    actions: list[int] = []

    if render and renderer is not None:
        renderer.render(
            board=info["board"],
            current_player=info["current_player"],
            last_action=info["last_action"],
        )

    while not (terminated or truncated):
        current_player = info["current_player"]
        active_player = black_player if current_player == BLACK else white_player
        action = int(active_player.select_action(observation, info["valid_actions"], info))
        actions.append(action)

        observation, reward, terminated, truncated, info = env.step(action)

        if render and renderer is not None:
            renderer.render(
                board=info["board"],
                current_player=info["current_player"],
                last_action=info["last_action"],
            )
            if move_delay > 0:
                time.sleep(move_delay)

    if render and renderer is not None and post_game_delay > 0:
        end_time = time.time() + post_game_delay
        while time.time() < end_time:
            renderer.process_events()
            time.sleep(0.05)

    winner = info.get("winner")
    if winner == BLACK:
        winner_name = black_player.name
    elif winner == WHITE:
        winner_name = white_player.name
    elif winner == DRAW:
        winner_name = "Draw"
    else:
        winner_name = "Unknown"

    result = MatchResult(
        black_name=black_player.name,
        white_name=white_player.name,
        winner=winner,
        winner_name=winner_name,
        reason=info.get("reason", "unknown"),
        move_count=len(actions),
        final_reward=reward,
        actions=actions,
    )

    env.close()
    if render and renderer is not None and close_renderer:
        renderer.close()

    return result

