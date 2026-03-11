from __future__ import annotations

import time
from typing import Any, Optional

import numpy as np

from gomoku_project.core.constants import DRAW, BLACK, WHITE
from gomoku_project.core.utils import copy_info_dict
from gomoku_project.envs.gomoku_env import GomokuEnv
from gomoku_project.players.base import BasePlayer
from gomoku_project.rl.replay_buffer import AlphaZeroExample


def _fallback_policy_target(
    chosen_action: int,
    action_space_size: int,
) -> np.ndarray:
    target = np.zeros(action_space_size, dtype=np.float32)
    target[int(chosen_action)] = 1.0
    return target


def _winner_to_value(winner: int | None, acting_player: int) -> float:
    if winner is None or winner == DRAW:
        return 0.0
    return 1.0 if winner == acting_player else -1.0


def play_recorded_game(
    black_player: BasePlayer,
    white_player: BasePlayer,
    env: Optional[GomokuEnv] = None,
    *,
    render: bool = False,
    renderer: Optional[Any] = None,
    move_delay: float = 0.0,
    post_game_delay: float = 0.0,
    close_renderer: bool = True,
) -> tuple[list[AlphaZeroExample], dict]:
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
    action_space_size = env.board_size * env.board_size
    pending_steps: list[dict[str, Any]] = []
    terminated = False
    truncated = False

    if render and renderer is not None:
        renderer.render(
            board=info["board"],
            current_player=info["current_player"],
            last_action=info["last_action"],
        )

    while not (terminated or truncated):
        current_player = info["current_player"]
        active_player = black_player if current_player == BLACK else white_player
        active_player.last_policy_target = None
        action = int(active_player.select_action(observation, info["valid_actions"], info))
        policy_target = getattr(active_player, "last_policy_target", None)
        if policy_target is None:
            policy_target = _fallback_policy_target(action, action_space_size)
        else:
            policy_target = np.asarray(policy_target, dtype=np.float32).copy()
            if policy_target.shape != (action_space_size,) or not np.isfinite(policy_target).all():
                policy_target = _fallback_policy_target(action, action_space_size)

        pending_steps.append(
            {
                "observation": observation.copy(),
                "policy_target": policy_target,
                "acting_player": current_player,
                "last_action": action,
            }
        )

        observation, _, terminated, truncated, info = env.step(action)

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

    final_info = copy_info_dict(info)
    winner = final_info.get("winner")
    examples = [
        AlphaZeroExample(
            observation=step["observation"],
            policy_target=step["policy_target"],
            value_target=_winner_to_value(winner, step["acting_player"]),
            info={
                "winner": winner,
                "reason": final_info.get("reason"),
                "acting_player": step["acting_player"],
                "last_action": step["last_action"],
            },
        )
        for step in pending_steps
    ]

    env.close()

    if render and renderer is not None and close_renderer:
        renderer.close()

    return examples, final_info


def play_self_play_game(
    black_player: BasePlayer,
    white_player: BasePlayer,
    env: Optional[GomokuEnv] = None,
) -> tuple[list[AlphaZeroExample], dict]:
    return play_recorded_game(
        black_player=black_player,
        white_player=white_player,
        env=env,
    )


def collect_self_play_games(
    black_player: BasePlayer,
    white_player: BasePlayer,
    num_games: int,
    *,
    env_factory: type[GomokuEnv] = GomokuEnv,
) -> list[AlphaZeroExample]:
    all_examples: list[AlphaZeroExample] = []
    for _ in range(num_games):
        examples, _ = play_self_play_game(
            black_player=black_player,
            white_player=white_player,
            env=env_factory(),
        )
        all_examples.extend(examples)
    return all_examples
