"""Agent-vs-agent play entrypoint for Athenan scaffolding."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from gomoku_ai.common.agents import BaseAgent
from gomoku_ai.env import BLACK, GomokuEnv, WHITE


@dataclass(frozen=True)
class AgentPlayResult:
    """Compact result of one full game between two agents."""

    moves: tuple[int, ...]
    winner: int
    move_count: int


def play_agent_game(
    black_agent: BaseAgent,
    white_agent: BaseAgent,
    *,
    env_factory: Callable[[], GomokuEnv] | None = None,
) -> AgentPlayResult:
    """Run one full game using the common `BaseAgent` interface."""

    env = (env_factory or GomokuEnv)()
    if not isinstance(env, GomokuEnv):
        raise TypeError("env_factory must create GomokuEnv instances.")
    env.reset()

    agents = {BLACK: black_agent, WHITE: white_agent}
    moves: list[int] = []

    while not env.done:
        acting_agent = agents[env.current_player]
        action = int(acting_agent.select_action(env))
        env.action_to_coord(action)
        if not env.get_valid_moves()[action]:
            raise ValueError(f"Agent selected illegal action: {action}.")
        env.apply_move(action)
        moves.append(action)

    if env.winner is None:
        raise RuntimeError("Game ended without a winner value.")
    return AgentPlayResult(moves=tuple(moves), winner=env.winner, move_count=env.move_count)
