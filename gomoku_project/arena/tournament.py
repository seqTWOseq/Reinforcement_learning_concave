from __future__ import annotations

from itertools import combinations
from typing import Callable, Sequence

from gomoku_project.arena.match_runner import MatchResult, run_match
from gomoku_project.core.constants import BLACK, DRAW, WHITE
from gomoku_project.envs.gomoku_env import GomokuEnv
from gomoku_project.players.base import BasePlayer


def _new_record(name: str) -> dict:
    return {
        "name": name,
        "games": 0,
        "wins": 0,
        "losses": 0,
        "draws": 0,
        "points": 0.0,
        "as_black": 0,
        "as_white": 0,
        "win_rate": 0.0,
        "score_rate": 0.0,
    }


def _apply_result(standings: dict[str, dict], result: MatchResult) -> None:
    black = standings[result.black_name]
    white = standings[result.white_name]

    black["games"] += 1
    white["games"] += 1
    black["as_black"] += 1
    white["as_white"] += 1

    if result.winner == BLACK:
        black["wins"] += 1
        black["points"] += 1.0
        white["losses"] += 1
    elif result.winner == WHITE:
        white["wins"] += 1
        white["points"] += 1.0
        black["losses"] += 1
    elif result.winner == DRAW:
        black["draws"] += 1
        white["draws"] += 1
        black["points"] += 0.5
        white["points"] += 0.5


def run_round_robin_tournament(
    players: Sequence[BasePlayer],
    *,
    env_factory: Callable[[], GomokuEnv] | None = None,
    render: bool = False,
    move_delay: float = 0.0,
) -> dict:
    env_factory = env_factory or GomokuEnv
    standings = {player.name: _new_record(player.name) for player in players}
    matches: list[MatchResult] = []

    for player_a, player_b in combinations(players, 2):
        for black_player, white_player in ((player_a, player_b), (player_b, player_a)):
            result = run_match(
                black_player=black_player,
                white_player=white_player,
                env=env_factory(),
                render=render,
                move_delay=move_delay,
            )
            matches.append(result)
            _apply_result(standings, result)

    for record in standings.values():
        games = max(int(record["games"]), 1)
        record["win_rate"] = record["wins"] / games
        record["score_rate"] = record["points"] / games

    ordered_standings = sorted(
        standings.values(),
        key=lambda item: (-item["points"], -item["wins"], item["losses"], item["name"]),
    )
    return {
        "standings": ordered_standings,
        "matches": matches,
    }
