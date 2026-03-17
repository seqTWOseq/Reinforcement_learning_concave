"""Human-vs-Athenan play entrypoint with optional search debug output."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

from gomoku_ai.athenan.network import load_athenan_value_net
from gomoku_ai.athenan.search import AthenanInferenceSearcher
from gomoku_ai.common.agents import BaseSearcher, SearchResult
from gomoku_ai.env import BLACK, DRAW, GomokuEnv, WHITE


@dataclass(frozen=True)
class AthenanAIDebugTurn:
    """Debug payload for one AI move."""

    move_index: int
    action: int
    root_value: float
    principal_variation: tuple[int, ...]
    top_actions: tuple[tuple[int, float], ...]
    nodes: int
    depth_reached: int
    forced_tactical: bool


@dataclass(frozen=True)
class HumanVsAthenanResult:
    """Result of one human-vs-Athenan game."""

    winner: int
    move_count: int
    moves: tuple[int, ...]
    human_color: int
    ai_debug_turns: tuple[AthenanAIDebugTurn, ...]


def play_human_vs_athenan_game(
    *,
    searcher: BaseSearcher,
    human_color: int = BLACK,
    env_factory: Callable[[], GomokuEnv] | None = None,
    human_moves: Sequence[int] | None = None,
    human_move_selector: Callable[[GomokuEnv], int] | None = None,
    debug: bool = False,
    debug_top_k: int = 5,
    input_fn: Callable[[str], str] = input,
    print_fn: Callable[[str], None] = print,
) -> HumanVsAthenanResult:
    """Play one game between a human side and an Athenan searcher."""

    if human_color not in {BLACK, WHITE}:
        raise ValueError(f"human_color must be BLACK({BLACK}) or WHITE({WHITE}).")
    if debug_top_k <= 0:
        raise ValueError("debug_top_k must be positive.")

    env = (env_factory or GomokuEnv)()
    env.reset()
    human_move_iter = iter(human_moves) if human_moves is not None else None
    moves: list[int] = []
    ai_debug_turns: list[AthenanAIDebugTurn] = []

    while not env.done:
        if env.current_player == human_color:
            action = _resolve_human_action(
                env,
                human_move_selector=human_move_selector,
                human_move_iter=human_move_iter,
                input_fn=input_fn,
            )
        else:
            search_result = searcher.search(env)
            action = int(search_result.best_action)
            if action < 0:
                raise RuntimeError("Athenan searcher returned best_action < 0 on non-terminal human-play turn.")
            debug_turn = _build_ai_debug_turn(
                move_index=len(moves),
                action=action,
                search_result=search_result,
                top_k=debug_top_k,
            )
            ai_debug_turns.append(debug_turn)
            if debug:
                print_fn(_format_ai_debug(debug_turn))

        _assert_legal_action(env, action)
        env.apply_move(action)
        moves.append(int(action))

    if env.winner is None:
        raise RuntimeError("Game ended without winner information.")
    if debug:
        print_fn(_format_game_result(winner=env.winner, move_count=env.move_count))
    return HumanVsAthenanResult(
        winner=int(env.winner),
        move_count=int(env.move_count),
        moves=tuple(moves),
        human_color=human_color,
        ai_debug_turns=tuple(ai_debug_turns),
    )


def _resolve_human_action(
    env: GomokuEnv,
    *,
    human_move_selector: Callable[[GomokuEnv], int] | None,
    human_move_iter: object | None,
    input_fn: Callable[[str], str],
) -> int:
    if human_move_selector is not None:
        return int(human_move_selector(env))
    if human_move_iter is not None:
        try:
            return int(next(human_move_iter))
        except StopIteration as exc:
            raise RuntimeError("No more scripted human moves are available.") from exc

    while True:
        raw = input_fn("Enter action (int) or row,col: ").strip()
        try:
            action = _parse_human_action(raw, env=env)
        except ValueError as exc:
            print(str(exc))
            continue
        return action


def _parse_human_action(raw: str, *, env: GomokuEnv) -> int:
    if "," in raw:
        parts = [part.strip() for part in raw.split(",", maxsplit=1)]
        if len(parts) != 2:
            raise ValueError("Input must be 'row,col' or one integer action index.")
        row = int(parts[0])
        col = int(parts[1])
        return int(env.coord_to_action(row, col))
    return int(raw)


def _assert_legal_action(env: GomokuEnv, action: int) -> None:
    legal_mask = env.get_valid_moves()
    if not (0 <= action < legal_mask.size):
        raise RuntimeError(f"Action {action} is out of range.")
    if not bool(legal_mask[action]):
        raise RuntimeError(f"Action {action} is illegal for current position.")


def _build_ai_debug_turn(
    *,
    move_index: int,
    action: int,
    search_result: SearchResult,
    top_k: int,
) -> AthenanAIDebugTurn:
    top_actions = tuple(
        sorted(
            ((int(candidate_action), float(value)) for candidate_action, value in search_result.action_values.items()),
            key=lambda item: (-item[1], item[0]),
        )[:top_k]
    )
    return AthenanAIDebugTurn(
        move_index=move_index,
        action=action,
        root_value=float(search_result.root_value),
        principal_variation=tuple(int(candidate) for candidate in search_result.principal_variation),
        top_actions=top_actions,
        nodes=int(search_result.nodes),
        depth_reached=int(search_result.depth_reached),
        forced_tactical=bool(search_result.forced_tactical),
    )


def _format_ai_debug(debug_turn: AthenanAIDebugTurn) -> str:
    return (
        f"[AI] move={debug_turn.move_index} action={debug_turn.action} "
        f"root_value={debug_turn.root_value:.4f} depth={debug_turn.depth_reached} "
        f"nodes={debug_turn.nodes} forced={debug_turn.forced_tactical} "
        f"pv={list(debug_turn.principal_variation)} top_actions={list(debug_turn.top_actions)}"
    )


def _format_game_result(*, winner: int, move_count: int) -> str:
    if winner == DRAW:
        winner_text = "draw"
    elif winner == BLACK:
        winner_text = "black"
    elif winner == WHITE:
        winner_text = "white"
    else:
        winner_text = str(winner)
    return f"[Result] winner={winner_text} move_count={move_count}"


def _parse_human_color(raw: str) -> int:
    normalized = raw.strip().lower()
    if normalized in {"black", "b"}:
        return BLACK
    if normalized in {"white", "w"}:
        return WHITE
    raise ValueError("human-color must be one of {'black', 'white'}.")


def main(argv: Sequence[str] | None = None) -> None:
    """CLI entrypoint for human-vs-Athenan play."""

    parser = argparse.ArgumentParser(
        description="Play Gomoku against Athenan inference search.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--human-color", type=str, default="black", help="Human side color: black or white.")
    parser.add_argument("--checkpoint-path", type=str, default="", help="Optional Athenan value-net checkpoint path.")
    parser.add_argument("--max-depth", type=int, default=4, help="Inference max depth.")
    parser.add_argument("--candidate-limit", type=int, default=64, help="Candidate action limit.")
    parser.add_argument("--candidate-radius", type=int, default=2, help="Candidate radius.")
    parser.add_argument("--time-budget-sec", type=float, default=0.0, help="0 means no time budget.")
    parser.add_argument("--no-iterative", action="store_true", help="Disable iterative deepening.")
    parser.add_argument("--debug", action="store_true", help="Print root debug info each AI move.")
    parser.add_argument("--debug-top-k", type=int, default=5, help="How many root actions to show in debug output.")
    parser.add_argument("--device", type=str, default="cpu", help="Torch device for model inference.")
    args = parser.parse_args(argv)

    human_color = _parse_human_color(args.human_color)
    checkpoint_path = args.checkpoint_path.strip()
    model = load_athenan_value_net(Path(checkpoint_path), device=args.device) if checkpoint_path else None
    time_budget_sec = None if args.time_budget_sec <= 0.0 else float(args.time_budget_sec)

    searcher = AthenanInferenceSearcher(
        model=model,
        max_depth=args.max_depth,
        candidate_limit=args.candidate_limit,
        candidate_radius=args.candidate_radius,
        iterative_deepening=not args.no_iterative,
        time_budget_sec=time_budget_sec,
        device=args.device,
    )
    play_human_vs_athenan_game(
        searcher=searcher,
        human_color=human_color,
        debug=args.debug,
        debug_top_k=args.debug_top_k,
    )


if __name__ == "__main__":
    main()
