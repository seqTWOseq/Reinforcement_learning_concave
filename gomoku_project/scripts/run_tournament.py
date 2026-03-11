from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gomoku_project.arena.tournament import run_round_robin_tournament
from gomoku_project.players.center_player import CenterPlayer
from gomoku_project.players.heuristic_player import HeuristicPlayer
from gomoku_project.players.random_player import RandomPlayer
from gomoku_project.rl.ppo_trainer import PPOTrainer
from gomoku_project.rl.trainer import AlphaZeroTrainer
from gomoku_project.utils.checkpoint_utils import load_checkpoint_or_maybe_warn


def _display_name(kind: str) -> str:
    return {
        "random": "Random",
        "center": "Center",
        "heuristic": "Heuristic",
        "ppo": "PPO",
        "alphazero": "AlphaZero",
    }[kind]


def _build_player(kind: str, name: str, args: argparse.Namespace):
    if kind == "random":
        return RandomPlayer(name=name), None
    if kind == "center":
        return CenterPlayer(name=name), None
    if kind == "heuristic":
        return HeuristicPlayer(name=name), None
    if kind == "ppo":
        trainer = PPOTrainer(device=args.device)
        load_checkpoint_or_maybe_warn(
            trainer=trainer,
            path=args.ppo_model_path,
            model_label=name,
            cli_flag="ppo-model-path",
            allow_random_init=args.allow_random_init,
        )
        return trainer.build_player(deterministic=True, name=name), trainer

    trainer = AlphaZeroTrainer(
        batch_size=args.batch_size,
        mcts_simulations=args.mcts_simulations,
        device=args.device,
    )
    load_checkpoint_or_maybe_warn(
        trainer=trainer,
        path=args.alphazero_model_path,
        model_label=name,
        cli_flag="alphazero-model-path",
        allow_random_init=args.allow_random_init,
    )
    return trainer.build_player(deterministic=True, name=name, use_root_noise=False), trainer


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a round-robin Gomoku tournament. PPO/AlphaZero require checkpoints unless --allow-random-init is set.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--players",
        nargs="+",
        choices=("random", "center", "heuristic", "ppo", "alphazero"),
        default=["random", "heuristic", "ppo", "alphazero", "center"],
        help="Player pool for the round-robin tournament.",
    )
    parser.add_argument(
        "--alphazero-model-path",
        type=str,
        default="gomoku_project/models/alphazero_checkpoint.pt",
        help="AlphaZero checkpoint path for any alphazero participant.",
    )
    parser.add_argument(
        "--ppo-model-path",
        type=str,
        default="gomoku_project/models/ppo/ppo_checkpoint.pt",
        help="PPO checkpoint path for any ppo participant.",
    )
    parser.add_argument(
        "--allow-random-init",
        action="store_true",
        help="Allow PPO/AlphaZero participants to use random initialization if checkpoint loading fails or the file is missing.",
    )
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--mcts-simulations", type=int, default=24)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--move-delay", type=float, default=0.0)
    parser.add_argument("--print-matches", action="store_true")
    args = parser.parse_args()

    total_counts = Counter(args.players)
    seen_counts: Counter[str] = Counter()
    players = []
    trainers: list[object] = []

    for kind in args.players:
        seen_counts[kind] += 1
        base_name = _display_name(kind)
        if total_counts[kind] > 1:
            name = f"{base_name}_{seen_counts[kind]}"
        else:
            name = base_name
        player, trainer = _build_player(kind, name, args)
        players.append(player)
        if trainer is not None:
            trainers.append(trainer)

    result = run_round_robin_tournament(
        players=players,
        render=args.render,
        move_delay=args.move_delay,
    )

    print("=== Standings ===")
    for row in result["standings"]:
        print(
            f"{row['name']}: "
            f"games={row['games']} "
            f"wins={row['wins']} "
            f"losses={row['losses']} "
            f"draws={row['draws']} "
            f"points={row['points']:.1f} "
            f"win_rate={row['win_rate']:.3f} "
            f"score_rate={row['score_rate']:.3f} "
            f"as_black={row['as_black']} "
            f"as_white={row['as_white']}"
        )

    if args.print_matches:
        print("\n=== Matches ===")
        for match in result["matches"]:
            print(match)


if __name__ == "__main__":
    main()
