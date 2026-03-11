from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gomoku_project.arena.match_runner import MatchResult, run_match
from gomoku_project.envs.gomoku_env import GomokuEnv
from gomoku_project.rl.ppo_trainer import PPOTrainer
from gomoku_project.rl.trainer import AlphaZeroTrainer
from gomoku_project.utils.checkpoint_utils import load_checkpoint_or_maybe_warn


def _summarize(results: list[MatchResult], ppo_name: str, alphazero_name: str) -> None:
    ppo_wins = 0
    alphazero_wins = 0
    draws = 0

    for result in results:
        if result.winner_name == ppo_name:
            ppo_wins += 1
        elif result.winner_name == alphazero_name:
            alphazero_wins += 1
        else:
            draws += 1

    total_games = max(len(results), 1)
    print(
        f"games={len(results)} "
        f"ppo_wins={ppo_wins} "
        f"alphazero_wins={alphazero_wins} "
        f"draws={draws} "
        f"ppo_win_rate={ppo_wins / total_games:.3f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate PPO vs AlphaZero. Both checkpoints are required unless --allow-random-init is set.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--ppo-model-path",
        type=str,
        default="gomoku_project/models/ppo/ppo_checkpoint.pt",
        help="PPO checkpoint path.",
    )
    parser.add_argument(
        "--alphazero-model-path",
        type=str,
        default="gomoku_project/models/alphazero_checkpoint.pt",
        help="AlphaZero checkpoint path.",
    )
    parser.add_argument(
        "--allow-random-init",
        action="store_true",
        help="Allow PPO/AlphaZero to use random initialization if checkpoint loading fails or the file is missing.",
    )
    parser.add_argument("--games-per-color", type=int, default=5)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--mcts-simulations", type=int, default=24)
    args = parser.parse_args()
    print("mode=evaluation_only matchup=ppo_vs_alphazero")

    ppo_trainer = PPOTrainer(device=args.device)
    load_checkpoint_or_maybe_warn(
        trainer=ppo_trainer,
        path=args.ppo_model_path,
        model_label="PPO",
        cli_flag="ppo-model-path",
        allow_random_init=args.allow_random_init,
    )

    alphazero_trainer = AlphaZeroTrainer(
        batch_size=args.batch_size,
        mcts_simulations=args.mcts_simulations,
        device=args.device,
    )
    load_checkpoint_or_maybe_warn(
        trainer=alphazero_trainer,
        path=args.alphazero_model_path,
        model_label="AlphaZero",
        cli_flag="alphazero-model-path",
        allow_random_init=args.allow_random_init,
    )

    ppo_player = ppo_trainer.build_player(deterministic=True, name="PPO")
    alphazero_player = alphazero_trainer.build_player(
        deterministic=True,
        name="AlphaZero",
        use_root_noise=False,
    )
    results: list[MatchResult] = []

    for _ in range(args.games_per_color):
        results.append(run_match(black_player=ppo_player, white_player=alphazero_player, env=GomokuEnv()))
        results.append(run_match(black_player=alphazero_player, white_player=ppo_player, env=GomokuEnv()))

    for index, result in enumerate(results, start=1):
        print(
            f"match={index} "
            f"black={result.black_name} "
            f"white={result.white_name} "
            f"winner={result.winner_name} "
            f"reason={result.reason} "
            f"moves={result.move_count}"
        )

    _summarize(results, ppo_name=ppo_player.name, alphazero_name=alphazero_player.name)


if __name__ == "__main__":
    main()
