from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gomoku_project.arena.match_runner import MatchResult, run_match
from gomoku_project.envs.gomoku_env import GomokuEnv
from gomoku_project.players.heuristic_player import HeuristicPlayer
from gomoku_project.rl.ppo_trainer import PPOTrainer
from gomoku_project.utils.checkpoint_utils import load_checkpoint_or_maybe_warn


def _summarize(results: list[MatchResult], ppo_name: str, heuristic_name: str) -> None:
    ppo_wins = 0
    heuristic_wins = 0
    draws = 0

    for result in results:
        if result.winner_name == ppo_name:
            ppo_wins += 1
        elif result.winner_name == heuristic_name:
            heuristic_wins += 1
        else:
            draws += 1

    total_games = max(len(results), 1)
    print(
        f"games={len(results)} "
        f"ppo_wins={ppo_wins} "
        f"heuristic_wins={heuristic_wins} "
        f"draws={draws} "
        f"ppo_win_rate={ppo_wins / total_games:.3f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate PPO vs Heuristic. The PPO checkpoint is required unless --allow-random-init is set.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model-path", type=str, default="gomoku_project/models/ppo/ppo_checkpoint.pt")
    parser.add_argument(
        "--allow-random-init",
        action="store_true",
        help="Allow PPO to use random initialization if the checkpoint is missing or invalid.",
    )
    parser.add_argument("--games-per-color", type=int, default=5)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--use-heuristic-prior", action="store_true")
    parser.add_argument("--heuristic-prior-beta-start", type=float, default=0.0)
    parser.add_argument("--heuristic-prior-beta-end", type=float, default=0.0)
    parser.add_argument("--heuristic-prior-decay-updates", type=int, default=0)
    parser.add_argument("--heuristic-prior-score-clip", type=float, default=2.5)
    args = parser.parse_args()

    trainer = PPOTrainer(
        device=args.device,
        use_heuristic_prior=args.use_heuristic_prior,
        heuristic_prior_beta_start=args.heuristic_prior_beta_start,
        heuristic_prior_beta_end=args.heuristic_prior_beta_end,
        heuristic_prior_decay_updates=args.heuristic_prior_decay_updates,
        heuristic_prior_score_clip=args.heuristic_prior_score_clip,
    )
    load_checkpoint_or_maybe_warn(
        trainer=trainer,
        path=args.model_path,
        model_label="PPO",
        cli_flag="model-path",
        allow_random_init=args.allow_random_init,
    )
    print(
        f"heuristic_prior_enabled={trainer.use_heuristic_prior} "
        f"heuristic_prior_beta={trainer.current_heuristic_prior_beta():.4f}"
    )

    ppo_player = trainer.build_player(deterministic=True, name="PPO")
    heuristic_player = HeuristicPlayer(name="Heuristic")
    results: list[MatchResult] = []

    for _ in range(args.games_per_color):
        results.append(run_match(black_player=ppo_player, white_player=heuristic_player, env=GomokuEnv()))
        results.append(run_match(black_player=heuristic_player, white_player=ppo_player, env=GomokuEnv()))

    for index, result in enumerate(results, start=1):
        print(
            f"match={index} "
            f"black={result.black_name} "
            f"white={result.white_name} "
            f"winner={result.winner_name} "
            f"reason={result.reason} "
            f"moves={result.move_count}"
        )

    _summarize(results, ppo_name=ppo_player.name, heuristic_name=heuristic_player.name)


if __name__ == "__main__":
    main()
