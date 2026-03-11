from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gomoku_project.arena.training_match import MatchParticipant, run_training_match
from gomoku_project.envs.gomoku_env import GomokuEnv
from gomoku_project.players.heuristic_player import HeuristicPlayer
from gomoku_project.rl.trainer import AlphaZeroTrainer
from gomoku_project.utils.checkpoint_utils import load_checkpoint_or_maybe_warn


def _format_optional(value: float | None) -> str:
    return "None" if value is None else f"{value:.4f}"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train AlphaZero through explicit AlphaZero vs Heuristic matches.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--games", type=int, default=10)
    parser.add_argument("--model-path", type=str, default="gomoku_project/models/alphazero_checkpoint.pt")
    parser.add_argument("--allow-random-init", action="store_true")
    parser.add_argument("--reset-model", action="store_true")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--mcts-simulations", type=int, default=32)
    parser.add_argument("--train-steps", type=int, default=8)
    parser.add_argument("--deterministic-alphazero", action="store_true")
    parser.add_argument("--disable-root-noise", action="store_true")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--move-delay", type=float, default=0.0)
    parser.add_argument("--post-game-delay", type=float, default=1.0)
    args = parser.parse_args()

    if args.games <= 0:
        parser.error("--games must be > 0.")
    if args.train_steps < 0:
        parser.error("--train-steps must be >= 0.")

    trainer = AlphaZeroTrainer(
        batch_size=args.batch_size,
        mcts_simulations=args.mcts_simulations,
        device=args.device,
    )
    checkpoint_path, _ = load_checkpoint_or_maybe_warn(
        trainer=trainer,
        path=args.model_path,
        model_label="AlphaZero",
        cli_flag="model-path",
        allow_random_init=args.allow_random_init,
        reset_model=args.reset_model,
    )

    for game_index in range(1, args.games + 1):
        alphazero_player = trainer.build_player(
            deterministic=args.deterministic_alphazero,
            name="AlphaZero",
            use_root_noise=not args.disable_root_noise and not args.deterministic_alphazero,
        )
        heuristic_player = HeuristicPlayer(
            name="Heuristic",
            seed=None if args.seed is None else args.seed + game_index,
        )

        if game_index % 2 == 1:
            black_participant = MatchParticipant(player=alphazero_player, trainer=trainer, algorithm="alphazero")
            white_participant = MatchParticipant(player=heuristic_player)
        else:
            black_participant = MatchParticipant(player=heuristic_player)
            white_participant = MatchParticipant(player=alphazero_player, trainer=trainer, algorithm="alphazero")

        result = run_training_match(
            black_participant=black_participant,
            white_participant=white_participant,
            matchup="alphazero_vs_heuristic",
            env=GomokuEnv(),
            render=args.render,
            move_delay=args.move_delay,
            post_game_delay=args.post_game_delay,
            train_after_game=True,
            alphazero_train_steps=args.train_steps,
        )
        trainer.save(checkpoint_path)

        metrics = next(
            (summary.update_metrics for summary in result.participant_summaries if summary.algorithm == "alphazero"),
            None,
        ) or {}
        print(
            f"game={game_index} "
            f"matchup={result.matchup} "
            f"black={result.black_name} "
            f"white={result.white_name} "
            f"winner={result.winner_name} "
            f"reason={result.reason} "
            f"moves={result.move_count}"
        )
        print(
            f"ppo_steps={result.ppo_steps_collected} "
            f"alphazero_examples={result.alphazero_examples_collected}"
        )
        print(
            f"alphazero_update={bool(metrics)} "
            f"alphazero_total_loss={_format_optional(metrics.get('total_loss'))} "
            f"alphazero_policy_loss={_format_optional(metrics.get('policy_loss'))} "
            f"alphazero_value_loss={_format_optional(metrics.get('value_loss'))} "
            f"checkpoint_saved=True "
            f"checkpoint={checkpoint_path}"
        )


if __name__ == "__main__":
    main()
