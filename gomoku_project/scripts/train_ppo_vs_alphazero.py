from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gomoku_project.arena.training_match import MatchParticipant, run_training_match
from gomoku_project.envs.gomoku_env import GomokuEnv
from gomoku_project.rl.ppo_trainer import PPOTrainer
from gomoku_project.rl.trainer import AlphaZeroTrainer
from gomoku_project.utils.checkpoint_utils import load_checkpoint_or_maybe_warn


def _format_optional(value: float | None) -> str:
    return "None" if value is None else f"{value:.4f}"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train PPO and AlphaZero together through direct PPO vs AlphaZero matches.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--games", type=int, default=10)
    parser.add_argument("--ppo-model-path", type=str, default="gomoku_project/models/ppo/ppo_checkpoint.pt")
    parser.add_argument("--alphazero-model-path", type=str, default="gomoku_project/models/alphazero_checkpoint.pt")
    parser.add_argument("--allow-random-init", action="store_true")
    parser.add_argument("--reset-ppo", action="store_true")
    parser.add_argument("--reset-alphazero", action="store_true")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--ppo-learning-rate", type=float, default=3e-4)
    parser.add_argument("--ppo-gamma", type=float, default=0.99)
    parser.add_argument("--ppo-gae-lambda", type=float, default=0.95)
    parser.add_argument("--ppo-clip-epsilon", type=float, default=0.2)
    parser.add_argument("--ppo-entropy-coef", type=float, default=0.01)
    parser.add_argument("--ppo-value-coef", type=float, default=0.5)
    parser.add_argument("--ppo-update-epochs", type=int, default=4)
    parser.add_argument("--ppo-minibatch-size", type=int, default=128)
    parser.add_argument("--alphazero-batch-size", type=int, default=32)
    parser.add_argument("--alphazero-mcts-simulations", type=int, default=24)
    parser.add_argument("--alphazero-train-steps", type=int, default=8)
    parser.add_argument("--deterministic-ppo", action="store_true")
    parser.add_argument("--deterministic-alphazero", action="store_true")
    parser.add_argument("--disable-root-noise", action="store_true")
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--move-delay", type=float, default=0.0)
    parser.add_argument("--post-game-delay", type=float, default=1.0)
    args = parser.parse_args()

    if args.games <= 0:
        parser.error("--games must be > 0.")
    if args.alphazero_train_steps < 0:
        parser.error("--alphazero-train-steps must be >= 0.")

    ppo_trainer = PPOTrainer(
        device=args.device,
        learning_rate=args.ppo_learning_rate,
        gamma=args.ppo_gamma,
        gae_lambda=args.ppo_gae_lambda,
        clip_epsilon=args.ppo_clip_epsilon,
        entropy_coef=args.ppo_entropy_coef,
        value_coef=args.ppo_value_coef,
        update_epochs=args.ppo_update_epochs,
        minibatch_size=args.ppo_minibatch_size,
    )
    ppo_checkpoint_path, _ = load_checkpoint_or_maybe_warn(
        trainer=ppo_trainer,
        path=args.ppo_model_path,
        model_label="PPO",
        cli_flag="ppo-model-path",
        allow_random_init=args.allow_random_init,
        reset_model=args.reset_ppo,
    )

    alphazero_trainer = AlphaZeroTrainer(
        batch_size=args.alphazero_batch_size,
        mcts_simulations=args.alphazero_mcts_simulations,
        device=args.device,
    )
    alphazero_checkpoint_path, _ = load_checkpoint_or_maybe_warn(
        trainer=alphazero_trainer,
        path=args.alphazero_model_path,
        model_label="AlphaZero",
        cli_flag="alphazero-model-path",
        allow_random_init=args.allow_random_init,
        reset_model=args.reset_alphazero,
    )

    for game_index in range(1, args.games + 1):
        ppo_player = ppo_trainer.build_player(deterministic=args.deterministic_ppo, name="PPO")
        alphazero_player = alphazero_trainer.build_player(
            deterministic=args.deterministic_alphazero,
            name="AlphaZero",
            use_root_noise=not args.disable_root_noise and not args.deterministic_alphazero,
        )

        if game_index % 2 == 1:
            black_participant = MatchParticipant(player=ppo_player, trainer=ppo_trainer, algorithm="ppo")
            white_participant = MatchParticipant(
                player=alphazero_player,
                trainer=alphazero_trainer,
                algorithm="alphazero",
            )
        else:
            black_participant = MatchParticipant(
                player=alphazero_player,
                trainer=alphazero_trainer,
                algorithm="alphazero",
            )
            white_participant = MatchParticipant(player=ppo_player, trainer=ppo_trainer, algorithm="ppo")

        result = run_training_match(
            black_participant=black_participant,
            white_participant=white_participant,
            matchup="ppo_vs_alphazero",
            env=GomokuEnv(),
            render=args.render,
            move_delay=args.move_delay,
            post_game_delay=args.post_game_delay,
            train_after_game=True,
            alphazero_train_steps=args.alphazero_train_steps,
        )
        ppo_trainer.save(ppo_checkpoint_path)
        alphazero_trainer.save(alphazero_checkpoint_path)

        ppo_metrics = next(
            (summary.update_metrics for summary in result.participant_summaries if summary.algorithm == "ppo"),
            None,
        ) or {}
        alphazero_metrics = next(
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
            f"ppo_update={bool(ppo_metrics)} "
            f"ppo_total_loss={_format_optional(ppo_metrics.get('total_loss'))} "
            f"ppo_policy_loss={_format_optional(ppo_metrics.get('policy_loss'))} "
            f"ppo_value_loss={_format_optional(ppo_metrics.get('value_loss'))}"
        )
        print(
            f"alphazero_update={bool(alphazero_metrics)} "
            f"alphazero_total_loss={_format_optional(alphazero_metrics.get('total_loss'))} "
            f"alphazero_policy_loss={_format_optional(alphazero_metrics.get('policy_loss'))} "
            f"alphazero_value_loss={_format_optional(alphazero_metrics.get('value_loss'))}"
        )
        print(
            f"checkpoint_saved=True "
            f"ppo_checkpoint={ppo_checkpoint_path} "
            f"alphazero_checkpoint={alphazero_checkpoint_path}"
        )


if __name__ == "__main__":
    main()
