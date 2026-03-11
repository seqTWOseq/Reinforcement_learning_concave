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
from gomoku_project.rl.ppo_trainer import PPOTrainer
from gomoku_project.utils.checkpoint_utils import load_checkpoint_or_maybe_warn


def _format_optional(value: float | None) -> str:
    return "None" if value is None else f"{value:.4f}"


def _ppo_summary(result) -> dict[str, float] | None:
    for summary in result.participant_summaries:
        if summary.algorithm == "ppo":
            return summary.update_metrics
    return None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train PPO through explicit PPO vs Heuristic matches.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--games", type=int, default=100)
    parser.add_argument("--model-path", type=str, default="gomoku_project/models/ppo/ppo_checkpoint.pt")
    parser.add_argument("--allow-random-init", action="store_true")
    parser.add_argument("--reset-model", action="store_true")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-epsilon", type=float, default=0.2)
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--update-epochs", type=int, default=4)
    parser.add_argument("--minibatch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--deterministic-ppo", action="store_true")
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--move-delay", type=float, default=0.0)
    parser.add_argument("--post-game-delay", type=float, default=1.0)
    args = parser.parse_args()

    if args.games <= 0:
        parser.error("--games must be > 0.")

    trainer = PPOTrainer(
        device=args.device,
        learning_rate=args.learning_rate,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_epsilon=args.clip_epsilon,
        entropy_coef=args.entropy_coef,
        value_coef=args.value_coef,
        update_epochs=args.update_epochs,
        minibatch_size=args.minibatch_size,
        seed=args.seed,
    )
    checkpoint_path, _ = load_checkpoint_or_maybe_warn(
        trainer=trainer,
        path=args.model_path,
        model_label="PPO",
        cli_flag="model-path",
        allow_random_init=args.allow_random_init,
        reset_model=args.reset_model,
    )

    for game_index in range(1, args.games + 1):
        ppo_player = trainer.build_player(deterministic=args.deterministic_ppo, name="PPO")
        heuristic_player = HeuristicPlayer(
            name="Heuristic",
            seed=None if args.seed is None else args.seed + game_index,
        )

        if game_index % 2 == 1:
            black_participant = MatchParticipant(player=ppo_player, trainer=trainer, algorithm="ppo")
            white_participant = MatchParticipant(player=heuristic_player)
        else:
            black_participant = MatchParticipant(player=heuristic_player)
            white_participant = MatchParticipant(player=ppo_player, trainer=trainer, algorithm="ppo")

        result = run_training_match(
            black_participant=black_participant,
            white_participant=white_participant,
            matchup="ppo_vs_heuristic",
            env=GomokuEnv(),
            render=args.render,
            move_delay=args.move_delay,
            post_game_delay=args.post_game_delay,
            train_after_game=True,
        )
        trainer.save(checkpoint_path)

        metrics = _ppo_summary(result) or {}
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
            f"ppo_update={bool(metrics)} "
            f"ppo_total_loss={_format_optional(metrics.get('total_loss'))} "
            f"ppo_policy_loss={_format_optional(metrics.get('policy_loss'))} "
            f"ppo_value_loss={_format_optional(metrics.get('value_loss'))} "
            f"checkpoint_saved=True "
            f"checkpoint={checkpoint_path}"
        )


if __name__ == "__main__":
    main()
